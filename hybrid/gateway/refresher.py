"""The single refresher.

The gateway process is the only code that calls a token endpoint for a refresh. One ``asyncio.Lock`` per
provider serializes refreshes; inside the lock the record is re-read, so concurrent callers that waited
share the refresh the first one did. The replacement refresh token is persisted (and read back) before the
new access token is returned to any caller. ``invalid_grant`` and its relatives mark the stored refresh
token dead: no later call touches the token endpoint until a new login replaces the record. The access token
that came with the dead refresh token is still used until it expires (OpenAI's errors page says to stop using a
token set once a refresh confirms it is unusable, but the access token keeps working until ``exp``, and the
devkit#5 reports show it does), so a refusal a few minutes before expiry does not end the session early.
A record that carries ``earliest_refresh_at`` is refreshed only from that time on (and only when the access token is
near expiry); before it, the current token is used. ``due_at`` is that rule as a time, so the proactive scheduler
(``refresh_schedule.py``) and on-demand callers agree on when a refresh is due.

Across processes, a refresh also holds the provider's flock on the store (``fsstore.py``) from the re-read
until the new record is published, so a login command and a gateway never rotate the same refresh token twice.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from .errors import LoginRequired, RefreshFailed, UpstreamError
from .redact import Redactor
from .store import Credentials, ProviderLock, TokenStore

DEFAULT_SKEW_S = 300.0     # refresh this long before expiry


class RefreshRejected(Exception):
    """The token endpoint says the refresh token is unusable (invalid_grant and similar). ``details`` holds what it
    said, without any credential."""

    def __init__(self, reason: str, *, details: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class TokenRefresh(Protocol):
    async def refresh(self, creds: Credentials) -> Credentials: ...


def _fingerprint(refresh_token: str | None) -> str:
    return hashlib.sha256((refresh_token or "").encode()).hexdigest()


def refresh_due_at(creds: Credentials, skew_s: float = DEFAULT_SKEW_S) -> float:
    """When a refresh of ``creds`` becomes due: ``skew_s`` before expiry, but never before ``earliest_refresh_at``."""
    earliest = creds.earliest_refresh_at
    return max(float("-inf") if earliest is None else earliest, creds.expires_at - skew_s)


def iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RefreshJournal:
    """One JSON line per refresh attempt (outcome, timing, the token endpoint's error fields and request id; never a
    token), so ``status`` in another process can show how the last refresh went."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()

    def record(self, provider: str, outcome: str, **fields) -> dict:
        event = {"ts": iso(self._clock()), "provider": provider, "outcome": outcome,
                 **{k: v for k, v in fields.items() if v is not None}}
        line = (json.dumps(event, sort_keys=True) + "\n").encode()
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        return event

    def last(self, provider: str) -> dict | None:
        if not self.path.exists():
            return None
        found = None
        for line in self.path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("provider") == provider:
                found = event
        return found


class Refresher:
    def __init__(self, store: TokenStore, providers: dict[str, TokenRefresh], *, redactor: Redactor,
                 clock: Callable[[], float] = time.time, skew_s: float = DEFAULT_SKEW_S,
                 on_persisted: Callable[[str], Awaitable[None]] | None = None,
                 journal: RefreshJournal | None = None) -> None:
        self._store = store
        self._providers = providers
        self._redactor = redactor
        self._clock = clock
        self._skew_s = skew_s
        self._locks: dict[str, asyncio.Lock] = {}
        self._dead: dict[str, tuple[str, str]] = {}      # provider -> (refresh token fingerprint, reason)
        self._on_persisted = on_persisted                # test seam: runs between persist and use
        self._journal = journal

    @property
    def skew_s(self) -> float:
        return self._skew_s

    def _lock(self, provider: str) -> asyncio.Lock:
        return self._locks.setdefault(provider, asyncio.Lock())

    def _window_open(self, creds: Credentials) -> bool:
        """False before the issuer's ``earliest_refresh_at``: a refresh then would be refused, so the current
        token is used until the window opens."""
        earliest = creds.earliest_refresh_at
        return earliest is None or self._clock() >= earliest

    def due_at(self, creds: Credentials) -> float:
        return refresh_due_at(creds, self._skew_s)

    def _needs_refresh(self, creds: Credentials) -> bool:
        return self._clock() >= self.due_at(creds)

    def is_dead(self, provider: str, creds: Credentials) -> bool:
        """True when this process saw the token endpoint refuse ``creds``' refresh token."""
        dead = self._dead.get(provider)
        return bool(dead and dead[0] == _fingerprint(creds.refresh_token))

    async def _load(self, provider: str) -> Credentials:
        creds = await asyncio.to_thread(self._store.load, provider)
        if creds is None:
            raise LoginRequired(provider, "no stored login")
        self._redactor.add(creds.access_token, creds.refresh_token, creds.id_token)
        return creds

    def _usable_after_rejection(self, provider: str, creds: Credentials, rejected_access_token: str | None) -> Credentials:
        """``creds`` whose refresh token is dead: the access token is returned while it has not expired and the API
        has not refused it; otherwise only a new login helps."""
        if creds.expires_at > self._clock() and creds.access_token != rejected_access_token:
            return creds
        raise LoginRequired(provider, self._dead[provider][1])

    async def credentials(self, provider: str, *, rejected_access_token: str | None = None) -> Credentials:
        """Credentials with an access token that is not near expiry.

        ``rejected_access_token`` is the token a caller just saw the API refuse. If the store still holds
        that token it is refreshed even though its expiry looks fine; if another caller already replaced
        it, the replacement is returned.
        """
        creds = await self._load(provider)
        if self.is_dead(provider, creds):
            return self._usable_after_rejection(provider, creds, rejected_access_token)
        if not self._stale(creds, rejected_access_token):
            return creds
        async with self._lock(provider):
            return await self._refresh_under_store_lock(provider, rejected_access_token)

    async def _refresh_under_store_lock(self, provider: str, rejected_access_token: str | None) -> Credentials:
        store_lock = self._store.lock(provider)
        try:
            await asyncio.to_thread(store_lock.acquire)
            creds = await self._load(provider)         # another process may have refreshed while we waited
            if self.is_dead(provider, creds):          # another caller in this process was just refused
                return self._usable_after_rejection(provider, creds, rejected_access_token)
            if not self._stale(creds, rejected_access_token):
                return creds
            return await self._refresh(provider, creds, store_lock, rejected_access_token)
        finally:
            store_lock.release()

    def _stale(self, creds: Credentials, rejected_access_token: str | None) -> bool:
        return self._needs_refresh(creds) or (self._window_open(creds) and creds.access_token == rejected_access_token)

    def _note(self, provider: str, outcome: str, **fields) -> None:
        if self._journal is not None:
            self._journal.record(provider, outcome, **fields)

    async def _refresh(self, provider: str, creds: Credentials, store_lock: ProviderLock,
                       rejected_access_token: str | None) -> Credentials:
        if not creds.refresh_token:
            raise LoginRequired(provider, "the stored login has no refresh token")
        timing = {"previous_expires_at": iso(creds.expires_at), "earliest_refresh_at": iso(creds.earliest_refresh_at),
                  "seconds_before_expiry": round(creds.expires_at - self._clock(), 1)}
        try:
            fresh = await self._providers[provider].refresh(creds)
        except RefreshRejected as exc:
            self._dead[provider] = (_fingerprint(creds.refresh_token), exc.reason)
            self._note(provider, "rejected", **timing, **exc.details)
            return self._usable_after_rejection(provider, creds, rejected_access_token)
        except RefreshFailed as exc:
            self._note(provider, "failed", **timing, **exc.details)
            raise
        except UpstreamError as exc:
            self._note(provider, "failed", **timing, error=exc.code, message=exc.message[:200])
            raise
        self._redactor.add(fresh.access_token, fresh.refresh_token, fresh.id_token)
        await asyncio.to_thread(store_lock.write, fresh)     # persist before anything uses it
        self._note(provider, "ok", **timing, expires_at=iso(fresh.expires_at),
                   next_earliest_refresh_at=iso(fresh.earliest_refresh_at),
                   rotated=fresh.refresh_token != creds.refresh_token)
        if self._on_persisted is not None:
            await self._on_persisted(provider)
        return fresh
