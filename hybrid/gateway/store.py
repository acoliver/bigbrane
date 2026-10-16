"""Token records on the private filesystem store (see ``fsstore.py`` for the storage rules)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .errors import StoreConflict, StoreError
from .fsstore import LOCK_WAIT_S, FileLock, PrivateDir

PROVIDERS = ("anthropic", "chatgpt")


def record_name(provider: str) -> str:
    return f"{provider}.json"


def lock_name(provider: str) -> str:
    return f"{provider}.lock"


@dataclass(frozen=True)
class Credentials:
    access_token: str
    refresh_token: str | None
    expires_at: float                       # unix seconds
    id_token: str | None = None             # ChatGPT only; kept for id_token_hint on re-login
    meta: dict = field(default_factory=dict)   # client_id, scopes, email, subject, host id, saved_at

    def __repr__(self) -> str:
        return f"Credentials(expires_at={self.expires_at}, meta_keys={sorted(self.meta)})"

    @property
    def earliest_refresh_at(self) -> float | None:
        """Unix seconds before which the issuer will not accept a refresh (ChatGPT's ``earliest_refresh_at``).

        The issuer sends a number; an ISO 8601 string is also accepted, as OpenAI's devkit does. Absent or
        unparseable values mean no restriction is known.
        """
        raw = self.meta.get("earliest_refresh_at")
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                try:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    return None
        return None

    def to_json(self) -> str:
        return json.dumps({"access_token": self.access_token, "refresh_token": self.refresh_token,
                           "expires_at": self.expires_at, "id_token": self.id_token, "meta": self.meta},
                          sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Credentials":
        data = json.loads(text)
        return cls(access_token=data["access_token"], refresh_token=data.get("refresh_token"),
                   expires_at=float(data["expires_at"]), id_token=data.get("id_token"), meta=data.get("meta") or {})


def _encode(creds: Credentials) -> bytes:
    return (creds.to_json() + "\n").encode()


def _parse(provider: str, raw: bytes) -> Credentials:
    try:
        return Credentials.from_json(raw.decode())
    except (ValueError, KeyError, TypeError) as exc:
        raise StoreError(f"stored {provider} record is unreadable ({type(exc).__name__}); discard it with "
                         f"`python -m hybrid.gateway logout {provider}`, then run: "
                         f"python -m hybrid.gateway login {provider}") from exc


@dataclass(frozen=True)
class Snapshot:
    """The exact bytes of a provider's record at one moment (None when there is no record).

    A login keeps the snapshot it started with and publishes only if the record still equals it.
    """

    provider: str
    raw: bytes | None

    def credentials(self) -> Credentials | None:
        return None if self.raw is None else _parse(self.provider, self.raw)


class ProviderLock:
    """Cross-process lock for one provider's record, with read and write that are valid only while held."""

    def __init__(self, provider: str, lock: FileLock) -> None:
        self.provider = provider
        self._lock = lock

    def acquire(self, timeout_s: float = LOCK_WAIT_S) -> None:
        self._lock.acquire(timeout_s)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "ProviderLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def snapshot(self) -> Snapshot:
        return Snapshot(self.provider, self._lock.read(record_name(self.provider)))

    def write(self, creds: Credentials) -> None:
        self._lock.write(record_name(self.provider), _encode(creds))

    def delete(self) -> None:
        self._lock.remove(record_name(self.provider))


class TokenStore:
    def __init__(self, oauth_dir: Path, *, uid: int | None = None) -> None:
        self._dir = PrivateDir(oauth_dir, uid=uid)

    @property
    def path(self) -> Path:
        return self._dir.path

    def snapshot(self, provider: str) -> Snapshot:
        """Current record bytes. Owner, mode and type are checked; the content is not parsed."""
        return Snapshot(provider, self._dir.read(record_name(provider)))

    def load(self, provider: str) -> Credentials | None:
        return self.snapshot(provider).credentials()

    def lock(self, provider: str) -> ProviderLock:
        return ProviderLock(provider, self._dir.lock(lock_name(provider)))

    def save(self, provider: str, creds: Credentials) -> None:
        with self.lock(provider) as held:
            held.write(creds)

    def publish_login(self, provider: str, creds: Credentials, started_from: Snapshot) -> None:
        """Store a fresh login unless the record changed since ``started_from`` was taken.

        A concurrent refresh or login wins: its record is kept and this login fails, so a slow browser
        sign-in can never overwrite a newer rotated refresh token.
        """
        with self.lock(provider) as held:
            if held.snapshot().raw != started_from.raw:
                raise StoreConflict(f"the {provider} record changed while this login was in progress (another "
                                    f"login or a gateway refresh); the newer record was kept. If you still "
                                    f"need to sign in, run: python -m hybrid.gateway login {provider}")
            held.write(creds)

    def delete(self, provider: str) -> None:
        with self.lock(provider) as held:
            held.delete()
