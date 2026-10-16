"""Proactive refresh: renew a provider's token as soon as a refresh is due, whether or not a request is waiting.

ChatGPT's refresh window is short. In the 2026-10-11 login, ``earliest_refresh_at`` was six minutes before expiry,
so an on-demand refresh happens only if a request arrives in those six minutes. A research loop that pauses for
longer (a long Claude phase, a sandbox run) would otherwise come back to an expired token and refresh after expiry,
a case no client has shown to work. The scheduler sleeps until ``Refresher.due_at`` and then asks the refresher for
credentials, which performs the one refresh under the usual locks, exactly as a request would at that moment.

After a refusal (``invalid_grant`` and similar) it makes no further attempt until the stored record changes (a new
login). After a transient failure it retries with backoff while the window is open.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from .errors import GatewayError, LoginRequired
from .refresher import Refresher, iso
from .store import TokenStore

log = logging.getLogger("bigbrane.gateway")

IDLE_POLL_S = 60.0          # how often a missing, dead or not-yet-due record is looked at again
RETRY_BASE_S = 15.0
RETRY_MAX_S = 120.0


class RefreshScheduler:
    def __init__(self, refresher: Refresher, store: TokenStore, provider: str, *,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self._refresher = refresher
        self._store = store
        self._provider = provider
        self._clock = clock
        self._sleep = sleep
        self._announced: float | None = None

    async def step(self) -> float:
        """One decision: refresh if due, else report how long to sleep. Returns the seconds to sleep next."""
        creds = await asyncio.to_thread(self._store.load, self._provider)
        if creds is None or not creds.refresh_token or self._refresher.is_dead(self._provider, creds):
            return IDLE_POLL_S
        due = self._refresher.due_at(creds)
        now = self._clock()
        if now < due:
            if self._announced != due:
                self._announced = due
                log.info("%s: proactive refresh scheduled for %s (earliest_refresh_at %s, access token expires %s)",
                         self._provider, iso(due), iso(creds.earliest_refresh_at), iso(creds.expires_at))
            return min(due - now, IDLE_POLL_S)
        log.info("%s: refresh due (earliest_refresh_at %s, expires %s); refreshing now", self._provider,
                 iso(creds.earliest_refresh_at), iso(creds.expires_at))
        fresh = await self._refresher.credentials(self._provider)
        if self._refresher.is_dead(self._provider, fresh):
            log.error("%s: the refresh token was refused; the current access token is used until %s, then a login "
                      "is needed (python -m hybrid.gateway login %s)", self._provider, iso(fresh.expires_at),
                      self._provider)
        else:
            log.info("%s: refreshed; new access token expires %s, next earliest_refresh_at %s", self._provider,
                     iso(fresh.expires_at), iso(fresh.earliest_refresh_at))
        return IDLE_POLL_S          # never a tight loop, even if the issuer answered with a token that is due at once

    async def run(self) -> None:
        failures = 0
        while True:
            try:
                delay = await self.step()
                failures = 0
            except LoginRequired as exc:
                log.error("%s: %s", self._provider, exc)
                delay = IDLE_POLL_S
            except GatewayError as exc:
                failures += 1
                delay = min(RETRY_BASE_S * 2 ** (failures - 1), RETRY_MAX_S)
                log.warning("%s: proactive refresh failed (%s: %s); retrying in %.0fs", self._provider, exc.code,
                            exc.message, delay)
            await self._sleep(delay)
