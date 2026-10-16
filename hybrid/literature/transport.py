"""Rate-limited, cached, retrying HTTP access to the keyless literature APIs.

Every network call passes through `Transport.get`. A cache hit returns the stored body
without touching the network or the rate limiter. OpenAlex and Semantic Scholar requests carry an
API key header when one is configured; the key is not part of the cache key, the cache envelope,
or any error text.

Every attempt, retries after a 429 or a server error included, waits for its slot in the `Pacer`
first. In production the pacer keeps its last-request times in files under the workspace's
var/slots directory, so the spacing holds across the processes of a research run, not only within one.
"""
from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import httpx

from hybrid.literature.cache import ResponseCache, request_key
from hybrid.literature.events import utc_now
from hybrid.literature.keys import ApiKey

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_LOG = logging.getLogger(__name__)

# arXiv asks for at most one request every three seconds per client, which is a policy across all of
# this machine's processes (the pacer's files), not per process. A Semantic Scholar key grants 1 request
# per second, cumulative across all endpoints, and Semantic Scholar asks clients to stay below that, so
# the spacing is 1.1 seconds. The keyless pool is shared by all anonymous users and needs the same care.
MIN_INTERVAL_S = {"arxiv": 3.0, "openalex": 0.1, "s2": 1.1}


class SourceError(RuntimeError):
    """A source could not answer. Callers record this as an event; it is never swallowed."""

    def __init__(self, source: str, url: str, reason: str) -> None:
        super().__init__(f"{source}: {reason} ({url})")
        self.source = source
        self.url = url
        self.reason = reason


@dataclass(frozen=True)
class Request:
    url: str
    params: dict[str, str]


@dataclass(frozen=True)
class Fetched:
    source: str
    url: str
    params: dict[str, str]
    body: str
    fetched_at: str
    from_cache: bool


class Pacer:
    """Minimum spacing between requests to one source. The lock is held while waiting, so threads
    calling in parallel (the prior-art searches run in worker threads) queue up instead of racing.
    Transports that must share one request budget, such as every transport of a production process,
    share one Pacer.

    With `slots_dir`, the last request time of each source also lives in files there, and the wait
    happens while holding an exclusive `fcntl.flock` on the source's lock file. Processes on this host
    then queue up on that lock and space their requests by the wall-clock time recorded in the file, so
    pass a wall clock (`time.time`) with it. The OS releases the lock if a process dies. Without
    `slots_dir` the times stay in memory, which is what tests with a fake clock use."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        *,
        slots_dir: Path | None = None,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._slots_dir = Path(slots_dir) if slots_dir is not None else None
        self._lock = threading.Lock()
        self._last_request_at: dict[str, float] = {}

    def wait_for_slot(self, source: str) -> float:
        """Block until `source` may be requested, record the request, and return its start time on
        this pacer's clock."""
        interval = MIN_INTERVAL_S[source]
        with self._lock:
            if self._slots_dir is None:
                started_at = self._wait_until_due(source, interval, self._last_request_at.get(source))
                self._last_request_at[source] = started_at
                return started_at
            return self._wait_across_processes(source, interval)

    def _wait_until_due(self, source: str, interval: float, last: float | None) -> float:
        waited = 0.0
        if last is not None:
            # A wall clock stepped back would leave `last` in the future; never wait longer than one interval.
            last = min(last, self._clock())
            # Loop because a sleep can return before the full interval has passed on the clock.
            while (remaining := last + interval - self._clock()) > 0:
                self._sleep(remaining)
                waited += remaining
        _LOG.debug("literature request slot: source=%s waited=%.3fs", source, waited)
        return self._clock()

    def _wait_across_processes(self, source: str, interval: float) -> float:
        assert self._slots_dir is not None
        self._slots_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._slots_dir / f"literature-{source}.last"
        fd = os.open(self._slots_dir / f"literature-{source}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                last = float(stamp.read_text()) if stamp.exists() else None
                started_at = self._wait_until_due(source, interval, last)
                pending = stamp.with_suffix(".new")
                pending.write_text(repr(started_at))  # replaced whole, so a reader never sees half a time
                os.replace(pending, stamp)
                return started_at
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class Transport:
    def __init__(
        self,
        cache: ResponseCache,
        http: httpx.Client,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], str] = utc_now,
        max_retries: int = 3,
        base_backoff_s: float = 5.0,
        max_backoff_s: float = 120.0,
        cooldown_s: float = 600.0,
        pacer: Pacer | None = None,
        api_keys: Mapping[str, ApiKey] | None = None,
    ) -> None:
        self._cache = cache
        self._http = http
        self._sleep = sleep
        self._clock = clock
        self._now = now
        self._max_retries = max_retries
        self._base_backoff_s = base_backoff_s
        self._max_backoff_s = max_backoff_s
        self._cooldown_s = cooldown_s
        self._pacer = pacer or Pacer(clock, sleep)
        self._api_keys = dict(api_keys or {})
        # After retries run out on HTTP 429, the source is skipped for a cooldown instead of
        # waiting out the backoff again for every query. Semantic Scholar's keyless pool can stay
        # saturated for hours, and one burst should not disable a source for a whole build.
        self._cooldown_until: dict[str, float] = {}

    def get(self, source: str, request: Request) -> Fetched:
        key = request_key(source, request.url, request.params)
        cached = self._cache.get(source, key)
        if cached is not None:
            return Fetched(source, request.url, request.params, cached["body"], cached["fetched_at"], True)
        cooldown_until = self._cooldown_until.get(source)
        if cooldown_until is not None and self._clock() < cooldown_until:
            raise SourceError(source, request.url, "skipped: source is cooling down after rate limiting")

        body, fetched_at = self._fetch_with_retries(source, request)
        self._cache.put(
            source,
            key,
            {
                "source": source,
                "url": request.url,
                "params": request.params,
                "status": 200,
                "fetched_at": fetched_at,
                "body": body,
            },
        )
        return Fetched(source, request.url, request.params, body, fetched_at, False)

    def _fetch_with_retries(self, source: str, request: Request) -> tuple[str, str]:
        attempts = self._max_retries + 1
        key = self._api_keys.get(source)
        headers = key.headers if key is not None else {}
        for attempt in range(attempts):
            self._pacer.wait_for_slot(source)
            try:
                response = self._http.get(request.url, params=request.params, headers=headers)
            except httpx.TransportError as exc:
                # The text comes from the HTTP library; strip the key in case it echoes a header.
                text = key.redact(str(exc)) if key is not None else str(exc)
                reason, retry_after = f"transport error: {text}", None
            else:
                if response.status_code == 200:
                    return response.text, self._now()
                if response.status_code not in RETRYABLE_STATUS:
                    raise SourceError(source, request.url, f"HTTP {response.status_code}")
                reason, retry_after = f"HTTP {response.status_code}", _retry_after_seconds(response)
            if attempt + 1 < attempts:
                self._sleep(self._backoff_s(attempt, retry_after))
        if reason == "HTTP 429":
            self._cooldown_until[source] = self._clock() + self._cooldown_s
        raise SourceError(source, request.url, f"{reason} after {attempts} attempts")

    def _backoff_s(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self._max_backoff_s)
        return min(self._base_backoff_s * (2**attempt), self._max_backoff_s)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None  # HTTP-date form; fall back to exponential backoff
