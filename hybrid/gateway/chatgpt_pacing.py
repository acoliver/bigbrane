"""Conservative pacing for Sign in with ChatGPT calls.

OpenAI's SIWC docs publish no concurrency, rate or ramp rule. The first Bigbrane grant was withdrawn 13 minutes after
registration, during a burst of up to four concurrent high-effort streams (results/gateway-20261010). Whether the
burst caused it is unknown, so the gateway paces itself: a minimum gap between request starts, a sliding hourly cap
that queues instead of failing, and a ceiling on ``reasoning.effort``. Concurrency is limited separately by the
backend's semaphore.

Start times are planned, not measured after the sleep: a request's start is the earliest moment both rules allow,
the caller sleeps until then, and that planned moment is recorded. This keeps the arithmetic exact with a fake clock.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

from .paths import EFFORT_ORDER, ChatGPTConfig

log = logging.getLogger("bigbrane.gateway")

WINDOW_S = 3600.0


@dataclass(frozen=True)
class PacingPolicy:
    max_concurrency: int = 1
    min_start_interval_s: float = 10.0
    max_requests_per_hour: int = 60
    max_reasoning_effort: str = "medium"       # "" means no cap

    @classmethod
    def from_config(cls, config: ChatGPTConfig, max_concurrency: int | None = None) -> "PacingPolicy":
        return cls(max_concurrency=max_concurrency or config.max_concurrency,
                   min_start_interval_s=float(config.min_start_interval_s),
                   max_requests_per_hour=config.max_requests_per_hour,
                   max_reasoning_effort=config.max_reasoning_effort)

    def describe(self) -> str:
        effort = f"reasoning effort capped at {self.max_reasoning_effort}" if self.max_reasoning_effort else \
            "reasoning effort not capped"
        return (f"chatgpt pacing: concurrency {self.max_concurrency}, at least {self.min_start_interval_s:g} s between "
                f"request starts, at most {self.max_requests_per_hour} requests per hour (extra requests wait), "
                f"{effort}")


class Pacer:
    """Admits one request start at a time under the policy's interval and hourly cap."""

    def __init__(self, policy: PacingPolicy, *, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.policy = policy
        self._clock = clock
        self._sleep = sleep
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()          # FIFO: waiters are admitted in arrival order

    def _next_start(self, now: float) -> float:
        while self._starts and self._starts[0] <= now - WINDOW_S:
            self._starts.popleft()
        start = now
        if self._starts:
            start = max(start, self._starts[-1] + self.policy.min_start_interval_s)
        if len(self._starts) >= self.policy.max_requests_per_hour:
            start = max(start, self._starts[-self.policy.max_requests_per_hour] + WINDOW_S)
        return start

    async def admit(self) -> float:
        """Wait until this request may start; return the seconds waited."""
        async with self._lock:
            now = self._clock()
            start = self._next_start(now)
            wait = start - now
            if wait > 0:
                full = len(self._starts) >= self.policy.max_requests_per_hour
                log.info("chatgpt pacing: waiting %.1fs before the next request (%s; %d starts in the last hour)",
                         wait, "hourly cap reached" if full else "minimum gap between starts", len(self._starts))
                await self._sleep(wait)
            self._starts.append(start)
            return max(wait, 0.0)

    def starts_in_last_hour(self) -> int:
        now = self._clock()
        return sum(1 for t in self._starts if t > now - WINDOW_S)


def cap_effort(payload: dict, ceiling: str) -> str | None:
    """Lower ``payload["reasoning"]["effort"]`` to ``ceiling`` and return the value that was replaced, or None.

    A request with no effort gets the ceiling, so the model's own default (which may be higher) cannot apply. An
    effort outside the known vocabulary is treated as above the ceiling.
    """
    if not ceiling:
        return None
    reasoning = payload.setdefault("reasoning", {})
    requested = reasoning.get("effort")
    if requested in EFFORT_ORDER and EFFORT_ORDER.index(requested) <= EFFORT_ORDER.index(ceiling):
        return None
    reasoning["effort"] = ceiling
    return requested if requested is not None else "(model default)"
