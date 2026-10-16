"""Retry and rate-limit throttling for Anthropic calls, ported from LLxprt Code.

Sources:
- packages/core/src/utils/retry.ts and packages/providers/src/RetryOrchestrator.ts: 6 attempts, 4 s initial delay,
  30 s ceiling, exponential backoff with +-30 % jitter, Retry-After honoured on 429 and on in-stream
  overloaded/rate-limit errors (capped at 300 s by providers/src/retryAfterHeader.ts). 400 and 403 are never
  retried. A 401 is handled by the backend (one forced token refresh), not here.
- packages/providers/src/anthropic/AnthropicRateLimitHandler.ts: the ``anthropic-ratelimit-*`` headers of the last
  answer decide whether the next request waits for a reset (threshold 5 %, at most 60 s).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, Mapping

from .errors import UpstreamError

# Anthropic error types that arrive as an `error` stream event with no HTTP status.
STREAM_RETRYABLE_TYPES = frozenset({"overloaded_error", "rate_limit_error", "api_error"})
TRANSIENT_CODES = frozenset({"upstream_unreachable", "stream_incomplete"})
RETRY_AFTER_CAP_S = 300.0


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    initial_delay_s: float = 4.0
    max_delay_s: float = 30.0
    jitter: float = 0.3


def is_retryable(error: UpstreamError) -> bool:
    detail = error.detail
    if detail is not None:
        return detail.http_status == 429 or detail.http_status >= 500
    return error.code in STREAM_RETRYABLE_TYPES or error.code in TRANSIENT_CODES


def _honors_retry_after(error: UpstreamError) -> bool:
    detail = error.detail
    return error.code in STREAM_RETRYABLE_TYPES if detail is None else detail.http_status == 429


def parse_retry_after(value: str | None, now: float) -> float | None:
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP date), capped; None when absent or unusable."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(int(value))
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - now
        except (TypeError, ValueError):
            return None
    return min(max(seconds, 0.0), RETRY_AFTER_CAP_S)


async def with_retries(call: Callable[[], Awaitable], policy: RetryPolicy, *,
                       sleep: Callable[[float], Awaitable[None]],
                       rand: Callable[[], float] = random.random,
                       on_retry: Callable[[int, UpstreamError, float], None] = lambda *_: None):
    """Run ``call`` until it succeeds, fails with a non-retryable error, or ``policy.max_attempts`` is used up."""
    delay = policy.initial_delay_s
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await call()
        except UpstreamError as error:
            if attempt >= policy.max_attempts or not is_retryable(error):
                raise
            explicit = error.detail.retry_after_s if error.detail is not None and _honors_retry_after(error) else None
            if explicit:
                wait, delay = explicit, policy.initial_delay_s
            else:
                wait = max(0.0, delay + delay * policy.jitter * (rand() * 2 - 1))
                delay = min(policy.max_delay_s, delay * 2)
            on_retry(attempt, error, wait)
            await sleep(wait)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class RateLimitInfo:
    requests_limit: int | None = None
    requests_remaining: int | None = None
    requests_reset: float | None = None          # epoch seconds
    tokens_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_reset: float | None = None
    input_tokens_limit: int | None = None
    input_tokens_remaining: int | None = None


def _int(headers: Mapping[str, str], name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _reset(headers: Mapping[str, str], name: str) -> float | None:
    value = headers.get(name)
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


def parse_rate_limit_headers(headers: Mapping[str, str]) -> RateLimitInfo:
    return RateLimitInfo(
        requests_limit=_int(headers, "anthropic-ratelimit-requests-limit"),
        requests_remaining=_int(headers, "anthropic-ratelimit-requests-remaining"),
        requests_reset=_reset(headers, "anthropic-ratelimit-requests-reset"),
        tokens_limit=_int(headers, "anthropic-ratelimit-tokens-limit"),
        tokens_remaining=_int(headers, "anthropic-ratelimit-tokens-remaining"),
        tokens_reset=_reset(headers, "anthropic-ratelimit-tokens-reset"),
        input_tokens_limit=_int(headers, "anthropic-ratelimit-input-tokens-limit"),
        input_tokens_remaining=_int(headers, "anthropic-ratelimit-input-tokens-remaining"))


def throttle_wait_s(info: RateLimitInfo, now: float, *, threshold_pct: float = 5.0, max_wait_s: float = 60.0) -> float:
    """How long to wait before the next request: until the reset of the first bucket below the threshold."""
    buckets = ((info.requests_remaining, info.requests_limit, info.requests_reset),
               (info.tokens_remaining, info.tokens_limit, info.tokens_reset),
               (info.input_tokens_remaining, info.input_tokens_limit, None))
    for remaining, limit, reset in buckets:
        if remaining is None or limit is None or limit <= 0 or remaining / limit * 100 >= threshold_pct:
            continue
        if reset is not None and reset - now > 0:
            return min(reset - now, max_wait_s)
    return 0.0
