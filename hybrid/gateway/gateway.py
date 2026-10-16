"""Routing, metering, the model list and the OpenAI-shaped surface, independent of the HTTP transport."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Protocol

from .backends import Completion
from .errors import BadRequest, GatewayError, LoginRequired, UpstreamError
from .meter import Meter

# The gateway server attaches the redacting filter to this logger, so upstream text is scrubbed before it is written.
log = logging.getLogger("bigbrane.gateway")


@dataclass(frozen=True)
class _CachedList:
    names: list[str]
    fetched_at: float


@dataclass(frozen=True)
class _Listing:
    """One provider's contribution to /v1/models: the ids to list, plus why it is not a live answer, if so."""
    names: list[str]
    not_logged_in: str | None = None
    failure: UpstreamError | None = None


def _log_upstream_failure(provider: str, error: UpstreamError) -> None:
    detail = error.detail
    if detail is None:
        log.warning("upstream %s failed: code=%s message=%s", provider, error.code, error.message)
    else:
        log.warning("upstream %s refused: http_status=%s code=%s type=%s message=%s%s", provider, detail.http_status,
                    detail.code, detail.error_type, detail.message,
                    "".join(f" {name}={value}" for name, value in detail.diagnostics().items()))


class Backend(Protocol):
    async def complete(self, body: dict, model: str) -> Completion: ...
    async def models(self) -> list[str]: ...


class Gateway:
    def __init__(self, backends: dict[str, Backend], meter: Meter, clock: Callable[[], float] = time.time, *,
                 models_timeout_s: float, models_cache_ttl_s: float, static_models: dict[str, list[str]]) -> None:
        self._backends = backends
        self._meter = meter
        self._clock = clock
        self._models_timeout_s = models_timeout_s
        self._models_cache_ttl_s = models_cache_ttl_s
        self._static_models = static_models
        self._model_cache: dict[str, _CachedList] = {}

    def route(self, model: str) -> tuple[str, str]:
        provider, sep, upstream = model.partition("/")
        if not sep or not upstream or provider not in self._backends:
            raise BadRequest(f"model {model!r} is not served here; use one of the prefixes "
                             f"{', '.join(p + '/<model>' for p in sorted(self._backends))}", code="model_not_found")
        return provider, upstream

    async def chat(self, body: dict, run_id: str | None = None) -> dict:
        """Non-streaming chat completion. Every call, successful or not, is metered."""
        model = body.get("model")
        if not isinstance(model, str) or not isinstance(body.get("messages"), list):
            raise BadRequest("the request needs a string model and a messages list")
        provider, upstream = self.route(model)
        call_id = f"call-{uuid.uuid4().hex[:12]}"
        started = self._clock()
        fields = {"call_id": call_id, "provider": provider, "model": model, "run_id": run_id}
        try:
            completion = await self._backends[provider].complete(body, upstream)
        except GatewayError as exc:
            if isinstance(exc, UpstreamError):
                _log_upstream_failure(provider, exc)
            error = {"code": exc.code, "status": exc.status}
            if isinstance(exc, UpstreamError) and exc.detail is not None:
                error.update(exc.detail.diagnostics())         # upstream request id and authorization headers
            self._meter.record(**fields, status="error", error=error, latency_s=round(self._clock() - started, 3))
            raise
        self._meter.record(**fields, status="ok", resolved_model=completion.resolved_model, usage=completion.usage,
                           latency_s=round(self._clock() - started, 3), dropped=list(completion.dropped))
        return completion.chat

    async def models(self) -> dict:
        """Models of every provider, fetched concurrently and each under the model-list timeout. A provider without a
        login is named under ``not_logged_in``. A provider whose live list times out or fails is named under
        ``upstream_errors`` and contributes its last good list, else its static ids, else nothing."""
        providers = sorted(self._backends)
        listings = await asyncio.gather(*(self._listing(provider) for provider in providers))
        data, not_logged_in, upstream_errors = [], {}, {}
        for provider, listing in zip(providers, listings):
            if listing.not_logged_in is not None:
                not_logged_in[provider] = listing.not_logged_in
            if listing.failure is not None:
                upstream_errors[provider] = {"status": listing.failure.status, "code": listing.failure.code,
                                             "message": listing.failure.message,
                                             **(listing.failure.detail.diagnostics() if listing.failure.detail else {})}
            data += [{"id": f"{provider}/{name}", "object": "model", "owned_by": provider} for name in listing.names]
        result: dict = {"object": "list", "data": data}
        if not_logged_in:
            result["not_logged_in"] = not_logged_in
        if upstream_errors:
            result["upstream_errors"] = upstream_errors
        return result

    async def _listing(self, provider: str) -> _Listing:
        cached = self._model_cache.get(provider)
        if cached is not None and self._clock() - cached.fetched_at < self._models_cache_ttl_s:
            return _Listing(list(cached.names))
        deadline = asyncio.timeout(self._models_timeout_s)
        started = time.monotonic()
        try:
            async with deadline:
                names = await self._backends[provider].models()
        except LoginRequired as exc:
            return _Listing([], not_logged_in=str(exc))
        except TimeoutError:
            if not deadline.expired():
                raise
            failure = UpstreamError(f"{provider} model list did not answer within {self._models_timeout_s:g} s",
                                    status=504, code="upstream_timeout")
        except UpstreamError as exc:
            failure = exc
        else:
            self._model_cache[provider] = _CachedList(list(names), self._clock())
            return _Listing(list(names))
        names, served = self._fallback(provider, cached)
        log.warning("upstream %s model list failed after %.2f s (%s); serving %s", provider,
                    time.monotonic() - started, failure.message, served)
        return _Listing(names, failure=failure)

    def _fallback(self, provider: str, cached: _CachedList | None) -> tuple[list[str], str]:
        """The ids served in place of a failed live list, with a description for the log."""
        if cached is not None:
            age = self._clock() - cached.fetched_at
            return list(cached.names), f"cached list of {_count(len(cached.names))}, fetched {age:.0f} s ago"
        static = self._static_models.get(provider, [])
        if static:
            return list(static), f"static list of {_count(len(static))}"
        return [], "no models"


def _count(models: int) -> str:
    return f"{models} model" if models == 1 else f"{models} models"
