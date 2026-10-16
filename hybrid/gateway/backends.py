"""Upstream calls for the two OAuth backends. Both stream upstream and aggregate to one chat completion."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

from . import anthropic_models
from . import anthropic_retry
from . import anthropic_translate as anthropic_tr
from . import responses_translate as responses_tr
from .chatgpt_pacing import Pacer, cap_effort
from .chatgpt_plan import MANAGE_USAGE_LEAD, PlanAccessGuard
from .errors import UpstreamDetail, UpstreamError
from .oauth_common import make_client
from .refresher import Refresher
from .sse import iter_events

log = logging.getLogger("bigbrane.gateway")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
UPSTREAM_TIMEOUT_S = 900.0
MAX_HEADER_VALUE_CHARS = 200

# Headers LLxprt Code sends for Claude Code subscription OAuth (AnthropicApiExecution.buildAnthropicCustomHeaders
# and AnthropicProviderInternals.instantiateClient). The user agent version is the one LLxprt pins because some
# subscription models reject older claude-cli versions. The Anthropic SDK LLxprt builds on adds
# anthropic-version and, because LLxprt sets dangerouslyAllowBrowser, anthropic-dangerous-direct-browser-access.
# Per-request betas (extended cache TTL, thinking binding controls) are merged in by AnthropicBackend.
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_BASE_BETAS = ("oauth-2025-04-20", "interleaved-thinking-2025-05-14")
ANTHROPIC_BETA = ", ".join(ANTHROPIC_BASE_BETAS)
ANTHROPIC_USER_AGENT = "claude-cli/2.1.293 (external, cli)"


@dataclass(frozen=True)
class Completion:
    chat: dict
    usage: dict
    resolved_model: str
    dropped: tuple[str, ...]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _error_fields(node: object) -> tuple[str, str | None, str | None]:
    """Message, code and type from one error node. A string is the message. An object carries message (or
    detail or error), code (or error_code) and type; an object whose message is itself an object is unwrapped,
    so its text and code are kept. OpenAI's authorization layer answers ``{"detail": {"error_code": ...}}``."""
    if isinstance(node, str):
        return node, None, None
    if not isinstance(node, dict):
        return ("" if node is None else json.dumps(node)), None, None
    code, error_type = _text(node.get("code")) or _text(node.get("error_code")), _text(node.get("type"))
    raw = node.get("message") or node.get("detail") or node.get("error") or ""
    if isinstance(raw, dict):
        message, inner_code, inner_type = _error_fields(raw)
        return message, code or inner_code, error_type or inner_type
    if isinstance(raw, str):
        return raw, code, error_type
    return json.dumps(raw), code, error_type


def _parse_error_body(body: bytes) -> tuple[str, str | None, str | None]:
    """Message, code and type from an upstream error body. The body is external input, so any JSON shape is
    accepted; text that is not JSON (or an empty body) is reported as the message."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return body[:200].decode(errors="replace").strip(), None, None
    if isinstance(parsed, dict):
        return _error_fields(parsed.get("error") or parsed)
    return _error_fields(parsed)


def upstream_http_error(provider: str, status: int, body: bytes, retry_after_s: float | None = None, *,
                        request_id: str | None = None, root_error_code: str | None = None,
                        authorization_error: str | None = None, lead: str = "") -> UpstreamError:
    message, code, error_type = _parse_error_body(body)
    message = message[:300]
    detail = UpstreamDetail(http_status=status, message=message, code=code, error_type=error_type,
                            retry_after_s=retry_after_s, request_id=request_id, root_error_code=root_error_code,
                            authorization_error=authorization_error)
    error_code = code or error_type
    text = lead + f"{provider} returned HTTP {status}" + (f": {message}" if message else f" ({code})" if code else "")
    if status == 401:
        return UpstreamError(text + " (credentials refused after a refresh)", status=401,
                             code="upstream_unauthorized", detail=detail)
    if status == 429:
        return UpstreamError(text, status=429, code=error_code or "rate_limited", detail=detail)
    if status >= 500:
        return UpstreamError(text, status=502, code=error_code or "upstream_error", detail=detail)
    return UpstreamError(text, status=status if 400 <= status < 500 else 502,
                         code=error_code or "upstream_rejected", detail=detail)


class _StreamingBackend:
    provider: str
    url: str

    def __init__(self, refresher: Refresher, transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._refresher = refresher
        self._transport = transport
        self._clock = clock

    def _headers(self, access_token: str) -> dict:
        raise NotImplementedError

    def _request_headers(self, access_token: str, payload: dict) -> dict:
        """Headers for one request. A backend whose headers depend on the payload overrides this."""
        return self._headers(access_token)

    def _refusal(self, response: httpx.Response, body: bytes) -> UpstreamError:
        """The error for a non-200 answer. A backend that reads more than the body (headers) overrides this."""
        return upstream_http_error(self.provider, response.status_code, body)

    def _observe(self, response: httpx.Response) -> None:
        """Called with each 200 answer before its stream is read."""

    def _note_unauthorized(self, response: httpx.Response) -> None:
        """Called with a 401 that is about to be retried once after a forced refresh."""

    async def _consume(self, response: httpx.Response):
        raise NotImplementedError

    async def _post(self, payload: dict):
        """POST with the stored token. One retry only if the API refuses a token that has not expired."""
        creds = await self._refresher.credentials(self.provider)
        for attempt in (0, 1):
            async with make_client(self._transport, UPSTREAM_TIMEOUT_S) as http:
                try:
                    async with http.stream("POST", self.url, headers=self._request_headers(creds.access_token, payload),
                                           json=payload) as response:
                        if response.status_code == 401 and attempt == 0:
                            self._note_unauthorized(response)
                            await response.aread()
                        elif response.status_code != 200:
                            raise self._refusal(response, await response.aread())
                        else:
                            self._observe(response)
                            return await self._consume(response)
                except httpx.TransportError as exc:
                    raise UpstreamError(f"{self.provider} request failed in transit: {type(exc).__name__}",
                                        code="upstream_unreachable") from exc
            creds = await self._refresher.credentials(self.provider, rejected_access_token=creds.access_token)
        raise AssertionError("unreachable")


class AnthropicBackend(_StreamingBackend):
    provider = "anthropic"
    url = ANTHROPIC_URL

    def __init__(self, refresher: Refresher, transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], float] = time.time, *, retry: anthropic_retry.RetryPolicy | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        super().__init__(refresher, transport, clock)
        self._retry = retry or anthropic_retry.RetryPolicy()
        self._sleep = sleep
        self._rate_limits = anthropic_retry.RateLimitInfo()      # from the last answer, as LLxprt keeps it

    def _headers(self, access_token: str, payload: dict | None = None) -> dict:
        betas = list(ANTHROPIC_BASE_BETAS) + anthropic_tr.extra_betas(payload or {})
        return {"authorization": f"Bearer {access_token}", "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta": ", ".join(betas), "user-agent": ANTHROPIC_USER_AGENT,
                "anthropic-dangerous-direct-browser-access": "true",
                "content-type": "application/json", "accept": "application/json"}

    def _request_headers(self, access_token: str, payload: dict) -> dict:
        return self._headers(access_token, payload)

    def _refusal(self, response: httpx.Response, body: bytes) -> UpstreamError:
        retry_after = anthropic_retry.parse_retry_after(response.headers.get("retry-after"), self._clock())
        return upstream_http_error(self.provider, response.status_code, body, retry_after)

    def _observe(self, response: httpx.Response) -> None:
        self._rate_limits = anthropic_retry.parse_rate_limit_headers(response.headers)

    async def _post(self, payload: dict):
        """The shared POST (with its single forced-refresh retry on 401), retried per LLxprt's policy on 429, 5xx
        (including 529), in-stream overloaded/rate-limit/api errors and transport failures."""
        wait = anthropic_retry.throttle_wait_s(self._rate_limits, self._clock())
        if wait > 0:
            log.info("anthropic rate limit nearly used up; waiting %.1fs for the reset", wait)
            await self._sleep(wait)

        def note(attempt: int, error: UpstreamError, delay: float) -> None:
            log.warning("anthropic attempt %d/%d failed (%s); retrying in %.1fs", attempt, self._retry.max_attempts,
                        error.code, delay)

        return await anthropic_retry.with_retries(lambda: super(AnthropicBackend, self)._post(payload), self._retry,
                                                  sleep=self._sleep, on_retry=note)

    async def _consume(self, response: httpx.Response) -> dict:
        accumulator = anthropic_tr.MessageAccumulator()
        async for _event, data in iter_events(response):
            accumulator.feed(data)
        if not accumulator.finished:
            raise UpstreamError("anthropic stream ended before message_stop", code="stream_incomplete")
        return accumulator.message

    async def complete(self, body: dict, model: str) -> Completion:
        payload, context = anthropic_tr.build_request(body, model)
        message = await self._post(payload)
        chat, usage = anthropic_tr.to_chat(message, context, self._clock())
        return Completion(chat, usage, chat["model"] or model, context.dropped)

    async def models(self) -> list[str]:
        """The model list LLxprt's claudecode alias serves (claudecode.config staticModels)."""
        return anthropic_models.model_ids()


class ChatGPTBackend(_StreamingBackend):
    """Sign in with ChatGPT inference. Calls are limited to ``max_concurrency`` at a time and paced by ``pacer``
    (the docs publish no allowance, so both are precautions), and stop for the life of the process after an
    admission 403."""

    provider = "chatgpt"
    url = OPENAI_RESPONSES_URL

    def __init__(self, refresher: Refresher, transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], float] = time.time, *, guard: PlanAccessGuard, pacer: Pacer) -> None:
        super().__init__(refresher, transport, clock)
        self._guard = guard
        self._pacer = pacer
        self._slots = asyncio.Semaphore(pacer.policy.max_concurrency)

    @property
    def pacer(self) -> Pacer:
        return self._pacer

    def _headers(self, access_token: str) -> dict:
        return {"authorization": f"Bearer {access_token}", "content-type": "application/json",
                "accept": "text/event-stream", "user-agent": "bigbrane-gateway/0.1"}

    @staticmethod
    def _diagnostics(response: httpx.Response) -> dict[str, str | None]:
        """The headers OpenAI asks apps to keep when a call is refused (request id) plus the authorization layer's
        own codes. The values are external input, so they are cut to a bounded length."""
        def header(name: str) -> str | None:
            value = response.headers.get(name)
            return value[:MAX_HEADER_VALUE_CHARS] if value else None
        return {"request_id": header("x-request-id"), "root_error_code": header("x-openai-ide-root-error-code"),
                "authorization_error": header("x-openai-authorization-error")}

    def _refusal(self, response: httpx.Response, body: bytes) -> UpstreamError:
        status = response.status_code
        lead = MANAGE_USAGE_LEAD if status in (403, 429) else ""
        return upstream_http_error(self.provider, status, body, lead=lead, **self._diagnostics(response))

    def _note_unauthorized(self, response: httpx.Response) -> None:
        shown = "".join(f" {name}={value}" for name, value in self._diagnostics(response).items() if value)
        log.warning("upstream chatgpt answered 401; refreshing once and retrying%s", shown)

    async def _consume(self, response: httpx.Response) -> dict:
        accumulator = responses_tr.ResponseAccumulator()
        async for event, data in iter_events(response):
            accumulator.feed(event, data)
        if accumulator.result is None:
            raise UpstreamError("chatgpt stream ended without response.completed", code="stream_incomplete")
        return accumulator.result

    async def _post(self, payload: dict):
        async with self._slots:
            await self._guard.check()      # before the pacing wait: a refused login fails at once
            await self._pacer.admit()
            await self._guard.check()      # re-checked after the waits: an earlier call may have been refused meanwhile
            try:
                return await super()._post(payload)
            except UpstreamError as exc:
                if exc.detail is not None and exc.detail.http_status == 403:
                    await self._guard.block(exc)
                raise

    async def complete(self, body: dict, model: str) -> Completion:
        payload, context = responses_tr.build_request(body, model)
        replaced = cap_effort(payload, self._pacer.policy.max_reasoning_effort)
        if replaced is not None:
            log.info("chatgpt pacing: reasoning effort %s lowered to %s", replaced, payload["reasoning"]["effort"])
        result = await self._post(payload)
        chat, usage = responses_tr.to_chat(result, model, self._clock())
        return Completion(chat, usage, chat["model"], context.dropped)

    async def models(self) -> list[str]:
        """The signed-in account's catalog (models with visibility "list"), per OpenAI's models-and-inference docs."""
        async with self._slots:            # the catalog call is paced like inference: it is a token-bearing request
            await self._guard.check()
            await self._pacer.admit()
            await self._guard.check()
            creds = await self._refresher.credentials(self.provider)
            async with make_client(self._transport) as http:
                try:
                    response = await http.get(OPENAI_MODELS_URL,
                                              headers={"authorization": f"Bearer {creds.access_token}"})
                except httpx.TransportError as exc:
                    raise UpstreamError(f"chatgpt model list failed in transit: {type(exc).__name__}") from exc
        if response.status_code != 200:
            error = self._refusal(response, response.content)
            if response.status_code == 403:
                await self._guard.block(error)
            raise error
        return [m["slug"] for m in response.json().get("models", []) if m.get("visibility") == "list"]
