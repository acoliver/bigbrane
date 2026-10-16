"""OpenAI-compatible HTTP surface on 127.0.0.1: POST /v1/chat/completions, GET /v1/models, GET /health."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import traceback

from .errors import BadRequest, GatewayError, Unauthorized
from .gateway import Gateway
from .http_min import MAX_HEADER_BYTES, ProtocolError, Request, read_request, write_json, write_response
from .redact import RedactingFilter, Redactor

log = logging.getLogger("bigbrane.gateway")

# The client closed the connection before an answer could be written.
_CLIENT_GONE = (ConnectionResetError, BrokenPipeError)


def error_body(message: str, error_type: str, code: str) -> dict:
    return {"error": {"message": message, "type": error_type, "code": code}}


def sse_body(chat: dict) -> bytes:
    """A completed chat.completion replayed as chat.completion.chunk events (buffered streaming)."""
    choice = chat["choices"][0]
    message = choice["message"]
    base = {"id": chat["id"], "object": "chat.completion.chunk", "created": chat["created"], "model": chat["model"]}
    delta: dict = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
    first = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    last = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
            "usage": chat["usage"]}
    return "".join(f"data: {json.dumps(c)}\n\n" for c in (first, last)).encode() + b"data: [DONE]\n\n"


class GatewayServer:
    def __init__(self, gateway: Gateway, token: str, redactor: Redactor, host: str = "127.0.0.1", port: int = 8765):
        self._gateway = gateway
        self._token = token
        self._redactor = redactor
        self._host = host
        self._port = port
        self._server: asyncio.Server | None = None
        redactor.add(token)
        for existing in [f for f in log.filters if isinstance(f, RedactingFilter)]:
            log.removeFilter(existing)
        log.addFilter(RedactingFilter(redactor))

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle, self._host, self._port, limit=MAX_HEADER_BYTES)
        return self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def _authorized(self, request: Request) -> bool:
        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(supplied.encode(), self._token.encode())

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        started = time.monotonic()
        status, summary, client_gone = 500, "-", False
        try:
            try:
                request = await read_request(reader, writer)
                summary = f"{request.method} {request.path}"
                status = await self._dispatch(request, writer)
            except _CLIENT_GONE:
                raise                       # the peer left, not a bug: the outer handler logs it once
            except ProtocolError as exc:
                status = exc.status
                await write_json(writer, status, error_body(str(exc), "invalid_request_error", "bad_http_request"))
            except GatewayError as exc:
                status = exc.status
                kind = "invalid_request_error" if status < 500 else "api_error"
                await write_json(writer, status, error_body(self._redactor.scrub(exc.message), kind, exc.code))
            except Exception as exc:                                  # a bug: report generically, log scrubbed detail
                log.error("%s", self._redactor.scrub(traceback.format_exc()))
                await write_json(writer, 500, error_body(f"internal error ({type(exc).__name__})", "api_error",
                                                         "internal_error"))
        except _CLIENT_GONE:
            client_gone = True
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000
            if client_gone:
                log.info("%s -> client disconnected after %dms; no response sent", summary, elapsed_ms)
            else:
                log.info("%s -> %s in %dms", summary, status, elapsed_ms)
            writer.close()

    async def _dispatch(self, request: Request, writer: asyncio.StreamWriter) -> int:
        if request.path == "/health" and request.method == "GET":
            await write_json(writer, 200, {"ok": True})
            return 200
        if not self._authorized(request):
            raise Unauthorized("missing or wrong gateway bearer token")
        if request.path == "/v1/models" and request.method == "GET":
            await write_json(writer, 200, self._redactor.scrub_value(await self._gateway.models()))
            return 200
        if request.path == "/v1/chat/completions" and request.method == "POST":
            try:
                body = json.loads(request.body)
            except ValueError as exc:
                raise BadRequest("the request body is not valid JSON") from exc
            if not isinstance(body, dict):
                raise BadRequest("the request body must be a JSON object")
            chat = await self._gateway.chat(body, run_id=request.headers.get("x-bigbrane-run-id"))
            if body.get("stream"):
                await write_response(writer, 200, sse_body(chat), "text/event-stream")
            else:
                await write_json(writer, 200, chat)
            return 200
        raise GatewayError(f"no route for {request.method} {request.path}", status=404, code="not_found")
