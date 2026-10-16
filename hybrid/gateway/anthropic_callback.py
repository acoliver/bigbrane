"""Loopback listener for the Claude login redirect, ported from LLxprt Code.

Source: packages/providers/src/auth/local-oauth-callback.ts (startLocalOAuthCallback) and
anthropic-oauth-provider.ts. LLxprt tries ports 8765 to 8795 in order, serves ``/callback`` on 127.0.0.1, advertises
``http://localhost:<port>/callback`` as the redirect URI, accepts the first request that carries ``code`` and the
expected ``state``, and gives up after 3 minutes. The pasted-code flow stays available alongside it.
"""
from __future__ import annotations

import asyncio
from urllib.parse import parse_qsl

from .errors import BadRequest
from .http_min import ProtocolError, read_request, write_response

CALLBACK_PATH = "/callback"
CALLBACK_PORT_RANGE = (8765, 8795)
CALLBACK_TIMEOUT_S = 3 * 60.0

_PAGE = ("<html><body style='font-family:sans-serif'><h3>Bigbrane gateway</h3><p>{text}</p></body></html>")
_DONE = _PAGE.format(text="Authorization finished. Return to the terminal.").encode()
_FAILED = _PAGE.format(text="Authorization failed. Return to the terminal and try again.").encode()


class CallbackListener:
    def __init__(self, server: asyncio.Server, redirect_uri: str, result: "asyncio.Future[str]") -> None:
        self._server = server
        self.redirect_uri = redirect_uri
        self._result = result

    async def wait(self) -> str:
        """``code#state`` from the redirect, the form the pasted flow uses."""
        return await self._result

    async def close(self) -> None:
        self._server.close()
        await self._server.wait_closed()


async def start_listener(state: str, ports: range | tuple[int, ...]) -> CallbackListener:
    """Listen on the first free port in ``ports`` (0 asks the OS). Raises BadRequest when none is free."""
    result: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_request(reader, writer)
            if request.path != CALLBACK_PATH:
                await write_response(writer, 404, b"not found", "text/plain")
                return
            params = dict(parse_qsl(request.query))
            code, returned_state = params.get("code"), params.get("state")
            if not code or not returned_state:
                await write_response(writer, 400, _FAILED, "text/html")
                _settle(result, error=BadRequest("the login callback is missing code or state"))
            elif returned_state != state:
                await write_response(writer, 400, _FAILED, "text/html")
                _settle(result, error=BadRequest("the login callback belongs to a different login attempt (state mismatch)"))
            else:
                await write_response(writer, 200, _DONE, "text/html")
                _settle(result, value=f"{code}#{returned_state}")
        except ProtocolError:
            pass
        finally:
            writer.close()

    for port in ports:
        try:
            server = await asyncio.start_server(handle, "127.0.0.1", port)
        except OSError:
            continue
        bound = server.sockets[0].getsockname()[1]
        return CallbackListener(server, f"http://localhost:{bound}{CALLBACK_PATH}", result)
    raise BadRequest("no free port for the login callback")


def _settle(future: "asyncio.Future[str]", *, value: str | None = None, error: Exception | None = None) -> None:
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(value)
