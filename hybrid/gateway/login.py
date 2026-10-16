"""Interactive logins and the status report. Run by a person on the host, never by a controller."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import parse_qsl

import httpx

from . import anthropic_auth
from .anthropic_callback import CALLBACK_TIMEOUT_S as CLAUDE_CALLBACK_TIMEOUT_S, start_listener
from .chatgpt_auth import CALLBACK_PATH, ChatGPTAuth, begin_login
from .chatgpt_pacing import PacingPolicy
from .refresher import refresh_due_at
from .chatgpt_plan import LOGIN_NOTICE, STATUS_NOTICE, USAGE_URL
from .oauth_common import pkce_pair
from .errors import BadRequest, StoreError, UpstreamError
from .http_min import ProtocolError, read_request, write_response
from .paths import ChatGPTConfig, host_id
from .store import PROVIDERS, Credentials, TokenStore

CALLBACK_TIMEOUT_S = 300.0
_DONE_PAGE = (b"<html><body style='font-family:sans-serif'><h3>Bigbrane gateway</h3>"
              b"<p>Login received. You can close this tab and return to the terminal.</p></body></html>")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def status_lines(store: TokenStore, now: float | None = None) -> list[str]:
    """One line per provider. Shows presence and expiry only; never a token value."""
    now = time.time() if now is None else now
    lines = []
    for provider in PROVIDERS:
        creds = store.load(provider)
        if creds is None:
            lines.append(f"{provider}: not logged in (run: python -m hybrid.gateway login {provider})")
            continue
        remaining = int(creds.expires_at - now)
        expiry = (f"access token valid until {_iso(creds.expires_at)} ({remaining // 60} min left)" if remaining > 0
                  else f"access token expired {_iso(creds.expires_at)} (refreshed on the next request)")
        extra = []
        if creds.meta.get("email"):
            extra.append(f"account {creds.meta['email']}")
        if creds.meta.get("client_id"):
            extra.append(f"client {creds.meta['client_id']}")
        extra.append("refresh token stored" if creds.refresh_token else "NO refresh token")
        lines.append(f"{provider}: logged in, {expiry}; {', '.join(extra)}")
        if provider == "chatgpt":
            lines.append(STATUS_NOTICE)
    return lines


def _calls_in_last_hour(meter_file: Path, provider: str, now: float) -> int:
    """Gateway calls to ``provider`` recorded in the usage log during the last hour (all outcomes)."""
    if not meter_file.exists():
        return 0
    cutoff = now - 3600
    count = 0
    for line in meter_file.read_text().splitlines():
        try:
            event = json.loads(line)
            ts = datetime.strptime(event["ts"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
        except (ValueError, KeyError, TypeError):
            continue
        if event.get("provider") == provider and ts > cutoff:
            count += 1
    return count


def chatgpt_durability_lines(creds: Credentials | None, *, pacing: PacingPolicy, skew_s: float,
                             proactive_refresh: bool, last_refresh: dict | None, meter_file: Path,
                             now: float | None = None) -> list[str]:
    """Pacing policy, the refresh window, the last refresh outcome and the last hour's call count. No token values."""
    now = time.time() if now is None else now
    lines = [pacing.describe(), f"chatgpt calls in the last hour (usage log): {_calls_in_last_hour(meter_file, 'chatgpt', now)}"]
    if creds is not None and creds.refresh_token:
        earliest = creds.earliest_refresh_at
        due = refresh_due_at(creds, skew_s)
        window = (f"earliest_refresh_at {_iso(earliest)}" if earliest is not None else "no earliest_refresh_at")
        mode = "proactively by the running gateway" if proactive_refresh else "on the first request after it"
        lines.append(f"chatgpt refresh: {window}, expires_at {_iso(creds.expires_at)}; refresh due from {_iso(due)} "
                     f"({int((due - now) // 60)} min from now), done {mode}")
    if last_refresh:
        said = ", ".join(f"{k}={last_refresh[k]}" for k in ("http_status", "error", "error_reason", "request_id",
                                                            "seconds_before_expiry", "rotated") if k in last_refresh)
        lines.append(f"chatgpt last refresh: {last_refresh.get('ts')} {last_refresh.get('outcome')}"
                     f"{' (' + said + ')' if said else ''}")
    else:
        lines.append("chatgpt last refresh: none recorded yet")
    return lines


PASTE_PROMPT = "Paste the code shown by Anthropic (code#state; input is hidden): "


async def login_anthropic(store: TokenStore, *, ask: Callable[[str], str], say: Callable[[str], None],
                          open_browser: Callable[[str], object], transport: httpx.AsyncBaseTransport | None = None,
                          clock: Callable[[], float] = time.time,
                          callback_ports: range | tuple[int, ...] | None = None,
                          ask_until: Callable[[str, threading.Event], str | None] | None = None,
                          callback_timeout_s: float = CLAUDE_CALLBACK_TIMEOUT_S) -> Credentials:
    """Paste-back login. With ``callback_ports`` (and ``ask_until``) a loopback listener races the paste, as in
    LLxprt: the browser is sent to ``http://localhost:<port>/callback`` and the first answer wins."""
    started_from = store.snapshot("anthropic")      # fails now, not after the browser step, if the store is unsafe
    auth = anthropic_auth.AnthropicAuth(transport, clock)
    listener = None
    if callback_ports is not None:
        try:
            verifier, challenge = pkce_pair()
            listener = await start_listener(verifier, callback_ports)
        except BadRequest as exc:
            say(f"No loopback callback ({exc}); paste the code instead.")
    if listener is None:
        url, verifier = anthropic_auth.new_login()
        say("Open this URL, sign in with the Claude account to use, and approve:")
        say(url)
        open_browser(url)
        pasted = ask(PASTE_PROMPT)
        creds = await auth.exchange_code(pasted, verifier)
    else:
        assert ask_until is not None, "a callback login needs ask_until"
        paste_url = anthropic_auth.authorization_url(verifier, challenge)
        callback_url = anthropic_auth.authorization_url(verifier, challenge, listener.redirect_uri)
        say("Opening the browser. Sign in with the Claude account to use and approve. If the browser is on another "
            "machine, open this URL there instead and paste the code it shows:")
        say(paste_url)
        open_browser(callback_url)
        try:
            creds = await _race_callback_and_paste(auth, listener, verifier, ask_until, callback_timeout_s, say)
        finally:
            await listener.close()
    store.publish_login("anthropic", creds, started_from)
    say(f"anthropic: saved. Access token valid until {_iso(creds.expires_at)}.")
    return creds


async def _race_callback_and_paste(auth: anthropic_auth.AnthropicAuth, listener, verifier: str,
                                   ask_until: Callable[[str, threading.Event], str | None],
                                   timeout_s: float, say: Callable[[str], None]) -> Credentials:
    cancel = threading.Event()
    paste = asyncio.get_running_loop().run_in_executor(None, ask_until, PASTE_PROMPT, cancel)
    callback = asyncio.ensure_future(listener.wait())
    pending = {paste, callback}
    deadline = asyncio.get_running_loop().time() + timeout_s
    try:
        while pending:
            done, pending = await asyncio.wait(pending, timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                                               return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise BadRequest(f"no login arrived within {int(timeout_s)} seconds")
            for finished in done:
                if finished is callback:
                    try:
                        return await auth.exchange_code(finished.result(), verifier, listener.redirect_uri)
                    except BadRequest as exc:       # a bad callback does not end the paste option
                        say(f"The browser callback was refused: {exc}")
                else:
                    pasted = finished.result()
                    if not pasted:
                        raise BadRequest("no authorization code was pasted")
                    return await auth.exchange_code(pasted, verifier)
        raise BadRequest("the login ended without an authorization code")
    finally:
        cancel.set()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def start_callback_listener(port: int) -> tuple[asyncio.Server, "asyncio.Future[dict[str, str]]", int]:
    """Loopback listener for the OAuth redirect. The future resolves with the callback's query parameters."""
    result: asyncio.Future[dict[str, str]] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_request(reader, writer)
            if request.path != CALLBACK_PATH:
                await write_response(writer, 404, b"not found", "text/plain")
                return
            await write_response(writer, 200, _DONE_PAGE, "text/html")
            if not result.done():
                result.set_result(dict(parse_qsl(request.query)))
        except ProtocolError:
            pass
        finally:
            writer.close()

    try:
        server = await asyncio.start_server(handle, "127.0.0.1", port)
    except OSError as exc:
        raise BadRequest(f"cannot listen on 127.0.0.1:{port} for the login callback ({exc.strerror}). Free the port "
                         f"or set callback_port in the [chatgpt] section of config.toml") from exc
    return server, result, server.sockets[0].getsockname()[1]


async def login_chatgpt(store: TokenStore, root: Path, config: ChatGPTConfig, *, say: Callable[[str], None],
                        open_browser: Callable[[str], object], transport: httpx.AsyncBaseTransport | None = None,
                        clock: Callable[[], float] = time.time, timeout_s: float = CALLBACK_TIMEOUT_S,
                        new_registration: bool = False) -> Credentials:
    """Browser login. ``new_registration`` registers a new client under the same host id; it is for the
    ``invalid_client`` case only (see ``begin_login``)."""
    started_from = store.snapshot("chatgpt")
    saved = started_from.credentials()
    server, callback, port = await start_callback_listener(config.callback_port)
    try:
        pending = begin_login(host_id=host_id(root), config=config, saved=saved, port=port,
                              new_registration=new_registration)
        if new_registration:
            say("Registering a new client with the same host id. Use this only after OpenAI answered invalid_client; "
                "it adds another app entry in ChatGPT settings, and the old entry can be disconnected there.")
        say("Opening the browser for Sign in with ChatGPT. If it does not open, visit:")
        say(pending.display_url())
        open_browser(pending.url)
        try:
            params = await asyncio.wait_for(callback, timeout_s)
        except TimeoutError:
            raise BadRequest(f"no login callback arrived within {int(timeout_s)} seconds") from None
    finally:
        server.close()
        await server.wait_closed()
    creds = await ChatGPTAuth(transport, clock).complete_login(pending, params)
    store.publish_login("chatgpt", creds, started_from)
    say(f"chatgpt: saved. Account {creds.meta.get('email')}, client {creds.meta['client_id']}, "
        f"access token valid until {_iso(creds.expires_at)}.")
    say(LOGIN_NOTICE)
    return creds


async def logout_chatgpt(store: TokenStore, *, say: Callable[[str], None],
                         transport: httpx.AsyncBaseTransport | None = None,
                         sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
    """Revoke the refresh token at the issuer, then delete the local record.

    The provider lock is held throughout, so a gateway refresh cannot rotate the token between the revocation and
    the delete. The record is deleted even when revocation is not confirmed (the sign-out guidance), and the
    message then says so and points to ChatGPT settings.
    """
    auth = ChatGPTAuth(transport, sleep=sleep)
    with store.lock("chatgpt") as held:
        snapshot = held.snapshot()
        if snapshot.raw is None:
            say("chatgpt: no stored login to remove")
            return
        try:
            creds = snapshot.credentials()
        except StoreError:
            creds = None         # an unreadable record is the case logout exists to clear
            say("chatgpt: the stored record is unreadable, so its refresh token cannot be revoked")
        confirmed = False
        if creds is not None:
            try:
                await auth.revoke(creds)
                confirmed = True
            except UpstreamError as exc:
                say(f"chatgpt: {exc}")
        held.delete()
    if confirmed:
        say("chatgpt: refresh token revoked and local login removed")
    else:
        say(f"chatgpt: local login removed; remote revocation was not confirmed. Disconnect the app at {USAGE_URL} "
            "if the grant should end")
