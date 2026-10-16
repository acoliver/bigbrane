"""Claude Code subscription OAuth, ported from LLxprt Code.

Sources: packages/auth/src/flows/anthropic-device-flow.ts (endpoints, client id, scopes, PKCE with the
verifier reused as state, JSON code exchange, form-encoded refresh) and
packages/providers/src/auth/anthropic-oauth-provider.ts (refresh keeps the old refresh token when the
response omits a new one). Login is paste-back: the user opens the authorize URL, approves, and pastes the
``code#state`` string Anthropic displays. As in LLxprt, a loopback callback (anthropic_callback.py) can answer instead.
"""
from __future__ import annotations

import time
from typing import Callable
from urllib.parse import urlencode

import httpx

from .errors import BadRequest, UpstreamError
from .meter import utc_now
from .oauth_common import expires_at, make_client, pkce_pair, raise_for_refresh_failure
from .store import Credentials

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
SCOPES = ["org:create_api_key", "user:profile", "user:inference"]


def authorization_url(verifier: str, challenge: str, redirect_uri: str = REDIRECT_URI) -> str:
    """The verifier doubles as OAuth state, as in LLxprt (and OpenCode before it).

    ``redirect_uri`` is Anthropic's code page (paste-back) unless a loopback callback listener is waiting.
    """
    params = {"code": "true", "client_id": CLIENT_ID, "response_type": "code", "redirect_uri": redirect_uri,
              "scope": " ".join(SCOPES), "code_challenge": challenge, "code_challenge_method": "S256",
              "state": verifier}
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def credentials_from_token_response(data: dict, now: float, previous: Credentials | None = None) -> Credentials:
    access = data.get("access_token")
    if not isinstance(access, str) or not access:
        raise UpstreamError("anthropic token response has no access_token", code="token_response_invalid")
    refresh = data.get("refresh_token") or (previous.refresh_token if previous else None)
    meta = {**(previous.meta if previous else {}), "client_id": CLIENT_ID, "saved_at": utc_now()}
    if data.get("scope"):
        meta["scope"] = data["scope"]
    return Credentials(access_token=access, refresh_token=refresh, expires_at=expires_at(now, data.get("expires_in"), access),
                       meta=meta)


class AnthropicAuth:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, clock: Callable[[], float] = time.time):
        self._transport = transport
        self._clock = clock

    async def exchange_code(self, pasted: str, verifier: str, redirect_uri: str = REDIRECT_URI) -> Credentials:
        """``redirect_uri`` must be the one the authorize URL carried."""
        code, _, state = pasted.strip().partition("#")
        if not code:
            raise BadRequest("no authorization code was pasted")
        if state and state != verifier:
            raise BadRequest("the pasted code belongs to a different login attempt (state mismatch); start again")
        body = {"grant_type": "authorization_code", "code": code, "state": state or verifier, "client_id": CLIENT_ID,
                "redirect_uri": redirect_uri, "code_verifier": verifier}
        async with make_client(self._transport) as client:
            try:
                response = await client.post(TOKEN_URL, json=body)
            except httpx.TransportError as exc:
                raise UpstreamError(f"anthropic token endpoint unreachable: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise UpstreamError(f"anthropic code exchange failed with HTTP {response.status_code}",
                                code="exchange_failed")
        return credentials_from_token_response(response.json(), self._clock())

    async def refresh(self, creds: Credentials) -> Credentials:
        form = {"grant_type": "refresh_token", "refresh_token": creds.refresh_token, "client_id": CLIENT_ID}
        async with make_client(self._transport) as client:
            try:
                response = await client.post(TOKEN_URL, data=form)
            except httpx.TransportError as exc:
                raise UpstreamError(f"anthropic token endpoint unreachable: {type(exc).__name__}",
                                    code="refresh_failed") from exc
        if response.status_code != 200:
            raise_for_refresh_failure("anthropic", response)
        return credentials_from_token_response(response.json(), self._clock(), previous=creds)


def new_login(redirect_uri: str = REDIRECT_URI) -> tuple[str, str]:
    """(authorize URL, verifier). Keep the verifier for exchange_code."""
    verifier, challenge = pkce_pair()
    return authorization_url(verifier, challenge, redirect_uri), verifier
