"""Sign in with ChatGPT, ChatGPT plan usage for open-source and locally hosted apps.

Source: https://developers.openai.com/siwc/token-sharing-open-source (overview, sign-in, accounts and
sessions, token reference, errors and recovery). This is not the Codex CLI client id. The first login uses
``client_id=dynamic_agent_client`` plus ``agent_name_hint`` and the host id; the callback returns an issued
``oaiapp_...`` client id that is saved with the tokens and used for every later authorization and refresh.
There is no secret and no dashboard registration.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlencode

import httpx

from .errors import BadRequest, UpstreamError
from .meter import utc_now
from .oauth_common import expires_at, make_client, pkce_pair, raise_for_refresh_failure, b64url
from .paths import ChatGPTConfig
from .refresher import RefreshRejected
from .store import Credentials

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
REVOKE_ATTEMPTS = 3
REVOKE_BACKOFF_S = 0.5
RESOURCE = "https://api.openai.com/v1"
DYNAMIC_CLIENT = "dynamic_agent_client"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = ["openid", "profile", "email", "offline_access", "resource.invoke", PLAN_SCOPE]
CALLBACK_PATH = "/auth/callback"
# The devkit's tokenRequest sends exactly these two headers (packages/local/src/oauth.ts). httpx sets the form
# content type itself; accept is otherwise */*.
TOKEN_HEADERS = {"content-type": "application/x-www-form-urlencoded", "accept": "application/json"}
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


@dataclass(frozen=True)
class PendingLogin:
    url: str
    state: str
    nonce: str
    verifier: str
    redirect_uri: str
    client_id: str | None      # None while registering a new client
    host_id: str

    def display_url(self) -> str:
        """The URL with any id_token_hint removed, safe to print."""
        from urllib.parse import parse_qsl
        base, _, query = self.url.partition("?")
        kept = [(k, v) for k, v in parse_qsl(query) if k != "id_token_hint"]
        return f"{base}?{urlencode(kept)}"


def begin_login(*, host_id: str, config: ChatGPTConfig, saved: Credentials | None, port: int,
                new_registration: bool = False) -> PendingLogin:
    """Authorization request. With ``new_registration`` the saved and configured client ids are ignored and the
    login registers a fresh client with ``dynamic_agent_client``; the host id stays the same. Use that only when
    OpenAI has answered ``invalid_client`` for the saved client: a refresh failure, sign-out or revoked session
    keeps the registered client, and every re-registration adds another app entry in ChatGPT settings."""
    client_id = None if new_registration else (saved.meta.get("client_id") if saved else None) or config.client_id
    verifier, challenge = pkce_pair()
    state, nonce = b64url(os.urandom(24)), b64url(os.urandom(24))
    redirect_uri = f"http://127.0.0.1:{port}{CALLBACK_PATH}"
    params = {"client_id": client_id or DYNAMIC_CLIENT, "ext_agent_host_id": host_id}
    if client_id is None:
        params["agent_name_hint"] = config.agent_name_hint
    elif saved and saved.id_token:
        params["id_token_hint"] = saved.id_token
        if saved.meta.get("email"):
            params["login_hint"] = saved.meta["email"]
    params.update({"response_type": "code", "redirect_uri": redirect_uri, "scope": " ".join(SCOPES),
                   "resource": RESOURCE, "state": state, "nonce": nonce, "code_challenge_method": "S256",
                   "code_challenge": challenge})
    return PendingLogin(f"{AUTHORIZE_URL}?{urlencode(params)}", state, nonce, verifier, redirect_uri, client_id, host_id)


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def verify_id_token(token: str, *, jwks: dict, client_id: str, nonce: str, now: float) -> dict:
    """Verify an RS256 ID token against the issuer's JWKS and return its claims."""
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(payload_b64))
        signature = _b64url_decode(signature_b64)
    except ValueError as exc:
        raise BadRequest("the ID token is malformed") from exc
    if header.get("alg") != "RS256":
        raise BadRequest(f"the ID token uses unsupported algorithm {header.get('alg')!r}")
    key = next((k for k in jwks.get("keys", []) if k.get("kid") == header.get("kid") and k.get("kty") == "RSA"), None)
    if key is None:
        raise BadRequest("the ID token is signed by a key that is not in the issuer's JWKS")
    n, e = int.from_bytes(_b64url_decode(key["n"]), "big"), int.from_bytes(_b64url_decode(key["e"]), "big")
    size = (n.bit_length() + 7) // 8
    digest_info = _SHA256_DIGEST_INFO + hashlib.sha256(f"{header_b64}.{payload_b64}".encode()).digest()
    expected = b"\x00\x01" + b"\xff" * (size - len(digest_info) - 3) + b"\x00" + digest_info
    recovered = pow(int.from_bytes(signature, "big"), e, n).to_bytes(size, "big") if len(signature) <= size else b""
    if not hmac.compare_digest(recovered, expected):
        raise BadRequest("the ID token signature does not verify")
    audience = claims.get("aud")
    if claims.get("iss") != ISSUER:
        raise BadRequest("the ID token issuer is not auth.openai.com")
    if client_id not in (audience if isinstance(audience, list) else [audience]):
        raise BadRequest("the ID token audience is not this client")
    if claims.get("exp", 0) <= now - 60:
        raise BadRequest("the ID token is expired")
    if claims.get("nonce") != nonce:
        raise BadRequest("the ID token nonce does not match this login attempt")
    return claims


class ChatGPTAuth:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self._transport = transport
        self._clock = clock
        self._sleep = sleep

    async def complete_login(self, pending: PendingLogin, callback: dict[str, str]) -> Credentials:
        """Finish a login from the loopback callback's query parameters."""
        if callback.get("state") != pending.state:
            raise BadRequest("the callback state does not match this login attempt")
        if callback.get("error") == "access_denied":
            raise BadRequest("consent was declined, so ChatGPT plan usage is not enabled. Run the login again and "
                             "approve the request, or enable the app in ChatGPT Settings")
        if callback.get("error"):
            raise BadRequest(f"authorization failed: {callback['error']}")
        code = callback.get("code")
        if not code:
            raise BadRequest("the callback carried no authorization code")
        issued = callback.get("client_id")
        if pending.client_id is None:
            if not issued or issued == DYNAMIC_CLIENT:
                raise BadRequest("registration did not return an issued client id; treat the login as incomplete")
            client_id = issued
        else:
            if issued and issued != pending.client_id:
                raise BadRequest("the callback names a different client id than the saved registration; rejecting it")
            client_id = pending.client_id
        form = {"grant_type": "authorization_code", "client_id": client_id, "code": code,
                "code_verifier": pending.verifier, "redirect_uri": pending.redirect_uri, "resource": RESOURCE}
        async with make_client(self._transport) as http:
            try:
                response = await http.post(TOKEN_URL, data=form, headers=TOKEN_HEADERS)
                if response.status_code != 200:
                    raise UpstreamError(f"chatgpt code exchange failed with HTTP {response.status_code}"
                                        f"{self._code_suffix(response)}; start the login again", code="exchange_failed")
                tokens = response.json()
                jwks = (await http.get(JWKS_URL)).json()
            except httpx.TransportError as exc:
                raise UpstreamError(f"chatgpt login endpoints unreachable: {type(exc).__name__}") from exc
        scopes = str(tokens.get("scope") or "").split()
        if PLAN_SCOPE not in scopes:
            raise BadRequest(f"the grant lacks {PLAN_SCOPE}, so ChatGPT plan usage is not enabled for this login. "
                             "Run the login again and approve plan usage, or enable it for the app in ChatGPT "
                             "Settings > Usage")
        id_token = tokens.get("id_token")
        if not id_token:
            raise UpstreamError("the token response has no id_token", code="token_response_invalid")
        claims = verify_id_token(id_token, jwks=jwks, client_id=client_id, nonce=pending.nonce, now=self._clock())
        meta = {"client_id": client_id, "issuer": ISSUER, "subject": claims.get("sub"), "email": claims.get("email"),
                "scopes": sorted(scopes), "ext_agent_host_id": pending.host_id, "saved_at": utc_now()}
        if tokens.get("earliest_refresh_at") is not None:
            meta["earliest_refresh_at"] = tokens["earliest_refresh_at"]       # enforced by the refresher
        return Credentials(access_token=tokens["access_token"], refresh_token=tokens.get("refresh_token"),
                           expires_at=expires_at(self._clock(), tokens.get("expires_in"), tokens["access_token"]), id_token=id_token, meta=meta)

    @staticmethod
    def _code_suffix(response: httpx.Response) -> str:
        try:
            code = response.json().get("error")
        except (ValueError, AttributeError):
            return ""
        return f" ({code})" if isinstance(code, str) else ""

    async def refresh(self, creds: Credentials) -> Credentials:
        client_id = creds.meta.get("client_id")
        if not client_id:
            raise RefreshRejected("the stored login has no issued client id")
        form = {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": creds.refresh_token,
                "resource": RESOURCE}
        async with make_client(self._transport) as http:
            try:
                response = await http.post(TOKEN_URL, data=form, headers=TOKEN_HEADERS)
            except httpx.TransportError as exc:
                raise UpstreamError(f"chatgpt token endpoint unreachable: {type(exc).__name__}",
                                    code="refresh_failed") from exc
        if response.status_code != 200:
            raise_for_refresh_failure("chatgpt", response)
        tokens = response.json()
        if not tokens.get("access_token"):
            raise UpstreamError("the refresh response has no access_token", code="token_response_invalid")
        meta = {**creds.meta, "saved_at": utc_now()}
        if tokens.get("earliest_refresh_at") is not None:
            meta["earliest_refresh_at"] = tokens["earliest_refresh_at"]
        if tokens.get("scope"):
            scopes = str(tokens["scope"]).split()
            if PLAN_SCOPE not in scopes:
                raise RefreshRejected(f"the grant no longer includes {PLAN_SCOPE}")
            meta["scopes"] = sorted(scopes)
        # Refresh tokens are single use and each successful refresh returns a replacement (token reference). The
        # consumed token is never kept as a fallback: sending it again would be a reuse. Without a replacement the
        # record keeps no refresh token, so the next refresh asks for a login instead.
        return Credentials(access_token=tokens["access_token"], refresh_token=tokens.get("refresh_token") or None,
                           expires_at=expires_at(self._clock(), tokens.get("expires_in"), tokens["access_token"]),
                           id_token=tokens.get("id_token") or creds.id_token, meta=meta)

    async def revoke(self, creds: Credentials) -> None:
        """Revoke the refresh token at the issuer's ``revocation_endpoint`` (sign-out, per the profiles-and-sessions
        page): a form POST with ``token``, ``token_type_hint=refresh_token`` and the issued ``client_id``. An empty
        200 is success, also for a token that is already invalid. Network failures and 5xx are retried with
        backoff; anything else raises ``UpstreamError`` with ``code="revocation_failed"``."""
        client_id = creds.meta.get("client_id")
        if not creds.refresh_token or not client_id:
            raise UpstreamError("the stored login has no refresh token or client id to revoke", code="revocation_failed")
        form = {"token": creds.refresh_token, "token_type_hint": "refresh_token", "client_id": client_id}
        endpoint: str | None = None
        failure = "no attempt was made"
        async with make_client(self._transport) as http:
            for attempt in range(REVOKE_ATTEMPTS):
                if attempt:
                    await self._sleep(REVOKE_BACKOFF_S * 2 ** (attempt - 1))
                try:
                    if endpoint is None:
                        discovery = await http.get(DISCOVERY_URL)
                        if discovery.status_code != 200:
                            failure = f"discovery answered HTTP {discovery.status_code}"
                            if discovery.status_code < 500:
                                break
                            continue
                        endpoint = self._revocation_endpoint(discovery)
                    response = await http.post(endpoint, data=form)
                except httpx.TransportError as exc:
                    failure = f"transport error {type(exc).__name__}"
                    continue
                if response.status_code == 200:
                    return
                failure = f"the revocation endpoint answered HTTP {response.status_code}"
                if response.status_code < 500:
                    break
        raise UpstreamError(f"chatgpt revocation not confirmed ({failure})", code="revocation_failed")

    @staticmethod
    def _revocation_endpoint(discovery: httpx.Response) -> str:
        try:
            endpoint = discovery.json().get("revocation_endpoint")
        except (ValueError, AttributeError):
            endpoint = None
        if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
            raise UpstreamError("the issuer's discovery document names no https revocation_endpoint",
                                code="revocation_failed")
        return endpoint
