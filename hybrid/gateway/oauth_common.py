"""Pieces both providers' OAuth code shares: PKCE, HTTP client construction, token-endpoint errors."""
from __future__ import annotations

import base64
import hashlib
import json
import os

import httpx

from .errors import RefreshFailed, UpstreamError
from .refresher import RefreshRejected

# Token endpoint error codes that mean the refresh token cannot be used again. ChatGPT documents this
# list; Anthropic reports invalid_grant.
REJECTED_CODES = frozenset({"invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
                            "refresh_token_invalidated", "refresh_token_reused"})
HTTP_TIMEOUT_S = 60.0


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def pkce_pair() -> tuple[str, str]:
    """(verifier, S256 challenge), both base64url without padding."""
    verifier = b64url(os.urandom(32))
    return verifier, b64url(hashlib.sha256(verifier.encode()).digest())


def make_client(transport: httpx.AsyncBaseTransport | None, timeout: float = HTTP_TIMEOUT_S) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, timeout=timeout)


def error_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        err = err.get("code") or err.get("type")
    return err if isinstance(err, str) else None


def _short_text(value: object, limit: int = 200) -> str | None:
    return value[:limit] if isinstance(value, str) and value else None


def refresh_failure_details(response: httpx.Response) -> dict:
    """What the token endpoint said, without any credential: HTTP status, ``error``, OpenAI's ``error_reason``
    (for example ``refresh_token_invalidated``), ``error_description`` and the ``x-request-id`` that OpenAI support
    and the devkit#5 reporters ask for. The body is external input, so every field is optional and cut short."""
    try:
        body = response.json()
    except ValueError:
        body = None
    body = body if isinstance(body, dict) else {}
    details = {"http_status": response.status_code, "error": error_code(response),
               "error_reason": _short_text(body.get("error_reason")),
               "error_description": _short_text(body.get("error_description")),
               "request_id": _short_text(response.headers.get("x-request-id"))}
    return {k: v for k, v in details.items() if v is not None}


def raise_for_refresh_failure(provider: str, response: httpx.Response) -> None:
    """Map a failed refresh response to RefreshRejected (login needed) or UpstreamError (try later)."""
    details = refresh_failure_details(response)
    code = details.get("error")
    said = ", ".join(f"{k}={details[k]}" for k in ("error_reason", "request_id") if k in details)
    if response.status_code in (400, 401, 403) and code in REJECTED_CODES:
        raise RefreshRejected(f"the {provider} refresh token was rejected ({code}{'; ' + said if said else ''})",
                              details=details)
    raise RefreshFailed(f"{provider} token refresh failed with HTTP {response.status_code}"
                        f"{f' ({code})' if code else ''}{'; ' + said if said else ''}",
                        status=502 if response.status_code >= 500 else 400, details=details)


def _jwt_exp(token: str) -> int | None:
    """The ``exp`` claim of a JWT-shaped access token, read without verifying it (lifetime metadata only)."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except ValueError:
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    return exp if isinstance(exp, int) and not isinstance(exp, bool) else None


def expires_at(now: float, expires_in: object, access_token: str) -> float:
    """Absolute expiry of a token response. No lifetime is ever guessed.

    ``expires_in`` is used when present. When the response omits it, the access token's JWT ``exp`` is used.
    A response with neither is rejected.
    """
    if expires_in is not None:
        if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)) or expires_in <= 0:
            raise UpstreamError("the token response has an invalid expires_in", code="token_response_invalid")
        return now + expires_in
    exp = _jwt_exp(access_token)
    if exp is None or exp <= now:
        raise UpstreamError("the token response has no expires_in and the access token carries no usable exp claim",
                            code="token_response_invalid")
    return float(exp)
