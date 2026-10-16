"""Shared HTTP plumbing: maps failures to TransportError (fallback) or BackendError (no fallback)."""
from __future__ import annotations

import asyncio

import httpx

from ..errors import BackendError, TransportError


async def request_json(transport: httpx.AsyncBaseTransport | None, method: str, url: str, *,
                       headers: dict, body: dict | None, timeout_s: float) -> dict:
    try:
        # A fresh client per request keeps the code independent of the event loop that created it.
        async with httpx.AsyncClient(transport=transport, timeout=timeout_s) as client:
            async with asyncio.timeout(timeout_s):
                response = await client.request(method, url, headers=headers, json=body)
    except TimeoutError as exc:
        raise TransportError(f"{method} {url} timed out after {timeout_s:g}s") from exc
    except httpx.TransportError as exc:
        raise TransportError(f"{method} {url} failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code >= 500 or response.status_code == 429:
        raise TransportError(f"{method} {url} returned HTTP {response.status_code}: {response.text[:300]}")
    if response.status_code >= 400:
        raise BackendError(f"{method} {url} returned HTTP {response.status_code}: {response.text[:300]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise BackendError(f"{method} {url} returned non-JSON body: {response.text[:200]!r}") from exc
    if not isinstance(payload, dict):
        raise BackendError(f"{method} {url} returned JSON that is not an object")
    return payload
