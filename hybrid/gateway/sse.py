"""Server-sent events parsing for upstream streams."""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .errors import UpstreamError


async def iter_events(response: httpx.Response) -> AsyncIterator[tuple[str | None, dict]]:
    """Yield (event name, decoded JSON data). ``[DONE]`` sentinels are skipped; undecodable data is an error."""
    event: str | None = None
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                payload = "\n".join(data)
                if payload.strip() != "[DONE]":
                    yield event, _decode(payload)
            event, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        payload = "\n".join(data)
        if payload.strip() != "[DONE]":
            yield event, _decode(payload)


def _decode(payload: str) -> dict:
    try:
        value = json.loads(payload)
    except ValueError as exc:
        raise UpstreamError("upstream sent an undecodable stream event", code="stream_invalid") from exc
    if not isinstance(value, dict):
        raise UpstreamError("upstream sent a stream event that is not an object", code="stream_invalid")
    return value
