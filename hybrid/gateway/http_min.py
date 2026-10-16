"""A deliberately small HTTP/1.1 server layer on asyncio streams (no third-party server is installed).

One request per connection (``Connection: close``), Content-Length bodies only. That is enough for the
local gateway and for the one-shot login callback.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

MAX_HEADER_BYTES = 64 * 1024
MAX_BODY_BYTES = 64 * 1024 * 1024
_REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
            405: "Method Not Allowed", 411: "Length Required", 413: "Payload Too Large", 429: "Too Many Requests",
            500: "Internal Server Error", 502: "Bad Gateway", 503: "Service Unavailable"}


class ProtocolError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Request:
    method: str
    target: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @property
    def path(self) -> str:
        return self.target.partition("?")[0]

    @property
    def query(self) -> str:
        return self.target.partition("?")[2]


async def read_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> Request:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError as exc:
        raise ProtocolError(400, "request headers too large") from exc
    except asyncio.IncompleteReadError as exc:
        raise ProtocolError(400, "connection closed before the request was complete") from exc
    lines = head.decode("latin-1").split("\r\n")
    try:
        method, target, _version = lines[0].split(" ", 2)
    except ValueError as exc:
        raise ProtocolError(400, "malformed request line") from exc
    headers = {}
    for line in lines[1:]:
        if line:
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
    if "transfer-encoding" in headers:
        raise ProtocolError(411, "chunked request bodies are not supported; send Content-Length")
    raw_length = headers.get("content-length", "0")
    if not raw_length.isdigit():
        raise ProtocolError(400, "invalid Content-Length")
    length = int(raw_length)
    if length > MAX_BODY_BYTES:
        raise ProtocolError(413, "request body too large")
    if length and headers.get("expect", "").lower() == "100-continue":
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
    try:
        body = await reader.readexactly(length) if length else b""
    except asyncio.IncompleteReadError as exc:
        raise ProtocolError(400, "connection closed before the body was complete") from exc
    return Request(method, target, headers, body)


async def write_response(writer: asyncio.StreamWriter, status: int, body: bytes,
                         content_type: str = "application/json") -> None:
    head = (f"HTTP/1.1 {status} {_REASONS.get(status, 'Status')}\r\nContent-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n")
    writer.write(head.encode("latin-1") + body)
    await writer.drain()


async def write_json(writer: asyncio.StreamWriter, status: int, payload: dict) -> None:
    await write_response(writer, status, json.dumps(payload).encode())
