"""Call-scoped transport facts forwarded through the caller's existing event sink."""
from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator
from uuid import uuid4

import httpx

from hybrid.literature.events import EventSink


def identity_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def endpoint(url: str | httpx.URL) -> str:
    """Keep only the origin: paths can contain credentials and account identifiers too."""
    parsed = httpx.URL(url)
    host = parsed.host
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{host}{port}" if parsed.scheme else ""


@dataclass
class Operation:
    name: str
    association_id: str
    call_id: str = field(default_factory=lambda: uuid4().hex)
    records: list[dict[str, Any]] = field(default_factory=list)


_OPERATION: ContextVar[Operation | None] = ContextVar("literature_operation", default=None)
_EXCHANGES: ContextVar[tuple[list[dict[str, Any]], Callable[[], float]] | None] = ContextVar(
    "literature_exchanges", default=None
)


@contextmanager
def operation(name: str, identity: str, sink: EventSink | None) -> Iterator[Operation]:
    context = Operation(name, identity_hash(identity))
    token = _OPERATION.set(context)
    try:
        yield context
    finally:
        _OPERATION.reset(token)
        # Emit after the existing summary event, preserving its ordering contract.
        if sink is not None:
            for record in context.records:
                sink.append("literature_transport", "literature", record)


def new_record(source: str, request_id: str, origin: str, started_at: str, started_s: float) -> dict[str, Any]:
    context = _OPERATION.get()
    record = {
        "source": source, "request_id": request_id, "endpoint": origin,
        "operation": context.name if context is not None else "get",
        "association_id": context.association_id if context is not None else request_id,
        "call_id": context.call_id if context is not None else uuid4().hex,
        "started_at": started_at, "started_s": started_s, "duration_s": 0.0,
        "outcome": "network", "terminal": "success", "error_type": None,
        "attempt_count": 0, "http_request_count": 0, "http_response_count": 0,
        "cooldown_remaining_s": 0.0, "attempts": [],
    }
    if context is not None:
        context.records.append(record)
    return record


@contextmanager
def exchanges(attempt: dict[str, Any], clock: Callable[[], float]) -> Iterator[None]:
    token = _EXCHANGES.set((attempt["exchanges"], clock))
    try:
        yield
    finally:
        _EXCHANGES.reset(token)


def request_started(request: httpx.Request) -> None:
    context = _EXCHANGES.get()
    if context is not None:
        records, clock = context
        records.append({"ordinal": len(records) + 1, "endpoint": endpoint(request.url),
                        "started_s": clock(), "duration_s": None, "http_status": None})


def response_received(response: httpx.Response) -> None:
    context = _EXCHANGES.get()
    if context is not None:
        records, clock = context
        record = records[-1]
        record["duration_s"] = clock() - record["started_s"]
        record["http_status"] = response.status_code


def retry_after(response: httpx.Response, now: str) -> tuple[str, float | None]:
    """Parse for measurement only; the existing retry policy remains in transport.py."""
    value = response.headers.get("Retry-After")
    if value is None:
        return "absent", None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.fromisoformat(now)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return "invalid", None
        return "date", max(0.0, seconds)
    return ("delta", seconds) if math.isfinite(seconds) else ("invalid", None)
