"""Event sink protocol and clock helper shared by the literature modules."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol


class EventSink(Protocol):
    """Anything with `append(type, actor, data)`, e.g. the research ledger."""

    def append(self, type: str, actor: str, data: dict[str, Any]) -> Any: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
