"""Public data types for hybrid/llm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResult:
    call_id: str
    role: str
    backend: str          # "llxprt" | "openai_compat" | "anthropic_messages" | "gateway"
    model: str            # resolved model id
    text: str             # final text
    parsed: dict | None   # set when json_schema given and parsing+validation succeeded
    usage: dict           # {"input": int, "output": int, "cached": int} best effort
    latency_s: float
    transcript_path: str  # file with full prompt+response


class EventSink(Protocol):
    """Anything that can record an event. The research ledger satisfies this."""

    def append(self, type: str, actor: str, data: dict) -> dict: ...
