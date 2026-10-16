"""Backend contract shared by every transport."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BackendRequest:
    call_id: str
    model: str
    system: str
    user: str
    allow_web: bool
    max_output_tokens: int | None
    timeout_s: float
    scratch_dir: Path
    effort: str | None = None        # low | medium | high, from role_settings; None keeps the backend default
    run_id: str | None = None        # sent to the gateway as x-bigbrane-run-id


@dataclass(frozen=True)
class BackendReply:
    text: str
    model: str                       # resolved model id reported by the backend
    usage: dict                      # {"input", "output", "cached"}
    detail: dict = field(default_factory=dict)   # backend metadata kept in the transcript


class Backend(Protocol):
    kind: str

    async def complete(self, request: BackendRequest) -> BackendReply: ...
