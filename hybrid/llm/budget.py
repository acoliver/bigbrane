"""Per-run call and token budgets."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Budgets
from .errors import BudgetExceeded


class Budget:
    """Counts backend requests and tokens. All methods are synchronous so that a check and
    its reservation cannot be interleaved by other coroutines on the same event loop."""

    def __init__(self, limits: Budgets, calls: int = 0, tokens: int = 0):
        self.limits = limits
        self.calls = calls
        self.tokens = tokens

    @classmethod
    def from_transcripts(cls, limits: Budgets, calls_dir: Path) -> "Budget":
        """Resume the counters of a run that already has transcripts on disk."""
        calls = tokens = 0
        if calls_dir.is_dir():
            for path in sorted(calls_dir.glob("call-*.json")):
                for attempt in json.loads(path.read_text()).get("attempts", []):
                    calls += 1
                    usage = attempt.get("usage") or {}
                    tokens += usage.get("input", 0) + usage.get("output", 0)
        return cls(limits, calls, tokens)

    def reserve_call(self) -> None:
        if self.calls + 1 > self.limits.max_calls:
            raise BudgetExceeded(f"max_calls={self.limits.max_calls} reached ({self.calls} requests made)")
        if self.tokens >= self.limits.max_tokens:
            raise BudgetExceeded(f"max_tokens={self.limits.max_tokens} reached ({self.tokens} tokens used)")
        self.calls += 1

    def add_usage(self, usage: dict) -> None:
        self.tokens += usage.get("input", 0) + usage.get("output", 0)
