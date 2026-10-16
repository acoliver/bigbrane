"""JsonlSink: a standalone EventSink that appends one JSON object per line."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = sum(1 for _ in self.path.open("rb")) if self.path.exists() else 0

    def append(self, type: str, actor: str, data: dict) -> dict:
        self._seq += 1
        event = {
            "seq": self._seq,
            "time": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "type": type,
            "actor": actor,
            "data": data,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event
