"""Per-call usage log, one JSON object per line, for hybrid/llm to merge into the run ledger."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Meter:
    """Fields: ts, call_id, provider, model (as requested), resolved_model, status, usage {input, output,
    cached}, latency_s, run_id (from the x-bigbrane-run-id request header), dropped (request fields the
    backend does not accept), error."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(self, **fields) -> dict:
        event = {"ts": utc_now(), **fields}
        line = (json.dumps(event, sort_keys=True) + "\n").encode()
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        return event
