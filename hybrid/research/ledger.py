"""Append-only, hash-chained event ledger for one research run.

Each line of ``ledger.jsonl`` is a sorted-key JSON object. ``hash`` is the SHA-256 of ``prev_hash``
concatenated with the sorted-key JSON of the event without its ``hash`` field, so editing any
event, or removing or reordering one, breaks the chain at a specific ``seq``.
"""
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

GENESIS_HASH = "0" * 64
LEDGER_FILE = "ledger.jsonl"
LOCK_FILE = "ledger.lock"
EVENT_KEYS = {"seq", "ts", "type", "actor", "data", "prev_hash", "hash"}


def sorted_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def event_hash(prev_hash: str, event_without_hash: dict) -> str:
    return hashlib.sha256((prev_hash + sorted_json(event_without_hash)).encode("utf-8")).hexdigest()


def _last_line(f) -> bytes | None:
    f.seek(0, os.SEEK_END)
    pos = f.tell()
    if pos == 0:
        return None
    data = b""
    while True:
        step = min(4096, pos)
        pos -= step
        f.seek(pos)
        data = f.read(step) + data
        stripped = data.rstrip(b"\n")
        newline = stripped.rfind(b"\n")
        if newline != -1 or pos == 0:
            return stripped[newline + 1:]


class Ledger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / LEDGER_FILE
        self._lock_path = self.run_dir / LOCK_FILE

    @contextmanager
    def _locked(self, mode: int):
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, mode)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def append(self, type: str, actor: str, data: dict) -> dict:
        if not isinstance(type, str) or not type:
            raise ValueError(f"event type must be a non-empty string, got {type!r}")
        if not isinstance(actor, str) or not actor:
            raise ValueError(f"event actor must be a non-empty string, got {actor!r}")
        if not isinstance(data, dict):
            raise TypeError(f"event data must be a dict, got {data.__class__.__name__}")
        ts = datetime.now(timezone.utc).isoformat()
        with self._locked(fcntl.LOCK_EX), open(self.path, "a+b") as f:
            last = _last_line(f)
            if last is None:
                seq, prev_hash = 1, GENESIS_HASH
            else:
                previous = json.loads(last)
                seq, prev_hash = previous["seq"] + 1, previous["hash"]
            body = {"seq": seq, "ts": ts, "type": type, "actor": actor, "data": data, "prev_hash": prev_hash}
            event = dict(body, hash=event_hash(prev_hash, body))
            f.write((sorted_json(event) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        return event

    def _read_events_locked(self) -> list[dict]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError(f"{self.path}: last line is unterminated (partial write?)")
        return [json.loads(line) for line in raw.decode("utf-8").splitlines()]

    def verify(self) -> tuple[bool, int | None]:
        """Return (True, None) for an intact chain, else (False, seq of the first bad event)."""
        with self._locked(fcntl.LOCK_SH):
            if not self.path.exists():
                return True, None
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                return False, len(raw.splitlines())
            prev_hash = GENESIS_HASH
            for position, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
                try:
                    event = json.loads(line)
                except ValueError:
                    return False, position
                if not isinstance(event, dict) or set(event) != EVENT_KEYS:
                    return False, position
                if event["seq"] != position or event["prev_hash"] != prev_hash:
                    return False, position
                body = {key: value for key, value in event.items() if key != "hash"}
                if event_hash(prev_hash, body) != event["hash"]:
                    return False, position
                prev_hash = event["hash"]
            return True, None

    def iter_events(self, type: str | None = None) -> Iterator[dict]:
        with self._locked(fcntl.LOCK_SH):
            events = self._read_events_locked()
        for event in events:
            if type is None or event["type"] == type:
                yield event
