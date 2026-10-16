"""On-disk cache of raw HTTP responses, keyed by a hash of the request.

Only successful (HTTP 200) responses are stored. Failures are never cached, so a
rate-limited request is retried on the next run instead of being replayed forever.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def request_key(source: str, url: str, params: dict[str, str]) -> str:
    encoded = json.dumps(
        {"source": source, "url": url, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, source: str, key: str) -> Path:
        return self.root / source / f"{key}.json"

    def get(self, source: str, key: str) -> dict[str, Any] | None:
        path = self.path_for(source, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, source: str, key: str, envelope: dict[str, Any]) -> Path:
        path = self.path_for(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path
