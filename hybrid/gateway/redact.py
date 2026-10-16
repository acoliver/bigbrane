"""Secret scrubbing for everything that leaves the process (responses, logs, error messages)."""
from __future__ import annotations

import logging
import threading

_MIN_LENGTH = 8   # shorter strings are not credentials and would corrupt ordinary text


class Redactor:
    def __init__(self) -> None:
        self._secrets: set[str] = set()
        self._lock = threading.Lock()

    def add(self, *values: str | None) -> None:
        with self._lock:
            self._secrets.update(v for v in values if v and len(v) >= _MIN_LENGTH)

    def scrub(self, text: str) -> str:
        with self._lock:
            secrets = sorted(self._secrets, key=len, reverse=True)
        for secret in secrets:
            text = text.replace(secret, "[redacted]")
        return text

    def scrub_value(self, value):
        """A JSON-shaped value with every string scrubbed, for payloads that carry upstream text."""
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, dict):
            return {key: self.scrub_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.scrub_value(item) for item in value]
        return value


class RedactingFilter(logging.Filter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redactor.scrub(record.getMessage())
        record.args = None
        return True
