"""API keys for the literature sources that accept one: OpenAlex and Semantic Scholar.

A key comes from an environment variable, else from a file under ~/.bigbrane/keys (surrounding whitespace
stripped). The file must not be readable by group or others. Keys are read only from the environment and from these files.

The key value lives only in `ApiKey.value`, which `repr` and `str` never show. It is sent as a request
header, never as a query parameter, so it cannot end up in a URL, a cache envelope or a ledger event.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

REDACTED = "***"
KEYS_SUBDIR = Path(".bigbrane") / "keys"


class ApiKeyError(RuntimeError):
    """A key source is unusable. The message never contains the key."""


@dataclass(frozen=True)
class KeySpec:
    source: str
    env_var: str
    file_name: str


KEY_SPECS = {
    "openalex": KeySpec("openalex", "BIGBRANE_OPENALEX_API_KEY", "openalex"),
    "s2": KeySpec("s2", "BIGBRANE_S2_API_KEY", "semantic_scholar"),
}


@dataclass(frozen=True)
class ApiKey:
    source: str
    value: str = field(repr=False)
    origin: str  # "env BIGBRANE_..." or "file <path>"

    def __post_init__(self) -> None:
        if not self.value or any(ch.isspace() or not ch.isprintable() for ch in self.value):
            raise ApiKeyError(f"{self.source} API key from {self.origin} is empty or contains whitespace")

    @property
    def last4(self) -> str:
        return self.value[-4:]

    @property
    def headers(self) -> dict[str, str]:
        if self.source == "openalex":
            return {"Authorization": f"Bearer {self.value}"}
        if self.source == "s2":
            return {"x-api-key": self.value}
        raise AssertionError(f"no key header defined for source {self.source!r}")

    def redact(self, text: str) -> str:
        return text.replace(self.value, REDACTED)

    def __str__(self) -> str:
        return f"{self.source} key {REDACTED}{self.last4} ({self.origin})"


def keys_dir(home: Path | None = None) -> Path:
    return (home if home is not None else Path(os.path.expanduser("~"))) / KEYS_SUBDIR


def load_api_key(
    source: str, *, environ: Mapping[str, str] | None = None, home: Path | None = None
) -> ApiKey | None:
    """The key for `source`, or None when no key is configured.

    Raises ApiKeyError for a key file that group or others can read (even when the environment variable
    wins, because the file is a leak either way), or whose content is unusable."""
    spec = KEY_SPECS[source]
    environ = os.environ if environ is None else environ
    path = keys_dir(home) / spec.file_name
    file_exists = path.is_file()
    if file_exists:
        _require_private(path)
    from_env = environ.get(spec.env_var, "").strip()
    if from_env:
        return ApiKey(source, from_env, f"env {spec.env_var}")
    if not file_exists:
        return None
    return ApiKey(source, path.read_text(encoding="utf-8").strip(), f"file {path}")


def load_api_keys(
    *, environ: Mapping[str, str] | None = None, home: Path | None = None
) -> dict[str, ApiKey]:
    loaded = {source: load_api_key(source, environ=environ, home=home) for source in KEY_SPECS}
    return {source: key for source, key in loaded.items() if key is not None}


def _require_private(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ApiKeyError(
            f"key file {path} is readable by group or others (mode {mode:04o}); run: chmod 600 {path}"
        )
