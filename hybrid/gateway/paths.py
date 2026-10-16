"""On-disk locations and the optional config file.

State lives under one directory per user (default ``~/.bigbrane-gateway``, override with
``BIGBRANE_GATEWAY_HOME``). It holds the pidfile, the per-run bearer token, the ChatGPT host id, the
usage meter and the optional ``config.toml``. Provider tokens live in the private ``oauth/`` subdirectory
(``fsstore.py``).
"""
from __future__ import annotations

import os
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

from .errors import GatewayError


def state_dir() -> Path:
    root = Path(os.environ.get("BIGBRANE_GATEWAY_HOME") or Path.home() / ".bigbrane-gateway")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def oauth_dir(root: Path) -> Path:
    return root / "oauth"


def pidfile_path(root: Path) -> Path:
    return root / "gateway.pid"


def token_path(root: Path) -> Path:
    return root / "run-token"


def meter_path(root: Path) -> Path:
    return root / "usage.jsonl"


def refresh_journal_path(root: Path) -> Path:
    return root / "refresh.jsonl"


def write_private(path: Path, text: str) -> None:
    """Atomic write with owner-only permissions."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def host_id(root: Path) -> str:
    """Stable opaque ChatGPT ``ext_agent_host_id`` for this machine, created once and reused."""
    path = root / "host-id"
    if path.exists():
        value = path.read_text().strip()
        if not value.startswith("urn:uuid:"):
            raise GatewayError(f"{path} does not hold a urn:uuid host id")
        return value
    value = f"urn:uuid:{uuid.uuid4()}"
    write_private(path, value + "\n")
    return value


@dataclass(frozen=True)
class ChatGPTConfig:
    client_id: str | None = None          # issued id; None means register a new client at first login
    agent_name_hint: str = "Bigbrane"
    callback_port: int = 1455
    max_concurrency: int = 1              # simultaneous ChatGPT calls through this gateway: 1 or 2
    # Pacing (chatgpt_pacing.py). OpenAI publishes no rate or ramp rule for this flow; these are our own
    # precautions after a grant was withdrawn 13 minutes into a burst of concurrent high-effort calls.
    min_start_interval_s: float = 10.0    # at least this long between the starts of two ChatGPT requests
    max_requests_per_hour: int = 60       # sliding one-hour window; requests beyond it wait, they do not fail
    max_reasoning_effort: str = "medium"  # reasoning.effort is lowered to this; "" disables the cap
    proactive_refresh: bool = True        # refresh once the refresh window opens, even with no traffic


@dataclass(frozen=True)
class ModelsConfig:
    # /v1/models asks each provider for its model list. A list that does not arrive within upstream_timeout_s is
    # replaced by the provider's last good list, else its static ids, else nothing.
    upstream_timeout_s: float = 10.0
    cache_ttl_s: float = 600.0            # a successful list is served from memory this long; 0 fetches every time


@dataclass(frozen=True)
class Config:
    chatgpt: ChatGPTConfig = ChatGPTConfig()
    models: ModelsConfig = ModelsConfig()


_CHATGPT_KEYS = {"client_id", "agent_name_hint", "callback_port", "max_concurrency", "min_start_interval_s",
                 "max_requests_per_hour", "max_reasoning_effort", "proactive_refresh"}
_MODELS_KEYS = {"upstream_timeout_s", "cache_ttl_s"}
CHATGPT_CONCURRENCY_CHOICES = (1, 2)
# OpenAI's reasoning.effort values, lowest first. "max" is accepted by some newer models.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def load_config(root: Path) -> Config:
    path = root / "config.toml"
    if not path.exists():
        return Config()
    data = tomllib.loads(path.read_text())
    unknown = set(data) - {"chatgpt", "models"}
    if unknown:
        raise GatewayError(f"{path}: unknown sections {sorted(unknown)}")
    section = data.get("chatgpt", {})
    bad = set(section) - _CHATGPT_KEYS
    if bad:
        raise GatewayError(f"{path}: unknown keys in [chatgpt]: {sorted(bad)}")
    chatgpt = ChatGPTConfig(**section)
    if chatgpt.max_concurrency not in CHATGPT_CONCURRENCY_CHOICES:
        raise GatewayError(f"{path}: [chatgpt] max_concurrency must be 1 or 2, not {chatgpt.max_concurrency!r}")
    interval = chatgpt.min_start_interval_s
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval < 0:
        raise GatewayError(f"{path}: [chatgpt] min_start_interval_s must be a number >= 0")
    hourly = chatgpt.max_requests_per_hour
    if isinstance(hourly, bool) or not isinstance(hourly, int) or hourly < 1:
        raise GatewayError(f"{path}: [chatgpt] max_requests_per_hour must be an integer >= 1")
    if chatgpt.max_reasoning_effort and chatgpt.max_reasoning_effort not in EFFORT_ORDER:
        raise GatewayError(f"{path}: [chatgpt] max_reasoning_effort must be one of {list(EFFORT_ORDER)} or """)
    if not isinstance(chatgpt.proactive_refresh, bool):
        raise GatewayError(f"{path}: [chatgpt] proactive_refresh must be true or false")
    models_section = data.get("models", {})
    bad = set(models_section) - _MODELS_KEYS
    if bad:
        raise GatewayError(f"{path}: unknown keys in [models]: {sorted(bad)}")
    models = ModelsConfig(**models_section)
    timeout = models.upstream_timeout_s
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise GatewayError(f"{path}: [models] upstream_timeout_s must be a number > 0")
    ttl = models.cache_ttl_s
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl < 0:
        raise GatewayError(f"{path}: [models] cache_ttl_s must be a number >= 0")
    return Config(chatgpt=chatgpt, models=models)
