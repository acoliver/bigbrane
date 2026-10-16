"""Gateway token lookup and the pre-run reachability check.

The gateway's bearer token comes from the backend's ``api_key_env`` (``BIGBRANE_GATEWAY_TOKEN``) or,
when that variable is unset, from the token file that ``python -m hybrid.gateway serve`` writes
(``~/.bigbrane-gateway/run-token``, or under ``BIGBRANE_GATEWAY_HOME``). The file is read on every
request, so a restarted gateway's new token is picked up without restarting the run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from hybrid.gateway.paths import token_path

from .config import BackendConfig, ModelsConfig
from .errors import LLMError

PYTHON = ".venv-hybrid/bin/python"
PROVIDERS = ("anthropic", "chatgpt")


class GatewayUnavailable(LLMError):
    """The configured gateway cannot serve this run. The message holds the commands that fix it."""


@dataclass(frozen=True)
class GatewayToken:
    value: str
    source: str            # "env BIGBRANE_GATEWAY_TOKEN" or the token file path


def token_file() -> Path:
    # Same root as hybrid.gateway.paths.state_dir(), without creating the directory.
    root = Path(os.environ.get("BIGBRANE_GATEWAY_HOME") or Path.home() / ".bigbrane-gateway")
    return token_path(root)


def gateway_token(config: BackendConfig) -> GatewayToken | None:
    if config.api_key_env and os.environ.get(config.api_key_env):
        return GatewayToken(os.environ[config.api_key_env], f"env {config.api_key_env}")
    path = token_file()
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return GatewayToken(value, str(path))
    return None


def serve_command(config: BackendConfig) -> str:
    port = urlparse(config.base_url).port or 8765
    return f"{PYTHON} -m hybrid.gateway serve --port {port}"


def login_command(provider: str) -> str:
    return f"{PYTHON} -m hybrid.gateway login {provider}"


def _not_running(config: BackendConfig, detail: str) -> GatewayUnavailable:
    logins = "\n".join(f"  {login_command(p)}" for p in PROVIDERS)
    return GatewayUnavailable(
        f"gateway backend '{config.name}' is not usable at {config.base_url}: {detail}\n"
        f"Start it in another terminal (it writes {token_file()}):\n  {serve_command(config)}\n"
        f"If `{PYTHON} -m hybrid.gateway status` shows a missing login, run:\n{logins}")


def required_models(config: ModelsConfig, backend: str) -> list[str]:
    return sorted({entry.model for entries in config.roles.values() for entry in entries if entry.backend == backend})


def check_backend(config: ModelsConfig, backend: str, transport: httpx.BaseTransport | None = None,
                  timeout_s: float = 10.0) -> dict:
    """GET <base_url>/models with the bearer token; every model the active routing sends here must be listed."""
    cfg = config.backends[backend]
    token = gateway_token(cfg)
    if token is None:
        raise _not_running(cfg, f"no token ({cfg.api_key_env} is unset and {token_file()} does not exist)")
    url = f"{cfg.base_url}/models"
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            response = client.get(url, headers={"authorization": f"Bearer {token.value}"})
    except httpx.HTTPError as exc:
        raise _not_running(cfg, f"GET {url} failed ({type(exc).__name__}: {exc})") from exc
    if response.status_code == 401:
        raise GatewayUnavailable(
            f"gateway at {cfg.base_url} rejected the bearer token from {token.source}. The gateway writes a new token "
            f"on every start unless {cfg.api_key_env} was set when it started. Either unset {cfg.api_key_env} so "
            f"{token_file()} is read, or run:\n  export {cfg.api_key_env}=$(cat {token_file()})")
    if response.status_code != 200:
        raise _not_running(cfg, f"GET {url} returned HTTP {response.status_code}: {response.text[:300]}")
    try:
        payload = response.json()
        served = {item["id"] for item in payload["data"]}
    except (ValueError, KeyError, TypeError) as exc:
        raise _not_running(cfg, f"GET {url} returned an unexpected body: {response.text[:300]}") from exc
    needed = required_models(config, backend)
    missing = [model for model in needed if model not in served]
    if missing:
        providers = sorted({model.split("/", 1)[0] for model in missing})
        not_logged_in = payload.get("not_logged_in") or {}
        upstream_errors = payload.get("upstream_errors") or {}
        lines = [f"gateway at {cfg.base_url} does not serve {missing} (routing '{config.routing}' needs them)."]
        for provider in providers:
            reason = not_logged_in.get(provider) or (upstream_errors.get(provider) or {}).get("message")
            lines.append(f"  {provider}: {reason or 'logged in, but the model id is not in its catalog'}")
        logins = [login_command(p) for p in providers if p in not_logged_in]
        if logins:
            lines.append("Log in, then start the run again (the running gateway reads new logins):")
            lines += [f"  {command}" for command in logins]
        else:
            lines.append(f"Check the ids with: curl -s {url} -H \"Authorization: Bearer $(cat {token_file()})\"")
        raise GatewayUnavailable("\n".join(lines))
    return {"backend": backend, "base_url": cfg.base_url, "token_source": token.source, "models": needed}


def check_gateways(config: ModelsConfig, transport: httpx.BaseTransport | None = None) -> list[dict]:
    """Check every gateway backend the active routing table uses. Returns one record per backend."""
    names = sorted(name for name in config.active_backends() if config.backends[name].kind == "gateway")
    return [check_backend(config, name, transport) for name in names]
