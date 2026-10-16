"""models.toml loading and validation.

A models file holds every backend, one or more complete routing tables under ``[routes.<name>]``,
per-role settings under ``[role_settings.<role>]`` and the run budgets. The top-level key
``routing`` names the active table. ``BIGBRANE_ROUTING`` overrides it, and an explicit
``routing`` argument to ``load_config`` (the ``--routing`` flag) overrides both.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

KINDS = ("llxprt", "openai_compat", "anthropic_messages", "gateway")
HTTP_KINDS = ("openai_compat", "anthropic_messages", "gateway")
EFFORTS = ("low", "medium", "high")
ROUTING_ENV = "BIGBRANE_ROUTING"
TOP_LEVEL_KEYS = {"routing", "budgets", "backends", "routes", "role_settings"}


@dataclass(frozen=True)
class BackendConfig:
    name: str
    kind: str
    concurrency: int
    timeout_s: float
    base_url: str | None = None
    api_key_env: str | None = None
    anthropic_version: str = "2023-06-01"
    default_max_output_tokens: int = 8192
    command: str = "llxprt"
    user_config_dir: str | None = None
    web_tools: tuple[str, ...] = ("exa_web_search", "direct_web_fetch")
    # openai_compat (local LM Studio) only: the loaded context every routed model must have before a research run
    # or plumbing check starts (hybrid/localmodels.py). None means the preflight default.
    min_loaded_context: int | None = None


@dataclass(frozen=True)
class RoleEntry:
    backend: str
    model: str          # llxprt: profile name; HTTP backends: the model id sent upstream
    timeout_s: float | None = None


@dataclass(frozen=True)
class RoleSettings:
    """Per-role request settings, applied to every entry of the role in every routing table."""
    effort: str | None = None              # low | medium | high
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class Budgets:
    max_calls: int
    max_tokens: int


@dataclass(frozen=True)
class ModelsConfig:
    backends: dict[str, BackendConfig]
    roles: dict[str, tuple[RoleEntry, ...]]          # the active routing table
    budgets: Budgets
    routing: str = "default"
    routing_source: str = "file"                      # argument | env | file
    routes: dict[str, dict[str, tuple[RoleEntry, ...]]] = field(default_factory=dict)
    role_settings: dict[str, RoleSettings] = field(default_factory=dict)
    path: str = field(default="")

    def settings_for(self, role: str) -> RoleSettings:
        return self.role_settings.get(role, RoleSettings())

    def active_backends(self) -> set[str]:
        return {entry.backend for entries in self.roles.values() for entry in entries}


def _require(table: dict, key: str, where: str):
    if key not in table:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return table[key]


def _positive(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{where}: must be a positive number, got {value!r}")
    return value


def _positive_int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where}: must be a positive integer, got {value!r}")
    return value


def _parse_backend(name: str, table: dict) -> BackendConfig:
    where = f"backends.{name}"
    kind = _require(table, "kind", where)
    if kind not in KINDS:
        raise ConfigError(f"{where}: kind must be one of {KINDS}, got {kind!r}")
    concurrency = int(_positive(_require(table, "concurrency", where), where + ".concurrency"))
    timeout_s = float(_positive(_require(table, "timeout_s", where), where + ".timeout_s"))
    base_url = table.get("base_url")
    if kind in HTTP_KINDS and not base_url:
        raise ConfigError(f"{where}: kind {kind} needs base_url")
    extras = {}
    for key in ("api_key_env", "anthropic_version", "command", "user_config_dir"):
        if key in table:
            extras[key] = table[key]
    if "default_max_output_tokens" in table:
        extras["default_max_output_tokens"] = int(_positive(table["default_max_output_tokens"], where))
    if "web_tools" in table:
        extras["web_tools"] = tuple(table["web_tools"])
    if "min_loaded_context" in table:
        if kind != "openai_compat":
            raise ConfigError(f"{where}.min_loaded_context: only for kind openai_compat (a local server)")
        extras["min_loaded_context"] = _positive_int(table["min_loaded_context"], where + ".min_loaded_context")
    known = {"kind", "concurrency", "timeout_s", "base_url", *extras}
    unknown = set(table) - known
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    return BackendConfig(name=name, kind=kind, concurrency=concurrency, timeout_s=timeout_s,
                         base_url=base_url.rstrip("/") if base_url else None, **extras)


def _parse_entries(where_role: str, entries, backends: dict[str, BackendConfig]) -> tuple[RoleEntry, ...]:
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{where_role}: needs at least one entry")
    parsed = []
    for index, entry in enumerate(entries):
        where = f"{where_role}[{index}]"
        unknown = set(entry) - {"backend", "model", "timeout_s"}
        if unknown:
            raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
        backend = _require(entry, "backend", where)
        if backend not in backends:
            raise ConfigError(f"{where}: unknown backend '{backend}'")
        model = _require(entry, "model", where)
        if not isinstance(model, str) or not model:
            raise ConfigError(f"{where}: model must be a non-empty string")
        timeout = entry.get("timeout_s")
        parsed.append(RoleEntry(backend=backend, model=model,
                                timeout_s=float(_positive(timeout, where)) if timeout is not None else None))
    return tuple(parsed)


def _parse_routes(raw: dict, backends: dict[str, BackendConfig]) -> dict[str, dict[str, tuple[RoleEntry, ...]]]:
    tables = _require(raw, "routes", "root")
    if not isinstance(tables, dict) or not tables:
        raise ConfigError("routes: needs at least one routing table")
    routes = {name: {role: _parse_entries(f"routes.{name}.{role}", entries, backends)
                     for role, entries in table.items()}
              for name, table in tables.items()}
    names = sorted(routes)
    reference = set(routes[names[0]])
    if not reference:
        raise ConfigError(f"routes.{names[0]}: has no roles")
    for name in names[1:]:
        if set(routes[name]) != reference:
            missing = sorted(reference - set(routes[name]))
            extra = sorted(set(routes[name]) - reference)
            raise ConfigError(f"routes.{name}: role set differs from routes.{names[0]} "
                              f"(missing {missing}, extra {extra}); every table must route every role")
    return routes


def _parse_role_settings(raw: dict, roles: set[str]) -> dict[str, RoleSettings]:
    settings = {}
    for role, table in raw.get("role_settings", {}).items():
        where = f"role_settings.{role}"
        if role not in roles:
            raise ConfigError(f"{where}: no routing table has a role named '{role}'")
        unknown = set(table) - {"effort", "max_output_tokens"}
        if unknown:
            raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
        effort = table.get("effort")
        if effort is not None and effort not in EFFORTS:
            raise ConfigError(f"{where}.effort: must be one of {EFFORTS}, got {effort!r}")
        cap = table.get("max_output_tokens")
        if cap is not None:
            cap = _positive_int(cap, where + ".max_output_tokens")
        settings[role] = RoleSettings(effort=effort, max_output_tokens=cap)
    return settings


def _select_routing(raw: dict, routes: dict, routing: str | None) -> tuple[str, str]:
    if routing:
        selected, source = routing, "argument"
    elif os.environ.get(ROUTING_ENV):
        selected, source = os.environ[ROUTING_ENV], "env"
    else:
        selected, source = _require(raw, "routing", "root"), "file"
    if selected not in routes:
        origin = {"argument": "--routing", "env": ROUTING_ENV, "file": "routing"}[source]
        raise ConfigError(f"{origin} = {selected!r} names no routing table; known: {sorted(routes)}")
    return selected, source


def load_config(path: str | Path, routing: str | None = None) -> ModelsConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"models config not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"root: unknown keys {sorted(unknown)}")
    backends = {name: _parse_backend(name, table)
                for name, table in _require(raw, "backends", "root").items()}
    routes = _parse_routes(raw, backends)
    selected, source = _select_routing(raw, routes, routing)
    role_settings = _parse_role_settings(raw, set(routes[selected]))
    for name, table in routes.items():
        for role, entries in table.items():
            if role_settings.get(role, RoleSettings()).effort is None:
                continue
            for entry in entries:
                if backends[entry.backend].kind == "anthropic_messages":
                    raise ConfigError(f"routes.{name}.{role}: backend '{entry.backend}' (anthropic_messages) "
                                      f"cannot apply role_settings.{role}.effort")
    budgets_raw = _require(raw, "budgets", "root")
    budgets = Budgets(max_calls=int(_positive(_require(budgets_raw, "max_calls", "budgets"), "budgets.max_calls")),
                      max_tokens=int(_positive(_require(budgets_raw, "max_tokens", "budgets"), "budgets.max_tokens")))
    return ModelsConfig(backends=backends, roles=routes[selected], budgets=budgets, routing=selected,
                        routing_source=source, routes=routes, role_settings=role_settings, path=str(path))
