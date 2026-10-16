"""llxprt CLI backend.

Tool control (verified, see README.md): the profile JSON is passed inline with
ephemeralSettings["tools.allowed"] set to [] (pure reasoning) or to the web tools only.
The CLI runs against a private LLXPRT_CONFIG_HOME that shares only auth with the real
one and carries a minimal system prompt. Afterwards stats.tools must show that no
disallowed tool ran.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ..config import BackendConfig
from ..errors import BackendError, ConfigError, ToolPolicyViolation, TransportError
from .base import BackendReply, BackendRequest

PROMPT_DIR = Path(__file__).resolve().parent.parent / "llxprt_prompts"
STDIN_INSTRUCTION = "Follow the instructions in the input above."
SHARED_WITH_REAL_HOME = ("oauth", "keys")
# Output caps a profile may carry in modelParams (llxprt passes modelParams to the request unchanged).
MODEL_PARAM_CAPS = {"max_tokens", "max_output_tokens", "max_completion_tokens", "maxOutputTokens"}


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


ProcessRunner = Callable[[list[str], bytes, Path, dict, float], Awaitable[ProcessResult]]


async def run_process(argv: list[str], stdin: bytes, cwd: Path, env: dict, timeout_s: float) -> ProcessResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=cwd, env=env)
    except OSError as exc:
        raise TransportError(f"cannot start {argv[0]}: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout_s)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise TransportError(f"{argv[0]} timed out after {timeout_s:g}s") from exc
    return ProcessResult(process.returncode, stdout, stderr)


def default_user_config_dir() -> Path:
    override = os.environ.get("LLXPRT_CONFIG_HOME")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Preferences" / "llxprt-code"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "llxprt-code"


def compose_prompt(system: str, user: str) -> str:
    return f"# SYSTEM INSTRUCTIONS\n{system}\n\n# USER REQUEST\n{user}\n"


class LlxprtBackend:
    kind = "llxprt"

    def __init__(self, config: BackendConfig, home_root: Path, runner: ProcessRunner = run_process):
        self.config = config
        self.home_root = Path(home_root)
        self.runner = runner
        self.user_config_dir = (Path(config.user_config_dir).expanduser()
                                if config.user_config_dir else default_user_config_dir())
        if config.command != "llxprt" and not Path(config.command).is_absolute():
            raise ConfigError(f"backend {config.name}: command must be 'llxprt' or an absolute path")

    def _home(self, allow_web: bool) -> Path:
        home = self.home_root / ("web" if allow_web else "pure")
        if not home.exists():
            home.mkdir(parents=True)
            for name in SHARED_WITH_REAL_HOME:
                source = self.user_config_dir / name
                if source.exists():
                    (home / name).symlink_to(source)
            settings = self.user_config_dir / "settings.json"
            if settings.is_file():
                shutil.copyfile(settings, home / "settings.json")
            (home / "prompts").mkdir()
            shutil.copyfile(PROMPT_DIR / ("web.md" if allow_web else "pure.md"), home / "prompts" / "core.md")
        return home

    def _profile(self, model: str, allow_web: bool, max_output_tokens: int | None, effort: str | None) -> dict:
        path = self.user_config_dir / "profiles" / f"{model}.json"
        if not path.is_file():
            raise ConfigError(f"llxprt profile not found: {path}")
        profile = json.loads(path.read_text())
        settings = profile.setdefault("ephemeralSettings", {})
        settings["tools.allowed"] = list(self.config.web_tools) if allow_web else []
        if allow_web and not settings["tools.allowed"]:
            raise ConfigError(f"backend {self.config.name}: web_tools is empty but allow_web was requested")
        if effort is not None:
            settings["reasoning.effort"] = effort
        if max_output_tokens is not None:
            # llxprt's generic cap; claudecode sends it as max_tokens, codex removes it from the request.
            settings["maxOutputTokens"] = max_output_tokens
            # A cap already in modelParams goes to the request as is and would win over the generic one.
            params = profile.get("modelParams") or {}
            for key in MODEL_PARAM_CAPS & set(params):
                params[key] = max_output_tokens
        return profile

    def _check_tool_policy(self, stats: dict, allow_web: bool) -> None:
        tools = stats.get("tools")
        if not isinstance(tools, dict) or "totalCalls" not in tools or "byName" not in tools:
            raise ToolPolicyViolation("llxprt stats.tools missing; tool usage cannot be verified")
        used = set(tools["byName"])
        if allow_web:
            extra = used - set(self.config.web_tools)
            if extra:
                raise ToolPolicyViolation(f"llxprt used tools outside the web allow-list: {sorted(extra)}")
        elif tools["totalCalls"] != 0 or used:
            raise ToolPolicyViolation(
                f"llxprt made {tools['totalCalls']} tool calls ({sorted(used)}) with tools disabled")

    async def complete(self, request: BackendRequest) -> BackendReply:
        home = self._home(request.allow_web)
        profile = self._profile(request.model, request.allow_web, request.max_output_tokens, request.effort)
        argv = [self.config.command, "--profile", json.dumps(profile), "--output-format", "json"]
        if request.allow_web:
            # yolo is safe only because tools.allowed is restricted to web tools above.
            argv += ["--approval-mode", "yolo"]
        argv += ["-p", STDIN_INSTRUCTION]
        request.scratch_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "LLXPRT_CONFIG_HOME": str(home)}
        stdin = compose_prompt(request.system, request.user).encode("utf-8")
        result = await self.runner(argv, stdin, request.scratch_dir, env, request.timeout_s)
        if result.returncode != 0:
            tail = result.stderr.decode("utf-8", "replace")[-500:] or result.stdout.decode("utf-8", "replace")[-500:]
            raise TransportError(f"llxprt exited {result.returncode}: {tail.strip()}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TransportError(f"llxprt stdout is not JSON: {exc}; head={result.stdout[:200]!r}") from exc
        response = payload.get("response")
        stats = payload.get("stats")
        if not isinstance(response, str) or not isinstance(stats, dict):
            raise TransportError("llxprt JSON lacks 'response' or 'stats'")
        self._check_tool_policy(stats, request.allow_web)
        if not response.strip():
            raise TransportError("llxprt returned an empty response")
        models = stats.get("models") or {}
        if not models:
            raise BackendError("llxprt stats.models is empty")
        resolved = max(models, key=lambda name: models[name]["tokens"]["total"])
        usage = {"input": 0, "output": 0, "cached": 0}
        for entry in models.values():
            tokens = entry["tokens"]
            usage["input"] += tokens["prompt"]
            usage["output"] += tokens["total"] - tokens["prompt"]
            usage["cached"] += tokens["cached"]
        detail = {"session_id": payload.get("session_id"), "profile": request.model,
                  "effort": profile["ephemeralSettings"].get("reasoning.effort"),
                  "max_output_tokens": profile["ephemeralSettings"].get("maxOutputTokens"),
                  "tools": stats["tools"], "models": models, "allow_web": request.allow_web}
        return BackendReply(text=response, model=resolved, usage=usage, detail=detail)
