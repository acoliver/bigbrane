"""llxprt backend with a fake process runner: command line, tool policy, parsing, failures."""
import json
import os
import sys
from pathlib import Path

import pytest

from hybrid.llm import ConfigError, LLMClient, ToolPolicyViolation, TransportError
from hybrid.llm.backends import BackendRequest, LlxprtBackend, ProcessResult, run_process
from hybrid.llm.config import BackendConfig

from .fakes import ListSink, llxprt_stdout

PROFILE = {"version": 1, "provider": "claudecode", "model": "claude-haiku-5-5",
           "ephemeralSettings": {"reasoning.effort": "high", "maxOutputTokens": 40000}}


@pytest.fixture
def user_config(tmp_path):
    root = tmp_path / "llxprt-user"
    (root / "profiles").mkdir(parents=True)
    (root / "oauth").mkdir()
    (root / "keys").mkdir()
    (root / "settings.json").write_text('{"oauthEnabledProviders": {"claudecode": true}}')
    (root / "profiles" / "haiku.json").write_text(json.dumps(PROFILE))
    return root


class Runner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def __call__(self, argv, stdin, cwd, env, timeout_s):
        self.calls.append({"argv": argv, "stdin": stdin.decode(), "cwd": cwd, "env": env, "timeout_s": timeout_s})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_backend(tmp_path, user_config, result, **cfg):
    config = BackendConfig(name="llxprt", kind="llxprt", concurrency=2, timeout_s=77,
                           user_config_dir=str(user_config), **cfg)
    runner = Runner(result)
    return LlxprtBackend(config, tmp_path / "homes", runner), runner


def req(tmp_path, *, allow_web=False, model="haiku", max_output_tokens=None, user="USER TEXT", effort=None):
    return BackendRequest(call_id="call-1", model=model, system="SYSTEM TEXT", user=user, allow_web=allow_web,
                          max_output_tokens=max_output_tokens, timeout_s=33, scratch_dir=tmp_path / "scratch" / "call-1",
                          effort=effort, run_id="run-x")


def profile_arg(argv):
    return json.loads(argv[argv.index("--profile") + 1])


@pytest.mark.asyncio
async def test_pure_call_disables_all_tools_and_passes_prompt_on_stdin(tmp_path, user_config):
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout("answer"), b""))
    reply = await backend.complete(req(tmp_path))
    call = runner.calls[0]
    argv = call["argv"]
    assert argv[0] == "llxprt" and "--profile-load" not in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--approval-mode" not in argv and "--yolo" not in argv       # yolo only with web-only tools
    assert profile_arg(argv)["ephemeralSettings"]["tools.allowed"] == []
    assert profile_arg(argv)["ephemeralSettings"]["reasoning.effort"] == "high"   # profile content is kept
    assert "SYSTEM TEXT" in call["stdin"] and "USER TEXT" in call["stdin"]
    assert call["stdin"].index("SYSTEM TEXT") < call["stdin"].index("USER TEXT")
    assert argv[argv.index("-p") + 1] not in call["stdin"]                # -p is only the short instruction
    assert call["cwd"] == tmp_path / "scratch" / "call-1" and call["cwd"].is_dir()
    assert call["timeout_s"] == 33
    assert reply.text == "answer" and reply.model == "claude-haiku-5-5"
    assert reply.usage == {"input": 1000, "output": 20, "cached": 0}


@pytest.mark.asyncio
async def test_long_prompt_goes_through_stdin_not_argv(tmp_path, user_config):
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(), b""))
    big = "x" * 300_000
    await backend.complete(req(tmp_path, user=big))
    assert big in runner.calls[0]["stdin"]
    assert sum(len(a) for a in runner.calls[0]["argv"] if a != runner.calls[0]["argv"][runner.calls[0]["argv"].index("--profile") + 1]) < 500


@pytest.mark.asyncio
async def test_private_config_home_shares_only_auth_and_has_minimal_prompt(tmp_path, user_config):
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(), b""))
    await backend.complete(req(tmp_path))
    home = Path(runner.calls[0]["env"]["LLXPRT_CONFIG_HOME"])
    assert home == tmp_path / "homes" / "pure"
    assert (home / "oauth").is_symlink() and (home / "oauth").resolve() == (user_config / "oauth").resolve()
    assert (home / "keys").is_symlink()
    assert not (home / "LLXPRT.md").exists() and not (home / "profiles").exists()
    assert "no tools" in (home / "prompts" / "core.md").read_text().lower()
    assert (home / "settings.json").read_text() == (user_config / "settings.json").read_text()


@pytest.mark.asyncio
async def test_web_call_allows_only_web_tools_and_uses_yolo(tmp_path, user_config):
    out = llxprt_stdout("found", tools_total=2, tools_by_name=["exa_web_search", "direct_web_fetch"])
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, out, b""))
    await backend.complete(req(tmp_path, allow_web=True))
    argv = runner.calls[0]["argv"]
    assert profile_arg(argv)["ephemeralSettings"]["tools.allowed"] == ["exa_web_search", "direct_web_fetch"]
    assert argv[argv.index("--approval-mode") + 1] == "yolo"
    home = Path(runner.calls[0]["env"]["LLXPRT_CONFIG_HOME"])
    assert home.name == "web" and "web search" in (home / "prompts" / "core.md").read_text()


@pytest.mark.asyncio
async def test_shell_or_write_tools_are_never_allowed_in_the_allow_list(tmp_path, user_config):
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(), b""),
                                   web_tools=("exa_web_search",))
    await backend.complete(req(tmp_path, allow_web=True))
    allowed = profile_arg(runner.calls[0]["argv"])["ephemeralSettings"]["tools.allowed"]
    assert allowed == ["exa_web_search"]
    assert not {"run_shell_command", "write_file", "replace", "shell"} & set(allowed)


@pytest.mark.asyncio
async def test_tool_use_with_tools_disabled_fails_the_call(tmp_path, user_config):
    out = llxprt_stdout("done", tools_total=1, tools_by_name=["read_file"])
    backend, _ = make_backend(tmp_path, user_config, ProcessResult(0, out, b""))
    with pytest.raises(ToolPolicyViolation, match="1 tool calls"):
        await backend.complete(req(tmp_path))


@pytest.mark.asyncio
async def test_non_web_tool_in_web_mode_fails_the_call(tmp_path, user_config):
    out = llxprt_stdout("done", tools_total=1, tools_by_name=["run_shell_command"])
    backend, _ = make_backend(tmp_path, user_config, ProcessResult(0, out, b""))
    with pytest.raises(ToolPolicyViolation, match="run_shell_command"):
        await backend.complete(req(tmp_path, allow_web=True))


@pytest.mark.asyncio
async def test_missing_tool_stats_cannot_be_verified(tmp_path, user_config):
    payload = json.loads(llxprt_stdout())
    del payload["stats"]["tools"]
    backend, _ = make_backend(tmp_path, user_config, ProcessResult(0, json.dumps(payload).encode(), b""))
    with pytest.raises(ToolPolicyViolation, match="cannot be verified"):
        await backend.complete(req(tmp_path))


@pytest.mark.asyncio
async def test_max_output_tokens_overrides_profile(tmp_path, user_config):
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(), b""))
    await backend.complete(req(tmp_path, max_output_tokens=123))
    assert profile_arg(runner.calls[0]["argv"])["ephemeralSettings"]["maxOutputTokens"] == 123


@pytest.mark.asyncio
async def test_effort_and_cap_override_the_profile_and_are_recorded(tmp_path, user_config):
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(), b""))
    reply = await backend.complete(req(tmp_path, effort="low", max_output_tokens=9000))
    settings = profile_arg(runner.calls[0]["argv"])["ephemeralSettings"]
    assert settings["reasoning.effort"] == "low" and settings["maxOutputTokens"] == 9000
    assert profile_arg(runner.calls[0]["argv"])["provider"] == "claudecode"     # the rest of the profile is kept
    assert (reply.detail["effort"], reply.detail["max_output_tokens"]) == ("low", 9000)
    assert "x-bigbrane-run-id" not in " ".join(runner.calls[0]["argv"])     # gateway only


@pytest.mark.asyncio
async def test_no_effort_keeps_the_profile_value_and_adds_no_model_params(tmp_path, user_config):
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(), b""))
    await backend.complete(req(tmp_path))
    profile = profile_arg(runner.calls[0]["argv"])
    assert profile["ephemeralSettings"]["reasoning.effort"] == "high"
    assert profile["ephemeralSettings"]["maxOutputTokens"] == 40000
    assert "modelParams" not in profile


@pytest.mark.asyncio
async def test_cap_also_replaces_caps_already_in_model_params(tmp_path, user_config):
    codex = {"version": 1, "provider": "codex", "model": "gpt-6.1-sol",
             "modelParams": {"maxOutputTokens": 40000, "max_tokens": 40000, "temperature": 1},
             "ephemeralSettings": {"reasoning.effort": "high"}}
    (user_config / "profiles" / "sol.json").write_text(json.dumps(codex))
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(), b""))
    await backend.complete(req(tmp_path, model="sol", effort="medium", max_output_tokens=5000))
    profile = profile_arg(runner.calls[0]["argv"])
    assert profile["modelParams"] == {"maxOutputTokens": 5000, "max_tokens": 5000, "temperature": 1}
    assert profile["ephemeralSettings"]["reasoning.effort"] == "medium"
    assert profile["ephemeralSettings"]["maxOutputTokens"] == 5000


@pytest.mark.asyncio
async def test_cached_tokens_are_reported(tmp_path, user_config):
    backend, _ = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(prompt=500, candidates=9, cached=300), b""))
    reply = await backend.complete(req(tmp_path))
    assert reply.usage == {"input": 500, "output": 9, "cached": 300}


@pytest.mark.asyncio
@pytest.mark.parametrize("result, message", [
    (ProcessResult(1, b"", b"auth exploded"), "exited 1: auth exploded"),
    (ProcessResult(0, b"not json at all", b""), "not JSON"),
    (ProcessResult(0, b'{"stats": {}}', b""), "lacks 'response'"),
    (ProcessResult(0, llxprt_stdout("   "), b""), "empty response"),
    (TransportError("llxprt timed out after 33s"), "timed out"),
])
async def test_subprocess_failures_are_transport_errors(tmp_path, user_config, result, message):
    backend, _ = make_backend(tmp_path, user_config, result)
    with pytest.raises(TransportError, match=message):
        await backend.complete(req(tmp_path))


@pytest.mark.asyncio
async def test_unknown_profile_is_a_config_error(tmp_path, user_config):
    backend, runner = make_backend(tmp_path, user_config, ProcessResult(0, llxprt_stdout(), b""))
    with pytest.raises(ConfigError, match="profile not found"):
        await backend.complete(req(tmp_path, model="nope"))
    assert runner.calls == []


@pytest.mark.asyncio
async def test_run_process_timeout_kills_and_raises_transport_error(tmp_path):
    with pytest.raises(TransportError, match="timed out"):
        await run_process([sys.executable, "-c", "import time; time.sleep(30)"], b"", tmp_path, dict(os.environ), 0.3)


@pytest.mark.asyncio
async def test_run_process_passes_stdin_and_cwd(tmp_path):
    script = "import sys,os; print(sys.stdin.read().upper() + os.getcwd())"
    result = await run_process([sys.executable, "-c", script], b"abc", tmp_path, dict(os.environ), 10)
    assert result.returncode == 0 and result.stdout.decode().startswith("ABC") and str(tmp_path.resolve()) in result.stdout.decode()


@pytest.mark.asyncio
async def test_run_process_missing_binary_is_transport_error(tmp_path):
    with pytest.raises(TransportError, match="cannot start"):
        await run_process(["/nonexistent/llxprt"], b"", tmp_path, {}, 5)


@pytest.mark.asyncio
async def test_client_runs_llxprt_end_to_end_with_scratch_dir(tmp_path, user_config):
    models = tmp_path / "models.toml"
    models.write_text(f"""
routing = "only"
[budgets]
max_calls = 5
max_tokens = 100000
[backends.llxprt]
kind = "llxprt"
concurrency = 1
timeout_s = 50
user_config_dir = "{user_config}"
[routes.only]
analyst = [ {{ backend = "llxprt", model = "haiku" }} ]
""")
    runner = Runner(ProcessResult(0, llxprt_stdout('{"ok": true}'), b""))
    sink = ListSink()
    client = LLMClient(str(models), str(tmp_path / "run"), sink, process_runner=runner)
    result = await client.complete("analyst", "s", "u", json_schema={"type": "object", "required": ["ok"]})
    assert result.parsed == {"ok": True} and result.backend == "llxprt" and result.model == "claude-haiku-5-5"
    assert runner.calls[0]["cwd"] == (tmp_path / "run").resolve() / "llm" / "scratch" / result.call_id
    assert runner.calls[0]["timeout_s"] == 50
    transcript = json.loads(Path(result.transcript_path).read_text())
    assert transcript["attempts"][0]["detail"]["tools"]["totalCalls"] == 0
