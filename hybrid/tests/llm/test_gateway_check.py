"""Pre-run gateway check against mocked transports: token lookup, reachability, model ids, fix-it commands."""
from pathlib import Path

import httpx
import pytest

from hybrid.llm import GatewayUnavailable, check_gateways, load_config
from hybrid.llm.gateway_check import gateway_token, required_models

MODELS_TOML = Path(__file__).resolve().parents[2] / "config" / "models.toml"
SERVE = ".venv-hybrid/bin/python -m hybrid.gateway serve --port 8765"
ALL_IDS = ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5",
           "chatgpt/gpt-6.1-sol", "chatgpt/gpt-5.6-terra"]


@pytest.fixture
def gateway_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path))
    monkeypatch.delenv("BIGBRANE_GATEWAY_TOKEN", raising=False)
    return tmp_path


def models_reply(ids, not_logged_in=None):
    body = {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}
    if not_logged_in:
        body["not_logged_in"] = not_logged_in
    return body


def transport(seen, status=200, body=None):
    def handler(request: httpx.Request):
        seen.append({"url": str(request.url), "auth": request.headers.get("authorization")})
        return httpx.Response(status, json=body if body is not None else models_reply(ALL_IDS))
    return httpx.MockTransport(handler)


def test_llxprt_routing_needs_no_gateway(gateway_home):
    seen = []
    assert check_gateways(load_config(MODELS_TOML), transport(seen)) == []
    assert seen == []


def test_reachable_gateway_with_env_token_passes(gateway_home, monkeypatch):
    monkeypatch.setenv("BIGBRANE_GATEWAY_TOKEN", "env-token")
    seen = []
    config = load_config(MODELS_TOML, routing="gateway")
    [record] = check_gateways(config, transport(seen))
    assert seen == [{"url": "http://127.0.0.1:8765/v1/models", "auth": "Bearer env-token"}]
    assert record["token_source"] == "env BIGBRANE_GATEWAY_TOKEN"
    assert record["models"] == sorted(ALL_IDS[:4])
    assert required_models(config, "gateway") == sorted(ALL_IDS[:4])


def test_token_file_is_used_when_env_is_unset(gateway_home):
    (gateway_home / "run-token").write_text("file-token\n")
    seen = []
    [record] = check_gateways(load_config(MODELS_TOML, routing="gateway"), transport(seen))
    assert seen[0]["auth"] == "Bearer file-token"
    assert record["token_source"] == str(gateway_home / "run-token")


def test_no_token_anywhere_prints_serve_and_login_commands(gateway_home):
    with pytest.raises(GatewayUnavailable) as info:
        check_gateways(load_config(MODELS_TOML, routing="gateway"), transport([]))
    message = str(info.value)
    assert "no token" in message and SERVE in message
    assert ".venv-hybrid/bin/python -m hybrid.gateway login anthropic" in message
    assert ".venv-hybrid/bin/python -m hybrid.gateway login chatgpt" in message


def test_unreachable_gateway_prints_serve_command(gateway_home):
    (gateway_home / "run-token").write_text("t\n")

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(GatewayUnavailable, match="ConnectError") as info:
        check_gateways(load_config(MODELS_TOML, routing="gateway"), httpx.MockTransport(refuse))
    assert SERVE in str(info.value)


def test_rejected_token_explains_how_to_refresh_it(gateway_home, monkeypatch):
    monkeypatch.setenv("BIGBRANE_GATEWAY_TOKEN", "stale")
    with pytest.raises(GatewayUnavailable, match="rejected the bearer token from env BIGBRANE_GATEWAY_TOKEN") as info:
        check_gateways(load_config(MODELS_TOML, routing="gateway"), transport([], status=401, body={"error": {}}))
    assert f"export BIGBRANE_GATEWAY_TOKEN=$(cat {gateway_home / 'run-token'})" in str(info.value)


def test_missing_login_names_the_provider_and_its_login_command(gateway_home):
    (gateway_home / "run-token").write_text("t\n")
    body = models_reply(ALL_IDS[:3], not_logged_in={"chatgpt": "chatgpt login needed (no stored login)"})
    with pytest.raises(GatewayUnavailable) as info:
        check_gateways(load_config(MODELS_TOML, routing="gateway"), transport([], body=body))
    message = str(info.value)
    assert "chatgpt/gpt-6.1-sol" in message and "chatgpt login needed" in message
    assert ".venv-hybrid/bin/python -m hybrid.gateway login chatgpt" in message
    assert "login anthropic" not in message


def test_logged_in_provider_without_the_model_id_says_so(gateway_home):
    (gateway_home / "run-token").write_text("t\n")
    body = models_reply([i for i in ALL_IDS if i != "anthropic/claude-haiku-5-5"])
    with pytest.raises(GatewayUnavailable, match="not in its catalog"):
        check_gateways(load_config(MODELS_TOML, routing="gateway"), transport([], body=body))


def test_provider_whose_model_list_failed_is_named_with_the_upstream_message(gateway_home):
    (gateway_home / "run-token").write_text("t\n")
    body = models_reply([i for i in ALL_IDS if not i.startswith("chatgpt/")])
    body["upstream_errors"] = {"chatgpt": {"status": 403, "code": "insufficient_scope",
                                           "message": "chatgpt returned HTTP 403: Your key lacks the scope"}}
    with pytest.raises(GatewayUnavailable) as info:
        check_gateways(load_config(MODELS_TOML, routing="gateway"), transport([], body=body))
    message = str(info.value)
    assert "chatgpt: chatgpt returned HTTP 403: Your key lacks the scope" in message
    assert "not in its catalog" not in message and "login chatgpt" not in message


def test_server_error_is_reported(gateway_home):
    (gateway_home / "run-token").write_text("t\n")
    with pytest.raises(GatewayUnavailable, match="HTTP 500"):
        check_gateways(load_config(MODELS_TOML, routing="gateway"), transport([], status=500, body={"x": 1}))


def test_empty_token_file_counts_as_missing(gateway_home):
    (gateway_home / "run-token").write_text("\n")
    config = load_config(MODELS_TOML, routing="gateway")
    assert gateway_token(config.backends["gateway"]) is None
