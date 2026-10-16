"""openai_compat, gateway and anthropic_messages backends against mocked transports."""
import json
from pathlib import Path

import httpx
import pytest

from hybrid.llm import BackendError, ConfigError, TransportError
from hybrid.llm.backends import AnthropicMessagesBackend, BackendRequest, OpenAICompatBackend
from hybrid.llm.config import BackendConfig


def request(model="m", **kw):
    return BackendRequest(call_id="call-x", model=model, system="SYS", user="USR", allow_web=False,
                          max_output_tokens=kw.get("max_output_tokens"), timeout_s=5, scratch_dir=Path("/unused"),
                          effort=kw.get("effort"), run_id=kw.get("run_id"))


def oai_config(**kw):
    return BackendConfig(name="lm", kind=kw.pop("kind", "openai_compat"), concurrency=1, timeout_s=5,
                         base_url="http://local/v1", **kw)


def oai_reply(text="hi", model="served-model"):
    return {"id": "c1", "model": model, "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}}


@pytest.mark.asyncio
async def test_openai_request_shape_and_reply_parsing(monkeypatch):
    seen = {}

    def handler(req: httpx.Request):
        seen.update(url=str(req.url), headers=dict(req.headers), body=json.loads(req.content))
        return httpx.Response(200, json=oai_reply("pong"))

    backend = OpenAICompatBackend(oai_config(), httpx.MockTransport(handler))
    reply = await backend.complete(request(max_output_tokens=50))
    assert seen["url"] == "http://local/v1/chat/completions"
    assert "authorization" not in seen["headers"]                  # local server: no key
    assert seen["body"]["model"] == "m" and seen["body"]["max_tokens"] == 50
    assert seen["body"]["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    assert reply.text == "pong" and reply.model == "served-model"
    assert reply.usage == {"input": 11, "output": 3, "cached": 4}


@pytest.mark.asyncio
async def test_openai_api_key_comes_from_named_env_var(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sekret")
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json=oai_reply())

    await OpenAICompatBackend(oai_config(api_key_env="MY_KEY"), httpx.MockTransport(handler)).complete(request())
    assert seen["auth"] == "Bearer sekret"


@pytest.mark.asyncio
async def test_openai_missing_key_env_is_config_error(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    backend = OpenAICompatBackend(oai_config(api_key_env="MY_KEY"), httpx.MockTransport(lambda r: httpx.Response(200, json=oai_reply())))
    with pytest.raises(ConfigError, match="MY_KEY"):
        await backend.complete(request())


@pytest.mark.asyncio
async def test_gateway_is_an_openai_compat_preset(monkeypatch):
    monkeypatch.setenv("GW_TOKEN", "t")
    backend = OpenAICompatBackend(oai_config(kind="gateway", api_key_env="GW_TOKEN"),
                                  httpx.MockTransport(lambda r: httpx.Response(200, json=oai_reply())))
    assert backend.kind == "gateway"
    assert (await backend.complete(request())).text == "hi"


@pytest.mark.asyncio
async def test_gateway_sends_run_id_cap_and_effort(monkeypatch):
    monkeypatch.setenv("GW_TOKEN", "tok-1")
    seen = {}

    def handler(req: httpx.Request):
        seen.update(headers=dict(req.headers), body=json.loads(req.content))
        return httpx.Response(200, json=oai_reply())

    backend = OpenAICompatBackend(oai_config(kind="gateway", api_key_env="GW_TOKEN"), httpx.MockTransport(handler))
    reply = await backend.complete(request("anthropic/claude-haiku-5-5", max_output_tokens=16000, effort="low",
                                           run_id="run-20261010T000322Z"))
    assert seen["headers"]["x-bigbrane-run-id"] == "run-20261010T000322Z"
    assert seen["headers"]["authorization"] == "Bearer tok-1"
    assert seen["body"]["max_tokens"] == 16000 and seen["body"]["reasoning_effort"] == "low"
    assert seen["body"]["model"] == "anthropic/claude-haiku-5-5"
    assert reply.detail["sent"] == {"max_tokens": 16000, "reasoning_effort": "low"}


@pytest.mark.asyncio
async def test_plain_openai_compat_sends_no_run_id_header():
    seen = {}

    def handler(req: httpx.Request):
        seen.update(headers=dict(req.headers), body=json.loads(req.content))
        return httpx.Response(200, json=oai_reply())

    await OpenAICompatBackend(oai_config(), httpx.MockTransport(handler)).complete(request(run_id="run-1"))
    assert "x-bigbrane-run-id" not in seen["headers"]
    assert "reasoning_effort" not in seen["body"] and "max_tokens" not in seen["body"]


@pytest.mark.asyncio
async def test_gateway_token_falls_back_to_the_token_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GW_TOKEN", raising=False)
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path))
    (tmp_path / "run-token").write_text("file-token\n")
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json=oai_reply())

    backend = OpenAICompatBackend(oai_config(kind="gateway", api_key_env="GW_TOKEN"), httpx.MockTransport(handler))
    await backend.complete(request())
    assert seen["auth"] == "Bearer file-token"
    (tmp_path / "run-token").unlink()
    with pytest.raises(ConfigError, match="hybrid.gateway serve --port"):
        await backend.complete(request())


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503, 429])
async def test_openai_server_errors_are_transport_errors(status):
    backend = OpenAICompatBackend(oai_config(), httpx.MockTransport(lambda r: httpx.Response(status, text="down")))
    with pytest.raises(TransportError, match=str(status)):
        await backend.complete(request())


@pytest.mark.asyncio
async def test_openai_client_error_is_not_a_transport_error():
    backend = OpenAICompatBackend(oai_config(), httpx.MockTransport(lambda r: httpx.Response(400, text="bad model")))
    with pytest.raises(BackendError, match="400"):
        await backend.complete(request())


@pytest.mark.asyncio
async def test_openai_connection_error_and_timeout_are_transport_errors():
    def refuse(req):
        raise httpx.ConnectError("refused")

    def slow(req):
        raise httpx.ReadTimeout("slow")

    for handler in (refuse, slow):
        backend = OpenAICompatBackend(oai_config(), httpx.MockTransport(handler))
        with pytest.raises(TransportError):
            await backend.complete(request())


@pytest.mark.asyncio
async def test_openai_empty_content_is_a_transport_error():
    backend = OpenAICompatBackend(oai_config(), httpx.MockTransport(lambda r: httpx.Response(200, json=oai_reply(""))))
    with pytest.raises(TransportError, match="empty"):
        await backend.complete(request())


@pytest.mark.asyncio
async def test_auto_model_lists_models_once_and_skips_embeddings():
    calls = []

    def handler(req: httpx.Request):
        calls.append((req.method, req.url.path))
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "text-embedding-nomic"}, {"id": "chat-one"}, {"id": "chat-two"}]})
        assert json.loads(req.content)["model"] == "chat-one"
        return httpx.Response(200, json=oai_reply(model="chat-one"))

    backend = OpenAICompatBackend(oai_config(), httpx.MockTransport(handler))
    await backend.complete(request("auto"))
    await backend.complete(request("auto"))
    assert [c for c in calls if c[0] == "GET"] == [("GET", "/v1/models")]


@pytest.mark.asyncio
async def test_auto_model_with_nothing_loaded_is_a_transport_error():
    backend = OpenAICompatBackend(oai_config(), httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "embed-only"}]})))
    with pytest.raises(TransportError, match="no chat model"):
        await backend.complete(request("auto"))


# ----- anthropic ------------------------------------------------------------------------

def anthropic_config(**kw):
    return BackendConfig(name="anth", kind="anthropic_messages", concurrency=1, timeout_s=5,
                         base_url="https://api.example", api_key_env="ANTH_KEY", anthropic_version="2023-06-01",
                         default_max_output_tokens=1234, **kw)


@pytest.mark.asyncio
async def test_anthropic_request_shape_and_reply_parsing(monkeypatch):
    monkeypatch.setenv("ANTH_KEY", "k-123")
    seen = {}

    def handler(req: httpx.Request):
        seen.update(url=str(req.url), headers=dict(req.headers), body=json.loads(req.content))
        return httpx.Response(200, json={
            "id": "msg_1", "model": "claude-x", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}],
            "usage": {"input_tokens": 7, "output_tokens": 9, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 3}})

    reply = await AnthropicMessagesBackend(anthropic_config(), httpx.MockTransport(handler)).complete(request("claude-x"))
    assert seen["url"] == "https://api.example/v1/messages"
    assert seen["headers"]["x-api-key"] == "k-123" and seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"] == {"model": "claude-x", "max_tokens": 1234, "system": "SYS",
                            "messages": [{"role": "user", "content": "USR"}]}
    assert reply.text == "Hello world" and reply.model == "claude-x"
    assert reply.usage == {"input": 110, "output": 9, "cached": 100}


@pytest.mark.asyncio
async def test_anthropic_explicit_max_tokens_wins(monkeypatch):
    monkeypatch.setenv("ANTH_KEY", "k")
    seen = {}

    def handler(req):
        seen["max_tokens"] = json.loads(req.content)["max_tokens"]
        return httpx.Response(200, json={"content": [{"type": "text", "text": "x"}], "usage": {}})

    await AnthropicMessagesBackend(anthropic_config(), httpx.MockTransport(handler)).complete(request(max_output_tokens=55))
    assert seen["max_tokens"] == 55


@pytest.mark.asyncio
async def test_anthropic_missing_key_and_errors(monkeypatch):
    monkeypatch.delenv("ANTH_KEY", raising=False)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"content": []}))
    with pytest.raises(ConfigError, match="ANTH_KEY"):
        await AnthropicMessagesBackend(anthropic_config(), transport).complete(request())
    monkeypatch.setenv("ANTH_KEY", "k")
    with pytest.raises(TransportError, match="529|500"):
        await AnthropicMessagesBackend(anthropic_config(), httpx.MockTransport(lambda r: httpx.Response(500, text="x"))).complete(request())
    with pytest.raises(BackendError, match="401"):
        await AnthropicMessagesBackend(anthropic_config(), httpx.MockTransport(lambda r: httpx.Response(401, text="no"))).complete(request())
