"""End to end through the real asyncio HTTP server on a loopback port, with mocked upstream HTTP."""
import json

import httpx

from hybrid.gateway.anthropic_models import model_ids
from hybrid.gateway.server import GatewayServer
from .support import (NOW, anthropic_ok, make_app, make_store, meter_lines, responses_ok, run, seed, sse_response,
                      fixture_json, chat_fixture)

TOKEN = "run-token-for-tests-1234567890"
AUTH = {"authorization": f"Bearer {TOKEN}"}


class Upstream:
    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.anthropic_status = 200
        self.responses_reply = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url.endswith("/v1/messages"):
            if self.anthropic_status != 200:
                return httpx.Response(self.anthropic_status, json={"error": {"type": "rate_limit_error", "message": "slow down"}})
            return anthropic_ok()
        if url.endswith("/v1/responses"):
            return self.responses_reply() if self.responses_reply else responses_ok()
        if url.endswith("/v1/models"):
            return httpx.Response(200, json={"models": [{"slug": "gpt-6.1-sol", "display_name": "Sol", "visibility": "list"},
                                                         {"slug": "hidden-model", "visibility": "hide"}]})
        raise AssertionError(f"unexpected upstream call {url}")


def serve(tmp_path, upstream):
    app = make_app(tmp_path, upstream)
    server = GatewayServer(app.gateway, TOKEN, app.redactor, port=0)
    return app, server


def log_in_both(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", access="anthropic-access-token", refresh="anthropic-refresh-token")
    seed(store, "chatgpt", access="chatgpt-access-token", refresh="chatgpt-refresh-token", meta={"client_id": "oaiapp_x"})


def with_server(tmp_path, scenario, *, anthropic_only=False, upstream=None):
    upstream = upstream or Upstream()
    if anthropic_only:
        seed(make_store(tmp_path), "anthropic")
    else:
        log_in_both(tmp_path)
    app, server = serve(tmp_path, upstream)

    async def go():
        port = await server.start()
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
                return await scenario(client)
        finally:
            await server.close()
    return run(go()), upstream, app


def test_anthropic_chat_completion_end_to_end_with_headers_and_meter(tmp_path):
    async def scenario(client):
        body = chat_fixture("chat_with_tools.json", "anthropic/claude-opus-5-5")
        return await client.post("/v1/chat/completions", json=body, headers={**AUTH, "x-bigbrane-run-id": "run-1"})

    response, upstream, _ = with_server(tmp_path, scenario)
    assert response.status_code == 200
    chat = response.json()
    assert chat["choices"][0]["message"]["content"] == "Paris."
    assert chat["usage"]["prompt_tokens"] == 135 and chat["usage"]["prompt_tokens_details"]["cached_tokens"] == 100

    sent = upstream.requests[0]
    assert str(sent.url) == "https://api.anthropic.com/v1/messages"
    assert sent.headers["authorization"] == "Bearer anthropic-access-token"
    assert sent.headers["anthropic-beta"] == ("oauth-2025-04-20, interleaved-thinking-2025-05-14, "
                                              "extended-cache-ttl-2025-04-11")
    assert sent.headers["anthropic-dangerous-direct-browser-access"] == "true"
    assert sent.headers["user-agent"] == "claude-cli/2.1.293 (external, cli)"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(sent.content) == fixture_json("golden", "anthropic_request_with_tools.json")

    [line] = meter_lines(tmp_path)
    assert line["status"] == "ok" and line["provider"] == "anthropic" and line["run_id"] == "run-1"
    assert line["usage"] == {"input": 135, "output": 7, "cached": 100}
    assert line["resolved_model"] == "claude-opus-5-5" and line["model"] == "anthropic/claude-opus-5-5"
    assert line["call_id"].startswith("call-") and line["latency_s"] >= 0


def test_chatgpt_chat_completion_end_to_end(tmp_path):
    async def scenario(client):
        body = chat_fixture("chat_json_schema.json", "chatgpt/gpt-6.1-sol")
        return await client.post("/v1/chat/completions", json=body, headers=AUTH)

    response, upstream, _ = with_server(tmp_path, scenario)
    assert response.status_code == 200
    sent = upstream.requests[0]
    assert str(sent.url) == "https://api.openai.com/v1/responses"
    assert sent.headers["authorization"] == "Bearer chatgpt-access-token"
    # The golden translation, plus the pacing policy's effort ceiling for a request that names no effort.
    expected = {**fixture_json("golden", "responses_request_json_schema.json"), "reasoning": {"effort": "medium"}}
    assert json.loads(sent.content) == expected
    [line] = meter_lines(tmp_path)
    assert line["usage"] == {"input": 120, "output": 9, "cached": 100}


def test_models_lists_both_providers_with_prefixes(tmp_path):
    async def scenario(client):
        return await client.get("/v1/models", headers=AUTH)

    response, _, _ = with_server(tmp_path, scenario)
    ids = [m["id"] for m in response.json()["data"]]
    assert "anthropic/claude-opus-5-5" in ids and "chatgpt/gpt-6.1-sol" in ids and "chatgpt/hidden-model" not in ids


def test_models_names_providers_without_a_login(tmp_path):
    async def scenario(client):
        return await client.get("/v1/models", headers=AUTH)

    response, _, _ = with_server(tmp_path, scenario, anthropic_only=True)
    body = response.json()
    assert [m["id"] for m in body["data"]] == [f"anthropic/{n}" for n in model_ids()]
    assert "login chatgpt" in body["not_logged_in"]["chatgpt"]


def test_requests_without_the_run_token_are_refused_and_never_reach_upstream(tmp_path):
    async def scenario(client):
        body = chat_fixture("chat_json_schema.json", "anthropic/claude-opus-5-5")
        return [await client.post("/v1/chat/completions", json=body),
                await client.post("/v1/chat/completions", json=body, headers={"authorization": "Bearer wrong"}),
                await client.get("/v1/models", headers={"authorization": "Basic abc"}),
                await client.get("/health")]

    responses, upstream, _ = with_server(tmp_path, scenario)
    assert [r.status_code for r in responses] == [401, 401, 401, 200]
    assert upstream.requests == []


def test_unknown_model_prefix_and_malformed_bodies_are_400(tmp_path):
    async def scenario(client):
        return [await client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []}, headers=AUTH),
                await client.post("/v1/chat/completions", content=b"{nope", headers=AUTH),
                await client.post("/v1/chat/completions", json={"model": "anthropic/x"}, headers=AUTH),
                await client.get("/v1/nothing", headers=AUTH)]

    responses, upstream, _ = with_server(tmp_path, scenario)
    assert [r.status_code for r in responses] == [400, 400, 400, 404]
    assert responses[0].json()["error"]["code"] == "model_not_found"
    assert upstream.requests == []


def test_upstream_rate_limit_is_429_and_recorded_as_an_error_event(tmp_path):
    upstream = Upstream()
    upstream.anthropic_status = 429

    async def scenario(client):
        return await client.post("/v1/chat/completions", headers=AUTH,
                                 json=chat_fixture("chat_json_schema.json", "anthropic/claude-opus-5-5"))

    response, _, _ = with_server(tmp_path, scenario, upstream=upstream)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_error"
    [line] = meter_lines(tmp_path)
    assert line["status"] == "error" and line["error"] == {"code": "rate_limit_error", "status": 429}


def test_chatgpt_plan_limit_failure_surfaces_as_429(tmp_path):
    upstream = Upstream()
    upstream.responses_reply = lambda: sse_response(("response.failed", {"type": "response.failed", "response": {
        "error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "limit"}}}))

    async def scenario(client):
        return await client.post("/v1/chat/completions", headers=AUTH,
                                 json=chat_fixture("chat_json_schema.json", "chatgpt/gpt-6.1-sol"))

    response, _, _ = with_server(tmp_path, scenario, upstream=upstream)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "subscription_sharing_usage_limit_exceeded"


def test_call_without_usage_fails_and_is_recorded(tmp_path):
    upstream = Upstream()
    from .support import responses_stream_text
    upstream.responses_reply = lambda: httpx.Response(200, content=responses_stream_text(usage=None))

    async def scenario(client):
        return await client.post("/v1/chat/completions", headers=AUTH,
                                 json=chat_fixture("chat_json_schema.json", "chatgpt/gpt-6.1-sol"))

    response, _, _ = with_server(tmp_path, scenario, upstream=upstream)
    assert response.status_code == 502 and response.json()["error"]["code"] == "usage_missing"
    assert meter_lines(tmp_path)[0]["error"]["code"] == "usage_missing"


def test_stream_true_replays_the_completed_answer_as_sse_chunks(tmp_path):
    async def scenario(client):
        body = {**chat_fixture("chat_json_schema.json", "anthropic/claude-opus-5-5"), "stream": True}
        return await client.post("/v1/chat/completions", json=body, headers=AUTH)

    response, _, _ = with_server(tmp_path, scenario)
    assert response.headers["content-type"] == "text/event-stream"
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    first, last = json.loads(events[0]), json.loads(events[1])
    assert first["choices"][0]["delta"]["content"] == "Paris."
    assert last["choices"][0]["finish_reason"] == "stop" and last["usage"]["prompt_tokens"] == 135


def test_hybrid_llm_openai_compat_backend_can_use_the_gateway(tmp_path):
    from hybrid.llm.backends.base import BackendRequest
    from hybrid.llm.backends.openai_compat import OpenAICompatBackend
    from hybrid.llm.config import BackendConfig
    import os

    upstream = Upstream()
    log_in_both(tmp_path)
    app, server = serve(tmp_path, upstream)
    os.environ["BIGBRANE_TEST_GATEWAY_KEY"] = TOKEN

    async def go():
        port = await server.start()
        try:
            config = BackendConfig(name="gateway", kind="gateway", concurrency=1, timeout_s=30,
                                   base_url=f"http://127.0.0.1:{port}/v1", api_key_env="BIGBRANE_TEST_GATEWAY_KEY")
            request = BackendRequest(call_id="call-x", model="anthropic/claude-opus-5-5", system="Be terse.", user="Capital of France?",
                                     allow_web=False, max_output_tokens=100, timeout_s=30, scratch_dir=tmp_path)
            return await OpenAICompatBackend(config).complete(request)
        finally:
            await server.close()

    try:
        reply = run(go())
    finally:
        del os.environ["BIGBRANE_TEST_GATEWAY_KEY"]
    assert reply.text == "Paris." and reply.model == "claude-opus-5-5"
    assert reply.usage == {"input": 135, "output": 7, "cached": 100}
