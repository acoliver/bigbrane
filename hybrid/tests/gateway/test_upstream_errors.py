"""Upstream refusals: the provider's status, code, type and message must reach the client and the log, whatever shape
the error body has. Scrubbing of tokens happens where the text leaves the process."""
import json
import logging

import httpx
import pytest

from hybrid.gateway.anthropic_models import model_ids
from hybrid.gateway.backends import upstream_http_error
from hybrid.gateway.errors import UpstreamError
from hybrid.gateway.server import GatewayServer
from .support import SENTINEL_ACCESS, make_app, make_store, meter_lines, run, seed

RUN_TOKEN = "SENTINEL-RUNTOKEN-upstream-errors-3320"
AUTH = {"authorization": f"Bearer {RUN_TOKEN}"}
ANTHROPIC_MODELS = model_ids()
CHAT_MODEL = {"anthropic": "anthropic/claude-opus-5-5", "chatgpt": "chatgpt/gpt-6.1-sol"}
CHAT_BODY = {"messages": [{"role": "user", "content": "hi"}]}
LEAD = "Manage usage: https://chatgpt.com/settings/usage. "      # ChatGPT 403 and 429 messages start with this

OBJECT_403 = json.dumps({"error": {"message": "Your key lacks the api.responses.write scope",
                                   "type": "invalid_request_error", "code": "insufficient_scope"}}).encode()
STRING_403 = json.dumps({"error": "Access denied by policy"}).encode()
EMPTY_403 = b""
OBJECT_429 = json.dumps({"error": {"type": "rate_limit_error", "message": "slow down"}}).encode()
NESTED_403 = json.dumps({"error": {"message": {"message": "Your key lacks the api.responses.write scope",
                                               "type": "invalid_request_error", "code": "insufficient_scope"}}}).encode()


def static_refusal(status: int, body: bytes):
    return lambda request: httpx.Response(status, content=body)


def echoing_refusal(status: int):
    """A hostile upstream that writes the bearer token it was sent into its error message."""
    def handler(request: httpx.Request) -> httpx.Response:
        bearer = request.headers["authorization"].removeprefix("Bearer ")
        return httpx.Response(status, json={"error": {"message": f"token {bearer} lacks scope",
                                                      "type": "invalid_request_error", "code": "insufficient_scope"}})
    return handler


def serve(tmp_path, handler):
    app = make_app(tmp_path, handler)
    return app, GatewayServer(app.gateway, RUN_TOKEN, app.redactor, port=0)


def over_http(server, scenario):
    async def go():
        port = await server.start()
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
                return await scenario(client)
        finally:
            await server.close()
    return run(go())


# ---- Backend level: both providers, three body shapes and a 429 ----------------------------------

REFUSALS = [
    pytest.param(403, OBJECT_403, 403, "insufficient_scope", "invalid_request_error",
                 "Your key lacks the api.responses.write scope", id="403-object-error"),
    pytest.param(403, NESTED_403, 403, "insufficient_scope", "invalid_request_error",
                 "Your key lacks the api.responses.write scope", id="403-message-is-object"),
    pytest.param(403, STRING_403, 403, "upstream_rejected", None, "Access denied by policy", id="403-string-error"),
    pytest.param(403, EMPTY_403, 403, "upstream_rejected", None, "", id="403-empty-body"),
    pytest.param(429, OBJECT_429, 429, "rate_limit_error", "rate_limit_error", "slow down", id="429-object-error"),
]


@pytest.mark.parametrize("provider", ["anthropic", "chatgpt"])
@pytest.mark.parametrize("http_status, body, gateway_status, code, error_type, message", REFUSALS)
def test_backend_refusal_keeps_status_code_type_and_message(tmp_path, provider, http_status, body, gateway_status,
                                                            code, error_type, message):
    seed(make_store(tmp_path), provider)
    app = make_app(tmp_path, static_refusal(http_status, body))

    with pytest.raises(UpstreamError) as err:
        run(app.gateway.chat({**CHAT_BODY, "model": CHAT_MODEL[provider]}))

    error = err.value
    assert (error.status, error.code) == (gateway_status, code)
    assert error.detail.http_status == http_status
    assert error.detail.error_type == error_type
    assert error.detail.message == message
    lead = LEAD if provider == "chatgpt" else ""
    expected_text = lead + f"{provider} returned HTTP {http_status}" + (f": {message}" if message else "")
    assert error.message == expected_text
    assert meter_lines(tmp_path)[0]["error"] == {"code": code, "status": gateway_status}


# ---- Parser shapes that the provider backends do not all produce -----------------------------------

def test_message_that_is_an_object_is_unwrapped_and_its_code_kept():
    body = json.dumps({"error": {"message": {"message": "nested text", "code": "nested_code"}}}).encode()
    error = upstream_http_error("chatgpt", 403, body)
    assert (error.status, error.code, error.detail.message) == (403, "nested_code", "nested text")


def test_message_that_is_a_list_is_kept_as_json_text():
    body = json.dumps({"error": {"message": ["first problem", "second problem"]}}).encode()
    error = upstream_http_error("chatgpt", 403, body)
    assert error.detail.message == '["first problem", "second problem"]'


def test_top_level_detail_is_the_message_when_there_is_no_error_key():
    error = upstream_http_error("chatgpt", 403, b'{"detail": "Forbidden for this account"}')
    assert error.detail.message == "Forbidden for this account" and error.code == "upstream_rejected"


def test_text_that_is_not_json_is_the_message():
    error = upstream_http_error("anthropic", 403, b"<html>denied by the edge</html>")
    assert error.detail.message == "<html>denied by the edge</html>"


def test_long_upstream_messages_are_cut_to_300_characters():
    body = json.dumps({"error": {"message": "x" * 1000}}).encode()
    error = upstream_http_error("anthropic", 403, body)
    assert len(error.detail.message) == 300


# ---- HTTP level: the client sees the provider's status, and nothing secret ----------------------------

def test_chat_refusal_reaches_the_client_as_403_and_the_log_names_the_upstream_error(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bigbrane.gateway")
    seed(make_store(tmp_path), "chatgpt", access=SENTINEL_ACCESS, meta={"client_id": "oaiapp_x"})
    _, server = serve(tmp_path, echoing_refusal(403))

    async def scenario(client):
        return await client.post("/v1/chat/completions", headers=AUTH, json={**CHAT_BODY, "model": CHAT_MODEL["chatgpt"]})

    response = over_http(server, scenario)
    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "insufficient_scope" and error["type"] == "invalid_request_error"
    assert error["message"] == LEAD + "chatgpt returned HTTP 403: token [redacted] lacks scope"
    assert ("upstream chatgpt refused: http_status=403 code=insufficient_scope type=invalid_request_error "
            "message=token [redacted] lacks scope") in caplog.text
    assert SENTINEL_ACCESS not in response.text + caplog.text


def test_transit_failure_is_logged_without_a_detail_block(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bigbrane.gateway")
    seed(make_store(tmp_path), "chatgpt", access=SENTINEL_ACCESS, meta={"client_id": "oaiapp_x"})

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _, server = serve(tmp_path, unreachable)

    async def scenario(client):
        return await client.post("/v1/chat/completions", headers=AUTH, json={**CHAT_BODY, "model": CHAT_MODEL["chatgpt"]})

    response = over_http(server, scenario)
    assert response.status_code == 502 and response.json()["error"]["code"] == "upstream_unreachable"
    assert "upstream chatgpt failed: code=upstream_unreachable message=chatgpt request failed in transit: ConnectError" \
        in caplog.text


@pytest.mark.parametrize("body, code, message", [
    pytest.param(OBJECT_403, "insufficient_scope",
                 LEAD + "chatgpt returned HTTP 403: Your key lacks the api.responses.write scope", id="object-error"),
    pytest.param(STRING_403, "upstream_rejected", LEAD + "chatgpt returned HTTP 403: Access denied by policy",
                 id="string-error"),
    pytest.param(EMPTY_403, "upstream_rejected", LEAD + "chatgpt returned HTTP 403", id="empty-body"),
])
def test_models_names_a_refused_backend_with_status_and_message_and_keeps_the_others(tmp_path, body, code, message):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    seed(store, "chatgpt")
    _, server = serve(tmp_path, static_refusal(403, body))

    async def scenario(client):
        return await client.get("/v1/models", headers=AUTH)

    response = over_http(server, scenario)
    assert response.status_code == 200
    payload = response.json()
    assert [m["id"] for m in payload["data"]] == [f"anthropic/{name}" for name in ANTHROPIC_MODELS]
    assert payload["upstream_errors"] == {"chatgpt": {"status": 403, "code": code, "message": message}}


def test_models_failure_text_is_scrubbed_before_it_leaves_the_process(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bigbrane.gateway")
    store = make_store(tmp_path)
    seed(store, "anthropic")
    seed(store, "chatgpt", access=SENTINEL_ACCESS, meta={"client_id": "oaiapp_x"})
    _, server = serve(tmp_path, echoing_refusal(403))

    async def scenario(client):
        return await client.get("/v1/models", headers=AUTH)

    response = over_http(server, scenario)
    assert response.json()["upstream_errors"]["chatgpt"] == {
        "status": 403, "code": "insufficient_scope",
        "message": LEAD + "chatgpt returned HTTP 403: token [redacted] lacks scope"}
    assert SENTINEL_ACCESS not in response.text + caplog.text
    assert "[redacted]" in caplog.text
