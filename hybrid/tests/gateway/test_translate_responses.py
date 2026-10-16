import json

import httpx
import pytest

from hybrid.gateway import responses_translate as tr
from hybrid.gateway.errors import BadRequest, UpstreamError
from hybrid.gateway.sse import iter_events
from .support import chat_fixture, fixture_json, responses_stream_text, run, sse


def aggregate(raw: bytes) -> dict:
    async def go():
        acc = tr.ResponseAccumulator()
        async for event, data in iter_events(httpx.Response(200, content=raw)):
            acc.feed(event, data)
        return acc.result
    return run(go())


def test_golden_request_with_tools_and_tool_result():
    payload, ctx = tr.build_request(chat_fixture("chat_with_tools.json", "chatgpt/gpt-6.1-sol"), "gpt-6.1-sol")
    assert payload == fixture_json("golden", "responses_request_with_tools.json")
    # The ChatGPT plan route refuses these, so they are dropped and reported.
    assert ctx.dropped == ("max_tokens", "temperature")
    assert payload["store"] is False and payload["stream"] is True
    assert not any(item.get("role") == "system" for item in payload["input"])


def test_golden_request_json_schema_goes_to_text_format():
    payload, _ = tr.build_request(chat_fixture("chat_json_schema.json", "chatgpt/gpt-6.1-sol"), "gpt-6.1-sol")
    assert payload == fixture_json("golden", "responses_request_json_schema.json")


def test_unsupported_preview_fields_never_reach_the_request():
    body = chat_fixture("chat_with_tools.json", "x")
    body.update({"top_p": 0.5, "user": "u", "metadata": {"a": 1}, "max_completion_tokens": 10})
    payload, _ = tr.build_request(body, "gpt-6.1-sol")
    for forbidden in ("max_output_tokens", "temperature", "top_p", "user", "metadata", "previous_response_id"):
        assert forbidden not in payload


def test_function_call_without_output_and_output_without_call_are_dropped():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_orphan", "content": "x"},
    ]
    payload, _ = tr.build_request({"model": "x", "messages": messages}, "gpt-6.1-sol")
    assert payload["input"] == [{"role": "user", "content": "go"}]


def test_call_ids_are_normalized_like_llxprt():
    assert tr.normalize_call_id("toolu_01AbC-x!") == "call_01AbC-x"
    assert tr.normalize_call_id("hist_tool_zz") == "call_zz"
    assert tr.normalize_call_id("call_ok_1") == "call_ok_1"


def test_reasoning_effort_and_forced_tool_choice():
    body = chat_fixture("chat_with_tools.json", "x")
    body["reasoning_effort"] = "high"
    body["tool_choice"] = {"type": "function", "function": {"name": "get_weather"}}
    payload, _ = tr.build_request(body, "gpt-6.1-sol")
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["tool_choice"] == {"type": "function", "name": "get_weather"}


def test_text_reply_with_cached_tokens():
    result = aggregate(responses_stream_text("Paris."))
    chat, usage = tr.to_chat(result, "gpt-6.1-sol", now=5)
    assert chat["choices"][0]["message"]["content"] == "Paris."
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert usage == {"input": 120, "output": 9, "cached": 100}
    assert chat["usage"]["prompt_tokens_details"] == {"cached_tokens": 100}


def test_function_call_items_become_chat_tool_calls_even_when_completed_output_is_empty():
    item = {"type": "function_call", "call_id": "call_9", "name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}
    raw = sse(
        ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item}),
        ("response.completed", {"type": "response.completed", "response": {
            "id": "r", "model": "gpt-6.1-sol", "status": "completed", "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 3}}}),
    )
    chat, usage = tr.to_chat(aggregate(raw), "gpt-6.1-sol", now=5)
    choice = chat["choices"][0]
    assert choice["message"]["tool_calls"] == [{"id": "call_9", "type": "function",
                                                "function": {"name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}}]
    assert choice["finish_reason"] == "tool_calls"
    assert usage == {"input": 5, "output": 3, "cached": 0}


def test_missing_usage_fails():
    with pytest.raises(UpstreamError) as err:
        tr.to_chat(aggregate(responses_stream_text(usage=None)), "m", now=1)
    assert err.value.code == "usage_missing"


@pytest.mark.parametrize("code,status", [("subscription_sharing_usage_limit_exceeded", 429),
                                         ("subscription_sharing_usage_unavailable", 503),
                                         ("subscription_sharing_unsupported_capability", 400),
                                         ("something_else", 502)])
def test_response_failed_maps_documented_codes(code, status):
    raw = sse(("response.failed", {"type": "response.failed", "response": {"error": {"code": code, "message": "nope"}}}))
    with pytest.raises(UpstreamError) as err:
        aggregate(raw)
    assert err.value.status == status and err.value.code == code


def test_stream_ending_without_a_terminal_event_gives_no_result():
    raw = sse(("response.output_text.delta", {"type": "response.output_text.delta", "delta": "x"}))
    assert aggregate(raw) is None


def test_json_object_format_and_unknown_format():
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}}
    payload, _ = tr.build_request(body, "m")
    assert payload["text"] == {"format": {"type": "json_object"}}
    body["response_format"] = {"type": "grammar"}
    with pytest.raises(BadRequest):
        tr.build_request(body, "m")
