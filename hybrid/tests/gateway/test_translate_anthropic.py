import json

import pytest

from hybrid.gateway import anthropic_translate as tr
from hybrid.gateway.errors import BadRequest, UpstreamError
from hybrid.gateway.sse import iter_events
from .support import chat_fixture, fixture_json, run, sse

import httpx


def aggregate(raw: bytes) -> dict:
    async def go():
        acc = tr.MessageAccumulator()
        response = httpx.Response(200, content=raw)
        async for _event, data in iter_events(response):
            acc.feed(data)
        assert acc.finished
        return acc.message
    return run(go())


def test_golden_request_with_tools_and_tool_result():
    # The fixture sets temperature 0.2, which the 5.5 models refuse, so it is dropped and reported.
    payload, ctx = tr.build_request(chat_fixture("chat_with_tools.json", "anthropic/claude-opus-5-5"), "claude-opus-5-5")
    assert payload == fixture_json("golden", "anthropic_request_with_tools.json")
    assert ctx.dropped == ("temperature",)


def test_golden_request_with_tools_keeps_temperature_on_a_model_that_takes_it():
    payload, ctx = tr.build_request(chat_fixture("chat_with_tools.json", "x"), "claude-opus-4-6")
    assert payload["temperature"] == 0.2 and ctx.dropped == ()


def test_golden_request_json_schema_uses_native_structured_output_on_models_that_refuse_a_forced_tool():
    payload, ctx = tr.build_request(chat_fixture("chat_json_schema.json", "anthropic/claude-opus-5-5"), "claude-opus-5-5")
    assert payload == fixture_json("golden", "anthropic_request_json_schema.json")
    assert not ctx.json_tool


def test_golden_request_json_schema_uses_a_forced_tool_elsewhere():
    payload, ctx = tr.build_request(chat_fixture("chat_json_schema.json", "anthropic/claude-haiku-5-5"), "claude-haiku-5-5")
    assert payload == fixture_json("golden", "anthropic_request_json_schema_forced_tool.json")
    assert ctx.json_tool


def test_system_field_is_only_the_claude_code_string_even_with_a_long_system_prompt():
    body = {"model": "x", "messages": [{"role": "system", "content": "A" * 5000}, {"role": "user", "content": "hi"}]}
    payload, _ = tr.build_request(body, "claude-sonnet-5-5")
    assert payload["system"] == "You are Claude Code, Anthropic's official CLI for Claude."
    assert "A" * 5000 in payload["messages"][0]["content"][0]["text"]
    assert payload["max_tokens"] == 128000       # default only when the caller sets no limit


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
def test_default_max_tokens_for_claude_5_5_models_matches_llxprt(model):
    payload, _ = tr.build_request({"model": model, "messages": [{"role": "user", "content": "hi"}]}, model)
    assert payload["max_tokens"] == 128000


def test_parallel_tool_results_merge_into_one_user_turn_with_results_first():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "r1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "r2"},
    ]
    payload, _ = tr.build_request({"model": "x", "messages": messages}, "claude-opus-5-5")
    last = payload["messages"][-1]
    assert [b["tool_use_id"] for b in last["content"]] == ["toolu_1", "toolu_2"]
    assert len(payload["messages"]) == 3


def test_history_tool_names_stay_unprefixed_and_declared_tools_are_prefixed():
    body = chat_fixture("chat_with_tools.json", "x")
    payload, _ = tr.build_request(body, "claude-opus-5-5")
    assert payload["tools"][0]["name"] == "llxprt_get_weather"
    assert payload["messages"][2]["content"][0]["name"] == "get_weather"


@pytest.mark.parametrize("choice,expected", [
    ("auto", {"type": "auto"}), ("required", {"type": "any"}),
    ({"type": "function", "function": {"name": "get_weather"}}, {"type": "tool", "name": "llxprt_get_weather"})])
def test_tool_choice_mapping(choice, expected):
    body = chat_fixture("chat_with_tools.json", "x")
    body["tool_choice"] = choice
    payload, _ = tr.build_request(body, "claude-opus-5-5")
    assert payload["tool_choice"] == expected


def test_tool_choice_none_sends_no_tools():
    body = chat_fixture("chat_with_tools.json", "x")
    body["tool_choice"] = "none"
    payload, _ = tr.build_request(body, "claude-opus-5-5")
    assert "tools" not in payload and "tool_choice" not in payload


def test_unconsumed_fields_are_reported_as_dropped():
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "seed": 1, "user": "u", "temperature": 0}
    payload, ctx = tr.build_request(body, "claude-opus-4-6")
    assert ctx.dropped == ("seed", "user")
    assert payload["temperature"] == 0


def test_json_schema_cannot_be_combined_with_tools():
    body = chat_fixture("chat_with_tools.json", "x")
    body["response_format"] = fixture_json("golden", "chat_json_schema.json")["response_format"]
    with pytest.raises(BadRequest):
        tr.build_request(body, "claude-opus-5-5")


def test_stream_with_tool_call_becomes_chat_tool_call_with_cached_tokens_split_out():
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_9", "model": "claude-opus-5-5", "role": "assistant",
                           "usage": {"input_tokens": 25, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 10, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking."}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "tool_use", "id": "toolu_abc", "name": "llxprt_get_weather", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": " \"Paris\"}"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 42}}),
        ("message_stop", {"type": "message_stop"}),
    )
    ctx = tr.RequestContext(json_tool=False, dropped=())
    chat, usage = tr.to_chat(aggregate(raw), ctx, now=1700000000)
    message = chat["choices"][0]["message"]
    assert message["content"] == "Checking."
    assert message["tool_calls"] == [{"id": "toolu_abc", "type": "function",
                                      "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]
    assert chat["choices"][0]["finish_reason"] == "tool_calls"
    assert usage == {"input": 135, "output": 42, "cached": 100}
    assert chat["usage"] == {"prompt_tokens": 135, "completion_tokens": 42, "total_tokens": 177,
                             "prompt_tokens_details": {"cached_tokens": 100}}
    assert chat["model"] == "claude-opus-5-5"


def test_forced_json_tool_reply_is_returned_as_message_content():
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "m", "model": "claude-opus-5-5", "role": "assistant",
                           "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "tool_use", "id": "toolu_j", "name": "llxprt_emit_json", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": \"Paris\"}"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 6}}),
        ("message_stop", {"type": "message_stop"}),
    )
    chat, _ = tr.to_chat(aggregate(raw), tr.RequestContext(json_tool=True, dropped=()), now=1)
    choice = chat["choices"][0]
    assert json.loads(choice["message"]["content"]) == {"city": "Paris"}
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"


def test_response_without_usage_fails_instead_of_returning_an_unmetered_call():
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "m", "model": "x", "role": "assistant", "usage": {}}}),
        ("message_stop", {"type": "message_stop"}),
    )
    with pytest.raises(UpstreamError) as err:
        tr.to_chat(aggregate(raw), tr.RequestContext(False, ()), now=1)
    assert err.value.code == "usage_missing"


def test_stream_error_event_raises_with_the_error_type():
    acc = tr.MessageAccumulator()
    with pytest.raises(UpstreamError) as err:
        acc.feed({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    assert err.value.code == "overloaded_error"
