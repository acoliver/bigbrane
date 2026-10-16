import json
import re

import httpx
import pytest

from hybrid.gateway import anthropic_translate as tr
from hybrid.gateway.backends import AnthropicBackend
from hybrid.gateway.errors import BadRequest
from hybrid.gateway.sse import iter_events
from .support import chat_fixture, fixture_json, run, sse

ADAPTIVE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
ADAPTIVE_THINKING = {"type": "adaptive", "display": "summarized"}
BINDING = {"prefix_mismatch_behavior": "drop_block"}


def reasoning_body(model: str, effort: str | None = "high", **extra) -> dict:
    body = chat_fixture("chat_reasoning.json", f"anthropic/{model}")
    if effort is not None:
        body["reasoning_effort"] = effort
    body.update(extra)
    return body


def aggregate(raw: bytes) -> dict:
    async def go():
        acc = tr.MessageAccumulator()
        async for _event, data in iter_events(httpx.Response(200, content=raw)):
            acc.feed(data)
        assert acc.finished
        return acc.message
    return run(go())


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_golden_request_maps_reasoning_effort_to_adaptive_thinking(effort):
    payload, ctx = tr.build_request(reasoning_body("claude-opus-5-5", effort), "claude-opus-5-5")
    assert payload == fixture_json("golden", f"anthropic_request_reasoning_{effort}.json")
    assert ctx.dropped == ()


@pytest.mark.parametrize("model", ADAPTIVE_MODELS)
def test_each_claude_5_5_model_gets_the_same_adaptive_mapping(model):
    payload, ctx = tr.build_request(reasoning_body(model, "medium"), model)
    assert payload["thinking"] == {**ADAPTIVE_THINKING, "block_binding": BINDING}
    assert payload["output_config"] == {"effort": "medium"}
    assert ctx.dropped == ()


def test_no_reasoning_effort_sends_no_thinking_field():
    payload, ctx = tr.build_request(reasoning_body("claude-opus-5-5", effort=None), "claude-opus-5-5")
    assert "thinking" not in payload and "output_config" not in payload
    assert ctx.dropped == ()


def test_reasoning_effort_on_a_legacy_model_enables_budgeted_thinking_like_llxprt():
    payload, ctx = tr.build_request(reasoning_body("claude-opus-4-5", "high"), "claude-opus-4-5")
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 10000}
    assert payload["output_config"] == {"effort": "high"}
    assert ctx.dropped == ()


@pytest.mark.parametrize("model", ["claude-sonnet-4-5", "claude-sonnet-4-5-20250929", "claude-haiku-4-5"])
def test_models_that_refuse_the_effort_parameter_get_the_thinking_budget_only(model):
    payload, ctx = tr.build_request(reasoning_body(model, "high"), model)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 10000}
    assert "output_config" not in payload
    assert ctx.dropped == ()


def test_reasoning_effort_on_a_model_that_is_not_claude_is_dropped_and_reported():
    payload, ctx = tr.build_request(reasoning_body("some-other-model", "high"), "some-other-model")
    assert "thinking" not in payload and "output_config" not in payload
    assert ctx.dropped == ("reasoning_effort",)


@pytest.mark.parametrize("effort,model,expected", [
    ("minimal", "claude-opus-5-5", "low"), ("low", "claude-opus-5-5", "low"), ("medium", "claude-opus-5-5", "medium"),
    ("high", "claude-opus-5-5", "high"), ("xhigh", "claude-opus-5-5", "max"), ("max", "claude-sonnet-5-5", "max"),
    ("max", "claude-opus-4-6", "max"), ("xhigh", "claude-fable-5", "max"),
    ("max", "claude-sonnet-4-6", "high"), ("xhigh", "claude-opus-4-5", "high"),
])
def test_effort_names_are_normalized_like_llxprt(effort, model, expected):
    payload, _ = tr.build_request(reasoning_body(model, effort), model)
    assert payload["output_config"] == {"effort": expected}


@pytest.mark.parametrize("effort", ["ultra", "", 3, True])
def test_reasoning_effort_outside_llxprts_vocabulary_is_rejected(effort):
    with pytest.raises(BadRequest):
        tr.build_request(reasoning_body("claude-opus-5-5", effort), "claude-opus-5-5")


@pytest.mark.parametrize("model,mode", [("claude-opus-5", "disabled"), ("claude-opus-5-latest", "disabled"),
                                        ("claude-haiku-5-5", "disabled"), ("claude-sonnet-5-5", "between_tools")])
def test_effort_none_turns_thinking_off_with_the_mode_the_model_takes(model, mode):
    payload, ctx = tr.build_request(reasoning_body(model, "none"), model)
    assert payload["thinking"] == {"type": mode}
    assert "output_config" not in payload
    assert ctx.dropped == ()


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-opus-4-6", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5"])
def test_effort_none_on_a_model_with_no_thinking_off_mode_omits_thinking_and_reports_it(model):
    payload, ctx = tr.build_request(reasoning_body(model, "none"), model)
    assert "thinking" not in payload and "output_config" not in payload
    assert ctx.dropped == ("reasoning_effort",)


@pytest.mark.parametrize("model,bound", [("claude-opus-5-5", True), ("claude-sonnet-5-5", True), ("claude-haiku-5-5", True),
                                         ("claude-fable-5-1", True), ("claude-opus-5", False), ("claude-fable-5", False),
                                         ("claude-opus-4-6", False), ("claude-sonnet-4-6", False)])
def test_block_binding_is_sent_only_where_llxprt_enforces_the_preserved_thinking_prefix(model, bound):
    payload, _ = tr.build_request(reasoning_body(model, "high"), model)
    assert ("block_binding" in payload["thinking"]) is bound


def test_thinking_off_modes_never_carry_block_binding():
    payload, _ = tr.build_request(reasoning_body("claude-sonnet-5-5", "none"), "claude-sonnet-5-5")
    assert payload["thinking"] == {"type": "between_tools"}


def test_caller_max_tokens_is_sent_unchanged_with_thinking():
    payload, _ = tr.build_request(reasoning_body("claude-opus-5-5", "high", max_tokens=1000), "claude-opus-5-5")
    assert payload["max_tokens"] == 1000


def test_default_max_tokens_applies_when_thinking_is_on_and_no_limit_is_set():
    body = reasoning_body("claude-haiku-5-5", "high")
    del body["max_tokens"]
    payload, _ = tr.build_request(body, "claude-haiku-5-5")
    assert payload["max_tokens"] == tr.default_max_tokens("claude-haiku-5-5") == 128000


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5-5", True), ("claude-opus-4-6", True), ("claude-opus-4-1", False),
    ("claude-sonnet-5", True), ("claude-sonnet-5-20260101", True), ("claude-sonnet-5-5", True),
    ("claude-sonnet-50", False), ("claude-sonnet-4-5", False),
    ("claude-haiku-5-5", True), ("claude-haiku-5", False),
    ("claude-fable-5-1", True), ("claude-opus-5-5-mini", False),
])
def test_adaptive_thinking_model_ids_match_llxprt_boundaries(model, expected):
    assert tr.supports_adaptive_thinking(model) is expected


def test_anthropic_requests_carry_the_interleaved_thinking_beta():
    headers = AnthropicBackend(refresher=None)._headers("test-access")
    assert "interleaved-thinking-2025-05-14" in [part.strip() for part in headers["anthropic-beta"].split(",")]


def test_anthropic_user_agent_version_is_at_least_2_1_280():
    # LLxprt Code: Claude subscription endpoints require claude-cli 2.1.280 or newer.
    user_agent = AnthropicBackend(refresher=None)._headers("test-access")["user-agent"]
    match = re.fullmatch(r"claude-cli/(\d+)\.(\d+)\.(\d+) \(external, cli\)", user_agent)
    assert match is not None, user_agent
    assert tuple(int(part) for part in match.groups()) >= (2, 1, 280)


def test_thinking_blocks_are_not_returned_as_content_and_their_tokens_stay_in_output():
    # output_tokens already includes the thinking tokens, so usage is read from it unchanged.
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_t", "model": "claude-opus-5-5",
                           "role": "assistant", "usage": {"input_tokens": 30, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "thinking_delta", "thinking": "Capital of France. It is Paris."}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "signature_delta", "signature": "sig-abc"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": "Paris."}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 57}}),
        ("message_stop", {"type": "message_stop"}),
    )
    chat, usage = tr.to_chat(aggregate(raw), tr.RequestContext(json_tool=False, dropped=()), now=1)
    choice = chat["choices"][0]
    assert choice["message"]["content"] == "Paris."
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"
    serialized = json.dumps(chat)
    assert "Capital of France" not in serialized
    assert "sig-abc" not in serialized
    assert usage == {"input": 30, "output": 57, "cached": 0}


def test_redacted_thinking_before_a_tool_call_is_dropped_and_the_call_is_kept():
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_r", "model": "claude-opus-5-5",
                           "role": "assistant", "usage": {"input_tokens": 12, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "redacted_thinking", "data": "OPAQUE-DATA"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "tool_use", "id": "toolu_w", "name": "llxprt_get_weather", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "input_json_delta", "partial_json": "{\"city\": \"Rome\"}"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}),
        ("message_stop", {"type": "message_stop"}),
    )
    chat, usage = tr.to_chat(aggregate(raw), tr.RequestContext(json_tool=False, dropped=()), now=1)
    choice = chat["choices"][0]
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == [{"id": "toolu_w", "type": "function",
                                                "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]
    assert choice["finish_reason"] == "tool_calls"
    assert "OPAQUE-DATA" not in json.dumps(chat)
    assert usage == {"input": 12, "output": 20, "cached": 0}
