"""Request construction ported from LLxprt Code: prompt caching, tool ordering, sequence repair, sampling limits,
native structured output, and the beta headers."""
import copy

import pytest

from hybrid.gateway import anthropic_sequence as seq
from hybrid.gateway import anthropic_translate as tr
from hybrid.gateway.anthropic_schema import to_structured_output_schema
from hybrid.gateway.backends import AnthropicBackend
from hybrid.gateway.errors import BadRequest

OPUS55 = "claude-opus-5-5"
CACHE = {"type": "ephemeral", "ttl": "1h"}


def request(messages, model=OPUS55, **extra):
    return tr.build_request({"model": model, "messages": messages, **extra}, model)[0]


def user(text):
    return {"role": "user", "content": text}


def text_block(text, **extra):
    return {"type": "text", "text": text, **extra}


# ---- prompt caching ----------------------------------------------------------------------------------------------

def test_context_prefix_and_last_block_carry_one_hour_cache_control():
    payload = request([{"role": "system", "content": "Be terse."}, user("first"), {"role": "assistant", "content": "ok"},
                       user("second")])
    first, *_, last = payload["messages"]
    assert first["content"] == [text_block(tr.format_context_prefix("Be terse."), cache_control=CACHE)]
    assert last["content"] == [text_block("second", cache_control=CACHE)]
    marked = [b for m in payload["messages"] for b in m["content"] if "cache_control" in b]
    assert len(marked) == 2


def test_without_a_system_prompt_only_the_last_block_is_marked():
    payload = request([user("hello")])
    assert payload["messages"] == [{"role": "user", "content": [text_block("hello", cache_control=CACHE)]}]


def test_cache_control_lands_on_the_last_non_empty_block_of_the_last_message():
    messages = [user("go"), {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"}]
    payload = request(messages)
    last_block = payload["messages"][-1]["content"][-1]
    assert last_block["type"] == "tool_result" and last_block["cache_control"] == CACHE


def test_empty_tool_result_is_skipped_when_choosing_the_cache_block():
    content = [text_block("keep"), {"type": "tool_result", "tool_use_id": "t", "content": "   "}]
    messages = [{"role": "user", "content": content}]
    tr.attach_prompt_caching(messages)
    assert messages[0]["content"][0]["cache_control"] == CACHE
    assert "cache_control" not in messages[0]["content"][1]


def test_extra_betas_follow_what_the_payload_uses():
    plain = {"messages": [{"role": "user", "content": [text_block("x")]}]}
    cached = {"messages": [{"role": "user", "content": [text_block("x", cache_control=CACHE)]}]}
    five_minute = {"messages": [{"role": "user", "content": [text_block("x", cache_control={"type": "ephemeral", "ttl": "5m"})]}]}
    bound = {**plain, "thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}}
    assert tr.extra_betas(plain) == [] and tr.extra_betas(five_minute) == []
    assert tr.extra_betas(cached) == ["extended-cache-ttl-2025-04-11"]
    assert tr.extra_betas({**cached, **{"thinking": bound["thinking"]}}) == [
        "extended-cache-ttl-2025-04-11", "thinking-binding-controls-2026-08-01"]


def test_headers_merge_the_betas_in_llxprts_order():
    backend = AnthropicBackend(refresher=None)
    payload = request([user("hi")], reasoning_effort="high")
    headers = backend._headers("tok", payload)
    assert headers["anthropic-beta"] == ("oauth-2025-04-20, interleaved-thinking-2025-05-14, "
                                         "extended-cache-ttl-2025-04-11, thinking-binding-controls-2026-08-01")
    assert headers["authorization"] == "Bearer tok"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"
    assert headers["user-agent"] == "claude-cli/2.1.293 (external, cli)"


def test_headers_without_a_payload_carry_only_the_oauth_betas():
    headers = AnthropicBackend(refresher=None)._headers("tok")
    assert headers["anthropic-beta"] == "oauth-2025-04-20, interleaved-thinking-2025-05-14"


# ---- tools -------------------------------------------------------------------------------------------------------

def test_tools_are_sent_sorted_by_name_with_sorted_property_keys():
    def tool(name, **props):
        return {"type": "function", "function": {"name": name, "parameters": {
            "type": "object", "properties": {k: {"type": "string"} for k in props}, "required": []}}}
    payload = request([user("go")], tools=[tool("zeta", b=1, a=1), tool("Alpha", z=1, m=1), tool("beta")])
    assert [t["name"] for t in payload["tools"]] == ["llxprt_Alpha", "llxprt_beta", "llxprt_zeta"]
    assert list(payload["tools"][0]["input_schema"]["properties"]) == ["m", "z"]
    assert list(payload["tools"][2]["input_schema"]["properties"]) == ["a", "b"]


# ---- sequence repair ---------------------------------------------------------------------------------------------

def tool_call_turn(*ids):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "f", "arguments": "{}"}} for i in ids]}


def test_orphaned_tool_results_are_dropped():
    payload = request([user("go"), {"role": "tool", "tool_call_id": "call_ghost", "content": "late"}, user("again")])
    blocks = [b for m in payload["messages"] for b in m["content"]]
    assert not any(b["type"] == "tool_result" for b in blocks)


def test_a_missing_tool_result_is_synthesized_as_an_interrupted_error():
    payload = request([user("go"), tool_call_turn("call_1", "call_2"),
                       {"role": "tool", "tool_call_id": "call_1", "content": "one"}, user("next")])
    results = [b for b in payload["messages"][2]["content"] if b["type"] == "tool_result"]
    assert [(b["tool_use_id"], b["content"], b.get("is_error")) for b in results] == [
        ("toolu_2", "[tool execution interrupted]", True), ("toolu_1", "one", None)]


def test_tool_use_at_the_end_gets_a_synthesized_result_turn():
    payload = request([user("go"), tool_call_turn("call_1")], model="claude-haiku-4-5")
    last = payload["messages"][-1]
    assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["content"] == seq.INTERRUPTED_RESULT and last["content"][0]["is_error"] is True


def test_a_result_that_arrives_later_is_moved_next_to_its_tool_use():
    messages = [{"role": "user", "content": [text_block("go")]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]},
                {"role": "user", "content": [text_block("interjection")]},
                {"role": "assistant", "content": [text_block("hm")]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "late"}]}]
    repaired = seq.repair(copy.deepcopy(messages), thinking=False, model="claude-haiku-4-5")
    assert repaired[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "late"}
    assert all(b["type"] != "tool_result" for m in repaired[3:] for b in m["content"])


def test_sequence_starts_with_a_user_turn():
    repaired = seq.repair([{"role": "assistant", "content": [text_block("hi")]},
                           {"role": "user", "content": [text_block("yes")]}], thinking=False, model="claude-haiku-4-5")
    assert repaired[0] == {"role": "user", "content": [text_block(seq.CONTINUE_TEXT)]}


def test_an_empty_conversation_becomes_hello():
    assert seq.repair([], thinking=False, model=OPUS55) == [{"role": "user", "content": [text_block("Hello")]}]


def test_empty_turns_get_role_placeholders():
    payload = request([user("a"), {"role": "assistant", "content": ""}, user("   "), ])
    contents = [m["content"][0]["text"] for m in payload["messages"]]
    assert contents[:2] == ["a", "[No content generated]"]
    assert payload["messages"][2]["content"][0]["text"] == "[Empty message]"


def test_a_trailing_assistant_turn_gets_a_user_turn_when_the_model_refuses_prefill():
    payload = request([user("a"), {"role": "assistant", "content": "partial"}], model=OPUS55)
    assert payload["messages"][-1]["role"] == "user" and payload["messages"][-1]["content"][0]["text"] == seq.CONTINUE_TEXT


def test_a_trailing_assistant_turn_is_kept_for_models_that_allow_prefill_unless_thinking_is_on():
    messages = [user("a"), {"role": "assistant", "content": "partial"}]
    assert request(messages, model="claude-haiku-4-5")["messages"][-1]["role"] == "assistant"
    with_thinking = request(messages, model="claude-opus-4-6", reasoning_effort="high")
    assert with_thinking["messages"][-1]["role"] == "user"


def test_adjacent_same_role_turns_are_merged_with_tool_results_first():
    repaired = seq.repair([{"role": "user", "content": [text_block("a")]},
                           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}]}],
                          thinking=False, model="claude-haiku-4-5")
    assert len(repaired) == 1


# ---- sampling parameters -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", [OPUS55, "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "claude-opus-4-7"])
def test_models_that_refuse_sampling_parameters_do_not_receive_them(model):
    payload, ctx = tr.build_request({"model": model, "messages": [user("x")], "temperature": 0.3, "top_p": 0.9, "top_k": 4}, model)
    assert not {"temperature", "top_p", "top_k"} & set(payload)
    assert ctx.dropped == ("temperature", "top_k", "top_p")


def test_models_that_take_sampling_parameters_get_them():
    payload, ctx = tr.build_request({"model": "x", "messages": [user("x")], "temperature": 0.3, "top_p": 0.9, "top_k": 4},
                                    "claude-opus-4-6")
    assert (payload["temperature"], payload["top_p"], payload["top_k"]) == (0.3, 0.9, 4)
    assert ctx.dropped == ()


def test_stop_sequences_and_max_tokens_pass_through():
    payload = request([user("x")], stop=["END"], max_tokens=77)
    assert payload["stop_sequences"] == ["END"] and payload["max_tokens"] == 77


# ---- structured output -------------------------------------------------------------------------------------------

SCHEMA = {"type": "object", "properties": {
    "capital": {"type": "string", "minLength": 2, "description": "City"},
    "pop": {"type": "integer", "minimum": 0},
    "tags": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "string"}}, "required": ["n"]}}},
    "required": ["capital"]}


def json_request(model, **extra):
    body = {"model": model, "messages": [user("x")], "response_format": {"type": "json_schema", "json_schema": {
        "name": "a", "schema": SCHEMA}}, **extra}
    return tr.build_request(body, model)


@pytest.mark.parametrize("model", [OPUS55, "claude-sonnet-5-5", "claude-fable-5-1"])
def test_models_that_refuse_a_forced_tool_get_native_structured_output(model):
    payload, ctx = json_request(model)
    assert "tools" not in payload and "tool_choice" not in payload
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert not ctx.json_tool


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-opus-5", "claude-sonnet-4-6", "claude-haiku-4-5"])
def test_other_models_keep_the_forced_tool(model):
    payload, ctx = json_request(model)
    assert payload["tool_choice"] == {"type": "tool", "name": "llxprt_emit_json"}
    assert "format" not in payload.get("output_config", {})
    assert ctx.json_tool


def test_native_structured_output_keeps_the_effort_in_the_same_output_config():
    payload, _ = json_request(OPUS55, reasoning_effort="high")
    assert payload["output_config"]["effort"] == "high" and "format" in payload["output_config"]
    assert payload["thinking"]["type"] == "adaptive"


def test_schema_rewrite_closes_objects_and_moves_refused_constraints_into_descriptions():
    original = copy.deepcopy(SCHEMA)
    rewritten = to_structured_output_schema(SCHEMA)
    assert SCHEMA == original                                   # the caller's schema is not modified
    assert rewritten["additionalProperties"] is False
    item = rewritten["properties"]["tags"]["items"]
    assert item["additionalProperties"] is False
    assert "minLength" not in rewritten["properties"]["capital"]
    assert rewritten["properties"]["capital"]["description"].startswith("City Constraint not enforced")
    assert "minLength: 2" in rewritten["properties"]["capital"]["description"]
    assert "minimum" not in rewritten["properties"]["pop"] and "minimum: 0" in rewritten["properties"]["pop"]["description"]


def test_schema_rewrite_reaches_into_any_of_and_defs():
    rewritten = to_structured_output_schema({"type": "object", "properties": {
        "v": {"anyOf": [{"type": "object", "properties": {"a": {"type": "string"}}}, {"type": "null"}]}},
        "$defs": {"d": {"type": ["object", "null"], "properties": {}, "maxProperties": 3}}})
    assert rewritten["properties"]["v"]["anyOf"][0]["additionalProperties"] is False
    assert rewritten["$defs"]["d"]["additionalProperties"] is False and "maxProperties" not in rewritten["$defs"]["d"]


def test_json_schema_still_cannot_be_combined_with_tools_for_native_models():
    tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}
    with pytest.raises(BadRequest):
        json_request(OPUS55, tools=[tool])
