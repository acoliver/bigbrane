"""FIX-5: a reply that spent the whole output cap on reasoning and returned no text.

Recorded case (hybrid/tests/research/fixtures/calibration_smoke_20261014/k2_prover_length_empty.json): k2-run1 of
runs/calibration-smoke-20261013T222017Z, prover call for the single construction statement O7.a, effort medium,
max_tokens 64000. The gateway metered both Claude Opus (685.8 s) and Claude Sonnet (540.7 s) at status ok with 64000
output tokens; the chat reply had no content and finish_reason length, the client raised a plain TransportError for
each, recorded no usage, and the stage ended the card-run.
"""
import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from hybrid.gateway.anthropic_translate import RequestContext, reasoning_plan, to_chat
from hybrid.llm import LLMClient, OutputExhausted, TransportError
from hybrid.llm.backends import AnthropicMessagesBackend, BackendRequest, OpenAICompatBackend
from hybrid.llm.config import BackendConfig, load_config

from .fakes import FakeBackend, ListSink, write_config

RECORDED = (Path(__file__).parents[1] / "research" / "fixtures" / "calibration_smoke_20261014"
            / "k2_prover_length_empty.json")
ROOT_CONFIG = Path(__file__).parents[2] / "config" / "models.toml"


def recorded() -> dict:
    return json.loads(RECORDED.read_text(encoding="utf-8"))


def request(**kw):
    return BackendRequest(call_id="call-x", model="anthropic/claude-opus-5-5", system="SYS", user="USR",
                          allow_web=False, max_output_tokens=kw.get("max_output_tokens", 64000), timeout_s=5,
                          scratch_dir=Path("/unused"), effort=kw.get("effort", "medium"))


def empty_at_cap(meter_row: dict) -> dict:
    """The chat completion the gateway returned: no content, finish_reason length, the metered usage."""
    usage = meter_row["usage"]
    return {"id": "msg_x", "model": meter_row["model"].split("/")[-1],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                      "prompt_tokens_details": {"cached_tokens": usage["cached"]}}}


# ----- the cause: adaptive thinking with no budget, thinking blocks dropped -----------------------------------------

def test_the_recorded_call_was_one_statement_at_effort_medium_and_64000_tokens_all_spent():
    rec = recorded()
    assert rec["statement_ids"] == ["O7.a"]
    assert [(a["requested_model"], a["effort"], a["max_output_tokens"]) for a in rec["attempts"]] == [
        ("anthropic/claude-opus-5-5", "medium", 64000), ("anthropic/claude-sonnet-5-5", "medium", 64000)]
    assert all("finish_reason='length'" in a["error"] for a in rec["attempts"])
    assert [row["usage"]["output"] for row in rec["gateway_meter"]] == [64000, 64000]
    assert rec["usage_recorded"] == {"cached": 0, "input": 0, "output": 0}     # the old client lost the spend


def test_the_gateway_sends_adaptive_thinking_without_a_budget_on_claude_5_5():
    plan = reasoning_plan("claude-opus-5-5", "medium")
    assert plan.thinking["type"] == "adaptive" and "budget_tokens" not in plan.thinking
    assert plan.output_config == {"effort": "medium"}


def test_a_message_of_thinking_only_at_max_tokens_becomes_empty_content_with_finish_length():
    message = {"id": "msg_x", "model": "claude-opus-5-5", "stop_reason": "max_tokens",
               "content": [{"type": "thinking", "thinking": "long deliberation", "signature": "s"}],
               "usage": {"input_tokens": 6098, "cache_read_input_tokens": 2154, "output_tokens": 64000}}
    body, usage = to_chat(message, RequestContext(json_tool=False, dropped=()), time.time())
    assert body["choices"][0]["message"]["content"] is None
    assert body["choices"][0]["finish_reason"] == "length"
    assert usage["output"] == 64000


# ----- backends: OutputExhausted with the spent usage --------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_recorded_empty_reply_is_output_exhausted_with_its_usage():
    row = recorded()["gateway_meter"][0]
    config = BackendConfig(name="gw", kind="openai_compat", concurrency=1, timeout_s=5, base_url="http://local/v1")
    backend = OpenAICompatBackend(config, httpx.MockTransport(lambda r: httpx.Response(200, json=empty_at_cap(row))))
    with pytest.raises(OutputExhausted, match=r"empty message content \(finish_reason='length'\)") as raised:
        await backend.complete(request())
    assert raised.value.usage == {"input": 8252, "output": 64000, "cached": 2154}
    assert isinstance(raised.value, TransportError)       # still falls back like any transport failure


@pytest.mark.asyncio
async def test_an_empty_reply_for_another_reason_stays_a_plain_transport_error():
    reply = {"id": "c", "model": "m", "choices": [{"message": {"content": ""}, "finish_reason": "stop"}], "usage": {}}
    config = BackendConfig(name="gw", kind="openai_compat", concurrency=1, timeout_s=5, base_url="http://local/v1")
    backend = OpenAICompatBackend(config, httpx.MockTransport(lambda r: httpx.Response(200, json=reply)))
    with pytest.raises(TransportError) as raised:
        await backend.complete(request())
    assert not isinstance(raised.value, OutputExhausted)


@pytest.mark.asyncio
async def test_anthropic_messages_empty_text_at_max_tokens_is_output_exhausted(monkeypatch):
    monkeypatch.setenv("ANTH_KEY", "k")
    payload = {"id": "m", "model": "claude-opus-5-5", "stop_reason": "max_tokens",
               "content": [{"type": "thinking", "thinking": "..."}], "usage": {"input_tokens": 10, "output_tokens": 99}}
    config = BackendConfig(name="anth", kind="anthropic_messages", concurrency=1, timeout_s=5,
                           base_url="https://api.example", api_key_env="ANTH_KEY", anthropic_version="2023-06-01",
                           default_max_output_tokens=99)
    backend = AnthropicMessagesBackend(config, httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    with pytest.raises(OutputExhausted) as raised:
        await backend.complete(request())
    assert raised.value.usage["output"] == 99


# ----- client: one retry at lower effort, recorded; usage counted; aggregate error ---------------------------------

PROVER_SETTINGS = '\n[role_settings.r]\neffort = "medium"\nmax_output_tokens = 64000\n'


def exhausted(output=64000):
    return OutputExhausted("gateway: empty message content (finish_reason='length')",
                           usage={"input": 8252, "output": output, "cached": 2154})


def make(tmp_path, a, b, extra=PROVER_SETTINGS):
    sink = ListSink()
    client = LLMClient(write_config(tmp_path, extra=extra), str(tmp_path / "run"), sink, backends={"a": a, "b": b})
    return client, sink


def test_an_exhausted_reply_is_retried_once_on_the_same_entry_at_lower_effort(tmp_path):
    a = FakeBackend(script=[exhausted(), "proof"])
    b = FakeBackend()
    client, sink = make(tmp_path, a, b)
    result = asyncio.run(client.complete("r", "s", "u"))
    assert result.text == "proof" and b.requests == []
    assert [(r.effort, r.max_output_tokens) for r in a.requests] == [("medium", 64000), ("low", 64000)]
    retry = sink.of("llm_length_retry")
    assert len(retry) == 1 and retry[0]["data"]["effort_from"] == "medium" and retry[0]["data"]["effort_to"] == "low"
    transcript = json.loads(Path(result.transcript_path).read_text())
    assert [(x["kind"], x["effort"], x["outcome"]) for x in transcript["attempts"]] == [
        ("request", "medium", "OutputExhausted"), ("length_retry", "low", "ok")]
    event = sink.of("llm_call")[0]["data"]
    assert event["attempts"][0]["usage"]["output"] == 64000 and event["attempts"][1]["kind"] == "length_retry"
    # The exhausted reply's tokens count against the call and the run budget.
    assert result.usage["output"] == 64000 + 5
    assert client.budget.tokens >= 8252 + 64000


def test_the_lower_effort_retry_happens_once_per_call_then_falls_back(tmp_path):
    a = FakeBackend(script=[exhausted(), exhausted()])
    b = FakeBackend(script=["from b"])
    client, sink = make(tmp_path, a, b)
    result = asyncio.run(client.complete("r", "s", "u"))
    assert result.text == "from b"
    assert [r.effort for r in a.requests] == ["medium", "low"] and [r.effort for r in b.requests] == ["medium"]
    assert len(sink.of("llm_length_retry")) == 1 and len(sink.of("llm_fallback")) == 1


def test_every_entry_exhausted_raises_output_exhausted_naming_each(tmp_path):
    a = FakeBackend(script=[exhausted(), exhausted()])
    b = FakeBackend(script=[exhausted()])
    client, sink = make(tmp_path, a, b)
    with pytest.raises(OutputExhausted, match="all backend entries failed for role 'r'") as raised:
        asyncio.run(client.complete("r", "s", "u"))
    assert raised.value.call_id
    transcript = json.loads((tmp_path / "run" / "llm" / "calls" / f"{raised.value.call_id}.json").read_text())
    assert transcript["usage"]["output"] == 3 * 64000


def test_a_mixed_failure_is_a_plain_transport_error(tmp_path):
    a = FakeBackend(script=[exhausted(), exhausted()])
    b = FakeBackend(script=[TransportError("HTTP 503")])
    client, _ = make(tmp_path, a, b)
    with pytest.raises(TransportError) as raised:
        asyncio.run(client.complete("r", "s", "u"))
    assert not isinstance(raised.value, OutputExhausted)


def test_a_role_at_effort_low_falls_back_without_a_length_retry(tmp_path):
    a = FakeBackend(script=[exhausted()])
    b = FakeBackend(script=["ok"])
    client, sink = make(tmp_path, a, b, extra='\n[role_settings.r]\neffort = "low"\n')
    assert asyncio.run(client.complete("r", "s", "u")).text == "ok"
    assert len(a.requests) == 1 and sink.of("llm_length_retry") == []


def test_the_prover_cap_leaves_room_for_thinking_within_the_gateway_timeout():
    config = load_config(str(ROOT_CONFIG), "gateway")
    settings = config.settings_for("prover")
    assert settings.effort == "medium" and settings.max_output_tokens == 96000
    # About 93 tokens/s measured on the recorded call (64000 tokens in 685.8 s): the full cap fits the timeout.
    assert settings.max_output_tokens / (64000 / 685.8) < config.backends["gateway"].timeout_s
