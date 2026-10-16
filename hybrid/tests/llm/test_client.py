"""LLMClient behavior with fake backends: routing, fallback, JSON mode, budgets, records, concurrency."""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from hybrid.llm import (BackendError, BudgetExceeded, ConfigError, JsonlSink, LLMClient, LLMResult,
                        SchemaValidationError, ToolPolicyViolation, TransportError)
from hybrid.llm.jsonmode import extract_first_json_object

from .fakes import FakeBackend, ListSink, transport_failure, write_config

SCHEMA = {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}},
          "additionalProperties": False}


def make_client(tmp_path, a=None, b=None, sink=None, **config_kwargs):
    a = a or FakeBackend()
    b = b or FakeBackend()
    sink = sink if sink is not None else ListSink()
    client = LLMClient(write_config(tmp_path, **config_kwargs), str(tmp_path / "run"), sink,
                       backends={"a": a, "b": b})
    return client, a, b, sink


# ----- routing and result ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_entry_is_used_and_result_is_populated(tmp_path):
    client, a, b, _ = make_client(tmp_path, a=FakeBackend(default="hello"))
    result = await client.complete("r", "sys", "usr")
    assert isinstance(result, LLMResult)
    assert (result.role, result.backend, result.model, result.text) == ("r", "openai_compat", "resolved-m1", "hello")
    assert result.parsed is None
    assert result.usage == {"input": 10, "output": 5, "cached": 0}
    assert result.call_id.startswith("call-") and len(result.call_id) == len("call-") + 12
    assert result.latency_s >= 0
    assert len(a.requests) == 1 and b.requests == []
    assert a.requests[0].system == "sys" and a.requests[0].user == "usr"


@pytest.mark.asyncio
async def test_unknown_role_is_a_config_error(tmp_path):
    client, *_ = make_client(tmp_path)
    with pytest.raises(ConfigError, match="unknown role"):
        await client.complete("nope", "s", "u")


def test_complete_sync_wrapper(tmp_path):
    client, *_ = make_client(tmp_path, a=FakeBackend(default="sync ok"))
    assert client.complete_sync("solo", "s", "u").text == "sync ok"
    assert client.complete_sync("solo", "s", "u").text == "sync ok"   # a second event loop works too


# ----- fallback -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transport_error_falls_back_and_is_recorded(tmp_path):
    a = FakeBackend(script=[transport_failure("HTTP 503")])
    client, a, b, sink = make_client(tmp_path, a=a, b=FakeBackend(default="from b"))
    result = await client.complete("r", "s", "u")
    assert result.text == "from b" and result.model == "resolved-m2"
    fallback = sink.of("llm_fallback")
    assert len(fallback) == 1
    assert fallback[0]["data"]["backend"] == "a" and "HTTP 503" in fallback[0]["data"]["error"]
    transcript = json.loads(Path(result.transcript_path).read_text())
    assert transcript["fallbacks"][0]["error"] == "HTTP 503"
    assert [attempt["outcome"] for attempt in transcript["attempts"]] == ["TransportError", "ok"]
    assert len(sink.of("llm_call")) == 1


@pytest.mark.asyncio
async def test_all_entries_failing_raises_and_still_records(tmp_path):
    a = FakeBackend(script=[transport_failure("a down")])
    b = FakeBackend(script=[transport_failure("b down")])
    client, _, _, sink = make_client(tmp_path, a=a, b=b)
    with pytest.raises(TransportError, match="a down.*b down"):
        await client.complete("r", "s", "u")
    event = sink.of("llm_call")[0]["data"]
    assert event["error"].startswith("TransportError") and event["response_sha256"] is None
    assert len(sink.of("llm_fallback")) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [BackendError("HTTP 400"), ToolPolicyViolation("used shell")])
async def test_non_transport_errors_do_not_fall_back(tmp_path, error):
    client, a, b, sink = make_client(tmp_path, a=FakeBackend(script=[error]))
    with pytest.raises(type(error)):
        await client.complete("r", "s", "u")
    assert b.requests == [] and sink.of("llm_fallback") == []
    assert sink.of("llm_call")[0]["data"]["error"].startswith(type(error).__name__)


# ----- JSON mode ------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ('{"city": "Paris"}', {"city": "Paris"}),
    ('```json\n{"city": "Paris"}\n```', {"city": "Paris"}),
    ('Sure! Here you go:\n```\n{"city": "Paris", "n": {"x": [1, 2]}}\n```\nDone.', {"city": "Paris", "n": {"x": [1, 2]}}),
    ('{"city": "a } brace"} trailing {"other": 1}', {"city": "a } brace"}),
    ('{not json} then {"city": "Rome"}', {"city": "Rome"}),
])
def test_extract_first_json_object(text, expected):
    assert extract_first_json_object(text) == expected


@pytest.mark.parametrize("text", ["no json here", "[1, 2, 3]", "{broken", ""])
def test_extract_rejects_non_objects(text):
    with pytest.raises(ValueError):
        extract_first_json_object(text)


@pytest.mark.asyncio
async def test_json_mode_parses_fenced_output_and_instructs_schema(tmp_path):
    client, a, *_ = make_client(tmp_path, a=FakeBackend(default='```json\n{"city": "Paris"}\n```'))
    result = await client.complete("r", "s", "capital?", json_schema=SCHEMA)
    assert result.parsed == {"city": "Paris"}
    assert len(a.requests) == 1
    assert "capital?" in a.requests[0].user and '"city"' in a.requests[0].user and "JSON" in a.requests[0].user


@pytest.mark.asyncio
async def test_json_mode_retries_once_with_validation_error_and_logs_both(tmp_path):
    a = FakeBackend(script=['{"city": 5}', '{"city": "Paris"}'])
    client, a, b, sink = make_client(tmp_path, a=a)
    result = await client.complete("r", "s", "capital?", json_schema=SCHEMA)
    assert result.parsed == {"city": "Paris"}
    assert len(a.requests) == 2 and b.requests == []
    assert "schema validation failed at city" in a.requests[1].user
    assert '{"city": 5}' in a.requests[1].user
    assert result.usage["input"] == 20            # both attempts are counted
    transcript = json.loads(Path(result.transcript_path).read_text())
    kinds = [attempt["kind"] for attempt in transcript["attempts"]]
    assert kinds == ["request", "schema_retry"]
    assert transcript["attempts"][0]["raw_output"] == '{"city": 5}'
    assert "city" in transcript["attempts"][0]["validation_error"]
    assert transcript["attempts"][1]["validation_error"] is None
    assert len(sink.of("llm_call")[0]["data"]["attempts"]) == 2


@pytest.mark.asyncio
async def test_json_mode_unparseable_output_also_retries(tmp_path):
    a = FakeBackend(script=["I cannot do that", '{"city": "Oslo"}'])
    client, *_ = make_client(tmp_path, a=a)
    result = await client.complete("r", "s", "u", json_schema=SCHEMA)
    assert result.parsed == {"city": "Oslo"}


@pytest.mark.asyncio
async def test_json_mode_failure_after_retry_raises_without_fallback(tmp_path):
    a = FakeBackend(script=['{"city": 1}', '{"wrong": true}'])
    client, a, b, sink = make_client(tmp_path, a=a)
    with pytest.raises(SchemaValidationError):
        await client.complete("r", "s", "u", json_schema=SCHEMA)
    assert len(a.requests) == 2 and b.requests == []
    assert sink.of("llm_fallback") == []
    event = sink.of("llm_call")[0]["data"]
    assert event["error"].startswith("SchemaValidationError")
    assert [attempt["kind"] for attempt in event["attempts"]] == ["request", "schema_retry"]
    transcript = json.loads(Path(event["transcript_path"]).read_text())
    assert transcript["attempts"][1]["raw_output"] == '{"wrong": true}'


@pytest.mark.asyncio
async def test_invalid_schema_is_rejected_before_any_call(tmp_path):
    client, a, *_ = make_client(tmp_path)
    with pytest.raises(Exception, match="not valid"):
        await client.complete("r", "s", "u", json_schema={"type": "nonsense"})
    assert a.requests == []


# ----- budgets --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_calls_raises_before_the_call(tmp_path):
    client, a, _, sink = make_client(tmp_path, max_calls=2)
    await client.complete("solo", "s", "u")
    await client.complete("solo", "s", "u")
    with pytest.raises(BudgetExceeded, match="max_calls=2"):
        await client.complete("solo", "s", "u")
    assert len(a.requests) == 2
    assert sink.of("llm_call")[-1]["data"]["error"].startswith("BudgetExceeded")


@pytest.mark.asyncio
async def test_retry_and_fallback_requests_count_against_max_calls(tmp_path):
    a = FakeBackend(script=['{"x": 1}'])        # invalid, forces a retry request
    client, a, *_ = make_client(tmp_path, a=a, max_calls=1)
    with pytest.raises(BudgetExceeded):
        await client.complete("solo", "s", "u", json_schema=SCHEMA)
    assert len(a.requests) == 1


@pytest.mark.asyncio
async def test_max_tokens_blocks_the_next_call(tmp_path):
    client, a, *_ = make_client(tmp_path, max_tokens=15)      # each fake call uses 15 tokens
    await client.complete("solo", "s", "u")
    with pytest.raises(BudgetExceeded, match="max_tokens=15"):
        await client.complete("solo", "s", "u")
    assert len(a.requests) == 1


@pytest.mark.asyncio
async def test_budget_resumes_from_existing_transcripts(tmp_path):
    client, *_ = make_client(tmp_path, max_calls=2)
    await client.complete("solo", "s", "u")
    await client.complete("solo", "s", "u")
    resumed, a, *_ = make_client(tmp_path, max_calls=2)
    with pytest.raises(BudgetExceeded):
        await resumed.complete("solo", "s", "u")
    assert a.requests == []


# ----- transcripts and events -----------------------------------------------------------

@pytest.mark.asyncio
async def test_transcript_and_event_content(tmp_path):
    sink = JsonlSink(tmp_path / "events.jsonl")
    client, *_ = make_client(tmp_path, a=FakeBackend(default="the answer"), sink=sink)
    result = await client.complete("solo", "SYSTEM", "USER", max_output_tokens=77)
    path = Path(result.transcript_path)
    assert path.parent == (tmp_path / "run" / "llm" / "calls").resolve()
    assert path.name == f"{result.call_id}.json"
    record = json.loads(path.read_text())
    for key in ("system", "user", "text", "parsed", "usage", "latency_s", "backend", "model", "error",
                "attempts", "prompt_sha256", "response_sha256"):
        assert key in record
    assert (record["system"], record["user"], record["text"]) == ("SYSTEM", "USER", "the answer")
    assert record["max_output_tokens"] == 77 and record["error"] is None
    assert record["response_sha256"] == hashlib.sha256(b"the answer").hexdigest()

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == ["llm_call"]
    data = events[0]["data"]
    assert events[0]["actor"] == "llm:solo"
    assert data["call_id"] == result.call_id and data["role"] == "solo"
    assert data["prompt_sha256"] == record["prompt_sha256"]
    assert data["response_sha256"] == hashlib.sha256(b"the answer").hexdigest()
    assert data["transcript_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert data["transcript_relpath"] == f"llm/calls/{result.call_id}.json"
    assert data["usage"] == result.usage and data["error"] is None


@pytest.mark.asyncio
async def test_prompt_hash_depends_on_both_system_and_user(tmp_path):
    client, *_ = make_client(tmp_path)
    one = await client.complete("solo", "s1", "u")
    two = await client.complete("solo", "s2", "u")
    hashes = [json.loads(Path(r.transcript_path).read_text())["prompt_sha256"] for r in (one, two)]
    assert hashes[0] != hashes[1]


def test_jsonl_sink_sequences_and_resumes(tmp_path):
    path = tmp_path / "e.jsonl"
    first = JsonlSink(path).append("t", "who", {"k": 1})
    second = JsonlSink(path).append("t", "who", {"k": 2})
    assert (first["seq"], second["seq"]) == (1, 2)
    assert len(path.read_text().splitlines()) == 2


# ----- concurrency ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backend_concurrency_is_capped_by_semaphore(tmp_path):
    a = FakeBackend(delay=0.05)
    client, a, *_ = make_client(tmp_path, a=a, concurrency_a=2)
    await asyncio.gather(*[client.complete("solo", "s", f"u{i}") for i in range(7)])
    assert len(a.requests) == 7
    assert a.max_in_flight == 2


@pytest.mark.asyncio
async def test_backends_have_independent_semaphores(tmp_path):
    a = FakeBackend(delay=0.05)
    b = FakeBackend(delay=0.05)
    client, a, b, _ = make_client(tmp_path, a=a, b=b, concurrency_a=1, concurrency_b=3)
    await asyncio.gather(*[client.complete("solo", "s", f"u{i}") for i in range(4)],
                         *[client.complete("solo_b", "s", f"u{i}") for i in range(6)])
    assert a.max_in_flight == 1
    assert b.max_in_flight == 3


# ----- role settings and run id ---------------------------------------------------------

ROLE_SETTINGS = '\n[role_settings.r]\neffort = "low"\nmax_output_tokens = 700\n'


@pytest.mark.asyncio
async def test_role_settings_reach_every_entry_and_the_transcript(tmp_path):
    client, a, b, sink = make_client(tmp_path, a=FakeBackend(script=[transport_failure()]), extra=ROLE_SETTINGS)
    result = await client.complete("r", "s", "u")
    for request in (a.requests[0], b.requests[0]):
        assert (request.effort, request.max_output_tokens) == ("low", 700)
        assert request.run_id == "run"                     # the run directory name
    transcript = json.loads(Path(result.transcript_path).read_text())
    assert transcript["routing"] == "main"
    assert [(x["effort"], x["max_output_tokens"]) for x in transcript["attempts"]] == [("low", 700), ("low", 700)]
    assert sink.of("llm_call")[0]["data"]["attempts"][1]["effort"] == "low"


@pytest.mark.asyncio
async def test_explicit_max_output_tokens_wins_over_role_settings(tmp_path):
    client, a, _, _ = make_client(tmp_path, extra=ROLE_SETTINGS)
    await client.complete("r", "s", "u", max_output_tokens=55)
    assert (a.requests[0].effort, a.requests[0].max_output_tokens) == ("low", 55)


@pytest.mark.asyncio
async def test_roles_without_settings_send_no_effort_or_cap(tmp_path):
    client, a, _, _ = make_client(tmp_path, extra=ROLE_SETTINGS)
    await client.complete("solo", "s", "u")
    assert (a.requests[0].effort, a.requests[0].max_output_tokens) == (None, None)


def test_client_routing_argument_selects_the_table(tmp_path):
    extra = '\n[routes.other]\nr = [ { backend = "b", model = "x" } ]\nsolo = [ { backend = "b", model = "y" } ]\n' \
            'solo_b = [ { backend = "b", model = "z" } ]\n'
    client = LLMClient(write_config(tmp_path, extra=extra), str(tmp_path / "run"), ListSink(), routing="other",
                       backends={"a": FakeBackend(), "b": FakeBackend()})
    assert client.config.routing == "other"
    assert [(e.backend, e.model) for e in client.config.roles["solo"]] == [("b", "y")]
