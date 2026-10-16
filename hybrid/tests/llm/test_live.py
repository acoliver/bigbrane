"""Live tests: real llxprt profiles and LM Studio. Run with `pytest -m live`. Four model calls in total.

Transcripts and events land in runs/llm-live-<timestamp>/ so measurements can be inspected afterwards.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from hybrid.llm import JsonlSink, LLMClient

ROOT = Path(__file__).resolve().parents[3]
MODELS_TOML = ROOT / "hybrid" / "config" / "models.toml"
LM_STUDIO = "http://127.0.0.1:1234/v1"
CITY_SCHEMA = {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}},
               "additionalProperties": False}
PAPER_SCHEMA = {"type": "object", "required": ["arxiv_id", "title"],
                "properties": {"arxiv_id": {"type": "string"}, "title": {"type": "string"}},
                "additionalProperties": False}


@pytest.fixture(scope="module")
def client():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"llm-live-{stamp}"
    run_dir.mkdir(parents=True)
    return LLMClient(str(MODELS_TOML), str(run_dir), JsonlSink(run_dir / "events.jsonl"))


def transcript(result):
    return json.loads(Path(result.transcript_path).read_text())


def assert_no_tools(result):
    attempts = transcript(result)["attempts"]
    assert attempts and all(a["detail"]["tools"]["totalCalls"] == 0 for a in attempts)


def record(result):
    print(f"\nMEASURE role={result.role} model={result.model} latency_s={result.latency_s} usage={result.usage}")


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("role, family", [("salad_critic", "haiku"), ("formalizer", "sonnet")])
async def test_llxprt_json_call_with_tools_disabled(client, role, family):
    result = await client.complete(role, "You answer with JSON only.", "What is the capital of France?",
                                   json_schema=CITY_SCHEMA)
    record(result)
    assert family in result.model
    assert result.backend == "llxprt"
    assert result.parsed["city"].lower() == "paris"
    assert_no_tools(result)
    assert result.usage["input"] < 6000, "system prompt overhead is not minimized"


@pytest.mark.live
@pytest.mark.asyncio
async def test_llxprt_web_call_uses_only_web_tools(client):
    result = await client.complete(
        "prior_art", "You are a careful literature searcher. Use web search; never invent identifiers.",
        "Find one real arXiv paper on reservoir computing. Reply with its arXiv id and exact title.",
        json_schema=PAPER_SCHEMA, allow_web=True)
    record(result)
    assert "haiku" in result.model
    assert re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", result.parsed["arxiv_id"]), result.parsed
    tools = transcript(result)["attempts"][0]["detail"]["tools"]
    assert tools["totalCalls"] >= 1
    assert set(tools["byName"]) <= {"exa_web_search", "direct_web_fetch"}


@pytest.mark.live
@pytest.mark.asyncio
async def test_lm_studio_call(client):
    try:
        listing = httpx.get(f"{LM_STUDIO}/models", timeout=5).json()
    except httpx.HTTPError as exc:
        pytest.skip(f"LM Studio is not reachable at {LM_STUDIO}: {exc}")
    models = [m["id"] for m in listing.get("data", []) if "embed" not in m["id"].lower()]
    if not models:
        pytest.skip(f"LM Studio at {LM_STUDIO} has no chat model loaded")
    result = await client.complete("local", "Reply with one word.", "Say the word: pong", max_output_tokens=2000)
    record(result)
    assert result.backend == "openai_compat" and result.model in models
    assert result.text.strip()
