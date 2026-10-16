"""classify_prior_art with a fake model callable and a fake web. No network, no hybrid/llm."""
import asyncio
import json

import httpx
import jsonschema
import pytest

from hybrid.literature.client import Literature
from hybrid.literature.prior_art import classify_prior_art
from hybrid.literature.sources import arxiv
from hybrid.tests.literature.support import FakeClock, FakeWeb, RecordingSink, fixture_text, make_literature, ok, status

QUERIES = ["linear attention recurrent state", "Mamba selective state space", "associative recall fixed state"]
CARD = "Card: a sample-efficient sequence learner based on sheaf diffusion over token graphs."


class Completion:
    def __init__(self, parsed):
        self.parsed = parsed


def fake_llm(*completions):
    """Returns the scripted completions in order and records each call's arguments."""
    queue = list(completions)
    calls = []

    async def complete(role, system, user, json_schema):
        calls.append({"role": role, "system": system, "user": user, "json_schema": json_schema})
        return queue.pop(0)

    complete.calls = calls
    return complete


def build(tmp_path):
    web = FakeWeb(
        {
            "export.arxiv.org": ok(fixture_text("arxiv_search.xml")),
            "api.openalex.org": ok(fixture_text("openalex_search.json")),
            "api.semanticscholar.org": status(404),
        }
    )
    literature = make_literature(tmp_path, web, clock=FakeClock())
    return literature, web


def grounded_id():
    return arxiv.parse_search(fixture_text("arxiv_search.xml"), "t")[0].source_id


def test_classification_is_grounded_in_search_hits_and_rejects_invented_ids(tmp_path):
    literature, _ = build(tmp_path)
    sink = RecordingSink()
    llm = fake_llm(
        Completion({"queries": QUERIES}),
        Completion(
            {
                "label": "prior_art_undertested",
                "evidence": [
                    {"source": "arxiv", "source_id": grounded_id(), "relation": "closest_prior_work"},
                    {"source": "arxiv", "source_id": "9999.99999", "relation": "same_mechanism"},
                ],
                "notes": "Close to selective state spaces; little testing on recall.",
            }
        ),
    )
    result = asyncio.run(classify_prior_art(literature, CARD, llm, sink))

    assert result["label"] == "prior_art_undertested"
    assert result["queries"] == QUERIES
    (evidence,) = result["evidence"]
    assert evidence["source_id"] == grounded_id()
    assert evidence["title"] == arxiv.parse_search(fixture_text("arxiv_search.xml"), "t")[0].title
    assert evidence["url"] == f"https://arxiv.org/abs/{grounded_id()}"
    assert result["rejected_evidence"] == [
        {"source": "arxiv", "source_id": "9999.99999", "relation": "same_mechanism"}
    ]
    assert f"arxiv:{grounded_id()}" in result["candidates"]


def test_every_query_and_result_is_recorded_through_the_sink(tmp_path):
    literature, _ = build(tmp_path)
    sink = RecordingSink()
    llm = fake_llm(
        Completion({"queries": QUERIES}),
        Completion({"label": "novel", "evidence": [], "notes": "No close work found."}),
    )
    asyncio.run(classify_prior_art(literature, CARD, llm, sink))

    assert sink.of_type("prior_art_queries") == [{"queries": QUERIES}]
    searches = sink.of_type("literature_search")
    assert [event["query"] for event in searches] == QUERIES
    assert all(event["sources"] == ["arxiv", "openalex", "s2"] for event in searches)
    (result_event,) = sink.of_type("prior_art_result")
    assert result_event["label"] == "novel"
    assert [event_type for event_type, _, _ in sink.events][0] == "prior_art_queries"


def test_judgment_prompt_lists_found_titles_and_abstracts(tmp_path):
    literature, _ = build(tmp_path)
    llm = fake_llm(
        Completion({"queries": QUERIES}),
        Completion({"label": "novel", "evidence": [], "notes": "none"}),
    )
    asyncio.run(classify_prior_art(literature, CARD, llm, RecordingSink()))
    judgment_call = llm.calls[1]
    assert judgment_call["role"] == "prior_art"
    assert CARD in judgment_call["user"]
    assert f"[arxiv:{grounded_id()}]" in judgment_call["user"]
    assert judgment_call["json_schema"]["properties"]["label"]["enum"] == [
        "novel",
        "prior_art_tested",
        "prior_art_undertested",
        "known_disproven",
    ]


def test_query_proposal_must_have_three_to_five_queries(tmp_path):
    literature, _ = build(tmp_path)
    llm = fake_llm(Completion({"queries": ["only one"]}))
    with pytest.raises(jsonschema.ValidationError):
        asyncio.run(classify_prior_art(literature, CARD, llm, RecordingSink()))


def test_label_outside_the_enum_is_rejected(tmp_path):
    literature, _ = build(tmp_path)
    llm = fake_llm(
        Completion({"queries": QUERIES}),
        Completion({"label": "maybe", "evidence": [], "notes": "unsure"}),
    )
    with pytest.raises(jsonschema.ValidationError):
        asyncio.run(classify_prior_art(literature, CARD, llm, RecordingSink()))


def test_a_model_call_without_parsed_output_stops_classification(tmp_path):
    literature, _ = build(tmp_path)
    llm = fake_llm(Completion(None))
    with pytest.raises(ValueError, match="no parsed JSON"):
        asyncio.run(classify_prior_art(literature, CARD, llm, RecordingSink()))
