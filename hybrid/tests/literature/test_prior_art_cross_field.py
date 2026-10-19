"""PA2: cross-field query expansion and the second round, with a fake literature (no network, no hybrid/llm)."""
import asyncio

import jsonschema
import pytest

from hybrid.literature.client import Related, SearchReceipt
from hybrid.literature.paper import Paper, paper_key_of
from hybrid.literature.prior_art import (CLASSIFIER_CANDIDATES, ROUND2_TITLES_SHOWN, SEED_PAPERS, classify_prior_art)
from hybrid.literature.schemas import QUERY_FIELDS
from hybrid.tests.literature.support import RecordingSink

CARD = "Card: a bounded Bayesian program beam with a pessimistic admission rule."


def paper(source_id, title, year=2020):
    return Paper("arxiv", source_id, title, (), year, f"Abstract of {title}.", f"https://arxiv.org/abs/{source_id}",
                 None, None, "t", arxiv_id=source_id)


class FakeLiterature:
    """Answers a search from a table of query -> papers and a related() from a table of paper key -> Related."""

    default_sources = ("arxiv", "openalex", "s2")

    def __init__(self, hits, related=None):
        self.hits, self.related_of, self.searched, self.related_calls = hits, related or {}, [], []

    def search_with_receipt(self, query, sources, limit, sink):
        self.searched.append(query)
        papers = tuple(self.hits.get(query, ()))
        return SearchReceipt(query, sources, limit, "t", {s: tuple(p for p in papers if p.source == s)
                                                       for s in sources}, {}, papers)

    def related(self, seed, sink=None):
        self.related_calls.append(paper_key_of(seed))
        refs, cites = self.related_of.get(paper_key_of(seed), ((), ()))
        return Related(paper_key_of(seed), "openalex", tuple(refs), tuple(cites), {})


class Completion:
    def __init__(self, parsed):
        self.parsed = parsed


def scripted(*completions):
    queue, calls = list(completions), []

    async def complete(role, system, user, json_schema):
        calls.append({"system": system, "user": user, "schema": json_schema})
        return queue.pop(0)

    complete.calls = calls
    return complete


def tagged(n=12):
    return [{"field": QUERY_FIELDS[i % len(QUERY_FIELDS)], "query": f"qry{i}"} for i in range(n)]


NOVEL = Completion({"coverage_assessment": {"adequate_for_novelty": True, "reason": "Scripted coverage for routing test."}, "label": "novel", "evidence": [], "notes": "n"})


def run(lit, *completions):
    llm = scripted(*completions)
    sink = RecordingSink()
    return asyncio.run(classify_prior_art(lit, CARD, llm, sink)), llm, sink


def test_round_one_runs_every_tagged_query_and_records_the_fields():
    lit = FakeLiterature({})
    result, llm, sink = run(lit, Completion({"queries": tagged(12)}), Completion({"queries": []}), NOVEL)
    assert lit.searched == [f"qry{i}" for i in range(12)]
    assert result["query_fields"] == [QUERY_FIELDS[i % len(QUERY_FIELDS)] for i in range(12)]
    assert sink.of_type("prior_art_queries")[0]["fields"] == result["query_fields"]
    assert "program_synthesis" in llm.calls[0]["system"] and "10 and 14" in llm.calls[0]["system"]


def test_a_query_without_a_known_field_is_refused():
    bad = tagged(10)
    bad[3] = {"field": "numerology", "query": "qry3"}
    with pytest.raises(jsonschema.ValidationError):
        run(FakeLiterature({}), Completion({"queries": bad}))


def test_round_two_follows_references_of_the_best_hits_and_runs_refined_queries():
    popular = paper("1111.0001", "Popular paper")
    single = paper("1111.0002", "Found once")
    ref = paper("1110.5667", "Inducing Probabilistic Programs by Bayesian Program Merging")
    cite = paper("2204.09837", "Memory Bounds for the Experts Problem")
    refined = paper("1904.06317", "Few-Shot Bayesian Imitation Learning with Logical Program Policies")
    lit = FakeLiterature(
        {"qry0": [popular, single], "qry1": [popular], "qry2": [popular], "refined query": [refined]},
        {paper_key_of(popular): ([ref], [cite])},
    )
    result, llm, sink = run(lit, Completion({"queries": tagged(10)}),
                            Completion({"queries": ["refined query"]}), NOVEL)

    assert lit.related_calls[0] == paper_key_of(popular)       # the paper three queries agree on seeds first
    assert len(lit.related_calls) <= SEED_PAPERS
    assert lit.searched[-1] == "refined query" and result["round2_queries"] == ["refined query"]
    keys = [c["key"] for c in result["candidates"]]
    assert {paper_key_of(p) for p in (ref, cite, refined)} <= set(keys)
    assert keys[0] == paper_key_of(popular)                    # best-ranked first
    refine_call = llm.calls[1]
    assert "Popular paper" in refine_call["user"] and refine_call["schema"]["properties"]["queries"]["maxItems"] == 4
    assert sink.of_type("prior_art_round2_queries") == [
        {"queries": ["refined query"], "seeds": lit.related_calls[:len(lit.related_calls)]}]


def test_more_than_four_refined_queries_are_refused():
    with pytest.raises(jsonschema.ValidationError):
        run(FakeLiterature({}), Completion({"queries": tagged(10)}),
            Completion({"queries": [f"r{i}" for i in range(5)]}))


def test_the_classifier_prompt_is_bounded_and_the_stored_candidates_are_not():
    many = [paper(f"2000.{i:05d}", f"Paper number {i}") for i in range(CLASSIFIER_CANDIDATES + 15)]
    lit = FakeLiterature({"qry0": many})
    result, llm, _ = run(lit, Completion({"queries": tagged(10)}), Completion({"queries": []}), NOVEL)
    judged = llm.calls[2]["user"]
    assert judged.count("\n- [") == CLASSIFIER_CANDIDATES
    assert f"the {CLASSIFIER_CANDIDATES} best-ranked of {len(many)} unique papers" in judged
    assert len(result["candidates"]) == len(many)
    assert ROUND2_TITLES_SHOWN == llm.calls[1]["user"].count("\n- (")


def test_a_paper_beyond_the_prompt_can_still_be_cited_when_it_was_returned():
    many = [paper(f"2000.{i:05d}", f"Paper number {i}") for i in range(CLASSIFIER_CANDIDATES + 5)]
    last = many[-1]
    lit = FakeLiterature({"qry0": many})
    cite = Completion({"coverage_assessment": {"adequate_for_novelty": True, "reason": "Scripted coverage for routing test."}, "label": "prior_art_undertested", "notes": "n", "evidence": [
        {"source": "arxiv", "source_id": last.source_id, "relation": "closest_prior_work"}]})
    result, _, _ = run(lit, Completion({"queries": tagged(10)}), Completion({"queries": []}), cite)
    assert [e["source_id"] for e in result["evidence"]] == [last.source_id]
