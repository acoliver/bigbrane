"""Coverage, distinct-query support and bounded field listings, without network or LLM calls."""
from dataclasses import replace

import jsonschema
import pytest

from hybrid.literature.client import Related
from hybrid.literature.paper import paper_key_of
from hybrid.literature.prior_art import _Pool, listing_receipt, render_receipt, select_candidates
from hybrid.tests.literature.test_prior_art_cross_field import Completion, FakeLiterature, NOVEL, paper, run, tagged


def test_related_fanout_and_duplicate_query_appearances_do_not_outrank_distinct_query_support():
    pool = _Pool()
    a, b = paper("2000.00001", "Two query work"), paper("2000.00002", "Related work")
    pool.add_all([b] * 5)
    pool.add_all([a, a], query="first query")
    pool.add_all([a], query="  FIRST   QUERY ")
    pool.add_all([a], query="second query")
    for _ in range(5):
        pool.add_all([b])
    assert list(pool.ranked()) == [paper_key_of(a), paper_key_of(b)]
    assert pool.provenance(paper_key_of(a))["query_hits"] == 2
    assert pool.provenance(paper_key_of(b))["query_hits"] == 0
    assert pool.provenance(paper_key_of(b))["related_only"] is True


def test_cross_source_identity_merges_support_aliases_and_richer_metadata():
    pool = _Pool()
    a = replace(paper("2000.00001v2", "Same work"), abstract=None, arxiv_id="2000.00001v2")
    b = replace(a, source="openalex", source_id="W1", arxiv_id="2000.00001", abstract="A richer abstract", doi="x")
    pool.add_all([a], query="one query", field="optimization")
    pool.add_all([b], query="two query", field="signal_processing")
    assert len(pool.ranked()) == 1
    key = paper_key_of(a)
    assert pool.papers[key].abstract == "A richer abstract"
    assert pool.papers[key].doi == "x"
    assert pool.provenance(key)["aliases"] == [key, "openalex:W1"]
    assert pool.provenance(key)["query_hits"] == 2
    pool.add_all([replace(b, abstract="A much longer abstract with more detail")])
    assert pool.papers[key].abstract == "A much longer abstract with more detail"


def test_late_identity_bridge_merges_previous_groups_without_losing_votes():
    pool = _Pool()
    a = paper("2000.00001", "Work A")
    b = replace(paper("2000.00002", "Alternate title"), source="s2", source_id="S", arxiv_id=None)
    pool.add_all([a], query="first")
    pool.add_all([b], query="second")
    pool.add_all([replace(b, arxiv_id=a.arxiv_id)], query="third")
    assert len(pool.ranked()) == 1
    assert pool.provenance(paper_key_of(a))["query_hits"] == 3


@pytest.mark.parametrize("bad", [
    [{"field": "optimization", "query": f"query {i}"} for i in range(10)],
    [dict(item, query="Repeated query") for item in tagged(10)],
    [dict(item, query="qry0" if i == 1 else item["query"]) for i, item in enumerate(tagged(10))],
    [dict(item, query="  QRY0  " if i == 1 else item["query"]) for i, item in enumerate(tagged(10))],
])
def test_query_contract_rejects_missing_diversity_and_duplicate_text(bad):
    with pytest.raises(jsonschema.ValidationError):
        run(FakeLiterature({}), Completion({"queries": bad}))


def test_schema_requires_explicit_coverage_assessment():
    bad = Completion({"label": "novel", "evidence": [], "notes": "none"})
    with pytest.raises(jsonschema.ValidationError):
        run(FakeLiterature({}), Completion({"queries": tagged(10)}), Completion({"queries": []}), bad)


class CoverageLiterature(FakeLiterature):
    def __init__(self, hits, failures, related_errors=None):
        super().__init__(hits)
        self.failures, self.related_errors = failures, related_errors or {}

    def search_with_receipt(self, query, sources, limit, sink):
        receipt = super().search_with_receipt(query, sources, limit, sink)
        return replace(receipt, errors=self.failures.get(query, {}))

    def related(self, seed, sink=None):
        return Related(paper_key_of(seed), "s2", (), (), self.related_errors)


def test_unrelated_hit_does_not_hide_unanswered_fields_even_if_model_says_adequate():
    lit = CoverageLiterature({"qry0": [paper("2000.00001", "Unrelated hit")]},
                             {f"qry{i}": {s: "429" for s in FakeLiterature.default_sources} for i in range(1, 10)})
    result, llm, _ = run(lit, Completion({"queries": tagged(10)}), Completion({"queries": []}), NOVEL)
    assert result["candidates"]
    assert result["novelty_unassessed"] is True
    assert result["coverage"]["searches"][1]["errors"]["arxiv"] == "429"
    assert '"arxiv": "429"' in llm.calls[2]["user"]
    assert result["novelty_unassessed_reasons"]


def test_usable_other_sources_and_related_fallback_do_not_automatically_block_novelty():
    lit = CoverageLiterature({"qry0": [paper("2000.00001", "Potential prior work")]},
                             {"qry0": {"arxiv": "429"}}, {"openalex": "503"})
    result, llm, _ = run(lit, Completion({"queries": tagged(10)}), Completion({"queries": []}), NOVEL)
    assert result["novelty_unassessed"] is False
    assert result["coverage"]["related"][0]["errors"] == {"openalex": "503"}
    assert '"openalex": "503"' in llm.calls[2]["user"]
    assert result["coverage"]["searches"][0]["usable_sources"] == ["openalex", "s2"]


def test_model_can_mark_partial_mechanism_coverage_unassessed_with_nonempty_candidates():
    judgment = Completion({"label": "novel", "evidence": [], "notes": "No work found for readout.",
                           "coverage_assessment": {"adequate_for_novelty": False,
                                                   "reason": "Only initialization is covered."}})
    result, _, _ = run(FakeLiterature({"qry0": [paper("2000.00001", "Initialization only")]}),
                       Completion({"queries": tagged(10)}), Completion({"queries": []}), judgment)
    assert result["novelty_unassessed"] is True
    assert "Only initialization is covered." in result["novelty_unassessed_reasons"]


def test_field_representatives_keep_low_ranked_distinct_field_within_listing_cap():
    records = [{"key": f"arxiv:{i}", "rank": i + 1, "query_fields": ["optimization"]} for i in range(80)]
    records += [{"key": "arxiv:other", "rank": 81, "query_fields": ["program_synthesis"]}]
    selected = select_candidates(records, 35)
    assert len(selected) == 35
    assert selected[-1]["key"] == "arxiv:other"
    receipt = listing_receipt(records, [r["key"] for r in selected], 35)
    assert receipt["shown"][-1] == {"key": "arxiv:other", "rank": 81, "position": 35,
                                   "query_fields": ["program_synthesis"]}
    assert receipt["truncated"] is True
    assert len(receipt["omitted_keys"]) == 46
    assert '"rank": 81' in render_receipt({"candidates": records, "coverage": {"errors": ["429"]}},
                                           [r["key"] for r in selected], 35)


def test_cross_round_repeated_query_is_not_searched_or_counted_twice():
    lit = FakeLiterature({"qry0": [paper("2000.00001", "Work")]})
    result, _, _ = run(lit, Completion({"queries": tagged(10)}),
                       Completion({"queries": [" QRY0 "]}), NOVEL)
    assert len(lit.searched) == 10
    assert result["round2_queries"] == []
    assert result["duplicate_round2_queries"] == [" QRY0 "]
    assert result["candidates"][0]["query_hits"] == 1


def test_failed_duplicate_field_query_does_not_block_when_other_query_covers_field():
    lit = CoverageLiterature({"qry0": [paper("2000.00001", "Potential prior work")]},
                             {"qry8": {s: "429" for s in FakeLiterature.default_sources}})
    result, _, _ = run(lit, Completion({"queries": tagged(10)}), Completion({"queries": []}), NOVEL)
    assert result["novelty_unassessed"] is False
    assert result["coverage"]["unanswered_queries"] == ["qry8"]


def test_all_three_system_prompts_match_the_documented_copy():
    from pathlib import Path
    from hybrid.literature.prior_art import CLASSIFY_SYSTEM, QUERY_SYSTEM, REFINE_SYSTEM

    root = Path(__file__).resolve().parents[3]
    text = (root / "hybrid/research/prompts/prior_art.md").read_text()
    assert QUERY_SYSTEM in text
    assert REFINE_SYSTEM in text
    assert CLASSIFY_SYSTEM in text
