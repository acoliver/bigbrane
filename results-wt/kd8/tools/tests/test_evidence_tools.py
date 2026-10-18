"""Offline regressions for saved-run scoring and exact evidence attribution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
EXTRACT = ROOT / "results-wt/kd8/tools/extract_live.py"
SCORE = ROOT / "results-wt/r2-bar/tools/score.py"
spec = importlib.util.spec_from_file_location("kd8_extract", EXTRACT)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


def event(seq, kind, cid, **data):
    return {"seq": seq, "type": kind, "ts": f"2026-10-18T00:00:{seq:02d}+00:00",
            "data": dict(data, **({"card_id": cid} if cid else {}))}


def test_span_pairing_never_zips_interleaved_cards_or_loses_missing_results():
    events = [event(1, "prior_art_queries", "card-aaa"), event(2, "prior_art_queries", "card-bbb"),
              event(3, "prior_art_result", "card-bbb"), event(4, "prior_art_result", "card-aaa"),
              event(5, "prior_art_queries", "card-aaa")]
    assert [(a["seq"], b["seq"] if b else None) for a, b in extractor.card_spans(events, "card-aaa")] == [(1, 4), (5, None)]
    assert [(a["seq"], b["seq"]) for a, b in extractor.card_spans(events, "card-bbb")] == [(2, 3)]


def test_attribution_uses_primary_json_card_only():
    assert extractor.prompt_card('CARD:\n{"id":"card-aaa"}\nHISTORY card-bbb') == "card-aaa"
    assert extractor.prompt_card("HISTORY card-bbb") is None
    assert extractor.prompt_card('{"id":"card-aaa-suffix"}') is None
    assert extractor.prompt_card('{"title":"card-aaa"}') is None


@pytest.mark.parametrize("title", ["Willshaw", "Storing sequences in a memory", "General factorization machines tutorial"])
def test_partial_titles_do_not_count_as_registered_works(title):
    candidate = {"source": "openalex", "source_id": "W1", "title": title}
    for works in extractor.REFEREE.values():
        assert not any(extractor.exact_match(candidate, work) for work in works.values())


def test_match_by_full_normalized_title_or_exact_arxiv_id():
    work = extractor.REFEREE["card-89248ee5"]["Sacramento"]
    assert extractor.exact_match({"source": "s2", "source_id": "ABC", "title": work[1].upper()}, work) == "full_title"
    assert extractor.exact_match({"source": "arxiv", "source_id": "1207.7196v2", "title": "Different title"}, work) == "arxiv_id"
    assert extractor.exact_match({"source": "s2", "source_id": "1207.7196", "title": "Wrong source"}, work) is None


def test_exposure_positions_are_read_from_prompt_listing_not_candidate_rank():
    call = {"role": "prior_art", "call_id": "x", "user": 'THEORY CARD\n{}\nSEARCH RESULTS (2 of 90)\n'
            '- [arxiv:1234.5678] (2020) Work A\n- [s2:abc] (2020) Work B\n'}
    assert extractor.exposure(call) == {"kind": "classifier", "call_id": "x", "keys": ["arxiv:1234.5678", "s2:abc"]}


def test_extract_costs_filter_card_and_attempt(tmp_path):
    run = tmp_path / "runs/interleaved"
    (run / "llm/calls").mkdir(parents=True)
    events = [event(1, "prior_art_queries", "card-aaa", queries=["query a"], fields=["optimization"]),
              event(2, "prior_art_queries", "card-bbb", queries=["query b"], fields=["signal_processing"]),
              event(3, "literature_search", "card-aaa", query="query a", errors={"arxiv": "429"}),
              event(4, "literature_receipt", "card-aaa"),
              event(5, "literature_search", "card-bbb", query="query b", errors={}),
              event(6, "prior_art_result", "card-bbb", candidates=[], label="novel"),
              event(7, "prior_art_result", "card-aaa", candidates=[], label="novel"),
              event(8, "run_end", None, reason="complete", queued_unfinished={})]
    (run / "ledger.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    for cid, end in (("card-aaa", 7), ("card-bbb", 6)):
        call = {"user": json.dumps({"id": cid}), "role": "prior_art", "call_id": cid,
                "started_at": "2026-10-18T00:00:00+00:00", "finished_at": f"2026-10-18T00:00:0{end}+00:00",
                "attempts": [{"kind": "request"}]}
        (run / f"llm/calls/{cid}.json").write_text(json.dumps(call))
    out = extractor.extract(tmp_path)
    a = out["cards"]["card-aaa"][0]["attempts"][0]
    b = out["cards"]["card-bbb"][0]["attempts"][0]
    assert a["cost"]["searches"] == b["cost"]["searches"] == 1
    assert a["cost"]["source_search_receipts"] == 1
    assert b["cost"]["source_search_receipts"] == 0
    assert a["source_errors"] == {"arxiv": 1} and b["source_errors"] == {}
    assert a["cost"]["model_request_attempts"] == b["cost"]["model_request_attempts"] == 1
    assert a["cost"]["http_requests"] is None
    assert a["cost"]["proposal_to_result_wall_s"] == 7
    assert b["cost"]["proposal_to_result_wall_s"] == 6


def test_scorer_keeps_named_gate_failures_even_when_every_card_dies(tmp_path):
    root = tmp_path / "replay"
    (root / "tools").mkdir(parents=True)
    (root / "tools/score.py").write_text(SCORE.read_text())
    run = root / "runs/test"
    run.mkdir(parents=True)
    cards = {f"card-1710000{i}": "gate: update rule reduces to other_known" for i in range(1, 9)}
    cards.update({"card-17100009": "gate: excluded_family:ssm", "card-1710000a": "gate: excluded_family:delta_rule"})
    counted = ["card-76ee768c", "card-440f0204", "card-62785c10", "card-83bd879a", "card-97baae43", "card-d6dfeac2"]
    cards.update({cid: "gate: update rule reduces to other_known" for cid in counted})
    cards["card-17100007"] = "gate: prior_art_tested and no untested element"
    for cid in counted[:3]:
        cards[cid] = "gate: no_difference"
    (run / "ledger.jsonl").write_text("\n".join(json.dumps({"type": "verdict", "data": {
        "stage": "S2", "card_id": cid, "verdict": "reduces_to_known" if reason.startswith(("gate: update rule reduces", "gate: excluded_family")) else "fail", "reasons": [reason]}}) for cid, reason in cards.items()))
    subprocess.run([sys.executable, str(root / "tools/score.py")], check=True, capture_output=True)
    text = (root / "SCORE.md").read_text()
    assert "registered named-gate predicate: NOT MET" in text
    assert "registered reduction-gate predicate: NOT MET" in text
    assert "9 of 10" in text and "3 of 6" in text
    assert "Probe: all dead; counted-before: all dead" in text
    for cid in cards:
        cards[cid] = "gate: excluded_family:ssm" if cid in ("card-17100009", "card-1710000a") else "gate: update rule reduces to other_known"
    (run / "ledger.jsonl").write_text("\n".join(json.dumps({"type": "verdict", "data": {
        "stage": "S2", "card_id": cid, "verdict": "reduces_to_known", "reasons": [reason]}}) for cid, reason in cards.items()))
    subprocess.run([sys.executable, str(root / "tools/score.py")], check=True, capture_output=True)
    text = (root / "SCORE.md").read_text()
    assert "registered named-gate predicate: MET" in text
    assert "registered reduction-gate predicate: MET" in text


def test_pa2_guard_does_not_substitute_an_unregistered_later_failure():
    assert not extractor.registered_guard({"verdict": "fail", "reasons": ["gate: no_difference"]})
    assert extractor.registered_guard({"verdict": "reduces_to_known", "reasons": ["gate: update rule reduces to other_known"]})
    assert extractor.registered_guard({"verdict": "fail", "reasons": ["gate: prior_art_tested and no untested element"]})


def test_incomplete_attempt_does_not_absorb_next_attempt_searches(tmp_path):
    run = tmp_path / "runs/incomplete"
    run.mkdir(parents=True)
    events = [event(1, "prior_art_queries", "card-aaa", queries=["query one"], fields=["optimization"]),
              event(2, "literature_search", "card-aaa", query="query one", errors={"arxiv": "429"}),
              event(3, "prior_art_queries", "card-aaa", queries=["query two"], fields=["optimization"]),
              event(4, "literature_search", "card-aaa", query="query two", errors={}),
              event(5, "prior_art_result", "card-aaa", candidates=[], label="novel")]
    (run / "ledger.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    attempts = extractor.extract(tmp_path)["cards"]["card-aaa"][0]["attempts"]
    assert [attempt["cost"]["searches"] for attempt in attempts] == [1, 1]
    assert [attempt["source_errors"] for attempt in attempts] == [{"arxiv": 1}, {}]
    assert [attempt["complete"] for attempt in attempts] == [False, True]
    assert attempts[0]["cost"]["queries_to_result_wall_s"] is None
