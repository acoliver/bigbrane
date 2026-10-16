"""Backfill: every card of every run becomes one record with its kill codes; the source runs are not written."""
import hashlib
import json
import os

import pytest

from hybrid.research.ledger import Ledger
from hybrid.research.outcome_records import backfill
from hybrid.research.outcomes import OutcomeStore
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.outcome_fixtures import KNOWN_MECHANISMS, build_run

PASS = ["ok"]
CRITIC = {"reduces_to": "other_known", "mapping": "m", "confidence": 0.9, "difference_if_any": "",
          "radical_enough": False, "why": "This is not a renamed recurrence. It is weighted union-find with labels.",
          "textbook_transplant": True, "textbook_why": "-"}
S3_GATE = {"rule": "check_violation", "verdict": "infeasible_proved"}
MATH_C = {"stage": "C", "verdict": "fail", "kill_codes": ["circular:C13", "statement_mismatch"]}
LEDGER_C = {"card_id": "card-0000c009", "stage": "C", "definitions": [{"id": "D1", "kind": "notation", "text": "t"}],
            "premises": [{"id": "A1", "status": "open", "text": "p"}]}
CLAIMS = {"falsified_by": [{"claim_id": "falsifier.1", "task_id": "copy_task", "rule_index": 0}]}


def cards():
    def card(n):
        return make_card(f"card-0000c00{n}", title=f"Card {n}")

    up_to = lambda stage: [(s, "pass", PASS) for s in ("S0", "S1", "S2", "S3", "S4", "S5")[:("S0", "S1", "S2", "S3",
                                                                                              "S4", "S5").index(stage)]]
    return [
        (card(1), [("S0", "fail", ["duplicate: TF-IDF cosine 0.712 with card-0000c002 is at or above 0.6"])], {}),
        (card(2), up_to("S1") + [("S1", "fail", ["gate: fatal salad-critic finding", "dimension error in u"])], {}),
        (card(3), up_to("S2") + [("S2", "reduces_to_known", ["gate: update rule reduces to other_known: m"])],
         {"reduction_critic": CRITIC}),
        (card(4), up_to("S2") + [("S2", "fail", ["gate: not radical enough: a renamed SSM"])],
         {"reduction_critic": dict(CRITIC, reduces_to="none")}),
        (card(5), up_to("S3") + [("S3", "infeasible_proved", [
            "gate rule: check_violation", "executed check violated stated obligation 'capacity bound' (replayed): x"])],
         {"s3_gate": S3_GATE}),
        (card(6), up_to("S3") + [("S3", "infeasible_proved", ["confirmed disproof of 'claim 2': counterexample"])],
         {}),
        (card(7), up_to("S2") + [("S2", "pass", PASS)], {}),
        (card(8), up_to("S5") + [("S5", "pass", PASS), ("S6", "pass", ["claims ledger: 1 supported"])],
         {"claims_ledger": CLAIMS}),
        (card(9), up_to("S2") + [("S2", "pass", PASS), ("M", "pass", PASS), ("C", "fail", ["math gate (C): fail"])],
         {"math_verdict": MATH_C, "assumption_ledger": LEDGER_C}),
    ]


def snapshot(root):
    """Every file and directory under root with its bytes hash and mtime; any write shows up."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        out[dirpath] = ("dir", os.stat(dirpath).st_mtime_ns, tuple(sorted(dirnames + filenames)))
        for name in filenames:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as handle:
                out[path] = ("file", os.stat(path).st_mtime_ns, hashlib.sha256(handle.read()).hexdigest())
    return out


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    build_run(root / "run-20261001T000000Z", cards())
    # A run with no cards (a bench run) is not a card run and is skipped.
    (root / "bench-20261001").mkdir()
    (root / "bench-20261001" / "result.json").write_text("{}")
    return root


def by_card(report):
    return {r["card_id"]: r for r in report.records}


def test_every_card_gets_its_kill_code(runs, tmp_path):
    report = backfill(runs, OutcomeStore(tmp_path / "store"))
    assert report.cards == report.added == 9 and report.skipped == 0
    records = by_card(report)
    expected = {
        "card-0000c001": ("dead", "S0", ["duplicate:card-0000c002"]),
        "card-0000c002": ("dead", "S1", ["ill_formed:fatal"]),
        "card-0000c003": ("dead", "S2", ["reduces_to_known:other_known"]),
        "card-0000c004": ("dead", "S2", ["not_radical:R2"]),
        "card-0000c005": ("dead", "S3", ["obligation_failed:capacity_bound"]),
        "card-0000c006": ("dead", "S3", ["infeasible:disproof"]),
        "card-0000c007": ("open", "S2", []),
        "card-0000c008": ("dead", "S6", ["falsified:copy_task"]),
        "card-0000c009": ("dead", "C", ["circular:C13", "statement_mismatch"]),
    }
    assert {cid: (r["outcome"], r["terminal_stage"], r["kill_codes"]) for cid, r in records.items()} == expected
    reduced = records["card-0000c003"]
    assert reduced["kill_detail"].startswith("reduced to other_known: It is weighted union-find")
    evidence = reduced["causes"][0]["evidence"]
    assert evidence[0]["kind"] == "reduction_critic" and len(evidence[0]["sha256"]) == 64
    assert {"kind": "job", "ref": "run-20261001T000000Z/job-0123456789ab"} in evidence
    assert reduced["cost"] == {"model_calls": 1, "tokens": 100, "executor_jobs": 1}


def test_old_cards_are_recorded_with_what_exists_and_flagged_legacy(runs, tmp_path):
    records = by_card(backfill(runs, OutcomeStore(tmp_path / "store")))
    old = records["card-0000c003"]
    assert old["legacy"] and old["difference_statement"] is None and old["premises"] == []
    assert {"difference_statement", "premises", "definitions", "commitments"} <= set(old["missing"])
    math = records["card-0000c009"]
    assert math["premises"] == LEDGER_C["premises"] and math["definitions"] == LEDGER_C["definitions"]
    assert math["legacy"] and "premises" not in math["missing"] and "difference_statement" in math["missing"]


def test_backfill_writes_nothing_inside_the_runs(runs, tmp_path):
    before = snapshot(runs)
    store = OutcomeStore(tmp_path / "store")
    backfill(runs, store)
    backfill(runs, store)
    assert snapshot(runs) == before
    assert not list(runs.rglob("outcomes*")) and not (store.dir / "ledger.lock").exists()


def test_a_second_backfill_skips_every_stored_record_and_adds_a_moved_card(runs, tmp_path):
    store = OutcomeStore(tmp_path / "store")
    backfill(runs, store)
    again = backfill(runs, store)
    assert again.added == 0 and again.skipped == 9
    from hybrid.research.store import Store
    run = runs / "run-20261001T000000Z"
    Store(run, Ledger(run)).set_verdict("card-0000c007", "S3", "undetermined", ["judges did not agree"],
                                        ["call-0123456789ab"])
    third = backfill(runs, store)
    assert third.added == 1 and store.verify() == (True, None)
    current = {s.record["card_id"]: s.record for s in store.current()}
    assert current["card-0000c007"]["kill_codes"] == ["undetermined:S3"]


def test_override_runs_are_flagged(tmp_path):
    root = tmp_path / "runs"
    run = build_run(root / "plumb-20261001T000000Z", [(KNOWN_MECHANISMS["ssm"], [("S0", "pass", PASS)], {})])
    Ledger(run).append("operator_override", "operator", {"source_run_id": "run-20261001T000000Z"})
    record = backfill(root, OutcomeStore(tmp_path / "store")).records[0]
    assert record["override"] and record["source_run"] == "run-20261001T000000Z"


def test_a_card_file_changed_after_the_ledger_recorded_it_stops_the_backfill(runs, tmp_path):
    path = runs / "run-20261001T000000Z" / "cards" / "card-0000c001" / "v2.json"
    card = json.loads(path.read_text())
    card["title"] = "edited"
    path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="sha256 differs"):
        backfill(runs, OutcomeStore(tmp_path / "store"))


def test_a_card_directory_the_ledger_does_not_know_stops_the_backfill(runs, tmp_path):
    (runs / "run-20261001T000000Z" / "cards" / "card-0000ffff").mkdir()
    with pytest.raises(ValueError, match="differ from the ledger"):
        backfill(runs, OutcomeStore(tmp_path / "store"))
