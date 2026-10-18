"""S6 labels follow from the claims ledger (review fix 2026-10-16).

Before the fix S6 recorded pass whatever its claims said, and the outcome record called the card finished. Card
card-779795a5 of run-20261011T042400Z passed S6 with 0 supported, 2 refuted, 1 inconclusive and 8 untested claims,
and card-03fa1d38 of the same run passed with its falsifier 4 refuted by the analyst; both ledgers are copied verbatim
into fixtures/s6_labels_20261011 (sha256 e4bd1060... and 9ba6ba77...).
"""
import asyncio
import json
from pathlib import Path

import pytest

from hybrid.research import graveyard
from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.stages.s6_verdict import ledger_verdict
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.outcome_fixtures import build_run
from hybrid.tests.research.test_orchestrator_fakes import analyst_supports_every_claim, make_orchestrator, orch_config

FIXTURES = Path(__file__).parent / "fixtures" / "s6_labels_20261011"
PASS = ["ok"]
EARLIER = [(stage, "pass", PASS) for stage in ("S0", "S1", "S2", "S3", "S4", "S5")]


def recorded_ledger(card_id: str) -> dict:
    return json.loads((FIXTURES / f"{card_id}-claims_ledger.json").read_text(encoding="utf-8"))


def ledger(*statuses: str) -> dict:
    return {"card_id": "card-00000001", "claims": [
        {"claim_id": f"capability.{i}", "claim": "c", "status": status, "decided_by": "analyst", "evidence": "e"}
        for i, status in enumerate(statuses, 1)]}


def test_the_verdict_is_pass_only_when_every_claim_is_supported():
    assert ledger_verdict(ledger("supported", "supported")) == ("pass", "S6 pass: all 2 claims supported")
    verdict, line = ledger_verdict(ledger("supported", "refuted", "untested"))
    assert verdict == "fail" and line == ("S6 fail: 1 of 3 claims refuted (capability.2); 1 not shown "
                                          "(capability.3)")
    assert ledger_verdict(ledger("supported", "inconclusive"))[0] == "undetermined"
    assert ledger_verdict(ledger("untested", "untested"))[0] == "undetermined"
    assert ledger_verdict(ledger()) == ("undetermined", "S6 undetermined: the claims ledger has no claims")
    with pytest.raises(ValueError, match="not in"):
        ledger_verdict(ledger("supported", "maybe"))


def test_the_recorded_ledgers_do_not_pass():
    assert ledger_verdict(recorded_ledger("card-779795a5"))[0] == "fail"
    assert ledger_verdict(recorded_ledger("card-03fa1d38"))[0] == "fail"


@pytest.mark.parametrize("card_id, codes", [("card-779795a5", ["predictions_missed"]),
                                            ("card-03fa1d38", ["falsified:falsifier.4"])])
def test_a_recorded_s6_pass_with_refuted_claims_is_a_death(tmp_path, card_id, codes):
    """The S6 pass the old rule recorded is read through the card's ledger: the record is dead, never finished, and
    the graveyard says it died."""
    claims = recorded_ledger(card_id)
    run_dir = build_run(tmp_path / "run-20261011T042400Z", [
        (make_card(card_id), EARLIER + [("S6", "pass", [f"claims ledger: {claims['counts']}"])],
         {"claims_ledger": claims})])
    record = build_record(RunView(run_dir), card_id, "backfill")
    assert (record["outcome"], record["terminal_stage"], record["kill_codes"]) == ("dead", "S6", codes)
    assert record["stage_verdicts"]["S6"]["verdict"] == "pass"          # the stored history is kept as recorded
    (grave,) = graveyard.graves(run_dir)
    assert grave.kind == "died" and grave.outcome.startswith("died at S6 (fail): S6 fail: ")


def test_a_fixed_rule_verdict_that_disagrees_with_its_ledger_is_refused(tmp_path):
    run_dir = build_run(tmp_path / "run-20261016T000000Z", [
        (make_card("card-0000aa01"), EARLIER + [("S6", "undetermined", ["claims ledger"])],
         {"claims_ledger": dict(ledger("supported"), card_id="card-0000aa01")})])
    with pytest.raises(ValueError, match="disagrees with its claims ledger"):
        build_record(RunView(run_dir), "card-0000aa01", "backfill")


def test_a_graveyard_s6_card_needs_its_claims_ledger(tmp_path):
    run_dir = build_run(tmp_path / "run-20261016T000001Z", [
        (make_card("card-0000aa02"), EARLIER + [("S6", "pass", ["claims ledger"])], {})])
    with pytest.raises(ValueError, match="no claims_ledger artifact"):
        graveyard.graves(run_dir)


# ----- through the stage ----------------------------------------------------------------------------------------

def s6_of(orch) -> tuple[str, dict, list[str]]:
    (card_id,) = orch.store.list_cards()
    entry = orch.store.get_card(card_id)["status"]["verdicts"]["S6"]
    return card_id, entry, entry["reasons"]


def test_a_refuted_claim_fails_s6_and_the_record_is_dead(tmp_path):
    script = {"analyst": {"claims": [{"claim_id": "capability.1", "claim": "c", "status": "refuted",
                                      "evidence": "kl 2.0 vs 0.1", "falsifier_tasks": []}]}}
    orch, _ = make_orchestrator(tmp_path / "run-20261016T000002Z", script=script,
                                cfg=orch_config(s5__falsifiers=False))
    asyncio.run(orch.run())
    card_id, entry, reasons = s6_of(orch)
    assert entry["verdict"] == "fail"
    assert any(r.startswith("S6 fail: 1 of 9 claims refuted (capability.1); 8 not shown") for r in reasons)
    record = build_record(RunView(orch.run_dir), card_id, "orchestrator")
    assert (record["outcome"], record["kill_codes"]) == ("dead", ["predictions_missed"])


def test_unshown_claims_leave_s6_undetermined_and_the_record_is_dead(tmp_path):
    """The default analyst comments on capability.1 alone: the other claims are untested, so S6 does not pass."""
    orch, _ = make_orchestrator(tmp_path / "run-20261016T000003Z", cfg=orch_config(s5__falsifiers=False))
    asyncio.run(orch.run())
    card_id, entry, reasons = s6_of(orch)
    assert entry["verdict"] == "undetermined"
    assert any(r.startswith("S6 undetermined: no claim refuted, but 8 of 9 claims not shown") for r in reasons)
    record = build_record(RunView(orch.run_dir), card_id, "orchestrator")
    assert (record["outcome"], record["kill_codes"]) == ("dead", ["undetermined:S6"])


def test_every_claim_supported_passes_s6_and_the_card_finishes(tmp_path):
    orch, _ = make_orchestrator(tmp_path / "run-20261016T000004Z", script={"analyst": analyst_supports_every_claim},
                                cfg=orch_config(s5__falsifiers=False))
    asyncio.run(orch.run())
    card_id, entry, reasons = s6_of(orch)
    assert entry["verdict"] == "pass" and "S6 pass: all 9 claims supported" in reasons
    record = build_record(RunView(orch.run_dir), card_id, "orchestrator")
    assert (record["outcome"], record["kill_codes"]) == ("finished", [])
