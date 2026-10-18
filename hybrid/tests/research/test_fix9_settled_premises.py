"""FIX-9 regressions for stage C (results/phase4-20261013T073748Z/FIX-9.md), from the recorded calibration line
runs/calibration-math-20261014T121552Z/k1-run2 (sanitized extract: fixtures/fix9_settled_premises/k1_run2_c.json,
written by results/phase4-20261013T073748Z/fix9-logs/make_fix9_fixture.py).

The statement fixer's conformance premise A1 ("the counted reference implements the card's equations with the product
spelling, parameter names and retained state specified in D1 and D2") was settled at stage M: the conformance
statement O2.a, which discharges it, held. Stage C still treated it as an open assumption: the challenger gave no
runnable magic-X for it (``magic_x_missing:A1``, in every finished run of the calibration), and the statement-match
role read it as restating P1, the claim it supports, which made it circular and scored P1 not shown
(``circular:A1``, k1-run2 only). A premise a discharging check settled is evidence, not an assumption: it is exempt
from both, and the report says "settled at M by <statement id>". A premise that is open, and one whose discharging
check failed, keep every check.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hybrid.research import mathrecords
from hybrid.research.calibration import load_cards
from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.stages import c_challenge, m_math
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_fix6_settlement import FAKE_K1_EQUATIONS
from hybrid.tests.research.test_profile_stage import Harness

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "fix9_settled_premises" / "k1_run2_c.json")
                     .read_text(encoding="utf-8"))
RECORD = FIXTURE["frozen_record"]
RECORDED = FIXTURE["recorded_c"]
FINDING = next(f for f in FIXTURE["match_premises"] if f["premise_id"] == "A1")
SETTLED = "settled at M by O2.a"


def recorded_checks() -> dict[str, dict]:
    return copy.deepcopy(FIXTURE["checks"])


def circular_finding() -> dict[str, str]:
    """What stage C passes to the ledger for the recorded statement-match reading of A1."""
    supports = c_challenge.supported_claims(RECORD, "A1", [])
    hit = sorted(set(FINDING["implies"]) & supports)
    assert hit == ["P1"]
    return {"A1": f"restates or implies {', '.join(hit)}, which it supports: {FINDING['why']}"[:1000]}


def stage_c(record: dict = RECORD, checks: dict | None = None, *, mismatched: dict | None = None,
            magic_items: list[dict] | None = None, magic_failures: list[str] = ()):
    """Stage C's ledger, records and gate on the recorded k1-run2 inputs, with the recorded magic-X item for A1."""
    checks = recorded_checks() if checks is None else checks
    blocked = {}
    for sid in mismatched or {}:
        claim_id = m_math.obligation_of(record, sid)["claim_id"]
        if claim_id:
            blocked[claim_id] = f"statement {sid} does not match the claim"
    _, ledger, records = m_math.evaluate(record, checks, circular=circular_finding(), citations_checked=True,
                                         mismatched=mismatched, blocked=blocked, unresolved_claims={})
    items = copy.deepcopy(RECORDED["magic_x"] if magic_items is None else magic_items)
    circularity = copy.deepcopy(RECORDED["circularity"])
    settled = mathrecords.settled_at_m(ledger)
    missing = c_challenge.settle_magic_x(items, settled)
    changed = c_challenge.settle_circularity(circularity, settled)
    gate = mathrecords.challenge_gate(records, ledger, conservation=[], magic_x=list(magic_failures),
                                      magic_x_missing=missing, counterexamples=[],
                                      mismatched=sorted(mismatched or {}))
    return {"ledger": {e["id"]: e for e in ledger}, "records": {r["claim_id"]: r for r in records}, "gate": gate,
            "magic_x": items, "circularity": circularity, "changed": changed, "settled": settled}


# ----- the recorded run --------------------------------------------------------------------------------------------

def test_the_recorded_run_treated_the_measured_a1_as_an_open_assumption():
    a1 = next(p for p in RECORD["premises"] if p["id"] == "A1")
    assert a1["settle_at"] == "M" and a1["subject"] == "frozen_definitions" and a1["does_work"]
    discharger = next(o for o in RECORD["obligations"] if "A1" in o["discharges"])
    assert discharger["id"] == "O2" and discharger["claim_id"] is None
    assert FIXTURE["checks"]["O2.a"]["form"] == "conformance"
    assert (FIXTURE["checks"]["O2.a"]["status"], FIXTURE["checks"]["O2.a"]["decided"]) == ("discharged", True)
    assert RECORDED["kill_codes"] == ["circular:A1", "magic_x_missing:A1", "obligation_failed:P4.composition"]
    assert RECORDED["claims"]["P1"]["status"] == "circular" and RECORDED["claims"]["P1"]["circular_premises"] == ["A1"]
    assert RECORDED["magic_x"][0]["missing"] and RECORDED["magic_x"][0]["required"]
    assert [e["claim"] for e in FIXTURE["challenger_magic_x"]] == [None, None]
    assert RECORDED["circularity"][0]["circular"] is True


def test_the_reconstruction_of_the_recorded_inputs_reproduces_the_recorded_ledger_entry():
    """Without the exemption (the discharging check removed from what settles A1), the inputs give what was recorded,
    so the tests below change only the rule."""
    record = copy.deepcopy(RECORD)
    record["obligations"] = [dict(o, discharges=[]) for o in record["obligations"]]
    out = stage_c(record)
    assert out["ledger"]["A1"]["status"] == "circular"
    assert out["ledger"]["A1"]["evidence"] == next(e for e in RECORDED["ledger"] if e["id"] == "A1")["evidence"]
    assert out["gate"] == (RECORDED["verdict"], RECORDED["kill_codes"])


# ----- a measured premise is evidence --------------------------------------------------------------------------------

def test_a_measured_a1_is_neither_circular_nor_missing_magic_x_and_p1_keeps_its_evidence():
    out = stage_c()
    a1 = out["ledger"]["A1"]
    assert a1["status"] == "measured" and a1["settled"] == SETTLED
    assert a1["evidence"].startswith("O2.a (executed-on-examples), math_check artifact ")
    assert "is not a circularity" in a1["basis"] and FINDING["why"] in a1["basis"]
    validate("assumption", a1)
    p1 = out["records"]["P1"]
    assert (p1["status"], p1["tier"], p1["circular_premises"], p1["open_premises"]) == ("observed", "executed", [], [])
    assert p1["phrase"].startswith("observed on the listed sizes")
    assert out["gate"] == ("fail", ["obligation_failed:P4.composition"])
    item = out["magic_x"][0]
    assert (item["premise_id"], item["settled"], item["missing"], item["required"]) == ("A1", SETTLED, False, False)
    assert item["evidence"] == RECORDED["magic_x"][0]["evidence"]          # what the challenger said is kept
    finding = out["circularity"][0]
    assert (finding["circular"], finding["settled"], finding["implies"]) == (False, SETTLED, ["P1"])
    assert finding["why"] == FINDING["why"] and out["changed"] == ["A1"]


def test_the_settled_premise_is_reported_with_its_check_and_nothing_else_changes():
    settled, before = stage_c(), stage_c(magic_items=[])
    assert settled["settled"] == {"A1": SETTLED}
    assert before["gate"] == settled["gate"]                                # no magic-X item: nothing was required
    for cid, record in settled["records"].items():
        if cid != "P1":
            assert record["status"] == RECORDED["claims"][cid]["status"], cid


def test_a_settled_premise_whose_magic_x_reached_the_claim_still_fails_the_card():
    item = dict(RECORDED["magic_x"][0], ran=True, status="violated", missing=False, evidence="baseline + X reached it")
    out = stage_c(magic_items=[item], magic_failures=["A1"])
    assert out["magic_x"][0]["settled"] == SETTLED and out["magic_x"][0]["status"] == "violated"
    assert out["gate"] == ("fail", ["magic_x:A1", "obligation_failed:P4.composition"])


# ----- an unmeasured premise keeps every check -------------------------------------------------------------------

def never_checked() -> tuple[dict, dict]:
    checks = recorded_checks()
    del checks["O2.a"]
    return RECORD, checks


def no_discharger() -> tuple[dict, dict]:
    record = copy.deepcopy(RECORD)
    record["obligations"] = [o for o in record["obligations"] if o["id"] != "O2"]
    a1 = next(p for p in record["premises"] if p["id"] == "A1")
    a1.update(subject="mechanism", settle_at="deferrable")
    checks = recorded_checks()
    del checks["O2.a"]
    return record, checks


@pytest.mark.parametrize("case", [never_checked, no_discharger], ids=["discharger never checked", "no discharger"])
def test_an_unmeasured_premise_is_still_flagged_by_both_checks(case):
    record, checks = case()
    out = stage_c(record, checks)
    a1 = out["ledger"]["A1"]
    assert a1["status"] == "circular" and "settled" not in a1 and out["settled"] == {}
    validate("assumption", a1)
    assert out["records"]["P1"]["status"] == "circular" and out["records"]["P1"]["circular_premises"] == ["A1"]
    assert out["gate"] == ("fail", ["circular:A1", "magic_x_missing:A1", "obligation_failed:P4.composition"])
    item = out["magic_x"][0]
    assert item["missing"] and item["required"] and "settled" not in item
    assert out["circularity"][0]["circular"] is True and "settled" not in out["circularity"][0]
    assert out["changed"] == []


def test_a_discharging_statement_mismatched_at_stage_c_settles_nothing():
    out = stage_c(mismatched={"O2.a": "the equations differ from the card's update"})
    assert out["ledger"]["A1"]["status"] == "circular" and out["settled"] == {}
    kill = out["gate"][1]
    assert {"circular:A1", "magic_x_missing:A1", "statement_mismatch"} <= set(kill)


# ----- a failed conformance still fails --------------------------------------------------------------------------

def failed_conformance() -> dict:
    checks = recorded_checks()
    checks["O2.a"].update(status="violated", decided=True,
                          evidence="retained array A1 differs from the frozen equations after token 3")
    return checks


def test_a_failed_conformance_still_fails():
    checks = failed_conformance()
    out = stage_c(checks=checks)
    a1 = out["ledger"]["A1"]
    assert a1["status"] == "circular" and "settled" not in a1
    assert out["gate"] == ("fail", ["circular:A1", "magic_x_missing:A1", "obligation_failed:P4.composition"])
    assert out["magic_x"][0]["missing"] and out["circularity"][0]["circular"]
    # Without the statement-match finding, the existing failure path: A1 stays open, cannot advance, is never
    # deferred to test design, P1 stays conditional on it, and the missing magic-X still fails stage C.
    outcomes, ledger, records = m_math.evaluate(RECORD, checks)
    entry = next(e for e in ledger if e["id"] == "A1")
    assert entry["status"] == "open" and "did not hold" in entry["basis"] and "settled" not in entry
    assert not mathrecords.can_advance(entry, stage="C") and not mathrecords.schedulable(entry)
    assert outcomes["O2"]["result"] == "failed"
    p1 = next(r for r in records if r["claim_id"] == "P1")
    assert p1["status"] == "conditional" and p1["open_premises"] == ["A1"]
    assert mathrecords.scheduled_measurements(records, ledger) == []
    missing = c_challenge.settle_magic_x(copy.deepcopy(RECORDED["magic_x"]), mathrecords.settled_at_m(ledger))
    verdict, codes = mathrecords.challenge_gate(records, ledger, conservation=[], magic_x=[],
                                                magic_x_missing=missing, counterexamples=[])
    assert verdict == "fail" and "magic_x_missing:A1" in codes


def test_the_ledger_schema_allows_settled_only_on_a_settled_premise():
    entry = dict(stage_c()["ledger"]["A1"])
    validate("assumption", entry)
    with pytest.raises(SchemaViolation):
        validate("assumption", dict(entry, status="open", evidence=None))
    with pytest.raises(SchemaViolation):
        validate("assumption", dict(entry, settled="settled by a model"))


# ----- stage C end to end (scripted LLMs, the k1 calibration card) -----------------------------------------------

def k1_fixer(card: dict, *, equations: str = FAKE_K1_EQUATIONS) -> dict:
    """The k1 calibration fixer answer with the k1-run2 shape: A1 about the frozen definitions, discharged by a
    conformance statement, and used by P1's cost obligation; and A2, a mechanism premise with no discharger, also
    used by P1."""
    answer = cf.fixer_output("k1", card)
    answer["definitions"] = [{"id": "D1", "kind": "accounting", "text": "Parameters are E, Wq, Wk, Wv and O; the "
                              "state retains the key and value rows and the last query."}]
    answer["premises"] = [
        mf.premise("A1", "The counted reference implements the card's equations with the parameter names and "
                   "retained state specified in D1.", "other", subject="frozen_definitions"),
        mf.premise("A2", "Each retrieval returns the stored row the query most resembles.", "store")]
    next(o for o in answer["obligations"] if o["claim_id"] == "P1")["premises"] = ["A1", "A2"]
    # FIX-20: the points, the tolerance and the precision are host values; the fixer leaves them out.
    answer["obligations"].append(mf.obligation("O12", None, "premise A1", [{"form": "conformance", "conformance": {
        "reference": "O1.a", "equations": equations, "streams": ["repeated", "distinct", "random"]}}],
        discharges=["A1"], definitions=["D1"]))
    return answer


def run_k1(tmp_path, *, equations: str = FAKE_K1_EQUATIONS) -> Harness:
    card = load_cards(("k1",))["k1"]
    script = cf.script("k1", card, fixer=k1_fixer(card, equations=equations))
    # The recorded challenger's magic-X reading of A1 (no construction task fits a conformance premise), and no
    # magic-X for A2; the recorded statement-match reading that A1 implies P1, and the same reading of A2.
    script["challenger"] = mf.challenger_output(magic_x=[copy.deepcopy(FIXTURE["challenger_magic_x"][0])])
    script["statement_match"] = mf.match_output(implies={"A1": ["P1"], "A2": ["P1"]})
    # Without the card's P4 protocols: with them K1 fails M on the composition learner (decision P4K) and the failed
    # check takes precedence over the unproven codes these tests read; they are about the settlement of A1 and A2.
    h = Harness(tmp_path, script=script, protocols=False)
    h.m()
    h.c()
    return h


def c_ledger(h: Harness) -> dict[str, dict]:
    verdict = h.verdict("C")
    data = json.loads((h.run_dir / "artifacts" / "assumption_ledger" / verdict["assumption_ledger"]).read_text())
    return {p["id"]: p for p in data["premises"]}


def test_stage_c_exempts_the_measured_premise_and_keeps_both_checks_on_the_open_one(tmp_path):
    h = run_k1(tmp_path)
    ledger = c_ledger(h)
    assert ledger["A1"]["status"] == "measured" and ledger["A1"]["settled"] == "settled at M by O12.a"
    assert ledger["A2"]["status"] == "circular" and "settled" not in ledger["A2"]
    verdict = h.verdict("C")
    codes = verdict["kill_codes"]
    assert "circular:A2" in codes and "magic_x_missing:A2" in codes
    assert "circular:A1" not in codes and "magic_x_missing:A1" not in codes
    magic = {m["premise_id"]: m for m in verdict["magic_x"]}
    assert magic["A1"]["settled"] == "settled at M by O12.a" and not magic["A1"]["missing"]
    assert not magic["A1"]["required"] and "no construction task fits" in magic["A1"]["evidence"]
    assert magic["A2"]["missing"] and magic["A2"]["required"] and "settled" not in magic["A2"]
    circularity = {c["premise_id"]: c for c in verdict["circularity"]}
    assert (circularity["A1"]["circular"], circularity["A1"]["settled"]) == (False, "settled at M by O12.a")
    assert circularity["A2"]["circular"] is True and "settled" not in circularity["A2"]
    assert h.slot("C", "P1")["circular_premises"] == ["A2"]
    events = [e["data"] for e in h.ledger.iter_events("challenge_check")
              if e["data"].get("premise_id") == "A1" and e["data"].get("settled")]
    assert {e["check"] for e in events} == {"magic_x", "circularity"}
    validate("math_verdict", verdict)


def test_stage_c_with_only_the_measured_premise_leaves_p1_observed(tmp_path):
    card = load_cards(("k1",))["k1"]
    fixer = k1_fixer(card)
    fixer["premises"] = fixer["premises"][:1]
    next(o for o in fixer["obligations"] if o["claim_id"] == "P1")["premises"] = ["A1"]
    script = cf.script("k1", card, fixer=fixer)
    script["challenger"] = mf.challenger_output(magic_x=[copy.deepcopy(FIXTURE["challenger_magic_x"][0])])
    script["statement_match"] = mf.match_output(implies={"A1": ["P1"]})
    h = Harness(tmp_path, script=script)
    h.m()
    h.c()
    p1 = h.slot("C", "P1")
    assert (p1["status"], p1["circular_premises"], p1["open_premises"]) == ("observed", [], [])
    assert not any(code.endswith(":A1") for code in h.verdict("C")["kill_codes"])


def test_stage_c_after_a_failed_audit_keeps_both_checks_on_a1(tmp_path):
    wrong = FAKE_K1_EQUATIONS.replace('q = x @ params["Wq"]', 'q = x @ params["Wk"]')
    h = run_k1(tmp_path, equations=wrong)
    audit = next(e["data"] for e in h.ledger.iter_events("math_check") if e["data"]["purpose"] == "conformance")
    assert audit["status"] == "violated"
    ledger = c_ledger(h)
    assert ledger["A1"]["status"] == "circular" and "settled" not in ledger["A1"]
    codes = h.verdict("C")["kill_codes"]
    assert {"circular:A1", "magic_x_missing:A1", "circular:A2", "magic_x_missing:A2"} <= set(codes)
    assert h.verdict("C")["verdict"] == "fail"
    assert h.slot("C", "P1")["status"] == "circular"
    assert "unproven:A1" in h.verdict("M")["kill_codes"]
