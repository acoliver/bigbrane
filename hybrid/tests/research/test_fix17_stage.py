"""FIX-17 at stages M and C (DECISIONS.md C17, P3 and XC).

A. The coupled measure in stage M: recorded in every verdict, a gate (obligation_failed:target_coupled) only when the
   target gate applies; calibration records it and never scores it.
B. The challenger reviews the card's resources: a confirmed ``resource_contradiction`` (settled by code over the
   frozen resources, never by a citation) fails the claims it names with kill code resource_contradiction.
C1. A P3 limit must be a mechanism limit: a limit that follows only from the output vocabulary or the task encoding is
   refused at freeze.
C2. A matcher/cross-check dispute goes to the statement tiebreak (a third family): upheld blocks, rejected clears,
   and a failed call leaves the claim unresolved and not shown.
"""
from __future__ import annotations

import copy

import pytest

from hybrid.llm import load_config
from hybrid.research import calibration, families
from hybrid.research.calibration import load_cards
from hybrid.research.schemas import validate
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages import match_adjudication as ma
from hybrid.research.stages import math_contracts as mc
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research import test_fix7_match as fix7
from hybrid.tests.research.test_profile_stage import Harness

TARGET_RULE = {"max_cost_exponent": 1.5, "state_growth": "log", "coupled_max_cost_exponent": 1.5,
               "coupled_max_state_exponent": 0.5}


# ----- A. the coupled measure at stage M ------------------------------------------------------------------------------

def test_with_the_target_gate_on_the_transformer_fails_the_coupled_gate_too(tmp_path):
    h = Harness(tmp_path, "k1", target=True)
    assert h.m() is None
    verdict = h.verdict("M")
    block = verdict["target_coupled"]
    assert block["applied"] and block["status"] == "fail" and block["kill_code"] == "obligation_failed:target_coupled"
    assert {slot: entry["growth"] for slot, entry in block["slots"].items()} == {
        "P2.copy": "none", "P2.recall": "none", "P2.nesting": "uncovered", "P2.composition": "none"}
    assert block["rule"] == {"max_cost_exponent": 1.5, "max_state_exponent": 0.5}
    assert {"obligation_failed:target", "obligation_failed:target_coupled"} <= set(verdict["kill_codes"])
    reasons = h.store.get_card(h.card["id"])["status"]["verdicts"]["M"]["reasons"]
    assert any(r.startswith("coupled target gate: no P2 task's skill grows as a power of n") for r in reasons)
    validate("math_verdict", verdict)


def test_the_lstm_passes_the_coupled_gate_with_recall_at_k_n_to_the_quarter(tmp_path):
    h = Harness(tmp_path, "k2", target=True)
    h.m()
    verdict = h.verdict("M")
    block = verdict["target_coupled"]
    assert block["status"] == "pass" and block["applied"]
    assert block["slots"]["P2.recall"]["scale"] == "n^(1/4)"
    assert block["tasks"]["recall"]["b_cost"] == "1/4" and block["tasks"]["recall"]["b_state"] == "1/2"
    assert "obligation_failed:target_coupled" not in verdict["kill_codes"]


def test_with_the_target_gate_off_the_coupled_measure_is_recorded_and_decides_nothing(tmp_path):
    h = Harness(tmp_path, "k3", target=False)
    h.m()
    verdict = h.verdict("M")
    assert verdict["target_coupled"]["status"] == "fail" and verdict["target_coupled"]["applied"] is False
    assert all(entry["growth"] == "bounded" for entry in verdict["target_coupled"]["slots"].values())
    assert "obligation_failed:target_coupled" not in verdict["kill_codes"]
    # Calibration records the block and its summary, and the cells come from the slots alone.
    outcome = calibration.outcome_record(h.ctx, "k3", h.card, 1, TARGET_RULE)
    assert outcome["target_coupled"]["status"] == "fail"
    assert m_math.coupled_summary(outcome["target_coupled"]) == {
        "status": "fail", "applied": False, "slots": {slot: "0 (bounded)" for slot in m_math.coupled.P2_SLOTS}}


def test_without_a_verified_p1_the_coupled_measure_does_not_apply():
    block = m_math.coupled_block({"target": dict(TARGET_RULE, enabled=True)}, {"obligations": []}, {})
    assert block["status"] == "not_applicable" and block["kill_code"] is None and block["applied"]
    assert block["reasons"] == ["P1 did not hold: no verified formula"]


# ----- B. resource contradictions -------------------------------------------------------------------------------------

GROWING_WIDTH = ('def verify(witness):\n'
                 '    d = witness.resources["tasks"]["recall"]["d"]\n'
                 '    small, large = witness.evaluate(d, {"K": 8}), witness.evaluate(d, {"K": 64})\n'
                 '    law = witness.conservation.associative_recall_check(32 * 2 * small, 64, 16, 1.0)\n'
                 '    return (large > small and law["status"] in ("discharged", "violated"),\n'
                 '            f"recall width {small} at K = 8 and {large} at K = 64")\n')
SAME_WIDTH = ('def verify(witness):\n'
              '    d = witness.resources["tasks"]["recall"]["d"]\n'
              '    return (witness.evaluate(d, {"K": 8}) == witness.evaluate(d, {"K": 64}), "the width is fixed")\n')
RECALL_QUOTE = {"source": "resources", "ref": None, "quote": '"d": "4*K+8"'}


def resource_finding(subject: str, used_by: list[str], **changes) -> dict:
    return mf.finding(subject, kind="resource_contradiction", evidence=dict(RECALL_QUOTE), used_by=used_by,
                      claim="the card states a fixed store while the recall width grows with K", **changes)


def run_c_with(tmp_path, findings_for) -> Harness:
    h = Harness(tmp_path, "k2")
    h.m()
    recall = [o["id"] for o in h.frozen()["obligations"] if o["claim_id"] == "P2.recall"]
    h.llm.script["challenger"] = mf.challenger_output(findings=findings_for(recall))
    h.c()
    return h


def test_a_confirmed_resource_contradiction_fails_the_claim_it_names(tmp_path):
    h = run_c_with(tmp_path, lambda recall: [resource_finding("fixed store, growing width", recall,
                                                               code=GROWING_WIDTH)])
    verdict = h.verdict("C")
    finding = next(f for f in verdict["findings"] if f["subject"] == "fixed store, growing width")
    assert finding["disposition"] == "violated" and finding["resource_contradiction"] is True
    assert "resource_contradiction" in verdict["kill_codes"] and verdict["verdict"] == "fail"
    assert h.slot("C", "P2.recall")["status"] == "failed"
    assert "resource contradiction" in h.slot("C", "P2.recall")["detail"]
    challenger = next(c for c in h.llm.calls if c["prompt"] == "challenger")
    assert "CARD RESOURCES" in challenger["user"] and '"4*K+8"' in challenger["user"]
    validate("math_verdict", verdict)


def test_a_resource_contradiction_the_code_does_not_confirm_or_that_cites_a_paper_blocks_nothing(tmp_path):
    h = run_c_with(tmp_path, lambda recall: [
        resource_finding("width said to be fixed", recall, code=SAME_WIDTH),
        resource_finding("a paper says so", recall, testable_by="citation", code=None)])
    verdict = h.verdict("C")
    dispositions = {f["subject"]: f["disposition"] for f in verdict["findings"]}
    assert dispositions == {"width said to be fixed": "rejected", "a paper says so": "unsupported"}
    assert "resource_contradiction" not in verdict["kill_codes"]
    assert h.slot("C", "P2.recall")["status"] != "failed"


# ----- C1. a P3 limit is about the mechanism -------------------------------------------------------------------------

def limits_k1(basis: dict) -> list[dict]:
    return [{"task": "dyck", "size": 8, "resources": "a fixed output alphabet", "why": "the readout has V symbols",
             "basis": basis}]


@pytest.mark.parametrize("basis, message", [
    ({"part": "output_vocabulary", "anchor": "mechanism.readout.readout", "quote": "over the V symbols"},
     "follows from the output vocabulary, not from the mechanism"),
    ({"part": "task_encoding", "anchor": "mechanism.update.embed", "quote": "a symbol table row"},
     "follows from the task encoding, not from the mechanism"),
    ({"part": "state", "anchor": "mechanism.state.Q", "quote": "anything"}, "is not one of the card's"),
    ({"part": "readout", "anchor": "mechanism.update.write", "quote": "appended as new rows of A_l and C_l"},
     "is part of the update, not the readout"),
    ({"part": "update", "anchor": "mechanism.update.write", "quote": "a sentence the card never wrote"},
     "basis quote is not text of mechanism.update.write"),
])
def test_limit_problems_refuse_a_limit_not_about_the_mechanism(basis, message):
    card = load_cards(("k1",))["k1"]
    problems = mc.limit_problems(limits_k1(basis), card)
    assert len(problems) == 1 and problems[0][0] == 0 and message in problems[0][1]


def test_a_mechanism_limit_with_its_quote_is_accepted():
    card = load_cards(("k1",))["k1"]
    assert mc.limit_problems(cf.LIMITS["k1"], card) == []
    for key in ("k2", "k3", "k4"):
        assert mc.limit_problems(cf.LIMITS[key], load_cards((key,))[key]) == [], key


def test_a_p3_resting_only_on_the_output_vocabulary_is_refused_at_freeze(tmp_path):
    card = load_cards(("k1",))["k1"]
    answer = cf.fixer_output("k1", card)
    answer["limits"] = limits_k1({"part": "output_vocabulary", "anchor": "mechanism.readout.readout",
                                  "quote": "over the V symbols"})
    h = Harness(tmp_path, "k1", fixer=answer)
    h.m()
    retry = [c["user"] for c in h.llm.calls if c["prompt"] == "statement_fixer"]
    assert len(retry) == 2 and "follows from the output vocabulary" in retry[1]
    record = h.frozen()
    assert record["profile"]["limits"] == []
    assert not any(o["claim_id"] == "P3" for o in record["obligations"])
    assert any(n["claim_id"] == "P3" and "DECISIONS.md P3" in n["why"] for n in record["not_formalized"])
    assert any(d.get("limit") for d in record["dropped"])
    assert h.slot("M", "P3")["status"] == "not_formalized" and "unproven:P3" in h.verdict("M")["kill_codes"]
    validate("math_statement", record)


def test_a_refused_limit_is_dropped_and_a_mechanism_limit_beside_it_is_kept(tmp_path):
    card = load_cards(("k1",))["k1"]
    answer = cf.fixer_output("k1", card)
    answer["limits"] = cf.LIMITS["k1"] + limits_k1({"part": "task_encoding", "anchor": "mechanism.update.embed",
                                                    "quote": "a symbol table row"})
    h = Harness(tmp_path, "k1", fixer=answer)
    h.m()
    record = h.frozen()
    assert record["profile"]["limits"] == cf.LIMITS["k1"]
    assert any(o["claim_id"] == "P3" for o in record["obligations"])


# ----- C2. the statement tiebreak -------------------------------------------------------------------------------------

class TiebreakReplay(fix7.Replay):
    """The FIX-7 replay of k3-run1 with the matcher's anchored D1 objection, a cross-check that rejects it, and the
    statement tiebreak as each test scripts (None: the call fails)."""

    def __init__(self, tmp_path, tiebreak):
        super().__init__(tmp_path, fix7.with_d1_objection(fix7.capacity_objection()),
                         crosscheck=mf.crosscheck_output(default="rejected"))
        if tiebreak is not None:
            self.llm.script["statement_tiebreak"] = tiebreak
        self.llm.models["statement_tiebreak"] = "chatgpt/gpt-6.1-sol"


def test_a_tiebreak_that_upholds_the_objection_settles_the_dispute_as_a_mismatch(tmp_path):
    replay = TiebreakReplay(tmp_path, mf.crosscheck_output(default="upheld"))
    replay.run()
    call = next(c for c in replay.llm.calls if c["prompt"] == "statement_tiebreak")
    assert call["role"] == "statement_tiebreak" and "THE TWO READINGS THAT DISAGREE" in call["user"]
    challenge = replay.challenge()
    assert challenge["tiebreak"]["model"] == "chatgpt/gpt-6.1-sol" and challenge["tiebreak"]["failed"] is None
    assert challenge["disputes"] == [dict(challenge["disputes"][0], objection_id="D1#1", matcher="upheld",
                                          crosscheck="rejected", tiebreak="upheld", outcome="mismatch")]
    verdict = replay.verdict()
    assert verdict["models"]["statement_tiebreak"] == ["chatgpt/gpt-6.1-sol"]
    assert not [i for i in verdict["statement_match"] if i["match"] == "unresolved"]
    assert "statement_mismatch" in verdict["kill_codes"]
    assert not any(code.startswith("match_unresolved:") for code in verdict["kill_codes"])
    validate("math_verdict", verdict)


def test_a_tiebreak_that_rejects_the_objection_lets_it_block_nothing(tmp_path):
    replay = TiebreakReplay(tmp_path, mf.crosscheck_output(default="rejected"))
    replay.run()
    challenge = replay.challenge()
    assert [d["outcome"] for d in challenge["disputes"]] == ["matched"]
    verdict = replay.verdict()
    assert not [i for i in verdict["statement_match"] if i["match"] == "unresolved"]
    assert "statement_mismatch" not in verdict["kill_codes"]
    assert not any(code.startswith("match_unresolved:") for code in verdict["kill_codes"])
    assert replay.statuses() == fix7.M_STATUSES


def test_a_failed_tiebreak_call_leaves_the_claim_unresolved_and_not_shown(tmp_path):
    replay = TiebreakReplay(tmp_path, None)   # the fakes' default: the tiebreak call raises TransportError
    replay.run()
    challenge = replay.challenge()
    assert challenge["tiebreak"]["failed"].startswith("the statement_tiebreak call failed")
    assert [(d["tiebreak"], d["outcome"]) for d in challenge["disputes"]] == [(None, "unresolved")]
    verdict = replay.verdict()
    unresolved = sorted(i["statement_id"] for i in verdict["statement_match"] if i["match"] == "unresolved")
    assert unresolved == fix7.D1_USERS
    assert "match_unresolved:P2.copy" in verdict["kill_codes"]
    assert "tiebreak not settled" in replay.claims()["P2.copy"]["match_detail"]
    outcome = calibration.outcome_record(replay.ctx, "k3", fix7.CARD, 1, TARGET_RULE)
    assert outcome["unresolved"] == list(calibration.CELLS)
    assert set(outcome["cells"].values()) == {"not_shown"}


def test_tiebreak_unit_rules():
    disputes = [{"objection_id": "D1#1", "why": "capacity", "crosscheck": "rejected", "crosscheck_why": "restates"}]
    upheld = ma.tiebreak(disputes, [{"objection_id": "D1#1", "verdict": "upheld", "why": "w"}])
    assert upheld["D1#1"]["adjudication"] == ma.CONFIRMED
    rejected = ma.tiebreak(disputes, [{"objection_id": "D1#1", "verdict": "rejected", "why": "w"}])
    assert rejected["D1#1"]["adjudication"] == ma.REJECTED
    with pytest.raises(ValueError, match="no verdict for D1#1"):
        ma.tiebreak(disputes, [])
    failed = ma.tiebreak_failed(disputes, "transport")
    assert failed["D1#1"] == {"adjudication": ma.DISPUTED, "tiebreak": None, "tiebreak_why": "transport"}
    record = ma.dispute_record(dict(disputes[0], **failed["D1#1"]))
    assert record["outcome"] == "unresolved" and record["matcher"] == "upheld"


def test_the_tiebreak_is_a_third_family_in_every_routing_table():
    for table in ("llxprt", "gateway", "gateway_claude_local"):
        config = load_config(fix7.MODELS_TOML, routing=table)
        primaries = {role: [config.roles[role][0].model] for role in
                     ("statement_fixer", "prover", "statement_match", "statement_crosscheck", "statement_tiebreak")}
        assert families.violations(primaries) == [], table
        assert len(config.roles["statement_tiebreak"]) == 1, table
        assert config.settings_for("statement_tiebreak").effort == "medium"
    gateway = load_config(fix7.MODELS_TOML, routing="gateway")
    assert gateway.roles["statement_tiebreak"][0].model == "chatgpt/gpt-6.1-sol"
    assert ("statement_tiebreak", "statement_match") in families.SEPARATED
    assert ("statement_tiebreak", "statement_crosscheck") in families.SEPARATED
