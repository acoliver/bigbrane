"""FIX-7 regressions: statement-match corrections (results/phase4-20261013T073748Z/match-diag/REPORT.md) and the
construction ranking against a trivial baseline (FIX-6.md, Limitations).

Recorded data:
- fixtures/calibration_math_20261014/k3_run1_match.json, a sanitized extract of
  runs/calibration-math-20261014T053341Z/k3-run1 (results/phase4-20261013T073748Z/fix7-logs/make_fix7_fixture.py):
  the frozen record, the stage M checks and claims, and the local matcher's recorded back-translation and comparison.
  The comparison called D1 (the card's sixteen states, update and readout) a substantive premise and read three P4
  size ceilings as ranges, which blocked sixteen statements and every scored cell.
- fixtures/calibration_smoke_20261013/k1_challenge.json (FIX-4): the K1 smoke's P3 objection, an existential limit
  read as a universal statement.
- fixtures/calibration_math_20261014/k1_run1.json (FIX-6): the k1-run1 composition and binding construction attempts.

Stage C runs for real on the recorded M results, with fakes for the model roles: the challenger finds nothing (the
recorded C ledger was empty and every recorded finding unsupported or out of scope), the matcher answers with the
recorded comparison in the FIX-7 schema, and the cross-check answers as each test scripts.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from hybrid.llm import load_config
from hybrid.math import constructions, profile
from hybrid.math.runner import MathRunner
from hybrid.research import calibration, families, mathrecords
from hybrid.research.calibration import load_cards, load_expected
from hybrid.research.ledger import Ledger
from hybrid.research.schemas import validate
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages import match_adjudication as ma
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages import prover_memory as pm
from hybrid.research.stages.common import Budget, StageContext, load_section
from hybrid.research.store import Store
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research import test_fix4_challenge as fix4
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, orch_config

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures"
K3 = json.loads((FIXTURES / "calibration_math_20261014" / "k3_run1_match.json").read_text(encoding="utf-8"))
K1_RUN1 = json.loads((FIXTURES / "calibration_math_20261014" / "k1_run1.json").read_text(encoding="utf-8"))
CARD = load_cards(("k3",))["k3"]
CARD_ID = CARD["id"]
MODELS_TOML = ROOT / "hybrid" / "config" / "models.toml"
STAGED_MODELS = ROOT / "results" / "phase4-20261013T073748Z" / "fix7-staging" / "models.toml"
P4_STATEMENTS = {"O13.a": "P4.copy", "O14.a": "P4.recall", "O15.a": "P4.nesting", "O16.a": "P4.composition"}
D1_USERS = sorted(s["id"] for o in K3["record"]["obligations"] if "D1" in o["definitions"] for s in o["statements"])
M_STATUSES = {c["claim_id"]: c["status"] for c in K3["m_claims"]}
D1_CLAUSE = "The model has k=16 latent states, one carried integer z"


def recorded(compare: dict | None = None) -> dict:
    """The recorded comparison as the matcher returned it (no objections: the old schema had none)."""
    reply = copy.deepcopy(compare or K3["replies"]["compare"])
    for entry in reply["statements"]:
        entry.setdefault("objections", [])
    for entry in reply["definitions"]:
        entry.setdefault("objection", None)
    return reply


def with_d1_objection(objection: dict) -> dict:
    reply = recorded()
    next(d for d in reply["definitions"] if d["definition_id"] == "D1")["objection"] = objection
    return reply


def with_statement_objection(sid: str, objection: dict) -> dict:
    reply = recorded()
    next(s for s in reply["statements"] if s["statement_id"] == sid)["objections"] = [objection]
    return reply


def capacity_objection() -> dict:
    """The recorded D1 verdict ("16 latent states ... are capacity constraints") stated as an anchored objection."""
    return {"definition_quote": D1_CLAUSE, "card_quote": None, "kind": "beyond_card", "asserts": "capacity",
            "difference": "it asserts the mechanism has exactly 16 latent states, a capacity constraint"}


class Replay:
    """Stage C on the recorded k3-run1 stage M: the frozen record and the M verdict stored as stage M stored them, a
    challenger that finds nothing, the recorded back-translation, the given comparison reply and cross-check."""

    def __init__(self, tmp_path, compare: dict, *, crosscheck=None, rec: dict | None = None):
        self.run_dir = (tmp_path / "run").resolve()
        self.ledger = Ledger(self.run_dir)
        self.store = Store(self.run_dir, self.ledger)
        self.store.put_card(copy.deepcopy(CARD))
        self.executor = mf.MathExecutor(tmp_path / "exec")
        script = {"challenger": mf.challenger_output(),
                  "statement_backtranslate": mf.recorded_backtranslation(K3["replies"]["backtranslate"]),
                  "statement_match": compare, "statement_crosscheck": crosscheck or mf.crosscheck_output()}
        self.llm = FakeLLM(self.run_dir, self.ledger, script, models={
            "challenger": "gpt-6.1-sol", "statement_match": "qwen3.6-35b-a3b-mlx",
            "statement_crosscheck": "anthropic/claude-sonnet-5-5"})
        self.ctx = StageContext(run_id=self.run_dir.name, run_dir=self.run_dir, ledger=self.ledger, store=self.store,
                                llm=self.llm, runner=None, bench=None, cfg=orch_config(),
                                budget=Budget(1, 500, 10**9, 3600.0), lenses={},
                                math=MathRunner(self.executor, mf.IMAGE, self.run_dir))
        rec = copy.deepcopy(rec or K3["record"])
        for statement in m_math.all_statements(rec):
            statement["sha256"] = m_math.statement_digest(statement)
        for definition in rec["definitions"]:
            definition["sha256"] = mc.definition_hash(definition)
        stored = self.ctx.put_json_artifact("math_statements", self.ctx.artifact_name(CARD_ID), rec, CARD_ID)
        hashes = {s["id"]: s["sha256"] for s in m_math.all_statements(rec)}
        hashes.update({d["id"]: d["sha256"] for d in rec["definitions"]})
        self.ctx.emit("math_statements_frozen", {"card_id": CARD_ID, "path": stored["path"].relative_to(
            self.run_dir).as_posix(), "sha256": stored["sha256"], "hashes": hashes})
        verdict = {"card_id": CARD_ID, "stage": "M", "verdict": K3["m_verdict"], "kill_codes": K3["m_kill_codes"],
                   "claims": K3["m_claims"], "models": K3["m_models"], "checks": copy.deepcopy(K3["m_checks"])}
        self.ctx.put_json_artifact("math_verdict", self.ctx.artifact_name(CARD_ID), verdict, CARD_ID)

    def run(self):
        return asyncio.run(c_challenge.run(self.ctx, CARD_ID))

    def verdict(self) -> dict:
        return [v for v in c_challenge._verdicts(self.ctx, CARD_ID) if v["stage"] == "C"][-1]

    def claims(self) -> dict:
        return {c["claim_id"]: c for c in self.verdict()["claims"]}

    def statuses(self) -> dict:
        return {cid: c["status"] for cid, c in self.claims().items()}

    def challenge(self) -> dict:
        return [json.loads((self.run_dir / e["data"]["path"]).read_text()) for e in self.ledger.iter_events("artifact")
                if e["data"]["kind"] == "challenge"][-1]

    def user(self, prompt: str) -> str:
        return next(c["user"] for c in self.llm.calls if c["prompt"] == prompt)

    def cells(self) -> dict:
        return {cell: calibration.cell_outcome(self.statuses().get(cell)) for cell in calibration.CELLS}


# ----- the recorded k3-run1 comparison --------------------------------------------------------------------------------

def test_under_the_old_rule_the_recorded_comparison_blocked_sixteen_statements():
    reasons = {e["statement_id"]: mc.mismatch_reasons(e) for e in K3["replies"]["compare"]["statements"]}
    substantive = {d["definition_id"] for d in K3["replies"]["compare"]["definitions"] if d["verdict"] == "substantive"}
    assert substantive == {"D1"}
    assert sorted(sid for sid, r in reasons.items() if r) == ["O13.a", "O14.a", "O16.a"]
    old = sorted(set(D1_USERS) | {sid for sid, r in reasons.items() if r})
    assert old == K3["recorded_c"]["mismatched"] and len(old) == 16
    assert set(K3["recorded_c"]["claims"].values()) <= {"blocked", "proven", "not_formalized"}
    assert all(K3["recorded_c"]["claims"][cell] == "blocked" for cell in calibration.CELLS)


def test_the_recorded_d1_and_p4_findings_quote_nothing_and_block_nothing(tmp_path):
    replay = Replay(tmp_path, recorded())
    replay.run()
    # Nothing was anchored, so the cross-check never ran.
    assert "statement_crosscheck" not in replay.llm.prompts()
    # Every claim keeps the status its checks gave at stage M (the challenger found nothing).
    assert replay.statuses() == M_STATUSES
    assert all(c["match"] == "matched" for c in replay.claims().values())
    assert replay.cells() == {"P1": "pass", "P2.copy": "fail", "P2.recall": "fail", "P2.nesting": "fail",
                              "P2.composition": "fail", "P3": "pass", "P5": "fail"}
    assert {s: replay.statuses()[s] for s in P4_STATEMENTS.values()} == {
        "P4.copy": "failed", "P4.recall": "failed", "P4.nesting": "observed", "P4.composition": "failed"}
    verdict = replay.verdict()
    assert verdict["verdict"] == "fail" and "statement_mismatch" not in verdict["kill_codes"]
    assert "obligation_failed:P2.copy" in verdict["kill_codes"]
    # The rejected findings are recorded with the host's reason.
    challenge = replay.challenge()
    d1 = next(d for d in challenge["definitions"] if d["definition_id"] == "D1")
    assert d1["verdict"] == "substantive" and d1["host"] == "rejected" and "no objection" in d1["host_reason"]
    o13 = next(i for i in challenge["statement_match"] if i["statement_id"] == "O13.a")
    assert o13["match"] == "matched" and {o["item"] for o in o13["objections"]} == {"constants", "domains",
                                                                                     "direction"}
    assert all(o["host"] == "rejected" for o in o13["objections"])


def test_a_d1_objection_that_names_the_cards_own_state_is_rejected_by_the_mechanism_rule(tmp_path):
    objection = dict(capacity_objection(), asserts="state")
    replay = Replay(tmp_path, with_d1_objection(objection))
    replay.run()
    d1 = next(d for d in replay.challenge()["definitions"] if d["definition_id"] == "D1")
    assert d1["host"] == "rejected" and "restates the card's state, update, readout or learning rule" in \
        d1["host_reason"]
    assert "statement_crosscheck" not in replay.llm.prompts()
    assert replay.statuses() == M_STATUSES


def test_a_p4_size_objection_cannot_anchor_to_the_ceiling(tmp_path):
    # The recorded O13.a objection ("requires success up to L=4"), quoting the slot's ceiling text.
    objection = {"item": "domains", "claim_quote": "the copy task at L at most 4",
                 "statement_quote": '"size": 2', "difference": "Formal fixes size=2, claim requires up to L=4."}
    replay = Replay(tmp_path, with_statement_objection("O13.a", objection))
    replay.run()
    item = next(i for i in replay.challenge()["statement_match"] if i["statement_id"] == "O13.a")
    assert item["objections"][0]["host"] == "rejected"
    assert "not text of its instantiated_claim" in item["objections"][0]["host_reason"]
    assert item["match"] == "matched" and replay.statuses()["P4.copy"] == "failed"
    assert "statement_crosscheck" not in replay.llm.prompts()


def test_the_p4_comparison_sees_the_selected_case_and_the_back_translation_stays_blind(tmp_path):
    replay = Replay(tmp_path, recorded())
    replay.run()
    compare, back = replay.user("statement_match"), replay.user("statement_backtranslate")
    items = {i["statement_id"]: i for i in json.loads(compare.split("STATEMENTS TO COMPARE:\n", 1)[1]
                                                       .split("\n\nDEFINITIONS:", 1)[0])}
    copy_item = items["O13.a"]
    assert copy_item["profile_slot"]["selected_case"] == {"task": "copy", "L": 2, "examples": 400,
                                                          "seeds": [0, 1, 2], "threshold": 0.99}
    assert copy_item["profile_slot"]["size_ceiling"] == {"L": 4}
    claim = copy_item["instantiated_claim"]
    assert "the selected L = 2 from m = 400 examples" in claim and "seeds 0, 1, 2" in claim
    assert "at least 0.99" in claim and "own learning rule" in claim and "random answers" in claim
    assert "at most 4" not in claim
    assert items["O16.a"]["profile_slot"]["size_ceiling"] == {"T": 4}
    assert "two-hop lookup at the selected T = 2" in items["O16.a"]["instantiated_claim"]
    assert "instantiated_claim" not in items["O8.a"] and "selected_case" not in items["O8.a"]["profile_slot"]
    # The comparison reads the card's mechanism with anchors and the definitions' provenance; the blind
    # back-translation reads neither, nor any claim.
    assert "CARD MECHANISM" in compare and '"anchor": "mechanism.state.z"' in compare
    assert "frozen by the statement fixer at stage M" in compare
    for section in ("match_mechanism", "profile_match_selected", "profile_match_limits"):
        assert load_section(section) in compare
    meaning = CARD["mechanism"]["state"][0]["meaning"]
    assert meaning in compare and meaning not in back
    assert "CARD MECHANISM" not in back and "instantiated_claim" not in back and "selected_case" not in back
    assert "profile P4.copy" not in back


def test_a_d1_objection_the_crosscheck_rejects_is_recorded_as_unresolved_and_keeps_the_checks(tmp_path):
    replay = Replay(tmp_path, with_d1_objection(capacity_objection()),
                    crosscheck=mf.crosscheck_output(default="rejected"))
    replay.run()
    # The disagreement is recorded: the matcher's anchored objection, the separate cross-check call and its verdict.
    cross = next(c for c in replay.llm.calls if c["prompt"] == "statement_crosscheck")
    assert cross["role"] == "statement_crosscheck" and '"objection_id": "D1#1"' in cross["user"]
    challenge = replay.challenge()
    assert challenge["crosscheck"]["model"] == "anthropic/claude-sonnet-5-5"
    assert challenge["crosscheck"]["verdicts"] == [{"objection_id": "D1#1", "verdict": "rejected",
                                                    "why": "read the quoted texts: rejected"}]
    d1 = next(d for d in challenge["definitions"] if d["definition_id"] == "D1")
    assert (d1["host"], d1["adjudication"], d1["crosscheck"]) == ("anchored", ma.DISPUTED, "rejected")
    verdict = replay.verdict()
    assert verdict["models"]["statement_crosscheck"] == ["anthropic/claude-sonnet-5-5"]
    unresolved = {i["statement_id"] for i in verdict["statement_match"] if i["match"] == "unresolved"}
    assert sorted(unresolved) == D1_USERS
    # Not blocked, not passed: each claim keeps the status its checks give, marked unresolved.
    assert replay.statuses() == M_STATUSES
    claims = replay.claims()
    assert claims["P2.copy"]["status"] == "failed" and claims["P2.copy"]["match"] == "unresolved"
    assert "D1" in claims["P2.copy"]["match_detail"] and "cross-check" in claims["P2.copy"]["match_detail"]
    assert claims["complexity.time_per_token"]["match"] == "matched"           # O1.a does not use D1
    assert verdict["verdict"] == "fail" and "statement_mismatch" not in verdict["kill_codes"]
    assert "match_unresolved:P2.copy" in verdict["kill_codes"] and "obligation_failed:P2.copy" in verdict["kill_codes"]
    assert "match_unresolved:complexity.time_per_token" not in verdict["kill_codes"]
    validate("math_verdict", verdict)


def test_an_unresolved_cell_is_not_shown_and_holds_back_acceptance(tmp_path):
    """FIX-17 (decisions XC and NS) replaced the FIX-7 rule that scored an unresolved cell from its checks: the
    dispute goes to the statement tiebreak, which is unavailable here (its call fails), so the cell is not shown."""
    replay = Replay(tmp_path, with_d1_objection(capacity_objection()),
                    crosscheck=mf.crosscheck_output(default="rejected"))
    replay.run()
    outcome = calibration.outcome_record(replay.ctx, "k3", CARD, 1, {"max_cost_exponent": 1.5,
                                                                     "state_growth": "log",
                                                                     "coupled_max_cost_exponent": 1.5,
                                                                     "coupled_max_state_exponent": 0.5})
    assert outcome["cells"] == {cell: "not_shown" for cell in calibration.CELLS}
    assert outcome["unresolved"] == list(calibration.CELLS)
    assert outcome["stages"]["C"]["slots"]["P2.copy"] == {"status": "failed", "tier": "argued", "match": "unresolved"}
    expected = load_expected()
    report = calibration.compare({"cards": {"k3": expected["cards"]["k3"]}, "runs_required": 1},
                                 [dict(outcome, stopped=None)])
    run = report["cards"]["k3"]["runs"][0]
    assert run["unresolved"] == list(calibration.CELLS)
    assert "P2.copy: statement match unresolved after the tiebreak (scored not shown)" in run["mismatches"]
    assert not run["reproduced"] and not report["accepted"]
    assert report["cards"]["k3"]["agreement"]["P2.copy"]["values"] == ["not_shown"]
    rendered = calibration.render_report(report, [outcome], {"started": "s", "finished": "f", "runs": 1})
    assert "not_shown (unresolved)" in rendered


def test_a_genuine_definition_contradiction_still_blocks_when_both_readings_agree(tmp_path):
    rec = copy.deepcopy(K3["record"])
    d1 = next(d for d in rec["definitions"] if d["id"] == "D1")
    added = "The carried state also holds an unbounded stack of every bracket read so far."
    d1["text"] += " " + added
    objection = {"definition_quote": added, "card_quote": "the only thing carried between tokens",
                 "kind": "contradicts_card", "asserts": "state",
                 "difference": "the card carries one index; the definition adds an unbounded stack"}
    replay = Replay(tmp_path, with_d1_objection(objection), rec=rec)
    replay.run()
    d1_reading = next(d for d in replay.challenge()["definitions"] if d["definition_id"] == "D1")
    assert (d1_reading["host"], d1_reading["adjudication"]) == ("anchored", ma.CONFIRMED)
    statuses = replay.statuses()
    assert all(statuses[cell] == "blocked" for cell in calibration.CELLS)
    assert statuses["complexity.time_per_token"] == "proven"
    verdict = replay.verdict()
    assert verdict["verdict"] == "fail" and "statement_mismatch" in verdict["kill_codes"]
    assert not any(code.startswith("match_unresolved:") for code in verdict["kill_codes"])
    assert all(c["match"] == "mismatched" for cid, c in replay.claims().items() if cid in calibration.CELLS)
    assert "definition D1 is a substantive premise" in replay.claims()["P2.copy"]["detail"]


def test_a_genuine_statement_mismatch_still_blocks_when_both_readings_agree(tmp_path):
    objection = {"item": "exponents", "claim_quote": "Theta(k V)", "statement_quote": "c*(k*v) ≤ tokenCost k v",
                 "difference": "the claim is linear in k V; suppose the statement bounded k*v*v"}
    replay = Replay(tmp_path, with_statement_objection("O1.a", objection))
    replay.run()
    claims = replay.claims()
    assert claims["complexity.time_per_token"]["status"] == "blocked"
    assert claims["complexity.time_per_token"]["match"] == "mismatched"
    assert "exponents" in claims["complexity.time_per_token"]["detail"]
    assert replay.verdict()["kill_codes"].count("statement_mismatch") == 1
    # The same objection with the cross-check disagreeing is unresolved, and the proven claim stays proven.
    replay = Replay(tmp_path / "again", with_statement_objection("O1.a", objection),
                    crosscheck=mf.crosscheck_output({"O1.a#1": "rejected"}))
    replay.run()
    claim = replay.claims()["complexity.time_per_token"]
    assert (claim["status"], claim["match"]) == ("proven", "unresolved")


def test_quotes_that_are_not_exact_text_anchor_nothing():
    item = {"claim": "complexity time_per_token: Theta(k V)", "claim_quote": "Theta(k V)"}
    entry = {"rubric": {name: {"verdict": "same", "note": ""} for name in mc.RUBRIC}, "direction": "match",
             "why": "x", "objections": [
                 {"item": "exponents", "claim_quote": "Theta(k V^2)", "statement_quote": "tokenCost k v",
                  "difference": "invented claim text"},
                 {"item": "exponents", "claim_quote": "Theta(k V)", "statement_quote": "tokenCost k v * v",
                  "difference": "invented statement text"},
                 {"item": "exponents", "claim_quote": "Theta(k V)", "statement_quote": "Theta(k V)",
                  "difference": "the same text on both sides"}]}
    spec = "def Claim : Prop := ∃ c C : ℕ, c*(k*v) ≤ tokenCost k v"
    found = ma.statement_objections("O1.a", entry, item, spec + " -- Theta(k V)", [])
    assert [o["host"] for o in found] == ["rejected"] * 3
    assert "claim quote" in found[0]["host_reason"] and "statement quote" in found[1]["host_reason"]
    assert "same text" in found[2]["host_reason"]


def test_a_mismatched_or_unresolved_statement_cannot_discharge_a_premise(tmp_path):
    from hybrid.tests.research.test_math_stages import Harness
    fixer = mf.fixer_output(premises=[mf.premise("A1", "the codebook has no collisions", does_work=False)],
                            capability_1_premises=["A1"],
                            extra_obligations=[mf.obligation("O7", None, "codebook", [mf.sympy_statement()],
                                                             discharges=["A1"])])
    script = mf.math_script(fixer=fixer, match=mf.match_output({"O7.a": ("weaker", "hypotheses")}))
    script["statement_crosscheck"] = mf.crosscheck_output(default="rejected")
    h = Harness(tmp_path, script)
    assert h.m() == "C"
    assert h.ledger_of("M")["A1"]["status"] == "proven"
    h.c()
    match = next(i for i in h.verdict("C")["statement_match"] if i["statement_id"] == "O7.a")
    assert match["match"] == "unresolved"
    assert h.ledger_of("C")["A1"]["status"] == "open"
    assert "statement_mismatch" not in h.verdict("C")["kill_codes"]


# ----- the K1 smoke's P3 objection --------------------------------------------------------------------------------------

def k1_replay(tmp_path, *, crosscheck: str, anchored: bool = True) -> fix4.Replay:
    replay = fix4.Replay(tmp_path, [fix4.converted_reply(), fix4.empty_reply()])
    replay.llm.script["statement_crosscheck"] = mf.crosscheck_output(default=crosscheck)
    if not anchored:
        replay.llm.script["statement_match"] = lambda index, user: dict(
            fix4.recorded_compare(fix4.FIX["replies"]["compare"])(index, user),
            statements=[dict(e, objections=[]) for e in fix4.FIX["replies"]["compare"]["statements"]])
    return replay


def test_the_k1_smoke_p3_objection_with_quotes_and_a_disagreeing_crosscheck_is_unresolved(tmp_path):
    replay = k1_replay(tmp_path, crosscheck="rejected")
    replay.run()
    verdict = replay.verdict()
    p3 = next(c for c in verdict["claims"] if c["claim_id"] == "P3")
    assert (p3["status"], p3["match"]) == ("proven", "unresolved")
    assert "quantifiers" in p3["match_detail"]
    assert "match_unresolved:P3" in verdict["kill_codes"] and "statement_mismatch" not in verdict["kill_codes"]
    item = next(i for i in verdict["statement_match"] if i["statement_id"] == "O15.a")
    assert item["match"] == "unresolved" and item["disputed"]
    assert replay.challenge()["crosscheck"]["verdicts"][0]["verdict"] == "rejected"


def test_the_k1_smoke_p3_objection_blocks_only_when_the_crosscheck_upholds_it(tmp_path):
    replay = k1_replay(tmp_path, crosscheck="upheld")
    replay.run()
    assert replay.statuses()["P3"] == "blocked"
    assert "statement_mismatch" in replay.verdict()["kill_codes"]


def test_the_recorded_k1_smoke_p3_objection_without_quotes_blocks_nothing(tmp_path):
    replay = k1_replay(tmp_path, crosscheck="upheld", anchored=False)
    replay.run()
    assert replay.statuses()["P3"] == "proven"
    assert "statement_crosscheck" not in [c["prompt"] for c in replay.llm.calls]
    assert not any(code.startswith(("statement_mismatch", "match_unresolved")) for code in
                   replay.verdict()["kill_codes"])


# ----- gate, schema, routing and families ---------------------------------------------------------------------------

def test_the_gate_turns_a_pass_with_unresolved_claims_into_undetermined():
    record = {"claim_id": "P1", "status": "observed", "match": "unresolved", "open_premises": []}
    assert mathrecords.challenge_gate([record], [], conservation=[], magic_x=[], magic_x_missing=[],
                                      counterexamples=[]) == ("undetermined", ["match_unresolved:P1"])
    failed = dict(record, claim_id="P2.copy", status="failed")
    assert mathrecords.challenge_gate([record, failed], [], conservation=[], magic_x=[], magic_x_missing=[],
                                      counterexamples=[]) == ("fail", ["match_unresolved:P1",
                                                                       "match_unresolved:P2.copy",
                                                                       "obligation_failed:P2.copy"])
    # Precedence is unchanged: a blocked claim is still blocked whatever its checks said.
    outcomes = {"O1": {"result": "failed", "tier": "argued", "statements": [{"result": "failed", "phrase": "x"}],
                       "obligation_id": "O1", "phrase": "x"}}
    records = mathrecords.claim_records([{"id": "P2.copy", "text": "t"}], [{"id": "O1", "claim_id": "P2.copy"}],
                                        outcomes, [], blocked={"P2.copy": "mismatch"}, unresolved={})
    assert records[0]["status"] == "blocked" and records[0]["match"] == "mismatched"


def test_the_match_schema_asks_for_anchors_and_offers_the_mechanism_verdict():
    schema = mc.match_schema(["O1.a"], ["D1"], [], ["c1"])
    statement = schema["properties"]["statements"]["items"]
    assert "objections" in statement["required"]
    assert set(statement["properties"]["objections"]["items"]["required"]) == {"item", "claim_quote",
                                                                               "statement_quote", "difference"}
    definition = schema["properties"]["definitions"]["items"]
    assert definition["properties"]["verdict"]["enum"] == ["convention", "mechanism", "substantive"]
    assert "objection" in definition["required"]
    assert "mechanism" in mathrecords.DEFINITION_KINDS
    fixer = mc.statement_fixer_schema(["c1"], 4)
    assert "mechanism" in fixer["properties"]["definitions"]["items"]["properties"]["kind"]["enum"]
    validate("math_statement", dict(copy.deepcopy(K3["record"]), definitions=[
        {"id": "D1", "kind": "mechanism", "text": "the card's state", "sha256": "0" * 64}]))


def test_the_crosscheck_may_share_the_provers_family_but_not_the_matchers_or_the_fixers():
    used = {"statement_fixer": ["gpt-6.1-sol"], "prover": ["claude-opus-5-5"],
            "statement_match": ["qwen3.6-35b-a3b-mlx"], "statement_crosscheck": ["anthropic/claude-sonnet-5-5"]}
    assert families.violations(used) == []
    assert families.violations(dict(used, statement_crosscheck=["qwen3.6-35b-a3b-mlx"])) == [
        "statement_crosscheck and statement_match share the qwen family"]
    assert families.violations(dict(used, statement_crosscheck=["gpt-6.1-sol"])) == [
        "statement_crosscheck and statement_fixer share the openai family"]


def test_the_staged_routing_adds_only_the_sonnet_crosscheck_in_every_table():
    for table in ("llxprt", "gateway", "gateway_claude_local"):
        staged = load_config(STAGED_MODELS, routing=table)
        shipped = load_config(MODELS_TOML, routing=table)
        entries = staged.roles["statement_crosscheck"]
        assert "sonnet" in entries[0].model and all(families.family(e.model) == "anthropic" for e in entries)
        assert staged.settings_for("statement_crosscheck").effort == "medium"
        # FIX-17 (decision XC) later added the statement tiebreak to the shipped tables; it is the only role the
        # FIX-7 staging file lacks, and every other role is unchanged.
        assert set(shipped.roles) - set(staged.roles) == {"statement_tiebreak"}
        assert {r: v for r, v in staged.roles.items() if r != "statement_crosscheck"} == \
            {r: v for r, v in shipped.roles.items() if r not in ("statement_crosscheck", "statement_tiebreak")}
        primaries = {role: [staged.roles[role][0].model] for role in ("statement_fixer", "prover", "statement_match",
                                                                      "statement_crosscheck")}
        assert families.violations(primaries) == [], table


def test_the_p4_slot_text_says_one_selected_size_and_k3s_case_passes():
    for slot in profile.P4_SLOTS:
        assert "ceiling bounds the choice" in profile.SLOT_TEXT[slot]
        assert "P2, not P4, covers every listed size" in profile.SLOT_TEXT[slot]
    assert "one selected length L with 1 <= L <= 4" in profile.SLOT_TEXT["P4.copy"]
    o13 = next(s for o in K3["record"]["obligations"] for s in o["statements"] if s["id"] == "O13.a")
    assert profile.statement_problems("P4.copy", "learnability", o13["spec"]) == []
    assert profile.statement_problems("P4.copy", "learnability", dict(o13["spec"], size=5)) == [
        "P4.copy: the solvable case has L at most 4, not 5"]
    # The quotes the fixer copied from the earlier slot text are still slot text.
    claims = {c["id"]: c["text"] for c in profile.profile_claims()}
    for obligation in K3["record"]["obligations"]:
        if profile.is_slot(obligation["claim_id"]):
            assert mc.quote_ok(obligation, CARD, claims), obligation["claim_id"]
    assert "each P4 statement selects one size" in profile.template_text(SimpleCaps())


class SimpleCaps:
    wall_s, memory_mb, cpus = 1800, 4096, 1


def test_the_fixer_is_told_about_mechanism_definitions():
    section = load_section("mechanism_definitions")
    assert 'kind "mechanism"' in section
    for name in ("mechanism_definitions", "match_mechanism", "profile_match_selected", "construction_baseline"):
        text = load_section(name)
        assert "\u2014" not in text and "canonical" not in text.lower() and "load-bearing" not in text.lower()
    prompt = (ROOT / "hybrid" / "research" / "prompts" / "statement_crosscheck.md").read_text(encoding="utf-8")
    assert prompt.startswith("# prompt: statement_crosscheck") and "\u2014" not in prompt


# ----- construction ranking against the query-independent baseline -------------------------------------------------

def k1_statement(sid: str) -> dict:
    return next(s for o in K1_RUN1["frozen_record"]["obligations"] for s in o["statements"] if s["id"] == sid)


def k1_attempts(sid: str) -> list[dict]:
    return [c for c in K1_RUN1["checks"] if c["statement_id"] == sid]


def test_the_baseline_is_the_best_query_independent_output_on_the_checked_instances():
    statement = k1_statement("O9.a")
    baseline = pm.query_independent_baseline(statement)
    assert sorted(baseline) == [4, 16, 32]
    # Recomputed by hand: per instance, the most frequent answer among its queries.
    instances = constructions.generate("composition", 4, statement["spec"]["count"], 0)
    right = sum(max([a for _, a in i.queries].count(v) for v in {a for _, a in i.queries}) for i in instances)
    assert baseline[4] == pytest.approx(right / (4 * len(instances)))
    assert baseline[4] > 0.6 and baseline[16] > 0.25 and baseline[32] > 0.15


@pytest.mark.parametrize("sid", ["O9.a", "O9.b"])
def test_the_recorded_k1_shortcuts_are_never_presented_as_the_best_attempt(sid):
    statement = k1_statement(sid)
    best = pm.BestAttempts()
    for check in k1_attempts(sid):
        best.note(statement, check["attempt"], check, check["text"])
    assert best.prompt_block(sid) is None
    ranking = best.ranking_block(sid)
    assert len(ranking["attempts"]) == 2 and ranking["best_attempt_shown"] is False
    for attempt in ranking["attempts"]:
        assert attempt["beats_baseline"] is False
        assert all(row["excess"] <= 0 for row in attempt["sizes"])
        assert all({"size", "accuracy", "baseline", "excess"} == set(row) for row in attempt["sizes"])
    if sid == "O9.a":
        modal = next(a for a in ranking["attempts"] if a["attempt"] == 4)
        assert [row["accuracy"] for row in modal["sizes"]] == pytest.approx([0.52625, 0.1934375, 0.10984375])


def test_a_construction_that_beats_the_baseline_ranks_above_a_higher_scoring_shortcut():
    statement = k1_statement("O9.a")
    baseline = pm.query_independent_baseline(statement)
    shortcut = copy.deepcopy(k1_attempts("O9.a")[3])
    for row in shortcut["details"]["results"]:
        row["accuracy"] = baseline[row["size"]]                       # as good as ignoring the query can be
    partial = copy.deepcopy(k1_attempts("O9.a")[0])
    for row, accuracy in zip(partial["details"]["results"], (baseline[4] + 0.05, 0.0, 0.0)):
        row["accuracy"] = accuracy                                    # lower mean, but it computes something
    best = pm.BestAttempts()
    best.note(statement, 1, shortcut, "shortcut source")
    best.note(statement, 2, partial, "partial source")
    block = best.prompt_block("O9.a")
    assert block["attempt"] == 2 and block["source"] == "partial source"
    ranking = best.ranking_block("O9.a")
    assert [a["attempt"] for a in ranking["attempts"]] == [2, 1]
    assert [a["beats_baseline"] for a in ranking["attempts"]] == [True, False]


def test_the_prover_sees_the_ranking_and_no_best_attempt_after_the_recorded_shortcuts():
    statement = k1_statement("O9.a")
    best = pm.BestAttempts()
    checks = k1_attempts("O9.a")
    for check in checks:
        best.note(statement, check["attempt"], check, check["text"])
    record = copy.deepcopy(K1_RUN1["frozen_record"])
    user = m_math.prover_user(K1_RUN1["card"], record, [statement], {"O9.a": "status violated"}, None,
                              {"O9.a": checks[-1]["text"]}, {"O9.a": ["attempt 4: ..."]}, best)
    item = cf._shown(user)[0]
    assert "best_attempt" not in item
    assert item["previous_source"] == checks[-1]["text"]
    assert len(item["attempts_vs_baseline"]["attempts"]) == 2
    assert load_section("construction_baseline") in user and load_section("construction_contract") in user
