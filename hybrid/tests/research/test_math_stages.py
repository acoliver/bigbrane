"""Stages M and C (DISCOVERY-PIPELINE-PLAN.md sections 4.1 to 4.3 and 5) with a scripted LLM and a real MathRunner
whose jobs are answered by math_fakes.MathExecutor (Lean scripted, constructions, conservation and magic-X run on
the host)."""
import asyncio
import json

import pytest

from hybrid.llm.errors import TransportError
from hybrid.math import manifest
from hybrid.math.runner import MathRunner
from hybrid.research.ledger import Ledger
from hybrid.research.schemas import validate
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages.common import Budget, StageContext, StageError
from hybrid.research.store import Store
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, orch_config

class Harness:
    def __init__(self, tmp_path, script, *, models=None, z3=mf.z3_default, host_kinds=None):
        self.run_dir = (tmp_path / "run").resolve()
        self.ledger = Ledger(self.run_dir)
        self.store = Store(self.run_dir, self.ledger)
        self.store.put_card(make_card(mf.CARD))
        options = {} if host_kinds is None else {"host_kinds": host_kinds}
        self.executor = mf.MathExecutor(tmp_path / "exec", z3=z3, **options)
        self.llm = FakeLLM(self.run_dir, self.ledger, script, models=models)
        cfg = orch_config()
        cfg["math"]["max_prover_attempts"] = 2
        self.ctx = StageContext(run_id=self.run_dir.name, run_dir=self.run_dir, ledger=self.ledger, store=self.store,
                                llm=self.llm, runner=None, bench=None, cfg=cfg,
                                budget=Budget(1, 500, 10**9, 3600.0), lenses={},
                                math=MathRunner(self.executor, mf.IMAGE, self.run_dir))

    def m(self):
        return asyncio.run(m_math.run(self.ctx, mf.CARD))

    def c(self):
        return asyncio.run(c_challenge.run(self.ctx, mf.CARD))

    def events(self, type):
        return [e for e in self.ledger.iter_events(type)]

    def verdict(self, stage):
        records = [r for r in c_challenge._verdicts(self.ctx, mf.CARD) if r["stage"] == stage]
        return records[-1]

    def claim(self, stage, claim_id):
        return next(c for c in self.verdict(stage)["claims"] if c["claim_id"] == claim_id)

    def ledger_of(self, stage):
        return {p["id"]: p for p in self.ledger_artifact(stage)["premises"]}

    def ledger_artifact(self, stage):
        name = self.verdict(stage)["assumption_ledger"]
        return json.loads((self.run_dir / "artifacts" / "assumption_ledger" / name).read_text())

    def artifacts(self, kind):
        return [json.loads((self.run_dir / e["data"]["path"]).read_text()) for e in self.events("artifact")
                if e["data"]["kind"] == kind]

    def card_verdict(self, stage):
        return self.store.get_card(mf.CARD)["status"]["verdicts"][stage]


# ----- freezing ---------------------------------------------------------------------------------------------------

def test_statements_are_frozen_and_hashed_before_any_prover_call(tmp_path):
    h = Harness(tmp_path, mf.math_script())
    assert h.m() == "C"
    frozen = h.events("math_statements_frozen")
    assert len(frozen) == 1
    first_prover = next(e for e in h.events("llm_call") if e["data"]["role"] == "prover")
    assert frozen[0]["seq"] < first_prover["seq"]
    record = json.loads((h.run_dir / frozen[0]["data"]["path"]).read_text())
    validate("math_statement", record)
    lean = next(s for o in record["obligations"] for s in o["statements"] if s["form"] == "lean")
    assert lean["spec"] == mf.LEAN_STATEMENT and frozen[0]["data"]["hashes"][lean["id"]] == lean["sha256"]
    prover_user = next(c["user"] for c in h.llm.calls if c["role"] == "prover")
    assert json.dumps(mf.LEAN_STATEMENT, ensure_ascii=False)[1:-1] in prover_user
    # Every check of a frozen statement carries that statement's frozen hash.
    for check in h.artifacts("math_check"):
        assert check["statement_sha256"] == frozen[0]["data"]["hashes"][check["statement_id"]]
    # The prover's schema has no field that could carry a statement.
    schema = next(c["schema"] for c in h.llm.calls if c["role"] == "prover")
    # FIX-5 added recipe_change_reason, a free-text reason a learner gives for changing a recipe that passed.
    assert set(schema["properties"]["attempts"]["items"]["properties"]) == {"statement_id", "proof", "source", "notes",
                                                                            "recipe_change_reason"}


def test_a_frozen_statement_edited_afterwards_stops_stage_c(tmp_path):
    h = Harness(tmp_path, mf.math_script())
    h.m()
    path = h.run_dir / h.events("math_statements_frozen")[0]["data"]["path"]
    path.write_text(path.read_text().replace("n + 0 = n", "n + 0 = n ∨ True"))
    calls = len(h.llm.calls)
    with pytest.raises(StageError, match="changed after it was frozen"):
        h.c()
    assert len(h.llm.calls) == calls


def test_a_resumed_m_reuses_the_frozen_statements(tmp_path):
    script = mf.math_script()
    h = Harness(tmp_path, script)
    h.llm.fail = {"prover": TransportError("down")}
    with pytest.raises(StageError):
        h.m()
    h.llm.fail = {}
    assert h.m() == "C"
    assert h.llm.prompts().count("statement_fixer") == 1
    assert len(h.events("math_statements_frozen")) == 1


# ----- the prover cannot change or weaken a statement ------------------------------------------------------------

def test_a_weaker_restated_claim_is_rejected_and_the_claim_stays_unproven(tmp_path):
    h = Harness(tmp_path, mf.math_script(proofs=mf.WEAKER_PROOF))
    assert h.m() is None
    attempts = [e["data"] for e in h.events("proof_attempt") if e["data"]["statement_id"] == "O5.a"]
    assert [(a["attempt"], a["status"]) for a in attempts] == [(1, "reject"), (2, "reject")]
    check = next(c for c in h.artifacts("math_check") if c["statement_id"] == "O5.a")
    assert any("not exactly the constant Claim (restated claim)" in r for r in check["reasons"])
    assert h.verdict("M")["verdict"] == "undetermined"
    assert h.verdict("M")["kill_codes"] == ["unproven:capability.1"]
    assert h.claim("M", "capability.1")["status"] == "unproven"


def test_a_proof_that_redeclares_claim_is_rejected_without_running_lean_then_a_retry_passes(tmp_path):
    h = Harness(tmp_path, mf.math_script(proofs=[mf.REDECLARE_PROOF, mf.GOOD_PROOF]))
    assert h.m() == "C"
    first, second = [c for c in h.artifacts("math_check") if c["statement_id"] == "O5.a"]
    assert first["status"] == "reject" and first["job_ids"] == [] and "redeclares Claim" in first["evidence"]
    assert second["status"] == "pass" and len(second["job_ids"]) == 2
    # The Lean statement and the construction go to separate prover calls; the second call that shows O5.a is the
    # retry.
    retry_user = [c["user"] for c in h.llm.calls if c["role"] == "prover" and '"statement_id": "O5.a"' in c["user"]][1]
    assert "previous_attempt_failed" in retry_user and "redeclares Claim" in retry_user


def test_a_sorry_proof_is_rejected(tmp_path):
    h = Harness(tmp_path, mf.math_script(proofs=mf.SORRY_PROOF))
    assert h.m() is None
    check = [c for c in h.artifacts("math_check") if c["statement_id"] == "O5.a"][-1]
    assert check["status"] == "reject" and check["decided"] and "uses sorry (sorryAx)" in check["reasons"]
    assert h.claim("M", "capability.1")["status"] == "unproven"
    assert h.claim("M", "capability.1")["tier"] == "argued"


# ----- tiers and the premise ledger --------------------------------------------------------------------------------

def test_tier_records_never_call_an_executed_result_proven(tmp_path):
    h = Harness(tmp_path, mf.math_script())
    h.m()
    verdict = h.verdict("M")
    validate("math_verdict", verdict)
    status = {c["claim_id"]: (c["status"], c["tier"], c["label"]) for c in verdict["claims"]}
    assert status == {
        "complexity.time_per_token": ("proven", "sympy_exact", "sympy-exact"),
        "complexity.memory": ("proven", "sympy_exact", "sympy-exact"),
        "complexity.total_sequence": ("proven", "sympy_exact", "sympy-exact"),
        "sample_efficiency.prediction": ("proven_up_to_bound", "smt_bounded", "z3-checked (bounded)"),
        "capability.1": ("proven", "lean", "lean-proven"),
        "capability.2": ("observed", "executed", "executed-on-examples")}
    assert h.claim("M", "sample_efficiency.prediction")["phrase"] == "proven up to bound m in [1, 64]"
    observed = h.claim("M", "capability.2")["phrase"]
    assert observed == "observed on the listed sizes: recall size=4" and "proven" not in observed
    reasons = h.card_verdict("M")["reasons"]
    assert any(r.startswith("claim capability.2: observed [executed-on-examples]") for r in reasons)


def test_real_sympy_checks_give_the_sympy_exact_tier(tmp_path):
    h = Harness(tmp_path, mf.math_script(), host_kinds=("audited_construction", "sympy"))
    assert h.m() == "C"
    sympy_checks = [c for c in h.artifacts("math_check") if c["form"] == "sympy"]
    assert len(sympy_checks) == 3 and all(c["status"] == "discharged" and c["decided"] for c in sympy_checks)
    assert h.claim("M", "complexity.memory")["status"] == "proven"


def test_a_premise_with_no_measurement_plan_leaves_the_claim_unproven_at_m(tmp_path):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "an oracle segments the input into constituents", plan=None)],
                            capability_1_premises=["A1"])
    h = Harness(tmp_path, mf.math_script(fixer=fixer))
    assert h.m() is None
    verdict = h.verdict("M")
    assert verdict["verdict"] == "undetermined" and verdict["kill_codes"] == ["unproven:A1"]
    # The fixer was asked once more for the missing plan; the premise was frozen as written.
    assert h.llm.prompts().count("statement_fixer") == 2
    claim = h.claim("M", "capability.1")
    # The proof itself passed; the open premise makes the claim conditional, and nothing can settle the premise.
    assert claim["status"] == "conditional" and claim["open_premises"] == ["A1"] and claim["tier"] == "lean"
    assert claim["phrase"] == "conditional on A1: proven"
    ledger = h.ledger_of("M")
    assert ledger["A1"]["status"] == "open" and ledger["A1"]["used_by"] == ["O5"] and ledger["A1"]["evidence"] is None
    assert "no measurement plan" in ledger["A1"]["basis"]
    assert h.events("math_decision")[-1]["data"]["premises"] == {"A1": "open"}


def test_a_premise_discharged_by_a_proof_is_proven(tmp_path):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "the codebook has no collisions")],
                            capability_1_premises=["A1"],
                            extra_obligations=[mf.obligation("O7", None, "codebook", [mf.sympy_statement()],
                                                             discharges=["A1"])])
    h = Harness(tmp_path, mf.math_script(fixer=fixer))
    assert h.m() == "C"
    assert h.ledger_of("M")["A1"]["status"] == "proven"
    assert h.claim("M", "capability.1")["status"] == "proven"


def test_a_premise_discharged_by_an_executed_check_is_measured_with_its_artifact(tmp_path):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "the table answers every stored key")],
                            capability_1_premises=["A1"],
                            extra_obligations=[mf.obligation("O7", None, "table", [mf.CONSTRUCTION],
                                                             discharges=["A1"])])
    h = Harness(tmp_path, mf.math_script(fixer=fixer))
    assert h.m() == "C"
    entry = h.ledger_of("M")["A1"]
    check = next(c for c in h.artifacts("math_check") if c["statement_id"] == "O7.a")
    assert entry["status"] == "measured" and "math_check artifact" in entry["evidence"]
    assert entry["evidence"].endswith(next(e["data"]["artifact"] for e in h.events("math_check")
                                           if e["data"]["statement_id"] == "O7.a")) and check["decided"]


# ----- stage C -----------------------------------------------------------------------------------------------------

def test_clean_challenge_passes_to_s4_and_records_every_call_check_and_decision(tmp_path):
    h = Harness(tmp_path, mf.math_script())
    h.m()
    assert h.c() == "S4"
    assert h.verdict("C")["verdict"] == "pass" and h.card_verdict("C")["verdict"] == "pass"
    roles = [(e["data"]["stage"], e["data"]["role"]) for e in h.events("math_role_call")]
    # One prover call for the Lean statement and one for the construction (m_math.prover_batches). FIX-19: the
    # statement match (back-translation and comparison) also runs at M on the fixer's answer, before the freeze.
    assert roles == [("M", "statement_fixer"), ("M", "statement_match"), ("M", "statement_match"), ("M", "prover"),
                     ("M", "prover"), ("C", "challenger"), ("C", "statement_match"), ("C", "statement_match")]
    assert len(h.events("math_check")) == len(h.artifacts("math_check"))
    assert [e["data"]["stage"] for e in h.events("math_decision")] == ["M", "C"]
    assert len(h.artifacts("challenge")) == 1 and len(h.artifacts("assumption_ledger")) == 2
    by_calls = h.card_verdict("M")["by_calls"]
    assert any(c.startswith("lean-") for c in by_calls) and any(c.startswith("mathlib-") for c in by_calls)
    assert any(e["data"]["check"] == "statement_match" for e in h.events("challenge_check"))
    # The statement-match role translates the frozen statements without seeing the claims.
    back = next(c["user"] for c in h.llm.calls if c["prompt"] == "statement_backtranslate")
    assert mf.CAPABILITY_1_QUOTE not in back and "capability" not in back
    # The verdict records the sha256 of every staged check file.
    assert h.verdict("C")["manifest"] == manifest.verify()


def test_a_magic_x_premise_fails_the_card(tmp_path):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "a perfect retrieval index", "store")],
                            capability_1_premises=["A1"])
    entry = {"premise_id": "A1", "code": mf.references.RETRIEVAL_ORACLE,
             "claim": dict(mf.references.MAGIC_X_CLAIM), "quote": mf.CAPABILITY_1_QUOTE,
             "why": "the index alone answers recall"}
    h = Harness(tmp_path, mf.math_script(fixer=fixer, challenger=mf.challenger_output(magic_x=[entry])))
    assert h.m() == "C"
    assert h.ledger_of("M")["A1"]["status"] == "open" and h.claim("M", "capability.1")["status"] == "conditional"
    assert h.c() is None
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("fail", ["magic_x:A1"])
    item = verdict["magic_x"][0]
    assert item["ran"] and item["status"] == "violated" and "attributable to A1" in item["evidence"]
    request = next(r for r in h.executor.requests() if r["kind"] == "magic_x")
    assert request["assumption"]["status"] == "open" and request["assumption"]["cost_in_p1"] is False


CAPACITY = "The state holds 256 bits; recall is claimed for K = 64 pairs over V = 256 values at accuracy 0.99."


def test_a_capacity_overrun_violates_the_conservation_law(tmp_path):
    # FIX-4: the numbers are quoted from a frozen definition (they used to need card text).
    numbers = {"claims": {"state_bits": 256, "recall": [{"K": 64, "V": 256, "accuracy": 0.99}]},
               "quotes": [{"source": "definition", "ref": "D1", "text": CAPACITY}],
               "why": "state of 256 bits, 64 pairs of 8-bit values"}
    fixer = mf.fixer_output(definitions=[mf.definition("D1", CAPACITY, "accounting")])
    h = Harness(tmp_path, mf.math_script(fixer=fixer, challenger=mf.challenger_output(conservation=numbers)))
    h.m()
    assert h.c() is None
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("infeasible_proved", ["conservation:associative_recall"])
    assert verdict["conservation"]["counterexample"]["smallest_failing_K"] == 33


def test_conservation_numbers_not_backed_by_card_text_are_not_run(tmp_path):
    numbers = {"claims": {"state_bits": 256, "recall": [{"K": 64, "V": 256, "accuracy": 0.99}]},
               "quotes": [{"source": "card", "ref": None, "text": "the card stores 64 pairs"}], "why": "invented"}
    h = Harness(tmp_path, mf.math_script(challenger=mf.challenger_output(conservation=numbers)))
    h.m()
    assert h.c() == "S4"
    assert h.verdict("C")["conservation"]["ran"] is False
    assert not any(r["kind"] == "conservation" for r in h.executor.requests())


def test_a_statement_mismatch_blocks_the_claim(tmp_path):
    h = Harness(tmp_path, mf.math_script(match=mf.match_output({"O5.a": ("stronger", "exponents")})))
    h.m()
    assert h.c() is None
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("fail", ["statement_mismatch"])
    claim = h.claim("C", "capability.1")
    assert claim["status"] == "blocked" and "exponents" in claim["phrase"]
    mismatch = next(i for i in verdict["statement_match"] if i["statement_id"] == "O5.a")
    assert mismatch["mismatch"] and any("stronger" in r for r in mismatch["reasons"])


def test_a_finite_domain_disagreement_is_a_mismatch(tmp_path):
    def z3(request):
        if request["kind"] == "predicate_agreement":
            return {"status": "violated", "evidence": "disagree at m = 3",
                    "counterexample": {"assignment": {"m": 3}, "formal": True, "natural": False}}
        return mf.z3_default(request)

    h = Harness(tmp_path, mf.math_script(), z3=z3)
    h.m()
    assert h.c() is None
    assert h.verdict("C")["kill_codes"] == ["statement_mismatch"]
    assert h.claim("C", "sample_efficiency.prediction")["status"] == "blocked"
    request = next(r for r in h.executor.requests() if r["kind"] == "predicate_agreement")
    assert request["predicate"] == mf.Z3_PREDICATE and request["count"] == 1000


def test_a_mismatched_statement_cannot_discharge_a_premise(tmp_path):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "the codebook has no collisions", does_work=False)],
                            capability_1_premises=["A1"],
                            extra_obligations=[mf.obligation("O7", None, "codebook", [mf.sympy_statement()],
                                                             discharges=["A1"])])
    h = Harness(tmp_path, mf.math_script(fixer=fixer, match=mf.match_output({"O7.a": ("weaker", "hypotheses")})))
    assert h.m() == "C"
    assert h.ledger_of("M")["A1"]["status"] == "proven"
    assert h.c() is None
    assert h.verdict("C")["kill_codes"] == ["statement_mismatch"]
    assert h.ledger_of("C")["A1"]["status"] == "open"


def test_a_z3_counterexample_in_wider_bounds_refutes_the_claim(tmp_path):
    def z3(request):
        if request["kind"] == "z3" and request["spec"]["variables"]["m"]["hi"] == 1000:
            return {"status": "violated", "tier": "smt_bounded", "bound": "m in [1, 1000]",
                    "counterexample": {"m": 999}, "evidence": "z3: sat; the property fails at {'m': 999}"}
        return mf.z3_default(request)

    search = {"statement_id": "O4.a", "bounds": {"m": {"lo": 1, "hi": 1000}}, "why": "larger m"}
    h = Harness(tmp_path, mf.math_script(challenger=mf.challenger_output(searches=[search])), z3=z3)
    h.m()
    assert h.c() is None
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("infeasible_proved",
                                                           ["obligation_failed:sample_efficiency.prediction"])
    # The property and hypotheses stayed frozen; only the box changed.
    wide = [r for r in h.executor.requests() if r["kind"] == "z3"][-1]["spec"]
    assert wide["property"] == mf.Z3_SPEC["property"] and wide["variables"]["m"] == {"sort": "int", "lo": 1, "hi": 1000}


def test_an_external_component_without_costed_code_becomes_an_open_challenger_premise(tmp_path):
    # FIX-4: an external component (an index supplied from outside the mechanism) whose cost no verified P1 covers
    # stays a blocking premise, under a stable id.
    component = mf.component("perfect index", quote=mf.CAPABILITY_1_QUOTE, statement_id="O6.a", function="predict",
                             used_by=["O5"], role="external", kind="store", why="recall reads it")
    h = Harness(tmp_path, mf.math_script(challenger=mf.challenger_output(components=[component])))
    h.m()
    assert h.c() is None
    pid = c_challenge.finding_id({"obligations": [{"id": "O5", "claim_id": "capability.1"}]}, "perfect index", ["O5"])
    assert h.verdict("C")["kill_codes"] == [f"unproven:{pid}"]
    ledger = h.ledger_of("C")
    assert ledger[pid]["status"] == "open" and ledger[pid]["added_by"] == "challenger" and ledger[pid]["used_by"] == ["O5"]
    assert ledger[pid]["measurement_plan"] is None and h.claim("C", "capability.1")["status"] == "conditional"
    assert "supplied from outside the mechanism" in ledger[pid]["basis"]


# ----- check integrity and model families ------------------------------------------------------------------------

def broken_manifest(*args, **kwargs):
    raise manifest.ManifestMismatch(["conservation.py: changed"])


def test_the_manifest_is_verified_before_any_model_call_or_job(tmp_path, monkeypatch):
    h = Harness(tmp_path, mf.math_script())
    monkeypatch.setattr(manifest, "verify", broken_manifest)
    with pytest.raises(StageError, match="MANIFEST.json"):
        h.m()
    assert h.llm.calls == [] and h.executor.jobs == []


def test_a_check_file_changed_after_freezing_stops_the_check(tmp_path, monkeypatch):
    h = Harness(tmp_path, mf.math_script())
    real = manifest.verify
    state = {"n": 0}

    def changes_later(*args, **kwargs):
        state["n"] += 1
        if state["n"] > 1:
            broken_manifest()
        return real(*args, **kwargs)

    monkeypatch.setattr(manifest, "verify", changes_later)
    with pytest.raises(StageError, match="MANIFEST.json"):
        h.m()
    # FIX-19: the statement match runs on the fixer's answer before the freeze; no prover call and no job follow.
    assert h.llm.prompts() == ["statement_fixer", "statement_backtranslate", "statement_match"]
    assert h.executor.jobs == [] and h.artifacts("math_check") == []


def test_stage_c_verifies_the_manifest_first(tmp_path, monkeypatch):
    h = Harness(tmp_path, mf.math_script())
    h.m()
    calls, jobs = len(h.llm.calls), len(h.executor.jobs)
    monkeypatch.setattr(manifest, "verify", broken_manifest)
    with pytest.raises(StageError, match="MANIFEST.json"):
        h.c()
    assert (len(h.llm.calls), len(h.executor.jobs)) == (calls, jobs)


def test_the_fixer_and_the_prover_must_come_from_different_families(tmp_path):
    h = Harness(tmp_path, mf.math_script(), models={"statement_fixer": "claude-sonnet-5-5",
                                                    "prover": "anthropic/claude-opus-5-5"})
    with pytest.raises(StageError, match="statement_fixer and prover share the anthropic family"):
        h.m()


def test_the_challenger_must_not_share_the_provers_family(tmp_path):
    h = Harness(tmp_path, mf.math_script(), models={"statement_fixer": "gpt-sol-high", "prover": "opus",
                                                    "challenger": "sonnet", "statement_match": "qwen3.6-35b-a3b-mlx"})
    assert h.m() == "C"
    with pytest.raises(StageError, match="challenger and prover share the anthropic family"):
        h.c()
