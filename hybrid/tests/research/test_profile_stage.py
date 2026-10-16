"""Stage M and C with the transformer profile on (Phase 4, plan section 4.4; hybrid/math/profile.py) and the
compute target gate (hybrid/math/target.py), with a scripted LLM and library checks run by the staged driver in the
test process (calibration_fakes.FastExecutor)."""
import asyncio
import copy
import json

import pytest

from hybrid.math import profile
from hybrid.math.runner import LIBRARY_CAPS, MathRunner
from hybrid.research.calibration import load_cards
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import load_run_config, ROOT
from hybrid.research.schemas import validate
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import Budget, StageContext, StageError
from hybrid.research.store import Store
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.llm.errors import TransportError
from hybrid.tests.research import test_math_stages as tms
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, make_orchestrator, orch_config


class Harness:
    def __init__(self, tmp_path, key="k1", *, script=None, target=False, fixer=None, passes_p2=None,
                 profile_on=True, card=None):
        self.key = key
        self.card = card or load_cards((key,))[key]
        self.run_dir = (tmp_path / "run").resolve()
        self.ledger = Ledger(self.run_dir)
        self.store = Store(self.run_dir, self.ledger)
        self.store.put_card(self.card)
        self.executor = cf.FastExecutor(tmp_path / "exec")
        self.llm = FakeLLM(self.run_dir, self.ledger,
                           script or cf.script(key, self.card, fixer=fixer, passes_p2=passes_p2))
        cfg = orch_config()
        cfg["math"]["profile"] = profile_on
        cfg["target"]["enabled"] = target
        cfg["math"]["max_prover_attempts"] = 2
        self.ctx = StageContext(run_id=self.run_dir.name, run_dir=self.run_dir, ledger=self.ledger, store=self.store,
                                llm=self.llm, runner=None, bench=None, cfg=cfg,
                                budget=Budget(1, 500, 10**9, 3600.0), lenses={},
                                math=MathRunner(self.executor, mf.IMAGE, self.run_dir))

    def m(self):
        return asyncio.run(m_math.run(self.ctx, self.card["id"]))

    def c(self):
        return asyncio.run(c_challenge.run(self.ctx, self.card["id"]))

    def verdict(self, stage):
        return [r for r in c_challenge._verdicts(self.ctx, self.card["id"]) if r["stage"] == stage][-1]

    def slot(self, stage, slot):
        return next(c for c in self.verdict(stage)["claims"] if c["claim_id"] == slot)

    def frozen(self):
        event = list(self.ledger.iter_events("math_statements_frozen"))[-1]["data"]
        return json.loads((self.run_dir / event["path"]).read_text())


def with_obligation(answer, slot, **changes):
    obligation = next(o for o in answer["obligations"] if o["claim_id"] == slot)
    obligation.update(changes)
    return answer


# ----- the fixer is held to the profile ------------------------------------------------------------------------------

def test_the_fixer_must_cover_every_slot_with_the_template_shown(tmp_path):
    h = Harness(tmp_path)
    h.m()
    call = next(c for c in h.llm.calls if c["prompt"] == "statement_fixer")
    assert "THE TRANSFORMER PROFILE (Phase 4)" in call["user"] and profile.template_text(LIBRARY_CAPS) in call["user"]
    for slot in profile.SLOTS:
        assert f"- {slot}: profile {slot}: " in call["user"]
    schema = call["schema"]
    assert "limits" in schema["required"]
    obligation = schema["properties"]["obligations"]["items"]
    assert set(profile.SLOTS) <= set(obligation["properties"]["claim_id"]["enum"])
    assert {"learnability", "conservation"} <= set(obligation["properties"]["statements"]["items"]["properties"]
                                                   ["form"]["enum"])
    record = h.frozen()
    validate("math_statement", record)
    assert record["profile"]["slots"] == list(profile.SLOTS) and record["profile"]["limits"] == cf.LIMITS["k1"]
    prover = next(c for c in h.llm.calls if c["prompt"] == "prover")
    assert "PROFILE OBLIGATIONS (Phase 4)" in prover["user"]


def test_without_the_profile_the_fixer_schema_keeps_the_phase3_forms():
    schema = mc.statement_fixer_schema(["capability.1"], 12)
    forms = schema["properties"]["obligations"]["items"]["properties"]["statements"]["items"]["properties"]["form"]
    # FIX-6: the conformance form settles a premise about the card's own cost reference, so it joins the Phase 3
    # forms; the profile forms stay out.
    assert forms["enum"] == ["lean", "sympy", "z3", "simulation", "construction", "cost", "conformance"]
    assert not {"learnability", "conservation"} & set(forms["enum"])
    assert "limits" not in schema["properties"]


def test_every_slot_records_its_status_and_the_tier_it_reached(tmp_path):
    h = Harness(tmp_path)
    assert h.m() is None   # the card's own claims are not formalized by this fixer, so M is undetermined
    block = h.verdict("M")["profile"]
    assert block["passed"] and set(block["slots"]) == set(profile.SLOTS)
    assert block["slots"]["P1"]["status"] == "observed" and block["slots"]["P1"]["tier"] == "executed"
    assert block["slots"]["P3"]["status"] == "proven" and block["slots"]["P3"]["tier"] == "sympy_exact"
    for slot in profile.SLOTS:
        assert block["slots"][slot]["accepted_tiers"] == profile.accepted_tiers(slot)
        assert block["slots"][slot]["tier"] in block["slots"][slot]["accepted_tiers"]
    assert "observed on the listed sizes" in h.slot("M", "P2.nesting")["phrase"]
    assert not any(code.startswith(("unproven:P", "obligation_failed:P")) for code in h.verdict("M")["kill_codes"])


def test_failed_constructions_fail_m_with_the_slot_in_the_kill_code(tmp_path):
    h = Harness(tmp_path, "k3")
    assert h.m() is None
    verdict = h.verdict("M")
    assert verdict["verdict"] == "fail"
    assert {"obligation_failed:P2.nesting", "obligation_failed:P2.composition"} <= set(verdict["kill_codes"])
    assert h.slot("M", "P2.nesting")["status"] == "failed" and "depth" in h.slot("M", "P2.nesting")["detail"]
    assert verdict["profile"]["passed"] is False


def test_a_statement_of_a_form_the_slot_does_not_accept_is_dropped_and_the_slot_unproven(tmp_path):
    card = load_cards(("k2",))["k2"]
    answer = with_obligation(cf.fixer_output("k2", card), "P1", statements=[mf.sympy_statement()])
    h = Harness(tmp_path, "k2", fixer=answer)
    h.m()
    record = h.frozen()
    assert any("P1 accepts only cost statements, not sympy" in p for p in record["first_problems"])
    assert any(d.get("statement_id") == "O1.a" for d in record["dropped"])
    assert h.slot("M", "P1")["status"] == "not_formalized"
    assert "unproven:P1" in h.verdict("M")["kill_codes"]
    assert h.verdict("M")["target"]["status"] == "not_applicable"


def test_a_missing_slot_gets_one_retry_then_leaves_the_card_unproven(tmp_path):
    card = load_cards(("k2",))["k2"]
    h = Harness(tmp_path, "k2", fixer=cf.fixer_output("k2", card, drop_slots=("P4.copy",)))
    h.m()
    assert [c["prompt"] for c in h.llm.calls].count("statement_fixer") == 2
    retry = [c["user"] for c in h.llm.calls if c["prompt"] == "statement_fixer"][1]
    assert "profile slots with no obligation: ['P4.copy']" in retry
    assert h.slot("M", "P4.copy")["status"] == "not_formalized"
    assert "unproven:P4.copy" in h.verdict("M")["kill_codes"]


def test_p2_composition_needs_both_variants_to_hold(tmp_path):
    h = Harness(tmp_path, "k1")
    sources = dict(cf.OPEN, binding=references_binding_broken())
    h.llm.script["prover"] = prover_with(sources)
    h.m()
    record = h.slot("M", "P2.composition")
    assert record["status"] == "failed"
    assert "obligation_failed:P2.composition" in h.verdict("M")["kill_codes"]
    checks = h.verdict("M")["checks"]
    assert (checks["O5.a"]["status"], checks["O5.b"]["status"]) == ("discharged", "violated")


def references_binding_broken() -> str:
    return cf.references.BINDING_ENV.replace("return state[\"env\"].get(state[\"key\"], state[\"vbase\"])",
                                             "return state[\"vbase\"]")


def prover_with(sources):
    def answer(index, user):
        attempts = []
        for item in cf._shown(user):
            spec = item["frozen_statement"]
            source = {"cost": cf.COST["k1"][0], "construction": sources.get(spec.get("task")),
                      "learnability": cf.LEARNERS.get(spec.get("task"))}[item["form"]]
            attempts.append({"statement_id": item["statement_id"], "proof": None, "source": source, "notes": ""})
        return {"attempts": attempts}
    return answer


def test_a_limit_that_p2_beat_fails_p3(tmp_path):
    # FIX-18 (decision L): the limit is the card's own, and the fixer copies it; a card limit that its own P2
    # construction beats is still caught by consistency.
    card = copy.deepcopy(load_cards(("k1",))["k1"])
    card["limits"] = [{"task": "dyck", "size": 8, "resources": "more rounds", "why": "claimed depth limit",
                       "kind": "update"}]
    answer = cf.fixer_output("k1", card)
    # FIX-17 (decision P3): a limit names the mechanism part it follows from (k1's write update).
    answer["limits"] = [{"task": "dyck", "size": 8, "resources": "more rounds", "why": "claimed depth limit",
                         "basis": dict(cf.LIMITS["k1"][0]["basis"])}]
    h = Harness(tmp_path, "k1", fixer=answer, card=card)
    h.m()
    record = h.slot("M", "P3")
    assert record["status"] == "failed" and "met the threshold at [8, 16]" in record["detail"]
    assert "obligation_failed:P3" in h.verdict("M")["kill_codes"]


def test_p5_that_does_not_account_for_what_p2_did_fails(tmp_path):
    card = load_cards(("k1",))["k1"]
    answer = cf.fixer_output("k1", card)
    p5 = next(o for o in answer["obligations"] if o["claim_id"] == "P5")["statements"][0]["conservation"]
    # FIX-11: a numeric accuracy cites the card claim that states it (k1's nesting claim).
    p5["dyck"] = [{"depth": 16, "accuracy": 0.5, "claim": "capability.3"}]
    h = Harness(tmp_path, "k1", fixer=answer)
    h.m()
    record = h.slot("M", "P5")
    assert record["status"] == "failed" and "P2 met the threshold on dyck at depth=16" in record["detail"]


def test_p5_violating_a_conservation_law_fails_with_its_counterexample(tmp_path):
    # FIX-16: the carried-state budget is the card's (resources.architecture.state_bits), and each P5 entry's
    # state_bits is derived from it. Here k1 states a 32-bit carried state while its numbers claim copy at 0.99 for L
    # up to 64 (128 bits at D = 4); the fixer copies the card's constant budget as the statement's state_bits.
    card = copy.deepcopy(load_cards(("k1",))["k1"])
    card["resources"]["architecture"]["state_bits"] = 32
    answer = cf.fixer_output("k1", card)
    p5 = next(o for o in answer["obligations"] if o["claim_id"] == "P5")["statements"][0]["conservation"]
    p5["state_bits"] = 32
    h = Harness(tmp_path, "k1", fixer=answer, card=card)
    h.m()
    assert not any("state_bits" in p for p in h.frozen()["first_problems"])
    assert h.slot("M", "P5")["status"] == "failed"
    assert "obligation_failed:P5" in h.verdict("M")["kill_codes"]
    check = next(c for c in h.verdict("M")["checks"].values() if c["form"] == "conservation")
    assert check["status"] == "violated"


# ----- the target gate ----------------------------------------------------------------------------------------------

def test_the_target_gate_fails_the_quadratic_card_at_m_when_enabled(tmp_path):
    h = Harness(tmp_path, "k1", target=True)
    assert h.m() is None
    verdict = h.verdict("M")
    assert verdict["verdict"] == "fail" and "obligation_failed:target" in verdict["kill_codes"]
    assert verdict["target"]["applied"] and verdict["target"]["cost_exponent"] == pytest.approx(2.0)
    assert verdict["profile"]["passed"]   # the profile itself holds: the target is a separate gate
    reasons = h.store.get_card(h.card["id"])["status"]["verdicts"]["M"]["reasons"]
    assert any(r.startswith("target gate: total cost grows as n^2") for r in reasons)


def test_the_target_gate_passes_a_linear_card_and_does_not_decide_when_off(tmp_path):
    h = Harness(tmp_path / "on", "k2", target=True)
    h.m()
    assert h.verdict("M")["target"]["status"] == "pass"
    assert "obligation_failed:target" not in h.verdict("M")["kill_codes"]
    off = Harness(tmp_path / "off", "k1", target=False)
    off.m()
    assert off.verdict("M")["target"]["status"] == "fail" and off.verdict("M")["target"]["applied"] is False
    assert "obligation_failed:target" not in off.verdict("M")["kill_codes"]


# ----- stage C keeps the slots ------------------------------------------------------------------------------------

def test_stage_c_recomputes_the_slots_and_records_the_profile(tmp_path):
    h = Harness(tmp_path, "k3")
    h.m()
    h.c()
    verdict = h.verdict("C")
    assert verdict["profile"]["slots"]["P2.nesting"]["status"] == "failed"
    assert verdict["profile"]["slots"]["P3"]["status"] == "proven"
    assert "obligation_failed:P2.nesting" in verdict["kill_codes"]
    match = next(c for c in h.llm.calls if c["prompt"] == "statement_match")
    assert "profile P2.nesting: construction, nesting" in match["user"]


# ----- configuration ------------------------------------------------------------------------------------------------

def test_d1_defaults_and_validation(tmp_path):
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    assert cfg["math"]["profile"] is True
    # FIX-17 (decision C17): the coupled bounds join the D1 keys.
    assert cfg["target"] == {"enabled": True, "max_cost_exponent": 1.5, "state_growth": "log",
                             "coupled_max_cost_exponent": 1.5, "coupled_max_state_exponent": 0.5}
    old = load_run_config(ROOT / "hybrid" / "config" / "run2.toml")
    assert old["math"]["profile"] is False and old["target"]["enabled"] is False
    text = (ROOT / "hybrid" / "config" / "run.toml").read_text()
    for bad, message in ((("state_growth = \"log\"", "state_growth = \"sublinear\""), "target.state_growth"),
                         (("\nmax_cost_exponent = 1.5", "\nmax_cost_exponent = 0"), "target.max_cost_exponent"),
                         (("coupled_max_cost_exponent = 1.5", "coupled_max_cost_exponent = 0"),
                          "target.coupled_max_cost_exponent"),
                         (("coupled_max_state_exponent = 0.5", "coupled_max_state_exponent = true"),
                          "target.coupled_max_state_exponent"),
                         (("profile = true", "profile = false"), "target.enabled needs math.profile")):
        path = tmp_path / f"run-{len(message)}.toml"
        path.write_text(text.replace(*bad))
        with pytest.raises(ValueError, match=message):
            load_run_config(path)


# ----- the switch: math.profile on and off ---------------------------------------------------------------------------

def test_with_the_profile_off_m_keeps_the_phase3_wiring(tmp_path):
    h = tms.Harness(tmp_path, mf.math_script())
    assert h.ctx.cfg["math"]["profile"] is False
    assert h.m() == "C"
    fixer = next(c for c in h.llm.calls if c["prompt"] == "statement_fixer")
    assert "THE TRANSFORMER PROFILE" not in fixer["user"] and "PROFILE TEMPLATE" not in fixer["user"]
    assert not any(f"- {slot}: profile {slot}: " in fixer["user"] for slot in profile.SLOTS)
    assert "limits" not in fixer["schema"]["properties"]
    prover = next(c for c in h.llm.calls if c["prompt"] == "prover")
    assert "PROFILE OBLIGATIONS" not in prover["user"]
    verdict = h.verdict("M")
    assert "profile" not in verdict and "target" not in verdict
    assert not any(profile.is_slot(c["claim_id"]) for c in verdict["claims"])
    assert not any(code.endswith(tuple(f":{slot}" for slot in profile.SLOTS)) for code in verdict["kill_codes"])
    record = json.loads((h.run_dir / h.events("math_statements_frozen")[0]["data"]["path"]).read_text())
    assert not record.get("profile")


def test_with_the_profile_on_the_same_card_is_held_to_every_slot(tmp_path):
    # The Phase 3 fixer output covers the card's own claims only: the host asks once more for the slots, then
    # every slot is left unproven and the card cannot pass M. (The schema makes "limits" and, since FIX-4,
    # "architecture" required, so the answer carries an empty list and the calibration contract.)
    h = tms.Harness(tmp_path, mf.math_script(fixer=dict(mf.fixer_output(), limits=[],
                                                        architecture=dict(cf.ARCHITECTURE))))
    h.ctx.cfg["math"]["profile"] = True
    # FIX-16: stage M first gates the card on its own resources; the Phase 3 card states K2's (complete for its
    # shapes), so the fixer is reached and the profile, not the gate, decides.
    card = copy.deepcopy(h.store.get_card(mf.CARD))
    # FIX-18: and K2's stated P3 limits, so the limit gate does not stop it either.
    card.update(resources=cf.card_resources_of(None, "k2"), limits=load_cards(("k2",))["k2"]["limits"],
                version=h.store.next_version(mf.CARD))
    h.store.put_card(card)
    assert h.m() is None
    fixers = [c for c in h.llm.calls if c["prompt"] == "statement_fixer"]
    assert len(fixers) == 2 and "PROFILE TEMPLATE" in fixers[0]["user"]
    assert "profile slots with no obligation" in fixers[1]["user"]
    verdict = h.verdict("M")
    assert verdict["verdict"] != "pass"
    assert {f"unproven:{slot}" for slot in profile.SLOTS} <= set(verdict["kill_codes"])
    assert verdict["profile"]["passed"] is False
    assert {entry["status"] for entry in verdict["profile"]["slots"].values()} == {"not_formalized"}
    assert verdict["target"]["status"] == "not_applicable" and verdict["target"]["applied"] is False


# ----- resume ------------------------------------------------------------------------------------------------------

def test_a_resumed_m_with_the_profile_reuses_the_frozen_slots_and_limits(tmp_path):
    h = Harness(tmp_path, "k2")
    h.llm.fail = {"prover": TransportError("down")}
    with pytest.raises(StageError):
        h.m()
    first = h.frozen()
    h.llm.fail = {}
    h.m()
    assert [c["prompt"] for c in h.llm.calls].count("statement_fixer") == 1
    assert len(list(h.ledger.iter_events("math_statements_frozen"))) == 1
    assert h.frozen() == first and first["profile"]["limits"] == cf.LIMITS["k2"]
    block = h.verdict("M")["profile"]
    assert block["passed"] and set(block["slots"]) == set(profile.SLOTS)


def test_a_resumed_m_keeps_the_profile_it_froze_when_the_fixer_would_answer_differently(tmp_path):
    # The second fixer answer drops a slot; it is never asked for, because the frozen record stands.
    card = load_cards(("k1",))["k1"]
    h = Harness(tmp_path, "k1")
    h.llm.fail = {"prover": TransportError("down")}
    with pytest.raises(StageError):
        h.m()
    h.llm.fail = {}
    h.llm.script["statement_fixer"] = cf.fixer_output("k1", card, drop_slots=("P5",))
    h.m()
    assert [c["prompt"] for c in h.llm.calls].count("statement_fixer") == 1
    assert h.slot("M", "P5")["status"] == "observed"


# ----- resume keeps the profile and the target gate ------------------------------------------------------------------

def _math_cfg(profile_on, target_on, **target):
    cfg = orch_config(budgets__max_calls=1)
    cfg["s3"]["mode"] = "math"
    cfg["math"]["profile"] = profile_on
    cfg["target"].update(enabled=target_on, **target)
    return cfg


def _orchestrator(run_dir, tmp_path, cfg, resume):
    return make_orchestrator(run_dir, cfg=cfg, resume=resume, script=mf.math_script(),
                             math=MathRunner(mf.MathExecutor(tmp_path / "exec"), mf.IMAGE, run_dir))[0]


def test_a_math_run_records_the_profile_and_target_and_resumes_only_with_them(tmp_path):
    run_dir = (tmp_path / "run-20261013T000000Z").resolve()
    orch = _orchestrator(run_dir, tmp_path, _math_cfg(True, True), False)
    asyncio.run(orch.run())
    start = next(Ledger(run_dir).iter_events("run_start"))["data"]
    assert start["math_profile"] is True
    assert start["target"] == {"enabled": True, "max_cost_exponent": 1.5, "state_growth": "log",
                               "coupled_max_cost_exponent": 1.5, "coupled_max_state_exponent": 0.5}
    with pytest.raises(ValueError, match="started with math.profile = true"):
        _orchestrator(run_dir, tmp_path, _math_cfg(False, False), True)
    with pytest.raises(ValueError, match=r"started with \[target\]"):
        _orchestrator(run_dir, tmp_path, _math_cfg(True, False), True)
    with pytest.raises(ValueError, match=r"started with \[target\]"):
        _orchestrator(run_dir, tmp_path, _math_cfg(True, True, max_cost_exponent=2.0), True)
    assert _orchestrator(run_dir, tmp_path, _math_cfg(True, True), True).resume


def test_a_math_run_from_before_phase_4_resumes_without_the_profile(tmp_path):
    run_dir = (tmp_path / "run-20261012T000000Z").resolve()
    Ledger(run_dir).append("run_start", "orchestrator", {"run_id": run_dir.name, "resume": False,
                                                          "s3_mode": "math"})
    with pytest.raises(ValueError, match="started with math.profile = false"):
        _orchestrator(run_dir, tmp_path, _math_cfg(True, False), True)
    with pytest.raises(ValueError, match=r"started with \[target\]"):
        cfg = _math_cfg(True, True)
        cfg["math"]["profile"] = False   # not a loadable config, but the gate check comes after the profile check
        _orchestrator(run_dir, tmp_path, cfg, True)
    assert _orchestrator(run_dir, tmp_path, _math_cfg(False, False, max_cost_exponent=2.0), True).resume


def test_a_judges_run_ignores_the_profile_and_target_settings_on_resume(tmp_path):
    run_dir = (tmp_path / "run-20261011T000000Z").resolve()
    Ledger(run_dir).append("run_start", "orchestrator", {"run_id": run_dir.name, "resume": False,
                                                          "s3_mode": "judges"})
    cfg = orch_config()
    cfg["math"]["profile"] = True
    cfg["target"]["enabled"] = True
    assert make_orchestrator(run_dir, cfg=cfg, resume=True)[0].resume


# ----- each slot names the tier it reached ---------------------------------------------------------------------------

def test_p3_by_a_bounded_z3_property_reaches_the_smt_bounded_tier(tmp_path):
    card = load_cards(("k2",))["k2"]
    answer = with_obligation(cf.fixer_output("k2", card), "P3", statements=[dict(mf.Z3)])
    h = Harness(tmp_path, "k2", fixer=answer)
    h.m()
    record = h.slot("M", "P3")
    assert record["status"] == "proven_up_to_bound" and record["tier"] == "smt_bounded"
    block = h.verdict("M")["profile"]
    assert block["slots"]["P3"]["tier"] in block["slots"]["P3"]["accepted_tiers"]
    assert block["passed"]


def test_every_slot_reports_the_tier_of_the_form_it_was_stated_in(tmp_path):
    h = Harness(tmp_path, "k2")
    h.m()
    block = h.verdict("M")["profile"]
    record = h.frozen()
    for obligation in record["obligations"]:
        slot = obligation["claim_id"]
        if not profile.is_slot(slot):
            continue
        tiers_stated = {profile.FORM_TIER[s["form"]] for s in obligation["statements"]}
        assert block["slots"][slot]["tier"] in tiers_stated, slot
        assert block["slots"][slot]["accepted_forms"] == list(profile.SLOT_FORMS[slot])
