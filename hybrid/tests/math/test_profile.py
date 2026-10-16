"""The transformer profile's fixed shape and host checks (hybrid/math/profile.py), and the Phase 4 statement forms in
spec_checks.py (declared state words of a construction, conservation numbers)."""
import copy

import pytest

from hybrid.math import constructions, profile, references, spec_checks, tiers
from hybrid.math.runner import LIBRARY_CAPS
from hybrid.tests.research import calibration_fakes as cf


def test_every_slot_names_its_forms_and_the_tiers_it_can_reach():
    assert profile.SLOTS == ("P1", "P2.copy", "P2.recall", "P2.nesting", "P2.composition", "P3", "P4.copy",
                             "P4.recall", "P4.nesting", "P4.composition", "P5")
    assert profile.accepted_tiers("P1") == [tiers.EXECUTED]
    assert all(profile.accepted_tiers(slot) == [tiers.EXECUTED] for slot in (*profile.P2_SLOTS, *profile.P4_SLOTS))
    assert profile.accepted_tiers("P3") == sorted([tiers.LEAN, tiers.SYMPY_EXACT, tiers.SMT_BOUNDED])
    assert profile.accepted_tiers("P5") == [tiers.EXECUTED]
    assert not any(tiers.ARGUED in profile.accepted_tiers(slot) for slot in profile.SLOTS)
    claims = profile.profile_claims()
    assert [c["id"] for c in claims] == list(profile.SLOTS)
    assert all(c["text"].startswith(f"profile {c['id']}: ") for c in claims)
    text = profile.template_text(LIBRARY_CAPS)
    assert all(f"- {slot} (forms: " in text for slot in profile.SLOTS)


@pytest.mark.parametrize("slot,form", [("P1", "construction"), ("P2.copy", "cost"), ("P3", "construction"),
                                       ("P3", "simulation"), ("P4.recall", "construction"),
                                       ("P5", "sympy")])
def test_a_slot_refuses_forms_it_does_not_accept(slot, form):
    problems = profile.statement_problems(slot, form, {})
    assert problems and "accepts only" in problems[0]


def test_p1_needs_an_exponent_and_enough_lengths():
    good = dict(references.REFERENCES["elman_rnn"][1], widths={"d": "d", "rounds": "1"})
    assert profile.statement_problems("P1", "cost", good) == []
    unwidthed = dict(references.REFERENCES["elman_rnn"][1])
    assert any("widths" in p for p in profile.statement_problems("P1", "cost", unwidthed))
    assert any("'depth'" in p for p in profile.statement_problems("P1", "cost", dict(good, widths={
        "d": "depth", "rounds": "1"})))
    assert profile.statement_problems("P1", "cost", dict(good, ns=[16, 32, 64, 128])) == []
    assert any("exponent" in p for p in profile.statement_problems("P1", "cost", dict(good, exponent=None)))
    assert any("ns must" in p for p in profile.statement_problems("P1", "cost", dict(good, ns=[64, 128, 256])))
    assert any("ns must" in p for p in profile.statement_problems("P1", "cost", dict(good, ns=[64, 70, 80, 90])))


# FIX-16: the scripted fixer leaves count, threshold, widths and state_words to the host, which fills them in from the
# card's resources and the protocol before freezing; these checks see the frozen statement (cf.frozen).
def binding_card() -> dict:
    """K2's resources with a binding P4 case in place of composition (P4.composition takes either task)."""
    resources = copy.deepcopy(cf.card_resources_of(None, "k2"))
    resources["learning"]["binding"] = {"size": 2, "examples": 1200, "d": "64", "rounds": "1"}
    del resources["learning"]["composition"]
    return {"resources": resources}


def test_p2_needs_its_task_the_required_sizes_the_threshold_and_state_words():
    good = cf.frozen("P2.nesting", cf.construction("dyck"))
    assert profile.statement_problems("P2.nesting", "construction", good) == []
    assert "task must be dyck" in profile.statement_problems("P2.nesting", "construction",
                                                             dict(good, task="copy"))[0]
    shrunk = profile.statement_problems("P2.nesting", "construction", dict(good, sizes=[2, 4]))
    assert any("missing [8, 16]" in p for p in shrunk)
    assert any("threshold" in p for p in profile.statement_problems("P2.nesting", "construction",
                                                                    dict(good, threshold=0.9)))
    assert any("state_words" in p for p in profile.statement_problems("P2.nesting", "construction",
                                                                      dict(good, state_words=None)))
    assert profile.statement_problems("P2.composition", "construction",
                                      cf.frozen("P2.composition", cf.construction("binding"))) == []


def test_p4_needs_the_solvable_case_and_three_seeds():
    good = cf.frozen("P4.nesting", cf.learnability("dyck"))
    assert good["size"] == 2
    assert profile.statement_problems("P4.nesting", "learnability", good) == []
    assert any("at most 2" in p for p in profile.statement_problems("P4.nesting", "learnability",
                                                                    dict(good, size=5)))
    assert any("3 distinct seeds" in p for p in profile.statement_problems("P4.nesting", "learnability",
                                                                           dict(good, seeds=[0, 1])))



@pytest.mark.parametrize("form,spec,tier", [
    ("lean", "import Mathlib\n\ndef Claim : Prop := True\n", tiers.LEAN),
    ("sympy", {"lhs": "2**(n + 4)", "rhs": "16*2**n"}, tiers.SYMPY_EXACT),
    ("z3", {"variables": {"m": {"sort": "int", "lo": 1, "hi": 8}}, "hypotheses": [], "property": "m + 0 == m"},
     tiers.SMT_BOUNDED),
])
def test_p3_accepts_each_derivation_form_and_names_its_tier(form, spec, tier):
    assert profile.statement_problems("P3", form, spec) == []
    assert profile.FORM_TIER[form] == tier and tier in profile.accepted_tiers("P3")


@pytest.mark.parametrize("slot", profile.SLOTS)
def test_the_template_names_each_slots_forms_and_tiers(slot):
    line = next(l for l in profile.template_text(LIBRARY_CAPS).splitlines() if l.startswith(f"- {slot} (forms: "))
    assert f"forms: {', '.join(profile.SLOT_FORMS[slot])}; tier reached: {', '.join(profile.accepted_tiers(slot))})" \
        in line
    assert profile.SLOT_TEXT[slot] in line


def test_p2_and_p4_slots_refuse_a_task_of_another_slot_and_a_lower_threshold():
    assert profile.statement_problems("P2.composition", "construction",
                                      cf.frozen("P2.copy", cf.construction("copy"))) == [
        "P2.composition: task must be composition or binding, not copy"]
    assert profile.statement_problems("P4.recall", "learnability", cf.frozen("P4.copy", cf.learnability("copy"))) == [
        "P4.recall: task must be recall, not copy"]
    low = dict(cf.frozen("P4.recall", cf.learnability("recall")), threshold=0.9)
    assert any("threshold must be at least 0.99" in p for p in profile.statement_problems("P4.recall", "learnability",
                                                                                         low))
    card = binding_card()
    assert profile.statement_problems("P4.composition", "learnability",
                                      cf.frozen("P4.composition", cf.learnability("binding", card=card), card)) == []
    k3 = {"form": "conservation", "conservation": cf.p5_numbers("k3")}
    assert profile.statement_problems("P5", "conservation", cf.frozen("P5", k3, key="k3")) == []
    assert profile.statement_problems("P1", "cost", "n**2") == ["P1: the statement must be an object"]

def obligations(key="k1", drop=()):
    """The scripted fixer's answer for calibration card ``key`` as the host freezes it (FIX-16: protocol, card and
    derived values filled in from the card's resources)."""
    from hybrid.research.calibration import load_cards
    from hybrid.research.stages.math_contracts import normalized_answer
    card = load_cards((key,))[key]
    answer, problems, arch_problems = normalized_answer(cf.fixer_output(key, card, drop_slots=drop), card)
    assert problems == {} and arch_problems == []
    return answer


def test_record_problems_accepts_a_complete_profile():
    answer = obligations()
    assert profile.record_problems(answer["obligations"], answer["limits"], answer["architecture"]) == []
    assert profile.record_problems(answer["obligations"], answer["limits"]) == [
        "architecture must be {hidden, precision_bits} with an optional state_bits"]


def test_record_problems_names_missing_slots_variants_coverage_and_limits():
    answer = obligations(drop=("P4.copy", "P3"))
    problems = profile.record_problems(answer["obligations"], [])
    assert any("['P3', 'P4.copy']" in p for p in problems)
    assert any("P3: state at least one limit" in p for p in problems)
    answer = obligations()
    composition = next(o for o in answer["obligations"] if o["claim_id"] == "P2.composition")
    composition["statements"] = composition["statements"][:1]
    p5 = next(o for o in answer["obligations"] if o["claim_id"] == "P5")["statements"][0]["conservation"]
    p5["dyck"] = [{"depth": 8, "accuracy": 0.99}]
    p5["sample"] = [dict(p5["sample"][0], examples=7)]
    problems = profile.record_problems(answer["obligations"], answer["limits"])
    assert any("composition construction and a binding construction" in p for p in problems)
    assert any("dyck entry at every P2 size" in p and "missing depth = [2, 4, 16]" in p for p in problems)
    # FIX-16: the P4 example counts are k1's stated ones (resources.learning), 3000, 3000, 1500 and 4000.
    assert any("missing m = [1500, 3000, 4000]" in p for p in problems)


def test_consistency_fails_p3_when_p2_beat_the_limit_and_p5_when_it_ignores_p2():
    p2 = {"dyck": {2: True, 4: True, 8: True, 16: False}, "copy": {8: True, 32: True, 64: True}}
    limits = [{"task": "dyck", "size": 8, "resources": "2^D states", "why": "finite state"},
              {"task": "other", "size": None, "resources": "-", "why": "parity"}]
    p5 = [{"state_bits": 10**6, "copy": [{"D": 4, "L": 64, "accuracy": 0.99}],
           "dyck": [{"depth": 4, "accuracy": 0.99}]}]
    failures = profile.consistency(limits, p2, p5)
    assert "contradicted by the P2 construction, which met the threshold at [8]" in failures["P3"]
    assert "dyck at depth=8" in failures["P5"] and "copy" not in failures["P5"]
    assert profile.consistency(limits[1:], p2, [dict(p5[0], dyck=[{"depth": 8, "accuracy": 0.99}])]) == {}


def test_p2_sizes_read_the_harness_results_of_decided_checks_only():
    statement = {"form": "construction", "spec": {"task": "dyck"}}
    decided = {"decided": True, "details": {"results": [{"size": 2, "passed": True}, {"size": 8, "passed": False}]}}
    assert profile.p2_sizes([(statement, decided)]) == {"dyck": {2: True, 8: False}}
    assert profile.p2_sizes([(statement, dict(decided, decided=False)), (statement, None)]) == {}


# ----- spec_checks: declared state words and conservation numbers ----------------------------------------------------

def test_a_construction_within_its_declared_state_words_holds_and_one_over_them_fails():
    spec = {"task": "dyck", "sizes": [2, 4, 8], "count": 20, "state_words": "depth + 1"}
    source = references.construction_source(references.DYCK_STACK)
    held = spec_checks.audited_construction(spec, source)
    assert held["status"] == "discharged" and "within the declared depth + 1" in held["evidence"]
    over = spec_checks.audited_construction(dict(spec, state_words="depth"), source)
    assert over["status"] == "violated" and over["counterexample"]["check"] == "state_words"
    assert all("declared_state_words" in row for row in over["results"])


def test_state_words_must_be_a_formula_in_the_task_size():
    problems = spec_checks.statement_problems("construction", {"task": "copy", "sizes": [8], "state_words": "K + 1"})
    assert any("unknown name 'K'" in p for p in problems)
    assert spec_checks.statement_problems("construction", {"task": "copy", "sizes": [8], "state_words": "2*L"}) == []
    assert constructions.SIZE_NAME["copy"] == "L"


def test_conservation_statements_must_be_numbers_the_library_reads():
    k1 = {"form": "conservation", "conservation": cf.p5_numbers("k1")}
    assert spec_checks.statement_problems("conservation", cf.frozen("P5", k1, key="k1")) == []
    assert spec_checks.statement_problems("conservation", {}) == ["conservation: the statement holds no numbers"]
    assert any("lacks" in p for p in spec_checks.statement_problems("conservation", {"copy": [{"D": 4}]}))
    assert any("unknown claim sections" in p for p in spec_checks.statement_problems("conservation",
                                                                                    {"parity": []}))
    assert spec_checks.statement_problems("conservation", {"state_bits": 8}) == [
        "conservation: no law applies to these numbers"]


def test_learnability_statements_are_checked_by_the_learnability_module():
    assert spec_checks.statement_problems("learnability", cf.learnability("copy", 3)["learnability"]) == []
    assert spec_checks.statement_problems("learnability", {"task": "copy"})
