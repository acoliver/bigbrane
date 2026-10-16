"""FIX-18: card-stated P3 limits and the Lean lint inside the fixer's revision rounds (DECISIONS.md L,
results/phase4-20261013T073748Z/FIX-18.md).

Cause: no calibration card stated a limit, so the statement fixer invented the P3 limit in every run (calibration
20261015T200147Z, K1: a positional table in run 1, an unseen next bit in runs 2 and 3). In K1 run 3 the invented P3
statement first appeared in the fixer's one retry and bound a variable named `prefix`, a Lean 4 keyword; the lint
refused it at freeze with no round left, and P3 was not shown."""
import asyncio
import copy
import json

import pytest

from hybrid.math import lean_check, profile, spec_checks
from hybrid.research import calibration as cal
from hybrid.research import planted
from hybrid.research.ledger import Ledger
from hybrid.research.mathrecords import formal_claims
from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.stages import card_resources as cr
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.falsify import claim_items
from hybrid.research.store import Store
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_profile_stage import Harness

CARDS = cal.load_cards()
# The P3 statement of K1 run 3 (runs/calibration-math-20261015T200147Z/k1-run3, dropped O13.a), as the fixer wrote it.
RUN3_STATEMENT = ("import Mathlib\n\ndef Claim : Prop :=\n  ∀ predict : List Bool → Bool,\n    ∀ prefix : List Bool,\n"
                  "      ∃ nextBit : Bool, predict prefix ≠ nextBit\n")
RENAMED = RUN3_STATEMENT.replace("prefix", "seen")
MODEL_NAMES = ("transformer", "lstm", "hmm", "hidden markov", "n-gram", "ngram", "attention", "rnn", "recurrent")


def without_limits(card: dict) -> dict:
    card = copy.deepcopy(card)
    card.pop("limits")
    return card


def claim_ids(card):
    return [c["id"] for c in formal_claims(claim_items(card))] + list(profile.SLOTS)


def texts(card):
    claims = formal_claims(claim_items(card)) + profile.profile_claims()
    return {c["id"]: c["text"] for c in claims}


def lean_p3(answer: dict, text: str) -> dict:
    answer = copy.deepcopy(answer)
    obligation = next(o for o in answer["obligations"] if o["claim_id"] == "P3")
    obligation["statements"] = [{"form": "lean", "lean": text}]
    return answer


# ----- the card field and the gate ----------------------------------------------------------------------------------

def test_the_schema_accepts_stated_limits_and_old_cards_without_them():
    for card in CARDS.values():
        validate("theory_card", card)
        validate("theory_card", without_limits(card))
    bad = copy.deepcopy(CARDS["k1"])
    bad["limits"][0]["kind"] = "output_vocabulary"
    with pytest.raises(SchemaViolation):
        validate("theory_card", bad)
    bad = copy.deepcopy(CARDS["k1"])
    del bad["limits"][0]["why"]
    with pytest.raises(SchemaViolation):
        validate("theory_card", bad)


def test_every_calibration_card_states_a_limit_and_passes_the_gate():
    for key, card in CARDS.items():
        assert card["limits"] and cr.missing(card) == [], key


def test_a_card_without_a_stated_limit_is_incomplete():
    items = cr.missing(without_limits(CARDS["k2"]))
    assert len(items) == 1 and items[0].startswith("limits: the card states no P3 limit")
    card = copy.deepcopy(CARDS["k3"])
    card["limits"] = []
    assert cr.missing(card)[0].startswith("limits:")
    card["limits"] = [dict(CARDS["k3"]["limits"][0], kind="readout", size=0, why=" ")]
    assert cr.missing(card) == ["limits[0].size (an integer of at least 1, or null)", "limits[0].why"]
    card["mechanism"] = dict(card["mechanism"], readout=[])
    assert "limits[0].kind: readout, but the card's mechanism states no readout entry" in cr.missing(card)


def test_calibration_refuses_a_fixture_without_limits_before_any_call(tmp_path):
    with pytest.raises(ValueError, match=r"card_incomplete\): k2 lacks limits: the card states no P3 limit"):
        cal.check_complete({"k2": without_limits(CARDS["k2"])})


def test_stage_m_ends_a_card_without_limits_with_card_incomplete_and_no_fixer_call(tmp_path):
    h = Harness(tmp_path, "k2", card=without_limits(CARDS["k2"]))
    h.m()
    verdict = h.verdict("M")
    assert verdict["verdict"] == "undetermined" and verdict["kill_codes"] == ["card_incomplete"]
    assert verdict["missing_resources"][0].startswith("limits:")
    assert not [c for c in h.llm.calls if c["prompt"] == "statement_fixer"]


def test_the_author_is_asked_for_limits_with_the_resources():
    schema = cr.request_schema()
    assert schema["properties"]["limits"]["items"]["required"] == ["task", "size", "resources", "why", "kind"]
    assert "limits" not in schema["required"]
    assert "`limits`" in (cal.ROOT / "hybrid" / "research" / "prompts" / "card_resources.md").read_text()
    for name in ("explorer.md", "explorer_escape.md"):
        assert "FIX-18" in (cal.ROOT / "hybrid" / "research" / "prompts" / name).read_text(), name


def test_the_planted_override_takes_the_limits_stated_in_r(tmp_path):
    ledger = Ledger(tmp_path / "run")
    store = Store(tmp_path / "run", ledger)
    sealed = without_limits(CARDS["k2"])
    store.put_card(dict(sealed, version=1))
    store.put_card(dict(copy.deepcopy(CARDS["k2"]), version=2))
    card, _ = planted.override_card(sealed, sealed["id"], store)
    assert card["limits"] == CARDS["k2"]["limits"]


# ----- the fixer copies the card's limits and invents none ------------------------------------------------------------

def test_the_fixer_is_shown_the_card_limits(tmp_path):
    h = Harness(tmp_path, "k1")
    h.m()
    call = next(c for c in h.llm.calls if c["prompt"] == "statement_fixer")
    assert "CARD LIMITS (FIX-18)" in call["user"] and CARDS["k1"]["limits"][0]["why"] in call["user"]
    assert "P3 limits come from the card" in call["system"]
    record = h.frozen()
    assert record["profile"]["limits"] == cf.LIMITS["k1"] and record["card_limits"] == CARDS["k1"]["limits"]
    validate("math_statement", record)


def test_an_invented_limit_is_fed_back_and_refused_at_freeze(tmp_path):
    invented = {"task": "dyck", "size": 8, "resources": "more rounds", "why": "an unseen next bit",
                "basis": dict(cf.LIMITS["k1"][0]["basis"])}
    answer = cf.fixer_output("k1", CARDS["k1"])
    answer["limits"] = cf.LIMITS["k1"] + [invented]
    h = Harness(tmp_path, "k1", fixer=answer)
    h.m()
    fixers = [c for c in h.llm.calls if c["prompt"] == "statement_fixer"]
    assert len(fixers) == 2 and "P3 limit 2 (task dyck, size 8, part update) is not one of the card's" in fixers[1]["user"]
    record = h.frozen()
    assert record["profile"]["limits"] == cf.LIMITS["k1"]
    assert any(d.get("limit") == invented and "Invent no limit" in d["problems"][0] for d in record["dropped"])
    assert any(o["claim_id"] == "P3" for o in record["obligations"])


def test_a_p3_resting_only_on_an_invented_limit_is_not_formalized(tmp_path):
    answer = cf.fixer_output("k1", CARDS["k1"])
    # Not K1's stated limit (task other, no size, update): a limit on copy from L = 128.
    answer["limits"] = [{"task": "copy", "size": 128, "resources": "an unseen bit", "why": "information",
                         "basis": dict(cf.LIMITS["k1"][0]["basis"])}]
    h = Harness(tmp_path, "k1", fixer=answer)
    h.m()
    record = h.frozen()
    assert record["profile"]["limits"] == []
    assert not any(o["claim_id"] == "P3" for o in record["obligations"])
    assert any(n["claim_id"] == "P3" and "not one of the card's stated limits" in n["why"]
               for n in record["not_formalized"])
    assert "unproven:P3" in h.verdict("M")["kill_codes"]


def test_a_left_out_card_limit_and_changed_text_are_problems_and_the_card_text_is_frozen():
    card = CARDS["k2"]
    assert cr.limit_copy_problems(card, []) == [
        (None, "card limit 1 (task recall, size 4225, part state) is not in the answer's limits; copy every limit "
               "the card states and formalize it in P3")]
    changed = [dict(cf.LIMITS["k2"][0], why="a store of 32 d bits")]
    problems = cr.limit_copy_problems(card, changed)
    assert len(problems) == 1 and problems[0][0] == 0 and "why must be the card's text" in problems[0][1]
    assert cr.frozen_limits(card, changed) == cf.LIMITS["k2"]
    answer = cf.fixer_output("k2", card)
    answer["limits"] = changed
    assert any("why must be the card's text" in p
               for p in mc.fixer_problems(answer, card, claim_ids(card), 40, texts(card)))
    assert mc.freeze(answer, card, claim_ids(card), 40, texts(card))["profile"]["limits"] == cf.LIMITS["k2"]


def test_a_card_from_before_fix18_keeps_the_fixers_limits():
    card = without_limits(CARDS["k2"])
    assert cr.limit_copy_problems(card, cf.LIMITS["k2"]) == [] and cr.invented_limits(card, cf.LIMITS["k2"]) == []
    assert cr.frozen_limits(card, cf.LIMITS["k2"]) == cf.LIMITS["k2"]


# ----- the K1 to K4 limits --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["k1", "k2", "k3", "k4"])
def test_each_calibration_limit_passes_the_profile_checks(key):
    card = CARDS[key]
    limits = cf.LIMITS[key]
    assert mc.limit_problems(limits, card) == []
    assert cr.limit_copy_problems(card, limits) == []
    answer = cf.fixer_output(key, card)
    assert not [p for p in mc.fixer_problems(answer, card, claim_ids(card), 40, texts(card)) if "limit" in p]
    assert profile.record_problems(answer["obligations"], answer["limits"], answer["architecture"]) == []


def p2_at_required_sizes(passed: bool) -> dict:
    return {task: {size: passed for size in sizes} for task, sizes in profile.REQUIRED_SIZES.items()}


def test_k1_and_k2_limits_sit_outside_the_protocol_and_are_consistent_with_passing_p2():
    for key in ("k1", "k2"):
        limit = CARDS[key]["limits"][0]
        assert limit["task"] == "other" or limit["size"] > max(profile.REQUIRED_SIZES[limit["task"]]), key
        assert "P3" not in profile.consistency(cf.LIMITS[key], p2_at_required_sizes(True), []), key


def test_k3_and_k4_limits_explain_their_p2_failures():
    for key in ("k3", "k4"):
        limit = CARDS[key]["limits"][0]
        assert limit["task"] in profile.REQUIRED_SIZES and limit["size"] <= max(profile.REQUIRED_SIZES[limit["task"]])
        assert "P3" not in profile.consistency(cf.LIMITS[key], p2_at_required_sizes(False), []), key
        # A P2 pass at the required sizes would contradict the limit: it is the reason those sizes fail.
        assert "P3" in profile.consistency(cf.LIMITS[key], p2_at_required_sizes(True), []), key


@pytest.mark.parametrize("key, lhs, rhs, tight", [
    # K1: the embed update needs row N + 1 of an N-row table.
    ("k1", "N + 1", "N", None),
    # K2: 4K bits of values from V = 16 against 2d = 528 numbers of 32 bits at d = 264, from K = 4225.
    ("k2", "4*(4224 + n)", "16896", "4*(4223 + n)"),
    # K3: depth-D stacks need D bits against log2 16 = 4, from depth 5.
    ("k3", "4 + n", "4", "3 + n"),
    # K4: L log2 4 = 2L bits against 2 ceiling(log2 5) = 6, from L = 4.
    ("k4", "2*(3 + n)", "6", "2*(2 + n)"),
])
def test_each_limit_is_provable_by_the_sympy_tier_at_its_stated_size(key, lhs, rhs, tight):
    symbol = "N" if key == "k1" else "n"
    spec = {"symbols": {symbol: "positive_integer"}, "lhs": lhs, "rhs": rhs, "relation": ">"}
    result = spec_checks.sympy_relation(spec)
    assert result["status"] == "discharged" and result["tier"] == "sympy_exact", result
    if tight:
        # One size below the stated size the counting bound no longer holds: the stated size is the boundary.
        assert spec_checks.sympy_relation(dict(spec, lhs=tight))["status"] == "violated"


def test_the_limits_name_no_model():
    for key, card in CARDS.items():
        text = json.dumps(card["limits"]).lower()
        assert not [name for name in MODEL_NAMES if name in text], key


# ----- the Lean lint inside the revision rounds -----------------------------------------------------------------------

def test_the_lint_names_a_keyword_bound_variable():
    assert lean_check.keyword_binders(RUN3_STATEMENT) == ["prefix"]
    assert lean_check.keyword_binders(RENAMED) == []
    assert lean_check.keyword_binders("def Claim : Prop := ∀ (at : ℕ), let from := 3; at = from") == ["at", "from"]
    assert lean_check.keyword_binders("def Claim : Prop := ∀ (n m : ℕ) (h : n < m), fun x => x = x") == []
    # The forbidden-construct check stays; the reserved-word message adds the rename.
    problems = mc.lean_statement_problems(RUN3_STATEMENT)
    assert "lean: forbidden construct 'prefix'" in problems
    assert any("bound name 'prefix' is a Lean 4 keyword" in p for p in problems)
    assert mc.lint_problems([f"O6 statement 1: {p}" for p in problems]) == [f"O6 statement 1: {p}" for p in problems]
    assert mc.lean_statement_problems(RENAMED) == []


def lean_harness(tmp_path, answers):
    card = CARDS["k1"]
    script = cf.script("k1", card, fixer=answers)
    script["prover"] = lean_and_scripted(cf.prover_output("k1"))
    h = Harness(tmp_path, "k1", script=script)
    h.ctx.math.executor = _Both(h.executor, mf.MathExecutor(tmp_path / "lean"))
    return h


class _Both:
    """The calibration fake executor, with Lean jobs answered by the Phase 3 Lean fake."""

    def __init__(self, fast, lean):
        self.fast, self.lean = fast, lean

    def run(self, job):
        from pathlib import Path
        from hybrid.math.runner import CHECK_FILE
        if (Path(job.input_dir) / CHECK_FILE).is_file():
            return self.lean.run(job)
        return self.fast.run(job)

    def __getattr__(self, name):
        return getattr(self.fast, name)


def lean_and_scripted(scripted):
    def answer(index, user):
        shown = json.JSONDecoder().raw_decode(mf._section(user, "FROZEN STATEMENTS:\n"))[0]
        lean = [item for item in shown if item["form"] == "lean"]
        rest = [item for item in shown if item["form"] != "lean"]
        attempts = [{"statement_id": item["statement_id"], "proof": mf.GOOD_PROOF, "source": None, "notes": "x"}
                    for item in lean]
        if rest:
            attempts += scripted(index, user)["attempts"]
        return {"attempts": attempts}
    return answer


def test_a_keyword_written_in_the_rewrite_gets_a_repair_round_and_p3_holds(tmp_path):
    base = cf.fixer_output("k1", CARDS["k1"])
    first = cf.fixer_output("k1", CARDS["k1"], drop_slots=("P3",))
    answers = [first, lean_p3(base, RUN3_STATEMENT), lean_p3(base, RENAMED)]
    h = lean_harness(tmp_path, answers)
    h.m()
    fixers = [c for c in h.llm.calls if c["prompt"] == "statement_fixer"]
    assert len(fixers) == 3
    assert "profile slots with no obligation: ['P3']" in fixers[1]["user"]
    assert "bound name 'prefix' is a Lean 4 keyword" in fixers[2]["user"]
    record = h.frozen()
    p3 = next(o for o in record["obligations"] if o["claim_id"] == "P3")
    assert p3["statements"][0]["spec"] == RENAMED
    assert record["lint_problems"] and not [d for d in record["dropped"] if d.get("statement_id", "").startswith("O6")]
    validate("math_statement", record)
    assert h.slot("M", "P3")["status"] == "proven"


def test_a_keyword_in_the_first_answer_is_repaired_in_the_existing_retry(tmp_path):
    base = cf.fixer_output("k1", CARDS["k1"])
    h = lean_harness(tmp_path, [lean_p3(base, RUN3_STATEMENT), lean_p3(base, RENAMED)])
    h.m()
    assert len([c for c in h.llm.calls if c["prompt"] == "statement_fixer"]) == 2
    record = h.frozen()
    assert "lint_problems" not in record
    assert h.slot("M", "P3")["status"] == "proven"


def test_the_repair_round_is_one_round(tmp_path):
    base = cf.fixer_output("k1", CARDS["k1"])
    stuck = lean_p3(base, RUN3_STATEMENT)
    h = lean_harness(tmp_path, [stuck, stuck, stuck, lean_p3(base, RENAMED)])
    h.m()
    assert len([c for c in h.llm.calls if c["prompt"] == "statement_fixer"]) == 3
    record = h.frozen()
    assert any("forbidden construct 'prefix'" in p for d in record["dropped"] for p in d.get("problems", []))
    assert h.slot("M", "P3")["status"] == "not_formalized"
