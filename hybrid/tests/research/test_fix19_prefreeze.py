"""FIX-19: statement-to-claim matching before the freeze, and P3 statements over the card's shared mechanism
definitions (DECISIONS.md M, results/phase4-20261013T073748Z/FIX-19.md).

Cause: in calibration 20261016T051933Z K1 run 2, P3 was Lean-proven at stage M and blocked at stage C. The frozen
statement O10.a was a generic list fact (a list has no entry at its own length) that never mentions the embed update
h0_t = E[x_t] + P[t] the card's limit names; the Sonnet cross-check upheld the matcher's objection. Runs 1 and 3
modeled the update and matched. The match ran only after the freeze, so the fixer never saw the mismatch.

All model calls are fakes."""
import copy
import json

import pytest

from hybrid.math import profile
from hybrid.research import calibration as cal
from hybrid.research.schemas import validate
from hybrid.research.stages import m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import StageError
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_fix18_card_limits import lean_harness

CARDS = cal.load_cards()

# runs/calibration-math-20261016T051933Z, the frozen P3 statements (k1-run2 O10.a was blocked at C).
K1_RUN2 = ("import Mathlib\ndef Claim : Prop := ∀ d : ℕ, ∀ table : List (Fin d → ℝ), table[table.length]? = none ∧ "
           "¬ (∀ t : ℕ, ∃ row : Fin d → ℝ, table[t]? = some row)\n")
K1_RUN1 = ("import Mathlib\n\ndef EmbedDomain (N t : ℕ) : Prop := 1 ≤ t ∧ t ≤ N\n\n"
           "def StreamInDomain (N len : ℕ) : Prop :=\n  ∀ t : ℕ, 1 ≤ t → t ≤ len → EmbedDomain N t\n\n"
           "def Claim : Prop :=\n  (∀ N : ℕ, ¬ EmbedDomain N (N+1)) ∧\n  (∀ N : ℕ, ¬ StreamInDomain N (N+1)) ∧\n"
           "  (¬ ∃ N : ℕ, ∀ len : ℕ, StreamInDomain N len)\n")
K1_RUN3 = ("import Mathlib\n\n"
           "def positionRow {N d : ℕ} (P : Fin N → Fin d → ℝ) (t : ℕ) : Option (Fin d → ℝ) :=\n"
           "  if ht : 1 ≤ t ∧ t ≤ N then\n    some (P ⟨t - 1, by omega⟩)\n  else none\n\n"
           "def embeddedRow {N d : ℕ} (P : Fin N → Fin d → ℝ) (symbol : Fin d → ℝ) (t : ℕ) : Option (Fin d → ℝ) :=\n"
           "  (positionRow P t).map (fun row j => symbol j + row j)\n\n"
           "def Claim : Prop :=\n  ∀ (N d : ℕ) (P : Fin N → Fin d → ℝ) (symbol : Fin d → ℝ),\n"
           "    embeddedRow P symbol (N + 1) = none\n")
K2_RUN1 = ("import Mathlib\n\ndef Claim : Prop :=\n  ∀ K : ℕ, 4225 ≤ K →\n"
           "    ∀ encode : (Fin K → Fin 16) → Fin (2 ^ 16896),\n    ∀ decode : Fin (2 ^ 16896) → (Fin K → Fin 16),\n"
           "      ∃ values : Fin K → Fin 16, decode (encode values) ≠ values\n")
K3_RUN2 = ("import Mathlib\n\ndef Claim : Prop :=\n  (∀ D q : ℕ,\n    ∀ enc : (Fin D → Bool) → Fin q,\n"
           "    ∀ dec : Fin q → (Fin D → Bool),\n    (∀ stack, dec (enc stack) = stack) → 2 ^ D ≤ q) ∧\n"
           "  (∀ enc : (Fin 5 → Bool) → Fin 16,\n    ∀ dec : Fin 16 → (Fin 5 → Bool),\n"
           "    ∃ stack, dec (enc stack) ≠ stack)\n")
K4_RUN2 = ("import Mathlib\ndef Claim : Prop := ∀ L : ℕ, 4 ≤ L → ∀ enc : Fin (4 ^ L) → Fin 64, "
           "∀ dec : Fin 64 → Fin (4 ^ L), ∃ x : Fin (4 ^ L), dec (enc x) ≠ x\n")
# The generic list fact with its list renamed P: it passes the host's name check, so only the match can catch it.
K1_RUN2_RENAMED = K1_RUN2.replace("table", "P")
MECHANISM = {"id": "D1", "kind": "mechanism",
             "text": "The model has symbol embeddings E and a learned position table P with N rows. At position t, "
                     "h = E[x_t] + P[t]; the rounds and the readout are as the card states."}
GENERIC_MARK = "[P.length]? = none"


def with_p3(text: str, *, cite: bool = True, card_key: str = "k1") -> dict:
    """The calibration fixer's answer with P3 as one Lean statement over the shared mechanism definition D1."""
    answer = cf.fixer_output(card_key, CARDS[card_key])
    answer["definitions"] = [copy.deepcopy(MECHANISM)]
    obligation = next(o for o in answer["obligations"] if o["claim_id"] == "P3")
    obligation["statements"] = [{"form": "lean", "lean": text}]
    obligation["definitions"] = ["D1"] if cite else []
    return answer


def objecting_to(*marks: str):
    """The statement matcher: an anchored objection (direction different, domains) to every statement whose formal
    text contains one of ``marks``; every other statement matches."""
    def answer(index, user):
        items = json.loads(mf._section(user, "STATEMENTS TO COMPARE:\n", "\n\nDEFINITIONS:"))
        bad = {i["statement_id"]: ("different", "domains") for i in items
               if isinstance(i["formal_statement"], str) and any(m in i["formal_statement"] for m in marks)}
        return mf.match_output(bad)(index, user)
    return answer


def harness(tmp_path, answers, *, matcher=None, crosscheck="upheld", tiebreak="upheld"):
    h = lean_harness(tmp_path, answers)
    h.llm.script["statement_match"] = matcher or objecting_to(GENERIC_MARK, "table[table.length]")
    h.llm.script["statement_crosscheck"] = mf.crosscheck_output(default=crosscheck)
    h.llm.script["statement_tiebreak"] = mf.crosscheck_output(default=tiebreak)
    return h


def calls(h, prompt):
    return [c for c in h.llm.calls if c["prompt"] == prompt]


def shown_statements(call) -> list[str]:
    return [s["statement_id"] for s in json.loads(mf._section(call["user"], "FORMAL STATEMENTS:\n",
                                                              "\n\nDEFINITIONS:"))]


def p3_statement(record) -> dict:
    return next(o for o in record["obligations"] if o["claim_id"] == "P3")["statements"][0]


def p3_sid(record) -> str:
    return p3_statement(record)["id"]


# ----- the match before the freeze ------------------------------------------------------------------------------------

def test_the_generic_list_fact_gets_an_objection_before_the_freeze_and_is_fixed_next_round(tmp_path):
    h = harness(tmp_path, [with_p3(K1_RUN2), with_p3(K1_RUN3)])
    h.m()
    fixers = calls(h, "statement_fixer")
    assert len(fixers) == 2
    retry = fixers[1]["user"]
    assert "statement match before the freeze" in retry and "O6 statement 1 (O6.a)" in retry
    # FIX-19 part 2: the host check names the same statement.
    assert "P3 (O6 statement 1): the Lean statement uses none of the names" in retry
    record = h.frozen()
    validate("math_statement", record)
    assert p3_statement(record)["spec"] == K1_RUN3
    rounds = record["prefreeze_match"]
    assert [r["round"] for r in rounds] == [1, 2]
    assert list(rounds[0]["mismatched"]) == ["O6.a"] and rounds[1]["mismatched"] == {}
    assert h.slot("M", "P3")["status"] == "proven"
    h.c()
    match = {i["statement_id"]: i for i in h.verdict("C")["statement_match"]}
    assert match[p3_sid(record)]["match"] == "matched"


def test_the_match_alone_sends_a_statement_back(tmp_path):
    # Renamed so the host check passes: the round comes from the upheld objection only.
    assert not any("P3 (O6" in p for p in mc.fixer_problems(with_p3(K1_RUN2_RENAMED), CARDS["k1"], [], 40))
    h = harness(tmp_path, [with_p3(K1_RUN2_RENAMED), with_p3(K1_RUN3)])
    h.m()
    fixers = calls(h, "statement_fixer")
    assert len(fixers) == 2
    assert "statement match before the freeze" in fixers[1]["user"]
    assert "uses none of the names" not in fixers[1]["user"]
    assert h.frozen()["first_problems"] == []
    assert h.slot("M", "P3")["status"] == "proven"


def test_a_mismatch_left_after_the_last_round_is_frozen_and_decided_at_c(tmp_path):
    stuck = with_p3(K1_RUN2_RENAMED)
    h = harness(tmp_path, [stuck])
    h.m()
    fixers = calls(h, "statement_fixer")
    assert len(fixers) == 3
    assert fixers[2]["user"].count("statement match before the freeze") == 1
    record = h.frozen()
    sid = p3_sid(record)
    assert p3_statement(record)["spec"] == K1_RUN2_RENAMED
    # The last answer has no round after it, so it is not matched again; the earlier reading is not a verdict.
    assert [r["round"] for r in record["prefreeze_match"]] == [1, 2]
    assert record["prefreeze_match"][1]["matched"] == [] and sid in record["prefreeze_match"][1]["reused"]
    assert h.slot("M", "P3")["status"] == "proven"
    h.c()
    match = {i["statement_id"]: i for i in h.verdict("C")["statement_match"]}
    assert match[sid]["match"] == "mismatched"
    assert h.slot("C", "P3")["status"] == "blocked"


def test_only_new_or_changed_statements_are_matched_again(tmp_path):
    h = harness(tmp_path, [with_p3(K1_RUN2), with_p3(K1_RUN3)])
    h.m()
    record = h.frozen()
    first, second = record["prefreeze_match"]
    every = [s["id"] for o in record["obligations"] for s in o["statements"]]
    assert sorted(first["matched"]) == sorted(every) and first["reused"] == []
    assert second["matched"] == ["O6.a"] and sorted(second["reused"]) == sorted(set(every) - {"O6.a"})
    back = calls(h, "statement_backtranslate")
    m_calls = [c for c in back if "O6.a" in shown_statements(c)][:2]
    assert len(shown_statements(m_calls[0])) == len(every) and shown_statements(m_calls[1]) == ["O6.a"]
    # The extra calls are recorded under their roles at stage M, before the freeze.
    events = list(h.ledger.iter_events())
    frozen_at = next(i for i, e in enumerate(events) if e["type"] == "math_statements_frozen")
    early = [e["data"] for e in events[:frozen_at] if e["type"] == "math_role_call"]
    assert {(e["role"], e["stage"]) for e in early} >= {("statement_match", "M"), ("statement_crosscheck", "M")}
    assert sum(1 for e in events[:frozen_at] if e["type"] == "llm_call" and e["data"]["role"] == "statement_match") == 4


def test_the_match_key_follows_the_statement_its_claim_and_its_definitions():
    record = {"claims": [{"id": "P3", "text": "a limit"}], "definitions": [dict(MECHANISM, sha256="x")],
              "profile": {"limits": [{"task": "other"}]}}
    obligation = {"id": "O6", "claim_id": "P3", "claim_quote": "q", "definitions": ["D1"], "discharges": [],
                  "statements": []}
    statement = {"id": "O6.a", "form": "lean", "spec": K1_RUN3}
    key = m_math.match_key(statement, obligation, record)
    assert m_math.match_key(dict(statement, id="O9.a"), dict(obligation, id="O9"), record) == key
    assert m_math.match_key(dict(statement, spec=K1_RUN1), obligation, record) != key
    changed = copy.deepcopy(record)
    changed["definitions"][0]["text"] += " Changed."
    assert m_math.match_key(statement, obligation, changed) != key
    changed = copy.deepcopy(record)
    changed["profile"]["limits"][0]["task"] = "copy"
    assert m_math.match_key(statement, obligation, changed) != key


def test_a_disputed_objection_goes_to_the_tiebreak_before_the_freeze(tmp_path):
    h = harness(tmp_path, [with_p3(K1_RUN2_RENAMED), with_p3(K1_RUN3)], crosscheck="rejected", tiebreak="upheld")
    h.m()
    assert len(calls(h, "statement_tiebreak")) == 1
    assert "statement match before the freeze" in calls(h, "statement_fixer")[1]["user"]
    record = h.frozen()
    assert list(record["prefreeze_match"][0]["mismatched"]) == ["O6.a"]
    assert record["prefreeze_match"][0]["models"]["statement_tiebreak"] == ["fake-statement_tiebreak"]


def test_a_tiebreak_that_rejects_the_objection_sends_nothing_back(tmp_path):
    h = harness(tmp_path, [with_p3(K1_RUN2_RENAMED), with_p3(K1_RUN3)], crosscheck="rejected", tiebreak="rejected")
    h.m()
    assert len(calls(h, "statement_tiebreak")) == 1
    assert len(calls(h, "statement_fixer")) == 1
    record = h.frozen()
    assert record["prefreeze_match"][0]["mismatched"] == {}
    assert p3_statement(record)["spec"] == K1_RUN2_RENAMED


def test_a_dispute_the_tiebreak_cannot_settle_is_left_to_stage_c(tmp_path):
    h = harness(tmp_path, [with_p3(K1_RUN2_RENAMED)], crosscheck="rejected")
    h.llm.fail["statement_tiebreak"] = StageError("tiebreak transport failed")
    h.m()
    assert len(calls(h, "statement_fixer")) == 1
    record = h.frozen()
    assert list(record["prefreeze_match"][0]["unresolved"]) == ["O6.a"]


def test_the_match_roles_before_the_freeze_are_held_to_the_family_rules(tmp_path):
    h = harness(tmp_path, [with_p3(K1_RUN3)])
    h.llm.models = {"statement_fixer": "gpt-fixer", "statement_match": "gpt-matcher"}
    with pytest.raises(StageError, match="family separation"):
        h.m()


# ----- P3 statements over the card's shared mechanism definitions ---------------------------------------------------

def p3_problems(key: str, text: str, *, cite: bool = True) -> list[str]:
    card = CARDS[key]
    answer = with_p3(text, cite=cite, card_key=key)
    mechanism = mc.p3_mechanism(card, answer["limits"], answer["definitions"])
    return profile.p3_mechanism_problems(answer["obligations"], mechanism)


@pytest.mark.parametrize("key, text", [("k1", K1_RUN1), ("k1", K1_RUN3), ("k2", K2_RUN1), ("k3", K3_RUN2),
                                       ("k4", K4_RUN2)])
def test_the_passing_p3_statements_of_calibration_051933z_pass_the_check(key, text):
    assert p3_problems(key, text) == []


def test_the_generic_list_fact_of_k1_run2_fails_the_check():
    problems = p3_problems("k1", K1_RUN2)
    assert len(problems) == 1 and "uses none of the names of the card's mechanism part the limit names" in problems[0]
    assert "E, P, embed, h" in problems[0]


def test_a_p3_statement_whose_obligation_cites_no_shared_mechanism_definition_fails():
    problems = p3_problems("k1", K1_RUN3, cite=False)
    assert len(problems) == 1 and "cites no shared mechanism definition" in problems[0]
    answer = with_p3(K1_RUN3)
    answer["definitions"][0]["scope"] = "cost_reference"
    mechanism = mc.p3_mechanism(CARDS["k1"], answer["limits"], answer["definitions"])
    assert mechanism["definitions"] == []
    assert any("cites no shared mechanism definition" in p
               for p in profile.p3_mechanism_problems(answer["obligations"], mechanism))


def test_a_state_limit_needs_the_state_or_its_stated_capacity():
    # K2's counting statement without the 16896-bit capacity names nothing of the state.
    bare = K2_RUN1.replace("16896", "N").replace("4225", "K0")
    assert any("uses none of the names" in p for p in p3_problems("k2", bare))
    assert p3_problems("k2", bare.replace("encode :", "s :").replace("encode values", "s values")) == []


def test_the_names_come_from_the_part_each_limit_anchors():
    def names(key):
        answer = cf.fixer_output(key, CARDS[key])
        return mc.p3_mechanism(CARDS[key], answer["limits"], [])
    assert names("k1")["names"] == ["E", "P", "embed", "h"] and names("k1")["numbers"] == set()
    assert names("k2")["names"] == ["s", "y"] and 16896 in names("k2")["numbers"]
    assert names("k3")["names"] == ["z"] and 16 in names("k3")["numbers"]
    assert names("k4")["names"] == ["w"] and 64 in names("k4")["numbers"]
    assert mc.equation_operands(r"h^0_t = E[x_t] + P[t]") == {"h", "E", "P"}


def test_one_letter_names_keep_their_case():
    assert profile.uses_name("def Claim : Prop := ∀ P : ℕ, P = P", ["P"], set())
    assert not profile.uses_name("def Claim : Prop := ∀ p : ℕ, p = p", ["P"], set())
    assert profile.uses_name("def EmbedDomain : Prop := True\ndef Claim : Prop := EmbedDomain", ["embed"], set())
    assert not profile.uses_name("-- P and E\ndef Claim : Prop := True", ["P", "E"], set())


def test_the_check_reaches_the_fixer_through_the_host_problems():
    problems = mc.fixer_problems(with_p3(K1_RUN2), CARDS["k1"], [], 40)
    assert any("P3 (O6 statement 1): the Lean statement uses none of the names" in p for p in problems)
    assert not any("P3 (O6" in p for p in mc.fixer_problems(with_p3(K1_RUN3), CARDS["k1"], [], 40))
