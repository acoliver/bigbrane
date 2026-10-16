"""FIX-11 regressions: how a P5 copy, recall or dyck entry states its accuracy.

In runs/calibration-math-20261014T204729Z/k3-run1 the statement fixer froze P5 statement O16.a with recall entries
{K: 8, V: 16, accuracy: 0.99, state_bits: 4} and {K: 32, ...}. The K3 card says recall fails at K = 8 to 64 and is at
chance at K = 64; 0.99 is the P2 threshold, not a number the card claims. The recall law needs 31.0 and 124.2 bits
for those entries, so O16.a was violated at M and K3's P5 cell failed. The host now accepts an entry's accuracy only
as "ceiling" (replaced before freezing by the information ceiling of the entry's state bits at its size) or as a
number with "claim", the id of a card claim. The laws, the inequalities and the post-execution consistency rule are
unchanged, so a cited number the laws forbid still freezes and fails P5 at M.

The recorded obligation is copied unchanged into fixtures/calibration_math_20261014T204729Z/k3_run1_o16a.json.
"""
import copy
import json
import math
from pathlib import Path

import pytest

from hybrid.math import conservation, profile, spec_checks
from hybrid.math.runner import LIBRARY_CAPS
from hybrid.research.calibration import load_cards
from hybrid.research.mathrecords import formal_claims
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import load_section
from hybrid.research.stages.falsify import claim_items
from hybrid.tests.research import calibration_fakes as cf

FIXTURE = (Path(__file__).resolve().parents[1] / "research" / "fixtures" / "calibration_math_20261014T204729Z"
           / "k3_run1_o16a.json")
MAX_STATEMENTS = 100
P5_SECTIONS = ("copy", "recall", "dyck")


def recorded() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def recorded_spec() -> dict:
    return copy.deepcopy(recorded()["obligation"]["statements"][0]["spec"])


def as_ceilings(spec: dict) -> dict:
    spec = copy.deepcopy(spec)
    for section in P5_SECTIONS:
        for entry in spec[section]:
            entry["accuracy"] = profile.CEILING
    return spec


def k3():
    card = load_cards(("k3",))["k3"]
    claims = formal_claims(claim_items(card)) + profile.profile_claims()
    return card, [c["id"] for c in claims], {c["id"]: c["text"] for c in claims}


def k3_answer(p5_spec: dict) -> dict:
    card, _, _ = k3()
    answer = cf.fixer_output("k3", card)
    p5 = next(o for o in answer["obligations"] if o["claim_id"] == "P5")
    # The scripted P4 statements use m = 40; their sample entry joins the recorded ones (FIX-8's rule that every
    # P4 example count has one is not what these tests are about).
    p5_spec = dict(p5_spec, sample=[*p5_spec.get("sample", []), *cf.p5_numbers("k3")["sample"]])
    p5["statements"] = [{"form": "conservation", "conservation": p5_spec}]
    return answer


def problems_of(answer: dict) -> list[str]:
    card, claim_ids, texts = k3()
    return mc.fixer_problems(answer, card, claim_ids, MAX_STATEMENTS, texts)


def problems_without_resources(answer: dict) -> list[str]:
    """The fixer's problems for k3 without its resources section. FIX-16: with resources the host derives each entry's
    state_bits and sets D and V (card_resources); without them it fills nothing in, and the P5 rules read the
    statement's own numbers. These are the rules the tests below guard."""
    card, claim_ids, texts = k3()
    card = {key: value for key, value in card.items() if key != "resources"}
    return mc.fixer_problems(answer, card, claim_ids, MAX_STATEMENTS, texts)


def frozen_of(answer: dict) -> dict:
    card, claim_ids, texts = k3()
    return mc.freeze(answer, card, claim_ids, MAX_STATEMENTS, texts)


def frozen_p5(frozen: dict) -> dict | None:
    found = [s["spec"] for o in frozen["obligations"] if o["claim_id"] == "P5" for s in o["statements"]]
    return found[0] if found else None


def p5_problems(problems: list[str]) -> list[str]:
    return [p for p in problems if "P5:" in p]


# ----- the recorded k3-run1 statement -----------------------------------------------------------------------------

def test_the_fixture_is_the_recorded_o16a_statement():
    data = recorded()
    statement = data["obligation"]["statements"][0]
    assert (data["obligation"]["claim_id"], statement["id"], statement["form"]) == ("P5", "O16.a", "conservation")
    assert spec_checks.spec_hash(statement["spec"]) == statement["sha256"]
    assert statement["spec"]["recall"][:2] == [{"K": 8, "V": 16, "accuracy": 0.99, "state_bits": 4},
                                               {"K": 32, "V": 16, "accuracy": 0.99, "state_bits": 4}]
    assert "capability.4" in data["claim_ids"]


def test_the_recorded_numbers_are_violated_under_the_unchanged_recall_law():
    report = conservation.report(conservation.check_claims(recorded_spec()))
    assert report["status"] == conservation.VIOLATED
    violated = [c for c in report["checks"] if c["status"] == conservation.VIOLATED]
    assert [(c["law"], c["claim"]["K"]) for c in violated] == [("associative_recall", 8), ("associative_recall", 32)]
    assert [round(c["bound"]["required_bits"], 1) for c in violated] == [31.0, 124.2]


def test_the_recorded_o16a_recall_entries_are_refused_before_freezing():
    answer = k3_answer(recorded_spec())
    problems = p5_problems(problems_of(answer))
    for K in (8, 32):
        assert any(f'P5: the recall entry at K = {K} gives accuracy 0.99 with no "claim"' in p for p in problems)
    # The message tells the fixer what to write instead and names the card's claims.
    assert any('{"accuracy": "ceiling"}' in p and "capability.4" in p for p in problems)
    # The copy and dyck entries carried numbers without a claim too.
    assert any("the copy entry at L = 8" in p for p in problems)
    assert any("the dyck entry at depth = 2" in p for p in problems)
    frozen = frozen_of(answer)
    assert frozen_p5(frozen) is None
    dropped = next(d for d in frozen["dropped"] if d.get("statement_id") == "O11.a")
    assert any("the recall entry at K = 32 gives accuracy 0.99" in p for p in dropped["problems"])
    assert dropped["statement"]["conservation"]["recall"][0]["accuracy"] == 0.99
    assert any(n["claim_id"] == "P5" for n in frozen["not_formalized"])


def test_the_same_entries_written_as_ceiling_freeze_and_discharge():
    answer = k3_answer(as_ceilings(recorded_spec()))
    assert p5_problems(problems_of(answer)) == []
    spec = frozen_p5(frozen_of(answer))
    assert spec is not None
    assert all(entry["ceiling"] is True and isinstance(entry["accuracy"], float)
               for section in P5_SECTIONS for entry in spec[section])
    assert [e["accuracy"] for e in spec["dyck"]] == [1.0, 1.0, 0.0625, 0.000244140625]
    assert [e["accuracy"] for e in spec["copy"]] == [2.0 ** -12, 2.0 ** -60, 2.0 ** -124]
    recall = [e["accuracy"] for e in spec["recall"]]
    assert recall == [conservation.recall_ceiling(4, K, 16) for K in (8, 32, 64)]
    assert all(1 / 16 <= a < 0.99 for a in recall) and recall == sorted(recall, reverse=True)
    report = conservation.report(conservation.check_claims(spec))
    assert report["status"] == conservation.DISCHARGED
    # The sample entries and everything else are frozen as written.
    assert spec["sample"] == recorded_spec()["sample"] + cf.p5_numbers("k3")["sample"]


def test_ceilings_still_fail_p5_when_a_p2_construction_beats_the_declared_state():
    spec = frozen_p5(frozen_of(k3_answer(as_ceilings(recorded_spec()))))
    # K3's recorded P2 results: nesting holds at depths 2 and 4 only, recall nowhere.
    assert profile.consistency([], {"dyck": {2: True, 4: True, 8: False, 16: False}}, [spec]) == {}
    # A recall construction that met the threshold at K = 8 on a four-bit state contradicts the ceiling.
    beaten = profile.consistency([], {"recall": {8: True, 32: False, 64: False}}, [spec])
    assert "P2 met the threshold on recall at K=8" in beaten["P5"]
    beaten = profile.consistency([], {"dyck": {2: True, 4: True, 8: True, 16: False}}, [spec])
    assert "dyck at depth=8" in beaten["P5"]


def test_a_cited_impossible_claim_freezes_and_is_violated():
    spec = as_ceilings(recorded_spec())
    for entry in spec["recall"][:2]:
        entry.update(accuracy=0.99, claim="capability.4")
    answer = k3_answer(spec)
    assert p5_problems(problems_of(answer)) == []
    frozen = frozen_p5(frozen_of(answer))
    assert frozen["recall"][0] == {"K": 8, "V": 16, "accuracy": 0.99, "state_bits": 4, "claim": "capability.4"}
    report = conservation.report(conservation.check_claims(frozen))
    assert report["status"] == conservation.VIOLATED
    assert report["kill_codes"] == ["conservation:associative_recall"]
    assert report["counterexample"]["K"] == 8


def test_a_cited_impossible_claim_fails_p5_at_m_end_to_end(tmp_path):
    from hybrid.tests.research.test_profile_stage import Harness
    spec = cf.p5_numbers("k3")
    spec["recall"] = [{"K": K, "V": 16, "accuracy": 0.99, "claim": "capability.4"} for K in (8, 32, 64)]
    card = load_cards(("k3",))["k3"]
    answer = cf.fixer_output("k3", card)
    next(o for o in answer["obligations"] if o["claim_id"] == "P5")["statements"][0]["conservation"] = spec
    h = Harness(tmp_path, "k3", fixer=answer)
    h.m()
    record = h.frozen()
    assert not any("P5:" in p for p in record["first_problems"])
    assert frozen_p5(record)["recall"][0]["claim"] == "capability.4"
    assert h.slot("M", "P5")["status"] == "failed"
    assert h.verdict("M")["checks"]["O11.a"]["status"] == "violated"
    assert "obligation_failed:P5" in h.verdict("M")["kill_codes"]


def test_the_recorded_answer_gets_one_retry_then_leaves_p5_unproven(tmp_path):
    from hybrid.tests.research.test_profile_stage import Harness
    h = Harness(tmp_path, "k3", fixer=k3_answer(recorded_spec()))
    h.m()
    users = [c["user"] for c in h.llm.calls if c["prompt"] == "statement_fixer"]
    assert len(users) == 2
    assert 'P5: the recall entry at K = 8 gives accuracy 0.99 with no "claim"' in users[1]
    assert "P5 ACCURACIES (FIX-11)" in users[0]
    assert h.slot("M", "P5")["status"] == "not_formalized"
    # K3's failed constructions come first in the kill codes (an unproven code is listed only without failures);
    # the profile block records the slot as not formalized and the profile as not passed.
    block = h.verdict("M")["profile"]
    assert block["slots"]["P5"]["status"] == "not_formalized" and block["passed"] is False
    assert "obligation_failed:P5" not in h.verdict("M")["kill_codes"]


# ----- refusals -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change, expected", [
    ({"accuracy": 0.0625}, 'gives accuracy 0.0625 with no "claim"'),
    ({"accuracy": 0.0625, "claim": None}, 'gives accuracy 0.0625 with no "claim"'),
    ({"accuracy": 0.0625, "claim": "capability.9"}, "cites claim 'capability.9', which is not one of the card's"),
    ({"accuracy": 0.0625, "claim": "P2.recall"}, "a profile slot is not a card claim"),
    ({"accuracy": 0.0625, "claim": "P5"}, "cites claim 'P5', which is not one of the card's claims"),
    ({"accuracy": "chance", "claim": "capability.4"}, "the accuracy must be \"ceiling\" or a number"),
    ({"accuracy": True, "claim": "capability.4"}, "the accuracy must be \"ceiling\" or a number"),
])
def test_a_missing_or_unknown_claim_id_is_refused(change, expected):
    spec = as_ceilings(recorded_spec())
    spec["recall"][2] = {key: value for key, value in dict(spec["recall"][2], **change).items() if value is not None}
    answer = k3_answer(spec)
    problems = p5_problems(problems_of(answer))
    assert len(problems) == 1 and "the recall entry at K = 64" in problems[0] and expected in problems[0]
    frozen = frozen_of(answer)
    assert frozen_p5(frozen) is None
    assert any(expected in p for d in frozen["dropped"] for p in d.get("problems", []))


def test_a_ceiling_needs_the_numbers_its_law_uses():
    spec = as_ceilings(recorded_spec())
    assert spec.pop("state_bits") == 4   # the recorded statement also gives a statement-level default
    del spec["dyck"][0]["state_bits"]
    spec["recall"][0]["V"] = 1
    problems = p5_problems(problems_without_resources(k3_answer(spec)))
    assert any("the dyck entry at depth = 2" in p and "lacks state_bits" in p for p in problems)
    assert any("the recall entry at K = 8: the ceiling cannot be computed: V must be at least 2" in p
               for p in problems)
    # FIX-16: with K3's resources the host supplies both numbers. V is the generator default, so the fixer's V = 1 is
    # refused; the entry's state_bits is derived from the card's 4-bit carried-state budget.
    problems = problems_of(k3_answer(spec))
    assert any("the recall entry at K = 8: V is the generator default 16; the entry gives 1" in p for p in problems)
    assert not any("lacks state_bits" in p for p in problems)
    spec["recall"][0]["V"] = 16
    assert frozen_p5(frozen_of(k3_answer(spec)))["dyck"][0]["state_bits"] == 4
    # A statement-level state_bits serves every entry that has none.
    spec = as_ceilings(recorded_spec())
    for section in P5_SECTIONS:
        for entry in spec[section]:
            del entry["state_bits"]
    spec["state_bits"] = 4
    assert p5_problems(problems_without_resources(k3_answer(spec))) == []
    assert p5_problems(problems_of(k3_answer(spec))) == []


def test_other_slots_and_sections_are_untouched():
    spec = as_ceilings(recorded_spec())
    spec["index"] = [{"n": 4, "accuracy": 1.0, "state_bits": 4}]
    resolved, problems = profile.resolve_p5_accuracies(spec, ["capability.1"])
    assert problems == [] and resolved["index"] == spec["index"] and resolved["sample"] == spec["sample"]
    statement = {"form": "conservation", "conservation": {"recall": [{"K": 8, "V": 16, "accuracy": 0.99,
                                                                      "state_bits": 4}]}}
    assert mc.resolved_statement({"claim_id": "capability.4"}, statement, []) == (statement, [])


# ----- FIX-8's per-size rules still hold --------------------------------------------------------------------------

def test_the_fix8_per_size_rules_still_hold_for_ceiling_entries():
    spec = as_ceilings(recorded_spec())
    spec["dyck"] = spec["dyck"][1:]
    spec["copy"][0]["D"] = 2
    problems = p5_problems(problems_without_resources(k3_answer(spec)))
    assert any("missing depth = [2]" in p for p in problems)
    assert any("copy entries at L = [8] must use the P2 construction's D (4)" in p for p in problems)
    # FIX-16: with K3's resources D is a protocol value (the generator default), so D = 2 is refused by the host.
    problems = problems_of(k3_answer(spec))
    assert any("missing depth = [2]" in p for p in problems)
    assert any("the copy entry at L = 8: D is the generator default 4; the entry gives 2" in p for p in problems)
    spec = as_ceilings(recorded_spec())
    spec["recall"] = [entry for entry in spec["recall"] if entry["K"] != 32]
    assert any("missing K = [32]" in p for p in p5_problems(problems_of(k3_answer(spec))))


def test_ceilings_sit_on_the_edge_of_each_law():
    for bits in (0, 1, 4, 7.5, 32):
        for D, L in ((2, 1), (4, 8), (4, 32), (16, 3)):
            ceiling = conservation.copy_ceiling(bits, D, L)
            assert ceiling == min(1.0, 2.0 ** (bits - L * math.log2(D)))
            assert conservation.copy_check(bits, D, L, ceiling).status == conservation.DISCHARGED
        for depth in (2, 4, 8, 16):
            ceiling = conservation.dyck_ceiling(bits, depth)
            assert conservation.dyck_check(bits, depth, ceiling).status == conservation.DISCHARGED
            if ceiling < 1.0:
                assert conservation.dyck_check(bits, depth, min(1.0, ceiling * 2)).status == conservation.VIOLATED
        for K, V in ((8, 16), (32, 16), (64, 16), (4, 2), (8, 256)):
            ceiling = conservation.recall_ceiling(bits, K, V)
            assert 1.0 / V <= ceiling <= 1.0
            assert conservation.associative_recall_check(bits, K, V, ceiling).status == conservation.DISCHARGED
            if ceiling < 1.0 - 1e-6:
                above = min(1.0, ceiling + 1e-6)
                assert conservation.associative_recall_check(bits, K, V, above).status == conservation.VIOLATED
    assert conservation.recall_ceiling(4, 4, 2) == 1.0


# ----- the prompt text --------------------------------------------------------------------------------------------

def test_the_rendered_requirement_and_the_sections_state_the_two_forms():
    text = profile.template_text(LIBRARY_CAPS)
    assert profile.P5_ACCURACY_TEXT in text and '"accuracy": "ceiling"' in text
    assert "Never write the P2 threshold 0.99 where the card claims failure" in text
    section = load_section("profile_obligations")
    assert '"accuracy": "ceiling"' in section and 'name it in `"claim"`' in section
    assert "each with the accuracy the card claims at that size" not in section
    assert "FIX-11" in load_section("profile_match") and '"ceiling": true' in load_section("profile_match")
    # The slot text keeps the FIX-8 wording as a prefix, so the recorded claim quote still matches.
    quote = recorded()["obligation"]["claim_quote"]
    assert profile.SLOT_TEXT["P5"].startswith(quote)
    card, _, texts = k3()
    assert mc.quote_ok(recorded()["obligation"], card, texts)
    for banned in ("\u2014", "canonical", "load-bearing"):
        assert banned not in profile.P5_ACCURACY_TEXT and banned not in section
