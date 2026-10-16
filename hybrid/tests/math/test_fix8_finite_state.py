"""FIX-8 regressions (results/phase4-20261013T073748Z/k3-p5-diag/REPORT.md).

1. P5 coverage before freezing. Card k3 in runs/calibration-math-20261014T121552Z/k3-run1 froze one dyck entry, at
   depth 16 with accuracy 0.000244140625. Its depth-4 construction met the threshold, and the post-execution
   consistency check failed P5 because nothing accounted for depth 4. The host now refuses a P5 statement that lacks
   an entry at any P2 size.
2. The finite carried-state contract. ``precision_bits`` is a float width; ``state_bits`` in the architecture
   contract is the budget of the whole carried state. The harness audits the carried state at initialization and
   after every token, counts distinct configurations, and fixes module-level data after init. A size that breaks a
   resource check never counts as a P2 success.

The honest witness is the recorded source of job mathlib-927063c878ef (k3-run1, statement O11.a), copied byte for
byte from in/request.json to fixtures/calibration_math_20261014T121552Z/k3_run1_nesting_request.json.
"""
import hashlib
import json
import math
from pathlib import Path

import pytest

from hybrid.math import architecture, conservation, constructions, cost, profile, spec_checks
from hybrid.math.runner import LIBRARY_CAPS
from hybrid.research.stages import prover_memory
from hybrid.tests.research import calibration_fakes as cf

FIXTURE = (Path(__file__).resolve().parents[1] / "research" / "fixtures" / "calibration_math_20261014T121552Z"
           / "k3_run1_nesting_request.json")
RECORDED_SOURCE_SHA256 = "0c6afa5d1c5813afec60ed516e701ade6705ec2c3a45c6ff6d6a39b2fde73490"
K3_GRID = [{"depth": 2, "accuracy": 1}, {"depth": 4, "accuracy": 1}, {"depth": 8, "accuracy": 0.0625},
           {"depth": 16, "accuracy": 0.000244140625}]
FOUR_BITS = {"hidden": "d", "precision_bits": 64, "state_bits": 4}
SPEC = {"task": "dyck", "sizes": [2, 4, 8, 16], "count": 60, "threshold": 0.99, "state_words": "1",
        "widths": {"d": "16", "rounds": "1", "positions": "1"}}

# Exact on every depth: the whole stack in one integer behind a marker bit (1 is the empty stack). At depth 8 it
# carries up to 511, far above the sixteen configurations four bits allow.
STACK_INDEX = '''
def init(layout):
    return 1

def update(state, token):
    if token in (0, 1):
        return (state << 1) | token
    return state >> 1

def predict(state):
    return 2 + (state & 1)
'''

# Seventeen states in one word: k3's four-bit window plus a start state 16 that behaves as 0. Accuracy is the
# window's (exact to depth 4); the index leaves 0 <= z < 16 at initialization.
SEVENTEEN_INDEX = '''
def init(layout):
    return 16

def update(state, token):
    z = 0 if state == 16 else state
    if token in (0, 1):
        return ((z << 1) & 15) | token
    return z >> 1

def predict(state):
    return 2 + (state & 1)
'''

# Seventeen configurations whose fields pack into four bits: an empty list at the start, then [z] with z in 0..15.
# Only the count of distinct whole configurations catches it.
SEVENTEEN_LIST = '''
def init(layout):
    return []

def update(state, token):
    z = state[0] if state else 0
    if token in (0, 1):
        return [((z << 1) & 15) | token]
    return [z >> 1]

def predict(state):
    return 2 + (state[0] & 1)
'''

# The stack kept in module data while the carried state is a dummy word.
HIDDEN_LIST = '''
STACK = []

def init(layout):
    STACK.clear()
    return 0

def update(state, token):
    if token in (0, 1):
        STACK.append(token)
    else:
        STACK.pop()
    return state

def predict(state):
    return 2 + STACK[-1]
'''

# The stack written into a module-level array; the carried state is only the depth.
HIDDEN_ARRAY = '''
BUF = np.zeros(16, dtype=np.int64)

def init(layout):
    BUF[:] = 0
    return 0

def update(state, token):
    if token in (0, 1):
        BUF[state] = token
        return state + 1
    return state - 1

def predict(state):
    return 2 + int(BUF[state - 1])
'''

# The stack in a mutable default argument.
HIDDEN_DEFAULT = '''
def init(layout):
    return 0

def update(state, token, memory=[]):
    if token in (0, 1):
        memory.append(token)
    else:
        memory.pop()
    return len(memory)

def predict(state, memory=None):
    return 2
'''


def recorded() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def k3_obligations(dyck=None, extra_dyck_size=None):
    from hybrid.research.calibration import load_cards
    answer = cf.fixer_output("k3", load_cards(("k3",))["k3"])
    p5 = next(o for o in answer["obligations"] if o["claim_id"] == "P5")["statements"][0]["conservation"]
    if dyck is not None:
        p5["dyck"] = dyck
    if extra_dyck_size is not None:
        nesting = next(o for o in answer["obligations"] if o["claim_id"] == "P2.nesting")["statements"][0]
        nesting["construction"]["sizes"].append(extra_dyck_size)
    return answer, p5


def p5_problems(answer) -> list[str]:
    return [p for p in profile.record_problems(answer["obligations"], answer["limits"], answer["architecture"])
            if p.startswith("P5")]


# ----- 1. per-size P5 coverage before freezing --------------------------------------------------------------------

def test_the_k3_freeze_with_only_a_depth_16_entry_is_refused_for_the_missing_small_sizes():
    answer, _ = k3_obligations(dyck=[{"depth": 16, "accuracy": 0.000244140625}])
    problems = p5_problems(answer)
    assert len(problems) == 1
    assert "dyck entry at every P2 size, depth in [2, 4, 8, 16]" in problems[0]
    assert "missing depth = [2, 4, 8]" in problems[0]


def test_a_larger_size_does_not_account_for_a_missing_smaller_one():
    answer, _ = k3_obligations(dyck=K3_GRID[1:])
    assert any("missing depth = [2]" in p for p in p5_problems(answer))
    answer, p5 = k3_obligations()
    p5["copy"] = [entry for entry in p5["copy"] if entry["L"] != 8]
    p5["recall"] = [entry for entry in p5["recall"] if entry["K"] != 32]
    problems = p5_problems(answer)
    assert any("missing L = [8]" in p for p in problems) and any("missing K = [32]" in p for p in problems)


def test_a_size_a_construction_adds_needs_its_own_entry_and_copy_entries_use_the_constructions_D():
    answer, _ = k3_obligations(dyck=K3_GRID, extra_dyck_size=6)
    assert any("depth in [2, 4, 6, 8, 16]" in p and "missing depth = [6]" in p for p in p5_problems(answer))
    answer, p5 = k3_obligations(dyck=K3_GRID)
    p5["copy"][0] = dict(p5["copy"][0], D=2)
    assert any("copy entries at L = [8] must use the P2 construction's D (4)" in p for p in p5_problems(answer))


def test_the_full_k3_grid_freezes_passes_conservation_and_the_unchanged_consistency_check():
    from hybrid.research.calibration import load_cards
    from hybrid.research.stages.math_contracts import normalized_answer
    answer, _ = k3_obligations(dyck=K3_GRID)
    assert p5_problems(answer) == []
    # FIX-16: each entry's state_bits is derived by the host from K3's resources (its 4-bit carried state), so the
    # conservation numbers are read from the statement as frozen.
    frozen, problems, _ = normalized_answer(answer, load_cards(("k3",))["k3"])
    assert problems == {}
    p5 = next(o for o in frozen["obligations"] if o["claim_id"] == "P5")["statements"][0]["conservation"]
    assert {entry["state_bits"] for section in ("copy", "recall", "dyck") for entry in p5[section]} == {4}
    report = conservation.report(conservation.check_claims(dict(p5, sample=[])))
    assert report["status"] == conservation.DISCHARGED
    k3_p2 = {"dyck": {2: True, 4: True, 8: False, 16: False}}
    assert profile.consistency([], k3_p2, [p5]) == {}
    # The frozen run's lone depth-16 entry still fails the same post-execution check.
    assert "dyck at depth=4" in profile.consistency([], k3_p2, [{"state_bits": 4, "dyck": K3_GRID[3:]}])["P5"]


def test_the_rendered_requirement_and_the_fixer_section_ask_for_every_size():
    text = profile.template_text(LIBRARY_CAPS)
    assert "a dyck entry at each depth in [2, 4, 8, 16]" in text
    assert "An entry at a larger size does not account for a smaller one" in text
    assert "dyck entry at depth >=" not in text
    section = (Path(profile.__file__).resolve().parents[1] / "research" / "prompt_sections"
               / "profile_obligations.md").read_text(encoding="utf-8")
    assert "a dyck entry {\"depth\", \"accuracy\"} at every P2 depth" in section
    assert "at the largest P2 depth" not in section


# ----- 2. the finite carried-state contract -----------------------------------------------------------------------

def test_the_architecture_contract_separates_a_bits_budget_from_float_precision():
    assert architecture.shared_problems({"hidden": "d", "precision_bits": 64}) == []
    assert architecture.shared_problems(FOUR_BITS) == []
    assert architecture.shared_problems(dict(FOUR_BITS, state_bits="log(d)/log(2)")) == []
    assert architecture.shared_problems(dict(FOUR_BITS, state_bits=None)) == []
    assert any("state_bits" in p for p in architecture.shared_problems(dict(FOUR_BITS, state_bits=-1)))
    assert any("state_bits" in p for p in architecture.shared_problems(dict(FOUR_BITS, state_bits="depth")))
    assert any("state_bits" in p for p in architecture.shared_problems(dict(FOUR_BITS, state_bits=True)))
    assert architecture.shared_problems(dict(FOUR_BITS, width=3))
    assert architecture.state_bits({"hidden": "d", "precision_bits": 64}, {"d": 16}) is None
    assert architecture.state_bits(FOUR_BITS, {"d": 16}) == 4
    assert architecture.state_bits(dict(FOUR_BITS, state_bits="log(d)/log(2)"), {"d": 16}) == pytest.approx(4)
    assert architecture.latent_states(4) == 16 and architecture.latent_states(math.log2(10)) == 10


def test_one_unbounded_integer_is_not_one_word_and_configurations_are_whole_states():
    assert cost.words(2 ** 63 - 1) == 1 and cost.words(2 ** 63) == 2 and cost.words(-(2 ** 200)) == 4
    assert cost.words(15) == 1
    assert cost.configuration(7) == (("int",), (7,))
    assert cost.configuration([7]) != cost.configuration((7,)) != cost.configuration(7)
    assert cost.configuration({"top": 1, "depth": 3})[1] == (3, 1)
    with pytest.raises(cost.StateDomainError):
        cost.configuration(0.5)
    with pytest.raises(cost.StateDomainError):
        cost.configuration({"z": constructions.np.zeros(2)})
    assert cost.packed_bits([15]) == 4 and cost.packed_bits([3, 4]) == 5 and cost.packed_bits([-1]) == 2


def test_the_recorded_honest_k3_witness_passes_depth_4_with_sixteen_states_under_four_bits():
    request = recorded()
    assert hashlib.sha256(request["source"].encode("utf-8")).hexdigest() == RECORDED_SOURCE_SHA256
    for contract in (request["architecture"], dict(request["architecture"], state_bits=4)):
        result = spec_checks.audited_construction(request["spec"], request["source"], contract)
        rows = {row["size"]: row for row in result["results"]}
        assert {size: row["accuracy"] for size, row in rows.items()} == {2: 1.0, 4: 1.0, 8: 0.058, 16: 0.002}
        assert rows[2]["passed"] and rows[4]["passed"] and not rows[8]["passed"] and not rows[16]["passed"]
        assert all(row["resource_violations"] == [] for row in rows.values())
        assert result["status"] == "violated" and "resource contract broken" not in result["evidence"]
        assert profile.p2_sizes([({"form": "construction", "spec": request["spec"]},
                                  {"decided": True, "details": result})]) == {
            "dyck": {2: True, 4: True, 8: False, 16: False}}
    # With the budget the counts are the report's: 4 states at depth 2, 16 at depths 4, 8 and 16.
    assert [rows[size]["carried_states"] for size in (2, 4, 8, 16)] == [4, 16, 16, 16]
    assert all(rows[size]["state_bits"] == 4 for size in rows)
    # Without it, float64 precision does not stand in for a carried-state budget: nothing is counted.
    plain = spec_checks.audited_construction(request["spec"], request["source"], request["architecture"])
    assert all(row["carried_states"] is None and row["state_bits"] is None for row in plain["results"])


def test_an_exact_depth_8_construction_cannot_pass_on_four_bits():
    result = spec_checks.audited_construction(SPEC, STACK_INDEX, FOUR_BITS)
    rows = {row["size"]: row for row in result["results"]}
    assert all(row["accuracy"] == 1.0 for row in rows.values())
    assert rows[2]["passed"] and rows[2]["resource_violations"] == []
    for size in (4, 8, 16):
        assert not rows[size]["passed"]
        assert any("outside the declared finite domain 0 <= z < 16" in v for v in rows[size]["resource_violations"])
        assert any("distinct carried configurations" in v for v in rows[size]["resource_violations"])
    assert result["status"] == "violated" and result["counterexample"]["check"] == "carried_state"
    p2 = profile.p2_sizes([({"form": "construction", "spec": SPEC}, {"decided": True, "details": result})])
    assert p2 == {"dyck": {2: True, 4: False, 8: False, 16: False}}
    # The same claim made in numbers: depth-8 success on four bits breaks the conservation law, and a depth-8 P2
    # success against the honest four-bit grid fails the unchanged consistency check.
    claimed = conservation.check_claims({"state_bits": 4, "dyck": [{"depth": 8, "accuracy": 0.99}]})
    assert claimed[0].status == conservation.VIOLATED and claimed[0].kill_code == "conservation:dyck"
    failures = profile.consistency([], {"dyck": {2: True, 4: True, 8: True, 16: False}},
                                   [{"state_bits": 4, "dyck": K3_GRID}])
    assert "dyck at depth=8" in failures["P5"]


@pytest.mark.parametrize("source,domain", [(SEVENTEEN_INDEX, True), (SEVENTEEN_LIST, False)])
def test_a_seventeen_state_construction_in_one_word_fails_although_accuracy_and_words_pass(source, domain):
    result = spec_checks.audited_construction(dict(SPEC, sizes=[2, 4]), source, FOUR_BITS)
    depth4 = next(row for row in result["results"] if row["size"] == 4)
    assert depth4["accuracy"] == 1.0 and depth4["state_peak_words"] <= depth4["declared_state_words"] == 1
    assert depth4["carried_states"] == 17 and not depth4["passed"]
    assert "17 distinct carried configurations at depth=4, above the 16 that state_bits 4 allows" in \
        depth4["resource_violations"]
    assert any("0 <= z < 16" in v for v in depth4["resource_violations"]) is domain
    assert result["status"] == "violated"
    p2 = profile.p2_sizes([({"form": "construction", "spec": SPEC}, {"decided": True, "details": result})])
    assert p2["dyck"][4] is False


@pytest.mark.parametrize("source", [HIDDEN_LIST, HIDDEN_ARRAY, HIDDEN_DEFAULT])
def test_mutable_hidden_storage_fails_the_size(source):
    result = constructions.run_construction(source, "dyck", [2, 4], count=20)
    assert result["status"] == "violated"
    for row in result["results"]:
        assert not row["passed"] and any("hidden storage" in v for v in row["resource_violations"])


def test_rebuilding_fixed_tables_in_init_is_not_hidden_storage():
    # k3's witness writes TABLES in every init and only reads it afterwards; arrays are writable again after a run.
    request = recorded()
    result = constructions.run_construction(request["source"], "dyck", [2, 4], count=20)
    assert result["status"] == "discharged"
    assert all(row["resource_violations"] == [] for row in result["results"])


def test_a_resource_invalid_size_is_never_a_p2_success():
    over = spec_checks.audited_construction(dict(SPEC, sizes=[2, 4], state_words="0"), STACK_INDEX)
    assert all(not row["passed"] and row["accuracy"] == 1.0 for row in over["results"])
    assert all(any("peak state words" in v for v in row["resource_violations"]) for row in over["results"])
    wide = spec_checks.audited_construction(dict(SPEC, sizes=[2, 4], widths={"d": "2", "rounds": "1",
                                                                             "positions": "1"}),
                                            recorded()["source"], {"hidden": "d", "precision_bits": 64})
    assert all(not row["passed"] and any("architecture" in v for v in row["resource_violations"])
               for row in wide["results"])
    for result in (over, wide):
        p2 = profile.p2_sizes([({"form": "construction", "spec": SPEC}, {"decided": True, "details": result})])
        assert p2 == {"dyck": {2: False, 4: False}}


def test_the_prover_reads_the_budget_in_its_frozen_contract_and_the_breach_in_its_memory():
    statement = {"id": "O11.a", "form": "construction", "spec": dict(SPEC)}
    facts = prover_memory.frozen_contract(statement, FOUR_BITS)
    assert all(point["state_bits"] == 4 and point["carried_states_allowed"] == 16 for point in facts["sizes"])
    assert all("state_bits" not in point for point in prover_memory.frozen_contract(
        statement, {"hidden": "d", "precision_bits": 64})["sizes"])
    row = spec_checks.audited_construction(dict(SPEC, sizes=[4]), SEVENTEEN_INDEX, FOUR_BITS)["results"][0]
    line = prover_memory.size_line(row)
    assert line["carried_states"] == 17 and line["resource_violations"] and line["passed"] is False
