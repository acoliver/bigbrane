"""FIX-20 (decision PR): one host-supplied precision policy for every check that compares a card's reference code with
its frozen equations, and conformance points and tolerance set by the protocol (results/phase4-20261013T073748Z/
FIX-20.md).

Cause: in K2 run 2 of calibration 20261016T051933Z conformance check O2.a compared the executed counted reference
(float64: the counter's normal, zeros and sigmoid made float64) with frozen equations that round state to float32,
under a grid and a 1e-6 tolerance the fixer chose. The 6.34e-06 store gap at position 205 is exactly 0 when both sides
use one precision (k2-o2-diag/REPORT.md). The fixture holds the recorded conformance requests of that calibration
(fixtures/calibration_math_20261016T051933Z/conformance_audits.json, built by tmp/fix19/scripts/fix20_fixture.py)."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hybrid.math import constructions, cost, findings, profile, spec_checks
from hybrid.research import calibration as cal
from hybrid.research.stages import card_resources as cr
from hybrid.research.stages import m_math
from hybrid.research.stages import math_contracts as mc

FIXTURE = Path(__file__).parent / "fixtures" / "calibration_math_20261016T051933Z" / "conformance_audits.json"
AUDITS = json.loads(FIXTURE.read_text())
K2_RUN2 = next(a for a in AUDITS if a["run"] == "k2-run2")
CARDS = cal.load_cards()


@pytest.fixture
def production(monkeypatch):
    """The production protocol (the research conftest sets small counts for the scripted runs)."""
    monkeypatch.undo()
    return profile.PROTOCOL


def protocol_spec(audit: dict, **changes) -> dict:
    """The recorded statement as the host now freezes it: points, tolerance and precision_bits from the protocol
    and the card's resources (card_resources), every claim field as the fixer wrote it."""
    given = {k: v for k, v in audit["spec"].items() if k not in ("points", "tolerance")}
    spec, problems = cr.normalize_statement({"resources": audit["resources"]}, None, "conformance", given)
    assert problems == []
    return dict(spec, **changes)


def first_gap(spec: dict, source: str, counter_bits, equations_bits) -> float:
    """The largest store-state gap on the failing K2 stream (V = d = 32, n = 256, repeated) with the counter and
    the equations' np at the given precisions (None: NumPy's defaults)."""
    frozen = spec_checks._load(spec["equations"], ("expected",), "equations",
                               np if equations_bits is None else spec_checks.PolicyNumPy(equations_bits))
    candidate = cost.load_candidate(source)
    counter = cost.OpCounter(counter_bits)
    params, state = candidate.init({"V": 32, "d": 32}, counter)
    tokens = spec_checks.conformance_streams("repeated", 256, 32)
    wanted = frozen["expected"](spec_checks._plain_tree(params), {"V": 32, "d": 32}, list(tokens))
    worst = 0.0
    for token, want in zip(tokens, wanted):
        state = candidate.update(params, state, token, counter)
        got = spec_checks._plain_tree(state)
        worst = max(worst, float(np.max(np.abs(np.asarray(got["s"], dtype=float)
                                               - np.asarray(want["state"]["s"], dtype=float)))))
    return worst


# ----- the recorded K2 O2.a case ---------------------------------------------------------------------------------------

def test_the_fixture_is_the_recorded_k2_o2_statement():
    assert K2_RUN2["statement_id"] == "O2.a" and K2_RUN2["spec"]["reference"] == "O1.a"
    assert spec_checks.spec_hash(K2_RUN2["spec"]) == K2_RUN2["statement_sha256"] \
        == "4993c2198e3e62293717d4969731d4143f1c6de6ab48d549c7d97493e7c513ab"
    assert K2_RUN2["architecture"]["precision_bits"] == 32 and K2_RUN2["spec"]["tolerance"] == 1e-6


def test_the_recorded_gap_comes_from_the_float64_counter_against_float32_equations():
    """The diagnosis, reproduced: the old counter (no policy, float64) against the recorded equations gives the
    6.34e-06 gap; one binary32 policy on both sides gives exactly 0."""
    spec, source = K2_RUN2["spec"], K2_RUN2["source"]
    assert first_gap(spec, source, None, None) == pytest.approx(6.546522385519893e-06, rel=1e-9)
    assert first_gap(spec, source, 32, 32) == 0.0


def test_the_recorded_case_discharges_under_the_shared_binary32_policy_and_the_protocol_grid(production):
    spec = protocol_spec(K2_RUN2)
    assert spec["precision_bits"] == 32 and spec["tolerance"] == 0.0
    assert spec["points"] == [{"d": d, "V": v, "n": 256} for d in (8, 32) for v in (8, 32)]
    result = spec_checks.reference_conformance(spec, K2_RUN2["source"])
    assert result["status"] == "discharged", result["evidence"]
    assert result["precision_bits"] == 32 and len(result["checked"]) == 12
    assert "both at 32-bit precision" in result["evidence"]


MECHANISM_CHANGES = {
    # The reference's store drops its retained term: s_new = i * u.
    "dropped store term": ("s_new = ops.np.add(kept, admitted)", "s_new = admitted"),
    # The reference's keep gate uses tanh where the card's logistic stands.
    "changed gate": ("r = ops.np.sigmoid(ar)", "r = ops.np.tanh(ar)"),
    # The reference retains an extra buffer.
    "added buffer": ("return {'s': s_new, 'y': y_new}", "return {'s': s_new, 'y': y_new, 'c': ops.np.add(s_new, 0.0)}"),
}


@pytest.mark.parametrize("change", sorted(MECHANISM_CHANGES))
def test_a_real_mechanism_change_still_fails_under_the_shared_policy(production, change):
    old, new = MECHANISM_CHANGES[change]
    assert K2_RUN2["source"].count(old) == 1
    result = spec_checks.reference_conformance(protocol_spec(K2_RUN2), K2_RUN2["source"].replace(old, new))
    assert result["status"] == "violated", result["evidence"]
    assert result["counterexample"]["position"] <= 1
    if change == "added buffer":
        assert "retains arrays the equations do not" in result["counterexample"]["detail"]


def test_a_reordered_sum_is_a_departure_from_the_frozen_operation_order(production):
    """The comparison is exact: equations that evaluate one affine map as two partial products (the same mechanism,
    another operation order) differ from the recipe the reference executes, and the audit says so."""
    old = "ar = (params['W_r'] @ z + params['b_r'])"
    new = "ar = (params['W_r'][:, :d] @ z[:d] + params['W_r'][:, d:] @ z[d:] + params['b_r'])"
    spec = protocol_spec(K2_RUN2)
    assert spec["equations"].count(old) == 1
    result = spec_checks.reference_conformance(dict(spec, equations=spec["equations"].replace(old, new)),
                                               K2_RUN2["source"])
    assert result["status"] == "violated"


# ----- K1 to K4 of calibration 20261016T051933Z ---------------------------------------------------------------------

@pytest.mark.parametrize("run", [a["run"] for a in AUDITS])
def test_each_recorded_conformance_statement_still_discharges_under_the_protocol(production, run):
    audit = next(a for a in AUDITS if a["run"] == run)
    spec = protocol_spec(audit)
    assert spec["precision_bits"] == audit["architecture"]["precision_bits"]
    assert spec_checks.conformance_problems(spec) == []
    result = spec_checks.reference_conformance(spec, audit["source"])
    assert result["status"] == "discharged", result["evidence"]
    assert len(result["checked"]) == len(spec["points"]) * len(spec["streams"])


def test_the_fixture_covers_k1_to_k4():
    assert {a["run"].split("-")[0] for a in AUDITS} == {"k1", "k2", "k3", "k4"}


# ----- the counted reference under the policy -------------------------------------------------------------------------

def test_the_counter_makes_every_float_at_the_policy_dtype():
    ops = cost.OpCounter(32)
    made = [ops.np.zeros(3), ops.np.ones((2, 2)), ops.np.full(2, 0.5), ops.np.eye(2), ops.np.arange(0.0, 1.0, 0.5),
            ops.np.array([1, 2]), ops.np.normal((2,), 1, 1.0), ops.np.sigmoid(ops.np.zeros(2))]
    assert [m._a.dtype for m in made] == [np.float32] * len(made)
    assert ops.np.arange(3)._a.dtype.kind == "i"


def test_the_counter_converts_an_integer_array_that_meets_a_float_one():
    ops = cost.OpCounter(32)
    mixed = ops.np.multiply(ops.np.arange(3), ops.np.ones(3))
    assert mixed._a.dtype == np.float32
    assert (ops.np.zeros(2) + 0.25)._a.dtype == np.float32


def test_the_counter_refuses_another_float_dtype():
    with pytest.raises(cost.PrecisionError, match="64-bit|float64"):
        cost.OpCounter(32).np.zeros(3, dtype=float)
    with pytest.raises(cost.PrecisionError):
        cost.OpCounter(64).np.zeros(3, dtype="float32")


def test_the_counter_draws_the_same_numbers_under_every_precision():
    wide = cost.OpCounter(64).np.normal((4, 4), 7, 0.3)._a
    narrow = cost.OpCounter(32).np.normal((4, 4), 7, 0.3)._a
    assert np.array_equal(wide.astype(np.float32), narrow)
    assert np.array_equal(cost.OpCounter().np.normal((4, 4), 7, 0.3)._a, wide)


def test_without_a_declared_precision_numpy_defaults_apply():
    ops = cost.OpCounter()
    assert ops.np.zeros(2)._a.dtype == np.float64 and ops.np.sigmoid(ops.np.zeros(2))._a.dtype == np.float64


def test_an_undeclared_width_is_refused():
    with pytest.raises(ValueError, match="precision_bits"):
        cost.OpCounter(8)


def test_the_policy_leaves_the_counted_cost_unchanged():
    candidate = cost.load_candidate(K2_RUN2["source"])
    for point in ({"V": 8, "d": 8, "n": 64}, {"V": 32, "d": 32, "n": 128}):
        plain = cost.measure(candidate, point)
        policed = cost.measure(candidate, point, precision_bits=32)
        assert (policed.ops, policed.by_kind, policed.state_peak_words, policed.lines) == \
            (plain.ops, plain.by_kind, plain.state_peak_words, plain.lines)


def test_the_cost_audit_runs_under_the_policy_and_holds_the_stored_arrays_to_it(production):
    claim = cost.CostClaim(**{k: K2_RUN2["cost_statement"]["spec"].get(k) for k in cost.CLAIM_FIELDS})
    architecture = {"shared": K2_RUN2["architecture"], "widths": K2_RUN2["cost_statement"]["spec"]["widths"]}
    result = cost.verify_cost(K2_RUN2["source"], claim, ns=(64, 128), architecture=architecture, precision_bits=32)
    assert result["precision_bits"] == 32
    assert not [f for f in result["failures"] if f["check"] in ("architecture", "ops_exact", "state_exact")]
    assert {p["float_bits"] for p in result["architecture"]["points"]} == {32}


def test_stage_m_sends_the_contract_precision_with_every_cost_check():
    statement = {"form": "cost", "spec": dict(K2_RUN2["cost_statement"]["spec"])}
    request = m_math.library_request(statement, K2_RUN2["source"], K2_RUN2["architecture"])
    assert request["precision_bits"] == 32 and request["architecture"]["shared"]["precision_bits"] == 32
    assert "precision_bits" not in m_math.library_request(statement, K2_RUN2["source"], None)


# ----- the equations under the policy --------------------------------------------------------------------------------

def test_the_equations_np_makes_new_floats_at_the_policy_dtype():
    np32 = spec_checks.PolicyNumPy(32)
    assert np32.zeros(3).dtype == np.float32 and np32.array([0.5]).dtype == np.float32
    assert np32.asarray(np.zeros(2, dtype=np.float32)).dtype == np.float32
    assert np32.arange(3).dtype.kind == "i" and np32.dtype("float32") == np.float32 and np32.tanh is np.tanh
    with pytest.raises(spec_checks.SpecError):
        np32.zeros(3, dtype=np.float64)


def test_equations_that_name_another_float_dtype_are_refused(production):
    spec = protocol_spec(K2_RUN2)
    wide = spec["equations"].replace("np.float32", "np.float64")
    problems = spec_checks.conformance_problems(dict(spec, equations=wide))
    assert len(problems) == 1 and "other than the declared 32-bit precision" in problems[0]
    assert findings.other_precision("y = x.astype(float)\nz = np.zeros(3, 'f8')\n", 32) == \
        ["line 1: float", "line 2: 'f8'"]
    assert findings.other_precision(spec["equations"], 32) == []


def test_equations_that_return_another_float_dtype_leave_the_audit_unknown(production):
    """A state the equations return at float64 under a binary32 policy is the statement's fault, not the
    reference's: the audit is unknown and not a prover fault. An int64 scalar (not a weak Python number) promotes
    float32 to float64 without naming a float dtype."""
    spec = protocol_spec(K2_RUN2)
    old = "'state': {'s': s.copy(), 'y': y.copy()}"
    leaking = spec["equations"].replace(old, "'state': {'s': s.copy() + np.longlong(0), 'y': y.copy()}")
    assert leaking != spec["equations"] and spec_checks.conformance_problems(dict(spec, equations=leaking)) == []
    result = spec_checks.reference_conformance(dict(spec, equations=leaking), K2_RUN2["source"])
    assert result["status"] == "unknown" and not result.get("prover_fault")
    assert "float64" in result["evidence"] and "under the declared 32-bit precision" in result["evidence"]


# ----- protocol points and tolerance --------------------------------------------------------------------------------

def test_the_protocol_n_covers_every_p2_protocol_stream_and_is_a_p1_length(production):
    longest = max(constructions.stream_lengths(task, size, constructions.CONTRACT_SAMPLE, 0)[1]
                  for task, sizes in profile.REQUIRED_SIZES.items() for size in sizes)
    n = profile.PROTOCOL["conformance_n"]
    assert longest == 193 and n in cost.DEFAULT_NS and n >= longest
    assert n == min(x for x in cost.DEFAULT_NS if x >= longest) == spec_checks.MAX_CONFORMANCE_N
    assert profile.PROTOCOL["conformance_tolerance"] == 0.0
    assert profile.FIELD_KINDS["conformance"]["tolerance"] == "protocol"
    assert profile.FIELD_KINDS["conformance"]["precision_bits"] == "card"


def test_the_protocol_grid_is_every_combination_of_the_cost_grid(production):
    assert profile.conformance_points(["V"]) == [{"V": 8, "n": 256}, {"V": 32, "n": 256}]
    assert len(profile.conformance_points(["d", "V"])) == 4
    assert len(profile.conformance_points(["a", "b", "c"])) == spec_checks.MAX_CONFORMANCE_POINTS


def test_a_fixer_chosen_grid_is_refused(production):
    spec, problems = cr.normalize_statement(CARDS["k2"], None, "conformance", copy.deepcopy(K2_RUN2["spec"]))
    assert any("points are protocol values" in p for p in problems)
    assert spec["points"] == profile.conformance_points(["d", "V"])


def test_a_fixer_chosen_tolerance_is_refused(production):
    given = {k: v for k, v in K2_RUN2["spec"].items() if k != "points"}
    spec, problems = cr.normalize_statement(CARDS["k2"], None, "conformance", given)
    assert problems == [p for p in problems if "tolerance is a protocol value" in p] and len(problems) == 1
    assert spec["tolerance"] == 0.0


def test_a_precision_other_than_the_cards_is_refused(production):
    given = {k: v for k, v in K2_RUN2["spec"].items() if k not in ("points", "tolerance")}
    _, problems = cr.normalize_statement(CARDS["k2"], None, "conformance", dict(given, precision_bits=64))
    assert len(problems) == 1 and "precision_bits" in problems[0] and "32" in problems[0]


def test_values_left_out_or_copied_exactly_are_filled_without_a_problem(production):
    given = {k: v for k, v in K2_RUN2["spec"].items() if k not in ("points", "tolerance")}
    filled, problems = cr.normalize_statement(CARDS["k2"], None, "conformance", given)
    assert problems == []
    again, problems = cr.normalize_statement(CARDS["k2"], None, "conformance", copy.deepcopy(filled))
    assert problems == [] and again == filled


def test_the_fixer_sees_the_refusal_as_a_problem_in_its_answer(production):
    """Through the fixer checks: an answer whose conformance statement carries its own grid and tolerance gets both
    problems back, so its one retry is told to leave them out."""
    answer = {"obligations": [{"id": "O2", "claim_id": None, "statements": [
        {"form": "conformance", "conformance": copy.deepcopy(K2_RUN2["spec"])}]}]}
    _, found, _ = mc.normalized_answer(answer, CARDS["k2"])
    problems = found[("O2", 0)]
    assert any("points are protocol values" in p for p in problems)
    assert any("tolerance is a protocol value" in p for p in problems)


def test_a_spec_without_the_host_values_cannot_run():
    """A conformance statement with no precision policy or tolerance is not run: the host sets both."""
    result = spec_checks.reference_conformance(dict(K2_RUN2["spec"], precision_bits=None), K2_RUN2["source"])
    assert result["status"] == "unknown" and "precision_bits" in result["evidence"]
    bare = {k: v for k, v in K2_RUN2["spec"].items() if k != "tolerance"}
    result = spec_checks.reference_conformance(dict(bare, precision_bits=32), K2_RUN2["source"])
    assert result["status"] == "unknown" and "tolerance" in result["evidence"]


def test_the_prompts_tell_the_fixer_and_the_prover_the_protocol():
    from hybrid.research.stages.common import load_section
    settlement = load_section("premise_settlement")
    assert "Leave `points`, `tolerance` and `precision_bits` out" in settlement and "(1 to 4 size points" not in settlement
    assert "the comparison is exact" in settlement
    assert "card's declared precision" in load_section("conformance_audit")
    assert f"n = {profile.PROTOCOL['conformance_n']}" in profile.protocol_text()
