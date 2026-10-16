"""Cost verifier: op counter rules, reference implementations with hand-derived formulas, the planted exponent
error, and the guards against uncounted work (plan section 4.4, P1)."""
import pytest

from hybrid.math import cost, references

NS = (32, 64, 128, 256)


def counted(source, point):
    return cost.measure(cost.load_candidate(source), point)


# ----- counting rules ----------------------------------------------------------------------------------------------

def test_counting_rules_by_kind():
    ops = cost.OpCounter()
    a, b = ops.np.ones((3, 4)), ops.np.ones((4, 5))
    a @ b                                   # 3*5 outputs, k=4
    assert ops.counts["mac"] == 60
    v = ops.np.ones(6)
    v + 1.0                                 # 6 arith
    v * v                                   # 6 arith
    v.sum()                                 # 5 arith (k - 1 for k elements)
    assert ops.counts["arith"] == 17
    ops.np.exp(v)                           # 6 transcendental
    ops.np.tanh(v)                          # 6 transcendental
    assert ops.counts["transcendental"] == 12
    v > 0.5                                 # 6 compare
    v.max()                                 # 5 compare
    v.argmax()                              # 5 compare
    ops.np.where(v > 0, v, 0.0)             # 6 compare for >, 6 for where
    assert ops.counts["compare"] == 28
    ops.np.outer(v, v)                      # 36 arith
    assert ops.counts["arith"] == 53
    ops.charge("mac", 7)
    assert ops.total == 60 + 7 + 53 + 12 + 28


def test_data_movement_counts_nothing():
    ops = cost.OpCounter()
    m = ops.np.zeros((4, 4))
    row = m[1]
    m[2] = row
    ops.np.concatenate([m, m])
    m.reshape(16).copy()
    m.T
    assert ops.total == 0


def test_plain_numbers_are_lifted_and_raw_arrays_refused():
    import numpy as np

    ops = cost.OpCounter()
    ops.np.exp(1.0)
    assert ops.counts["transcendental"] == 1
    with pytest.raises(cost.CandidateError):
        ops.np.ones(3) + np.ones(3)
    with pytest.raises(cost.CandidateError):
        np.asarray(ops.np.ones(3))


def test_words_measures_state_and_refuses_unknown_objects():
    ops = cost.OpCounter()
    assert cost.words({"K": ops.np.zeros((5, 4)), "t": 3, "table": {7: 1, 8: 2}}) == 20 + 1 + 4
    with pytest.raises(cost.CandidateError):
        cost.words({"name": "x"})


@pytest.mark.parametrize("source,problem", [
    ("import numpy as np\ndef init(s, o): pass\n", "imports"),
    ("def init(s, o):\n    return o._a\n", "private attribute"),
    ("def init(s, o):\n    return eval('1')\n", "'eval'"),
    ("def init(s, o):\n    return o.__class__\n", "private attribute"),
])
def test_audit_refuses_escapes_from_the_counter(source, problem):
    problems = cost.audit_source(source)
    assert problems and problem in problems[0]
    with pytest.raises(cost.CandidateError):
        cost.load_candidate(source)


def test_missing_functions_are_refused():
    with pytest.raises(cost.CandidateError, match="predict"):
        cost.load_candidate("def init(s, o):\n    return {}, {}\ndef update(p, s, t, o):\n    return s\n")


# ----- references with known formulas ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["elman_rnn", "softmax_attention", "decayed_unigram", "hmm_forward"])
def test_reference_counts_equal_the_hand_derived_formula(name):
    source, claim, expected = references.REFERENCES[name]
    result = cost.verify_cost(source, cost.CostClaim(**claim), ns=NS)
    assert expected == "discharged"
    assert result["status"] == "discharged", result["failures"]
    for row in result["slopes"]:
        assert row["measured"] == row["formula"]


def test_exact_count_at_one_point_by_hand():
    # Elman RNN, d=8, V=16, n=10: 10 * (64 + 16 + 128 + 15) = 2230.
    m = counted(references.ELMAN_RNN, {"n": 10, "d": 8, "V": 16})
    assert m.ops == 2230 and m.state_peak_words == 8
    assert m.by_kind == {"mac": 10 * (64 + 128), "arith": 80, "transcendental": 80, "compare": 150}
    # Attention, d=4, V=8, n=3: 3*(48 + 32 + 8 - 3) + 13 * 6 = 333; state 2*3*4 + 4 = 28.
    m = counted(references.SOFTMAX_ATTENTION, {"n": 3, "d": 4, "V": 8})
    assert m.ops == 333 and m.state_peak_words == 28


def test_attention_formula_has_exponent_two_and_fails_a_linear_claim():
    source, claim, _ = references.REFERENCES["softmax_attention"]
    linear = dict(claim, ops="n*(3*d**2 + d*V + V - 3) + (2*d + 5)*n", exponent=1)
    result = cost.verify_cost(source, cost.CostClaim(**linear), ns=NS)
    assert result["status"] == "violated"
    assert result["counterexample"]["check"] == "exponent_n"


# ----- planted errors ----------------------------------------------------------------------------------------------

def test_planted_exponent_error_is_caught():
    """Code that is quadratic while the card claims linear cost."""
    source, claim, expected = references.REFERENCES["planted_quadratic"]
    result = cost.verify_cost(source, cost.CostClaim(**claim), ns=NS)
    assert expected == "violated" and result["status"] == "violated"
    first = result["counterexample"]
    assert first["check"] == "exponent_n" and first["formula_slope"] == 1.0
    assert first["measured_slope"] > 1.9
    checks = {f["check"] for f in result["failures"]}
    assert {"exponent_n", "ops_exact", "lines_exponent_n"} <= checks
    # The true count is d*n(n+1)/2 + n(d-1): the verifier accepts that formula.
    honest = dict(claim, ops="d*n*(n + 1)/2 + n*(d - 1)", exponent=2)
    assert cost.verify_cost(source, cost.CostClaim(**honest), ns=NS)["status"] == "discharged"


def test_planted_n_log_n_claim_on_a_nested_loop_is_caught():
    source, claim, _ = references.REFERENCES["planted_quadratic"]
    nlogn = dict(claim, ops="n*log(n)*d", exponent=1)
    result = cost.verify_cost(source, cost.CostClaim(**nlogn), ns=NS)
    assert result["status"] == "violated" and result["counterexample"]["check"] == "exponent_n"


def test_uncounted_python_loop_is_caught_by_the_line_slope():
    source, claim, expected = references.REFERENCES["planted_uncounted_loop"]
    result = cost.verify_cost(source, cost.CostClaim(**claim), ns=NS)
    assert expected == "violated" and result["status"] == "violated"
    assert {f["check"] for f in result["failures"]} == {"lines_exponent_n"}


def test_state_claim_is_checked():
    source, claim, _ = references.REFERENCES["softmax_attention"]
    fixed_state = dict(claim, state="d")
    result = cost.verify_cost(source, cost.CostClaim(**fixed_state), ns=NS)
    assert result["status"] == "violated"
    assert any(f["check"] == "state_exact" for f in result["failures"])


def test_params_that_grow_during_the_sequence_are_refused():
    source = '''
def init(sizes, ops):
    return {"log": []}, {"h": ops.np.zeros(2)}

def update(params, state, token, ops):
    params["log"].append(token)
    return {"h": state["h"] + 1.0}

def predict(params, state, ops):
    return 0
'''
    result = cost.verify_cost(source, cost.CostClaim(ops="2*n", state="2"), ns=(16, 32))
    assert result["status"] == "violated"
    assert {f["check"] for f in result["failures"]} == {"params_grew"}


def test_declared_remainder_is_honoured():
    source, claim, _ = references.REFERENCES["decayed_unigram"]
    loose = dict(claim, ops="2*n*V + n", ops_remainder="n", ops_remainder_lower_in=["V"])
    assert cost.verify_cost(source, cost.CostClaim(**loose), ns=NS)["status"] == "discharged"
    tight = dict(claim, ops="2*n*V + n")
    assert cost.verify_cost(source, cost.CostClaim(**tight), ns=NS)["status"] == "violated"


def test_claimed_exponent_must_match_the_formula():
    source, claim, _ = references.REFERENCES["decayed_unigram"]
    result = cost.verify_cost(source, cost.CostClaim(**dict(claim, exponent=0.5)), ns=NS)
    assert result["counterexample"]["check"] == "claimed_exponent"


def test_formula_symbols_must_be_size_parameters():
    source, claim, _ = references.REFERENCES["decayed_unigram"]
    with pytest.raises(ValueError, match="not size parameters"):
        cost.verify_cost(source, cost.CostClaim(**dict(claim, ops="2*n*W")), ns=NS)


def test_asymptotic_exponent_handles_log_factors():
    symbols = {"n": cost.sp.Symbol("n", positive=True)}
    assert cost.asymptotic_exponent(cost.parse_formula("n*log(n) + 5", symbols), symbols, {}) == 1.0
    assert cost.asymptotic_exponent(cost.parse_formula("n**1.5 + n", symbols), symbols, {}) == 1.5


def test_runs_are_deterministic():
    source, claim, _ = references.REFERENCES["hmm_forward"]
    a = cost.verify_cost(source, cost.CostClaim(**claim), ns=(16, 32))
    b = cost.verify_cost(source, cost.CostClaim(**claim), ns=(16, 32))
    strip = lambda r: [dict(m, seconds=None) for m in r["measurements"]]  # noqa: E731
    assert strip(a) == strip(b)


def test_every_size_parameter_needs_two_values():
    source, claim, _ = references.REFERENCES["decayed_unigram"]
    with pytest.raises(ValueError, match="two values"):
        cost.verify_cost(source, cost.CostClaim(**dict(claim, sizes={"V": [16]})), ns=NS)


def test_a_comprehension_over_the_history_is_caught_by_the_line_slope():
    source = '''
def init(sizes, ops):
    return {}, {"h": ops.np.zeros(sizes["d"]), "seen": []}

def update(params, state, token, ops):
    seen = state["seen"] + [token]
    total = [s * 2 for s in seen]
    return {"h": state["h"] * 0.5, "seen": seen}

def predict(params, state, ops):
    return 0
'''
    claim = cost.CostClaim(ops="n*d", state="d + n", sizes={"d": [4, 8]})
    result = cost.verify_cost(source, claim, ns=NS)
    assert "lines_exponent_n" in {f["check"] for f in result["failures"]}
