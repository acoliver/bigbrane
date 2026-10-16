"""The compute target gate (hybrid/math/target.py; plan sections 1.1 and 4.4, decision D1 defaults)."""
import pytest

from hybrid.math import references, target

D1 = {"max_cost_exponent": 1.5, "state_rule": "log"}


def spec(ops, state, sizes=None):
    return {"ops": ops, "state": state, "sizes": sizes or {}}


def test_the_softmax_reference_fails_on_cost_and_state():
    formula = references.REFERENCES["softmax_attention"][1]
    result = target.evaluate(formula, **D1)
    assert result["status"] == "fail" and result["kill_code"] == "obligation_failed:target"
    assert result["cost_exponent"] == pytest.approx(2.0)
    assert result["state_growth"] == "power" and result["state_exponent"] == pytest.approx(1.0)
    assert len(result["reasons"]) == 2


@pytest.mark.parametrize("name", ["elman_rnn", "hmm_forward", "decayed_unigram"])
def test_linear_references_with_fixed_state_pass(name):
    result = target.evaluate(references.REFERENCES[name][1], **D1)
    assert result["status"] == "pass" and result["kill_code"] is None
    assert result["cost_exponent"] == pytest.approx(1.0) and result["state_growth"] == "fixed"


def test_exponent_at_the_ceiling_passes_and_above_it_fails():
    assert target.evaluate(spec("n**(3/2) + n", "d", {"d": [4, 8]}), **D1)["status"] == "pass"
    above = target.evaluate(spec("n**(8/5)", "1"), **D1)
    assert above["status"] == "fail" and above["cost_exponent"] == pytest.approx(1.6)


def test_n_log_n_cost_counts_as_exponent_one():
    result = target.evaluate(spec("n*log(n)*d", "d", {"d": [2, 4]}), **D1)
    assert result["status"] == "pass" and result["cost_exponent"] == pytest.approx(1.0)


def test_logarithmic_state_passes_under_log_and_fails_under_fixed():
    s = spec("n*d", "d + log(n)", {"d": [2, 4]})
    assert target.evaluate(s, **D1)["state_growth"] == "log"
    assert target.evaluate(s, **D1)["status"] == "pass"
    fixed = target.evaluate(s, max_cost_exponent=1.5, state_rule="fixed")
    assert fixed["status"] == "fail" and "fixed state" in fixed["reasons"][0]


def test_state_growing_as_a_power_of_n_fails_even_when_slow():
    result = target.evaluate(spec("n", "sqrt(n)"), **D1)
    assert result["status"] == "fail" and result["state_exponent"] == pytest.approx(0.5)


def test_the_largest_exponent_over_the_listed_sizes_decides():
    # The n^2 term vanishes only at k = 1, which is not the listed value that decides.
    result = target.evaluate(spec("n + (k - 1)*n**2", "1", {"k": [1, 3]}), **D1)
    assert result["status"] == "fail" and result["cost_exponent"] == pytest.approx(2.0)


def test_bad_rules_are_refused():
    with pytest.raises(ValueError, match="state rule"):
        target.evaluate(spec("n", "1"), max_cost_exponent=1.5, state_rule="sublinear")
    with pytest.raises(ValueError, match="max_cost_exponent"):
        target.evaluate(spec("n", "1"), max_cost_exponent=0, state_rule="log")
    with pytest.raises(ValueError, match="not size parameters"):
        target.evaluate(spec("n*q", "1"), **D1)
