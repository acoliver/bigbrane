"""FIX-17 (decision C17, as amended 2026-10-15): the skill scale at target, hybrid/math/coupled.py.

The P1 formulas are the ones the calibration math runs verified (results/calibration-math-*/outcomes.jsonl,
target.formula); the resources are the calibration cards' own (hybrid/fixtures/calibration/k*.json). The bounds are
run.toml [target] coupled_max_cost_exponent = 1.5 and coupled_max_state_exponent = 0.5.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hybrid.math import constructions, coupled

ROOT = Path(__file__).resolve().parents[3]
BOUNDS = {"max_cost_exponent": 1.5, "max_state_exponent": 0.5}
P1 = {"k1": ("(2*d+4)*n*(n+1)+n*(12*d**2+7*d+d*V+V-3)", "4*n*d+d+1"),
      "k2": ("n*(8*d**2+d*V+13*d+V-1)", "2*d"),
      "k3": ("n*(k*V+2*k+V-2)", "1"),
      "k4": ("n*(V-1)", "2")}
SLOTS = ("P2.copy", "P2.recall", "P2.nesting", "P2.composition")


def card(key: str) -> dict:
    return json.loads((ROOT / "hybrid" / "fixtures" / "calibration" / f"{key}.json").read_text(encoding="utf-8"))


def spec(key: str, resources: dict, ops: str | None = None, state: str | None = None) -> dict:
    return {"ops": ops or P1[key][0], "state": state or P1[key][1],
            "sizes": {name: [8, 32] for name in resources["cost"]["sizes"]},
            "widths": {"d": resources["cost"]["d"], "rounds": resources["cost"]["rounds"], "positions": "n"}}


def measure(key: str, **bounds) -> dict:
    resources = card(key)["resources"]
    return coupled.evaluate(spec(key, resources), resources, **dict(BOUNDS, **bounds))


@pytest.fixture(scope="module")
def results() -> dict:
    return {key: measure(key) for key in P1}


def test_k1_the_transformer_has_no_skill_at_target_since_even_a_constant_size_costs_n_squared(results):
    out = results["k1"]
    assert out["status"] == "fail" and out["kill_code"] == coupled.KILL_CODE and out["problems"] == []
    assert {slot: out["slots"][slot]["growth"] for slot in SLOTS} == {
        "P2.copy": "none", "P2.recall": "none", "P2.nesting": "uncovered", "P2.composition": "none"}
    recall = out["tasks"]["recall"]
    assert recall["b"] is None and recall["cost_exponent_fixed_size"] == 2.0
    assert any("total cost grows as n^2" in r for r in recall["reasons"])
    assert any("state grows as n^1" in r and "O(n^0.5)" in r for r in recall["reasons"])
    # P1 counted the cost of two rounds; nesting and binding run depth and hops + 1 rounds, never counted.
    assert "never counted" in out["tasks"]["dyck"]["reasons"][0]


def test_k2_the_lstm_reaches_recall_at_k_n_to_the_quarter(results):
    out = results["k2"]
    assert out["status"] == "pass" and out["kill_code"] is None
    recall = out["tasks"]["recall"]
    assert recall["substitutions"]["d"] == "4*K + 8"
    # Cost n d^2 with d = 4K+8 allows K = n^(1/4); the state 2d at most n^(1/2) alone would allow K = n^(1/2).
    assert (recall["growth"], recall["b_exact"], recall["b_cost"], recall["b_state"]) == ("power", "1/4", "1/4", "1/2")
    assert recall["b"] == pytest.approx(0.25) and recall["scale"] == "n^(1/4)"
    assert recall["capacity"]["capped"] is False
    for slot in SLOTS:
        assert (out["slots"][slot]["growth"], out["slots"][slot]["b"]) == ("power", pytest.approx(0.25))


@pytest.mark.parametrize("key", ["k3", "k4"])
def test_k3_and_k4_have_no_growing_skill_since_their_state_holds_only_a_bounded_size(results, key):
    out = results[key]
    assert out["status"] == "fail" and out["kill_code"] == coupled.KILL_CODE
    for slot in SLOTS:
        assert (out["slots"][slot]["growth"], out["slots"][slot]["b"]) == ("bounded", 0.0), slot
    # Cost and state alone would let recall grow; the conservation library's K log2 V bits cap it.
    recall = out["tasks"]["recall"]
    assert recall["capacity"]["capped"] and recall["capacity"]["largest_size_held"] >= 1
    assert recall["reasons"][0].startswith("cost and state alone allow s = n^(1/2)")
    # Binding has no law in the library; the composition slot takes the worse of composition and binding.
    assert out["tasks"]["binding"]["growth"] == "power"
    assert out["slots"]["P2.composition"]["decided_by"] == "composition"


def test_the_state_bound_is_read_from_the_arguments():
    out = measure("k2", max_state_exponent=0.2)
    recall = out["tasks"]["recall"]
    assert (recall["b_exact"], recall["b_state"]) == ("1/5", "1/5")
    assert out["rule"] == {"max_cost_exponent": 1.5, "max_state_exponent": 0.2}
    # A cost bound of n^1 leaves the n d^2 cost no room to grow: b = 0, and the gate fails the card.
    flat = measure("k2", max_cost_exponent=1.0)
    assert flat["status"] == "fail" and all(flat["slots"][s]["b"] == 0.0 for s in SLOTS)


def test_log_growth_is_reported_with_b_zero_and_does_not_pass_the_gate():
    resources = card("k2")["resources"]
    # State sqrt(n) d: at a constant size it sits on the bound, so no power of the task size fits, but log n does.
    out = coupled.evaluate(spec("k2", resources, state="2*d*n**(1/2)"), resources, **BOUNDS)
    recall = out["tasks"]["recall"]
    assert (recall["growth"], recall["b"], recall["scale"]) == ("log", 0.0, "log n (b = 0)")
    assert out["status"] == "fail" and out["kill_code"] == coupled.KILL_CODE
    assert "b > 0" in out["reasons"][0]


def test_a_p1_width_that_differs_from_the_cards_cost_width_fails_with_the_disagreement():
    resources = copy.deepcopy(card("k2")["resources"])
    p1 = spec("k2", resources)
    p1["widths"]["d"] = "2*d"
    out = coupled.evaluate(p1, resources, **BOUNDS)
    assert out["status"] == "fail" and any("resources.cost.d is d" in p for p in out["problems"])


@pytest.mark.parametrize("bad", [0, -1, True, "1.5", None])
def test_the_bounds_must_be_positive_numbers(bad):
    resources = card("k2")["resources"]
    with pytest.raises(ValueError, match="coupled_max_state_exponent"):
        coupled.evaluate(spec("k2", resources), resources, max_cost_exponent=1.5, max_state_exponent=bad)
    with pytest.raises(ValueError, match="coupled_max_cost_exponent"):
        coupled.evaluate(spec("k2", resources), resources, max_cost_exponent=bad, max_state_exponent=0.5)


@pytest.mark.parametrize("task", sorted(constructions.SIZE_NAME))
def test_the_vocabulary_formula_is_the_generator_layout(task):
    import sympy as sp

    s = sp.Symbol(constructions.SIZE_NAME[task], positive=True)
    formula = sp.sympify(coupled.vocab_formula(task), locals={s.name: s, "Max": sp.Max})
    for size in (1, 2, 3, 8, 9, 33, 64):
        assert int(formula.subs(s, size)) == constructions.layout(task, size)["vocab"], (task, size)


@pytest.mark.parametrize("task", sorted(constructions.SIZE_NAME))
def test_the_symbolic_law_is_the_conservation_librarys_value(task):
    import sympy as sp

    s = sp.Symbol(constructions.SIZE_NAME[task], positive=True)
    law = coupled.required_bits(task, s)
    for size in (1, 4, 16, 100):
        value = coupled.required_bits_value(task, size)
        if law is None:
            assert value is None
        else:
            assert float(law.subs(s, size)) == pytest.approx(value), (task, size)
