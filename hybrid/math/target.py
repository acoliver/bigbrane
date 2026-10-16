"""The compute target gate (DISCOVERY-PIPELINE-PLAN.md sections 1.1 and 4.4, decision D1).

The gate is separate from the transformer profile. Profile obligation P1 verifies that a card's cost formula is
exact by running its reference code (cost.py); this gate then compares the verified formula with the compute target,
so calibration can check that the profile describes a known model correctly while the target still rejects it.

Target (D1 defaults, recorded in hybrid/config/run.toml [target] until Andrew decides otherwise):

* total cost over a sequence of length n is O(n^a) with a at most ``max_cost_exponent`` (1.5);
* inference state is fixed in n (``state_growth = "fixed"``) or grows at most as O(log n) (``"log"``). A store that
  grows with the corpus rather than the context is constant in n, so it passes when its size is part of the verified
  state formula; the formula is what P1 checked, so an undeclared store would already have failed P1.

Input is the frozen P1 cost statement ({ops, state, sizes, ...}), used only after its check held. The exponent is
lim log f / log n with the other sizes fixed (cost.asymptotic_exponent), at every listed combination of the other
sizes; the largest value decides. Self-contained (stdlib and SymPy, relative imports) like the other check modules.
"""
from __future__ import annotations

import itertools

import sympy as sp

from .cost import asymptotic_exponent, parse_formula

STATE_RULES = ("fixed", "log")
PASS, FAIL = "pass", "fail"
KILL_CODE = "obligation_failed:target"


def _symbols(spec: dict) -> dict:
    return {name: sp.Symbol(name, positive=True) for name in ("n", *spec.get("sizes", {}))}


def _settings(spec: dict) -> list[dict]:
    sizes = spec.get("sizes") or {}
    names = list(sizes)
    return [dict(zip(names, combo)) for combo in itertools.product(*(sizes[k] for k in names))] or [{}]


def state_growth(expr: sp.Expr, symbols: dict, fixed: dict) -> tuple[str, float]:
    """How a state formula grows in n with the other sizes fixed: ("fixed", 0), ("log", 0) or ("power", a)."""
    n = symbols["n"]
    value = expr.subs({symbols[k]: v for k, v in fixed.items()})
    if n not in value.free_symbols:
        return "fixed", 0.0
    if sp.limit(value / sp.log(n), n, sp.oo).is_finite:
        return "log", 0.0
    return "power", asymptotic_exponent(expr, symbols, fixed)


def evaluate(spec: dict, *, max_cost_exponent: float, state_rule: str) -> dict:
    """Compare a verified P1 cost statement with the target. Returns {status, kill_code, cost_exponent,
    state_growth, state_exponent, reasons, rule}; status is ``pass`` or ``fail``."""
    if state_rule not in STATE_RULES:
        raise ValueError(f"state rule must be one of {STATE_RULES}, got {state_rule!r}")
    if isinstance(max_cost_exponent, bool) or not isinstance(max_cost_exponent, (int, float)) or max_cost_exponent <= 0:
        raise ValueError(f"max_cost_exponent must be a positive number, got {max_cost_exponent!r}")
    symbols = _symbols(spec)
    ops, state = parse_formula(spec["ops"], symbols), parse_formula(spec["state"], symbols)
    settings = _settings(spec)
    exponent = max(asymptotic_exponent(ops, symbols, fixed) for fixed in settings)
    growths = [state_growth(state, symbols, fixed) for fixed in settings]
    order = {"fixed": 0, "log": 1, "power": 2}
    growth, state_exponent = max(growths, key=lambda g: (order[g[0]], g[1]))
    reasons = []
    if exponent > max_cost_exponent + 1e-9:
        reasons.append(f"total cost grows as n^{exponent:g}, above the target exponent {max_cost_exponent:g}")
    allowed = ("fixed",) if state_rule == "fixed" else ("fixed", "log")
    if growth not in allowed:
        shown = f"n^{state_exponent:g}" if growth == "power" else growth
        reasons.append(f"inference state grows as {shown}; the target allows "
                       + ("a fixed state" if state_rule == "fixed" else "a fixed state or O(log n)"))
    status = FAIL if reasons else PASS
    return {"status": status, "kill_code": KILL_CODE if reasons else None, "cost_exponent": exponent,
            "state_growth": growth, "state_exponent": state_exponent, "reasons": reasons,
            "rule": {"max_cost_exponent": max_cost_exponent, "state_growth": state_rule},
            "formula": {"ops": spec["ops"], "state": spec["state"], "sizes": spec.get("sizes") or {}}}
