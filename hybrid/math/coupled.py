"""Capability coupled to cost (FIX-17, decision C17; DISCOVERY-PIPELINE-PLAN.md sections 1.1 and 4.4).

The target gate (target.py) reads the verified P1 formula with every size other than n held fixed. P2 constructions
are proven at widths that grow with the task size: card K2 states a recall width d = 4K+8. A card could claim O(n)
cost at a fixed d and prove recall at K = 64 with d = 264; each statement is true, but they use different resources.
This module judges the skill and the cost at the same resources. It is host arithmetic in SymPy, no model.

For every P2 construction task the card's stated resources for that task (``resources.tasks.<task>``: d, rounds and
state words as formulas in the task size s) are substituted into the verified P1 cost and state formulas:

* the P1 width symbol (the frozen P1 ``widths.d``, which must equal ``resources.cost.d``) becomes the task's d(s);
  when ``resources.cost.d`` is an expression in one size symbol, that symbol is solved for;
* ``V`` becomes the task's vocabulary (the generator layout) unless V is itself the width symbol;
* a size named in ``resources.tasks.<task>.symbols`` becomes that formula, a size in ``resources.dimensions`` its
  formula at d(s); any other size stays at its grid values and the worst setting decides, as in target.py;
* the P1 rounds after substitution must equal the task's rounds. P1 counts the cost of its own rounds only, so a task
  whose rounds differ is ``uncovered``: its cost was never verified.

Then, with s = Theta(n^b), the largest b in [0, b_stream] is computed for which the total cost stays O(n^a_max) and
the state (the P1 state formula and the task's state words) stays O(n^c_max). The bounds are run.toml [target]
``coupled_max_cost_exponent`` (1.5) and ``coupled_max_state_exponent`` (0.5, sublinear state; driver amendment of
2026-10-15 to C17). They differ from the D1 state rule on purpose: recalling K pairs needs at least K log V bits in any
mechanism, so a fixed or O(log n) state would give b = 0 to every card. b_stream = 1 / deg(stream length): the task's
token stream must fit in n. Each formula is reduced to exponent pairs (p_i in n, q_i
in s) of its terms, b follows exactly, and the result is confirmed by direct limits at s = n^b.

A capacity cap comes from the conservation library: the bits the task needs at size s (copy L log2 D, recall
K log2 V, nesting depth, composition T by the INDEX reduction with f a bijection) against the card's carried bits
at d(s) (the carried-state budget when stated, else state words times precision, as the P5 entries derive them).
When the needed bits outgrow the carried bits, only a bounded size fits. Binding has no law in the library and no cap.

Growth classes, worst first: ``none`` (even a constant size breaks the target, as for a transformer),
``uncovered`` (the task's rounds are not the ones P1 verified), ``bounded`` (b = 0: only a bounded size fits),
``log`` (b = 0, but s = Theta(log n) fits; reported, still b = 0), ``power`` (b > 0). A P2 slot takes the worst
class of its tasks.

The gate (C17): a card needs b > 0 on at least one P2 slot; when every slot is ``none``, ``uncovered``, ``bounded`` or
``log`` it fails with ``obligation_failed:target_coupled``. Stage M applies it only when the target gate applies (live runs);
calibration records it. Self-contained (stdlib and SymPy, relative imports) like the other check modules.
"""
from __future__ import annotations

import copy
import functools
import itertools
import json
import math
from fractions import Fraction

import sympy as sp

from . import conservation, constructions
from .cost import asymptotic_exponent, parse_formula
from .target import state_growth

KILL_CODE = "obligation_failed:target_coupled"
GROWTHS = ("none", "uncovered", "bounded", "log", "power")
RANK = {name: index for index, name in enumerate(GROWTHS)}
PASSING = ("power",)
P2_SLOTS = {"P2.copy": ("copy",), "P2.recall": ("recall",), "P2.nesting": ("dyck",),
            "P2.composition": ("composition", "binding")}
SIZE_NAME = dict(constructions.SIZE_NAME)
FUNCTIONS = {"ceiling": sp.ceiling, "floor": sp.floor, "log": sp.log, "sqrt": sp.sqrt, "Max": sp.Max, "Min": sp.Min}
CAPACITY_SEARCH = 4096
STEP_ABOVE = sp.Rational(1, 16)


class CouplingError(ValueError):
    """The formulas cannot be coupled (a stated value cannot be read, or the direct limits disagree with the terms)."""


# ----- the task side, from the generators and the conservation library -----------------------------------------------

def vocab_formula(task: str) -> str:
    """The generator layout's vocabulary size as a formula in the task size (checked against layout() in tests)."""
    p = constructions.DEFAULTS
    if task == "copy":
        return str(p["copy"]["D"] + 1)
    if task == "recall":
        return f"Max(2*K, 16)+{p['recall']['V'] + 1}"
    if task == "dyck":
        return "4"
    if task == "composition":
        return "3*T+3"
    if task == "binding":
        return f"2*hops+{2 + p['binding']['values'] + 1}"
    raise ValueError(f"unknown task {task!r}")


def required_bits(task: str, s: sp.Symbol) -> sp.Expr | None:
    """Bits a one-pass state must hold to answer the task exactly at size s, from the conservation library's laws;
    None when the library has no law for the task (binding)."""
    p = constructions.DEFAULTS
    if task == "copy":
        return s * sp.log(p["copy"]["D"], 2)
    if task == "recall":
        return s * sp.log(p["recall"]["V"], 2)
    if task == "dyck":
        return s
    if task == "composition":
        # f a bijection makes f(g(x)) reveal g(x): INDEX on T items, keys before queries.
        return s * sp.Integer(1)
    return None


def required_bits_value(task: str, size: int) -> float | None:
    """``required_bits`` through the library's own functions, at one size."""
    p = constructions.DEFAULTS
    if task == "copy":
        return conservation.facts_bits(size, math.log2(p["copy"]["D"]))
    if task == "recall":
        return conservation.associative_recall_required_bits(size, p["recall"]["V"])
    if task == "dyck":
        return float(size)
    if task == "composition":
        return conservation.streaming_index_required_bits(size)
    return None


# ----- formulas ---------------------------------------------------------------------------------------------------

def _expr(text, names: dict) -> sp.Expr:
    if isinstance(text, bool):
        raise CouplingError(f"{text!r} is not a formula")
    if isinstance(text, (int, float)):
        return sp.nsimplify(text)
    if not isinstance(text, str) or not text.strip():
        raise CouplingError(f"{text!r} is not a formula")
    try:
        expr = sp.sympify(text, locals=dict(names, **FUNCTIONS))
    except (sp.SympifyError, TypeError, SyntaxError) as exc:
        raise CouplingError(f"{text!r} cannot be read: {exc}") from exc
    unknown = {str(x) for x in expr.free_symbols} - set(names)
    if unknown:
        raise CouplingError(f"{text!r} uses {sorted(unknown)}; allowed: {sorted(names)}")
    return expr


def surrogate(expr: sp.Expr) -> sp.Expr:
    """An expression of the same order for positive arguments: ceiling(x) and floor(x) become x, Max(a, b) becomes
    a + b. Used only for limits; exact values are never read from it."""
    expr = expr.replace(lambda e: isinstance(e, (sp.ceiling, sp.floor)), lambda e: e.args[0])
    return expr.replace(lambda e: isinstance(e, sp.Max), lambda e: sp.Add(*e.args))


def _exponent(expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
    return sp.limit(sp.log(expr) / sp.log(var), var, sp.oo)


def _powers(term: sp.Expr, n: sp.Symbol, s: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
    """(p, q) read off a product of powers of n, s, logarithms and numbers; None for any other factor."""
    p = q = sp.Integer(0)
    for base, power in term.as_powers_dict().items():
        if base.is_number:
            continue
        if not power.is_number:
            return None
        if base == n:
            p += power
        elif base == s:
            q += power
        elif not (isinstance(base, sp.log) and power.is_positive):
            return None
    return p, q


def exponent_pairs(expr: sp.Expr, n: sp.Symbol, s: sp.Symbol) -> list[tuple[sp.Rational, sp.Rational]]:
    """(exponent in n, exponent in s) of every term of the expanded surrogate. With s = n^b a term grows as
    n^(p + q b) up to logarithms, so the formula grows as n^max(p + q b). A term that is a product of powers of n, s,
    logarithms and numbers is read directly; any other term by limits."""
    pairs = []
    for term in sp.Add.make_args(sp.expand(surrogate(expr))):
        magnitude = term.as_coeff_Mul()[1]
        if magnitude.is_number:
            pairs.append((sp.Integer(0), sp.Integer(0)))
            continue
        direct = _powers(magnitude, n, s)
        if direct is not None:
            pairs.append((sp.nsimplify(direct[0]), sp.nsimplify(direct[1])))
            continue
        p = _exponent(magnitude.subs(s, 2), n) if n in magnitude.free_symbols else sp.Integer(0)
        q = _exponent(magnitude.subs(n, 2), s) if s in magnitude.free_symbols else sp.Integer(0)
        if not (p.is_finite and q.is_finite):
            raise CouplingError(f"term {term} has no finite exponent")
        pairs.append((sp.nsimplify(p), sp.nsimplify(q)))
    return pairs


def _growth_at(cost_expr, states, n, s, path) -> tuple[float, list]:
    """The cost exponent in n and the state growths when s follows ``path`` (an expression in n)."""
    cost = surrogate(cost_expr).subs(s, path)
    exponent = float(_exponent(cost, n))
    growths = [state_growth(surrogate(st).subs(s, path), {"n": n}, {}) for st in states]
    return exponent, growths


def _allowed(growth: tuple[str, float], c_max) -> bool:
    """A state growth from target.state_growth within O(n^c_max), up to logarithms."""
    return growth[0] != "power" or growth[1] <= float(c_max) + 1e-9


def _rule_text(c_max) -> str:
    return f"O(n^{float(c_max):g})"


def _fraction(value) -> str:
    frac = Fraction(str(sp.nsimplify(value))).limit_denominator(10 ** 6)
    return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


# ----- one task -------------------------------------------------------------------------------------------------

def _width_substitution(p1_d, size_syms: dict, task_d: sp.Expr, s: sp.Symbol):
    """({P1 size symbol: expression in s}, problem or None) for the width; None as the map means uncovered."""
    free = p1_d.free_symbols
    if not free:
        if sp.simplify(p1_d - task_d) == 0:
            return {}, None
        return None, (f"P1 was verified at the fixed width {p1_d}; the task's width is {task_d}, whose cost P1 "
                      "never counted")
    if len(free) != 1:
        raise CouplingError(f"P1 widths.d {p1_d} uses several sizes {sorted(map(str, free))}; it cannot be matched "
                            "to one task width")
    (symbol,) = free
    if symbol not in size_syms.values():
        raise CouplingError(f"P1 widths.d {p1_d} uses {symbol}, which is not a P1 size")
    if p1_d == symbol:
        return {symbol: task_d}, None
    solutions = [x for x in sp.solve(sp.Eq(p1_d, task_d), symbol) if x.free_symbols <= {s}]
    if len(solutions) != 1:
        raise CouplingError(f"P1 widths.d {p1_d} = task width {task_d} has no single solution for {symbol}")
    return {symbol: solutions[0]}, None


def task_measure(task: str, p1_spec: dict, resources: dict, *, max_cost_exponent: float,
                 max_state_exponent: float) -> dict:
    """The skill scale at target of one P2 construction task (see the module docstring)."""
    size_name = SIZE_NAME[task]
    sizes = dict(p1_spec.get("sizes") or {})
    n = sp.Symbol("n", positive=True)
    s = sp.Symbol(size_name, positive=True)
    size_syms = {name: sp.Symbol(name, positive=True) for name in sizes}
    p1_names = {"n": n, **size_syms}
    stated = (resources.get("tasks") or {}).get(task) or {}
    arch = resources.get("architecture") or {}
    entry = resources.get("cost") or {}
    widths = p1_spec.get("widths") or {}
    a_max = sp.nsimplify(max_cost_exponent)
    c_max = sp.nsimplify(max_state_exponent)
    task_names = {size_name: s}
    task_d = _expr(stated.get("d"), task_names)
    task_rounds = _expr(stated.get("rounds"), task_names)
    task_words = _expr(stated.get("state_words"), task_names)
    p1_d = _expr(widths.get("d", entry.get("d")), p1_names)
    p1_rounds = _expr(widths.get("rounds", entry.get("rounds")), p1_names)
    result = {"task": task, "size": size_name, "d": str(task_d), "rounds": str(task_rounds),
              "state_words": str(task_words), "reasons": []}

    mapping, problem = _width_substitution(p1_d, size_syms, task_d, s)
    if mapping is None:
        return dict(result, growth="uncovered", b=None, scale="uncovered", reasons=[problem])
    width_symbol = next(iter(mapping), None)
    dims = resources.get("dimensions") or {}
    symbols = stated.get("symbols") or {}
    fixed_names = []
    for name, sym in size_syms.items():
        if sym in mapping:
            continue
        if name in symbols:
            mapping[sym] = _expr(symbols[name], task_names)
        elif name in dims:
            mapping[sym] = _expr(dims[name], {"d": sp.Symbol("d", positive=True)}).subs(
                sp.Symbol("d", positive=True), task_d)
        elif name == "V":
            mapping[sym] = _expr(vocab_formula(task), task_names)
        else:
            fixed_names.append(name)
    result["substitutions"] = {str(k): str(v) for k, v in mapping.items()}
    result["held_at_grid"] = fixed_names

    rounds_after = p1_rounds.subs(mapping)
    if sp.simplify(rounds_after - task_rounds) != 0:
        return dict(result, growth="uncovered", b=None, scale="uncovered",
                    reasons=[f"P1 verified the cost of {rounds_after} round(s); the task uses {task_rounds}, whose "
                             "cost P1 never counted"])

    ops = parse_formula(p1_spec["ops"], p1_names)
    state = parse_formula(p1_spec["state"], p1_names)
    positions = constructions.length_formula(task)
    b_stream = sp.Integer(1)
    if positions:
        degree = sp.degree(sp.Poly(_expr(positions, task_names), s))
        b_stream = sp.Rational(1, max(degree, 1))
    result["b_stream"] = _fraction(b_stream)

    settings = [dict(zip(fixed_names, combo)) for combo in itertools.product(*(sizes[k] for k in fixed_names))] or [{}]
    worst = None
    for setting in settings:
        fixed = {size_syms[k]: v for k, v in setting.items()}
        cost_expr = ops.subs(mapping).subs(fixed)
        states = [state.subs(mapping).subs(fixed), task_words]
        measure = _one_setting(cost_expr, states, n, s, a_max, b_stream, c_max)
        measure["setting"] = setting
        if worst is None or _key(measure) < _key(worst):
            worst = measure
    result.update(worst)

    law = required_bits(task, s)
    if law is None:
        result["capacity"] = {"law": None, "note": "the conservation library has no law for this task; no cap"}
    else:
        result["capacity"] = capacity(task, law, s, task_d, task_words, arch)
        if result["capacity"]["capped"] and result["growth"] in PASSING:
            alone = scale_text(result["growth"], result.get("b_exact"))
            result.update(growth="bounded", b=0.0, b_exact="0",
                          reasons=[f"cost and state alone allow s = {alone}", result["capacity"]["reason"]])
    result["scale"] = scale_text(result["growth"], result.get("b_exact"))
    result["width_symbol"] = str(width_symbol) if width_symbol is not None else None
    return result


def _key(measure: dict) -> tuple:
    return RANK[measure["growth"]], measure.get("b") or 0.0


def _one_setting(cost_expr, states, n, s, a_max, b_stream, c_max) -> dict:
    cost_pairs = exponent_pairs(cost_expr, n, s)
    p_max = max(p for p, _ in cost_pairs)
    constant = sp.Integer(8)
    constant_exponent, constant_growths = _growth_at(cost_expr, states, n, s, constant)
    if abs(constant_exponent - float(p_max)) > 1e-9:
        raise CouplingError(f"cost exponent at a constant size is {constant_exponent}, the terms give {p_max}")
    out = {"cost_exponent_fixed_size": float(p_max), "cost_pairs": [[_fraction(p), _fraction(q)]
                                                                    for p, q in cost_pairs]}
    reasons = []
    if constant_exponent > float(a_max) + 1e-9:
        reasons.append(f"at a constant task size the total cost grows as n^{constant_exponent:g}, above the target "
                       f"exponent {float(a_max):g}")
    rule = _rule_text(c_max)
    bad_states = [g for g in constant_growths if not _allowed(g, c_max)]
    if bad_states:
        reasons.append("at a constant task size the state grows as "
                       + ", ".join(f"n^{g[1]:g}" if g[0] == "power" else g[0] for g in bad_states)
                       + f", above the coupled state bound {rule}")
    if reasons:
        return dict(out, growth="none", b=None, b_exact=None, b_cost=None, b_state=None, reasons=reasons)
    growing = [(p, q) for p, q in cost_pairs if q > 0]
    b_cost = min([(a_max - p) / q for p, q in growing] + [b_stream])
    state_pairs = [pair for st in states for pair in exponent_pairs(st, n, s)]
    b_state = min([(c_max - p) / q for p, q in state_pairs if q > 0] + [b_stream])
    b = max(sp.Integer(0), min(b_cost, b_state, b_stream))
    out.update(b_cost=_fraction(max(b_cost, 0)), b_state=_fraction(max(b_state, 0)))
    if b > 0:
        exponent, growths = _growth_at(cost_expr, states, n, s, n ** b)
        if exponent > float(a_max) + 1e-9:
            raise CouplingError(f"at s = n^{b} the cost grows as n^{exponent}, the terms allow it")
        if all(_allowed(g, c_max) for g in growths):
            if b < b_stream:
                above_exp, above_growths = _growth_at(cost_expr, states, n, s, n ** (b + STEP_ABOVE))
                if above_exp <= float(a_max) + 1e-9 and all(_allowed(g, c_max) for g in above_growths):
                    raise CouplingError(f"s = n^{b + STEP_ABOVE} still meets the target; the terms gave b = {b}")
            return dict(out, growth="power", b=float(b), b_exact=_fraction(b),
                        reasons=[f"s = Theta({_power(_fraction(b))}) keeps the cost within n^{float(a_max):g} and the "
                                 f"state within {rule}"])
        out["reasons_power"] = [f"the state leaves {rule} at s = n^b"]
    exponent, growths = _growth_at(cost_expr, states, n, s, sp.log(n))
    why = []
    if exponent > float(a_max) + 1e-9:
        why.append(f"at s = log n the cost grows as n^{exponent:g}")
    if not all(_allowed(g, c_max) for g in growths):
        why.append(f"at s = log n the state leaves {rule}")
    if not why:
        limit_by = "state" if b_state <= b_cost else "cost"
        return dict(out, growth="log", b=0.0, b_exact="0",
                    reasons=[f"no power of n fits ({limit_by} limits b to 0); s = Theta(log n) keeps the cost within "
                             f"n^{float(a_max):g} and the state within {rule}"])
    return dict(out, growth="bounded", b=0.0, b_exact="0", reasons=["only a bounded size fits: " + "; ".join(why)])


def carried_bits(task_d: sp.Expr, task_words: sp.Expr, arch: dict) -> sp.Expr:
    """The card's carried bits at the task's width: the stated carried-state budget, else state words times
    precision (the same derivation as the P5 entries, card_resources._entry_bits)."""
    budget = arch.get("state_bits")
    if budget is not None:
        d = sp.Symbol("d", positive=True)
        return _expr(budget, {"d": d}).subs(d, task_d)
    return task_words * int(arch["precision_bits"])


def capacity(task: str, law: sp.Expr, s: sp.Symbol, task_d, task_words, arch: dict) -> dict:
    bits = carried_bits(task_d, task_words, arch)
    ratio = sp.limit(surrogate(law) / surrogate(bits), s, sp.oo)
    capped = bool(ratio == sp.oo or (ratio.is_finite and ratio > 1))
    largest = 0
    for size in range(1, CAPACITY_SEARCH + 1):
        need = required_bits_value(task, size)
        have = float(bits.subs(s, size))
        if have + conservation.EPS >= need:
            largest = size
        elif capped and size > 64 and largest < size - 64:
            break
    out = {"law": str(law), "carried_bits": str(bits), "ratio_limit": str(ratio), "capped": capped,
           "largest_size_held": largest if capped else None}
    if capped:
        out["reason"] = (f"the needed bits {law} outgrow the carried bits {bits}: the state holds the task only up to "
                         f"{SIZE_NAME[task]} = {largest}")
    return out


def _power(b_exact: str) -> str:
    return f"n^({b_exact})" if "/" in b_exact else f"n^{b_exact}"


def scale_text(growth: str, b_exact) -> str:
    if growth == "power":
        return _power(b_exact)
    if growth == "log":
        return "log n (b = 0)"
    if growth == "bounded":
        return "0 (bounded)"
    return growth


# ----- the card -----------------------------------------------------------------------------------------------------

def evaluate(p1_spec: dict, resources: dict, *, max_cost_exponent: float, max_state_exponent: float) -> dict:
    """The skill scale at target of every P2 task and slot, the resource agreement problems, and the gate. Pure in
    its inputs, so equal inputs share one computation (the limits are the slow part)."""
    key = json.dumps({"spec": {k: p1_spec.get(k) for k in ("ops", "state", "sizes", "widths")},
                      "resources": resources, "a": max_cost_exponent, "c": max_state_exponent}, sort_keys=True)
    return copy.deepcopy(_evaluate_cached(key))


@functools.lru_cache(maxsize=256)
def _evaluate_cached(key: str) -> dict:
    data = json.loads(key)
    return _evaluate({k: v for k, v in data["spec"].items() if v is not None}, data["resources"],
                     max_cost_exponent=data["a"], max_state_exponent=data["c"])


def check_bounds(max_cost_exponent, max_state_exponent) -> None:
    """Both coupled bounds are positive numbers (run.toml [target] coupled_max_cost_exponent and
    coupled_max_state_exponent)."""
    for name, value in (("coupled_max_cost_exponent", max_cost_exponent),
                        ("coupled_max_state_exponent", max_state_exponent)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value!r}")


def _evaluate(p1_spec: dict, resources: dict, *, max_cost_exponent: float, max_state_exponent: float) -> dict:
    check_bounds(max_cost_exponent, max_state_exponent)
    entry = (resources or {}).get("cost") or {}
    widths = p1_spec.get("widths") or {}
    problems = []
    names = {name: sp.Symbol(name, positive=True) for name in ("n", *(p1_spec.get("sizes") or {}))}
    for key in ("d", "rounds"):
        if key in widths and key in entry:
            try:
                if sp.simplify(_expr(widths[key], names) - _expr(entry[key], names)) != 0:
                    problems.append(f"P1 widths.{key} is {widths[key]}; the card's resources.cost.{key} is "
                                    f"{entry[key]}: the cost statement and the resources disagree")
            except CouplingError as exc:
                problems.append(f"P1 widths.{key}: {exc}")
    tasks = {}
    for task in SIZE_NAME:
        try:
            tasks[task] = task_measure(task, p1_spec, resources, max_cost_exponent=max_cost_exponent,
                                       max_state_exponent=max_state_exponent)
        except CouplingError as exc:
            tasks[task] = {"task": task, "growth": "uncovered", "b": None, "scale": "uncovered",
                           "reasons": [f"the formulas could not be coupled: {exc}"]}
    slots = {}
    for slot, members in P2_SLOTS.items():
        worst = min((tasks[t] for t in members), key=_key)
        slots[slot] = {"growth": worst["growth"], "b": worst.get("b"), "scale": worst["scale"],
                       "decided_by": worst["task"]}
    passed = not problems and any(slot_entry["growth"] in PASSING for slot_entry in slots.values())
    reasons = list(problems)
    if not passed and not problems:
        reasons.append("no P2 task's skill grows as a power of n within the coupled bounds (b > 0): every slot is "
                       + ", ".join(f"{slot} {slot_entry['scale']}" for slot, slot_entry in slots.items()))
    return {"status": "pass" if passed else "fail", "kill_code": None if passed else KILL_CODE, "slots": slots,
            "tasks": tasks, "problems": problems, "reasons": reasons,
            "rule": {"max_cost_exponent": max_cost_exponent, "max_state_exponent": max_state_exponent}}
