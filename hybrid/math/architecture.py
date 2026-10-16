"""The shared architecture and resource contract of P1, P2 and P4 (DISCOVERY-PIPELINE-PLAN.md section 4.4, decision
FIX-4, 2026-10-13).

The k1 stage C diagnosis found the nesting witness of card k1 using a feed-forward decoder of hidden width
H = 8 * depth * (depth + 1) (2176 at depth 16) against a model width d = 224, while the P1 cost reference counted
square d-by-d matrices. Nothing compared the two: P1 counted one architecture and P2 ran another. This module makes the
architecture explicit and checks every witness against it.

The contract has two parts.

- Shared, frozen with the profile (``profile.architecture``): ``hidden``, the feed-forward hidden width as a formula
  in d (``"d"`` for square matrices, ``"4*d"`` for a 4x expansion), and ``precision_bits``, the float width of stored
  numbers (32 for binary32).
- Optional in the shared contract (FIX-8): ``state_bits``, the information budget of the whole carried state of a
  finite-state mechanism, as a number or a formula in d. It is separate from ``precision_bits``: precision is how
  wide stored floats are, ``state_bits`` is how many distinct configurations the carried state may take
  (2**state_bits). With it, a P2 construction's carried state is audited at initialization and after every token.
- Per statement (``widths`` in a P1 cost, P2 construction or P4 learnability statement): ``d`` and ``rounds`` as
  formulas in the statement's own size names (n and the cost sizes for P1, the task size for P2 and P4), and
  optionally ``positions``, the most positions a table may hold (default: n for P1, the longest token stream the
  task generator makes at that size for P2 and P4).

The check surveys every array a witness holds (the module globals of the source and the state or parameters it
returns) and every list, tuple or dict of arrays. Each axis must be at most the largest declared dimension,
max(d, hidden, rounds, positions, vocab). A table that keeps one row per hidden unit cannot be wider than the
declared hidden width, a list of per-unit vectors counts its length as an axis, and a float array stored at more bits
than ``precision_bits`` breaks the precision. A P1 cost reference must also realize the declared hidden width when
it differs from d: some array has an axis of exactly that width, so the counted cost is the cost of the declared
architecture.

The check is a resource audit, not a proof of faithfulness: it does not show that a witness uses the card's update
rule. It closes the specific gap the diagnosis found, a construction holding a hidden width larger than the costed
one.

Self-contained (stdlib, NumPy, SymPy) like the other check modules.
"""
from __future__ import annotations

import ast
import math
from types import ModuleType, SimpleNamespace

import numpy as np
import sympy as sp

SHARED_FIELDS = ("hidden", "precision_bits")
# FIX-8: optional in the shared contract. The information budget of the carried state of a finite-state mechanism.
OPTIONAL_SHARED_FIELDS = ("state_bits",)
WIDTH_FIELDS = ("d", "rounds", "positions")
PRECISIONS = (16, 32, 64)
FUNCTIONS = {"ceiling": sp.ceiling, "floor": sp.floor, "log": sp.log, "sqrt": sp.sqrt, "Max": sp.Max, "Min": sp.Min}
# How deep the survey walks nested containers, and how many offending arrays a report names.
MAX_DEPTH = 8
SHOWN = 4


class ContractError(ValueError):
    """The contract or a width formula cannot be read."""


# ----- formulas ------------------------------------------------------------------------------------------------------

def formula_problems(text, names) -> list[str]:
    """Why ``text`` is not a formula over ``names`` (numbers, + - * / **, and the functions in FUNCTIONS)."""
    if not isinstance(text, str) or not text.strip():
        return ["must be a non-empty formula"]
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        return [f"{text!r} is not a formula: {exc.msg}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in names and node.id not in FUNCTIONS:
                problems.append(f"{text!r} uses {node.id!r}; allowed names: {sorted(names)}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                problems.append(f"{text!r}: only {sorted(FUNCTIONS)} may be called")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load, ast.Add, ast.Sub,
                                   ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.FloorDiv)):
            problems.append(f"{text!r}: {type(node).__name__} is not allowed in a width formula")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            problems.append(f"{text!r}: only numeric constants are allowed")
    return list(dict.fromkeys(problems))


def value(text: str, values: dict) -> float:
    """``text`` evaluated at ``values`` ({name: number})."""
    problems = formula_problems(text, set(values))
    if problems:
        raise ContractError("; ".join(problems))
    symbols = {name: sp.Symbol(name, positive=True) for name in values}
    expr = sp.sympify(text, locals=dict(symbols, **FUNCTIONS))
    result = expr.subs({symbols[name]: sp.Integer(v) if float(v).is_integer() else sp.Rational(str(v))
                        for name, v in values.items()})
    return float(result)


def shared_problems(shared, size_names=()) -> list[str]:
    """Problems with the profile's shared contract; ``size_names`` are names ``hidden`` may use besides d."""
    if not isinstance(shared, dict) or not set(SHARED_FIELDS) <= set(shared) \
            or set(shared) - set(SHARED_FIELDS) - set(OPTIONAL_SHARED_FIELDS):
        return [f"architecture must be {{{', '.join(SHARED_FIELDS)}}} with an optional "
                f"{', '.join(OPTIONAL_SHARED_FIELDS)}"]
    problems = [f"architecture.hidden: {p}" for p in formula_problems(shared["hidden"], {"d", *size_names})]
    if shared["precision_bits"] not in PRECISIONS:
        problems.append(f"architecture.precision_bits must be one of {list(PRECISIONS)}")
    bits = shared.get("state_bits")
    if isinstance(bits, bool) or not (bits is None or isinstance(bits, (int, float, str))):
        problems.append("architecture.state_bits must be a number, a formula in d, or null")
    elif isinstance(bits, (int, float)) and not (math.isfinite(bits) and 0 <= bits <= MAX_STATE_BITS):
        problems.append(f"architecture.state_bits must be between 0 and {MAX_STATE_BITS}")
    elif isinstance(bits, str):
        problems += [f"architecture.state_bits: {p}" for p in formula_problems(bits, {"d", *size_names})]
    return problems


# ----- the finite carried-state budget (FIX-8) -----------------------------------------------------------------------
#
# ``precision_bits`` is the float width of stored numbers. It says nothing about how much information the carried
# state holds: a float64 parameter table does not give a sixteen-state machine 64 bits, and one Python integer can
# hold any number of bits. A mechanism whose carried state is a finite discrete set (a state index, a finite automaton,
# a fixed-width code) declares ``state_bits``, the budget of the whole carried state. The construction harness then
# holds the state returned by init and by every update to it (constructions.CarriedStateAudit): integers only, a
# packed width within the budget (one index z needs 0 <= z < 2**state_bits), and at most 2**state_bits distinct
# carried configurations at each size. Card k3's sixteen-state HMM (state_bits = 4) is the case that motivated it
# (results/phase4-20261013T073748Z/k3-p5-diag/REPORT.md).

# Larger budgets are not finite-state cards in any useful sense; the audit keeps one set of seen configurations.
MAX_STATE_BITS = 32


def state_bits(shared: dict | None, values: dict) -> float | None:
    """The declared carried-state budget in bits at one point (``values`` holds d), or None when the contract
    declares none (a mechanism with continuous state)."""
    bits = (shared or {}).get("state_bits")
    if bits is None:
        return None
    number = value(bits, values) if isinstance(bits, str) else float(bits)
    if not math.isfinite(number) or not 0 <= number <= MAX_STATE_BITS:
        raise ContractError(f"declared state_bits is {number} at {values}; it must be between 0 and "
                            f"{MAX_STATE_BITS}")
    return number


def latent_states(bits: float) -> int:
    """How many distinct carried configurations ``bits`` bits allow: floor(2**bits), so 4 bits allow 16 and
    log2(10) bits allow 10."""
    return int(math.floor(2.0 ** bits + 1e-6))


def widths_problems(widths, names) -> list[str]:
    """Problems with a statement's ``widths``; ``names`` are the statement's own size names."""
    if not isinstance(widths, dict) or not {"d", "rounds"} <= set(widths) or set(widths) - set(WIDTH_FIELDS):
        return ["widths must be {\"d\": formula, \"rounds\": formula} with an optional \"positions\" formula"]
    problems = []
    for key in WIDTH_FIELDS:
        if widths.get(key) is not None:
            problems += [f"widths.{key}: {p}" for p in formula_problems(widths[key], set(names))]
    return problems


def dimensions(shared: dict, widths: dict, values: dict, *, positions: int, vocab: int = 0) -> dict:
    """The declared dimensions at one point: d, hidden, rounds, positions, vocab and their maximum ``bound``.
    ``values`` holds the statement's size names; ``positions`` is the default when the widths give none."""
    d = value(widths["d"], values)
    rounds = value(widths["rounds"], values)
    hidden = value(shared["hidden"], dict(values, d=d))
    pos = value(widths["positions"], values) if widths.get("positions") else positions
    dims = {"d": d, "hidden": hidden, "rounds": rounds, "positions": float(pos), "vocab": float(vocab)}
    for key, number in dims.items():
        if not math.isfinite(number) or number < 0:
            raise ContractError(f"declared {key} is {number} at {values}")
    return dict(dims, bound=max(dims.values()))


# ----- the survey ----------------------------------------------------------------------------------------------------

def _raw(obj):
    """The NumPy array behind ``obj``: itself, or the array a counted wrapper of cost.py holds."""
    if isinstance(obj, np.ndarray):
        return obj
    inner = getattr(obj, "_a", None)
    return inner if isinstance(inner, np.ndarray) else None


def _array(obj) -> bool:
    return _raw(obj) is not None


def survey(**roots) -> dict:
    """Every array reachable from ``roots`` ({label: value}), with its path, shape and float bits, and every
    container of arrays with its length. Functions, modules and classes are skipped."""
    arrays: list[dict] = []
    containers: list[dict] = []
    seen: set[int] = set()

    def walk(obj, path: str, depth: int):
        if depth > MAX_DEPTH or id(obj) in seen:
            return
        if callable(obj) and not _array(obj) or isinstance(obj, (ModuleType, type)):
            return
        if _array(obj):
            seen.add(id(obj))
            obj = _raw(obj)
            dtype = np.dtype(obj.dtype)
            bits = dtype.itemsize * 8 if dtype.kind in "fc" else None
            if dtype.kind == "c" and bits:
                bits //= 2
            arrays.append({"path": path, "shape": [int(s) for s in obj.shape], "float_bits": bits})
            return
        if isinstance(obj, (list, tuple)):
            seen.add(id(obj))
            if any(_array(item) for item in obj):
                containers.append({"path": path, "length": len(obj)})
            for index, item in enumerate(obj):
                walk(item, f"{path}[{index}]", depth + 1)
        elif isinstance(obj, dict):
            seen.add(id(obj))
            if any(_array(item) for item in obj.values()):
                containers.append({"path": path, "length": len(obj)})
            for key, item in obj.items():
                if isinstance(key, str) and key.startswith("__"):
                    continue
                walk(item, f"{path}[{key!r}]", depth + 1)
        elif isinstance(obj, SimpleNamespace):
            walk(vars(obj), path, depth + 1)

    for label, root in roots.items():
        walk(root, label, 0)
    return {"arrays": arrays, "containers": containers}


def namespace_values(namespace: dict) -> dict:
    """The data a loaded source keeps at module level (its globals without builtins, modules and functions)."""
    return {k: v for k, v in namespace.items()
            if not k.startswith("__") and k != "np" and not isinstance(v, (ModuleType, type)) and not callable(v)}


def violations(dims: dict, found: dict, precision_bits: int | None, *, realize_hidden: bool = False) -> list[str]:
    """Where the surveyed arrays break the declared dimensions at one point."""
    bound = dims["bound"] + 1e-9
    problems = []
    wide = sorted(((max(a["shape"]), a) for a in found["arrays"] if a["shape"] and max(a["shape"]) > bound),
                  key=lambda item: -item[0])
    if wide:
        shown = ", ".join(f"{a['path']} shape {tuple(a['shape'])}" for _, a in wide[:SHOWN])
        problems.append(f"{len(wide)} array(s) have an axis above the largest declared dimension {_g(dims['bound'])} "
                        f"(d={_g(dims['d'])}, hidden={_g(dims['hidden'])}, rounds={_g(dims['rounds'])}, "
                        f"positions={_g(dims['positions'])}, vocab={_g(dims['vocab'])}): {shown}")
    long = [c for c in found["containers"] if c["length"] > bound]
    if long:
        shown = ", ".join(f"{c['path']} holds {c['length']} arrays" for c in long[:SHOWN])
        problems.append(f"containers of arrays longer than the largest declared dimension {_g(dims['bound'])}: "
                        f"{shown}")
    if precision_bits is not None:
        precise = [a for a in found["arrays"] if a["float_bits"] and a["float_bits"] > precision_bits]
        if precise:
            shown = ", ".join(f"{a['path']} at {a['float_bits']} bits" for a in precise[:SHOWN])
            problems.append(f"{len(precise)} float array(s) stored above the declared {precision_bits}-bit "
                            f"precision: {shown}")
    if realize_hidden and abs(dims["hidden"] - dims["d"]) > 1e-9:
        target = round(dims["hidden"])
        if not any(target in a["shape"] for a in found["arrays"]):
            problems.append(f"no array has an axis of the declared hidden width {target}: the cost reference does not "
                            "realize the declared architecture")
    return problems


def _g(number: float) -> str:
    return f"{number:g}"


def check_point(shared: dict, widths: dict, values: dict, found: dict, *, positions: int, vocab: int = 0,
                precision: bool = True, realize_hidden: bool = False) -> dict:
    """One point's record: the declared dimensions, the widest array, its float bits and the violations."""
    dims = dimensions(shared, widths, values, positions=positions, vocab=vocab)
    widest = max((max(a["shape"]) for a in found["arrays"] if a["shape"]), default=0)
    bits = max((a["float_bits"] or 0 for a in found["arrays"]), default=0)
    problems = violations(dims, found, shared["precision_bits"] if precision else None,
                          realize_hidden=realize_hidden)
    return {"point": dict(values), "dimensions": {k: _g(v) for k, v in dims.items()}, "widest_axis": widest,
            "float_bits": bits or None, "arrays": len(found["arrays"]), "violations": problems}


def summary(points: list[dict]) -> dict:
    """The ``architecture`` detail a check reports: checked, every point and the first violations."""
    bad = [p for p in points if p["violations"]]
    return {"checked": True, "points": points, "violations": [f"at {p['point']}: {v}" for p in bad
                                                               for v in p["violations"]][:SHOWN]}
