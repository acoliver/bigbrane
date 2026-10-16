"""Checks of frozen formal statements (DISCOVERY-PIPELINE-PLAN.md sections 4.1, 4.2 and 5.4), run in the tools image.

The statement fixer writes each statement as data, never as a program; this module turns the data into a check, so
the code that decides lives here (hashed in MANIFEST.json) and no model writes it.

* ``z3_property(spec)``: a property over bounded variables. Every int and real variable has a lower and an upper
  bound; the negation of the property is asserted under the bounds and hypotheses. ``unsat`` is discharged
  (tier ``smt_bounded``, proven up to the stated bound), ``sat`` is violated with the model as counterexample.
* ``sympy_relation(spec)``: ``lhs <relation> rhs`` over typed symbols. An identity is discharged when
  ``simplify(lhs - rhs) == 0`` and 20 random rational points agree (tier ``sympy_exact``); an inequality when SymPy
  proves the sign of the difference under the symbol assumptions and the points agree. A point where the relation
  fails is a counterexample.
* ``simulation(spec, source)``: the fixer's ``generate(size, rng)`` makes ``{"input", "target"}`` instances, the
  prover's ``run(input)`` answers, the fixer's ``metric(given, target, output)`` scores in [0, 1]; discharged when
  the mean meets the threshold at every listed size (tier ``executed``, observed on the listed sizes).
* ``audited_construction(spec, source)``: the construction harness of constructions.py on prover code that passed
  the cost verifier's source audit.
* ``audited_construction`` also compares the peak state words the harness measured at every size with the
  ``state_words`` formula when the statement declares one (Phase 4: the resources a P2 construction needs as a
  function of its size); more words than declared is a violation. With the profile's architecture contract
  (FIX-4, architecture.py) every array the mechanism holds is checked against the statement's declared widths and
  the shared hidden width and precision at every size; an array wider than declared is a violation. When the contract
  declares ``state_bits`` (FIX-8), the harness audits the carried state at initialization and after every token
  against that budget and counts its distinct configurations; module-level data is fixed after init in every
  construction. A size that breaks any of these resource checks has ``passed`` False and names the breach in
  ``resource_violations``, so a resource-invalid size never counts as a P2 success.
* ``conservation`` statements (Phase 4, P5) are the card's numbers for the conservation laws, checked by
  conservation.check_claims in the job; ``learnability`` statements (P4) run learnability.run_learnability.
* ``reference_conformance(spec, source)`` (FIX-6): the cost reference that held for the statement ``reference``
  and the fixer's frozen ``expected(params, sizes, tokens)`` run on the same parameters over repeated, distinct and
  random token streams; the retained state and the output must agree after every token. It settles a premise that
  the reference executes the card's own equations and buffers, which equal operation and state counts do not.
  FIX-20 (decision PR): both sides run under one precision policy, the statement's ``precision_bits`` (the card's
  declared precision, frozen by the host): the reference in ``cost.OpCounter(precision_bits)``, the equations with
  an ``np`` whose float constructors default to that dtype, on the reference's parameters at that dtype. Equations
  that name another float dtype are refused, and a state or output the equations return at another float dtype makes
  the audit unknown. The points and the tolerance are protocol values the host freezes (profile.PROTOCOL).
* ``predicate_agreement(spec, predicate)``: the statement-match harness for finite domains. The z3 property and a
  Python predicate written from the natural-language claim are evaluated on random assignments inside the bounds;
  any disagreement is a mismatch.

Expressions are parsed with ``ast`` and evaluated by a whitelist walker (numbers, the declared names, arithmetic,
one comparison, calls to listed functions), never by ``eval``. Model-written functions (generator, metric, run,
predicate) pass ``cost.audit_source`` (no imports, no private attributes, no eval or getattr) and run with the
cost verifier's safe builtins. Every result follows the obligation-check contract (status, evidence,
counterexample) and adds its tier. Stdlib, NumPy and SymPy at the top level; z3 is imported where it is used.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from fractions import Fraction

import numpy as np

from . import conservation, constructions, cost, findings, learnability, tiers
from .cost import SAFE_BUILTINS, CandidateError, audit_source, crash_line, describe_exception

DISCHARGED, VIOLATED, UNKNOWN = "discharged", "violated", "unknown"
Z3_SORTS = ("int", "real", "bool")
Z3_FUNCS = ("And", "Or", "Not", "Implies", "If", "Xor", "Distinct", "Sum", "Product", "ToReal", "ToInt", "IsInt")
SYMPY_DOMAINS = ("real", "positive", "nonnegative", "integer", "positive_integer", "nonnegative_integer")
SYMPY_FUNCS = ("sqrt", "log", "exp", "Abs", "Min", "Max", "floor", "ceiling", "binomial", "factorial", "Rational",
               "sin", "cos")
SYMPY_CONSTANTS = ("pi", "E")
RELATIONS = ("==", "<=", ">=", "<", ">")
SPOT_POINTS = 20
DEFAULT_Z3_TIMEOUT_MS = 60_000
MAX_SIMULATION_COUNT = 1000
MAX_SIZES = 8
NAME = r"^[A-Za-z][A-Za-z0-9_]{0,30}$"

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b}
_CMPOPS = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
           ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}


class SpecError(ValueError):
    """A frozen statement that cannot be read as written (bad expression, unknown name, missing bound)."""


def spec_hash(spec: dict) -> str:
    """sha256 of a statement's canonical JSON. The stage records it when the statement is frozen."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


# ----- expressions ---------------------------------------------------------------------------------------------------

def expression_problems(text: str, names, funcs, *, allow_compare: bool) -> list[str]:
    """Why ``text`` is not an expression the walker evaluates: syntax, node types, unknown names, chained or
    disallowed comparisons, calls to anything but a listed function."""
    if not isinstance(text, str) or not text.strip():
        return ["empty expression"]
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        return [f"syntax error in {text!r}: {exc.msg}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)) or type(node) in _BINOPS or type(node) in _CMPOPS:
            continue
        if isinstance(node, (ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                problems.append(f"operator {type(node.op).__name__} is not allowed")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                problems.append(f"unary {type(node.op).__name__} is not allowed (use Not(...))")
        elif isinstance(node, ast.Compare):
            if not allow_compare:
                problems.append("comparisons are not allowed here")
            elif len(node.ops) != 1:
                problems.append("chained comparisons are not allowed (use And(a <= x, x <= b))")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in funcs:
                problems.append(f"call to {ast.unparse(node.func)!r} is not allowed; functions: {sorted(funcs)}")
            if node.keywords:
                problems.append("keyword arguments are not allowed")
        elif isinstance(node, ast.Name):
            if node.id not in names and node.id not in funcs:
                problems.append(f"unknown name {node.id!r}")
        elif isinstance(node, ast.Constant):
            if not (isinstance(node.value, bool) or _number(node.value)):
                problems.append(f"constant {node.value!r} is not a number")
        elif isinstance(node, ast.BoolOp):
            problems.append("and/or are not allowed (use And(...), Or(...))")
        else:
            problems.append(f"{type(node).__name__} is not allowed")
    return list(dict.fromkeys(problems))


def evaluate(text: str, env: dict, funcs: dict, const):
    """Evaluate a checked expression: names from ``env``, calls from ``funcs``, numbers through ``const``."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return const(node.value)
        if isinstance(node, ast.Name):
            return env[node.id] if node.id in env else funcs[node.id]
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Compare):
            return _CMPOPS[type(node.ops[0])](walk(node.left), walk(node.comparators[0]))
        if isinstance(node, ast.Call):
            return funcs[node.func.id](*[walk(arg) for arg in node.args])
        raise SpecError(f"cannot evaluate {type(node).__name__}")

    return walk(ast.parse(text.strip(), mode="eval"))


# ----- validation of statement data (used on the host before freezing and again in the job) --------------------------

def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def z3_problems(spec: dict) -> list[str]:
    problems = []
    variables = spec.get("variables")
    if not isinstance(variables, dict) or not variables:
        return ["z3: variables must be a non-empty object"]
    for name, var in variables.items():
        if not re.match(NAME, name) or name in Z3_FUNCS:
            problems.append(f"z3: bad variable name {name!r}")
            continue
        if not isinstance(var, dict) or var.get("sort") not in Z3_SORTS:
            problems.append(f"z3: variable {name} needs sort in {Z3_SORTS}")
            continue
        if var["sort"] != "bool":
            lo, hi = var.get("lo"), var.get("hi")
            if not (_number(lo) and _number(hi)) or lo > hi:
                problems.append(f"z3: variable {name} needs numeric bounds lo <= hi (the statement must be bounded)")
    names = set(variables)
    for index, text in enumerate(spec.get("hypotheses") or []):
        problems += [f"z3: hypothesis {index}: {p}" for p in expression_problems(text, names, Z3_FUNCS,
                                                                                 allow_compare=True)]
    problems += [f"z3: property: {p}" for p in expression_problems(spec.get("property", ""), names, Z3_FUNCS,
                                                                    allow_compare=True)]
    return problems


def sympy_problems(spec: dict) -> list[str]:
    problems = []
    symbols = spec.get("symbols")
    if not isinstance(symbols, dict) or not symbols:
        return ["sympy: symbols must be a non-empty object"]
    for name, domain in symbols.items():
        if not re.match(NAME, name) or name in SYMPY_FUNCS or name in SYMPY_CONSTANTS:
            problems.append(f"sympy: bad symbol name {name!r}")
        if domain not in SYMPY_DOMAINS:
            problems.append(f"sympy: symbol {name} has domain {domain!r}; known: {SYMPY_DOMAINS}")
    if spec.get("relation") not in RELATIONS:
        problems.append(f"sympy: relation must be one of {RELATIONS}")
    names = set(symbols) | set(SYMPY_CONSTANTS)
    for side in ("lhs", "rhs"):
        problems += [f"sympy: {side}: {p}" for p in expression_problems(spec.get(side, ""), names, SYMPY_FUNCS,
                                                                         allow_compare=False)]
    for index, text in enumerate(spec.get("conditions") or []):
        problems += [f"sympy: condition {index}: {p}" for p in expression_problems(text, names, SYMPY_FUNCS,
                                                                                   allow_compare=True)]
    return problems


def _code_problems(label: str, source, names: tuple) -> list[str]:
    if not isinstance(source, str) or not source.strip():
        return [f"{label}: code is missing"]
    problems = [f"{label}: {p}" for p in audit_source(source)]
    if not problems:
        tree = ast.parse(source)
        defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        problems += [f"{label}: does not define {name}()" for name in names if name not in defined]
    return problems


def simulation_problems(spec: dict) -> list[str]:
    problems = _code_problems("simulation generator", spec.get("generator"), ("generate",))
    problems += _code_problems("simulation metric", spec.get("metric"), ("metric",))
    sizes = spec.get("sizes")
    if not isinstance(sizes, list) or not sizes or len(sizes) > MAX_SIZES or not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in sizes):
        problems.append(f"simulation: sizes must be 1 to {MAX_SIZES} non-negative integers")
    count = spec.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_SIMULATION_COUNT:
        problems.append(f"simulation: count must be an integer from 1 to {MAX_SIMULATION_COUNT}")
    threshold = spec.get("threshold")
    if not _number(threshold) or not 0.0 <= threshold <= 1.0:
        problems.append("simulation: threshold must be a number in [0, 1]")
    return problems


def construction_problems(spec: dict) -> list[str]:
    problems = []
    if spec.get("task") not in constructions.SIZE_NAME:
        problems.append(f"construction: task must be one of {sorted(constructions.SIZE_NAME)}")
    sizes = spec.get("sizes")
    if not isinstance(sizes, list) or not sizes or len(sizes) > MAX_SIZES or not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 1 for s in sizes):
        problems.append(f"construction: sizes must be 1 to {MAX_SIZES} positive integers")
    threshold = spec.get("threshold", constructions.DEFAULT_THRESHOLD)
    if not _number(threshold) or not 0.0 <= threshold <= 1.0:
        problems.append("construction: threshold must be a number in [0, 1]")
    params = spec.get("params")
    if params is not None and not isinstance(params, dict):
        problems.append("construction: params must be an object or null")
    if spec.get("state_words") is not None and spec.get("task") in constructions.SIZE_NAME:
        name = constructions.SIZE_NAME[spec["task"]]
        problems += [f"construction: state_words: {p}" for p in expression_problems(
            spec["state_words"], {name}, SYMPY_FUNCS, allow_compare=False)]
    if spec.get("widths") is not None and spec.get("task") in constructions.SIZE_NAME:
        from . import architecture
        problems += [f"construction: {p}" for p in architecture.widths_problems(
            spec["widths"], {constructions.SIZE_NAME[spec["task"]]})]
    if problems:
        return problems
    for size in sizes:
        try:
            constructions.layout(spec["task"], size, params)
        except ValueError as exc:
            problems.append(f"construction: {exc} (at {constructions.SIZE_NAME[spec['task']]}={size})")
    if not problems and spec.get("token_store") is not None:
        problems += token_store_problems(spec)
    return problems


def size_formula(spec: dict, text: str, size: int) -> float:
    """A formula in the construction's size variable (L, K, depth, T or hops), evaluated at ``size``."""
    sp, _, funcs, const = _sympy_env({"symbols": {"x": "positive"}})
    name = constructions.SIZE_NAME[spec["task"]]
    return float(evaluate(text, {name: sp.Integer(size)}, funcs, const))


def state_words_bound(spec: dict, size: int) -> float:
    """The declared peak state words of a construction at ``size`` (its ``state_words`` formula in the size)."""
    return size_formula(spec, spec["state_words"], size)


def token_store_problems(spec: dict) -> list[str]:
    """A construction that keeps rows per input token declares ``token_store`` = {"arrays": a, "tokens": formula,
    "width": formula}: a arrays of ``tokens`` rows of ``width`` words. Before freezing, the declared capacity is
    checked against the generator's real token streams (the instances the check will run: its count and seed) and the
    stores alone against the declared state words. In the 2026-10-13 smoke card k1 froze capacities of 4T+3 and
    3hops+3 tokens against streams of 5T+3 and 4hops+7, and at T=4 its four stores alone needed 4*23*96 = 8832 words
    against a declared 8300, so no faithful construction could meet the statement."""
    store = spec["token_store"]
    task, name = spec["task"], constructions.SIZE_NAME[spec["task"]]
    if not isinstance(store, dict) or set(store) != {"arrays", "tokens", "width"}:
        return ['construction: token_store must be {"arrays": integer, "tokens": formula, "width": formula}']
    problems = []
    arrays = store["arrays"]
    if not isinstance(arrays, int) or isinstance(arrays, bool) or arrays < 1:
        problems.append("construction: token_store.arrays must be a positive integer")
    for key in ("tokens", "width"):
        problems += [f"construction: token_store.{key}: {p}" for p in expression_problems(
            store[key], {name}, SYMPY_FUNCS, allow_compare=False)]
    if problems:
        return problems
    count = spec.get("count") or constructions.DEFAULT_COUNT
    seed = spec.get("seed") or 0
    real = {size: constructions.stream_lengths(task, size, count, seed, spec.get("params"))[1]
            for size in spec["sizes"]}
    formula = constructions.length_formula(task, spec.get("params"))
    lengths = (", ".join(f"{name}={size}: {length}" for size, length in real.items())
               + (f"; {formula}" if formula else "; the longest of the check's instances"))
    for size, length in real.items():
        tokens, width = size_formula(spec, store["tokens"], size), size_formula(spec, store["width"], size)
        if width < 1 or width != int(width):
            problems.append(f"construction: token_store.width {store['width']} = {width:g} at {name}={size} is not a "
                            "positive integer")
            continue
        if tokens < length:
            problems.append(f"construction: token_store.tokens {store['tokens']} = {tokens:g} at {name}={size}, but "
                            f"the {task} generator feeds {length} tokens there (real stream lengths {lengths}); a store "
                            "of every input position needs a row for each of them")
        if spec.get("state_words") is not None:
            need = arrays * length * width
            declared = state_words_bound(spec, size)
            if need > declared:
                problems.append(f"construction: the token stores alone need arrays x tokens x width = {arrays} x "
                                f"{length} x {width:g} = {need:g} words at {name}={size}, above state_words "
                                f"{spec['state_words']} = {declared:g}; state_words must include every stored row")
    return problems



def conservation_problems(spec: dict) -> list[str]:
    """A conservation statement must be numbers the library can read (conservation.check_claims raises otherwise)."""
    if not spec:
        return ["conservation: the statement holds no numbers"]
    try:
        checks = conservation.check_claims(spec)
    except (ValueError, TypeError, KeyError) as exc:
        return [f"conservation: {exc}"]
    return [] if checks else ["conservation: no law applies to these numbers"]


def cost_problems(spec: dict) -> list[str]:
    problems = []
    for key in ("ops", "state"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            problems.append(f"cost: {key} formula is missing")
    sizes = spec.get("sizes", {})
    if not isinstance(sizes, dict) or not all(isinstance(v, list) and len(v) >= 2 for v in sizes.values()):
        problems.append("cost: sizes must map every size name other than n to at least two values")
    exponent = spec.get("exponent")
    if exponent is not None and not _number(exponent):
        problems.append("cost: exponent must be a number or null")
    for key in ("ops_remainder_lower_in", "state_remainder_lower_in"):
        value = spec.get(key)
        if value is not None and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            problems.append(f"cost: {key} must be a list of size names or null")
    if spec.get("widths") is not None:
        from . import architecture
        problems += [f"cost: {p}" for p in architecture.widths_problems(spec["widths"], {"n", *sizes})]
    if problems:
        return problems
    claim = cost.CostClaim(**{k: spec.get(k) for k in cost.CLAIM_FIELDS if k != "sizes"}, sizes=sizes)
    ns = spec.get("ns") or cost.DEFAULT_NS
    return [f"cost: {problem}" for problem in cost.remainder_problems(claim, tuple(ns))]


CONFORMANCE_STREAMS = ("repeated", "distinct", "random")
# FIX-20: the protocol grid is every combination of the cost grid's values (profile.PROTOCOL), up to three cost sizes.
MAX_CONFORMANCE_POINTS = 8
MAX_CONFORMANCE_N = 256
MAX_CONFORMANCE_TOLERANCE = 1e-6
REFERENCE_ID = r"^O[0-9]{1,2}\.[abc]$"


def conformance_problems(spec: dict) -> list[str]:
    """A conformance statement (FIX-6): the frozen ``equations`` (``expected(params, sizes, tokens)``, written by the
    statement fixer from the card's equations and its frozen definitions), the cost statement whose reference source
    it audits, the size points (each with n) and the token streams. Whether ``reference`` names a cost statement of
    the same record is checked by the host before freezing."""
    problems = _code_problems("conformance equations", spec.get("equations"), ("expected",))
    if not isinstance(spec.get("reference"), str) or not re.match(REFERENCE_ID, spec["reference"]):
        problems.append("conformance: reference must be the id of a cost statement, such as O4.a")
    points = spec.get("points")
    if not isinstance(points, list) or not 1 <= len(points) <= MAX_CONFORMANCE_POINTS:
        problems.append(f"conformance: points must list 1 to {MAX_CONFORMANCE_POINTS} size points")
    else:
        for index, point in enumerate(points):
            if not isinstance(point, dict) or not all(
                    isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 1
                    for k, v in point.items()):
                problems.append(f"conformance: point {index} must map size names to positive integers")
            elif not 1 <= point.get("n", 0) <= MAX_CONFORMANCE_N:
                problems.append(f"conformance: point {index} needs n from 1 to {MAX_CONFORMANCE_N}")
    streams = spec.get("streams")
    if not isinstance(streams, list) or not streams or not set(streams) <= set(CONFORMANCE_STREAMS):
        problems.append(f"conformance: streams must be a non-empty list from {list(CONFORMANCE_STREAMS)}")
    tolerance = spec.get("tolerance")
    if not _number(tolerance) or not 0.0 <= tolerance <= MAX_CONFORMANCE_TOLERANCE:
        problems.append(f"conformance: tolerance must be a number in [0, {MAX_CONFORMANCE_TOLERANCE:g}] (the host "
                        "sets it from the protocol)")
    bits = spec.get("precision_bits")
    if isinstance(bits, bool) or bits not in cost.FLOAT_DTYPES:
        problems.append(f"conformance: precision_bits must be one of {sorted(cost.FLOAT_DTYPES)} (the card's declared "
                        "precision, set by the host)")
    elif not problems:
        other = findings.other_precision(spec["equations"], bits)
        if other:
            problems.append(f"conformance: the equations name a float dtype other than the declared {bits}-bit "
                            f"precision ({'; '.join(other[:4])}); under the precision policy they compute at "
                            f"{bits} bits, so leave dtypes out or use the declared one")
    return problems


FORM_PROBLEMS = {"z3": z3_problems, "sympy": sympy_problems, "simulation": simulation_problems,
                 "construction": construction_problems, "cost": cost_problems,
                 "conservation": conservation_problems, "learnability": learnability.spec_problems,
                 "conformance": conformance_problems}


def statement_problems(form: str, spec) -> list[str]:
    """Problems with a non-Lean statement's data; empty means it can be frozen and run."""
    if form not in FORM_PROBLEMS:
        return [f"unknown form {form!r}"]
    if not isinstance(spec, dict):
        return [f"{form}: statement must be an object"]
    return FORM_PROBLEMS[form](spec)


def _unknown(evidence: str, **extra) -> dict:
    return {"status": UNKNOWN, "evidence": evidence[:2000], "counterexample": None, **extra}


# ----- Z3 ------------------------------------------------------------------------------------------------------------

def bound_text(spec: dict) -> str:
    parts = [f"{name} in [{var['lo']}, {var['hi']}]" for name, var in spec["variables"].items()
             if var["sort"] != "bool"]
    return ", ".join(parts) or "booleans only"


def _z3_env(spec: dict):
    import z3

    env = {}
    for name, var in spec["variables"].items():
        env[name] = {"int": z3.Int, "real": z3.Real, "bool": z3.Bool}[var["sort"]](name)
    funcs = {name: getattr(z3, name) for name in Z3_FUNCS}

    def const(value):
        if isinstance(value, bool):
            return z3.BoolVal(value)
        if isinstance(value, float):
            return z3.RealVal(str(Fraction(value).limit_denominator(10**12)))
        return value

    return z3, env, funcs, const


def _z3_value(z3, value):
    if z3.is_int_value(value):
        return value.as_long()
    if z3.is_rational_value(value):
        fraction = Fraction(value.numerator_as_long(), value.denominator_as_long())
        return int(fraction) if fraction.denominator == 1 else str(fraction)
    if z3.is_true(value) or z3.is_false(value):
        return bool(z3.is_true(value))
    return str(value)


def z3_property(spec: dict) -> dict:
    problems = z3_problems(spec)
    if problems:
        return _unknown("statement cannot be read: " + "; ".join(problems), tier=tiers.SMT_BOUNDED)
    z3, env, funcs, const = _z3_env(spec)
    solver = z3.Solver()
    solver.set("timeout", int(spec.get("timeout_ms") or DEFAULT_Z3_TIMEOUT_MS))
    for name, var in spec["variables"].items():
        if var["sort"] != "bool":
            solver.add(env[name] >= const(var["lo"]), env[name] <= const(var["hi"]))
    for text in spec.get("hypotheses") or []:
        solver.add(evaluate(text, env, funcs, const))
    solver.add(z3.Not(evaluate(spec["property"], env, funcs, const)))
    answer = solver.check()
    bound = bound_text(spec)
    if answer == z3.unsat:
        return {"status": DISCHARGED, "tier": tiers.SMT_BOUNDED, "bound": bound, "counterexample": None,
                "evidence": f"z3: the negation is unsat under the bounds and hypotheses; "
                            f"{tiers.verdict_phrase(tiers.SMT_BOUNDED, bound=bound)}"}
    if answer == z3.sat:
        model = solver.model()
        witness = {name: _z3_value(z3, model.eval(var, model_completion=True)) for name, var in env.items()}
        return {"status": VIOLATED, "tier": tiers.SMT_BOUNDED, "bound": bound, "counterexample": witness,
                "evidence": f"z3: sat; the property fails at {witness} within {bound}"}
    return _unknown(f"z3: {answer} ({solver.reason_unknown()}) within {bound}", tier=tiers.SMT_BOUNDED, bound=bound)


# ----- SymPy ---------------------------------------------------------------------------------------------------------

def _sympy_env(spec: dict):
    import sympy as sp

    flags = {"real": {"real": True}, "positive": {"positive": True}, "nonnegative": {"nonnegative": True},
             "integer": {"integer": True}, "positive_integer": {"integer": True, "positive": True},
             "nonnegative_integer": {"integer": True, "nonnegative": True}}
    env = {name: sp.Symbol(name, **flags[domain]) for name, domain in spec["symbols"].items()}
    env.update(pi=sp.pi, E=sp.E)
    funcs = {name: getattr(sp, name) for name in SYMPY_FUNCS}

    def const(value):
        if isinstance(value, bool):
            return sp.true if value else sp.false
        if isinstance(value, float):
            return sp.Rational(str(value))
        return sp.Integer(value)

    return sp, env, funcs, const


def _sample(domain: str, rng: random.Random) -> Fraction:
    if domain.endswith("integer"):
        low = 1 if domain.startswith("positive") else (0 if domain.startswith("nonnegative") else -12)
        return Fraction(rng.randint(low, 12))
    value = Fraction(rng.randint(1, 60), rng.randint(1, 9))
    if domain == "real" and rng.random() < 0.5:
        value = -value
    return value


def _relation_holds(relation: str, a, b) -> bool:
    return {"==": a == b, "<=": a <= b, ">=": a >= b, "<": a < b, ">": a > b}[relation]


def sympy_relation(spec: dict, *, seed: int = 0) -> dict:
    problems = sympy_problems(spec)
    if problems:
        return _unknown("statement cannot be read: " + "; ".join(problems), tier=tiers.SYMPY_EXACT)
    sp, env, funcs, const = _sympy_env(spec)
    lhs = evaluate(spec["lhs"], env, funcs, const)
    rhs = evaluate(spec["rhs"], env, funcs, const)
    conditions = [evaluate(text, env, funcs, const) for text in spec.get("conditions") or []]
    relation = spec["relation"]
    rng = random.Random(seed)
    points, attempts, counterexample = [], 0, None
    symbols = {name: env[name] for name in spec["symbols"]}
    while len(points) < SPOT_POINTS and attempts < SPOT_POINTS * 50:
        attempts += 1
        point = {symbols[name]: sp.Rational(v.numerator, v.denominator)
                 for name, v in ((name, _sample(domain, rng)) for name, domain in spec["symbols"].items())}
        try:
            if not all(bool(c.subs(point)) for c in conditions):
                continue
            a, b = sp.nsimplify(lhs.subs(point)), sp.nsimplify(rhs.subs(point))
            if not (a.is_finite and b.is_finite):
                continue
            if a.is_Rational and b.is_Rational:
                holds = _relation_holds(relation, a, b)
            elif relation == "==":
                holds = abs(sp.N(a - b, 60)) < sp.Rational(1, 10**40)
            else:
                holds = bool(_relation_holds(relation, sp.N(a, 60), sp.N(b, 60)))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        points.append({str(k): str(v) for k, v in point.items()})
        if not holds:
            counterexample = {"point": points[-1], "lhs": str(a), "rhs": str(b), "relation": relation}
            break
    if counterexample is not None:
        return {"status": VIOLATED, "tier": tiers.SYMPY_EXACT, "counterexample": counterexample,
                "evidence": f"sympy: {spec['lhs']} {relation} {spec['rhs']} fails at {counterexample['point']} "
                            f"(lhs {counterexample['lhs']}, rhs {counterexample['rhs']})"}
    if len(points) < SPOT_POINTS:
        return _unknown(f"sympy: only {len(points)} of {SPOT_POINTS} spot points satisfy the domain and conditions",
                        tier=tiers.SYMPY_EXACT)
    difference = sp.simplify(lhs - rhs)
    if relation == "==":
        proved = difference == 0
        how = "simplify(lhs - rhs) == 0"
    else:
        sign = {"<=": (rhs - lhs), "<": (rhs - lhs), ">=": (lhs - rhs), ">": (lhs - rhs)}[relation]
        sign = sp.simplify(sign)
        proved = bool(sign.is_nonnegative) if relation in ("<=", ">=") else bool(sign.is_positive)
        how = f"SymPy proves the sign of {sign} from the symbol assumptions"
    if proved and not conditions:
        return {"status": DISCHARGED, "tier": tiers.SYMPY_EXACT, "counterexample": None,
                "evidence": f"sympy: {how}; {SPOT_POINTS} rational spot checks agree; "
                            f"{tiers.verdict_phrase(tiers.SYMPY_EXACT)}"}
    if proved and conditions:
        # The symbolic step did not use the conditions, so a proof without them is a proof with them.
        return {"status": DISCHARGED, "tier": tiers.SYMPY_EXACT, "counterexample": None,
                "evidence": f"sympy: {how} without the conditions; {SPOT_POINTS} rational spot checks agree; "
                            f"{tiers.verdict_phrase(tiers.SYMPY_EXACT)}"}
    return _unknown(f"sympy: no symbolic proof ({how} failed; difference {difference}); "
                    f"{SPOT_POINTS} spot checks agree", tier=tiers.SYMPY_EXACT)


# ----- executed checks -----------------------------------------------------------------------------------------------

def _load(source: str, names: tuple, label: str, numpy=np) -> dict:
    problems = _code_problems(label, source, names)
    if problems:
        raise CandidateError("; ".join(problems))
    namespace = {"__builtins__": dict(SAFE_BUILTINS), "np": numpy}
    exec(compile(source, f"<{label}>", "exec"), namespace)
    return namespace


def _plain(value):
    """JSON round trip, so the prover and the metric see plain data and cannot share objects."""
    def default(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"{type(obj).__name__} is not JSON data")
    return json.loads(json.dumps(value, default=default))


def simulation(spec: dict, source: str, *, seed: int = 0) -> dict:
    problems = simulation_problems(spec)
    if problems:
        return _unknown("statement cannot be read: " + "; ".join(problems), tier=tiers.EXECUTED)
    try:
        fixed = _load(spec["generator"], ("generate",), "generator")
        scorer = _load(spec["metric"], ("metric",), "metric")
    except Exception as exc:  # the frozen statement code failed to load
        return _unknown(f"statement code failed to load: {type(exc).__name__}: {exc}", tier=tiers.EXECUTED)
    try:
        prover = _load(source, ("run",), "prover")
    except Exception as exc:  # the prover's code is refused or fails to load; another attempt may fix it
        return _unknown(f"prover code refused: {type(exc).__name__}: {exc}", tier=tiers.EXECUTED, prover_fault=True)
    base = int(spec.get("seed", seed))
    results, first_failure = [], None
    for size in spec["sizes"]:
        total, scores = 0.0, []
        for index in range(spec["count"]):
            rng = np.random.default_rng([base, size, index])
            try:
                instance = _plain(fixed["generate"](size, rng))
                if not isinstance(instance, dict) or set(instance) != {"input", "target"}:
                    raise SpecError("generate() must return {'input': ..., 'target': ...}")
            except Exception as exc:
                return _unknown(f"generator failed at size {size}, instance {index}: {type(exc).__name__}: {exc}",
                                tier=tiers.EXECUTED)
            try:
                output = _plain(prover["run"](_plain(instance["input"])))
            except Exception as exc:
                where = crash_line(exc, source, "<prover>")
                return _unknown(f"prover run() failed at size {size}, instance {index}: {describe_exception(exc)}"
                                + (f"; raised at {where}" if where else ""), tier=tiers.EXECUTED, prover_fault=True)
            try:
                score = scorer["metric"](_plain(instance["input"]), _plain(instance["target"]), output)
                score = float(score)
            except Exception as exc:
                return _unknown(f"metric failed at size {size}, instance {index}: {type(exc).__name__}: {exc}",
                                tier=tiers.EXECUTED)
            if not 0.0 <= score <= 1.0:
                return _unknown(f"metric returned {score}, outside [0, 1]", tier=tiers.EXECUTED)
            scores.append(score)
            total += score
            if score < 1.0 and first_failure is None:
                first_failure = {"size": size, "instance": index, "score": score,
                                 "input": json.dumps(instance["input"])[:600],
                                 "target": json.dumps(instance["target"])[:300], "output": json.dumps(output)[:300]}
        mean = total / len(scores)
        results.append({"size": size, "mean": round(mean, 6), "count": len(scores),
                        "passed": mean >= spec["threshold"]})
    failed = [r for r in results if not r["passed"]]
    listed = [f"size={r['size']}" for r in results]
    evidence = (", ".join(f"size {r['size']} mean {r['mean']:.4f}" for r in results)
                + f"; threshold {spec['threshold']}; {tiers.verdict_phrase(tiers.EXECUTED, sizes=listed)}")
    if failed:
        return {"status": VIOLATED, "tier": tiers.EXECUTED, "evidence": evidence, "results": results,
                "counterexample": dict(first_failure or {}, failing_size=failed[0]["size"],
                                       mean=failed[0]["mean"], threshold=spec["threshold"])}
    return {"status": DISCHARGED, "tier": tiers.EXECUTED, "evidence": evidence, "results": results,
            "counterexample": None}


def construction_survey(shared: dict, spec: dict, points: list[dict]):
    """The ``inspect`` hook of run_construction for the architecture contract (hybrid/math/architecture.py): after
    each size, every array the mechanism holds at module level and in a fresh state is checked against the declared
    dimensions at that size. Positions default to the longest token stream the generator made at that size."""
    from . import architecture
    name = constructions.SIZE_NAME[spec["task"]]

    def inspect(size, lay, instances, mechanism):
        found = architecture.survey(module=architecture.namespace_values(mechanism), state=mechanism["init"](
            dict(lay)))
        values = {name: size, **{k: v for k, v in (spec.get("params") or {}).items()
                                 if isinstance(v, (int, float)) and not isinstance(v, bool)}}
        points.append(architecture.check_point(shared, spec["widths"], values, found,
                                               positions=max(len(i.tokens) for i in instances),
                                               vocab=int(lay.get("vocab") or 0)))
    return inspect


def audited_construction(spec: dict, source: str, architecture_contract: dict | None = None) -> dict:
    problems = construction_problems(spec) + [f"prover: {p}" for p in audit_source(source)]
    if architecture_contract is not None:
        from . import architecture
        problems += architecture.widths_problems(spec.get("widths"), {constructions.SIZE_NAME.get(spec.get("task"),
                                                                                                    "size")})
    if problems:
        return _unknown("refused: " + "; ".join(problems), tier=tiers.EXECUTED, prover_fault=True)
    options = {k: spec[k] for k in ("count", "seed", "params", "threshold") if spec.get(k) is not None}
    points: list[dict] = []
    if architecture_contract is not None:
        options["inspect"] = construction_survey(architecture_contract, spec, points)
        try:
            budget = carried_state_budget(architecture_contract, spec)
        except ValueError as exc:  # the frozen state_bits or d formula cannot be evaluated at a listed size
            return _unknown(f"refused: architecture.state_bits: {exc}", tier=tiers.EXECUTED)
        if budget:
            options["state_bits"] = budget
    try:
        result = constructions.run_construction(source, spec["task"], spec["sizes"], **options)
    except constructions.MechanismCrashed as crash:  # the prover's mechanism crashed; another attempt may fix it
        # The size, the instance and the source line: a bare assert at K=64 reached the k1 prover in the 2026-10-13
        # smoke as "AssertionError: " and nothing else.
        name = constructions.SIZE_NAME[spec["task"]]
        where = crash_line(crash.exc, source, "<construction>")
        return _unknown(f"construction failed at {name}={crash.size}, instance {crash.instance}: "
                        f"{describe_exception(crash.exc)}" + (f"; raised at {where}" if where else ""),
                        tier=tiers.EXECUTED, prover_fault=True,
                        crash={"size": crash.size, "instance": crash.instance, "raised_at": where,
                               "error": describe_exception(crash.exc)[:400]})
    except Exception as exc:  # the prover's mechanism failed to load; another attempt may fix it
        where = crash_line(exc, source, "<construction>")
        return _unknown(f"construction failed: {describe_exception(exc)}" + (f"; raised at {where}" if where else ""),
                        tier=tiers.EXECUTED, prover_fault=True)
    result = _state_words_result(spec, result)
    if architecture_contract is None:
        return result
    return _architecture_result(spec, result, points)


def carried_state_budget(shared: dict, spec: dict) -> dict[int, float]:
    """FIX-8: the architecture contract's carried-state budget (``state_bits``, a number or a formula in d) at each
    listed size, with d from the statement's widths there; empty when the contract declares none."""
    from . import architecture
    if shared.get("state_bits") is None:
        return {}
    name = constructions.SIZE_NAME[spec["task"]]
    budget = {}
    for size in spec["sizes"]:
        values = {name: size, **{k: v for k, v in (spec.get("params") or {}).items()
                                 if isinstance(v, (int, float)) and not isinstance(v, bool)}}
        budget[size] = architecture.state_bits(shared, {"d": architecture.value(spec["widths"]["d"], values)})
    return budget


def _resource_failure(row: dict, message: str) -> None:
    """A size that broke its resource contract is never a P2 success, whatever its accuracy (FIX-8)."""
    row["passed"] = False
    row.setdefault("resource_violations", []).append(message)


def _state_words_result(spec: dict, result: dict) -> dict:
    if spec.get("state_words") is None:
        return result
    over = []
    name = constructions.SIZE_NAME[spec["task"]]
    for row in result["results"]:
        declared = state_words_bound(spec, row["size"])
        row["declared_state_words"] = declared
        if row["state_peak_words"] > declared + 1e-9:
            over.append(row)
            _resource_failure(row, f"peak state words {row['state_peak_words']} exceed the declared "
                                   f"{spec['state_words']} = {declared:g} at {name}={row['size']}")
    if not over:
        result["evidence"] += f"; peak state words within the declared {spec['state_words']} at every size"
        return result
    first = over[0]
    name = constructions.SIZE_NAME[spec["task"]]
    evidence = (result["evidence"] + f"; peak state words {first['state_peak_words']} exceed the declared "
                f"{spec['state_words']} = {first['declared_state_words']:g} at {name}={first['size']}")
    counterexample = result["counterexample"] or {
        "task": spec["task"], "size": first["size"], "check": "state_words",
        "state_peak_words": first["state_peak_words"], "declared_state_words": first["declared_state_words"]}
    return dict(result, status=VIOLATED, evidence=evidence, counterexample=counterexample)


def _architecture_result(spec: dict, result: dict, points: list[dict]) -> dict:
    """The architecture contract applied to a construction that ran: an array wider than the declared dimensions,
    or stored above the declared precision, at any size is a violation of the statement (FIX-4)."""
    from . import architecture
    detail = architecture.summary(points)
    result = dict(result, architecture=detail)
    if not detail["violations"]:
        result["evidence"] += "; every array within the declared architecture at every size"
        return result
    bad = next(p for p in points if p["violations"])
    name = constructions.SIZE_NAME[spec["task"]]
    rows = {row["size"]: row for row in result["results"]}
    for point in points:
        for violation in point["violations"]:
            _resource_failure(rows[point["point"][name]], f"architecture at {name}={point['point'][name]}: "
                                                          f"{violation}")
    evidence = result["evidence"] + "; architecture contract broken: " + "; ".join(detail["violations"])
    counterexample = result["counterexample"] or {"task": spec["task"], "check": "architecture",
                                                  "point": bad["point"], "dimensions": bad["dimensions"],
                                                  "violations": bad["violations"]}
    return dict(result, status=VIOLATED, evidence=evidence[:4000], counterexample=counterexample)


# ----- conformance of a cost reference to the card's equations (FIX-6) -----------------------------------------------

def _plain_tree(value):
    """Parameters and states as plain data: arrays (counted wrappers included) as NumPy copies, numbers kept,
    containers rebuilt, so the frozen equations cannot reach the counter or the reference's objects."""
    from . import architecture
    raw = architecture._raw(value)
    if raw is not None:
        return np.array(raw, copy=True)
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return value.item() if hasattr(value, "item") else value
    if isinstance(value, dict):
        return {k: _plain_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_tree(v) for v in value]
    if value is None:
        return None
    raise SpecError(f"a {type(value).__name__} is not plain data")


def _leaves(value) -> list[np.ndarray]:
    """Every array and number held in a state, as float arrays; containers and their keys are not compared, so a
    buffer can be named or ordered as the source chooses."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in _leaves(v)]
    if isinstance(value, (list, tuple)):
        # A list of numbers, or of vectors of one length, is one buffer: rows kept as a list equal a stacked array.
        if value and all(isinstance(v, (bool, int, float)) for v in value):
            return [np.asarray(value, dtype=float)]
        if value and all(isinstance(v, np.ndarray) and v.ndim <= 1 for v in value) \
                and len({v.shape for v in value}) == 1:
            return [np.stack([np.asarray(v, dtype=float) for v in value])]
        return [leaf for v in value for leaf in _leaves(v)]
    return [np.asarray(value, dtype=float)]


def _describe(leaf: np.ndarray) -> str:
    flat = leaf.ravel()
    return f"shape {list(leaf.shape)}, first values {[round(float(x), 6) for x in flat[:4]]}"


def _state_mismatch(expected, got, tolerance: float) -> str | None:
    """None when the reference holds exactly the arrays the equations retain (a one-to-one match by shape and value
    within ``tolerance``); otherwise the first array that is missing or left over."""
    wanted, held = _leaves(expected), _leaves(got)
    free = list(range(len(held)))
    for leaf in sorted(wanted, key=lambda a: -a.size):
        match = next((i for i in free if held[i].shape == leaf.shape
                      and np.allclose(held[i], leaf, rtol=tolerance, atol=tolerance)), None)
        if match is None:
            same_shape = [i for i in free if held[i].shape == leaf.shape]
            gap = (f"; largest difference from a held array of that shape "
                   f"{min(float(np.max(np.abs(held[i] - leaf))) for i in same_shape):.3g}" if same_shape else "")
            return (f"the equations retain an array ({_describe(leaf)}) that the reference does not hold{gap}; "
                    f"reference holds {[list(held[i].shape) for i in free]}")
        free.remove(match)
    if free:
        return ("the reference retains arrays the equations do not: "
                + "; ".join(_describe(held[i]) for i in free[:3]))
    return None


class PolicyNumPy:
    """NumPy for the frozen conformance equations under a precision policy (FIX-20). Every attribute is NumPy's,
    except that the float constructors make arrays at the policy dtype when no dtype is given, and refuse an explicit
    float dtype other than it. Arithmetic on the reference's parameters (already at the policy dtype) and Python
    numbers stays at that dtype under NumPy's promotion rules. Its own attributes are private, so ``np.dtype`` and
    every other NumPy name reach NumPy (model code cannot read private attributes, cost.audit_source)."""

    _CONSTRUCTORS = ("zeros", "ones", "empty", "full", "eye", "identity", "array", "asarray", "arange", "linspace")

    def __init__(self, precision_bits: int):
        self._bits = precision_bits
        self._dtype = cost.policy_dtype(precision_bits)

    def __getattr__(self, name):
        value = getattr(np, name)
        return self._constructor(value) if name in self._CONSTRUCTORS else value

    def _constructor(self, maker):
        def make(*args, **kwargs):
            if kwargs.get("dtype") is not None:
                asked = np.dtype(kwargs["dtype"])
                if asked.kind == "f" and asked != self._dtype:
                    raise SpecError(f"dtype {asked} under the declared {self._bits}-bit precision")
                return maker(*args, **kwargs)
            result = maker(*args, **kwargs)
            if np.asarray(result).dtype.kind != "f" or np.asarray(result).dtype == self._dtype:
                return result
            try:
                return maker(*args, **dict(kwargs, dtype=self._dtype))
            except TypeError as exc:   # a positional dtype at another precision
                raise SpecError(f"{maker.__name__} makes {np.asarray(result).dtype} under the declared "
                                f"{self._bits}-bit precision") from exc
        return make


def _off_policy(value, dtype) -> list[str]:
    """The float leaves of a returned state or output at a dtype other than the policy's."""
    if isinstance(value, dict):
        return [p for v in value.values() for p in _off_policy(v, dtype)]
    if isinstance(value, (list, tuple)):
        return [p for v in value for p in _off_policy(v, dtype)]
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind == "f" and value.dtype != dtype:
        return [f"{value.dtype} of shape {list(np.shape(value))}"]
    return []


def conformance_streams(kind: str, n: int, vocab: int) -> list[int]:
    """Token streams for a conformance audit: one symbol repeated, distinct symbols in turn, uniform random."""
    if kind == "repeated":
        return [1 % vocab] * n
    if kind == "distinct":
        return [i % vocab for i in range(n)]
    return cost.token_stream(n, vocab, 0)


def reference_conformance(spec: dict, source: str) -> dict:
    """Run the cost reference (``init``/``update``/``predict`` of the cost contract, the source that held for the
    statement ``spec["reference"]``) and the frozen ``expected(params, sizes, tokens)`` on the same parameters, at
    every point and stream, and compare after every token: the retained state (every array and number, matched one
    to one by shape and value) and predict's output. Discharged when they agree everywhere (tier executed); violated
    at the first position where the reference computes something else, omits work, or retains other buffers. This
    checks equation conformance on the listed streams, not only operation and state counts. Both sides run under the
    statement's precision policy (FIX-20): the counted reference in ``cost.OpCounter(precision_bits)`` and the
    equations with ``PolicyNumPy``, on the reference's parameters at that dtype; equations that return a state or
    output at another float dtype leave the audit unknown (the statement's fault, not the reference's)."""
    problems = conformance_problems(spec)
    if problems:
        return _unknown("statement cannot be read: " + "; ".join(problems), tier=tiers.EXECUTED)
    bits = spec["precision_bits"]
    dtype = cost.policy_dtype(bits)
    try:
        frozen = _load(spec["equations"], ("expected",), "equations", PolicyNumPy(bits))
    except Exception as exc:  # the frozen statement code failed to load
        return _unknown(f"statement code failed to load: {type(exc).__name__}: {exc}", tier=tiers.EXECUTED)
    try:
        candidate = cost.load_candidate(source)
    except Exception as exc:  # the reference is refused or fails to load
        return _unknown(f"reference source refused: {describe_exception(exc)}", tier=tiers.EXECUTED,
                        prover_fault=True)
    tolerance = float(spec["tolerance"])
    audited = []
    for point in spec["points"]:
        sizes = {k: v for k, v in point.items() if k != "n"}
        vocab = int(sizes.get("V", cost.DEFAULT_VOCAB))
        for kind in spec["streams"]:
            tokens = conformance_streams(kind, point["n"], vocab)
            where = {"point": dict(point), "stream": kind}
            try:
                counter = cost.OpCounter(bits)
                params, state = candidate.init(dict(sizes), counter)
                plain_params = _plain_tree(params)
                trace = []
                for token in tokens:
                    state = candidate.update(params, state, token, counter)
                    trace.append((_plain_tree(state), _plain_tree(candidate.predict(params, state, counter))))
            except Exception as exc:
                at = crash_line(exc, source, cost.CANDIDATE_FILE)
                return _unknown(f"reference failed at {where}: {describe_exception(exc)}"
                                + (f"; raised at {at}" if at else ""), tier=tiers.EXECUTED, prover_fault=True)
            try:
                returned = frozen["expected"](plain_params, dict(sizes), list(tokens))
                wanted = _plain_tree(returned)
                if not isinstance(wanted, list) or len(wanted) != len(tokens) or not all(
                        isinstance(w, dict) and set(w) == {"state", "output"} for w in wanted):
                    raise SpecError("expected() must return one {'state': ..., 'output': ...} per token")
                off = _off_policy(returned, dtype)
                if off:
                    raise SpecError(f"the equations return {off[0]} under the declared {bits}-bit precision "
                                    f"({len(off)} value(s) off the policy)")
            except Exception as exc:
                return _unknown(f"the frozen equations failed at {where}: {type(exc).__name__}: {exc}",
                                tier=tiers.EXECUTED)
            for position, ((got_state, got_output), want) in enumerate(zip(trace, wanted)):
                detail = _state_mismatch(want["state"], got_state, tolerance)
                part = "state"
                if detail is None:
                    a, b = np.asarray(want["output"], dtype=float), np.asarray(got_output, dtype=float)
                    if a.shape != b.shape or not np.allclose(a, b, rtol=tolerance, atol=tolerance):
                        part = "output"
                        detail = f"predict returned {_compact_value(got_output)}, the equations give " \
                                 f"{_compact_value(want['output'])}"
                if detail is not None:
                    counterexample = dict(where, position=position, token=tokens[position], part=part,
                                          detail=detail[:600])
                    return {"status": VIOLATED, "tier": tiers.EXECUTED, "counterexample": counterexample,
                            "evidence": (f"the reference departs from the frozen equations at {where} position "
                                         f"{position} ({part}): {detail}")[:2000], "checked": audited}
            audited.append(dict(where, positions=len(tokens)))
    listed = [f"n={a['point']['n']} {a['stream']}" for a in audited]
    return {"status": DISCHARGED, "tier": tiers.EXECUTED, "counterexample": None, "checked": audited,
            "precision_bits": bits,
            "evidence": (f"the reference's retained state and output equal the frozen equations after every token "
                         f"(tolerance {tolerance:g}, both at {bits}-bit precision) on {len(audited)} streams; "
                         + tiers.verdict_phrase(tiers.EXECUTED, sizes=listed))[:2000]}


def _compact_value(value) -> str:
    text = json.dumps(_plain(value))
    return text if len(text) <= 120 else text[:120] + "..."


# ----- statement match on finite domains ------------------------------------------------------------------------------

def _random_value(var: dict, rng: random.Random):
    if var["sort"] == "bool":
        return rng.random() < 0.5
    lo, hi = Fraction(str(var["lo"])), Fraction(str(var["hi"]))
    if var["sort"] == "int":
        return rng.randint(math.ceil(lo), math.floor(hi))
    denominator = rng.randint(1, 8)
    low, high = int(lo * denominator), int(hi * denominator)
    value = Fraction(rng.randint(low, high), denominator) if high >= low else lo
    return min(max(value, lo), hi)


def predicate_agreement(spec: dict, predicate: str, *, count: int = 1000, seed: int = 0) -> dict:
    """Evaluate the frozen z3 property and a predicate written from the claim on ``count`` random assignments
    inside the bounds (hypotheses must hold). Any disagreement is a mismatch (violated)."""
    problems = z3_problems(spec)
    if problems:
        return _unknown("statement cannot be read: " + "; ".join(problems))
    try:
        claim = _load(predicate, ("claim",), "predicate")["claim"]
    except Exception as exc:
        return _unknown(f"predicate refused: {type(exc).__name__}: {exc}")
    z3, env, funcs, const = _z3_env(spec)
    hypotheses = [evaluate(text, env, funcs, const) for text in spec.get("hypotheses") or []]
    prop = evaluate(spec["property"], env, funcs, const)
    rng = random.Random(seed)
    checked = 0
    for _ in range(count * 5):
        if checked >= count:
            break
        values = {name: _random_value(var, rng) for name, var in spec["variables"].items()}
        pairs = []
        for name, value in values.items():
            if isinstance(value, bool):
                pairs.append((env[name], z3.BoolVal(value)))
            elif isinstance(value, Fraction):
                pairs.append((env[name], z3.RealVal(str(value))))
            elif spec["variables"][name]["sort"] == "real":
                pairs.append((env[name], z3.RealVal(str(value))))
            else:
                pairs.append((env[name], z3.IntVal(value)))
        if not all(z3.is_true(z3.simplify(z3.substitute(h, *pairs))) for h in hypotheses):
            continue
        formal = z3.simplify(z3.substitute(prop, *pairs))
        if not (z3.is_true(formal) or z3.is_false(formal)):
            return _unknown(f"the formal property did not reduce to true or false at {values}")
        try:
            natural = claim(**values)
        except Exception as exc:
            return _unknown(f"predicate failed at {values}: {type(exc).__name__}: {exc}")
        if not isinstance(natural, (bool, np.bool_)):
            return _unknown(f"predicate returned {natural!r}, not a bool")
        checked += 1
        if bool(natural) != z3.is_true(formal):
            shown = {k: (str(v) if isinstance(v, Fraction) else v) for k, v in values.items()}
            return {"status": VIOLATED, "counterexample": {"assignment": shown, "formal": z3.is_true(formal),
                                                           "natural": bool(natural)},
                    "evidence": f"the formal statement and the claim's predicate disagree at {shown}: formal "
                                f"{z3.is_true(formal)}, claim {bool(natural)}", "checked": checked}
    if checked < min(100, count):
        return _unknown(f"only {checked} assignments satisfied the hypotheses", checked=checked)
    return {"status": DISCHARGED, "counterexample": None, "checked": checked,
            "evidence": f"the formal statement and the claim's predicate agree on {checked} random assignments "
                        f"within {bound_text(spec)}"}
