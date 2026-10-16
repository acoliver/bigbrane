"""Verification code of a challenger finding (DISCOVERY-PIPELINE-PLAN.md section 5, decision FIX-4, 2026-10-13).

A finding the challenger raises at stage C (a missing assumption, a witness that breaks the frozen architecture, a
cost the reference does not count) says how it can be settled. When it says code, it comes with Python defining
``verify(witness)`` that returns ``(confirmed, evidence)``: ``confirmed`` is True when the fact the finding rests on
holds (for example: the O7.a decoder at depth 16 holds a 2176-wide matrix while d is 224), False when it does not.
The code runs here, in the tools image, never on the host's word.

``witness`` gives read access to what stage M ran:

- ``witness.sources``: {statement id: the prover's source or proof text};
- ``witness.specs``: {statement id: {"form": ..., "spec": the frozen statement}};
- ``witness.load(statement_id)``: the module data and functions of a construction, learner or cost source, loaded
  as the checks load it (a dict);
- ``witness.layout(task, size, params=None)`` and ``witness.generate(task, size, count, seed, params=None)``: the
  task harness of constructions.py (instances have ``tokens`` and ``queries``);
- ``witness.survey(value)``: every array reachable from ``value`` with its shape and float bits
  (architecture.survey);
- FIX-17: ``witness.resources`` (the card's frozen ``resources`` section), ``witness.evaluate(formula, values)`` (a
  resource formula at given sizes) and ``witness.conservation`` (storage_bound, copy_check, dyck_check,
  associative_recall_check, streaming_index_bound, each returning the library's check as a dict with ``status``), so
  a resource contradiction is settled by code: a claim at a size the stated width and precision cannot hold, or a
  fixed-state claim against a width that grows with the task.

The code is audited like every model-written source (no imports, no private attributes, no eval or getattr) and runs
with the safe builtins and NumPy. Results: ``discharged`` (confirmed) or ``violated`` (not confirmed, the
evidence as counterexample, so the replay compares the verdict) when the code ran and returned one; ``unknown`` with
``prover_fault`` when the code was refused, crashed or returned something else (the challenger's fault: the finding
is then unsupported); ``unknown`` without it when the job itself failed.

Confirmation and materiality are separate decisions (FIX-13, 2026-10-14). A confirmed witness violation shows only
that the deviation exists. Whether it decides a claim is settled per affected statement by the host at stage C: the
challenger names the deviating text of that statement's source and a narrow edit that removes it
(``conforming_variant`` validates it here, on the host, without running anything), and the variant runs through the
statement's own frozen check (``spec_checks.audited_construction`` for a construction) with the frozen spec, sizes,
count, seed and architecture contract. A passing variant makes the deviation immaterial to that claim; a failing one
makes it material; no valid edit, or a replay that did not finish, leaves it unresolved. In k1-run1 of the
2026-10-14 calibration one finding ("max subtraction, which D2 forbids") failed both P2.copy and P2.recall; replayed
without the subtraction, copy still passes and recall overflows.

A precision deviation has its own conforming edit (FIX-15, 2026-10-15): the witness computes at a precision other
than the frozen contract's ``precision_bits``, and the conforming variant is the same source with its dtype casts and
dtype arguments set to the declared precision. ``conforming_variant(..., precision_bits=...)`` accepts such an edit
when it changes dtype casts and dtype arguments only, toward the declared precision only. In k2-run2 of the
2026-10-15 calibration both O9 witnesses computed in float64 under a binary32 contract; the challenger could not
offer an edit under the FIX-13 rules, so P2.composition was left not shown.

FIX-20 (decision PR): ``other_precision(source, precision_bits)`` lists the float dtypes a source names other than the
declared precision. The conformance audit refuses frozen equations that name one, since both sides of that audit run
under the card's one precision policy (spec_checks.reference_conformance). Materiality replays of a cost reference
run through the statement's own frozen check, which carries the policy (stage M's library request).

Self-contained (stdlib, NumPy, SymPy) like the other check modules.
"""
from __future__ import annotations

import ast
import re
from types import SimpleNamespace

import numpy as np
import sympy as sp

from . import architecture, conservation, constructions, cost, learnability
from .cost import SAFE_BUILTINS, audit_source, crash_line, describe_exception

FINDING_FILE = "<finding>"
MAX_EVIDENCE = 2000


def _load_source(form: str, source: str) -> dict:
    if form == "construction":
        return constructions.load_mechanism(source)
    if form == "learnability":
        return learnability.load_learner(source)
    if form == "cost":
        return vars(cost.load_candidate(source))
    namespace = {"__builtins__": dict(SAFE_BUILTINS), "np": np}
    exec(compile(source, f"<{form}>", "exec"), namespace)
    return namespace


def evaluate_formula(text, values: dict) -> float:
    """A card resource formula (numbers, + - * / **, ceiling, floor, log, sqrt, Max, Min) at the given sizes."""
    names = {name: sp.Symbol(name, positive=True) for name in values}
    if isinstance(text, bool) or not isinstance(text, (str, int, float)):
        raise ValueError(f"{text!r} is not a formula")
    problems = architecture.formula_problems(str(text), set(values))
    if problems:
        raise ValueError("; ".join(problems))
    expr = sp.sympify(str(text), locals=dict(names, **architecture.FUNCTIONS))
    return float(expr.subs({names[k]: sp.Integer(v) for k, v in values.items()}))


def _law(function):
    def run(*args, **kwargs) -> dict:
        return function(*args, **kwargs).to_dict()
    return run


def witness_for(sources: dict, specs: dict, resources: dict | None = None) -> SimpleNamespace:
    def load(statement_id: str) -> dict:
        if statement_id not in sources or not sources[statement_id]:
            raise KeyError(f"no source for statement {statement_id}")
        form = (specs.get(statement_id) or {}).get("form", "construction")
        return {k: v for k, v in _load_source(form, sources[statement_id]).items() if not k.startswith("__")}

    def survey(value) -> dict:
        return architecture.survey(value=value)

    # FIX-17: the card's frozen resources and the conservation library, so a resource contradiction (a claim the
    # stated widths, precision or state cannot hold) is settled by runnable code like every other finding.
    laws = SimpleNamespace(storage_bound=_law(conservation.storage_bound), copy_check=_law(conservation.copy_check),
                           dyck_check=_law(conservation.dyck_check),
                           associative_recall_check=_law(conservation.associative_recall_check),
                           streaming_index_bound=_law(conservation.streaming_index_bound))
    return SimpleNamespace(sources=dict(sources), specs=dict(specs), load=load, layout=constructions.layout,
                           generate=constructions.generate, survey=survey, resources=dict(resources or {}),
                           evaluate=evaluate_formula, conservation=laws)


def _unknown(evidence: str, *, prover_fault: bool) -> dict:
    return {"status": "unknown", "confirmed": None, "evidence": evidence[:MAX_EVIDENCE], "counterexample": None,
            "prover_fault": prover_fault}


def verify_finding(code: str, sources: dict, specs: dict, resources: dict | None = None) -> dict:
    """Run a finding's ``verify(witness)``; see the module docstring for the result."""
    problems = audit_source(code) if isinstance(code, str) and code.strip() else ["code is missing"]
    if problems:
        return _unknown("finding code refused: " + "; ".join(problems), prover_fault=True)
    namespace = {"__builtins__": dict(SAFE_BUILTINS), "np": np}
    try:
        exec(compile(code, FINDING_FILE, "exec"), namespace)
        if not callable(namespace.get("verify")):
            return _unknown("finding code does not define verify(witness)", prover_fault=True)
        answer = namespace["verify"](witness_for(sources, specs, resources))
    except Exception as exc:  # the challenger's code failed; the finding is then unsupported
        where = crash_line(exc, code, FINDING_FILE)
        return _unknown(f"finding code failed: {describe_exception(exc)}" + (f"; raised at {where}" if where else ""),
                        prover_fault=True)
    if (not isinstance(answer, (tuple, list)) or len(answer) != 2 or not isinstance(answer[0], (bool, np.bool_))
            or not isinstance(answer[1], str)):
        return _unknown(f"verify must return (confirmed: bool, evidence: str), not {answer!r}"[:400], prover_fault=True)
    confirmed, evidence = bool(answer[0]), answer[1].strip() or "(no evidence text)"
    if confirmed:
        return {"status": "discharged", "confirmed": True, "counterexample": None,
                "evidence": ("confirmed: " + evidence)[:MAX_EVIDENCE]}
    return {"status": "violated", "confirmed": False, "counterexample": {"confirmed": False, "evidence": evidence[:400]},
            "evidence": ("not confirmed: " + evidence)[:MAX_EVIDENCE]}


# ----- materiality of a witness violation (FIX-13) ------------------------------------------------------------------

MAX_EDITS = 8
# Names that set or change the precision of a computation. An edit keeps every one its anchor has and adds none.
PRECISION_NAMES = frozenset({"float16", "float32", "float64", "float96", "float128", "half", "single", "double",
                             "longdouble", "longfloat", "float_", "astype", "dtype", "finfo", "complex64",
                             "complex128", "clongdouble"})
_TOKEN = re.compile(r"[A-Za-z_]\w*|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _tokens(text: str) -> tuple[list[str], list[float]]:
    """The names and numeric literals of a code fragment (a fragment need not parse on its own)."""
    names, numbers = [], []
    for token in _TOKEN.findall(text):
        if token[0].isalpha() or token[0] == "_":
            names.append(token)
        else:
            numbers.append(float(token))
    return names, numbers


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    found, start = [], text.find(needle)
    while needle and start >= 0:
        found.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    return found


def _lines(text: str, start: int, end: int) -> set[int]:
    """The line numbers the characters text[start:end] lie on (an empty span: the line it sits on)."""
    first = text.count("\n", 0, start)
    return set(range(first, text.count("\n", 0, max(start, end - 1)) + 1))



# ----- precision-restoring edits (FIX-15) ---------------------------------------------------------------------------

# Float dtype names and their widths in bits (NumPy's ``double`` and ``float_`` are float64). ``longdouble`` is
# recorded as wider than 64 bits whatever the platform gives it: it is never the frozen precision.
FLOAT_BITS = {"float16": 16, "half": 16, "float32": 32, "single": 32, "float64": 64, "double": 64, "float_": 64,
              "float96": 96, "float128": 128, "longdouble": 128, "longfloat": 128}
# dtype strings: the names above and the array-protocol codes.
FLOAT_STRING_BITS = {**FLOAT_BITS, **{f"{order}f{n // 8}": n for n in (16, 32, 64) for order in ("", "<", ">", "=")}}
NUMPY_NAMES = frozenset({"np", "numpy"})
# Array constructors that take a dtype as a positional argument, and its position.
DTYPE_POSITION = {"zeros": 1, "ones": 1, "empty": 1, "identity": 1, "array": 1, "asarray": 1,
                  "ascontiguousarray": 1, "zeros_like": 1, "ones_like": 1, "empty_like": 1, "full": 2,
                  "full_like": 2}
_LEX = re.compile(r"""[A-Za-z_]\w*|'[^'\n]*'|"[^"\n]*"|\d[\w.]*|\S""")


def _float_bits(node: ast.AST, *, builtin: bool = False) -> int | None:
    """The width of a float dtype the node names (np.float32, float32, 'float32', 'f4'), else None. ``builtin``
    also reads Python's ``float`` (64 bits), which names a dtype only as an ``astype`` or ``dtype=`` argument."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in NUMPY_NAMES:
        return FLOAT_BITS.get(node.attr)
    if isinstance(node, ast.Name):
        return 64 if builtin and node.id == "float" else FLOAT_BITS.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return FLOAT_STRING_BITS.get(node.value)
    return None


def _call_name(func: ast.AST) -> str | None:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


class _Uncast(ast.NodeTransformer):
    """Removes float dtype casts and dtype arguments: ``x.astype(DT)`` and ``DT(x)`` become ``x``, a ``dtype=DT``
    keyword and the trailing positional dtype of a NumPy array constructor go. Two sources that differ only in
    those compare equal after it. A dtype named anywhere else (an alias, ``finfo``, a comparison) stays and must
    compare as written."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        name = _call_name(node.func)
        if name in ("finfo", "iinfo"):
            return node   # machine constants of a dtype are numbers the witness uses, not a cast
        self.generic_visit(node)
        if (name == "astype" and isinstance(node.func, ast.Attribute) and len(node.args) == 1 and not node.keywords
                and _float_bits(node.args[0], builtin=True) is not None):
            return node.func.value
        if _float_bits(node.func) is not None and len(node.args) == 1 and not node.keywords:
            return node.args[0]
        node.keywords = [k for k in node.keywords
                         if not (k.arg == "dtype" and _float_bits(k.value, builtin=True) is not None)]
        position = DTYPE_POSITION.get(name) if (isinstance(node.func, ast.Attribute)
                                                and isinstance(node.func.value, ast.Name)
                                                and node.func.value.id in NUMPY_NAMES) else None
        if (position is not None and len(node.args) == position + 1
                and _float_bits(node.args[position], builtin=True) is not None):
            node.args = node.args[:position]
        return node


def _uncast(tree: ast.AST) -> str:
    return ast.dump(_Uncast().visit(tree), include_attributes=False)


def _lexed_bits(token: str) -> int | None:
    if token[0] in "'\"":
        return FLOAT_STRING_BITS.get(token[1:-1])
    return FLOAT_BITS.get(token)


def _changes(old: list[str], new: list[str]) -> list[tuple[list[str], list[str]]]:
    """The (removed, added) token runs between two token lists, aligned on a longest common subsequence."""
    longest = [[0] * (len(new) + 1) for _ in range(len(old) + 1)]
    for i in range(len(old) - 1, -1, -1):
        for j in range(len(new) - 1, -1, -1):
            longest[i][j] = (longest[i + 1][j + 1] + 1 if old[i] == new[j]
                             else max(longest[i + 1][j], longest[i][j + 1]))
    runs, removed, added, i, j = [], [], [], 0, 0
    while i < len(old) or j < len(new):
        if i < len(old) and j < len(new) and old[i] == new[j]:
            if removed or added:
                runs.append((removed, added))
                removed, added = [], []
            i, j = i + 1, j + 1
        elif j < len(new) and (i == len(old) or longest[i][j + 1] >= longest[i + 1][j]):
            added.append(new[j])
            j += 1
        else:
            removed.append(old[i])
            i += 1
    if removed or added:
        runs.append((removed, added))
    return runs


def precision_direction(anchor: str, replacement: str, precision_bits: int) -> list[str]:
    """Why an edit's dtype changes do not all move toward the frozen precision, else []. The edit keeps the anchor's
    lines, and each line is compared token by token: every float dtype the replacement adds is the frozen one, none
    it drops is the frozen one, and a narrower one is dropped only where the frozen one takes its place. Dropping a
    wider cast (``y.astype(np.float64)`` to ``y`` for a float32 ``y``) moves toward it."""
    old_lines, new_lines = anchor.split("\n"), replacement.split("\n")
    if len(old_lines) != len(new_lines):
        return [f"it has {len(new_lines)} lines where its anchor has {len(old_lines)}; a precision edit keeps the "
                "lines and changes dtype casts within them"]
    problems, changed = [], False
    for old_line, new_line in zip(old_lines, new_lines):
        if old_line == new_line:
            continue
        for removed_tokens, added_tokens in _changes(_LEX.findall(old_line), _LEX.findall(new_line)):
            removed = [b for b in map(_lexed_bits, removed_tokens) if b is not None]
            added = [b for b in map(_lexed_bits, added_tokens) if b is not None]
            changed = changed or bool(removed or added)
            other = sorted({b for b in added if b != precision_bits})
            if other:
                problems.append(f"it sets {', '.join(f'{b}-bit' for b in other)} precision where "
                                f"{' '.join(added_tokens)[:80]!r} stands; the frozen precision is "
                                f"{precision_bits}-bit")
            if precision_bits in removed:
                problems.append(f"it drops a {precision_bits}-bit cast ({' '.join(removed_tokens)[:80]!r}), which is "
                                "the frozen precision")
            if any(b < precision_bits for b in removed) and precision_bits not in added:
                problems.append(f"it drops a narrower cast ({' '.join(removed_tokens)[:80]!r}) without setting the "
                                f"frozen {precision_bits}-bit precision in its place")
    if not changed:
        problems.append(f"it sets no float dtype; a precision edit casts to the frozen {precision_bits}-bit "
                        "precision (astype, dtype=, np.float32(...))")
    return list(dict.fromkeys(problems))


def wider_precision_left(source: str, precision_bits: int) -> list[str]:
    """The float dtype names wider than the frozen precision a source still has, as "line N: name". Recorded with a
    precision variant: a name left in weight construction can be legitimate, one left in update or predict means the
    restoration was partial."""
    left = []
    for number, line in enumerate(source.splitlines(), start=1):
        left += [f"line {number}: {t}" for t in _LEX.findall(line)
                 if (_lexed_bits(t) or 0) > precision_bits]
    return left


def other_precision(source: str, precision_bits: int) -> list[str]:
    """FIX-20: the float dtypes a source names other than the declared precision, as "line N: name": a NumPy dtype
    name or dtype string, and Python's ``float`` where it names a dtype (an ``astype``, ``dtype=`` or the positional
    dtype of an array constructor). The frozen conformance equations run under the card's precision policy, and one
    that names another float dtype computes outside it."""
    found = {}

    def note(node: ast.AST, bits: int | None) -> None:
        if bits is not None and bits != precision_bits:
            found.setdefault((node.lineno, node.col_offset), f"line {node.lineno}: {ast.unparse(node)}")

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            dtypes = [k.value for k in node.keywords if k.arg == "dtype"]
            if name == "astype" and node.args:
                dtypes.append(node.args[0])
            position = DTYPE_POSITION.get(name)
            if position is not None and len(node.args) > position:
                dtypes.append(node.args[position])
            for value in dtypes:
                note(value, _float_bits(value, builtin=True))
        elif isinstance(node, (ast.Attribute, ast.Name, ast.Constant)):
            note(node, _float_bits(node))
    return [found[key] for key in sorted(found)]


def _precision_names(text: str) -> list[str]:
    """The precision names of a fragment, dtype strings ('float64', 'f8') included."""
    return sorted(t for t in _LEX.findall(text) if t in PRECISION_NAMES or _lexed_bits(t) is not None)


def edit_kind(edits: list[dict], precision_bits: int | None) -> str:
    """``precision`` when every edit changes precision names (a precision restoration, FIX-15), ``narrow`` when none
    does (the FIX-13 rules), ``mixed`` otherwise. Without a frozen precision every edit is held to the FIX-13 rules,
    which refuse a change of precision names."""
    if precision_bits is None:
        return "narrow"
    flags = {_precision_names(e["anchor"]) != _precision_names(e["replacement"]) for e in edits
             if isinstance(e.get("anchor"), str) and isinstance(e.get("replacement"), str)}
    return "mixed" if len(flags) > 1 else "precision" if flags == {True} else "narrow"


def conforming_variant(source: str, deviation: str, edits: list[dict],
                       precision_bits: int | None = None) -> tuple[str | None, list[str]]:
    """The challenger's conforming replacement of one witness source: (variant, []) or (None, the problems).

    ``deviation`` is the exact text of ``source`` the finding says breaks the frozen definitions; ``edits`` are
    {"anchor", "replacement"} pairs; ``precision_bits`` is the frozen architecture contract's float width (None when
    no contract was frozen). Always:
    - the deviation is text of the source, and no occurrence of it is left in the variant;
    - each anchor is text of the source exactly once, and the anchors do not overlap;
    - the variant parses and differs from the source.
    A narrow edit (FIX-13), which cannot retune the witness, also has:
    - every line an anchor touches holds part of an occurrence of the deviation (the edit stays on those lines);
    - a replacement adds no number its anchor does not have (weights and scales stay), no name the source does not
      use (no new operation, store or library call), and keeps the precision names of its anchor (no change of
      dtype or precision).
    A precision edit (FIX-15) changes precision names, and restores the frozen precision and nothing else:
    - a frozen precision is declared, and every edit of the target is a precision edit (no mix with narrow edits);
    - each edit moves toward the frozen precision only (``precision_direction``);
    - the variant equals the source once float dtype casts and dtype arguments are removed from both (``_Uncast``):
      the weights, the numbers, the names, the operations and the structure are the source's. The edit may reach
      any line, since a precision deviation is usually spread over the whole update.
    The frozen spec, sizes, count, seed and architecture contract are not part of the edit: the host replays the
    variant under the frozen statement's own check."""
    if not isinstance(source, str) or not source.strip():
        return None, ["the statement has no recorded source"]
    if not isinstance(deviation, str) or not deviation.strip():
        return None, ["the deviation is empty"]
    deviations = _occurrences(source, deviation)
    if not deviations:
        return None, [f"the deviation {deviation[:120]!r} is not text of the source"]
    if not edits:
        return None, ["no edit is given"]
    if len(edits) > MAX_EDITS:
        return None, [f"{len(edits)} edits; at most {MAX_EDITS}"]
    kind = edit_kind(edits, precision_bits)
    if kind == "mixed":
        return None, [f"the edits mix a restoration of the frozen {precision_bits}-bit precision with other "
                      "changes; a conforming variant either changes dtype casts only, toward the frozen precision, "
                      "or makes a narrow edit that keeps every precision name"]
    deviation_lines = set().union(*(_lines(source, a, b) for a, b in deviations))
    source_names = set(_tokens(source)[0])
    problems, spans = [], []
    for index, edit in enumerate(edits, start=1):
        anchor, replacement = edit.get("anchor"), edit.get("replacement")
        if not isinstance(anchor, str) or not anchor or not isinstance(replacement, str):
            problems.append(f"edit {index}: anchor and replacement must be text, the anchor non-empty")
            continue
        found = _occurrences(source, anchor)
        if len(found) != 1:
            problems.append(f"edit {index}: the anchor {anchor[:120]!r} is text of the source {len(found)} times, "
                            "not once")
            continue
        start, end = found[0]
        spans.append((start, end, replacement))
        if kind == "precision":
            problems += [f"edit {index}: {p}" for p in precision_direction(anchor, replacement, precision_bits)]
            continue
        outside = sorted(_lines(source, start, end) - deviation_lines)
        if outside:
            problems.append(f"edit {index}: the anchor reaches line(s) {[n + 1 for n in outside]}, which hold no "
                            "part of the deviation")
        anchor_names, anchor_numbers = _tokens(anchor)
        names, numbers = _tokens(replacement)
        new_numbers = sorted({x for x in numbers if not any(x == y for y in anchor_numbers)})
        if new_numbers:
            problems.append(f"edit {index}: the replacement adds numbers {new_numbers} its anchor does not have")
        new_names = sorted(set(names) - source_names)
        if new_names:
            problems.append(f"edit {index}: the replacement uses names {new_names} the source does not use")
        if sorted(n for n in names if n in PRECISION_NAMES) != sorted(n for n in anchor_names
                                                                      if n in PRECISION_NAMES):
            problems.append(f"edit {index}: the replacement changes the precision names of its anchor"
                            + ("" if precision_bits is None else
                               f" (a precision edit sets the frozen {precision_bits}-bit precision and nothing else)"))
    spans.sort()
    for (_, end, _), (start, _, _) in zip(spans, spans[1:]):
        if start < end:
            problems.append("two anchors overlap")
            break
    if problems:
        return None, problems
    variant = source
    for start, end, replacement in reversed(spans):
        variant = variant[:start] + replacement + variant[end:]
    if variant == source:
        return None, ["the edits leave the source unchanged"]
    if deviation in variant:
        return None, [f"the variant still contains the deviation {deviation[:120]!r}"]
    try:
        tree = ast.parse(variant)
    except SyntaxError as exc:
        return None, [f"the variant does not parse: {exc.msg} at line {exc.lineno}"]
    if kind == "precision" and _uncast(tree) != _uncast(ast.parse(source)):
        return None, ["the variant changes more than float dtype casts and dtype arguments (astype, dtype=, "
                      f"np.float{precision_bits}(...), an array constructor's dtype): a precision edit restores the "
                      "frozen precision and leaves the weights, numbers, names and structure as they are"]
    return variant, []
