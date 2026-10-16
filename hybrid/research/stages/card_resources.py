"""Card completeness (FIX-16): the card's own resources, the completeness gate and the host verification of every
frozen value of a profile statement.

Cause. Calibration card K2 never stated its store width, so the statement fixer invented one per run, and that
choice decided P2.recall: at d = 64 the recall witness shares cells between keys and scores 0.752, at d = 128, 4K+8
or 8K+32 it passes (results/phase4-20261013T073748Z/STATUS.md, "K2 recall: cause found"). The audit of three
calibrations (fix16/audit-varying-frozen-values.txt) found the same freedom everywhere: widths, rounds, positions,
hidden width, state_bits, state_words, the test-instance count, the P1 cost grid and even the pass threshold. The
rule now is that a frozen statement contains no value that is neither in the card nor set by the protocol.

Three kinds of frozen field (profile.FIELD_KINDS):

- protocol: set by the host (profile.PROTOCOL, REQUIRED_SIZES, the thresholds, the generators' defaults and stream
  lengths). Never the fixer's choice.
- card: the model's own resources, from the card's structured ``resources`` section (theory_card.schema.json):
  the shared architecture (hidden width, precision, carried-state bits), the P1 cost sizes and widths, and per P2
  task d, rounds and state_words, per P4 task the solvable size, the example count m, d and rounds, as formulas in
  the task's size (L, K, depth, T, hops).
- derived: computed by the host from the other two (a P5 entry's state_bits and its D or V).

``normalize_statement`` and ``normalize_architecture`` fill in every protocol, card and derived value the fixer left
out and return a problem for every value it wrote that differs (formulas compared after SymPy simplification). The
host shows the problems in the fixer's one retry; at freeze a statement that still carries one is dropped
(math_contracts.freeze), so no frozen value can come from anywhere else.

Completeness gate. ``missing(card)`` lists what the card's ``resources`` do not state: the architecture, the P1
cost entry, d, rounds and state_words for every P2 task, size, examples, d and rounds for every P4 task (composition
or binding for P4.composition), and every symbol of the mechanism's state shapes that is neither a stream symbol, a
stated dimension, d, nor stated per task. Stage M runs it before the statement fixer and stage R after its dialogue
(profile on); ``request_resources`` asks the card's author for the missing items.

Conformance (FIX-20, decision PR). A conformance statement's points and tolerance are protocol values
(profile.conformance_points, PROTOCOL["conformance_tolerance"]) and its ``precision_bits`` is the card's
resources.architecture.precision_bits, the one precision policy both the counted reference and the frozen equations
run under (spec_checks.reference_conformance).

Stated limits (FIX-18, decision L). The card states its own P3 limits in its top-level ``limits`` list (task, size,
resources, why and kind: the part of the mechanism, state, update or readout, the limit follows from). ``missing``
reports a card that states none, so it is gated exactly like missing resources. The statement fixer copies the
card's limits and formalizes them: ``limit_copy_problems`` returns a problem for a limit in the fixer's answer that is
not one of the card's, and for a card limit the answer leaves out; ``frozen_limits`` freezes only the card's limits,
with the card's own text, so a limit the fixer invents never reaches the profile.
"""
from __future__ import annotations

import ast
import copy
import json

import sympy as sp

from hybrid.math import architecture, constructions, cost, learnability, profile
from hybrid.research.schemas import SchemaViolation, load_schema, validate
from hybrid.research.stages.common import StageContext, StageError, card_for_prompt, load_prompt

KILL_CODE = "card_incomplete"
TASKS = tuple(constructions.SIZE_NAME)
P4_REQUIRED = ("copy", "recall", "dyck")
P4_COMPOSITION = ("composition", "binding")
TASK_FIELDS = ("d", "rounds", "state_words")
LEARNING_FIELDS = ("size", "examples", "d", "rounds")
COST_FIELDS = ("sizes", "d", "rounds")
# FIX-18: a stated limit's task (a P2 task or "other") and the part of the mechanism it follows from (decision P3).
LIMIT_TASKS = (*TASKS, "other")
LIMIT_KINDS = ("state", "update", "readout")
# Shape symbols that the protocol sets: the vocabulary size of the task's layout.
PROTOCOL_SYMBOLS = frozenset({"V"})
STATE_P5 = {"copy": ("copy", "L"), "recall": ("recall", "K"), "dyck": ("dyck", "depth")}
P5_PARAM = {"copy": ("D", constructions.DEFAULTS["copy"]["D"]), "recall": ("V", constructions.DEFAULTS["recall"]["V"])}


# ----- formulas ------------------------------------------------------------------------------------------------------

def _symbols(names) -> dict:
    return {name: sp.Symbol(name, positive=True) for name in names}


def expression(text, names):
    """``text`` (a formula over ``names``, or a number) as a SymPy expression."""
    if isinstance(text, bool):
        raise architecture.ContractError(f"{text!r} is not a formula")
    if isinstance(text, (int, float)):
        return sp.nsimplify(text)
    problems = architecture.formula_problems(text, set(names))
    if problems:
        raise architecture.ContractError("; ".join(problems))
    return sp.sympify(text, locals=dict(_symbols(names), **architecture.FUNCTIONS))


def same(left, right, names) -> bool:
    """Whether two formulas over ``names`` are equal after SymPy simplification."""
    try:
        a, b = expression(left, names), expression(right, names)
    except (architecture.ContractError, sp.SympifyError, TypeError, ValueError):
        return False
    return a == b or sp.simplify(a - b) == 0


def evaluate(text, values: dict) -> float:
    return float(expression(text, set(values)).subs({_symbols(values)[k]: sp.Integer(v) for k, v in values.items()}))


def shape_symbols(card: dict) -> list[str]:
    """Every name used in the shapes of the card's state objects, in order of first use."""
    found: list[str] = []
    for item in card.get("mechanism", {}).get("state", []):
        for dim in item.get("shape", []):
            if not isinstance(dim, str):
                continue
            try:
                tree = ast.parse(dim, mode="eval")
            except SyntaxError:
                found.append(dim.strip())
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id not in architecture.FUNCTIONS and node.id not in found:
                    found.append(node.id)
    return found


# ----- the completeness gate ---------------------------------------------------------------------------------------

def _formula_item(where: str, value, names) -> list[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return [f"{where} (a formula in {', '.join(sorted(names)) or 'no size'})"]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return [f"{where}: {value!r} is not a formula"]
    if isinstance(value, str):
        problems = architecture.formula_problems(value, set(names))
        if problems:
            return [f"{where}: {'; '.join(problems)}"]
    return []


def missing(card: dict) -> list[str]:
    """What the card's ``resources`` leave unstated, as items naming the field (empty when the card is complete)."""
    resources = card.get("resources")
    if not isinstance(resources, dict):
        resources = {}
    items: list[str] = []
    if not resources:
        items.append("resources: the card states no resources section")
    arch = resources.get("architecture") or {}
    items += _formula_item("resources.architecture.hidden", arch.get("hidden"), {"d"})
    if arch.get("precision_bits") not in architecture.PRECISIONS:
        items.append(f"resources.architecture.precision_bits (one of {list(architecture.PRECISIONS)})")
    if arch.get("state_bits") is not None:
        items += _formula_item("resources.architecture.state_bits", arch["state_bits"], {"d"})
    if not any(item.startswith("resources.architecture") for item in items):
        # A stated contract the architecture check refuses (a carried-state budget above MAX_STATE_BITS) could never
        # be frozen, so the gate asks for it again rather than letting every fixer answer fail on it.
        shared = {key: arch[key] for key in ("hidden", "precision_bits", "state_bits") if arch.get(key) is not None}
        items += [f"resources.{problem}" for problem in architecture.shared_problems(shared)]
    dims = resources.get("dimensions") or {}
    for name, value in dims.items():
        items += _formula_item(f"resources.dimensions.{name}", value, {"d"})
    streams = set(resources.get("stream_symbols") or [])
    per_task = [s for s in shape_symbols(card) if s not in streams and s not in dims and s not in PROTOCOL_SYMBOLS
                and s != "d"]
    entry = resources.get("cost") or {}
    names = entry.get("sizes")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.isidentifier() and n != "n"
                                                           for n in names):
        items.append("resources.cost.sizes (the size parameters other than n that the P1 cost formula uses)")
        names = []
    for key in ("d", "rounds"):
        items += _formula_item(f"resources.cost.{key}", entry.get(key), {"n", *names})
    tasks = resources.get("tasks") or {}
    for task in TASKS:
        size = constructions.SIZE_NAME[task]
        stated = tasks.get(task) or {}
        for key in TASK_FIELDS:
            items += _formula_item(f"resources.tasks.{task}.{key}", stated.get(key), {size})
        symbols = stated.get("symbols") or {}
        for name in per_task:
            items += _formula_item(f"resources.tasks.{task}.symbols.{name}", symbols.get(name), {size})
    learning = resources.get("learning") or {}
    wanted = list(P4_REQUIRED) + [next((t for t in P4_COMPOSITION if t in learning), "composition")]
    for task in wanted:
        size = constructions.SIZE_NAME[task]
        stated = learning.get(task) or {}
        top = profile.SOLVABLE_MAX[task]
        value = stated.get("size")
        if not _int(value) or not 1 <= value <= top:
            items.append(f"resources.learning.{task}.size (the solvable case, 1 <= {size} <= {top})")
        value = stated.get("examples")
        if not _int(value) or not 1 <= value <= learnability.MAX_EXAMPLES:
            items.append(f"resources.learning.{task}.examples (the example count m the card states)")
        for key in ("d", "rounds"):
            items += _formula_item(f"resources.learning.{task}.{key}", stated.get(key), {size})
        symbols = stated.get("symbols") or {}
        for name in per_task:
            items += _formula_item(f"resources.learning.{task}.symbols.{name}", symbols.get(name), {size})
    items += limit_items(card)
    return items


def limit_items(card: dict) -> list[str]:
    """FIX-18: what the card's ``limits`` leave unstated: at least one limit, each naming a P2 task or "other", a size
    (an integer of at least 1, or null), the resources it holds within, why, and a kind for which the card's mechanism
    has an entry (a state limit needs a state object, an update limit an update equation, a readout limit a readout)."""
    limits = card.get("limits")
    if not isinstance(limits, list) or not limits:
        return ["limits: the card states no P3 limit (a task the mechanism cannot solve within stated resources: "
                "task, size, resources, why and kind, the part of the mechanism it follows from)"]
    mechanism = card.get("mechanism") or {}
    items = []
    for index, limit in enumerate(limits):
        where = f"limits[{index}]"
        if not isinstance(limit, dict):
            items.append(f"{where}: {limit!r} is not a limit")
            continue
        if limit.get("task") not in LIMIT_TASKS:
            items.append(f"{where}.task (one of {list(LIMIT_TASKS)})")
        size = limit.get("size")
        if size is not None and not (_int(size) and size >= 1):
            items.append(f"{where}.size (an integer of at least 1, or null)")
        for key in ("resources", "why"):
            if not isinstance(limit.get(key), str) or not limit[key].strip():
                items.append(f"{where}.{key}")
        kind = limit.get("kind")
        if kind not in LIMIT_KINDS:
            items.append(f"{where}.kind (one of {list(LIMIT_KINDS)}: the part of the mechanism the limit follows "
                         "from, decision P3)")
        elif not mechanism.get(kind):
            items.append(f"{where}.kind: {kind}, but the card's mechanism states no {kind} entry")
    return items


def complete(card: dict) -> bool:
    return not missing(card)


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ----- host verification of the fixer's values ---------------------------------------------------------------------

class _Fields:
    """Collects the filled-in spec and the problems of one statement."""

    def __init__(self, label: str, spec: dict):
        self.label = label
        self.spec = copy.deepcopy(spec) if isinstance(spec, dict) else spec
        self.problems: list[str] = []

    def number(self, key: str, value, kind: str) -> None:
        given = self.spec.get(key)
        if given is not None and given != value:
            self.problems.append(f"{self.label}: {key} is a {kind} value, {_json(value)}; the statement gives "
                                 f"{_json(given)}. Leave it out or copy it.")
        self.spec[key] = copy.deepcopy(value)

    def formula(self, path: tuple[str, ...], value, names, kind: str, source: str) -> None:
        holder = self.spec
        for key in path[:-1]:
            if not isinstance(holder.get(key), dict):
                holder[key] = {}
            holder = holder[key]
        key = path[-1]
        given = holder.get(key)
        if value is None:
            if given is not None:
                self.problems.append(f"{self.label}: {'.'.join(path)} is a {kind} value that is left out "
                                     f"({source}); the statement gives {_json(given)}")
            holder.pop(key, None)
            return
        if given is not None and not same(given, value, names):
            self.problems.append(f"{self.label}: {'.'.join(path)} must be {_json(value)} ({source}); the statement "
                                 f"gives {_json(given)}. A frozen value comes from the card or the protocol, never "
                                 "from the statement fixer.")
        holder[key] = value if isinstance(value, str) else str(value)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _params(fields: _Fields, task: str, sizes) -> None:
    given = fields.spec.get("params")
    defaults = dict(constructions.DEFAULTS[task])
    if given:
        try:
            same_layout = all(constructions.layout(task, s, given) == constructions.layout(task, s, None)
                              for s in sizes if _int(s) and s >= 1)
        except ValueError:
            same_layout = False
        if not same_layout:
            fields.problems.append(f"{fields.label}: params are protocol values, the generator defaults "
                                   f"{_json(defaults)}; the statement gives {_json(given)}")
            fields.spec["params"] = defaults
        return
    if defaults:
        fields.spec["params"] = defaults
    else:
        fields.spec.pop("params", None)


def _task_entry(resources: dict, section: str, task: str) -> dict:
    return ((resources.get(section) or {}).get(task)) or {}


def normalize_statement(card: dict, claim_id, form: str, spec) -> tuple[dict, list[str]]:
    """FIX-16: ``spec`` with every protocol, card and derived value filled in, and the problems with the values the
    fixer wrote. A card without a ``resources`` section is returned as written (the completeness gate keeps such
    cards from a profile run)."""
    resources = card.get("resources")
    if not isinstance(resources, dict) or not isinstance(spec, dict):
        return spec, []
    slot = profile.is_slot(claim_id)
    label = f"{claim_id or 'premise obligation'} {form}"
    if form == "construction":
        return _construction(resources, spec, slot, label)
    if form == "learnability":
        return _learnability(resources, spec, slot, label)
    if form == "cost":
        return _cost(resources, spec, label)
    if form == "conformance":
        return _conformance(resources, spec, label)
    if form == "conservation" and claim_id == "P5":
        return _conservation(resources, spec, label)
    return spec, []


def _construction(resources: dict, spec: dict, slot: bool, label: str) -> tuple[dict, list[str]]:
    task = spec.get("task")
    if task not in constructions.SIZE_NAME:
        return spec, []
    fields, size = _Fields(label, spec), constructions.SIZE_NAME[task]
    if slot:
        required = list(profile.REQUIRED_SIZES[task])
        given = spec.get("sizes")
        if given is not None and sorted(given) != required:
            fields.problems.append(f"{label}: sizes are protocol values, exactly {required}; the statement gives "
                                   f"{_json(given)}")
        fields.spec["sizes"] = required
    _params(fields, task, fields.spec.get("sizes") or [])
    fields.number("count", profile.PROTOCOL["p2_count"], "protocol")
    fields.number("threshold", profile.P2_THRESHOLD, "protocol")
    stated = _task_entry(resources, "tasks", task)
    source = f"the card's resources.tasks.{task}"
    for key in ("d", "rounds"):
        fields.formula(("widths", key), stated.get(key), {size}, "card", source)
    fields.formula(("widths", "positions"), profile.protocol_positions(task), {size}, "protocol",
                   "the generator's stream length")
    fields.formula(("state_words",), stated.get("state_words"), {size}, "card", source)
    return fields.spec, fields.problems


def _learnability(resources: dict, spec: dict, slot: bool, label: str) -> tuple[dict, list[str]]:
    task = spec.get("task")
    if task not in constructions.SIZE_NAME:
        return spec, []
    fields, size = _Fields(label, spec), constructions.SIZE_NAME[task]
    learned = _task_entry(resources, "learning", task)
    if slot:
        if not learned:
            fields.problems.append(f"{label}: the card's resources.learning states no P4 case for task {task}; "
                                   f"use a task it states ({', '.join(sorted(resources.get('learning') or {}))})")
        else:
            fields.number("size", learned.get("size"), "card")
            fields.number("examples", learned.get("examples"), "card")
    chosen = spec.get("size") if not slot else fields.spec.get("size")
    _params(fields, task, [chosen])
    fields.spec["seeds"] = _seeds(fields, spec.get("seeds"))
    fields.number("test_count", profile.PROTOCOL["p4_test_count"], "protocol")
    fields.number("threshold", profile.P4_THRESHOLD, "protocol")
    widths = learned if learned and learned.get("size") == chosen else _task_entry(resources, "tasks", task)
    source = (f"the card's resources.learning.{task}" if widths is learned
              else f"the card's resources.tasks.{task}")
    for key in ("d", "rounds"):
        fields.formula(("widths", key), widths.get(key), {size}, "card", source)
    fields.formula(("widths", "positions"), profile.protocol_positions(task), {size}, "protocol",
                   "the generator's stream length")
    return fields.spec, fields.problems


def _seeds(fields: _Fields, given) -> list[int]:
    wanted = list(profile.PROTOCOL["p4_seeds"])
    if given is not None and sorted(given) != wanted:
        fields.problems.append(f"{fields.label}: seeds are protocol values, {wanted}; the statement gives "
                               f"{_json(given)}")
    return wanted


def cost_grid(resources: dict) -> dict[str, list[int]]:
    """The P1 grid the protocol sets for the card's cost sizes."""
    names = (resources.get("cost") or {}).get("sizes") or []
    return {name: list(profile.PROTOCOL["cost_grid"]) for name in names}


def _cost(resources: dict, spec: dict, label: str) -> tuple[dict, list[str]]:
    fields = _Fields(label, spec)
    entry = resources.get("cost") or {}
    grid = cost_grid(resources)
    given = spec.get("sizes")
    if given is not None and given != grid:
        fields.problems.append(f"{label}: sizes must be {_json(grid)}: the size names are the card's "
                               f"(resources.cost.sizes) and the values the protocol's grid; the statement gives "
                               f"{_json(given)}")
    fields.spec["sizes"] = grid
    ns = profile.PROTOCOL["p1_ns"]
    if ns is None:
        if spec.get("ns") is not None:
            fields.problems.append(f"{label}: ns is a protocol value: leave it out (the plan's lengths "
                                   f"{list(cost.DEFAULT_NS)}); the statement gives {_json(spec['ns'])}")
        fields.spec.pop("ns", None)
    else:
        fields.number("ns", list(ns), "protocol")
    names = {"n", *grid}
    for key in ("d", "rounds"):
        fields.formula(("widths", key), entry.get(key), names, "card", "the card's resources.cost")
    fields.formula(("widths", "positions"), "n", names, "protocol", "P1 positions are n")
    return fields.spec, fields.problems


def _point_key(point) -> list:
    return sorted(point.items()) if isinstance(point, dict) else [("", repr(point))]


def _conformance(resources: dict, spec: dict, label: str) -> tuple[dict, list[str]]:
    """FIX-20 (decision PR): a conformance audit's points and tolerance are protocol values and its precision is the
    card's declared precision. The host fills all three; a value the fixer wrote that differs is a problem fed back
    to it (in K2 run 2 of calibration 20261016T051933Z the fixer chose the grid, n = 256 at one point only, and the
    tolerance 1e-6, and the audit compared float64 reference arithmetic with float32 equations)."""
    fields = _Fields(label, spec)
    points = profile.conformance_points((resources.get("cost") or {}).get("sizes") or [])
    given = spec.get("points")
    if given is not None and (not isinstance(given, list)
                              or sorted(map(_point_key, given)) != sorted(map(_point_key, points))):
        fields.problems.append(f"{label}: points are protocol values, {_json(points)} (every combination of the cost "
                               f"grid {list(profile.PROTOCOL['cost_grid'])} over the card's cost sizes, each at n = "
                               f"{profile.PROTOCOL['conformance_n']}); the statement gives {_json(given)}. Leave them "
                               "out.")
    fields.spec["points"] = points
    tolerance = profile.PROTOCOL["conformance_tolerance"]
    if "tolerance" in spec and spec["tolerance"] != tolerance:
        fields.problems.append(f"{label}: tolerance is a protocol value, {tolerance:g} (both sides run at the card's "
                               f"precision, so they agree exactly); the statement gives {_json(spec['tolerance'])}. "
                               "Leave it out.")
    fields.spec["tolerance"] = tolerance
    fields.number("precision_bits", (resources.get("architecture") or {}).get("precision_bits"),
                  "card (resources.architecture.precision_bits)")
    return fields.spec, fields.problems


def _entry_bits(resources: dict, task: str, size: int):
    """A P5 entry's derived state bits at one size: the card's carried-state budget at that size's d when it states
    one, else its state words times its precision."""
    arch = resources.get("architecture") or {}
    stated = _task_entry(resources, "tasks", task)
    name = constructions.SIZE_NAME[task]
    if arch.get("state_bits") is not None:
        d = evaluate(stated["d"], {name: size})
        bits = arch["state_bits"]
        value = float(bits) if not isinstance(bits, str) else evaluate(bits, {"d": d})
    else:
        value = evaluate(stated["state_words"], {name: size}) * arch["precision_bits"]
    return int(value) if float(value).is_integer() else value


def _conservation(resources: dict, spec: dict, label: str) -> tuple[dict, list[str]]:
    fields = _Fields(label, spec)
    arch = resources.get("architecture") or {}
    bits = arch.get("state_bits")
    if "state_bits" in spec:
        constant = bits is not None and not isinstance(bits, str)
        if not constant or spec["state_bits"] != bits:
            fields.problems.append(f"{label}: a statement-wide state_bits is allowed only as the card's constant "
                                   f"carried-state budget; the host derives each copy, recall and dyck entry's "
                                   f"state_bits from the card (got {_json(spec['state_bits'])})")
        fields.spec.pop("state_bits", None)
    for task, (section, field) in STATE_P5.items():
        entries = spec.get(section)
        if not isinstance(entries, list):
            continue
        filled = []
        for entry in entries:
            if not isinstance(entry, dict) or not _int(entry.get(field)):
                filled.append(entry)
                continue
            entry = dict(entry)
            where = f"{label}: the {section} entry at {field} = {entry[field]}"
            try:
                derived = _entry_bits(resources, task, entry[field])
            except (KeyError, architecture.ContractError, TypeError, ValueError) as exc:
                fields.problems.append(f"{where}: its state_bits cannot be derived from the card: {exc}")
                filled.append(entry)
                continue
            given = entry.get("state_bits")
            if given is not None and abs(float(given) - float(derived)) > 1e-9:
                fields.problems.append(f"{where}: state_bits is derived from the card's resources, {derived}; "
                                       f"the entry gives {_json(given)}")
            entry["state_bits"] = derived
            if task in P5_PARAM:
                name, value = P5_PARAM[task]
                if entry.get(name) is not None and entry[name] != value:
                    fields.problems.append(f"{where}: {name} is the generator default {value}; the entry gives "
                                           f"{_json(entry[name])}")
                entry[name] = value
            filled.append(entry)
        fields.spec[section] = filled
    return fields.spec, fields.problems


def normalize_architecture(card: dict, given) -> tuple[dict | None, list[str]]:
    """The shared architecture contract from the card's resources, and the problems with the fixer's copy of it."""
    resources = card.get("resources")
    if not isinstance(resources, dict):
        return given, []
    arch = resources.get("architecture") or {}
    wanted = {"hidden": arch.get("hidden"), "precision_bits": arch.get("precision_bits")}
    if arch.get("state_bits") is not None:
        wanted["state_bits"] = arch["state_bits"]
    problems = []
    given = given if isinstance(given, dict) else {}
    if given.get("hidden") is not None and not same(given["hidden"], wanted["hidden"], {"d"}):
        problems.append(f"architecture.hidden must be {_json(wanted['hidden'])} (the card's resources); the answer "
                        f"gives {_json(given['hidden'])}")
    if given.get("precision_bits") is not None and given["precision_bits"] != wanted["precision_bits"]:
        problems.append(f"architecture.precision_bits must be {wanted['precision_bits']} (the card's resources); the "
                        f"answer gives {given['precision_bits']}")
    stated, found = wanted.get("state_bits"), given.get("state_bits")
    if found is not None and (stated is None or not same(found, stated, {"d"})):
        problems.append(f"architecture.state_bits must be {_json(stated)} (the card's resources); the answer gives "
                        f"{_json(found)}")
    return wanted, problems


def resources_text(card: dict) -> str:
    """The card's resources for the statement fixer's message: what it copies."""
    resources = card.get("resources")
    if not isinstance(resources, dict):
        return ""
    return ("CARD RESOURCES (FIX-16). Copy these values; invent none. The architecture contract is "
            "resources.architecture. Every P2 statement for a task uses resources.tasks.<task> (widths.d, "
            "widths.rounds, state_words); every P4 statement uses resources.learning.<task> (size, examples, "
            "widths.d, widths.rounds); every P1 cost statement uses resources.cost (the size names and widths.d, "
            "widths.rounds). Each P5 copy, recall and dyck entry's state_bits is derived by the host from these "
            "(the carried-state budget at the task's d, else state_words times precision_bits): leave it out. Any "
            "frozen value that differs from the card or from the PROTOCOL VALUES is refused at freeze.\n"
            + json.dumps(resources, indent=2, ensure_ascii=False))


# ----- stated limits (FIX-18) -----------------------------------------------------------------------------------------

def stated_limits(card: dict) -> list[dict] | None:
    """The card's own limits, or None for a card made before FIX-18 (the gate keeps such cards from a profile run)."""
    limits = card.get("limits")
    return limits if isinstance(limits, list) else None


def _text(value) -> str:
    return " ".join(str(value or "").split())


def _limit_of_card(limit, stated: list[dict]) -> int | None:
    """The index of the card limit a fixer limit copies: the same task, size and part (its basis part is the card
    limit's kind)."""
    if not isinstance(limit, dict):
        return None
    part = (limit.get("basis") or {}).get("part") if isinstance(limit.get("basis"), dict) else None
    for index, card_limit in enumerate(stated):
        if (card_limit.get("task"), card_limit.get("size"), card_limit.get("kind")) == (
                limit.get("task"), limit.get("size"), part):
            return index
    return None


def _limit_label(limit) -> str:
    if not isinstance(limit, dict):
        return repr(limit)
    part = (limit.get("basis") or {}).get("part") if isinstance(limit.get("basis"), dict) else None
    return f"task {limit.get('task')}, size {limit.get('size')}, part {part or limit.get('kind')}"


def limit_copy_problems(card: dict, limits) -> list[tuple[int | None, str]]:
    """FIX-18: (index of the fixer's limit, or None for a card limit it left out, problem). A limit the card does not
    state is invented; a card limit missing from the answer was not formalized; a copied limit whose resources or why
    differ from the card's text is shown too (the host freezes the card's text). Empty for a card made before FIX-18."""
    stated = stated_limits(card)
    if stated is None:
        return []
    found: list[tuple[int | None, str]] = []
    copied = set()
    for index, limit in enumerate(limits or []):
        match = _limit_of_card(limit, stated)
        label = f"P3 limit {index + 1} ({_limit_label(limit)})"
        if match is None:
            found.append((index, f"{label} is not one of the card's stated limits (CARD LIMITS). The card states its "
                                 "own P3 limits; copy them (task, size, resources, why, and basis.part equal to the "
                                 "limit's kind) and formalize them in P3. Invent no limit (DECISIONS.md L)"))
            continue
        copied.add(match)
        for key in ("resources", "why"):
            if _text(limit.get(key)) != _text(stated[match].get(key)):
                found.append((index, f"{label}: {key} must be the card's text for card limit {match + 1}, "
                                     f"{_json(stated[match].get(key))}; copy it"))
    for index, card_limit in enumerate(stated):
        if index not in copied:
            found.append((None, f"card limit {index + 1} ({_limit_label(card_limit)}) is not in the answer's limits; "
                                "copy every limit the card states and formalize it in P3"))
    return found


def invented_limits(card: dict, limits) -> list[tuple[int, str]]:
    """FIX-18: (index, problem) for each limit of the fixer's answer that the card does not state."""
    stated = stated_limits(card)
    if stated is None:
        return []
    return [(index, problem) for index, problem in limit_copy_problems(card, limits)
            if index is not None and _limit_of_card(limits[index], stated) is None]


def frozen_limits(card: dict, limits: list[dict]) -> list[dict]:
    """FIX-18: the limits the host freezes, given the fixer limits that passed the other checks: each copied card
    limit with the card's own task, size, resources and why and the fixer's basis (the anchor and quote FIX-17
    checks). A fixer limit the card does not state is left out. A card made before FIX-18 keeps the fixer's limits."""
    stated = stated_limits(card)
    if stated is None:
        return copy.deepcopy(limits)
    kept, seen = [], set()
    for limit in limits:
        match = _limit_of_card(limit, stated)
        if match is None or match in seen:
            continue
        seen.add(match)
        card_limit = stated[match]
        kept.append({"task": card_limit["task"], "size": card_limit["size"], "resources": card_limit["resources"],
                     "why": card_limit["why"], "basis": copy.deepcopy(limit["basis"])})
    return kept


def limits_text(card: dict) -> str:
    """The card's limits for the statement fixer's message: what P3 formalizes."""
    stated = stated_limits(card)
    if stated is None:
        return ""
    return ("CARD LIMITS (FIX-18). The card states its own P3 limits. Copy each one into `limits` with the same task, "
            "size, resources and why, and with basis.part equal to its kind (give the anchor and a quote of the "
            "card's mechanism text there); write P3 statements that formalize them. Invent no limit and leave none "
            "out: a limit the card does not state is refused at freeze.\n"
            + json.dumps(stated, indent=2, ensure_ascii=False))


# ----- asking the card's author --------------------------------------------------------------------------------------

def resources_schema() -> dict:
    """The card schema's resources section with its references inlined, for a reply schema of its own."""
    card = load_schema("theory_card")

    def inline(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return inline(card["$defs"][ref.removeprefix("#/$defs/")])
            return {key: inline(value) for key, value in node.items()}
        if isinstance(node, list):
            return [inline(item) for item in node]
        return copy.deepcopy(node)

    return inline(card["properties"]["resources"])


def limits_schema() -> dict:
    """FIX-18: the card schema's limits list with its references inlined."""
    card = load_schema("theory_card")
    return {"type": "array", "items": copy.deepcopy(card["$defs"]["card_limit"])}


def request_schema() -> dict:
    """The author's answer: the complete resources section, the card's P3 limits (FIX-18) and why."""
    return {"type": "object", "additionalProperties": False, "required": ["resources", "why"],
            "properties": {"resources": resources_schema(), "limits": limits_schema(),
                           "why": {"type": "string", "minLength": 1}}}


async def request_resources(ctx: StageContext, card_id: str, items: list[str], stage: str) -> dict:
    """Ask the card's author for the missing resources and store the answer as the card's next version. Returns the
    revised card (validated); raises StageError when the answer does not make a valid card."""
    from hybrid.research.stages.s1_wellformed import explorer_role, store_revision

    card = ctx.card(card_id)
    system = (load_prompt("card_resources").rstrip() + "\n\n## General card rules (shared with every explorer)\n\n"
              + load_prompt("explorer").split("\n", 1)[1].strip() + "\n")
    user = "\n\n".join(["YOUR CARD:\n" + card_for_prompt(card),
                        "MISSING RESOURCES (state each one):\n" + "\n".join(f"- {item}" for item in items),
                        "Return the complete resources section (keep what the card already states unless it is "
                        "wrong), the card's P3 limits under `limits` (keep the ones it already states unless they "
                        "are wrong) and one sentence on why these are the mechanism's resources, as JSON."])
    reply = await ctx.call(explorer_role(ctx, card_id), system, user, request_schema())
    revised = copy.deepcopy(ctx.store.get_card(card_id))
    revised["resources"] = reply.parsed["resources"]
    if "limits" in reply.parsed:
        revised["limits"] = reply.parsed["limits"]
    revised["version"] = ctx.store.next_version(card_id)
    revised["created_by_call"] = reply.call_id
    try:
        validate("theory_card", revised)
    except SchemaViolation as exc:
        raise StageError(f"card with the requested resources fails the theory_card schema: {exc}") from exc
    store_revision(ctx, revised, reply.call_id, stage=stage, purpose="resources", missing=items)
    ctx.emit("card_resources_requested", {"card_id": card_id, "stage": stage, "call_id": reply.call_id,
                                          "missing_before": items, "missing_after": missing(revised),
                                          "version": revised["version"]})
    return revised
