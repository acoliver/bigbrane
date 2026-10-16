"""The transformer profile: obligations P1 to P5 every card must discharge (DISCOVERY-PIPELINE-PLAN.md sections 4.4
and 4.5, Phase 4).

Each obligation corresponds to something known about the transformer, so a candidate is held to what is known about
the model it must beat. The profile is a template with a fixed shape: the statement fixer fills in the card's
parameters (formulas, resources, numbers, the learner's example count) but cannot change the slots, the forms a slot
accepts, the sizes it must cover or the thresholds. The shape lives here, in a hashed check file (MANIFEST.json), so
a model role cannot edit it.

Slots (each is a claim with this id that the statement fixer must cover; kill codes name the slot, for example
``obligation_failed:P2.nesting`` or ``unproven:P3``):

| Slot | Forms accepted | Tier it can reach | What it requires |
| --- | --- | --- | --- |
| P1 | cost | executed | exact op and state formulas, verified by running the reference code at several lengths |
| P2.copy, P2.recall, P2.nesting, P2.composition | construction | executed | a hand-set configuration meets 0.99 at every required size, within its declared state words |
| P3 | lean, sympy, z3 | lean, sympy_exact or smt_bounded | a stated limit with a derivation, consistent with P2 |
| P4.copy, P4.recall, P4.nesting, P4.composition | learnability | executed | learned from m examples in a solvable case on three seeds |
| P5 | conservation | executed | the card's numbers pass the conservation laws, with an entry at every P2 size and P4's m; each copy, recall and dyck accuracy cites a card claim or is the host-computed ceiling |

A profile slot holds only when every statement it has holds (``require_all``): P2.composition needs both the two-hop
lookup and the variable-binding variant. Statements of a form the slot does not accept are dropped before freezing.

Checks after the statements ran (``consistency``): a P3 limit that a P2 construction beat at or above the stated size
fails P3; a P2 construction that met the threshold at a size P5 does not account for (an entry at that size or above
with accuracy at least the P2 threshold) fails P5, so the information bound is computed for what the construction
actually did.

Shared architecture contract (FIX-4, architecture.py). The fixer freezes ``architecture`` with the profile: the
feed-forward hidden width as a formula in d, the precision of stored numbers and, for a finite-state mechanism, the
carried-state budget ``state_bits`` (FIX-8; a bits budget, not a float width). Every P1, P2
and P4 statement declares ``widths`` (d and rounds, optionally positions) in its own size names, and the checks hold
the reference code, the witnesses and the trained models to those dimensions. A P5 sample entry defines the
target-rule class whose size it uses (``hypothesis_class``: what the class is, why it has that many members, and
log2 of the size, equal to ``log2_H``).

Self-contained (stdlib and the sibling modules) like the other check modules.
"""
from __future__ import annotations

import re

from . import architecture, conservation, constructions, cost, lean_check, learnability, tiers

PROFILE_VERSION = "phase4-2026-10-13"
P2_SLOTS = {"P2.copy": ("copy",), "P2.recall": ("recall",), "P2.nesting": ("dyck",),
            "P2.composition": ("composition", "binding")}
P4_SLOTS = {"P4.copy": ("copy",), "P4.recall": ("recall",), "P4.nesting": ("dyck",),
            "P4.composition": ("composition", "binding")}
SLOTS = ("P1", *P2_SLOTS, "P3", *P4_SLOTS, "P5")
DERIVATION_FORMS = ("lean", "sympy", "z3")
SLOT_FORMS = {"P1": ("cost",), **{slot: ("construction",) for slot in P2_SLOTS}, "P3": DERIVATION_FORMS,
              **{slot: ("learnability",) for slot in P4_SLOTS}, "P5": ("conservation",)}
FORM_TIER = {"cost": tiers.EXECUTED, "construction": tiers.EXECUTED, "learnability": tiers.EXECUTED,
             "conservation": tiers.EXECUTED, "lean": tiers.LEAN, "sympy": tiers.SYMPY_EXACT, "z3": tiers.SMT_BOUNDED}
# Forms a profile statement may use beyond the Phase 3 forms.
PROFILE_FORMS = ("conservation", "learnability")

# P2: sizes every construction must be run at (the fixer may add sizes, up to the harness limit of 8) and the
# threshold (plan section 4.4: at least 0.99 for every listed size).
REQUIRED_SIZES = {"copy": (8, 32, 64), "recall": (8, 32, 64), "dyck": (2, 4, 8, 16), "composition": (4, 16, 32),
                  "binding": (1, 4, 8)}
P2_THRESHOLD = 0.99
# P1: the plan's grid of lengths is the default; a statement that names its own lengths needs this many distinct
# values spanning at least this factor.
P1_MIN_LENGTHS = 4
P1_MIN_SPAN = 8
# P4: the solvable case (small V, small K, depth 2) and the seeds.
SOLVABLE_MAX = {"copy": 4, "recall": 4, "dyck": 2, "composition": 4, "binding": 2}
P4_SEEDS = 3
P4_THRESHOLD = 0.99
# P5: conservation sections that must cover a P2 task, with the size field they use.
P5_SECTIONS = {"copy": ("copy", "L"), "recall": ("recall", "K"), "dyck": ("dyck", "depth")}
# FIX-16: the protocol values of every profile statement. They are set by the host, never by the statement fixer:
# a fixer value that differs is refused at freeze, and an absent one is filled in from here. In the 2026-10-14 and
# 2026-10-15 calibrations the fixer chose count 200 or 500, test_count, the P1 cost grid and even the K2 threshold
# (0.99 or 1) per run (results/phase4-20261013T073748Z/fix16/audit-varying-frozen-values.txt). Where the plan names a
# value it is used (sizes, thresholds, seeds, the P1 lengths); where runs disagreed and the plan is silent one host
# rule is chosen (FIX-16.md): P2 count 200, the instances the TASK CONTRACTS are measured on (CONTRACT_SAMPLE);
# P4 test_count 500, the most any run used and the harness maximum; every P1 size other than n at 8 and 32.
PROTOCOL = {
    "p2_count": 200,
    "p4_test_count": 500,
    "p4_seeds": (0, 1, 2),
    # None: the cost statement leaves ns out and the verifier runs the plan's lengths 64 to 4096.
    "p1_ns": None,
    "cost_grid": (8, 32),
    # FIX-20 (decision PR): a conformance audit runs at every combination of the cost grid's values over the card's
    # cost sizes, each at this n: the smallest P1 length that covers the longest token stream of a P2 protocol size
    # (recall K = 64, 193 tokens), and the audit's upper limit. The fixer chose 2 to 4 points and n from 64 to 256 per
    # run in calibration 20261016T051933Z.
    "conformance_n": 256,
    # FIX-20: exact agreement. Both sides run at the card's declared precision and the operation sequence the cost
    # reference definition fixes, so they perform the same IEEE operations on the same inputs: every recorded K1 to K4
    # audit of 20261016T051933Z gave a gap of exactly 0.0 under the shared policy. A reordered sum moved K2's binary32
    # state by 36 to 124 eps over 256 tokens and a mechanism change by 1e7 eps or more
    # (results/phase4-20261013T073748Z/FIX-20.md), so no slack between 0 and those separates anything real.
    "conformance_tolerance": 0.0,
}
# FIX-16: how every frozen field of a profile statement is set. "protocol": by the host (PROTOCOL, REQUIRED_SIZES,
# P2_THRESHOLD, the generators); "card": copied from the card's `resources` and verified symbol for symbol;
# "derived": computed by the host from the other two; "claim": the fixer's formalization of what the card claims,
# checked by the existing rules (statement match, the checks themselves).
FIELD_KINDS = {
    "construction": {"task": "protocol", "sizes": "protocol", "params": "protocol", "count": "protocol",
                     "threshold": "protocol", "widths.positions": "protocol", "widths.d": "card",
                     "widths.rounds": "card", "state_words": "card", "token_store": "claim"},
    "learnability": {"task": "protocol", "params": "protocol", "seeds": "protocol", "threshold": "protocol",
                     "test_count": "protocol", "widths.positions": "protocol", "size": "card", "examples": "card",
                     "widths.d": "card", "widths.rounds": "card"},
    "cost": {"ns": "protocol", "sizes (values)": "protocol", "widths.positions": "protocol", "sizes (names)": "card",
             "widths.d": "card", "widths.rounds": "card", "ops": "claim", "state": "claim", "exponent": "claim",
             "ops_remainder": "claim", "state_remainder": "claim"},
    "conformance": {"points (sizes other than n)": "protocol", "points (n)": "protocol", "tolerance": "protocol",
                    "precision_bits": "card", "reference": "claim", "equations": "claim", "streams": "claim"},
    "architecture": {"hidden": "card", "precision_bits": "card", "state_bits": "card"},
    "conservation": {"copy/recall/dyck sizes": "protocol", "D, V": "protocol", "entry state_bits": "derived",
                     "accuracy": "claim", "sample": "claim"},
}


def protocol_positions(task: str) -> str | None:
    """The frozen ``widths.positions`` of a P2 or P4 statement (FIX-16): the generator's token-stream length as a
    formula in the size when it is exactly linear, else None (the field is left out and the harness uses the
    longest stream at each size, as for dyck, whose length varies per instance)."""
    return constructions.length_formula(task)


# FIX-7: P4 is existential in the size. The fixer selects one solvable case at or below the ceiling; P2 is the slot
# held to every listed size. The k3 calibration matcher read "at L at most 4" as "at every L up to 4"
# (results/phase4-20261013T073748Z/match-diag/REPORT.md).
P4_SELECTED = ("; the case is one selected {what} {name} with 1 <= {name} <= {top}: the ceiling bounds the "
               "choice, it is not a size grid (P2, not P4, covers every listed size)")

SLOT_TEXT = {
    "P1": "exact cost: symbolic formulas for the total counted operations over n tokens and the peak inference state "
          "words, in n and the card's own size parameters, equal the operations and state counted when the card's "
          "reference code runs at several sequence lengths, with the claimed exponent in n",
    "P2.copy": "construction, copy: a hand-set configuration of the mechanism, with no learning, reproduces a uniform "
               "string after a separator with per-string accuracy at least 0.99 at every length L in "
               + ", ".join(map(str, REQUIRED_SIZES["copy"])) + ", using at most its declared state words",
    "P2.recall": "construction, associative recall: a hand-set configuration reads K key-value pairs, then answers "
                 "every key once with per-query accuracy at least 0.99 at every K in "
                 + ", ".join(map(str, REQUIRED_SIZES["recall"])) + ", using at most its declared state words",
    "P2.nesting": "construction, nesting: a hand-set configuration predicts the legal closing bracket of two-type "
                  "balanced bracket prefixes with per-sequence accuracy at least 0.99 at every depth in "
                  + ", ".join(map(str, REQUIRED_SIZES["dyck"])) + ", using at most its declared state words",
    "P2.composition": "construction, composition: a hand-set configuration answers f(g(x)) for tables f and g given "
                      "in context with per-query accuracy at least 0.99 at every table size in "
                      + ", ".join(map(str, REQUIRED_SIZES["composition"]))
                      + ", and resolves chained variable assignments at every chain length in "
                      + ", ".join(map(str, REQUIRED_SIZES["binding"])) + ", using at most its declared state words",
    "P3": "derived limit: at least one task family the mechanism cannot solve within its resources, with a derivation "
          "checked by Lean, SymPy or Z3, and consistent with the constructions of P2",
    "P4.copy": "toy learnability, copy: from its initial state, using only its own learning rule, the mechanism learns "
               "the copy task at L at most 4 from the stated number of examples m, reaching 0.99 on three seeds"
               + P4_SELECTED.format(what="length", name="L", top=SOLVABLE_MAX["copy"]),
    "P4.recall": "toy learnability, recall: the mechanism learns associative recall at K at most 4 from the stated m "
                 "examples with its own learning rule, reaching 0.99 on three seeds"
                 + P4_SELECTED.format(what="pair count", name="K", top=SOLVABLE_MAX["recall"]),
    "P4.nesting": "toy learnability, nesting: the mechanism learns the bracket task at depth at most 2 from the stated "
                  "m examples with its own learning rule, reaching 0.99 on three seeds"
                  + P4_SELECTED.format(what="nesting", name="depth", top=SOLVABLE_MAX["dyck"]),
    "P4.composition": "toy learnability, composition: the mechanism learns two-hop lookup at table size at most 4, or "
                      "variable binding at chain length at most 2, from the stated m examples with its own learning "
                      "rule, reaching 0.99 on three seeds; the case is one selected table size T with 1 <= T <= "
                      f"{SOLVABLE_MAX['composition']}, or one selected chain length with 1 <= hops <= "
                      f"{SOLVABLE_MAX['binding']}: the ceiling bounds the choice, it is not a size grid (P2, not P4, "
                      "covers every listed size)",
    # FIX-8 appends the per-size clause and FIX-11 the accuracy clause; the earlier text stays a prefix so recorded
    # claim quotes still match it.
    "P5": "information bounds: the card's state bits against the bits the P2 tasks need at the sizes P2 covers, and "
          "the stated example counts m against the bits needed to identify the target rule, pass the conservation "
          "laws, with an entry at every size P2 covers (each copy L, recall K and dyck depth); each such entry's "
          "accuracy is either the card's own number at that size, citing the card claim that states it, or the "
          "information ceiling of the card's state bits at that size",
}


def is_slot(claim_id) -> bool:
    return claim_id in SLOT_FORMS


def profile_claims() -> list[dict]:
    """The profile slots as claims for the statement fixer, in slot order."""
    return [{"id": slot, "text": f"profile {slot}: {SLOT_TEXT[slot]}"} for slot in SLOTS]


def accepted_tiers(slot: str) -> list[str]:
    return sorted({FORM_TIER[form] for form in SLOT_FORMS[slot]})


def tasks_of(slot: str) -> tuple:
    return P2_SLOTS.get(slot) or P4_SLOTS.get(slot) or ()


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def statement_problems(slot: str, form: str, spec) -> list[str]:
    """Why a statement cannot serve ``slot`` (empty when it can). The form's own data checks are separate
    (spec_checks.statement_problems); these are the profile's fixed shape."""
    if form not in SLOT_FORMS[slot]:
        return [f"{slot} accepts only {', '.join(SLOT_FORMS[slot])} statements, not {form}"]
    if not isinstance(spec, dict) and form != "lean":
        return [f"{slot}: the statement must be an object"]
    if slot == "P1":
        problems = widths_required(slot, spec, {"n", *(spec.get("sizes") or {})})
        if spec.get("exponent") is None:
            problems.append("P1: the cost statement must claim its exponent in n")
        ns = spec.get("ns")
        if ns is not None and (len(set(ns)) < P1_MIN_LENGTHS or max(ns) < P1_MIN_SPAN * min(ns)):
            problems.append(f"P1: ns must hold at least {P1_MIN_LENGTHS} distinct lengths spanning a factor of "
                            f"{P1_MIN_SPAN} (or be left out for the plan's grid 64 to 4096)")
        return problems
    if slot in P2_SLOTS:
        problems = []
        task = spec.get("task")
        if task not in P2_SLOTS[slot]:
            return [f"{slot}: task must be {' or '.join(P2_SLOTS[slot])}, not {task}"]
        missing = [s for s in REQUIRED_SIZES[task] if s not in (spec.get("sizes") or [])]
        if missing:
            problems.append(f"{slot}: sizes must include {list(REQUIRED_SIZES[task])}; missing {missing}")
        if spec.get("threshold", P2_THRESHOLD) < P2_THRESHOLD:
            problems.append(f"{slot}: threshold must be at least {P2_THRESHOLD}")
        if not spec.get("state_words"):
            problems.append(f"{slot}: state_words (the construction's peak state words as a formula in "
                            f"{constructions.SIZE_NAME[task]}) is required")
        return problems + widths_required(slot, spec, {constructions.SIZE_NAME[task]})
    if slot in P4_SLOTS:
        problems = []
        task = spec.get("task")
        if task not in P4_SLOTS[slot]:
            return [f"{slot}: task must be {' or '.join(P4_SLOTS[slot])}, not {task}"]
        size = spec.get("size")
        if _int(size) and size > SOLVABLE_MAX[task]:
            problems.append(f"{slot}: the solvable case has {constructions.SIZE_NAME[task]} at most "
                            f"{SOLVABLE_MAX[task]}, not {size}")
        seeds = spec.get("seeds") or []
        if len(set(seeds)) < P4_SEEDS:
            problems.append(f"{slot}: at least {P4_SEEDS} distinct seeds are required")
        if spec.get("threshold", P4_THRESHOLD) < P4_THRESHOLD:
            problems.append(f"{slot}: threshold must be at least {P4_THRESHOLD}")
        return problems + widths_required(slot, spec, {constructions.SIZE_NAME[task]})
    if slot == "P5":
        return [f"P5: {p}" for entry in spec.get("sample") or [] for p in hypothesis_class_problems(entry)]
    return []


def widths_required(slot: str, spec: dict, names) -> list[str]:
    """P1, P2 and P4 statements declare their widths under the shared architecture contract (FIX-4)."""
    if spec.get("widths") is None:
        return [f"{slot}: widths (d and rounds as formulas in {', '.join(sorted(names))}, optionally positions) is "
                "required: the architecture contract checks the code against them"]
    return [f"{slot}: {p}" for p in architecture.widths_problems(spec["widths"], names)]


HYPOTHESIS_FIELDS = ("definition", "size_justification", "log2_size")


def hypothesis_class_problems(entry) -> list[str]:
    """A sample entry's ``hypothesis_class``: the target-rule class it counts, why the class has that size, and log2
    of the size, which must equal the entry's log2_H. The class is the family of rules the task could be (for
    example every readout table from 4 symbols to 4 answers, 4**4 members); it is not the set of every function the
    mechanism can represent. k1's smoke used log2_H = 8 with no class defined (k1 stage C diagnosis, C9 and C11)."""
    if not isinstance(entry, dict):
        return ["a sample entry must be an object"]
    found = entry.get("hypothesis_class")
    if not isinstance(found, dict) or set(found) != set(HYPOTHESIS_FIELDS):
        return [f"sample entry with log2_H={entry.get('log2_H')} needs hypothesis_class = {{definition, "
                "size_justification, log2_size}}: the target-rule class and why it has that many members"]
    problems = [f"hypothesis_class.{key} must be non-empty text" for key in ("definition", "size_justification")
                if not isinstance(found[key], str) or len(found[key].strip()) < 10]
    log2_size, log2_h = found["log2_size"], entry.get("log2_H")
    if isinstance(log2_size, bool) or not isinstance(log2_size, (int, float)) or not isinstance(log2_h, (int, float)) \
            or abs(log2_size - log2_h) > 1e-9:
        problems.append(f"hypothesis_class.log2_size ({log2_size}) must equal log2_H ({log2_h})")
    return problems


def undefined_hypothesis_classes(p5_specs: list[dict]) -> list[str]:
    """For a frozen record: why the P5 sample entries do not define their target-rule class (empty when every entry
    does). Stage C raises this as one finding on P5."""
    return [p for spec in p5_specs for entry in spec.get("sample") or [] for p in hypothesis_class_problems(entry)]


def record_problems(obligations: list[dict], limits: list[dict], shared=None, *, p3_mechanism: dict | None = None
                    ) -> list[str]:
    """Cross-slot problems with the statement fixer's profile obligations (parsed output: statements as
    {"form": ..., <form>: spec}). Every slot covered; P2.composition with both variants; P5 with an entry at every
    P2 size (p5_coverage_problems) and every P4 example count; at least one P3 limit; the shared architecture contract,
    whose hidden width is a formula in d (it is evaluated at every P1, P2 and P4 point, and only d is common to all of
    them). FIX-19 (decision M): with ``p3_mechanism`` (the card's side, math_contracts.p3_mechanism) a P3 statement is
    stated over the card's shared mechanism definitions (p3_mechanism_problems)."""
    problems = architecture.shared_problems(shared, set())
    if p3_mechanism is not None:
        problems += p3_mechanism_problems(obligations, p3_mechanism)
    by_slot: dict[str, list[dict]] = {}
    for obligation in obligations:
        if is_slot(obligation.get("claim_id")):
            by_slot.setdefault(obligation["claim_id"], []).extend(
                {"form": s["form"], "spec": s.get(s["form"])} for s in obligation["statements"])
    missing = [slot for slot in SLOTS if slot not in by_slot]
    if missing:
        problems.append(f"profile slots with no obligation: {missing} (every slot must be stated; an unstated slot "
                        "leaves the card unproven)")
    composition_tasks = {s["spec"].get("task") for s in by_slot.get("P2.composition", [])
                         if s["form"] == "construction" and isinstance(s["spec"], dict)}
    if "P2.composition" in by_slot and not {"composition", "binding"} <= composition_tasks:
        problems.append("P2.composition needs a composition construction and a binding construction")
    p5 = [s["spec"] for s in by_slot.get("P5", []) if s["form"] == "conservation" and isinstance(s["spec"], dict)]
    if p5:
        problems += p5_coverage_problems(p5, [s["spec"] for slot in P2_SLOTS for s in by_slot.get(slot, [])
                                              if s["form"] == "construction" and isinstance(s["spec"], dict)])
    p4_examples = {s["spec"].get("examples") for slot in P4_SLOTS for s in by_slot.get(slot, [])
                   if s["form"] == "learnability" and isinstance(s["spec"], dict)}
    p5_examples = {entry.get("examples") for spec in p5 for entry in spec.get("sample") or []}
    if p5 and not p4_examples <= p5_examples:
        problems.append(f"P5: every P4 example count needs a sample entry; missing m = "
                        f"{sorted(m for m in p4_examples - p5_examples if m is not None)}")
    if not limits:
        problems.append("P3: state at least one limit (task, size, resources, why)")
    return problems


LEAN_NAME = re.compile(r"[^\W\d][\w'.]*")
LEAN_NUMBER = re.compile(r"(?<![\w.])\d+(?![\w])")


def name_words(name: str) -> set[str]:
    """The words of an identifier: split at dots, underscores and lower-to-upper case changes (``EmbedDomain`` gives
    Embed and Domain, ``h0_t`` gives h0 and t)."""
    words = set()
    for part in re.split(r"[._']", name):
        words.update(w for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+|[A-Z]", part) if w)
    return words


def uses_name(text: str, names, numbers) -> bool:
    """Whether Lean source (comments and strings removed) uses one of ``names`` as an identifier or a word of one,
    or one of ``numbers`` as a numeral. A one-letter name is matched with its case (P is the position table, p the
    retrieval query); a longer one in any case."""
    code = lean_check.strip_comments_and_strings(text)
    found = set()
    for identifier in LEAN_NAME.findall(code):
        found.add(identifier)
        found.update(name_words(identifier))
    exact = {w for w in found if len(w) == 1}
    folded = {w.lower() for w in found if len(w) > 1}
    for name in names:
        if (name in exact) if len(name) == 1 else (name.lower() in folded or name in found):
            return True
    return any(int(n) in numbers for n in LEAN_NUMBER.findall(code))


def p3_mechanism_problems(obligations: list[dict], mechanism: dict) -> list[str]:
    """FIX-19 (DECISIONS.md M): a P3 Lean statement is stated over the card's shared mechanism definitions, for the
    part (update, state or readout) its limit names. Two structural tests on each P3 Lean statement of the fixer's
    answer: its obligation cites a shared mechanism definition (kind ``mechanism``, scope shared,
    ``mechanism["definitions"]``), and the statement uses a name of the mechanism part the limits name
    (``mechanism["names"]``: the state's names, or an update or readout equation's name and operands) or, for a state
    limit, the state's stated capacity (``mechanism["numbers"]``: a number of the limit's resources, as a counting
    statement over the state space writes it). In calibration 20261016T051933Z K1 run 2 the frozen P3 statement was a
    list-indexing fact over a ``table`` of rows that names no part of the embed update h0_t = E[x_t] + P[t] the
    card's limit names; stage C blocked it. Statements of other forms are left to the statement match."""
    problems = []
    wanted = ", ".join(sorted(mechanism["names"])) or "none"
    for obligation in obligations:
        if obligation.get("claim_id") != "P3":
            continue
        oid = obligation.get("id")
        cites = set(obligation.get("definitions") or []) & set(mechanism["definitions"])
        for index, statement in enumerate(obligation.get("statements") or []):
            text = statement.get("lean") if statement.get("form") == "lean" else None
            if not isinstance(text, str):
                continue
            label = f"P3 ({oid} statement {index + 1})"
            if not cites:
                problems.append(
                    f"{label}: the obligation cites no shared mechanism definition (kind mechanism, scope shared; "
                    f"this answer has {', '.join(mechanism['definitions']) or 'none'}). A P3 statement is stated "
                    "over the card's mechanism: cite the definition that restates the update, state or readout the "
                    "limit names")
            elif not uses_name(text, mechanism["names"], mechanism["numbers"]):
                capacity = (", or the state's stated capacity ("
                            + ", ".join(str(n) for n in sorted(mechanism["numbers"])) + ")"
                            if mechanism["numbers"] else "")
                problems.append(
                    f"{label}: the Lean statement uses none of the names of the card's mechanism part the limit "
                    f"names ({wanted}){capacity}. A statement that never mentions the update, state or readout the "
                    "limit is about formalizes a different fact (DECISIONS.md M): state it over that part, with the "
                    "names the card's mechanism and its shared mechanism definition give it")
    return problems


def p5_sizes(constructions_specs: list[dict]) -> dict[str, list[int]]:
    """Per P5 task, every size P2 runs: the required sizes and any size a P2 construction statement adds."""
    found = {task: set(REQUIRED_SIZES[task]) for task in P5_SECTIONS}
    for spec in constructions_specs:
        if spec.get("task") in found:
            found[spec["task"]].update(s for s in spec.get("sizes") or [] if _int(s))
    return {task: sorted(sizes) for task, sizes in found.items()}


def p5_parameters(constructions_specs: list[dict]) -> dict[str, set]:
    """The task parameter a P5 entry must share with the construction it accounts for: copy D and recall V, from the
    P2 statements' params (the generator's defaults when a statement gives none)."""
    found: dict[str, set] = {}
    for spec in constructions_specs:
        task = spec.get("task")
        if task in P5_PARAMETER:
            name = P5_PARAMETER[task]
            params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
            found.setdefault(task, set()).add(params.get(name, constructions.DEFAULTS[task][name]))
    return found


# FIX-8: the task parameter, besides the size, that sets the information a copy or recall entry accounts for.
P5_PARAMETER = {"copy": "D", "recall": "V"}


def p5_coverage_problems(p5_specs: list[dict], construction_specs: list[dict]) -> list[str]:
    """FIX-8: before freezing, P5 needs an entry at every size P2 runs, for copy (L), recall (K) and dyck (depth),
    with copy's D and recall's V equal to those of the P2 construction. In the 2026-10-14 calibration card k3 froze one
    dyck entry at depth 16 (accuracy 0.000244140625); its depth-4 construction met the threshold and the consistency
    check failed P5 after the run, because no entry accounted for depth 4 (results/phase4-20261013T073748Z/
    k3-p5-diag/REPORT.md). An entry at a larger size does not stand in for a smaller one."""
    problems = []
    parameters = p5_parameters(construction_specs)
    for task, sizes in p5_sizes(construction_specs).items():
        section, field = P5_SECTIONS[task]
        entries = [entry for spec in p5_specs for entry in spec.get(section) or [] if isinstance(entry, dict)]
        missing = [size for size in sizes if not any(entry.get(field) == size for entry in entries)]
        if missing:
            problems.append(f"P5: the conservation numbers need a {section} entry at every P2 size, {field} in "
                            f"{sizes}, each with the accuracy the card claims there; missing {field} = {missing} (an "
                            "entry at a larger size does not account for a smaller one)")
        name = P5_PARAMETER.get(task)
        if name and parameters.get(task):
            wanted = parameters[task]
            unmatched = [size for size in sizes if size not in missing and not any(
                entry.get(field) == size and entry.get(name) in wanted for entry in entries)]
            if unmatched:
                problems.append(f"P5: the {section} entries at {field} = {unmatched} must use the P2 construction's "
                                f"{name} ({', '.join(map(str, sorted(wanted)))}), the {name} its generator runs")
    return problems


# FIX-11: how a P5 copy, recall or dyck entry states its accuracy. In the 2026-10-14 calibration (runs/
# calibration-math-20261014T204729Z, k3-run1) the fixer froze P5 statement O16.a with recall entries at K = 8 and 32
# of accuracy 0.99 on 4 state bits, the P2 threshold, where the K3 card claims recall fails at K = 8 to 64; the
# recall law needs 31.0 and 124.2 bits for those, so O16.a was violated at M and K3's P5 cell failed. An entry now
# gives either "ceiling" (the host fills in the largest accuracy the law allows the entry's state bits at its size)
# or a number with the id of the card claim that states it.
CEILING = "ceiling"
# The JSON spellings the problem messages show the fixer.
CEILING_JSON, CLAIM_JSON, ACCURACY_JSON = f'"{CEILING}"', '"claim"', '"accuracy"'


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def resolve_p5_accuracies(spec, claim_ids) -> tuple[dict, list[str]]:
    """FIX-11, before freezing a P5 conservation statement: every copy, recall and dyck entry gives ``"accuracy":
    "ceiling"`` or a numeric accuracy with ``"claim"``, the id of one of the record's card claims (``claim_ids``
    without the profile slots) that states that accuracy at that size. Returns the statement with each ceiling
    replaced by the information ceiling of the entry's state bits at its size (conservation.CEILINGS, the same laws
    the check runs, so a ceiling entry discharges) and marked ``"ceiling": true``, and the problems: a numeric
    accuracy without a claim id or with an id that is not a card claim, an accuracy that is neither, or a ceiling
    entry without the numbers its law needs. A cited number is frozen as written: whether the claim supports it is
    stage C's statement match, and a number the laws forbid is violated at M."""
    if not isinstance(spec, dict):
        return spec, []
    known = [cid for cid in claim_ids if not is_slot(cid)]
    listed = ", ".join(known) or "none"
    resolved, problems = dict(spec), []
    default_bits = spec.get("state_bits")
    for task, (section, field) in P5_SECTIONS.items():
        entries = spec.get(section)
        if not isinstance(entries, list):
            continue
        resolved[section] = []
        for entry in entries:
            if not isinstance(entry, dict):
                resolved[section].append(entry)
                continue
            where = f"P5: the {section} entry at {field} = {entry.get(field)}"
            accuracy = entry.get("accuracy")
            if accuracy == CEILING:
                bits = entry.get("state_bits", default_bits)
                needs = [key for key in conservation.CLAIM_KEYS[section] if key != "accuracy"
                         and not _number(entry.get(key))]
                if not _number(bits):
                    needs.append("state_bits (on the entry or the statement)")
                if needs:
                    problems.append(f"{where}: an accuracy of {CEILING_JSON} is computed from the entry's numbers; "
                                    f"it lacks {', '.join(needs)}")
                    resolved[section].append(entry)
                    continue
                try:
                    ceiling = conservation.CEILINGS[section](bits, entry)
                except ValueError as exc:
                    problems.append(f"{where}: the ceiling cannot be computed: {exc}")
                    resolved[section].append(entry)
                    continue
                resolved[section].append(dict(entry, accuracy=ceiling, ceiling=True))
                continue
            if not _number(accuracy):
                problems.append(f"{where}: the accuracy must be {CEILING_JSON} or a number with {CLAIM_JSON}, the "
                                f"card claim that states it, not {accuracy!r}")
            elif entry.get("claim") is None:
                problems.append(
                    f"{where} gives accuracy {accuracy} with no {CLAIM_JSON}. A numeric accuracy must name, in "
                    f"{CLAIM_JSON}, the card claim that states that accuracy at that size (card claims: {listed}). "
                    "When the card states no accuracy at that size, or says only that the task fails there, write "
                    f"{{{ACCURACY_JSON}: {CEILING_JSON}}}: the host fills in the largest accuracy the conservation "
                    f"law allows the entry's state_bits at that size. Never write the P2 threshold {P2_THRESHOLD} "
                    "where the card does not claim it.")
            elif entry["claim"] not in known:
                problems.append(f"{where} cites claim {entry['claim']!r}, which is not one of the card's claims "
                                f"({listed}); a profile slot is not a card claim. Cite the card claim that states "
                                f"this accuracy, or write {ACCURACY_JSON}: {CEILING_JSON}")
            resolved[section].append(entry)
    return resolved, problems


def p2_sizes(statement_checks: list[tuple[dict, dict | None]]) -> dict[str, dict[int, bool]]:
    """Per construction task, the sizes P2 ran and whether each met its threshold, from (statement, check) pairs
    whose checks carry the harness's per-size results in ``details``."""
    found: dict[str, dict[int, bool]] = {}
    for statement, check in statement_checks:
        if statement["form"] != "construction" or not check or not check.get("decided"):
            continue
        for row in (check.get("details") or {}).get("results") or []:
            sizes = found.setdefault(statement["spec"]["task"], {})
            sizes[row["size"]] = sizes.get(row["size"], False) or bool(row["passed"])
    return found


def consistency(limits: list[dict], p2: dict[str, dict[int, bool]], p5_specs: list[dict]) -> dict[str, str]:
    """Slot failures found after the checks ran: {slot: reason}."""
    failures = {}
    contradicted = []
    for limit in limits:
        task, size = limit.get("task"), limit.get("size")
        if task in p2 and _int(size):
            beaten = sorted(s for s, ok in p2[task].items() if ok and s >= size)
            if beaten:
                contradicted.append(f"the limit on {task} from size {size} is contradicted by the P2 construction, "
                                    f"which met the threshold at {beaten}")
    if contradicted:
        failures["P3"] = "; ".join(contradicted)
    gaps = []
    for task, (section, field) in P5_SECTIONS.items():
        passed = [s for s, ok in p2.get(task, {}).items() if ok]
        if not passed:
            continue
        top = max(passed)
        covered = any(entry.get(field, 0) >= top and entry.get("accuracy", 0) >= P2_THRESHOLD
                      for spec in p5_specs for entry in spec.get(section) or [])
        if not covered:
            gaps.append(f"P2 met the threshold on {task} at {field}={top}, but P5 has no {section} entry at that size "
                        f"or above with accuracy at least {P2_THRESHOLD}")
    if gaps:
        failures["P5"] = "; ".join(gaps)
    return failures


def template_text(caps) -> str:
    """The fixed shape, rendered for the statement fixer's message. ``caps`` are the runner's library caps (an object
    with ``wall_s``, ``memory_mb`` and ``cpus``, hybrid.math.runner.LIBRARY_CAPS in a run): the P4 budget is stated
    from them, never from a literal here."""
    lines = [f"PROFILE TEMPLATE ({PROFILE_VERSION}). Claim ids and what each requires:"]
    for slot in SLOTS:
        lines.append(f"- {slot} (forms: {', '.join(SLOT_FORMS[slot])}; tier reached: "
                     f"{', '.join(accepted_tiers(slot))}): {SLOT_TEXT[slot]}")
    lines.append("Required P2 sizes: " + "; ".join(f"{task} {constructions.SIZE_NAME[task]} in {list(sizes)}"
                                                  for task, sizes in REQUIRED_SIZES.items())
                 + f"; threshold at least {P2_THRESHOLD}; state_words required.")
    lines.append(P2_STORE_TEXT)
    lines.append("TASK CONTRACTS at the required P2 sizes, with the default params (a statement's own params, such as "
                 "recall V or n_keys, change the layout as the generator does):")
    lines += [constructions.task_contract(task, sizes) for task, sizes in REQUIRED_SIZES.items()]
    lines.append("P4 solvable cases: " + "; ".join(f"{task} {constructions.SIZE_NAME[task]} <= {top}"
                                                   for task, top in SOLVABLE_MAX.items())
                 + " (each P4 statement selects one size at or below its ceiling; P2, not P4, covers every listed size)"
                 + f"; at least {P4_SEEDS} distinct seeds; threshold at least {P4_THRESHOLD}. The same generators "
                 "make the training and test data; token streams at the largest solvable size: "
                 + "; ".join(f"{task} {constructions.SIZE_NAME[task]}={top}: "
                             + _length_text(constructions.stream_lengths(task, top, constructions.CONTRACT_SAMPLE, 0))
                             + f" tokens, vocab {constructions.layout(task, top)['vocab']}"
                             for task, top in SOLVABLE_MAX.items()) + ".")
    lines.append(p4_budget_text(caps))
    lines.append("P1: the lengths and the grid of the other sizes are protocol values (PROTOCOL VALUES below); the "
                 "exponent is required.")
    lines.append(cost.counting_text())
    lines.append(ARCHITECTURE_TEXT)
    lines.append(protocol_text())
    lines.append("P5: one conservation statement with an entry at every P2 size: a copy entry at each L in "
                 + str(list(REQUIRED_SIZES["copy"])) + ", a recall entry at each K in "
                 + str(list(REQUIRED_SIZES["recall"])) + " and a dyck entry at each depth in "
                 + str(list(REQUIRED_SIZES["dyck"]))
                 + ", plus an entry at any size a P2 construction adds. Each entry gives the accuracy the card claims "
                 "at that size and the resources it has there (per-entry state_bits when the state depends on the "
                 "size); copy entries use the P2 copy construction's D and recall entries its V. An entry at a larger "
                 "size does not account for a smaller one, and the host refuses a statement with a missing size "
                 "before freezing. " + P5_ACCURACY_TEXT + " Add a sample entry for every P4 example count. "
                 + HYPOTHESIS_TEXT)
    return "\n".join(lines)


def protocol_text() -> str:
    """FIX-16: the values the host sets in every profile statement, rendered for the statement fixer."""
    grid = list(PROTOCOL["cost_grid"])
    ns = ("leave ns out (the plan's lengths " + ", ".join(map(str, cost.DEFAULT_NS)) + ")"
          if PROTOCOL["p1_ns"] is None else f"ns = {list(PROTOCOL['p1_ns'])}")
    positions = "; ".join(f"{task} {protocol_positions(task) or 'left out (the longest stream at each size)'}"
                          for task in constructions.SIZE_NAME)
    return ("PROTOCOL VALUES (FIX-16). The host sets these in every P1, P2 and P4 statement; leave them out or copy "
            "them exactly, because a statement that gives any other value is refused at freeze. P2: sizes exactly "
            "the required sizes above (no added sizes), params the generator defaults ("
            + "; ".join(f"{task}: {dict(defaults) or 'none'}" for task, defaults in constructions.DEFAULTS.items())
            + f"), count {PROTOCOL['p2_count']}, threshold {P2_THRESHOLD}. P4: seeds {list(PROTOCOL['p4_seeds'])}, "
            f"test_count {PROTOCOL['p4_test_count']}, threshold {P4_THRESHOLD}, params the generator defaults. "
            f"widths.positions: {positions}; P1 n. P1: {ns}, and each size other than n takes the two values "
            f"{grid}. Conformance (FIX-20): points, tolerance and precision_bits are set by the host: one point per "
            f"combination of {grid} over the cost statement's sizes, each at n = {PROTOCOL['conformance_n']}, "
            f"tolerance {PROTOCOL['conformance_tolerance']:g} (exact: the reference and your equations both run at "
            "the card's precision_bits, so they agree exactly when they compute the same operations in the same "
            "order), precision_bits the card's resources.architecture.precision_bits.")


def conformance_points(cost_sizes) -> list[dict]:
    """FIX-20: the protocol points of a conformance audit for the card's cost size names (resources.cost.sizes):
    every combination of the cost grid's values, each at PROTOCOL["conformance_n"]."""
    names = list(cost_sizes)
    grid = PROTOCOL["cost_grid"]
    combos = [[]]
    for _ in names:
        combos = [c + [v] for c in combos for v in grid]
    return [dict(zip(names, combo), n=PROTOCOL["conformance_n"]) for combo in combos]


P5_ACCURACY_TEXT = (
    "P5 ACCURACIES (FIX-11). Each copy, recall and dyck entry states its accuracy in one of two ways. (1) "
    "\"accuracy\": \"ceiling\": the host replaces it, before freezing, with the largest accuracy the conservation "
    "law allows the entry's state_bits at that size (copy min(1, 2^S / D^L), dyck min(1, 2^S / 2^depth), recall the "
    "accuracy at which K (log2 V - h2(e) - e log2(V - 1)) falls to S, never below 1/V). Use it when the card states "
    "no accuracy at that size, or says only that the task fails there or is exact there. (2) A number with "
    "\"claim\": the id of the card claim (from CLAIMS TO FORMALIZE, not a profile slot) that states that accuracy at "
    "that size, for example {\"K\": 64, \"V\": 16, \"accuracy\": 0.0625, \"state_bits\": 4, \"claim\": "
    "\"capability.4\"} for a card that claims chance accuracy 1/16 at K = 64. The host refuses a number without a "
    "claim id, or with an id that is not a card claim, before freezing. Never write the P2 threshold 0.99 where the "
    "card claims failure: P5 states what the card claims, and P2 measures what a construction does. A cited number "
    "the laws forbid is frozen and fails P5, and stage C checks that the cited claim states it.")


ARCHITECTURE_TEXT = (
    "ARCHITECTURE CONTRACT (shared by P1, P2 and P4). State `architecture` = {\"hidden\": formula, "
    "\"precision_bits\": 16, 32 or 64}: the feed-forward hidden width H as a formula in d "
    "(\"d\" when the card's feed-forward matrices are square, \"4*d\" for a 4x expansion; only what the card states), "
    "and the float width of stored numbers. Every P1, P2 and P4 statement states `widths` = {\"d\": formula, "
    "\"rounds\": formula, \"positions\": optional formula} in its own size names (n and the cost sizes for P1, the "
    "task size for P2 and P4): the model width, the number of rounds or layers, and the most positions a table "
    "holds (default: n for P1, the longest token stream of the task for P2 and P4). The checks survey every array the "
    "code holds: no axis may exceed max(d, H, rounds, positions, vocab) at that size, P2 witnesses store floats at "
    "no more than precision_bits, and the P1 reference must hold an array of width H when H differs from d. A "
    "witness that needs a wider hidden layer than the cost reference counts breaks the contract; state the width the "
    "construction needs, and count it in P1. CARRIED-STATE BUDGET (optional `state_bits`, a number or a formula in "
    "d): give it when the card's carried inference state is a finite discrete set (a state index, a finite automaton, "
    "a fixed-width code), as the bits of the whole carried state the card's mechanism defines, for example 4 for a "
    "sixteen-state machine; leave it out for continuous state. It is not precision_bits: float64 tables do not give "
    "the carried state 64 bits. With state_bits, every P2 construction's carried state (what init and update return) "
    "must be integers at initialization and after every token: one index z with 0 <= z < 2**state_bits, or "
    "several integers that pack into state_bits bits together, with at most 2**state_bits distinct configurations "
    "at each size. In every construction, module-level data is fixed once init returns: a write to it after init is "
    "hidden storage. A size that breaks any resource check fails, whatever its accuracy.")
HYPOTHESIS_TEXT = (
    "Every sample entry defines its target-rule class: hypothesis_class = {\"definition\": the family of rules the "
    "task's answer could follow, \"size_justification\": why the family has that many members (a count or a "
    "derivation, with the prior bits the mechanism's built-in structure supplies), \"log2_size\": equal to log2_H}. "
    "The class is the family of target rules, not every function the mechanism can represent.")


def _length_text(lengths: tuple[int, int]) -> str:
    low, high = lengths
    return str(low) if low == high else f"{low} to {high}"


P2_STORE_TEXT = ("P2 capacities: a construction that keeps rows per input token (stored keys and values, a buffer of "
                 "seen tokens) declares token_store = {\"arrays\": a, \"tokens\": formula, \"width\": formula} in "
                 "the size: a arrays of `tokens` rows of `width` words. Before freezing, the host evaluates `tokens` "
                 "at every listed size against the real stream lengths of the TASK CONTRACTS below (the instances "
                 "the check runs) and a x real tokens x width against state_words, and returns a mismatch as a "
                 "problem to fix. Count every token the harness feeds: separators, both tables and the queries. When "
                 "a construction needs resources that grow with the size (rounds, layers, width), state them as "
                 "formulas in the size; do not freeze a fixed count the construction cannot meet at the larger sizes.")


def p4_budget_text(caps) -> str:
    """What a P4 check may spend, and who chooses the training recipe. In the 2026-10-13 calibration the fixer froze
    1000 full-batch epochs on m=3000 for card k1 (definition D7) with no budget in view and the check ran out of its
    1800 s cap; in the smoke that followed it froze width 16, learning rate 0.01 and 50 epochs, which fit the budget
    but did not learn (seed-0 loss 3.000009 to 2.987527), and the prover could not change it. Decision FIX-3: the
    statement fixes the model class, m, seeds, threshold, controls and budget; the prover chooses the recipe."""
    per_training = learnability.PROBE_FIT * caps.wall_s / (learnability.TRAININGS_ON_M * P4_SEEDS)
    return (learnability.budget_text(caps.wall_s, caps.memory_mb, caps.cpus, P4_SEEDS)
            + " P4 TRAINING RECIPE. The frozen learnability statement and its definitions fix the model class (the "
            "card's architecture and learning rule, with the widths and sizes the card states; when the card leaves "
            "a width open, a definition may give the range the prover chooses from), the example count m, the "
            f"seeds, the threshold {P4_THRESHOLD}, both data controls and this budget. They do not fix the training "
            "recipe: initialization, optimizer, learning rate, epochs or passes and batch size are chosen by the "
            "prover, within the card's learning rule and this budget, and declared in the learner's RECIPE; the "
            "harness records it and applies the same learn and the same RECIPE to the true data and to both controls, "
            "so a recipe that succeeds without the data still fails. Do not freeze optimizer settings, learning "
            "rates, epoch counts or initialization scales in a definition unless the card itself states them as "
            "part of its learning rule: an untested frozen schedule that cannot learn leaves the slot failed with no "
            f"repair. The prover's recipe must fit about {per_training:.0f} s per training on m examples on one core. "
            f"Never fit the budget by lowering m below the card's count, using fewer than {P4_SEEDS} seeds, lowering "
            f"the threshold below {P4_THRESHOLD} or dropping a control: those are fixed.")
