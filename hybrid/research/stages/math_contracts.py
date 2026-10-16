"""Output schemas of the math roles and the host checks on the statement fixer's output (stages M and C).

Roles (DISCOVERY-PIPELINE-PLAN.md sections 4.1, 5 and 16): ``statement_fixer`` writes the formal statements, the
definitions they use (conventions and protocol choices, frozen with the statements) and the premises (substantive
claims a proof relies on but does not prove); ``prover`` writes Lean proofs and the sources an executed statement
runs; ``challenger`` writes the magic-X code, verifies cited sources against the literature record, gives the
card's numbers for the conservation laws, the components the card relies on and wider bounds for Z3
counterexample searches; ``statement_match`` translates the frozen statements and definitions back to plain
language without seeing the claims, then compares them with the claims on a fixed rubric, says whether each
definition is only a convention, and whether a premise restates or implies a claim (circularity).

Every role returns JSON only. Nothing a role returns can change a frozen statement or definition: the prover
schema has no statement field, and the challenger's searches reuse the frozen property with other bounds.
"""
from __future__ import annotations

import copy
import re

from hybrid.math import constructions, cost, lean_check, learnability, profile, spec_checks
from hybrid.research.mathrecords import (BASE_FORMS, CITATION_SOURCES, DEFINITION_KINDS, FORMS, PREMISE_KINDS,
                                         SUBJECTS, TESTABLE_BY)
from hybrid.research.stages.falsify import _normal, quote_in_card
from hybrid.research.stages import card_resources

OBLIGATION_ID = r"^O[0-9]{1,2}$"
PREMISE_ID = r"^A[0-9]{1,2}$"
DEFINITION_ID = r"^D[0-9]{1,2}$"
RUBRIC = ("quantifiers", "domains", "constants", "exponents", "hypotheses", "inequality_direction")
LEAN_IMPORT = re.compile(r"^\s*import\s+(\S+)\s*$")
CLAIM_DECLARATION = re.compile(r"\b(def|abbrev|theorem|lemma|axiom|instance|opaque|structure|inductive|class)\s+"
                               r"Claim\b")
# FIX-13: where a definition applies. "shared" (also when the field is absent) binds every obligation that cites it;
# "cost_reference" is a P1 cost-reference convention (the operation sequence chosen for the exact count, such as a
# product spelling or omitting max subtraction) and binds only cost and conformance statements. Scope is never
# inferred from a definition's kind: accounting definitions also hold shared state and resource rules.
DEFINITION_SCOPES = ("shared", "cost_reference")
COST_REFERENCE_FORMS = ("cost", "conformance")

_STRING = {"type": "string", "minLength": 1}
# FIX-4: the widths a P1, P2 or P4 statement declares under the shared architecture contract (architecture.py).
WIDTHS = {"type": ["object", "null"], "additionalProperties": False, "required": ["d", "rounds"],
          "properties": {"d": {"type": "string", "minLength": 1}, "rounds": {"type": "string", "minLength": 1},
                         "positions": {"type": ["string", "null"]}}}
ARCHITECTURE = {"type": "object", "additionalProperties": False, "required": ["hidden", "precision_bits"],
                "properties": {"hidden": {"type": "string", "minLength": 1}, "precision_bits": {"enum": [16, 32, 64]},
                               # FIX-8: the carried-state budget of a finite-state mechanism (architecture.py).
                               "state_bits": {"type": ["string", "number", "null"]}}}
_BOUND = {"type": "object", "additionalProperties": False, "required": ["sort"],
          "properties": {"sort": {"enum": list(spec_checks.Z3_SORTS)}, "lo": {"type": "number"},
                         "hi": {"type": "number"}}}
FORM_SCHEMAS = {
    "lean": {"type": "string", "minLength": 20},
    "z3": {"type": "object", "additionalProperties": False, "required": ["variables", "hypotheses", "property"],
           "properties": {"variables": {"type": "object", "minProperties": 1, "additionalProperties": _BOUND},
                          "hypotheses": {"type": "array", "items": _STRING}, "property": _STRING,
                          "timeout_ms": {"type": "integer", "minimum": 1000, "maximum": 60000}}},
    "sympy": {"type": "object", "additionalProperties": False, "required": ["symbols", "lhs", "rhs", "relation"],
              "properties": {"symbols": {"type": "object", "minProperties": 1,
                                         "additionalProperties": {"enum": list(spec_checks.SYMPY_DOMAINS)}},
                             "lhs": _STRING, "rhs": _STRING, "relation": {"enum": list(spec_checks.RELATIONS)},
                             "conditions": {"type": "array", "items": _STRING}}},
    "simulation": {"type": "object", "additionalProperties": False,
                   "required": ["generator", "metric", "sizes", "count", "threshold"],
                   "properties": {"generator": _STRING, "metric": _STRING,
                                  "sizes": {"type": "array", "minItems": 1, "maxItems": spec_checks.MAX_SIZES,
                                            "items": {"type": "integer", "minimum": 0}},
                                  "count": {"type": "integer", "minimum": 1,
                                            "maximum": spec_checks.MAX_SIMULATION_COUNT},
                                  "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                  "seed": {"type": "integer"}}},
    "construction": {"type": "object", "additionalProperties": False, "required": ["task", "sizes"],
                     "properties": {"task": {"enum": sorted(constructions.SIZE_NAME)},
                                    "sizes": {"type": "array", "minItems": 1, "maxItems": spec_checks.MAX_SIZES,
                                              "items": {"type": "integer", "minimum": 1}},
                                    "params": {"type": ["object", "null"]}, "count": {"type": "integer",
                                                                                     "minimum": 1, "maximum": 500},
                                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                    # Phase 4: declared peak state words, a formula in the task's size
                                    "state_words": {"type": ["string", "null"]},
                                    # FIX-3: rows kept per input token, checked against the real streams
                                    "token_store": {"type": ["object", "null"], "additionalProperties": False,
                                                    "required": ["arrays", "tokens", "width"],
                                                    "properties": {"arrays": {"type": "integer", "minimum": 1},
                                                                   "tokens": _STRING, "width": _STRING}},
                                    "widths": WIDTHS}},
    "cost": {"type": "object", "additionalProperties": False, "required": ["ops", "state", "sizes", "exponent"],
             "properties": {"ops": _STRING, "state": _STRING,
                            "sizes": {"type": "object", "additionalProperties": {
                                "type": "array", "minItems": 2, "items": {"type": "integer", "minimum": 1}}},
                            "exponent": {"type": ["number", "null"]}, "ops_remainder": {"type": ["string", "null"]},
                            "state_remainder": {"type": ["string", "null"]},
                            # FIX-3: the sizes in which each remainder is of strictly lower order than its formula
                            "ops_remainder_lower_in": {"type": ["array", "null"], "items": {"type": "string"}},
                            "state_remainder_lower_in": {"type": ["array", "null"], "items": {"type": "string"}},
                            "ns": {"type": "array", "minItems": 3, "items": {"type": "integer", "minimum": 8}},
                            "widths": WIDTHS}},
    # Phase 4 (profile obligations P5 and P4, hybrid/math/profile.py)
    "conservation": {"type": "object", "minProperties": 1, "additionalProperties": False,
                     "properties": {"state_bits": {"type": "number", "minimum": 0},
                                    **{section: {"type": "array", "items": {"type": "object"}}
                                       for section in ("storage", "copy", "dyck", "index", "recall", "sample",
                                                       "reads")}}},
    "learnability": {"type": "object", "additionalProperties": False, "required": ["task", "size", "examples", "seeds"],
                     "properties": {"task": {"enum": sorted(constructions.SIZE_NAME)},
                                    "size": {"type": "integer", "minimum": 1},
                                    "params": {"type": ["object", "null"]},
                                    "examples": {"type": "integer", "minimum": 1,
                                                 "maximum": learnability.MAX_EXAMPLES},
                                    "seeds": {"type": "array", "minItems": 1, "maxItems": learnability.MAX_SEEDS,
                                              "items": {"type": "integer", "minimum": 0}},
                                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                    "test_count": {"type": "integer", "minimum": 1,
                                                   "maximum": learnability.MAX_TEST_COUNT},
                                    "widths": WIDTHS}},
}
# FIX-6: a cost reference run against the card's equations, frozen by the fixer (spec_checks.reference_conformance).
FORM_SCHEMAS["conformance"] = {
    "type": "object", "additionalProperties": False, "required": ["reference", "equations", "streams"],
    "properties": {"reference": {"type": "string", "pattern": spec_checks.REFERENCE_ID},
                   "equations": {"type": "string", "minLength": 20},
                   # FIX-20: points, tolerance and precision_bits are set by the host (card_resources); the fixer
                   # leaves them out, and a value that differs is a problem in its retry.
                   "points": {"type": "array", "minItems": 1, "maxItems": spec_checks.MAX_CONFORMANCE_POINTS,
                              "items": {"type": "object", "required": ["n"],
                                        "additionalProperties": {"type": "integer", "minimum": 1}}},
                   "streams": {"type": "array", "minItems": 1,
                               "items": {"enum": list(spec_checks.CONFORMANCE_STREAMS)}},
                   "tolerance": {"type": "number"},
                   "precision_bits": {"enum": sorted(cost.FLOAT_DTYPES)}}}
# FIX-17 (decision P3): what a P3 limit follows from. Only the mechanism's own state, update or readout counts; a limit
# that follows only from the output vocabulary or the task encoding is refused at freeze and returned as a problem.
LIMIT_PARTS = ("state", "update", "readout")
LIMIT_BASES = (*LIMIT_PARTS, "output_vocabulary", "task_encoding")
LIMIT_BASIS = {"type": "object", "additionalProperties": False, "required": ["part", "anchor", "quote"],
               "properties": {"part": {"enum": list(LIMIT_BASES)}, "anchor": {"type": "string"},
                              "quote": {"type": "string"}}}
LIMIT = {"type": "object", "additionalProperties": False, "required": ["task", "size", "resources", "why", "basis"],
         "properties": {"task": {"enum": [*sorted(constructions.SIZE_NAME), "other"]},
                        "size": {"type": ["integer", "null"], "minimum": 1}, "resources": _STRING, "why": _STRING,
                        "basis": LIMIT_BASIS}}


def mechanism_parts(card: dict) -> dict[str, tuple[str, list[str]]]:
    """{anchor: (part, texts)} for the card's state, update and readout (the anchors stage C's comparison uses)."""
    mechanism = card.get("mechanism") or {}
    found = {f"mechanism.state.{item['name']}": ("state", [item.get("meaning") or ""])
             for item in mechanism.get("state", [])}
    for part in ("update", "readout"):
        for eq in mechanism.get(part, []):
            found[f"mechanism.{part}.{eq['name']}"] = (part, [eq.get("plain") or "", eq.get("latex") or ""])
    return found


LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
LATEX_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*")
NUMBER_TEXT = re.compile(r"(?<![\w.])\d+(?![\w.]\d)")
# Numbers too common to name a state's capacity (exponent bases, counts of one or two).
TRIVIAL_NUMBERS = {0, 1, 2}


def equation_operands(latex: str) -> set[str]:
    """The operands of an equation's LaTeX: names outside index brackets, braces and sub- or superscripts
    (``h^0_t = E[x_t] + P[t]`` gives h, E and P). Commands such as \\arg\\max or \\tanh are not names."""
    text = LATEX_COMMAND.sub(" ", latex or "")
    out, depth, i = [], 0, 0
    while i < len(text):
        char = text[i]
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth = max(depth - 1, 0)
        elif char in "_^" and depth == 0:
            # a script: one name or digit, or a braced group (skipped by the depth count)
            i += 1
            while i < len(text) and text[i] == " ":
                i += 1
            if i < len(text) and text[i] not in "{[":
                match = LATEX_NAME.match(text, i) or re.compile(r"\d+").match(text, i)
                i = match.end() if match else i + 1
                continue
            continue
        elif depth == 0 and (char.isalpha()):
            match = LATEX_NAME.match(text, i)
            out.append(match.group())
            i = match.end()
            continue
        i += 1
    return set(out)


def p3_mechanism(card: dict, limits: list, definitions: list[dict]) -> dict:
    """FIX-19 (DECISIONS.md M): what a P3 statement must be stated over (profile.p3_mechanism_problems). The shared
    mechanism definitions of the answer (kind ``mechanism``, scope shared: FIX-7 kind, FIX-13 scope); the names of
    the mechanism part each limit's basis anchors (FIX-17): for a state, every state entry's name (and its base
    before an index, A for A_l); for an update or readout equation, its name and the operands of its LaTeX; and for a
    state limit, the numbers of its resources text other than 0, 1 and 2, the state's stated capacity (K2: 16896
    bits, K3: 16 states, K4: 64 states). A limit whose anchor is not one of the card's entries adds nothing
    (limit_problems reports it)."""
    parts = mechanism_parts(card)
    mechanism = card.get("mechanism") or {}
    names: set[str] = set()
    numbers: set[int] = set()
    for limit in limits or []:
        basis = limit.get("basis") if isinstance(limit, dict) else None
        anchor = basis.get("anchor") if isinstance(basis, dict) else None
        if anchor not in parts:
            continue
        part, _ = parts[anchor]
        if part == "state":
            for entry in mechanism.get("state", []):
                names.update({entry["name"], entry["name"].split("_")[0]})
            numbers.update(int(n) for n in NUMBER_TEXT.findall(str(limit.get("resources") or ""))
                           if int(n) not in TRIVIAL_NUMBERS)
            continue
        name = anchor.rsplit(".", 1)[1]
        equation = next(eq for eq in mechanism.get(part, []) if eq["name"] == name)
        names.add(name)
        names.update(equation_operands(equation.get("latex") or ""))
    shared = sorted(d["id"] for d in definitions or []
                    if d.get("kind") == "mechanism" and d.get("scope", "shared") == "shared")
    return {"definitions": shared, "names": sorted(names), "numbers": numbers}


def limit_problems(limits: list, card: dict) -> list[tuple[int, str]]:
    """FIX-17 (DECISIONS.md P3): (index, problem) for each P3 limit that is not about the mechanism. Decided
    structurally: the limit's ``basis`` names the part it follows from and quotes the card's text of that part. A basis
    of ``output_vocabulary`` or ``task_encoding``, an anchor that is not one of the card's state, update or readout
    entries, a part that is not the anchor's, or a quote that is not text of it, is a problem."""
    parts = mechanism_parts(card)
    found = []
    for index, limit in enumerate(limits or []):
        label = f"P3 limit {index + 1} ({limit.get('task') if isinstance(limit, dict) else limit!r})"
        basis = limit.get("basis") if isinstance(limit, dict) else None
        if not isinstance(basis, dict):
            found.append((index, f"{label}: no basis; name the part of the mechanism (state, update or readout) the "
                                 "limit follows from, with its anchor and a quote of the card's text there"))
            continue
        if basis.get("part") not in LIMIT_PARTS:
            found.append((index, f"{label}: it follows from the {str(basis.get('part')).replace('_', ' ')}, not from "
                                 "the mechanism. A limit must be about the mechanism's state, update or readout "
                                 "(DECISIONS.md P3); a limit that follows only from the output vocabulary or the task "
                                 "encoding does not count. State a limit of the mechanism itself"))
            continue
        anchor = basis.get("anchor")
        if anchor not in parts:
            found.append((index, f"{label}: basis anchor {anchor!r} is not one of the card's state, update or readout "
                                 f"entries ({', '.join(sorted(parts)) or 'none'})"))
            continue
        part, texts = parts[anchor]
        if part != basis["part"]:
            found.append((index, f"{label}: basis anchor {anchor} is part of the {part}, not the {basis['part']}"))
            continue
        needle = _normal(basis.get("quote") or "").strip(" .;:'\"")
        if not needle or not any(needle in _normal(text) for text in texts if text):
            found.append((index, f"{label}: basis quote is not text of {anchor}"))
    return found


# ----- statement fixer --------------------------------------------------------------------------------------------

MEASUREMENT_PLAN = {"type": ["object", "null"], "additionalProperties": False,
                    "required": ["measure", "data", "procedure", "accept_if"],
                    "properties": {"measure": _STRING, "data": _STRING, "procedure": _STRING, "accept_if": _STRING}}
CITATION = {"type": ["object", "null"], "additionalProperties": False, "required": ["source", "id", "quote"],
            "properties": {"source": {"enum": list(CITATION_SOURCES)}, "id": _STRING,
                           "quote": {"type": "string", "minLength": 10}}}


def statement_fixer_schema(claim_ids: list[str], max_obligations: int, *, profile_required: bool = False) -> dict:
    """The fixer's output schema. With ``profile_required`` (Phase 4) the claim ids include the profile slots, the
    profile forms are offered and the answer has a ``limits`` list (P3)."""
    forms = FORMS if profile_required else BASE_FORMS
    statement = {"type": "object", "additionalProperties": False, "required": ["form"],
                 "properties": {"form": {"enum": list(forms)}, **{f: FORM_SCHEMAS[f] for f in forms}}}
    obligation = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "claim_id", "claim_quote", "statements", "definitions", "premises", "discharges"],
        "properties": {"id": {"type": "string", "pattern": OBLIGATION_ID},
                       "claim_id": {"enum": [*claim_ids, None]},
                       "claim_quote": _STRING,
                       "statements": {"type": "array", "minItems": 1, "maxItems": 3, "items": statement},
                       "definitions": {"type": "array", "items": {"type": "string", "pattern": DEFINITION_ID}},
                       "premises": {"type": "array", "items": {"type": "string", "pattern": PREMISE_ID}},
                       "discharges": {"type": "array", "items": {"type": "string", "pattern": PREMISE_ID}}}}
    definition = {"type": "object", "additionalProperties": False, "required": ["id", "kind", "text"],
                  "properties": {"id": {"type": "string", "pattern": DEFINITION_ID},
                                 "kind": {"enum": list(DEFINITION_KINDS)}, "text": _STRING,
                                 # FIX-13: a choice made only for P1's exact count binds only the cost reference.
                                 "scope": {"enum": list(DEFINITION_SCOPES)}}}
    premise = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "text", "kind", "subject", "does_work", "why_needed", "testable_by", "measurement_plan",
                     "citation"],
        "properties": {"id": {"type": "string", "pattern": PREMISE_ID}, "text": _STRING,
                       "kind": {"enum": list(PREMISE_KINDS)}, "subject": {"enum": list(SUBJECTS)},
                       "does_work": {"type": "boolean"},
                       "why_needed": _STRING, "testable_by": {"enum": list(TESTABLE_BY)},
                       "measurement_plan": MEASUREMENT_PLAN, "citation": CITATION}}
    schema = {"type": "object", "additionalProperties": False,
              "required": ["obligations", "definitions", "premises", "not_formalized"],
              "properties": {"obligations": {"type": "array", "minItems": 1, "maxItems": max_obligations,
                                             "items": obligation},
                             "definitions": {"type": "array", "maxItems": 20, "items": definition},
                             "premises": {"type": "array", "maxItems": 20, "items": premise},
                             "not_formalized": {"type": "array", "items": {
                                 "type": "object", "additionalProperties": False, "required": ["claim_id", "why"],
                                 "properties": {"claim_id": {"enum": list(claim_ids)}, "why": _STRING}}}}}
    if profile_required:
        schema["required"] += ["limits", "architecture"]
        schema["properties"]["limits"] = {"type": "array", "maxItems": 6, "items": LIMIT}
        schema["properties"]["architecture"] = ARCHITECTURE
    return schema


def quote_ok(obligation: dict, card: dict, claims: dict[str, str]) -> bool:
    """An obligation's claim_quote is text of what it formalizes: the card's claims, or, for a profile slot, the
    slot's template text."""
    if profile.is_slot(obligation["claim_id"]):
        needle = _normal(obligation["claim_quote"]).strip(" .;:'\"")
        return bool(needle) and needle in _normal(claims.get(obligation["claim_id"], ""))
    return quote_in_card(obligation["claim_quote"], card)


def slot_statement_problems(obligation: dict, statement: dict) -> list[str]:
    """The profile's fixed shape for one statement of a slot obligation (empty for other obligations)."""
    if not profile.is_slot(obligation["claim_id"]):
        return []
    form = statement["form"]
    return profile.statement_problems(obligation["claim_id"], form, statement.get(form))


def resolved_statement(obligation: dict, statement: dict, claim_ids) -> tuple[dict, list[str]]:
    """FIX-11: a P5 conservation statement with each "ceiling" accuracy replaced by the information ceiling at the
    entry's size, and the problems with how its copy, recall and dyck entries state their accuracy
    (profile.resolve_p5_accuracies). Any other statement is returned as written with no problems."""
    if obligation["claim_id"] != "P5" or statement.get("form") != "conservation":
        return statement, []
    spec, problems = profile.resolve_p5_accuracies(statement.get("conservation"), claim_ids)
    return dict(statement, conservation=spec), problems


def premise_problems(premise: dict) -> list[str]:
    """A premise must say how it can be settled: code with a measurement plan, or a citation."""
    pid = premise["id"]
    if premise["testable_by"] == "code" and premise["measurement_plan"] is None:
        return [f"premise {pid}: testable_by code needs a measurement_plan"]
    if premise["testable_by"] == "citation" and premise["citation"] is None:
        return [f"premise {pid}: testable_by citation needs a citation"]
    return []


# ----- where a premise is settled (FIX-6) ----------------------------------------------------------------------------

DEFINITION_NAME = re.compile(r"\bD[0-9]{1,2}\b")


def named_definitions(premise: dict, definition_ids) -> list[str]:
    """The record's frozen definitions a premise names in what it claims or measures (its text and measurement plan;
    why_needed only says where it is used)."""
    plan = premise.get("measurement_plan") or {}
    text = " ".join([premise.get("text") or "", *(str(v) for v in plan.values())])
    known = set(definition_ids)
    return sorted({name for name in DEFINITION_NAME.findall(text) if name in known}, key=lambda d: int(d[1:]))


def settle_at(premise: dict, definition_ids) -> str:
    """``M`` for a premise about the card's own frozen definitions (by its subject, or because it names one of them),
    else ``deferrable`` (mathrecords, Where a premise is settled)."""
    if premise.get("subject") == "frozen_definitions" or named_definitions(premise, definition_ids):
        return "M"
    return "deferrable"


def dischargers(premise_id: str, obligations: list[dict]) -> list[str]:
    """The obligations that settle a premise at stage M: claim_id null, the premise in ``discharges`` and not among
    their own premises, at least one statement. A profile slot or card claim that also lists the premise does not
    count: its statement is compared with its own claim, so a count check could pass as equation conformance."""
    return [o["id"] for o in obligations if o["claim_id"] is None and premise_id in o["discharges"]
            and premise_id not in o["premises"] and o["statements"]]


def settlement_problems(premises: list[dict], obligations: list[dict], definition_ids) -> list[str]:
    """Problems with premises about the card's own frozen definitions: each needs an obligation that settles it at
    stage M, cannot be settled by a citation, and is not a world premise."""
    problems = []
    for premise in premises:
        pid, names = premise["id"], named_definitions(premise, definition_ids)
        if settle_at(premise, definition_ids) != "M":
            continue
        named = f" (it names {', '.join(names)})" if names else ""
        if premise.get("subject") in ("world", "mechanism") and names:
            problems.append(f"premise {pid}: subject {premise['subject']} but it names the frozen definitions "
                            f"{', '.join(names)}; a premise about the card's own definitions has subject "
                            "frozen_definitions and is settled at stage M")
        if premise["testable_by"] == "citation":
            problems.append(f"premise {pid} is about the card's own frozen definitions{named}; a citation cannot "
                            "settle it: make it testable by code and discharge it at stage M")
        if not dischargers(pid, obligations):
            problems.append(
                f"premise {pid} is about the card's own frozen definitions{named}, and no obligation settles it at "
                "stage M. Calibration runs only stages M and C, so a measurement plan for test design would never "
                "run. Add an obligation with claim_id null and discharges [\"" + pid + "\"] whose statement checks it "
                "(for the cost reference executing the card's equations and buffers: a conformance statement whose "
                "reference is the P1 cost statement). Without one the host refuses the premise at freeze and every "
                "claim that uses it stays unproven.")
    return problems


def conformance_problems(obligations: list[dict]) -> list[str]:
    """A conformance statement must name a cost statement of the same answer (ids as the host assigns them:
    obligation id, then a, b, c by position)."""
    forms = {f"{o['id']}.{'abc'[i]}": s["form"] for o in obligations for i, s in enumerate(o["statements"][:3])}
    problems = []
    for obligation in obligations:
        for index, statement in enumerate(obligation["statements"]):
            spec = statement.get("conformance") if statement.get("form") == "conformance" else None
            if isinstance(spec, dict) and forms.get(spec.get("reference")) != "cost":
                problems.append(f"{obligation['id']} statement {index + 1}: conformance reference "
                                f"{spec.get('reference')!r} is not a cost statement of this answer")
    return problems


def lean_statement_problems(text: str) -> list[str]:
    """The Lean lint of a statement. FIX-18: also each bound name that is a Lean 4 keyword, with a rename hint (in
    calibration 200147Z K1 run 3 a bound variable named ``prefix``, the keyword of the prefix-notation command, was
    refused as a forbidden construct; the statement would not have parsed either)."""
    problems = [f"lean: {p}" for p in lean_check.scan(text, allow_imports=True)]
    problems += [f"lean: bound name {name!r} is a Lean 4 keyword, so the statement does not parse; rename the "
                 "variable (for example "
                 f"{name + '0'!r})" for name in lean_check.keyword_binders(text)]
    code = lean_check.strip_comments_and_strings(text)
    for line in code.splitlines():
        match = LEAN_IMPORT.match(line)
        if match and not match.group(1).startswith("Mathlib"):
            problems.append(f"lean: import {match.group(1)} (only Mathlib imports are allowed)")
    if not re.search(r"\bdef\s+Claim\s*:\s*Prop\s*:=", code):
        problems.append("lean: the statement must define `def Claim : Prop := ...`")
    if re.search(r"\bsorry\b", code):
        problems.append("lean: the statement uses sorry")
    if re.search(r"\b(theorem|lemma)\s+claim_holds\b", code):
        problems.append("lean: the statement must not contain the proof")
    return problems


LINT_PREFIX = "lean: "


def lint_problems(problems: list[str]) -> list[str]:
    """FIX-18: the Lean lint findings (forbidden constructs, reserved words, imports) among the fixer problems. They
    are renameable, so stage M gives an answer that still carries one a repair round before freezing."""
    return [p for p in problems if LINT_PREFIX in p and ("forbidden construct" in p or "Lean 4 keyword" in p
                                                         or "set_option" in p)]


def statement_problems(statement: dict) -> list[str]:
    form = statement["form"]
    present = [name for name in FORMS if name in statement]
    if present != [form]:
        return [f"form {form} needs exactly the field {form!r}; found {present}"]
    if form == "lean":
        return lean_statement_problems(statement["lean"])
    return spec_checks.statement_problems(form, statement[form])


def normalized_answer(parsed: dict, card: dict) -> tuple[dict, dict[tuple[str, int], list[str]], list[str]]:
    """FIX-16: the fixer's answer with every protocol, card and derived value of its statements and of the shared
    architecture filled in from the card's ``resources`` and the protocol (card_resources), the problems per
    statement ((obligation id, index) to problems) and the architecture's problems. A value the fixer wrote that
    differs from the card or the protocol is a problem; one it left out is filled in. A card without resources is
    returned as written."""
    if not isinstance(card.get("resources"), dict):
        return parsed, {}, []
    answer = copy.deepcopy(parsed)
    found: dict[tuple[str, int], list[str]] = {}
    for obligation in answer["obligations"]:
        for index, statement in enumerate(obligation["statements"]):
            form = statement.get("form")
            if form not in statement:
                continue
            spec, problems = card_resources.normalize_statement(card, obligation["claim_id"], form, statement[form])
            statement[form] = spec
            if problems:
                found[(obligation["id"], index)] = problems
    arch_problems: list[str] = []
    if "limits" in answer:
        answer["architecture"], arch_problems = card_resources.normalize_architecture(card,
                                                                                       answer.get("architecture"))
    return answer, found, arch_problems


def fixer_problems(parsed: dict, card: dict, claim_ids: list[str], max_statements: int,
                   claims: dict[str, str] | None = None) -> list[str]:
    """Problems with the statement fixer's output beyond its schema. Empty means everything can be frozen.
    ``claims`` maps claim id to text; it is needed for profile slots (whose quotes come from the template)."""
    claims = claims or {}
    parsed, resource_problems, problems = normalized_answer(parsed, card)
    obligation_ids = [o["id"] for o in parsed["obligations"]]
    premise_ids = [p["id"] for p in parsed["premises"]]
    definition_ids = [d["id"] for d in parsed["definitions"]]
    for label, ids in (("obligation", obligation_ids), ("premise", premise_ids), ("definition", definition_ids)):
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            problems.append(f"repeated {label} ids {repeated}")
    for premise in parsed["premises"]:
        problems += premise_problems(premise)
    total = 0
    for obligation in parsed["obligations"]:
        oid = obligation["id"]
        for pid in obligation["premises"] + obligation["discharges"]:
            if pid not in premise_ids:
                problems.append(f"{oid}: premise {pid} is not in the premise list")
        for did in obligation["definitions"]:
            if did not in definition_ids:
                problems.append(f"{oid}: definition {did} is not in the definition list")
        if obligation["claim_id"] is None and not obligation["discharges"]:
            problems.append(f"{oid}: claim_id null needs a non-empty discharges list")
        if obligation["claim_id"] is not None and not quote_ok(obligation, card, claims):
            problems.append(f"{oid}: claim_quote is not text of the card's claims"
                            + (" (for a profile slot: of the slot's template text)"
                               if profile.is_slot(obligation["claim_id"]) else ""))
        for index, statement in enumerate(obligation["statements"]):
            total += 1
            statement, accuracy = resolved_statement(obligation, statement, claim_ids)
            problems += [f"{oid} statement {index + 1}: {p}" for p in resource_problems.get((oid, index), [])
                         + accuracy + (statement_problems(statement) if not accuracy else [])
                         + slot_statement_problems(obligation, statement)]
    if total > max_statements:
        problems.append(f"{total} statements; at most {max_statements}")
    covered = {o["claim_id"] for o in parsed["obligations"]} | {n["claim_id"] for n in parsed["not_formalized"]}
    missing = [cid for cid in claim_ids if cid not in covered]
    if missing:
        problems.append(f"claims with no obligation and no not_formalized entry: {missing}")
    if "limits" in parsed:
        problems += profile.record_problems(parsed["obligations"], parsed["limits"], parsed.get("architecture"),
                                            p3_mechanism=p3_mechanism(card, parsed["limits"], parsed["definitions"]))
        problems += [problem for _, problem in limit_problems(parsed["limits"], card)]
        # FIX-18 (decision L): the limits are the card's; a limit it does not state is refused.
        problems += [problem for _, problem in card_resources.limit_copy_problems(card, parsed["limits"])]
    problems += conformance_problems(parsed["obligations"])
    problems += settlement_problems(parsed["premises"], parsed["obligations"], definition_ids)
    problems += scope_problems(parsed["obligations"], parsed["definitions"])
    return problems


def cost_reference_definitions(definitions: list[dict]) -> set[str]:
    return {d["id"] for d in definitions if d.get("scope") == "cost_reference"}


def misplaced_references(obligation: dict, scoped: set[str]) -> list[str]:
    """FIX-13: the cost-reference definitions an obligation cites although one of its statements is not a cost or
    conformance statement (a P2 or P4 witness follows the mechanism and the numerical contract, not P1's recipe)."""
    if all(s.get("form") in COST_REFERENCE_FORMS for s in obligation["statements"]):
        return []
    return [did for did in obligation["definitions"] if did in scoped]


def scope_problems(obligations: list[dict], definitions: list[dict]) -> list[str]:
    scoped = cost_reference_definitions(definitions)
    problems = []
    for obligation in obligations:
        cited = misplaced_references(obligation, scoped)
        if cited:
            problems.append(f"{obligation['id']}: cites {', '.join(cited)}, scoped to the P1 cost reference, but has "
                            "statements that are not cost or conformance statements. A convention chosen for P1's "
                            "exact count does not bind a witness: cite the mechanism and numerical-contract "
                            "definitions instead, or give the definition scope shared if the card itself requires it "
                            "of every computation")
    return problems


def definition_hash(definition: dict) -> str:
    # The scope is hashed when it is given; a definition frozen without one keeps its earlier hash.
    scope = {"scope": definition["scope"]} if definition.get("scope") else {}
    return spec_checks.spec_hash({"kind": definition["kind"], "text": definition["text"], **scope})


def _first_of_each(items: list[dict], label: str, dropped: list[dict]) -> list[dict]:
    kept, seen = [], set()
    for item in items:
        if item["id"] in seen:
            dropped.append({f"{label}_id": item["id"], "problems": [f"repeated {label} id"]})
            continue
        seen.add(item["id"])
        kept.append(item)
    return kept


def freeze(parsed: dict, card: dict, claim_ids: list[str], max_statements: int,
           claims: dict[str, str] | None = None) -> dict:
    """Statements that pass the host checks, with ids and hashes, ready to freeze, and the definitions and premises
    they use. Returns {obligations, definitions, premises, not_formalized, dropped}. Invalid statements are dropped
    (recorded); an obligation left without statements is dropped and its claim goes to not_formalized; statements
    past ``max_statements`` are dropped; a repeated id keeps its first entry. A premise that still lacks its
    measurement plan or citation is frozen as written: it then cannot let a claim advance.

    Settlement (FIX-6): every premise records its ``subject`` and ``settle_at``. A premise about the card's own frozen
    definitions (``settle_at`` M) with no frozen obligation that discharges it at stage M is kept with ``refused``,
    the reason it cannot be settled: it never advances, so the claims that use it stay unproven, and its plan is not
    deferred to test design. A conformance statement whose reference is not a frozen cost statement is dropped.

    P5 accuracies (FIX-11): a P5 conservation statement is frozen with each "ceiling" accuracy replaced by the
    information ceiling at its size (resolved_statement); one whose copy, recall or dyck entry gives a number without
    a card claim id, or with an id that is not a card claim, is dropped.

    Definition scope (FIX-13): a definition with scope "cost_reference" stays cited only by obligations whose
    statements are all cost or conformance statements; its reference from any other obligation is removed and
    recorded in ``dropped``, so a P2 or P4 witness is never held to a choice made for P1's exact count."""
    dropped: list[dict] = []
    raw = parsed
    parsed, resource_problems, arch_problems = normalized_answer(parsed, card)
    definitions = [dict(d, sha256=definition_hash(d))
                   for d in _first_of_each(parsed["definitions"], "definition", dropped)]
    premises = [dict(p, added_by="statement_fixer") for p in _first_of_each(parsed["premises"], "premise", dropped)]
    known_premises = {p["id"] for p in premises}
    known_definitions = {d["id"] for d in definitions}
    scoped = cost_reference_definitions(definitions)
    obligations, seen = [], set()
    total = 0
    for obligation in parsed["obligations"]:
        oid = obligation["id"]
        if oid in seen:
            dropped.append({"obligation_id": oid, "problems": ["repeated obligation id"]})
            continue
        seen.add(oid)
        problems = []
        if obligation["claim_id"] is not None and not quote_ok(obligation, card, claims or {}):
            problems.append("claim_quote is not text of the card's claims")
        if obligation["claim_id"] is None and not obligation["discharges"]:
            problems.append("claim_id null needs a non-empty discharges list")
        if problems:
            dropped.append({"obligation_id": oid, "problems": problems})
            continue
        statements = []
        for index, statement in enumerate(obligation["statements"]):
            sid = f"{oid}.{'abc'[index]}"
            resolved, found = resolved_statement(obligation, statement, claim_ids)
            # FIX-16: a statement that still carries a value from neither the card nor the protocol is refused.
            found = (resource_problems.get((oid, index)) or found or statement_problems(resolved)
                     or slot_statement_problems(obligation, resolved))
            if not found and total >= max_statements:
                found = [f"over the limit of {max_statements} statements"]
            if found:
                written = _written(raw, oid, index, statement)
                dropped.append({"statement_id": sid, "statement": written, "problems": found})
                continue
            total += 1
            form = statement["form"]
            spec = copy.deepcopy(resolved[form])
            digest = lean_check.statement_hash(spec) if form == "lean" else spec_checks.spec_hash(spec)
            statements.append({"id": sid, "form": form, "spec": spec, "sha256": digest,
                               "bound": spec_checks.bound_text(spec) if form == "z3" else None})
        if statements:
            cited = list(dict.fromkeys(d for d in obligation["definitions"] if d in known_definitions))
            misplaced = misplaced_references({"statements": statements, "definitions": cited}, scoped)
            if misplaced:
                # FIX-13: the fixer scoped these to the P1 cost reference; they do not bind this obligation.
                dropped.append({"obligation_id": oid, "definitions": misplaced, "problems": [
                    f"definitions {', '.join(misplaced)} are scoped to the P1 cost reference; their references were "
                    "removed from this obligation, whose statements are not all cost or conformance statements"]})
                cited = [d for d in cited if d not in misplaced]
            obligations.append({"id": oid, "claim_id": obligation["claim_id"],
                                "claim_quote": obligation["claim_quote"],
                                "definitions": cited,
                                "premises": list(dict.fromkeys(p for p in obligation["premises"]
                                                               if p in known_premises)),
                                "discharges": list(dict.fromkeys(p for p in obligation["discharges"]
                                                                 if p in known_premises)),
                                "statements": statements})
    obligations = _drop_unreferenced_conformance(obligations, dropped)
    known_ids = [d["id"] for d in definitions]
    for premise in premises:
        if named_definitions(premise, known_ids):
            premise["subject"] = "frozen_definitions"
        premise["settle_at"] = settle_at(premise, known_ids)
        if premise["settle_at"] == "M" and not dischargers(premise["id"], obligations):
            premise["refused"] = ("about the card's own frozen definitions, and no frozen obligation discharges it "
                                  "at stage M; calibration runs only M and C, so its measurement plan would never "
                                  "run. The claims that use it cannot be shown.")
    not_formalized = list(parsed["not_formalized"])
    covered = {o["claim_id"] for o in obligations} | {n["claim_id"] for n in not_formalized}
    for cid in claim_ids:
        if cid not in covered:
            not_formalized.append({"claim_id": cid, "why": "every statement written for it was invalid or missing"})
    frozen = {"obligations": obligations, "definitions": definitions, "premises": premises,
              "not_formalized": not_formalized, "dropped": dropped}
    if "limits" in parsed:
        # FIX-17 (decision P3): a limit not about the mechanism is refused; P3's statements are dropped when every
        # limit is, since a statement of a refused limit cannot instantiate the slot.
        refused = limit_problems(parsed["limits"], card)
        # FIX-18 (decision L): a limit the card does not state is invented and refused as well.
        invented = card_resources.invented_limits(card, parsed["limits"])
        refused = refused + [(i, p) for i, p in invented if i not in {j for j, _ in refused}]
        kept = [limit for index, limit in enumerate(parsed["limits"]) if index not in {i for i, _ in refused}]
        for index, problem in refused:
            dropped.append({"limit": copy.deepcopy(parsed["limits"][index]), "problems": [problem]})
        kept = card_resources.frozen_limits(card, kept)
        if refused and not kept:
            for obligation in [o for o in obligations if o["claim_id"] == "P3"]:
                obligations.remove(obligation)
                dropped.append({"obligation": obligation["id"], "claim_id": "P3",
                                "problems": ["every P3 limit was refused at freeze (DECISIONS.md P3)"]})
            not_formalized = [n for n in not_formalized if n["claim_id"] != "P3"] + [
                {"claim_id": "P3", "why": "every stated limit follows from the output vocabulary or the task "
                                          "encoding, names no part of the mechanism, or is not one of the card's "
                                          "stated limits, and was refused at freeze (DECISIONS.md P3 and L)"}]
            frozen["not_formalized"] = not_formalized
        frozen["profile"] = {"version": profile.PROFILE_VERSION, "slots": list(profile.SLOTS),
                             "limits": copy.deepcopy(kept)}
        if parsed.get("architecture") is not None:
            frozen["profile"]["architecture"] = copy.deepcopy(parsed["architecture"])
        if arch_problems:
            # FIX-16: the contract is frozen from the card's resources; the fixer's differing copy is refused.
            dropped.append({"architecture": raw.get("architecture"), "problems": arch_problems})
    if isinstance(card.get("resources"), dict):
        frozen["resources"] = copy.deepcopy(card["resources"])
    if card_resources.stated_limits(card) is not None:
        frozen["card_limits"] = copy.deepcopy(card["limits"])
    return frozen


def _written(raw: dict, oid: str, index: int, fallback: dict) -> dict:
    """The statement as the fixer wrote it (before the host filled in card and protocol values)."""
    for obligation in raw["obligations"]:
        if obligation["id"] == oid and index < len(obligation["statements"]):
            return obligation["statements"][index]
    return fallback


def _drop_unreferenced_conformance(obligations: list[dict], dropped: list[dict]) -> list[dict]:
    """Frozen obligations without conformance statements whose reference is not a frozen cost statement; an
    obligation left with no statement is dropped (recorded)."""
    cost_ids = {s["id"] for o in obligations for s in o["statements"] if s["form"] == "cost"}
    kept = []
    for obligation in obligations:
        statements = []
        for statement in obligation["statements"]:
            if statement["form"] == "conformance" and statement["spec"]["reference"] not in cost_ids:
                dropped.append({"statement_id": statement["id"], "statement": {"conformance": statement["spec"]},
                                "problems": [f"conformance reference {statement['spec']['reference']} is not a "
                                             "frozen cost statement"]})
                continue
            statements.append(statement)
        if statements:
            kept.append(dict(obligation, statements=statements))
        else:
            dropped.append({"obligation_id": obligation["id"], "problems": ["no statement left"]})
    return kept


# ----- prover -----------------------------------------------------------------------------------------------------

def prover_schema(statement_ids: list[str]) -> dict:
    """``recipe_change_reason`` (optional, FIX-5): why a learner changes a RECIPE that passed on every seed earlier
    in the run (prover_memory.BestAttempts.discarded_without_reason)."""
    return {"type": "object", "additionalProperties": False, "required": ["attempts"],
            "properties": {"attempts": {"type": "array", "minItems": 1, "maxItems": len(statement_ids), "items": {
                "type": "object", "additionalProperties": False,
                "required": ["statement_id", "proof", "source", "notes"],
                "properties": {"statement_id": {"enum": list(statement_ids)},
                               "proof": {"type": ["string", "null"]}, "source": {"type": ["string", "null"]},
                               "notes": {"type": "string"},
                               "recipe_change_reason": {"type": ["string", "null"]}}}}}}


def proof_redeclares_claim(proof: str) -> bool:
    return bool(CLAIM_DECLARATION.search(lean_check.strip_comments_and_strings(proof)))


# ----- challenger -------------------------------------------------------------------------------------------------

def _ids(values) -> dict:
    return {"enum": list(values)} if values else {"type": "string", "maxLength": 0}


# FIX-21 (decision MO): "measurement", a definition that belongs to a test or bound (a P5 hypothesis class, a P2 task
# generator, an evaluation set, a protocol constant), never supplied to the model at inference or training. The host
# settles it by the frozen statement that defines it (c_challenge.measurement_definition), and only if that held.
COMPONENT_ROLES = ("mechanism", "external", "measurement")
# FIX-17: "resource_contradiction", a claim of the card that its own stated resources cannot hold.
FINDING_KINDS = ("missing_assumption", "witness_violation", "accounting", "explanation", "resource_contradiction")
FINDING_TESTS = ("code", "derivation", "citation")
# FIX-17: "resources", the card's resources frozen with the statements (the JSON text the challenger is shown).
EVIDENCE_SOURCES = ("card", "check", "definition", "statement", "resources")
QUOTE_SOURCES = ("card", "definition", "statement")


def _magic_claim() -> dict:
    return {"type": ["object", "null"], "additionalProperties": False,
            "required": ["task", "size", "value", "tolerance", "examples", "params"],
            "properties": {"task": {"enum": sorted(constructions.SIZE_NAME)}, "size": {"type": "integer", "minimum": 1},
                           "value": {"type": "number", "minimum": 0, "maximum": 1},
                           "tolerance": {"type": "number", "minimum": 0, "maximum": 0.5},
                           "examples": {"type": "integer", "minimum": 1, "maximum": 100000},
                           "params": {"type": ["object", "null"]}}}


def _attached_magic_x() -> dict:
    """Magic-X code attached to a component or finding the challenger adds (FIX-4): it runs once the host has
    substantiated the dependency, like the magic-X of an original premise."""
    return {"type": ["object", "null"], "additionalProperties": False, "required": ["code", "claim", "quote", "why"],
            "properties": {"code": {"type": ["string", "null"]}, "claim": _magic_claim(), "quote": {"type": "string"},
                           "why": _STRING}}


def challenger_schema(premise_ids: list[str], obligation_ids: list[str], executed_ids: list[str],
                      z3_ids: list[str], cited_ids: list[str], *, statement_ids: list[str] = (),
                      definition_ids: list[str] = ()) -> dict:
    """The challenger's output (FIX-4). Components say whether they are part of the mechanism or supplied from
    outside it, or (FIX-21) belong to a test or bound as measurement objects; the host decides whether P1 covers
    their cost and whether a measurement object is one (it corrects the role either way). New dependencies are
    findings with an exact evidence anchor, the dependency they claim, and how they can be settled (runnable code, a
    SymPy derivation or a citation); the host settles each before it can block anything. Conservation numbers carry
    quotes from the card, a frozen definition or a frozen statement, and derivations for numbers that are not
    quoted."""
    evidence_refs = sorted({*executed_ids, *statement_ids, *definition_ids})
    citation = {"type": ["object", "null"], "additionalProperties": False, "required": ["source", "id", "quote"],
                "properties": {"source": {"enum": list(CITATION_SOURCES)}, "id": _STRING,
                               "quote": {"type": "string", "minLength": 10}}}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["magic_x", "citations", "conservation", "components", "counterexample_searches", "new_edges",
                     "findings"],
        "properties": {
            "magic_x": {"type": "array", "maxItems": len(premise_ids), "items": {
                "type": "object", "additionalProperties": False,
                "required": ["premise_id", "code", "claim", "quote", "why"],
                "properties": {"premise_id": _ids(premise_ids), "code": {"type": ["string", "null"]},
                               "claim": _magic_claim(), "quote": {"type": "string"}, "why": _STRING}}},
            "citations": {"type": "array", "maxItems": len(cited_ids), "items": {
                "type": "object", "additionalProperties": False,
                "required": ["premise_id", "supports", "record_quote", "why"],
                "properties": {"premise_id": _ids(cited_ids), "supports": {"type": "boolean"},
                               "record_quote": {"type": "string"}, "why": _STRING}}},
            "conservation": {"type": "object", "additionalProperties": False,
                             "required": ["claims", "quotes", "derivations", "why"],
                             "properties": {
                                 "claims": {"type": ["object", "null"]},
                                 "quotes": {"type": "array", "items": {
                                     "type": "object", "additionalProperties": False,
                                     "required": ["source", "ref", "text"],
                                     "properties": {"source": {"enum": list(QUOTE_SOURCES)},
                                                    "ref": {"type": ["string", "null"]}, "text": _STRING}}},
                                 "derivations": {"type": "array", "items": {
                                     "type": "object", "additionalProperties": False,
                                     "required": ["value", "expression", "inputs"],
                                     "properties": {"value": {"type": "number"}, "expression": _STRING,
                                                    "inputs": {"type": "object",
                                                               "additionalProperties": {"type": "number"}}}}},
                                 "why": _STRING}},
            "components": {"type": "array", "maxItems": 12, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "subject", "quote", "role", "kind", "premise_id", "statement_id", "function",
                             "used_by", "why", "magic_x"],
                "properties": {"name": _STRING, "subject": _STRING, "quote": _STRING,
                               "role": {"enum": list(COMPONENT_ROLES)}, "kind": {"enum": list(PREMISE_KINDS)},
                               "premise_id": {"enum": [*premise_ids, None]},
                               "statement_id": {"enum": [*executed_ids, None]},
                               "function": {"type": ["string", "null"]},
                               "used_by": {"type": "array", "items": _ids(obligation_ids)}, "why": _STRING,
                               "magic_x": _attached_magic_x()}}},
            "counterexample_searches": {"type": "array", "maxItems": len(z3_ids) * 2, "items": {
                "type": "object", "additionalProperties": False, "required": ["statement_id", "bounds", "why"],
                "properties": {"statement_id": _ids(z3_ids),
                               "bounds": {"type": "object", "additionalProperties": {
                                   "type": "object", "additionalProperties": False, "required": ["lo", "hi"],
                                   "properties": {"lo": {"type": "number"}, "hi": {"type": "number"}}}},
                               "why": _STRING}}},
            "new_edges": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["premise_id", "obligation_id", "why"],
                "properties": {"premise_id": _ids(premise_ids), "obligation_id": _ids(obligation_ids),
                               "why": _STRING}}},
            "findings": {"type": "array", "maxItems": 12, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["subject", "kind", "claim", "evidence", "dependency", "used_by", "premise_kind",
                             "does_work", "testable_by", "code", "derivation", "citation", "record_quote",
                             "measurement_plan", "magic_x", "targets"],
                "properties": {
                    "subject": _STRING, "kind": {"enum": list(FINDING_KINDS)}, "claim": _STRING,
                    "evidence": {"type": "object", "additionalProperties": False,
                                 "required": ["source", "ref", "quote"],
                                 "properties": {"source": {"enum": list(EVIDENCE_SOURCES)},
                                                "ref": {"enum": [*evidence_refs, None]}, "quote": _STRING}},
                    "dependency": _STRING,
                    "used_by": {"type": "array", "minItems": 1, "items": _ids(obligation_ids)},
                    "premise_kind": {"enum": list(PREMISE_KINDS)}, "does_work": {"type": "boolean"},
                    "testable_by": {"enum": list(FINDING_TESTS)}, "code": {"type": ["string", "null"]},
                    "derivation": {"type": ["object", "null"]}, "citation": citation,
                    "record_quote": {"type": ["string", "null"]}, "measurement_plan": MEASUREMENT_PLAN,
                    "magic_x": _attached_magic_x(), "targets": _witness_targets(executed_ids)}}},
        }}


def _witness_targets(executed_ids: list[str]) -> dict:
    """FIX-13: for a witness violation, each affected witness with the exact deviating text of its source and a
    narrow edit that removes it (hybrid/math/findings.conforming_variant). The host replays the edited source under
    the statement's own frozen check; the result decides materiality for that statement. Empty for other kinds."""
    return {"type": "array", "maxItems": 12, "items": {
        "type": "object", "additionalProperties": False, "required": ["statement_id", "deviation", "edits", "why"],
        "properties": {"statement_id": _ids(executed_ids), "deviation": _STRING,
                       "edits": {"type": "array", "maxItems": 8, "items": {
                           "type": "object", "additionalProperties": False, "required": ["anchor", "replacement"],
                           "properties": {"anchor": _STRING, "replacement": {"type": "string"}}}},
                       "why": _STRING}}}


# ----- statement match ----------------------------------------------------------------------------------------------

def backtranslate_schema(statement_ids: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["translations"],
            "properties": {"translations": {"type": "array", "minItems": len(statement_ids),
                                            "maxItems": len(statement_ids), "items": {
                "type": "object", "additionalProperties": False, "required": ["statement_id", "plain"],
                "properties": {"statement_id": _ids(statement_ids), "plain": _STRING}}}}}


# FIX-7: a mismatch objection is anchored. A statement objection names the rubric item (or the direction) and quotes
# the claim text and the formal statement or definition text that differ. A definition objection quotes the clause it
# objects to, and either the card's mechanism text the clause contradicts or what the clause asserts beyond the card.
OBJECTION_ITEMS = (*RUBRIC, "direction")
DEFINITION_VERDICTS = ("convention", "mechanism", "substantive")
DEFINITION_OBJECTION_KINDS = ("contradicts_card", "beyond_card")
# What a definition clause specifies. The card-defining parts restate the card's own mechanism: a clause that only
# restates them is part of the claim. An addition to them contradicts the card's mechanism text.
CARD_DEFINING = ("state", "update", "readout", "learning_rule")
SUBSTANTIVE_ASSERTS = ("performance", "capacity", "cost", "data", "world")
CROSSCHECK_VERDICTS = ("upheld", "rejected")


def statement_objection_schema() -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": ["item", "claim_quote", "statement_quote", "difference"],
            "properties": {"item": {"enum": list(OBJECTION_ITEMS)}, "claim_quote": {"type": "string"},
                           "statement_quote": {"type": "string"}, "difference": _STRING}}


def definition_objection_schema() -> dict:
    return {"type": ["object", "null"], "additionalProperties": False,
            "required": ["definition_quote", "card_quote", "kind", "asserts", "difference"],
            "properties": {"definition_quote": {"type": "string"}, "card_quote": {"type": ["string", "null"]},
                           "kind": {"enum": list(DEFINITION_OBJECTION_KINDS)},
                           "asserts": {"enum": [*CARD_DEFINING, *SUBSTANTIVE_ASSERTS]}, "difference": _STRING}}


def match_schema(statement_ids: list[str], definition_ids: list[str], premise_ids: list[str],
                 claim_ids: list[str]) -> dict:
    item = {"type": "object", "additionalProperties": False, "required": ["verdict", "note"],
            "properties": {"verdict": {"enum": ["same", "differs"]}, "note": {"type": "string"}}}
    return {
        "type": "object", "additionalProperties": False, "required": ["statements", "definitions", "premises"],
        "properties": {
            "statements": {"type": "array", "minItems": len(statement_ids), "maxItems": len(statement_ids), "items": {
                "type": "object", "additionalProperties": False,
                "required": ["statement_id", "rubric", "direction", "predicate", "objections", "why"],
                "properties": {"statement_id": _ids(statement_ids),
                               "rubric": {"type": "object", "additionalProperties": False, "required": list(RUBRIC),
                                          "properties": {name: item for name in RUBRIC}},
                               "direction": {"enum": ["match", "weaker", "stronger", "different"]},
                               "predicate": {"type": ["string", "null"]},
                               "objections": {"type": "array", "maxItems": 8, "items": statement_objection_schema()},
                               "why": _STRING}}},
            "definitions": {"type": "array", "minItems": len(definition_ids), "maxItems": len(definition_ids),
                            "items": {"type": "object", "additionalProperties": False,
                                      "required": ["definition_id", "verdict", "objection", "why"],
                                      "properties": {"definition_id": _ids(definition_ids),
                                                     "verdict": {"enum": list(DEFINITION_VERDICTS)},
                                                     "objection": definition_objection_schema(),
                                                     "why": _STRING}}},
            "premises": {"type": "array", "minItems": len(premise_ids), "maxItems": len(premise_ids), "items": {
                "type": "object", "additionalProperties": False, "required": ["premise_id", "implies", "why"],
                "properties": {"premise_id": _ids(premise_ids),
                               "implies": {"type": "array", "items": _ids(claim_ids)}, "why": _STRING}}},
        }}


def crosscheck_schema(objection_ids: list[str]) -> dict:
    """The cross-check judge's answer (FIX-7): one verdict per anchored objection."""
    return {"type": "object", "additionalProperties": False, "required": ["verdicts"],
            "properties": {"verdicts": {"type": "array", "minItems": len(objection_ids),
                                        "maxItems": len(objection_ids), "items": {
                "type": "object", "additionalProperties": False, "required": ["objection_id", "verdict", "why"],
                "properties": {"objection_id": _ids(objection_ids), "verdict": {"enum": list(CROSSCHECK_VERDICTS)},
                               "why": _STRING}}}}}


def mismatch_reasons(entry: dict) -> list[str]:
    """The fixed rubric: any item that differs, or a direction other than match, is a mismatch."""
    reasons = [f"{name}: {entry['rubric'][name]['note'] or 'differs'}" for name in RUBRIC
               if entry["rubric"][name]["verdict"] == "differs"]
    if entry["direction"] != "match":
        reasons.append(f"the formal statement is {entry['direction']} than the claim: {entry['why']}")
    return reasons
