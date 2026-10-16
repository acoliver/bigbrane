"""Stage C, the challenge stage (DISCOVERY-PIPELINE-PLAN.md section 5). Runs after stage M on the frozen statements.

1. The check files are verified against MANIFEST.json (StageError on a difference, before anything runs).
2. Cited sources are resolved in the literature record (``hybrid.research.citations``: the local arXiv index, or
   OpenAlex), so the challenger can read what each cited premise points to.
3. Challenger (role ``challenger``, a family other than the prover's). It sees the card, the frozen statements and
   definitions, the M checks (with the prover's proofs and sources, complete: in batches of at most
   ``CHECK_BATCH_CHARS`` characters, one call per batch, never cut inside a source), the premise ledger and the
   resolved sources, and returns:
   - magic-X code ``def x(prefix, layout, ops)`` for each original premise marked as doing work, with the card's
     claimed advantage as a construction task (``hybrid/math/magic_x.py``). Baseline + X reaching the claim fails the
     card (kill code ``magic_x:<id>``) unless X is proven or measured and its cost is covered by a verified P1
     (``cost_in_p1``: a component that implements the premise is covered, see below). For a doing-work premise that
     is testable by code, magic-X code that is missing, has no task, quotes text that is not the card's, or does not
     run is a challenge failure (``magic_x_missing:<id>``), never a skip. FIX-9: a premise a discharging check
     settled at M (ledger ``proven`` or ``measured``, on the final ledger of this stage) is evidence, not an
     assumption, and needs no magic-X code; its item says ``settled`` ("settled at M by <statement id>"). Magic-X code
     the challenger gave for it still runs, and a baseline + X that reaches the claim still fails the card.
   - for each cited premise whose source resolved: whether the record supports the premise, with a sentence quoted
     from the record. Only a resolved source, a supporting answer and a quote that is text of the record make the
     premise ``cited``; otherwise it stays open and its reason is recorded.
   - the card's numbers for the conservation laws. Each quote is text of the card, of a frozen definition or of a
     frozen statement; a number no quote contains needs a derivation whose literals and inputs are quoted. A
     violated law fails the card (``conservation:<law>``).
   - the components the card relies on (section 5.3), each as part of the mechanism or supplied from outside it.
     Implementation evidence and cost coverage are separate (FIX-4). A component is runnable when a function of that
     name is defined in the source of an executed statement that held. Its cost is covered when that statement is a
     cost statement, or when it is a P2 or P4 witness checked under the profile's architecture contract and P1 held
     under the same contract. A runnable mechanism component without covered cost is an accounting finding on the
     cost obligation (P1), never a premise on the capability it runs for; a component named only in prose, or
     supplied from outside the mechanism without covered cost, is an open premise that blocks the claims using it.
     A training procedure is outside P1's inference cost and is recorded only.
     FIX-21 (decision MO): a component may also be a measurement object (role ``measurement``), a definition that
     belongs to a test or bound and is never supplied to the model: a P5 hypothesis class, a P2 task generator, an
     evaluation set, a protocol constant. The host decides it on structure (``measurement_definition``): not
     runnable code, used only by obligations whose frozen statements define that object (P5 conservation entries
     with defined hypothesis classes, P2 task statements), and named by its name or subject. The host corrects an
     external or prior label that passes to ``measurement`` and a ``measurement`` label that fails to ``external``,
     and records each correction with its reason in the component and the ledger (``component_role`` events). A
     measurement object never becomes a premise or a magic-X item: it is settled by its defining statements when they
     held (``measurement``) and stays open with the reason when one did not, or shows nothing after statement match
     (``measurement_open``); the claims that use it rest on those statements as before, so it never passes one.
   - wider bounds for the frozen Z3 properties. The property and hypotheses stay frozen; a model found in the wider
     box refutes the obligation's claim (``obligation_failed:<claim>``).
   - new dependency edges (only ever added) and findings: new dependencies with a subject, an exact evidence anchor,
     the dependency claimed, and a way to settle them (runnable ``verify(witness)`` code, hybrid/math/findings.py; a
     SymPy derivation; or a citation). The host settles each finding: an anchor that is not exact text, a
     code-testable finding without code, or code that crashes is ``unsupported`` and blocks nothing; a verification
     that says no is ``rejected``; a confirmed missing assumption is ``substantiated`` and becomes an open premise
     (with its measurement plan when it gives one); a verification the executor could not finish is ``unknown``
     and blocks as an open premise. A confirmed witness violation shows only that the deviation exists (FIX-13): it
     is split by statement, and each named witness is replayed with the challenger's narrow conforming replacement
     (hybrid/math/findings.conforming_variant) under its own frozen check. For a precision deviation the
     conforming replacement is the same source with its dtype casts set to the frozen contract's precision_bits, and
     nothing else changed (FIX-15). A variant that holds makes the deviation
     ``immaterial`` (the claim keeps its evidence), one decided violated makes it material (``violated``, the claim
     fails), and no valid replacement or a replay that decided nothing leaves it ``unresolved``, an open premise
     that blocks without asserting a measured failure. A definition binds only obligations that cite it, and one
     scoped to the P1 cost reference binds only cost and conformance statements. An explanatory
     objection attaches only to the card's own capability claims, never to a profile slot. Equivalent findings
     (same subject, same claims) share one stable id and are merged.
   FIX-17: the challenger also receives the card's resources (frozen with the statements) and tests them against the
   card's own claims (prompt section challenge_resources). A ``resource_contradiction`` finding (a "fixed state"
   claim against a width that grows with the task; a capability claimed at a size the stated width and precision
   cannot hold by the conservation library's storage bound) is settled by runnable code over ``witness.resources``
   and ``witness.conservation`` or by a SymPy derivation, never by a citation. Confirmed, it fails the claims it
   names and adds the kill code ``resource_contradiction``.
   Host findings: a P5 sample entry without a defined target-rule class, and a profile frozen without an
   architecture contract (P1's coverage of the witnesses cannot be verified), are raised by the stage itself.
   Magic-X runs on every substantiated doing-work dependency that comes with X code, original or new.
4. Statement match (role ``statement_match``, a family other than both the fixer's and the prover's), two calls:
   the frozen statements and their definitions alone are translated back to plain language (the claims are not
   shown), then each translation is compared with the claim it came from on a fixed rubric (quantifiers, domains,
   constants, exponents, hypotheses, inequality direction). A P3 statement is compared with the selected limits
   record and the limit it instantiates. Any item that differs, or a direction other than ``match``, is a mismatch:
   the statement then shows nothing (it cannot discharge a premise), its claim is blocked, and the card fails with
   ``statement_mismatch`` (section 5.4). For a Z3 statement the checker also writes a Python predicate from the
   claim, and the harness evaluates it against the frozen property on ``math.match_instances`` random assignments;
   a disagreement blocks the claim. The same call says whether each definition is only a convention (a definition
   that is a substantive premise is a mismatch of every statement that uses it), and which card claims each premise
   restates or implies: a premise that implies a claim it supports is circular, the claim fails with
   ``circular:<id>``, and the premise is never left open. FIX-9: circularity applies to premises assumed without
   support. A premise a discharging check settled at M is not circular whatever the finding says: the finding is
   recorded with ``circular`` False and ``settled`` ("settled at M by <statement id>"). If that check failed, or its
   statement is mismatched or unresolved here, nothing is settled and the finding applies.
   FIX-7 (match_adjudication.py). The comparison, never the blind back-translation, also reads the card's mechanism
   with anchors and the definitions' provenance; a definition that restates the card's state, update, readout or
   learning rule is part of the claim. A P4 statement is compared with the selected solvable case it instantiates
   (``instantiated_claim``), not with the profile's size ceiling. Every objection is anchored: it quotes the claim text
   and the statement or definition text that differ, and the host discards one that does not. The anchored objections
   go to a separately recorded cross-check (role ``statement_crosscheck``, gateway Sonnet; Qwen stays the
   third-family matcher). Both agree: the statement is mismatched as above. They disagree: the statement and its
   claim are ``unresolved``, the claim keeps the status its checks give, and the gate lists
   ``match_unresolved:<claim>``. The finite-domain predicate check still decides on its own.
   FIX-17 (decision XC). A dispute (the matcher upheld an anchored objection, the cross-check rejected it) goes to a
   third judge from a family other than both (role ``statement_tiebreak``, gateway ChatGPT): upheld makes the
   statement mismatched, rejected lets the objection block nothing. A failed tiebreak call settles nothing, so the
   claim stays unresolved and is not shown. Every dispute is recorded in the verdict (``disputes``).
   FIX-7 item 6. The back-translation reads each distinct statement once (identical statements share the reading;
   k2-run1's duplicate copy statement made the matcher shift ids and leave one out). A matcher reply (back-translation
   or comparison) that fails its schema after the client's retry is no reading: every statement and every claim is
   ``unresolved`` and keeps the status its checks give, and the failed call is listed under ``matcher_unread``. A
   cross-check reply that fails its schema leaves each objection it was to read ``unread``, so its statement is
   unresolved. Neither is a stage error. The cross-check never stands in for the matcher: it is Claude, the prover's
   family, and reads anchored objections, not statements.
5. Model families are checked (``hybrid.research.families``); a breach raises StageError.
6. Ledger and tier records are recomputed and the gate gives the verdict C (``mathrecords.challenge_gate``). A
   conditional claim passes only when each open premise is testable by code with a measurement plan; the verdict
   lists those plans as measurements test design must schedule.

Artifacts: ``challenge`` (everything the challenger and matcher returned and every check), ``math_check`` per
executed check, ``assumption_ledger`` and ``math_verdict`` for stage C. Events: ``math_role_call``, ``math_check``,
``challenge_check`` and ``math_decision``.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import math
import re

from hybrid.llm.errors import SchemaValidationError
from hybrid.math import architecture, constructions, manifest, profile, spec_checks
from hybrid.math import findings as finding_checks
from hybrid.research import citations, families, mathrecords
from hybrid.research.stages import m_math
from hybrid.research.stages import match_adjudication as ma
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import StageContext, StageError, card_for_prompt, load_section
from hybrid.research.stages.falsify import _normal, quote_in_card

STAGE = "C"
# Most characters of serialized stage M checks in one challenger call (FIX-4: the checks used to be cut at this many
# characters, inside O8.b's source in the k1 smoke; now they go in complete batches, one call per batch).
CHECK_BATCH_CHARS = 60000
TRAINING_FUNCTIONS = ("learn",)
# Dispositions of components and findings, strongest first (two equivalent findings keep the stronger).
DISPOSITIONS = ("violated", "substantiated", "accounting", "unresolved", "unknown", "measurement_open", "covered",
                "measurement", "immaterial", "training_scope", "rejected", "out_of_scope", "unsupported")
BLOCKING = ("substantiated", "accounting", "unresolved", "unknown")
# FIX-13: the forms whose witness is a source the host can replay under the statement's own frozen check.
REPLAYABLE = ("simulation", "construction", "cost", "learnability")
# FIX-17: a claim of the card that its own stated resources cannot hold; confirmed, it fails the claims it names.
RESOURCE_KIND = "resource_contradiction"
RESOURCE_CODE = "resource_contradiction"
# FIX-21 (decision MO): measurement objects, the definitions a profile statement's own spec freezes for its test or
# bound. Each kind: what it is called in reports, and the narrow name test a component must pass to be that object
# (matched against the component's name and subject only, never its reasoning).
MEASUREMENT_ROLE = "measurement"
MEASUREMENT_OBJECTS = {
    "hypothesis_class": ("the P5 sample entries' hypothesis_class definitions",
                         re.compile(r"hypothesis_class|\b(?:hypothes[ie]s|(?:target[- ])?rules?) class(?:es)?\b",
                                    re.IGNORECASE)),
    "task_generator": ("the P2 task generator, its frozen instances and protocol constants",
                       re.compile(r"\btask generators?\b|\b(?:evaluation|test) (?:sets?|instances)\b"
                                  r"|\bprotocol constants?\b|\b(?:accuracy|evaluation) thresholds?\b",
                                  re.IGNORECASE)),
}


def defines(source: str | None, name: str | None) -> bool:
    if not source or not name:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == name for node in ast.walk(tree))


# ----- what the challenger reads ---------------------------------------------------------------------------------------

def _check_view(check: dict) -> dict:
    view = {k: check.get(k) for k in ("form", "status", "decided", "evidence", "counterexample", "text")}
    if check.get("learner"):
        # Decision P4P: the learner in `text` is the card's supplied protocol, used verbatim.
        view["learner"] = check["learner"]
    return view


def card_supplied(check: dict | None) -> bool:
    """Whether a stage M check ran a learner the card supplied (decision P4P)."""
    return bool(check) and (check.get("learner") or {}).get("origin") == "card"


def _dumps(value) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False, default=str)


def check_batches(m_checks: dict[str, dict], limit: int = CHECK_BATCH_CHARS) -> list[dict[str, dict]]:
    """The stage M checks in complete batches: each batch serializes to at most ``limit`` characters, except a single
    check larger than that, which goes alone and whole. No check is cut or left out."""
    batches: list[dict[str, dict]] = []
    current: dict[str, dict] = {}
    for sid, check in m_checks.items():
        view = _check_view(check)
        if current and len(_dumps({**current, sid: view})) > limit:
            batches.append(current)
            current = {}
        current[sid] = view
    batches.append(current)
    return batches


def challenger_user(card: dict, record: dict, m_verdict: dict, ledger: list[dict], resolved: dict[str, dict | None],
                    lookup_problems: dict[str, str], batch: dict[str, dict] | None = None, index: int = 1,
                    total: int = 1) -> str:
    checks = batch if batch is not None else {sid: _check_view(c) for sid, c in m_verdict["checks"].items()}
    statements = [{"statement_id": s["id"], "obligation_id": o["id"], "claim_id": o["claim_id"], "form": s["form"],
                   "frozen_statement": s["spec"], "definitions": o["definitions"], "premises": o["premises"]}
                  for o in record["obligations"] for s in o["statements"]]
    sources = [{"premise_id": pid, "citation": next(p["citation"] for p in record["premises"] if p["id"] == pid),
                "resolved_record": record_or_none,
                "not_resolved_because": lookup_problems.get(pid)} for pid, record_or_none in resolved.items()]
    if total > 1:
        others = sorted(set(m_verdict["checks"]) - set(checks))
        heading = (f"\n\nSTAGE M CHECKS, BATCH {index} OF {total} (prover proofs and sources in `text`, complete; the "
                   f"checks of {', '.join(others)} are complete in the other batches, each sent in a call of its own "
                   "with everything else repeated; answer from what this batch shows):\n")
    else:
        heading = "\n\nSTAGE M CHECKS (prover proofs and sources in `text`, complete):\n"
    text = ("CARD:\n" + card_for_prompt(card)
            + "\n\nFROZEN DEFINITIONS (conventions, part of the statements):\n"
            + _dumps([{k: d[k] for k in ("id", "kind", "text", "scope") if k in d} for d in record["definitions"]])
            + "\n\nFROZEN STATEMENTS:\n" + _dumps(statements)
            + heading + _dumps(checks)
            + "\n\nPREMISE LEDGER:\n" + _dumps(ledger)
            + "\n\nCITED PREMISES AND THE LITERATURE RECORDS THEIR SOURCES RESOLVED TO:\n" + _dumps(sources)
            + "\n\nCLAIM TIER RECORDS FROM STAGE M:\n"
            + _dumps([{k: r[k] for k in ("claim_id", "status", "label", "phrase")} for r in m_verdict["claims"]])
            + "\n\n" + load_section("challenge_findings"))
    if record.get("profile"):
        contract = m_math.architecture_contract(record)
        text += ("\n\n" + load_section("challenge_profile") + "\nFROZEN ARCHITECTURE CONTRACT: "
                 + (json.dumps(contract, sort_keys=True) if contract else "none (frozen before FIX-4)"))
    if resources_text(record) is not None:
        text += ("\n\n" + load_section("challenge_resources")
                 + "\nCARD RESOURCES (frozen with the statements; quote them with source \"resources\"):\n"
                 + resources_text(record))
    return text


def merge_replies(replies: list[dict]) -> dict:
    """One challenger answer from the answers to every batch: the first answer per premise for magic-X and
    citations, the first conservation numbers given, and every component, search, edge and finding (repeats of the
    same component or edge once)."""
    merged = {"magic_x": [], "citations": [], "conservation": None, "components": [], "counterexample_searches": [],
              "new_edges": [], "findings": []}
    seen_magic, seen_cited, seen_components, seen_edges = set(), set(), set(), set()
    for reply in replies:
        for entry in reply["magic_x"]:
            if entry["premise_id"] not in seen_magic:
                seen_magic.add(entry["premise_id"])
                merged["magic_x"].append(entry)
        for entry in reply["citations"]:
            if entry["premise_id"] not in seen_cited:
                seen_cited.add(entry["premise_id"])
                merged["citations"].append(entry)
        if merged["conservation"] is None or (merged["conservation"]["claims"] is None
                                              and reply["conservation"]["claims"] is not None):
            merged["conservation"] = reply["conservation"]
        for component in reply["components"]:
            key = (component["name"].strip().lower(), component["statement_id"], component["function"])
            if key not in seen_components:
                seen_components.add(key)
                merged["components"].append(component)
        merged["counterexample_searches"] += reply["counterexample_searches"]
        for edge in reply["new_edges"]:
            key = (edge["premise_id"], edge["obligation_id"])
            if key not in seen_edges:
                seen_edges.add(key)
                merged["new_edges"].append(edge)
        merged["findings"] += reply["findings"]
    return merged


# ----- statement match presentation ----------------------------------------------------------------------------------

def definitions_text(record: dict) -> list[dict]:
    return [{"definition_id": d["id"], "kind": d["kind"], "text": d["text"]} for d in record["definitions"]]


def statements_for_match(record: dict) -> list[dict]:
    return [{"statement_id": s["id"], "form": s["form"], "formal_statement": s["spec"],
             "definitions": o["definitions"]} for o in record["obligations"] for s in o["statements"]]


def distinct_statements(shown: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """The statements the blind back-translation reads, each distinct statement once (form, formal statement and
    definitions), and {statement id: id of the statement translated for it}. Identical statements have the same
    plain reading by definition. FIX-7 item 6: k2-run1's O4.a and O11.a were the same copy learnability statement
    (the learning claim and P4.copy); the local matcher translated it once, shifted the ids of the four statements
    after it by one and left out O15.a, on the request and on the retry."""
    distinct: list[dict] = []
    same_as: dict[str, str] = {}
    first: dict[str, str] = {}
    for statement in shown:
        key = json.dumps({k: v for k, v in statement.items() if k != "statement_id"}, sort_keys=True)
        if key not in first:
            first[key] = statement["statement_id"]
            distinct.append(statement)
        same_as[statement["statement_id"]] = first[key]
    return distinct, same_as


async def matcher_reading(ctx: StageContext, role: str, prompt: str, user: str, schema: dict):
    """(reply, None), or (None, the failure) when the reply failed its schema after the client's retry (FIX-7 item 6).
    A matcher or cross-check reply that does not validate is no reading: the statements it was to settle are
    ``unresolved``, never a stage error, and nothing is salvaged from it (k2-run1's invalid back-translation carried
    the wrong ids on four valid-looking entries). Every other failure still raises."""
    try:
        return await ctx.ask(role, prompt, user, schema), None
    except StageError as exc:
        cause = exc.__cause__
        if not isinstance(cause, SchemaValidationError):
            raise
        return None, {"role": role, "prompt": prompt, "call_id": cause.call_id, "error": str(cause)[:2000]}


def unread_reason(failure: dict) -> str:
    return (f"the {failure['role']} reply ({failure['prompt']}, call {failure['call_id']}) failed its schema after "
            f"one retry, so no reading settles this match: {failure['error'][:300]}")


def supported_claims(record: dict, premise_id: str, edges: list[dict]) -> set[str]:
    """The claims whose obligations depend on a premise (the fixer's edges and the challenger's)."""
    obligations = {o["id"]: o for o in record["obligations"]}
    used = {o["id"] for o in record["obligations"] if premise_id in o["premises"]}
    used |= {e["obligation_id"] for e in edges if e["premise_id"] == premise_id}
    return {obligations[oid]["claim_id"] for oid in used if obligations[oid]["claim_id"]}


def slot_info(obligation: dict, statement: dict) -> dict | None:
    """For a profile slot's statement: the slot, the slot's statements and the task this one covers (a slot such as
    P2.composition is split over one statement per task and holds only when all hold)."""
    if not profile.is_slot(obligation["claim_id"]):
        return None
    spec = statement["spec"]
    part = spec.get("task") if isinstance(spec, dict) else None
    return {"slot": obligation["claim_id"], "statements": [s["id"] for s in obligation["statements"]],
            "part": part}


def instantiated_limit(limit: dict) -> str:
    """One selected P3 limit as the plain-language claim it instantiates."""
    task, size = limit["task"], limit.get("size")
    if task == "other":
        scope = "a task family outside the P2 tasks (the one the limit and its definitions describe)"
    else:
        name = constructions.SIZE_NAME[task]
        scope = f"the {task} task" + (f" at every {name} of {size} or more" if size else "")
    return (f"There is a task family the mechanism cannot solve within its resources: {scope}. Resources: "
            f"{limit['resources']} Derivation: {limit['why']}")


TASK_TEXT = {"copy": "the copy task", "recall": "associative recall", "dyck": "the bracket nesting task",
             "composition": "two-hop lookup", "binding": "variable binding"}


def instantiated_learnability(spec: dict) -> str:
    """One frozen P4 learnability statement as the claim it instantiates (FIX-7): the selected solvable case, with the
    example count, the seeds, the threshold, the learning rule and both data controls. The profile's size ceiling is
    not part of it: the ceiling bounds which case may be selected (profile_slot.size_ceiling)."""
    task = spec["task"]
    metric = "per-instance" if constructions.METRIC[task] == "instance" else "per-query"
    seeds = ", ".join(str(seed) for seed in spec["seeds"])
    return (f"From its initial state, using only its own learning rule, the mechanism learns {TASK_TEXT[task]} at the "
            f"selected {constructions.SIZE_NAME[task]} = {spec['size']} from m = {spec['examples']} examples, reaching "
            f"{metric} accuracy at least {spec.get('threshold', profile.P4_THRESHOLD)} on each of the seeds {seeds}, "
            "while the same learner and recipe, trained on no examples and on the same token sequences with random "
            "answers, do not both reach that accuracy.")


def selected_case(spec: dict) -> dict:
    task = spec["task"]
    return {"task": task, constructions.SIZE_NAME[task]: spec["size"], "examples": spec["examples"],
            "seeds": spec["seeds"], "threshold": spec.get("threshold", profile.P4_THRESHOLD)}


def mechanism_anchors(card: dict) -> list[dict]:
    """The card's mechanism, one entry per part with its anchor (FIX-7: the comparison reads definitions against the
    card's own state, update, readout and learning rule; the blind back-translation never sees it)."""
    mechanism = card["mechanism"]
    out = [{"anchor": f"mechanism.state.{s['name']}", "part": "state", "text": s["meaning"],
            "type": s.get("type"), "shape": s.get("shape")} for s in mechanism["state"]]
    for part in ("update", "readout"):
        out += [{"anchor": f"mechanism.{part}.{eq['name']}", "part": part, "text": eq["plain"],
                 "latex": eq.get("latex")} for eq in mechanism[part]]
    learning = mechanism["learning"]
    out.append({"anchor": "mechanism.learning.description", "part": "learning_rule", "text": learning["description"]})
    out.append({"anchor": "card.summary", "part": "summary", "text": card["summary"]})
    return out


def definitions_view(record: dict) -> list[dict]:
    """The frozen definitions as the comparison reads them, with their provenance: frozen by the statement fixer at
    stage M with the statements, never by the prover (FIX-7)."""
    return [dict(item, provenance="frozen by the statement fixer at stage M, hashed with the statements")
            for item in definitions_text(record)]


def match_items(record: dict, translations: dict[str, str]) -> list[dict]:
    items = []
    limits = (record.get("profile") or {}).get("limits") or []
    for obligation in record["obligations"]:
        claim = next((c for c in record["claims"] if c["id"] == obligation["claim_id"]), None)
        for s in obligation["statements"]:
            item = {"statement_id": s["id"], "form": s["form"],
                    "claim": claim["text"] if claim else f"(establishes premises {obligation['discharges']})",
                    "claim_quote": obligation["claim_quote"], "back_translation": translations.get(s["id"], ""),
                    "formal_statement": s["spec"], "definitions": obligation["definitions"],
                    "z3_variables": sorted(s["spec"]["variables"]) if s["form"] == "z3" else None}
            slot = slot_info(obligation, s)
            if slot is not None:
                item["profile_slot"] = slot
                if slot["slot"] == "P3" and limits:
                    # FIX-4: the comparison sees the limit the statement instantiates (the back-translation above was
                    # written without it).
                    slot["selected_limits"] = limits
                    item["instantiated_claim"] = " ".join(instantiated_limit(limit) for limit in limits)
                if slot["slot"] in profile.P4_SLOTS and s["form"] == "learnability":
                    # FIX-7: P4 selects one solvable case within the ceiling; P2 covers every listed size.
                    task = s["spec"]["task"]
                    slot["selected_case"] = selected_case(s["spec"])
                    slot["size_ceiling"] = {constructions.SIZE_NAME[task]: profile.SOLVABLE_MAX[task]}
                    item["instantiated_claim"] = instantiated_learnability(s["spec"])
            items.append(item)
    return items


def match_sections(items: list[dict]) -> str:
    """The prompt sections a comparison of these items needs (the statement-match and cross-check calls)."""
    text = "\n\n" + load_section("match_mechanism")
    if any("profile_slot" in item for item in items):
        text += "\n\n" + load_section("profile_match")
    if any(item.get("profile_slot", {}).get("slot") == "P3" and "instantiated_claim" in item for item in items):
        text += "\n\n" + load_section("profile_match_limits")
    if any("selected_case" in item.get("profile_slot", {}) for item in items):
        text += "\n\n" + load_section("profile_match_selected")
    return text


def match_user(record: dict, translations: dict[str, str], premises: list[dict], edges: list[dict], *,
               card: dict) -> str:
    items = match_items(record, translations)
    shown = [{"premise_id": p["id"], "text": p["text"], "why_needed": p["why_needed"],
              "supports_claims": sorted(supported_claims(record, p["id"], edges))} for p in premises]
    return ("STATEMENTS TO COMPARE:\n" + _dumps(items)
            + "\n\nDEFINITIONS:\n" + _dumps(definitions_view(record))
            + "\n\nCARD CLAIMS:\n" + _dumps([{"claim_id": c["id"], "text": c["text"]} for c in record["claims"]])
            + "\n\nPREMISES:\n" + _dumps(shown)
            + "\n\nCARD MECHANISM (the card's own state, update, readout and learning rule, with anchors):\n"
            + _dumps(mechanism_anchors(card)) + match_sections(items))


def crosscheck_user(record: dict, card: dict, items: list[dict], objections: list[dict]) -> str:
    """The cross-check's message: the anchored objections, the statements and definitions they concern (without the
    matcher's back-translation), the card's mechanism and claims, and the same comparison rules."""
    involved = {o["on"].split(" ", 1)[1] for o in objections if o["on"].startswith("statement ")}
    definition_ids = {o["definition_id"] for o in objections if o.get("definition_id")}
    for obligation in record["obligations"]:
        if set(obligation["definitions"]) & definition_ids:
            involved |= {s["id"] for s in obligation["statements"]}
    shown_items = [{k: v for k, v in item.items() if k != "back_translation"} for item in items
                   if item["statement_id"] in involved]
    shown = []
    for o in objections:
        if o.get("definition_id"):
            shown.append({"objection_id": o["objection_id"], "on": f"definition {o['definition_id']}",
                          **o["objection"], "matcher_why": o["why"]})
        else:
            shown.append({k: o[k] for k in ("objection_id", "on", "item", "claim_quote", "statement_quote",
                                             "difference")})
    return ("OBJECTIONS TO ADJUDICATE:\n" + _dumps(shown)
            + "\n\nSTATEMENTS THEY CONCERN:\n" + _dumps(shown_items)
            + "\n\nDEFINITIONS:\n" + _dumps(definitions_view(record))
            + "\n\nCARD CLAIMS:\n" + _dumps([{"claim_id": c["id"], "text": c["text"]} for c in record["claims"]])
            + "\n\nCARD MECHANISM (the card's own state, update, readout and learning rule, with anchors):\n"
            + _dumps(mechanism_anchors(card)) + match_sections(shown_items))


def tiebreak_user(record: dict, card: dict, items: list[dict], disputes: list[dict]) -> str:
    """The tiebreak judge's message (FIX-17): the cross-check's message for the disputed objections, plus both
    readings (the matcher upheld each objection, the cross-check rejected it) with their reasons."""
    readings = [{"objection_id": o["objection_id"], "matcher": "upheld", "matcher_why": o.get("why"),
                 "crosscheck": o.get("crosscheck"), "crosscheck_why": o.get("crosscheck_why")} for o in disputes]
    return (crosscheck_user(record, card, items, disputes)
            + "\n\nTHE TWO READINGS THAT DISAGREE (the statement-match checker upheld each objection, the "
              "cross-check rejected it):\n" + _dumps(readings))


async def tiebreak_disputes(ctx: StageContext, card_id: str, record: dict, card: dict, items: list[dict],
                            disputes: list[dict], calls: list[str], models: dict, report: dict, *,
                            stage: str = STAGE, event: str = "challenge_check") -> dict[str, dict]:
    """FIX-17 (decision XC): one call of role ``statement_tiebreak`` (a family other than the matcher's and the
    cross-check's) settles every dispute. Any failure of the call (transport, schema, a missing verdict) settles
    nothing: the disputes stay disputed, the statements unresolved and the claims not shown."""
    ids = [o["objection_id"] for o in disputes]
    try:
        reply = await ctx.ask("statement_tiebreak", "statement_tiebreak", tiebreak_user(record, card, items, disputes),
                              mc.crosscheck_schema(ids))
    except StageError as exc:
        why = f"the statement_tiebreak call failed, so the dispute stays unresolved: {exc}"[:800]
        report["tiebreak"] = {"call_id": getattr(exc.__cause__, "call_id", None), "model": None, "verdicts": None,
                              "failed": why}
        ctx.emit(event, {"card_id": card_id, "check": "statement_tiebreak", "status": "failed", "objections": ids})
        return ma.tiebreak_failed(disputes, why)
    m_math.role_call(ctx, card_id, stage, "statement_tiebreak", "settle matcher/cross-check disputes", reply)
    calls.append(reply.call_id)
    models["statement_tiebreak"].add(reply.model)
    try:
        settled = ma.tiebreak(disputes, reply.parsed["verdicts"])
        failed = None
    except ValueError as exc:
        failed = f"the statement_tiebreak reply settles nothing: {exc}"
        settled = ma.tiebreak_failed(disputes, failed)
    report["tiebreak"] = {"call_id": reply.call_id, "model": reply.model, "verdicts": reply.parsed["verdicts"],
                          "failed": failed}
    ctx.emit(event, {"card_id": card_id, "check": "statement_tiebreak", "status": "failed" if failed else "settled",
                     "objections": ids})
    return settled


async def match_statements(ctx: StageContext, card_id: str, record: dict, card: dict, premises: list[dict],
                           edges: list[dict], *, stage: str, event: str, calls: list[str], models: dict,
                           job_ids: list[str]) -> dict:
    """The statement match of a record's statements (section 5.4, FIX-7, FIX-17): the blind back-translation, the
    comparison on the rubric with the definitions and the premises, the host's anchoring of every objection, the
    cross-check of the anchored ones, the tiebreak of a dispute, and the finite-domain predicate check of a Z3
    statement. Stage C runs it on the frozen record as the independent audit; stage M runs it before the freeze on
    the statements of a fixer answer that are new or changed (FIX-19, decision M), with the same roles and rules.

    Returns {"items": one entry per statement with ``match`` (matched, mismatched or unresolved), ``reasons``,
    ``disputed`` and ``mismatch``; "readings": {definition id: the host's reading}; "premise_findings": the
    comparison's premise entries (circularity, read by stage C), or None when nothing was read; "unread": the failure
    when the back-translation or the comparison failed its schema after the retry (every item is then unresolved),
    else None; "matcher_unread", "crosscheck", "tiebreak", "disputes": what the stage records}. Role calls are
    recorded under ``stage``, check events are emitted as ``event``, and call ids, models and job ids are appended to
    ``calls``, ``models`` and ``job_ids``."""
    statements = {s["id"]: s for s in m_math.all_statements(record)}
    shown = statements_for_match(record)
    ids = [s["statement_id"] for s in shown]
    distinct, same_as = distinct_statements(shown)
    out: dict = {"items": [], "readings": {}, "premise_findings": None, "unread": None, "matcher_unread": [],
                 "crosscheck": None, "tiebreak": None, "disputes": []}
    translations: dict[str, str] = {}
    compare = None
    back, failure = await matcher_reading(
        ctx, "statement_match", "statement_backtranslate",
        "FORMAL STATEMENTS:\n" + json.dumps(distinct, indent=1, ensure_ascii=False)
        + "\n\nDEFINITIONS:\n" + json.dumps(definitions_text(record), indent=1, ensure_ascii=False),
        mc.backtranslate_schema([s["statement_id"] for s in distinct]))
    if back is not None:
        m_math.role_call(ctx, card_id, stage, "statement_match", "back-translate", back)
        calls.append(back.call_id)
        models["statement_match"].add(back.model)
        plain = {t["statement_id"]: t["plain"] for t in back.parsed["translations"]}
        translations = {sid: plain[same_as[sid]] for sid in ids}
        compare, failure = await matcher_reading(
            ctx, "statement_match", "statement_match", match_user(record, translations, premises, edges, card=card),
            mc.match_schema(ids, [d["id"] for d in record["definitions"]], [p["id"] for p in premises],
                            [c["id"] for c in record["claims"]]))
    items = {item["statement_id"]: item for item in match_items(record, translations)}
    definition_text = {d["id"]: d["text"] for d in record["definitions"]}

    if compare is None:
        # FIX-7 item 6: no valid reading of the statements (the back-translation or the comparison failed its schema
        # after the retry). Every statement is unresolved; the caller decides what that means for the claims.
        out["matcher_unread"].append(failure)
        out["unread"] = failure
        calls.append(failure["call_id"])
        reason = unread_reason(failure)
        ctx.emit(event, {"card_id": card_id, "check": "statement_match_unread",
                         "prompt": failure["prompt"], "call_id": failure["call_id"]})
        for sid in ids:
            claim_id = m_math.obligation_of(record, sid)["claim_id"]
            out["items"].append({
                "statement_id": sid, "claim_id": claim_id, "back_translation": translations.get(sid, ""),
                "reasons": [], "agreement": None, "objections": [], "match": ma.UNRESOLVED, "disputed": [reason],
                "mismatch": False})
            ctx.emit(event, {"card_id": card_id, "check": "statement_match", "statement_id": sid,
                             "mismatch": False, "match": ma.UNRESOLVED})
        return out

    m_math.role_call(ctx, card_id, stage, "statement_match", "compare", compare)
    calls.append(compare.call_id)
    models["statement_match"].add(compare.model)
    out["premise_findings"] = compare.parsed["premises"]

    # FIX-7: the host anchors every objection, then a separately recorded cross-check reads the anchored ones.
    card_side = ma.card_texts(card)
    definition_findings = _one_per_id(compare.parsed["definitions"], "definition_id",
                                      [d["id"] for d in record["definitions"]])
    readings = {did: ma.definition_objection(finding, definition_text[did], card_side)
                for did, finding in definition_findings.items()}
    entries = {}
    for entry in compare.parsed["statements"]:
        entries.setdefault(entry["statement_id"], entry)
    objections = {sid: ma.statement_objections(sid, entries[sid], items[sid], statements[sid]["spec"],
                                               [definition_text[did] for did in
                                                m_math.obligation_of(record, sid)["definitions"]])
                  for sid in ids if sid in entries}
    to_check = ma.pending(objections, readings)
    if to_check:
        cross, failure = await matcher_reading(
            ctx, "statement_crosscheck", "statement_crosscheck",
            crosscheck_user(record, card, list(items.values()), to_check),
            mc.crosscheck_schema([o["objection_id"] for o in to_check]))
        if cross is None:
            # FIX-7 item 6: an objection the cross-check could not read is neither confirmed nor rejected.
            out["matcher_unread"].append(failure)
            calls.append(failure["call_id"])
            settled_objections = ma.unread(to_check, unread_reason(failure))
            out["crosscheck"] = {"call_id": failure["call_id"], "model": None, "verdicts": None,
                                 "unread": failure["error"]}
        else:
            m_math.role_call(ctx, card_id, stage, "statement_crosscheck", "cross-check anchored objections", cross)
            calls.append(cross.call_id)
            models["statement_crosscheck"].add(cross.model)
            try:
                settled_objections = ma.adjudicate(to_check, cross.parsed["verdicts"])
            except ValueError as exc:
                raise StageError(f"statement_crosscheck: {exc}") from exc
            out["crosscheck"] = {"call_id": cross.call_id, "model": cross.model, "verdicts": cross.parsed["verdicts"]}
        for objection in to_check:
            objection.update(settled_objections[objection["objection_id"]])
        # FIX-17 (decision XC): a dispute between the matcher and the cross-check goes to a third family.
        disputes = [o for o in to_check if o.get("adjudication") == ma.DISPUTED]
        if disputes:
            settled_disputes = await tiebreak_disputes(ctx, card_id, record, card, list(items.values()), disputes,
                                                       calls, models, out, stage=stage, event=event)
            for objection in disputes:
                objection.update(settled_disputes[objection["objection_id"]])
            out["disputes"] = [ma.dispute_record(o) for o in disputes]
            for entry in out["disputes"]:
                ctx.emit(event, {"card_id": card_id, "check": "statement_dispute",
                                 "objection_id": entry["objection_id"], "on": entry.get("on"),
                                 "tiebreak": entry["tiebreak"], "outcome": entry["outcome"]})
    for did, reading in readings.items():
        ctx.emit(event, {"card_id": card_id, "check": "definition", "definition_id": did,
                         "substantive": reading.get("adjudication") == ma.CONFIRMED,
                         "host": reading["host"], "adjudication": reading.get("adjudication")})
    out["readings"] = readings

    for sid in ids:
        entry = entries.get(sid)
        obligation = m_math.obligation_of(record, sid)
        claim_id = obligation["claim_id"]
        item = {"statement_id": sid, "claim_id": claim_id, "back_translation": translations.get(sid, ""),
                "reasons": [], "agreement": None, "objections": objections.get(sid, [])}
        hard = []
        if entry is None:
            hard.append("the checker gave no comparison for this statement")
        else:
            item.update(direction=entry["direction"], rubric=entry["rubric"], why=entry["why"])
            if statements[sid]["form"] == "z3" and entry["predicate"]:
                check = await _library(ctx, card_id, sid if stage == STAGE else f"prefreeze:{sid}", {
                    "kind": "predicate_agreement", "spec": statements[sid]["spec"],
                    "predicate": entry["predicate"], "count": ctx.cfg["math"]["match_instances"], "seed": 0},
                    "predicate_agreement", stage=stage, kind="math_check" if stage == STAGE else "prefreeze_check")
                job_ids.extend(check["job_ids"])
                item["agreement"] = {"status": check["status"], "evidence": check["evidence"],
                                     "counterexample": check["counterexample"], "artifact": check["artifact"],
                                     "predicate": entry["predicate"]}
                if check["decided"] and check["status"] == "violated":
                    hard.append(f"finite-domain disagreement: {check['evidence'][:300]}")
        match, reasons, disputed = ma.settle(item["objections"], [readings[did] for did in obligation["definitions"]],
                                             hard)
        item.update(match=match, reasons=reasons, disputed=disputed, mismatch=match == ma.MISMATCHED)
        out["items"].append(item)
        ctx.emit(event, {"card_id": card_id, "check": "statement_match", "statement_id": sid,
                         "mismatch": item["mismatch"], "match": match})
    return out


# ----- sources, evidence anchors and numbers --------------------------------------------------------------------------

async def resolve_sources(ctx: StageContext, card_id: str, cited: list[dict]) -> tuple[dict, dict]:
    """Every cited premise's source looked up in the literature record: ({premise id: record or None},
    {premise id: why it did not resolve})."""
    resolved: dict[str, dict | None] = {}
    problems: dict[str, str] = {}
    if not cited:
        return resolved, problems
    lookup = ctx.lookup_source or citations.default_lookup(ctx.cfg)
    for premise in cited:
        pid, source = premise["id"], premise["citation"]
        try:
            resolved[pid] = await asyncio.to_thread(lookup, source["source"], source["id"])
        except citations.LookupFailed as exc:
            resolved[pid], problems[pid] = None, str(exc)
        ctx.emit("challenge_check", {"card_id": card_id, "check": "citation_lookup", "premise_id": pid,
                                     "source": f"{source['source']}:{source['id']}",
                                     "resolved": resolved[pid] is not None, "problem": problems.get(pid)})
    return resolved, problems


def _statement_text(record: dict, statement_id: str | None) -> str | None:
    statement = next((s for s in m_math.all_statements(record) if s["id"] == statement_id), None)
    if statement is None:
        return None
    return statement["spec"] if isinstance(statement["spec"], str) else json.dumps(statement["spec"],
                                                                                    ensure_ascii=False)


def source_text(source: str, ref: str | None, card: dict, record: dict, checks: dict[str, dict]) -> str | None:
    """The text a quote must come from: a frozen definition, a frozen statement or an M check's source or proof
    (None when the reference names nothing); the card is checked by quote_in_card."""
    if source == "definition":
        return next((d["text"] for d in record["definitions"] if d["id"] == ref), None)
    if source == "statement":
        return _statement_text(record, ref)
    if source == "check":
        return (checks.get(ref) or {}).get("text")
    if source == "resources":
        return resources_text(record)
    return None


def resources_text(record: dict) -> str | None:
    """FIX-17: the card's resources frozen with the statements, as the challenger reads them."""
    resources = record.get("resources")
    return json.dumps(resources, indent=1, ensure_ascii=False) if isinstance(resources, dict) else None


def quote_in(quote: str, source: str, ref: str | None, card: dict, record: dict, checks: dict[str, dict]) -> bool:
    """The quote is exact text (up to whitespace and case) of the named source."""
    if source == "card":
        return quote_in_card(quote, card)
    text = source_text(source, ref, card, record, checks)
    needle = _normal(quote).strip(" .;:'\"")
    return bool(text) and bool(needle) and needle in _normal(text)


NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _numbers(text: str) -> set[float]:
    return {float(match) for match in NUMBER.findall(text)}


def _leaves(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _leaves(v)]
    if isinstance(value, list):
        return [x for v in value for x in _leaves(v)]
    return []


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def conservation_problems(conservation: dict, card: dict, record: dict) -> list[str]:
    """Provenance of the challenger's conservation numbers (FIX-4). Every quote is text of the card, of a frozen
    definition or of a frozen statement. Every number in the claims is in some quote, or is the value of a
    derivation whose expression evaluates to it and whose numeric literals and input values are all in quotes. 0 and
    1 (an exact accuracy, no prior bits) need no quote. An invented number or an unrelated quote stops the check."""
    problems = []
    quotes = conservation["quotes"]
    if not quotes:
        return ["no quotes"]
    for quote in quotes:
        if not quote_in(quote["text"], quote["source"], quote["ref"], card, record, {}):
            problems.append(f"quote {quote['text'][:120]!r} is not text of {quote['source']} {quote['ref'] or ''}"
                            .rstrip())
    quoted = set().union(*(_numbers(q["text"]) for q in quotes))
    derived = set()
    for derivation in conservation.get("derivations") or []:
        expression, inputs = derivation["expression"], derivation["inputs"]
        literals = _numbers(re.sub(r"[A-Za-z_]\w*", " ", expression))
        unquoted = sorted(x for x in literals | {float(v) for v in inputs.values()}
                          if not any(_same(x, q) for q in quoted) and x not in (0.0, 1.0))
        try:
            got = architecture.value(expression, inputs)
        except (architecture.ContractError, ValueError, TypeError, ZeroDivisionError) as exc:
            problems.append(f"derivation {expression!r} cannot be evaluated: {exc}")
            continue
        if unquoted:
            problems.append(f"derivation {expression!r} uses numbers no quote contains: {unquoted}")
        elif not _same(got, float(derivation["value"])):
            problems.append(f"derivation {expression!r} at {inputs} is {got:g}, not {derivation['value']}")
        else:
            derived.add(float(derivation["value"]))
    invented = sorted({x for x in _leaves(conservation["claims"]) if x not in (0.0, 1.0)
                       and not any(_same(x, q) for q in quoted | derived)})
    if invented:
        problems.append(f"numbers neither quoted nor derived from quotes: {invented}")
    return problems


# ----- coverage of components and settlement of findings --------------------------------------------------------------

def architecture_checked(check: dict | None) -> bool:
    """The check ran under the architecture contract and every array stayed within it."""
    detail = ((check or {}).get("details") or {}).get("architecture")
    return bool(detail and detail.get("checked") and not detail.get("violations"))


def p1_coverage(record: dict, checks: dict[str, dict]) -> dict | None:
    """The verified P1 cost reference: P1 held, checked under the profile's frozen architecture contract. None for a
    record without the profile or without a contract, or when P1 did not hold that way."""
    contract = m_math.architecture_contract(record)
    if contract is None:
        return None
    for statement in m_math.slot_statements(record, ("P1",)):
        check = checks.get(statement["id"])
        if statement["form"] == "cost" and m_math.holds(check) and architecture_checked(check):
            return {"statement_id": statement["id"], "contract": contract}
    return None


def cost_obligations(record: dict) -> list[str]:
    """Where an accounting finding attaches: the P1 obligation of a profile card, else every obligation with a cost
    statement."""
    slot = [o["id"] for o in record["obligations"] if o["claim_id"] == "P1"]
    return slot or [o["id"] for o in record["obligations"] if any(s["form"] == "cost" for s in o["statements"])]


def subject_key(subject: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_") or "unnamed"


def claim_targets(record: dict, obligation_ids) -> list[str]:
    obligations = {o["id"]: o for o in record["obligations"]}
    return sorted({obligations[oid]["claim_id"] or oid for oid in obligation_ids if oid in obligations})


def finding_id(record: dict, subject: str, obligation_ids) -> str:
    """A stable id: the same subject against the same claims is the same finding, in any run or revision."""
    key = subject_key(subject) + "|" + ",".join(claim_targets(record, obligation_ids))
    return "C-" + hashlib.sha256(key.encode()).hexdigest()[:8]


def measured_object(record: dict, obligation: dict) -> str | None:
    """The measurement object an obligation's frozen statements define (FIX-21), or None. A P5 obligation whose
    statements are all conservation statements with sample entries, every entry with a defined hypothesis_class
    (definition and log2_size: profile.undefined_hypothesis_classes is empty), defines hypothesis classes. A P2
    obligation whose statements are all construction statements naming their task defines the task generator (the
    host's, hybrid/math/constructions.py, run on the frozen sizes and count). Any other obligation (cost,
    conformance, learnability, Lean, a card claim) defines none: its components stay mechanism or external parts."""
    claim_id, statements = obligation["claim_id"] or "", obligation["statements"]
    if not record.get("profile") or not statements:
        return None
    specs = [s["spec"] for s in statements]
    if claim_id == "P5" and all(s["form"] == "conservation" for s in statements) \
            and all(isinstance(spec, dict) and spec.get("sample") for spec in specs) \
            and not profile.undefined_hypothesis_classes(specs):
        return "hypothesis_class"
    if claim_id.startswith("P2.") and all(s["form"] == "construction" for s in statements) \
            and all(isinstance(spec, dict) and spec.get("task") for spec in specs):
        return "task_generator"
    return None


def measurement_definition(component: dict, record: dict, runnable: bool) -> tuple[dict | None, str]:
    """Whether a component is a measurement object, decided on structure (FIX-21, decision MO): (definition, why).

    It is one when (1) it is not runnable code (no function of its name in a check that held: code a witness defines
    is part of what the model runs), (2) every obligation in ``used_by`` defines the same measurement object
    (``measured_object``), so a component that any mechanism, cost, conformance, learnability or card-claim
    obligation uses is never one, (3) its ``statement_id`` is null or one of those obligations' statements, and (4)
    its name or subject names that object (``MEASUREMENT_OBJECTS``: narrow phrases such as "hypothesis class",
    "target-rule classes", "task generator", "test set", "protocol constant", "accuracy threshold"). The definition
    lists the frozen statements that define it.
    """
    if runnable:
        return None, "it is runnable code in a check that held, so it is part of what the model runs"
    obligations = {o["id"]: o for o in record["obligations"]}
    used = [obligations[oid] for oid in component["used_by"] if oid in obligations]
    if not used:
        return None, "it names no obligation"
    objects = {o["id"]: measured_object(record, o) for o in used}
    others = sorted(oid for oid, obj in objects.items() if obj is None)
    if others:
        return None, (f"{', '.join(others)} define no measurement object (only P5 conservation statements with "
                      "defined hypothesis classes and P2 task statements do), so it is a part of what the model uses")
    kinds = set(objects.values())
    if len(kinds) != 1:
        return None, f"its obligations define different measurement objects ({', '.join(sorted(kinds))})"
    (kind,) = kinds
    defining = [s["id"] for o in used for s in o["statements"]]
    if component["statement_id"] is not None and component["statement_id"] not in defining:
        return None, f"it names {component['statement_id']}, which is not a statement of {', '.join(objects)}"
    label, pattern = MEASUREMENT_OBJECTS[kind]
    found = pattern.search(f"{component['name']} | {component['subject']}")
    if not found:
        return None, f"its name and subject do not name {label}"
    return {"object": kind, "label": label, "statement_ids": defining, "obligations": sorted(objects),
            "matched": found.group(0)}, ""


def measurement_role(component: dict, record: dict, runnable: bool) -> tuple[dict | None, dict | None]:
    """The host's role check (FIX-21): (measurement definition or None, role correction or None). A component the
    challenger labels external, mechanism with kind prior, or measurement is tested by ``measurement_definition``.
    One that passes is a measurement object whatever its label (an external or prior label is corrected); one
    labelled measurement that fails is corrected to external, so it is settled like any part supplied from outside
    the mechanism. A mechanism component of another kind is not re-labelled."""
    role = component["role"]
    if role not in ("external", MEASUREMENT_ROLE) and component["kind"] != "prior":
        return None, None
    definition, why_not = measurement_definition(component, record, runnable)
    if definition is not None and role != MEASUREMENT_ROLE:
        return definition, {"from": role, "to": MEASUREMENT_ROLE, "reason": (
            f"host role check: {', '.join(definition['obligations'])} only use it, their frozen statements "
            f"{', '.join(definition['statement_ids'])} define {definition['label']}, and the component names it "
            f"('{definition['matched']}'); a measurement object belongs to the test or bound and is never supplied "
            "to the model at inference or training")}
    if definition is None and role == MEASUREMENT_ROLE:
        return None, {"from": role, "to": "external", "reason": (
            f"host role check: labelled a measurement object, but {why_not}; it is settled as a part supplied from "
            "outside the mechanism")}
    return definition, None


def measurement_disposition(item: dict, definition: dict, checks: dict[str, dict]) -> dict:
    """A measurement object is settled by the frozen statements that define it, and only when each held at stage M
    (``measurement``). Otherwise it stays open (``measurement_open``) with the reason, and the claims that use it
    rest on those statements as they do without it. Neither produces a premise or a magic-X item; neither passes a
    claim, whose status comes from its own statements."""
    sids = definition["statement_ids"]
    item = dict(item, measurement=definition)
    unheld = [f"{sid} {(checks.get(sid) or {}).get('status') or 'not checked'}" for sid in sids
              if not m_math.holds(checks.get(sid))]
    if not unheld:
        return dict(item, disposition="measurement", settled_by=list(sids),
                    reason=f"measurement object ({definition['label']}): settled by {', '.join(sids)}, which held at "
                           "stage M and freezes it; it is not a mechanism part and is never supplied to the model")
    return dict(item, disposition="measurement_open", settled_by=[],
                reason=f"measurement object ({definition['label']}) frozen in {', '.join(sids)}, which did not hold "
                       f"at stage M ({'; '.join(unheld)}): it stays open, and the claims that use it rest on that "
                       "statement")


def settle_measurements(components: list[dict], shows_nothing: dict[str, str]) -> list[dict]:
    """After statement match (FIX-21): a measurement object whose defining statement shows nothing at stage C
    (mismatched, match unresolved, or refuted) is no longer settled by it; it stays open with the reason. Returns the
    components changed."""
    changed = []
    for component in components:
        if component["disposition"] != "measurement":
            continue
        hit = [f"{sid}: {shows_nothing[sid][:200]}" for sid in component["settled_by"] if sid in shows_nothing]
        if hit:
            component.update(disposition="measurement_open", settled_by=[], reason=(
                f"measurement object ({component['measurement']['label']}): its defining statement shows nothing at "
                f"stage C ({'; '.join(hit)}); it stays open, and the claims that use it rest on that statement"))
            changed.append(component)
    return changed


def component_role_event(card_id: str, component: dict) -> dict:
    return {"card_id": card_id, "check": "component_role", "component": component["name"],
            "subject": component["subject"], "role": component["role"], "disposition": component["disposition"],
            "settled_by": component.get("settled_by"), "role_correction": component.get("role_correction"),
            "reason": component["reason"][:600]}


def component_disposition(component: dict, card: dict, record: dict, checks: dict[str, dict],
                          statements: dict[str, dict], coverage: dict | None) -> dict:
    """How the host classifies one component (module docstring, components). Pure: magic-X runs later."""
    sid, function = component["statement_id"], component["function"]
    check = checks.get(sid) if sid else None
    runnable = bool(check) and m_math.holds(check) and defines(check.get("text"), function)
    form = statements[sid]["form"] if sid in statements else None
    definition, correction = measurement_role(component, record, runnable)
    role = correction["to"] if correction else component["role"]
    item = dict(component, role=role, runnable=runnable, cost_measured=False, covered_by=None)
    if correction:
        item["role_correction"] = correction
    if definition is not None:
        return measurement_disposition(item, definition, checks)
    if not runnable:
        if not quote_in_card(component["quote"], card):
            return dict(item, disposition="unsupported",
                        reason="no function of that name in a check that held, and the quote is not card text")
        return dict(item, disposition="substantiated",
                    reason="the card names it, but no executed check that held defines it (section 5.3: a component "
                           "named only in prose is open)")
    if form == "cost":
        return dict(item, disposition="covered", cost_measured=True, covered_by=sid,
                    reason=f"defined in the cost statement {sid}, which held")
    if form == "learnability" and function in TRAINING_FUNCTIONS:
        return dict(item, disposition="training_scope",
                    reason=f"the training procedure of {sid}: P1 counts inference work, and P4's controls already run "
                           "the same recipe without the data")
    if coverage is not None and architecture_checked(check):
        return dict(item, disposition="covered", cost_measured=True, covered_by=coverage["statement_id"],
                    reason=f"{sid} held under the frozen architecture contract, and P1 ({coverage['statement_id']}) "
                           "held under the same contract")
    if role == "mechanism":
        why = ("no architecture contract was frozen, so nothing ties its widths to what P1 counts"
               if m_math.architecture_contract(record) is None and record.get("profile")
               else "no verified P1 cost covers it")
        return dict(item, disposition="accounting",
                    reason=f"implemented in {sid}, which held (its observed result stands), but {why}")
    if coverage is not None:
        # Runnable, inside a witness held to the contract that P1 counts: the "external" label is an allegation,
        # which only magic-X can substantiate.
        return dict(item, disposition="unsupported", allegation=True,
                    reason=f"implemented in {sid} within the architecture P1 counts; called external, which only a "
                           "magic-X run can show")
    return dict(item, disposition="substantiated",
                reason=f"supplied from outside the mechanism in {sid}, and no verified P1 cost covers it")


def finding_scope(finding: dict, record: dict) -> tuple[list[str], str | None]:
    """The obligations a finding can attach to, and why it attaches to none. An explanatory objection concerns the
    card's own account of how it works: it attaches to the card's claims, never to a profile slot, whose obligation
    is the executed result. An accounting finding attaches to the cost obligation."""
    obligations = {o["id"]: o for o in record["obligations"]}
    used = [oid for oid in finding["used_by"] if oid in obligations]
    if finding["kind"] == "explanation":
        used = [oid for oid in used if not profile.is_slot(obligations[oid]["claim_id"])]
        if not used:
            return [], ("an objection to the card's explanation of its mechanism does not attach to a profile slot; "
                        "the slot asks for the executed result")
    if finding["kind"] == "accounting":
        used = cost_obligations(record) or used
    return used, None


def merge_findings(items: list[dict]) -> list[dict]:
    """Equivalent findings (same stable id) merged: the strongest disposition, every source and reason kept."""
    merged: dict[str, dict] = {}
    for item in items:
        fid = item["id"]
        if fid not in merged:
            merged[fid] = dict(item, sources=[item["origin"]], reasons=[item["reason"]])
            continue
        kept = merged[fid]
        kept["sources"].append(item["origin"])
        kept["reasons"].append(item["reason"])
        kept["used_by"] = sorted(set(kept["used_by"]) | set(item["used_by"]))
        if DISPOSITIONS.index(item["disposition"]) < DISPOSITIONS.index(kept["disposition"]):
            for key in ("disposition", "reason", "premise", "verification", "magic_x", "does_work", "kind",
                        "measurement_plan", "text", "materiality", "resource_contradiction"):
                if key in item:
                    kept[key] = item[key]
        elif not kept.get("magic_x") and item.get("magic_x"):
            kept["magic_x"] = item["magic_x"]
    return list(merged.values())


def finding_premise(item: dict) -> dict:
    """The open premise a blocking finding becomes."""
    plan = item.get("measurement_plan")
    if item["disposition"] == "accounting":
        tail = "accounting (section 5.3): a cost check covering it, or a frozen architecture both P1 and the witness " \
               "are held to, settles it"
    elif item["disposition"] == "unknown":
        tail = "its verification did not complete (an executor stop, not a result); it blocks until it runs"
    elif item["disposition"] == "unresolved":
        tail = ("the deviation is confirmed but its effect on this witness is not established (no validated conforming "
                "replacement, or its replay did not finish); it blocks the claim without asserting a measured failure")
    else:
        tail = ("its measurement plan is for test design to schedule" if plan
                else "no measurement plan: it blocks the claims it names until a revision settles it")
    return {"id": item["id"], "text": item["text"][:600], "kind": item.get("premise_kind", "other"),
            "does_work": bool(item.get("does_work")), "why_needed": item["why"][:1000], "testable_by": "code",
            "measurement_plan": plan if item["disposition"] == "substantiated" else None, "citation": None,
            "added_by": "challenger",
            "open_basis": f"{item['disposition']} at stage C ({'; '.join(item['reasons'])[:600]}); {tail}"}


async def verify_finding(ctx: StageContext, card_id: str, finding: dict, fid: str, record: dict,
                         checks: dict[str, dict]) -> dict:
    """Run a finding's verification: {status: confirmed, rejected, unsupported or unknown, evidence, artifact}."""
    how = finding["testable_by"]
    if how == "code":
        if not finding["code"]:
            return {"status": "unsupported", "evidence": "claims to be testable by code but gives no runnable code"}
        sources = {sid: c["text"] for sid, c in checks.items() if c.get("text")}
        specs = {s["id"]: {"form": s["form"], "spec": s["spec"]} for s in m_math.all_statements(record)}
        check = await _library(ctx, card_id, statement_for(fid), {"kind": "finding", "code": finding["code"],
                                                                  "sources": sources, "specs": specs,
                                                                  "resources": record.get("resources") or {}},
                               "finding")
        if check["decided"] and check["status"] in ("discharged", "violated"):
            return {"status": "confirmed" if check["status"] == "discharged" else "rejected",
                    "evidence": check["evidence"], "artifact": check["artifact"]}
        fault = bool((check.get("details") or {}).get("prover_fault"))
        return {"status": "unsupported" if fault else "unknown", "evidence": check["evidence"] or "no result",
                "artifact": check["artifact"]}
    if how == "derivation":
        spec = finding["derivation"]
        problems = spec_checks.statement_problems("sympy", spec) if isinstance(spec, dict) else ["no derivation"]
        if problems:
            return {"status": "unsupported", "evidence": "derivation cannot be checked: " + "; ".join(problems)}
        check = await _library(ctx, card_id, statement_for(fid), {"kind": "sympy", "spec": spec}, "finding")
        if check["decided"] and check["status"] in ("discharged", "violated"):
            return {"status": "confirmed" if check["status"] == "discharged" else "rejected",
                    "evidence": check["evidence"], "artifact": check["artifact"]}
        return {"status": "unsupported", "evidence": check["evidence"] or "SymPy decided nothing",
                "artifact": check["artifact"]}
    if not finding["citation"] or not finding["record_quote"]:
        return {"status": "unsupported", "evidence": "testable by citation but gives no source and quote"}
    resolved, problems = await resolve_sources(ctx, card_id, [{"id": fid, "citation": finding["citation"]}])
    found = citations.verify(fid, resolved[fid], problems.get(fid), {
        "premise_id": fid, "supports": True, "record_quote": finding["record_quote"], "why": finding["dependency"]})
    if found["verified"]:
        return {"status": "confirmed", "evidence": found["evidence"]}
    return {"status": "unsupported", "evidence": found["reason"]}


async def settle_finding(ctx: StageContext, card_id: str, finding: dict, card: dict, record: dict,
                         checks: dict[str, dict]) -> list[dict]:
    """One challenger finding settled (module docstring, findings). A confirmed witness violation comes back as one
    item per affected statement, each settled by its own materiality replay (FIX-13); every other finding is one
    item."""
    used, out_of_scope = finding_scope(finding, record)
    fid = finding_id(record, finding["subject"], used or finding["used_by"])
    item = {"id": fid, "origin": "challenger finding", "subject": finding["subject"], "kind": finding["kind"],
            "text": finding["claim"], "why": finding["dependency"], "used_by": used, "evidence": finding["evidence"],
            "premise_kind": finding["premise_kind"], "does_work": finding["does_work"],
            "measurement_plan": finding["measurement_plan"], "magic_x": finding["magic_x"], "verification": None}
    anchor = finding["evidence"]
    if out_of_scope:
        return [dict(item, disposition="out_of_scope", reason=out_of_scope)]
    if not quote_in(anchor["quote"], anchor["source"], anchor["ref"], card, record, checks):
        return [dict(item, disposition="unsupported",
                     reason=f"the evidence quote is not text of {anchor['source']} {anchor['ref'] or ''}".rstrip())]
    if finding["kind"] == RESOURCE_KIND and finding["testable_by"] not in ("code", "derivation"):
        return [dict(item, disposition="unsupported",
                     reason="a resource contradiction is settled by code over the card's resources or by a SymPy "
                            "derivation, not by a citation")]
    verification = await verify_finding(ctx, card_id, finding, fid, record, checks)
    item["verification"] = verification
    status = verification["status"]
    if status == "confirmed" and finding["kind"] == "witness_violation":
        return await settle_materiality(ctx, card_id, finding, item, record, checks)
    if status == "confirmed" and finding["kind"] == RESOURCE_KIND:
        # FIX-17: the card's own resources cannot hold the claims it names; they fail (kill code
        # resource_contradiction). The claim is the card's, so no replay can make it immaterial.
        return [dict(item, disposition="violated", resource_contradiction=True,
                     reason=f"resource contradiction confirmed by {finding['testable_by']}: "
                            f"{verification['evidence'][:400]}")]
    if status == "confirmed":
        disposition = "accounting" if finding["kind"] == "accounting" else "substantiated"
        return [dict(item, disposition=disposition, reason=f"confirmed by {finding['testable_by']}: "
                                                          f"{verification['evidence'][:400]}")]
    return [dict(item, disposition=status, reason=f"{finding['testable_by']}: {verification['evidence'][:400]}")]


# ----- materiality of a confirmed witness violation (FIX-13) ----------------------------------------------------------

def source_hash(text: str | None) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None


def witness_items(finding: dict, item: dict, record: dict) -> list[tuple[dict, dict | None, dict | None]]:
    """The statements a confirmed witness violation is settled on, one per item: (obligation, statement or None,
    target or None). Each target the challenger names in an obligation the finding names is one item (the first
    target per statement); an obligation the finding names with no target is one item without a statement, which
    stays unresolved because nothing identifies the witness to replay. A target outside those obligations is one
    item that attaches to nothing."""
    obligations = {o["id"]: o for o in record["obligations"]}
    out, seen, named = [], set(), set()
    for target in finding.get("targets") or []:
        sid = target["statement_id"]
        if sid in seen:
            continue
        seen.add(sid)
        obligation = m_math.obligation_of(record, sid)
        statement = next(s for s in obligation["statements"] if s["id"] == sid)
        out.append((obligation, statement, target))
        named.add(obligation["id"])
    out += [(obligations[oid], None, None) for oid in item["used_by"] if oid not in named]
    return out


def bound_by(obligation: dict, statement: dict, anchor: dict, record: dict) -> str | None:
    """Why a definition anchor does not bind this witness, or None. A deviation from a frozen definition is a
    violation only for an obligation that cites the definition, and a definition scoped to the P1 cost reference
    binds only cost and conformance statements (FIX-13)."""
    if anchor["source"] != "definition":
        return None
    did = anchor["ref"]
    if did not in obligation["definitions"]:
        return f"{obligation['id']} does not cite {did}, so its witness is not held to it"
    definition = next((d for d in record["definitions"] if d["id"] == did), {})
    if definition.get("scope") == "cost_reference" and statement["form"] not in mc.COST_REFERENCE_FORMS:
        return (f"{did} is scoped to the P1 cost reference (a convention chosen for the exact count); it does not "
                f"bind the {statement['form']} witness {statement['id']}")
    return None


async def replay_conforming(ctx: StageContext, card_id: str, statement: dict, target: dict,
                            original: str | None, precision_bits: int | None = None) -> dict:
    """The materiality record of one witness: the validated conforming variant run through the statement's own frozen
    check (m_math.run_check: the frozen spec and the frozen architecture contract, so ``audited_construction`` for a
    construction). ``outcome`` is ``immaterial`` (the variant holds), ``material`` (decided violated), ``unresolved``
    (no valid variant, or a check that decided nothing) or ``rejected`` (the deviation is not text of this witness).
    ``precision_bits`` is the frozen contract's float width: with it, an edit that only restores that precision is a
    valid variant (FIX-15); ``edit_kind`` records which rules the edit was held to, and a precision variant records
    the wider float names it still has."""
    record = {"statement_id": statement["id"], "form": statement["form"], "deviation": target["deviation"],
              "edits": target["edits"], "why": target["why"], "original_sha256": source_hash(original),
              "variant_sha256": None, "variant_source": None, "problems": [], "check": None,
              "precision_bits": precision_bits, "edit_kind": finding_checks.edit_kind(target["edits"], precision_bits)}
    if not isinstance(original, str) or target["deviation"] not in original:
        return dict(record, outcome="rejected", evidence=f"the deviation {target['deviation'][:120]!r} is not text of "
                                                          f"{statement['id']}'s recorded source")
    variant, problems = finding_checks.conforming_variant(original, target["deviation"], target["edits"],
                                                          precision_bits)
    if variant is None:
        return dict(record, outcome="unresolved", problems=problems,
                    evidence="no valid conforming replacement: " + "; ".join(problems))
    record.update(variant_sha256=source_hash(variant), variant_source=variant)
    if record["edit_kind"] == "precision":
        record["wider_precision_left"] = finding_checks.wider_precision_left(variant, precision_bits)
    check = await m_math.run_check(ctx, card_id, STAGE, statement, purpose="materiality", text=variant)
    record["check"] = {"artifact": check["artifact"], "status": check["status"], "decided": check["decided"],
                       "evidence": check["evidence"], "job_ids": check["job_ids"],
                       "results": (check.get("details") or {}).get("results"),
                       "prover_fault": check.get("prover_fault", False)}
    if m_math.holds(check):
        outcome = "immaterial"
    elif check["decided"] and check["status"] == "violated":
        outcome = "material"
    else:
        outcome = "unresolved"
    return dict(record, outcome=outcome, evidence=check["evidence"] or "; ".join(check["reasons"]) or "no result")


def materiality_summary(item: dict) -> dict:
    """What the verdict records of a finding split by statement (the challenge artifact keeps the whole replay,
    variant source included)."""
    if "parent" not in item:
        return {}
    replay = item.get("materiality") or {}
    check = replay.get("check") or {}
    return {"statement_id": item.get("statement_id"), "parent": item["parent"],
            "materiality": {"outcome": replay.get("outcome"), "original_sha256": replay.get("original_sha256"),
                            "variant_sha256": replay.get("variant_sha256"), "artifact": check.get("artifact"),
                            "status": check.get("status"), "decided": check.get("decided")} if replay else None}


async def settle_materiality(ctx: StageContext, card_id: str, finding: dict, item: dict, record: dict,
                             checks: dict[str, dict]) -> list[dict]:
    """A confirmed witness violation split by statement (FIX-13). Confirmation shows that the deviation exists; it
    decides a claim only through a host replay of a conforming replacement under the frozen check:

    - the variant holds: ``immaterial``; the claim keeps its stage M evidence and the validated variant is recorded
      as a conforming witness;
    - the variant is decided violated (below the threshold, a resource or numerical violation): ``violated``, a
      material violation that fails the claim;
    - no target names the witness, the edit is not a valid conforming replacement, or the check decided nothing
      (a crash, an executor stop): ``unresolved``, which blocks the claim without asserting a measured failure;
    - the deviation is not text of that witness's source: ``rejected``; a definition the obligation does not cite,
      or one scoped to the P1 cost reference, does not bind the witness: ``out_of_scope``.

    When the frozen precision or evaluation order excludes a numerical shortcut, the replay runs without it, so
    agreement in exact arithmetic cannot excuse an overflow: in k1-run1 the recall witness without max subtraction
    is decided violated. A witness that computes above the frozen precision is replayed at the frozen precision
    (FIX-15): in k2-run2 both O9 witnesses ran in float64 under a binary32 contract."""
    out = []
    anchor = finding["evidence"]
    precision_bits = (m_math.architecture_contract(record) or {}).get("precision_bits")
    for obligation, statement, target in witness_items(finding, item, record):
        oid = obligation["id"]
        sid = statement["id"] if statement else None
        piece = dict(item, id=finding_id(record, f"{finding['subject']} {sid or oid}", [oid]), used_by=[oid],
                     statement_id=sid, parent=item["id"], materiality=None)
        if oid not in item["used_by"]:
            out.append(dict(piece, used_by=[], disposition="out_of_scope",
                            reason=f"{sid} is in {oid}, which the finding does not name"))
            continue
        if statement is None:
            out.append(dict(piece, disposition="unresolved",
                            reason=f"confirmed ({item['verification']['evidence'][:300]}), but no target names a "
                                   f"witness of {oid} with a conforming replacement, so its effect is not "
                                   "established"))
            continue
        unbound = bound_by(obligation, statement, anchor, record)
        if unbound:
            out.append(dict(piece, disposition="out_of_scope", reason=unbound))
            continue
        if statement["form"] not in REPLAYABLE:
            out.append(dict(piece, disposition="unresolved",
                            reason=f"{sid} is a {statement['form']} statement, which has no witness to replay"))
            continue
        if card_supplied(checks.get(sid)):
            # Decision P4P: the card's measured learning protocol is used verbatim; no variant replaces it.
            out.append(dict(piece, disposition="out_of_scope",
                            reason=f"{sid}'s learner is the card's supplied learning protocol (sha256 "
                                   f"{checks[sid]['learner'].get('sha256')}), used verbatim (decision P4P): its "
                                   "recipe, width and m are the card's, and no conforming variant replaces it"))
            continue
        replay = await replay_conforming(ctx, card_id, statement, target, (checks.get(sid) or {}).get("text"),
                                         precision_bits)
        disposition = {"immaterial": "immaterial", "material": "violated", "unresolved": "unresolved",
                       "rejected": "rejected"}[replay["outcome"]]
        reason = {"immaterial": f"immaterial: the deviation is confirmed, and {sid} without it holds under its frozen "
                                "check, so the claim keeps its evidence",
                  "violated": f"material: {sid} without the deviation fails its frozen check",
                  "unresolved": f"unresolved: the deviation is confirmed, but its effect on {sid} is not established",
                  "rejected": f"the deviation is not in {sid}"}[disposition]
        out.append(dict(piece, materiality=replay, disposition=disposition,
                        reason=f"{reason}: {replay['evidence'][:400]}"))
        ctx.emit("challenge_check", {"card_id": card_id, "check": "materiality", "finding_id": piece["id"],
                                     "statement_id": sid, "outcome": replay["outcome"],
                                     "artifact": (replay["check"] or {}).get("artifact")})
    return out


def host_findings(record: dict, components: list[dict]) -> list[dict]:
    """Findings the stage raises itself: P5 entries without a defined target-rule class, and a profile frozen
    without an architecture contract while runnable mechanism components have no verified cost coverage."""
    found = []
    if record.get("profile"):
        p5 = [o for o in record["obligations"] if o["claim_id"] == "P5"]
        specs = [s["spec"] for o in p5 for s in o["statements"] if s["form"] == "conservation"]
        gaps = profile.undefined_hypothesis_classes(specs)
        if p5 and gaps:
            used = [o["id"] for o in p5]
            found.append({"id": finding_id(record, "hypothesis_class", used), "origin": "host", "kind":
                          "missing_assumption", "subject": "hypothesis_class", "used_by": used, "premise_kind": "prior",
                          "does_work": False, "measurement_plan": None, "magic_x": None, "verification": None,
                          "text": "the target-rule class whose size P5's sample bounds use is defined, and its size "
                                  "is justified",
                          "why": "P5's sample-information inequality takes log2_H as an input; a passing inequality "
                                 "does not show the class it counts",
                          "disposition": "substantiated",
                          "reason": "host: the frozen P5 sample entries do not define their class: " + "; ".join(
                              dict.fromkeys(gaps))[:400]})
        accounting = [c for c in components if c["disposition"] == "accounting"]
        if m_math.architecture_contract(record) is None and accounting:
            used = cost_obligations(record)
            names = ", ".join(f"{c['name']} ({c['statement_id']})" for c in accounting)
            found.append({"id": finding_id(record, "architecture_contract", used), "origin": "host",
                          "kind": "accounting", "subject": "architecture_contract", "used_by": used,
                          "premise_kind": "other", "does_work": False, "measurement_plan": None, "magic_x": None,
                          "verification": None,
                          "text": "P1's cost reference counts the architecture the P2 and P4 witnesses run",
                          "why": "every component the witnesses use must appear in P1's counted cost (section 5.3)",
                          "disposition": "accounting",
                          "reason": f"host: no architecture contract was frozen, so P1 coverage of {names} cannot "
                                    "be verified; their observed results stand"})
    return found


def component_findings(components: list[dict], record: dict) -> list[dict]:
    """Components that become findings. Accounting findings that come from a missing architecture contract are one
    finding, the host's (they are the same gap)."""
    out = []
    no_contract = record.get("profile") and m_math.architecture_contract(record) is None
    for component in components:
        disposition = component["disposition"]
        if disposition not in BLOCKING:
            continue
        if disposition == "accounting":
            if no_contract:
                continue
            used = cost_obligations(record) or component["used_by"]
        else:
            used = component["used_by"]
        out.append({"id": finding_id(record, component["subject"], used), "origin": f"component {component['name']}",
                    "kind": "accounting" if disposition == "accounting" else "missing_assumption",
                    "subject": component["subject"], "used_by": used,
                    "premise_kind": component["kind"], "does_work": disposition != "accounting",
                    "measurement_plan": None, "magic_x": component.get("magic_x"), "verification": None,
                    "text": f"component {component['name']}: "
                            + ("its cost is counted in P1" if disposition == "accounting"
                               else "it exists as costed mechanism code, or the claims that use it hold without it"),
                    "why": component["why"], "disposition": disposition, "reason": component["reason"]})
    return out


# ----- the stage -------------------------------------------------------------------------------------------------------

async def run(ctx: StageContext, card_id: str) -> str | None:
    m_math.require_math(ctx)
    m_math.verify_manifest()
    card = ctx.card(card_id)
    record = m_math.load_frozen(ctx, card_id)
    m_verdicts = [v for v in _verdicts(ctx, card_id) if v["stage"] == "M"]
    if record is None or not m_verdicts:
        raise RuntimeError(f"{card_id}: stage C needs stage M's frozen statements and verdict")
    m_verdict = m_verdicts[-1]
    statements = {s["id"]: s for s in m_math.all_statements(record)}
    checks = {sid: dict(c) for sid, c in m_verdict["checks"].items()}
    _, ledger, _ = m_math.evaluate(record, checks)
    calls: list[str] = []
    models: dict[str, set] = {"statement_fixer": set(m_verdict["models"]["statement_fixer"]),
                              "prover": set(m_verdict["models"]["prover"]), "challenger": set(),
                              "statement_match": set(), "statement_crosscheck": set(), "statement_tiebreak": set()}
    report: dict = {"magic_x": [], "citations": [], "conservation": None, "components": [], "counterexamples": [],
                    "new_edges": [], "new_premises": [], "findings": [], "statement_match": [], "definitions": [],
                    "circularity": [], "challenger_calls": []}

    # ----- cited sources, resolved before the challenger reads them
    cited_premises = [p for p in record["premises"] if p["testable_by"] == "citation" and p["citation"]]
    resolved, lookup_problems = await resolve_sources(ctx, card_id, cited_premises)

    # ----- challenger, one call per complete batch of checks
    obligation_ids = [o["id"] for o in record["obligations"]]
    premise_ids = [p["id"] for p in record["premises"]]
    executed_ids = [sid for sid, s in statements.items()
                    if s["form"] in ("simulation", "construction", "cost", "learnability")]
    z3_ids = [sid for sid, s in statements.items() if s["form"] == "z3"]
    resolved_ids = [pid for pid, found in resolved.items() if found is not None]
    schema = mc.challenger_schema(premise_ids, obligation_ids, executed_ids, z3_ids, resolved_ids,
                                  statement_ids=list(statements), definition_ids=[d["id"] for d in record["definitions"]])
    batches = check_batches(m_verdict["checks"])
    replies = []
    for index, batch in enumerate(batches, start=1):
        reply = await ctx.ask("challenger", "challenger", challenger_user(
            card, record, m_verdict, ledger, resolved, lookup_problems, batch, index, len(batches)), schema)
        m_math.role_call(ctx, card_id, STAGE, "challenger", "challenge" if len(batches) == 1
                         else f"challenge batch {index} of {len(batches)}", reply)
        calls.append(reply.call_id)
        report["challenger_calls"].append({"call_id": reply.call_id, "checks": sorted(batch)})
        models["challenger"].add(reply.model)
        replies.append(reply.parsed)
    challenge = merge_replies(replies)
    job_ids: list[str] = []

    # ----- components: implementation evidence, then cost coverage (FIX-4)
    coverage = p1_coverage(record, checks)
    components = [component_disposition(c, card, record, checks, statements, coverage)
                  for c in challenge["components"]]
    # FIX-21: every role the host corrected and every measurement object goes to the ledger with its reason.
    for component in components:
        if component.get("role_correction") or component["role"] == MEASUREMENT_ROLE:
            ctx.emit("challenge_check", component_role_event(card_id, component))

    # ----- findings: the host's, the components', and the challenger's, settled before anything blocks
    settled = [piece for finding in challenge["findings"]
               for piece in await settle_finding(ctx, card_id, finding, card, record, checks)]
    findings = merge_findings(host_findings(record, components) + component_findings(components, record) + settled)
    for item in findings:
        ctx.emit("challenge_check", {"card_id": card_id, "check": "finding", "finding_id": item["id"],
                                     "disposition": item["disposition"], "sources": item["sources"]})

    # ----- magic-X on every premise that does work: the original premises, then the substantiated new ones
    status = {entry["id"]: entry["status"] for entry in ledger}
    covered_premises = {c["premise_id"] for c in components if c.get("premise_id") and c["disposition"] == "covered"
                        and coverage is not None}
    magic_failures: list[str] = []
    by_premise = {}
    for entry in challenge["magic_x"]:
        by_premise.setdefault(entry["premise_id"], entry)
    for premise in record["premises"]:
        if not premise["does_work"]:
            continue
        pid = premise["id"]
        required = premise["testable_by"] == "code"
        item = await _magic_x(ctx, card_id, card, pid, premise["kind"], status[pid], pid in covered_premises,
                              by_premise.get(pid), job_ids)
        item.update(testable_by=premise["testable_by"], required=required, missing=False, origin="original")
        if item["status"] == "violated":
            magic_failures.append(pid)
        if required and not (item["ran"] and item["status"] != "unknown"):
            item["missing"] = True
        report["magic_x"].append(item)
        ctx.emit("challenge_check", {"card_id": card_id, "check": "magic_x", "premise_id": pid,
                                     "status": item["status"], "ran": item["ran"], "missing": item["missing"]})
    # A substantiated new dependency is open; without X code it stays open and blocks (never a pass), with X code a
    # baseline that reaches the claim through it fails the card.
    for item in findings:
        if item["disposition"] != "substantiated" or not item.get("does_work") or not item.get("magic_x"):
            continue
        result = await _magic_x(ctx, card_id, card, item["id"], item.get("premise_kind", "other"), "open", False,
                                item["magic_x"], job_ids)
        result.update(testable_by="code", required=False, missing=False, origin=item["origin"])
        if result["status"] == "violated":
            magic_failures.append(item["id"])
        report["magic_x"].append(result)
        ctx.emit("challenge_check", {"card_id": card_id, "check": "magic_x", "premise_id": item["id"],
                                     "status": result["status"], "ran": result["ran"], "missing": False})
    # A runnable component called external inside a contract-held witness: magic-X decides the allegation.
    for component in components:
        if not component.get("allegation"):
            continue
        result = await _magic_x(ctx, card_id, card, finding_id(record, component["subject"], component["used_by"]),
                                component["kind"], "open", True, component.get("magic_x"), job_ids)
        result.update(testable_by="code", required=False, missing=False, origin=f"component {component['name']}")
        component["magic_x_status"] = result["status"]
        if result["status"] == "violated":
            magic_failures.append(result["premise_id"])
        elif result["ran"]:
            component.update(disposition="rejected", reason=component["reason"] + f"; magic-X: {result['evidence']}"
                             [:300])
        report["magic_x"].append(result)

    # ----- cited premises: the challenger's reading of each resolved record
    answers = {}
    for answer in challenge["citations"]:
        answers.setdefault(answer["premise_id"], answer)
    cited_ok: dict[str, str] = {}
    citation_problems: dict[str, str] = {}
    for premise in cited_premises:
        pid = premise["id"]
        found = citations.verify(pid, resolved[pid], lookup_problems.get(pid), answers.get(pid))
        report["citations"].append(dict(found, citation=premise["citation"]))
        if found["verified"]:
            cited_ok[pid] = found["evidence"]
        else:
            citation_problems[pid] = found["reason"]
        ctx.emit("challenge_check", {"card_id": card_id, "check": "citation", "premise_id": pid,
                                     "verified": found["verified"], "reason": found["reason"]})

    # ----- conservation laws on the card's numbers
    laws: list[str] = []
    conservation = challenge["conservation"]
    if conservation["claims"]:
        bad = conservation_problems(conservation, card, record)
        if bad:
            report["conservation"] = {"ran": False, "evidence": "numbers not backed by quotes: " + "; ".join(bad)}
        else:
            check = await _library(ctx, card_id, "conservation", {"kind": "conservation",
                                                                  "claims": conservation["claims"]}, "conservation")
            job_ids += check["job_ids"]
            report["conservation"] = {"ran": True, "status": check["status"], "evidence": check["evidence"],
                                      "counterexample": check["counterexample"], "artifact": check["artifact"]}
            if check["decided"] and check["status"] == "violated":
                laws = sorted({code.split(":", 1)[1] for code in (check.get("details") or {}).get("kill_codes", [])}
                              or {(check["counterexample"] or {}).get("law", "unknown")})
        ctx.emit("challenge_check", {"card_id": card_id, "check": "conservation",
                                     "status": (report["conservation"] or {}).get("status")})
    else:
        report["conservation"] = {"ran": False, "evidence": conservation["why"]}

    # ----- the dependencies that block, and the witnesses that failed
    new_premises, new_edges, failed_claims = [], [], {}
    obligations = {o["id"]: o for o in record["obligations"]}
    for item in findings:
        if item["disposition"] in BLOCKING:
            premise = finding_premise(item)
            new_premises.append(premise)
            new_edges += [{"premise_id": premise["id"], "obligation_id": oid} for oid in item["used_by"]]
        elif item["disposition"] == "violated":
            what = "resource contradiction" if item.get("resource_contradiction") else "witness violation"
            for oid in item["used_by"]:
                claim_id = obligations[oid]["claim_id"]
                if claim_id:
                    failed_claims.setdefault(claim_id, f"{what} {item['id']}: {item['reason']}"[:1000])
    new_edges += [{"premise_id": e["premise_id"], "obligation_id": e["obligation_id"]}
                  for e in challenge["new_edges"]]
    report.update(components=components, findings=findings, new_premises=new_premises, new_edges=new_edges)

    # ----- counterexample searches on the frozen Z3 properties
    refuted_statements: dict[str, dict] = {}
    counterexample_claims: list[str] = []
    for search in challenge["counterexample_searches"]:
        statement = statements[search["statement_id"]]
        spec = json.loads(json.dumps(statement["spec"]))
        unknown_vars = sorted(set(search["bounds"]) - set(spec["variables"]))
        if unknown_vars or any(b["lo"] > b["hi"] for b in search["bounds"].values()):
            report["counterexamples"].append({"statement_id": statement["id"], "ran": False,
                                              "evidence": f"bad bounds (unknown variables {unknown_vars} or lo > hi)"})
            continue
        for name, bound in search["bounds"].items():
            if spec["variables"][name]["sort"] != "bool":
                spec["variables"][name].update(lo=bound["lo"], hi=bound["hi"])
        check = await m_math.run_check(ctx, card_id, STAGE, statement, purpose="counterexample", spec_override=spec)
        job_ids += check["job_ids"]
        report["counterexamples"].append({"statement_id": statement["id"], "ran": True, "bounds": search["bounds"],
                                          "status": check["status"], "evidence": check["evidence"],
                                          "counterexample": check["counterexample"], "artifact": check["artifact"]})
        if check["decided"] and check["status"] == "violated":
            refuted_statements[statement["id"]] = check
            claim_id = m_math.obligation_of(record, statement["id"])["claim_id"]
            if claim_id:
                counterexample_claims.append(claim_id)
        ctx.emit("challenge_check", {"card_id": card_id, "check": "counterexample", "statement_id": statement["id"],
                                     "status": check["status"]})

    # ----- statement match: blind back-translation, then the rubric, the definitions and circularity
    blocked: dict[str, str] = {}
    mismatched: dict[str, str] = {}
    unresolved_statements: dict[str, str] = {}
    unresolved_claims: dict[str, str] = {}
    circular: dict[str, str] = {}
    premises = list(record["premises"]) + new_premises
    match = await match_statements(ctx, card_id, record, card, premises, new_edges, stage=STAGE,
                                   event="challenge_check", calls=calls, models=models, job_ids=job_ids)
    for key in ("matcher_unread", "crosscheck", "tiebreak", "disputes"):
        report[key] = match[key]
    report["definitions"] += list(match["readings"].values())
    if match["unread"] is not None:
        reason = unread_reason(match["unread"])
        for item in match["items"]:
            unresolved_statements[item["statement_id"]] = reason[:1000]
            report["statement_match"].append(item)
        # Neither the definitions nor the premises' circularity were read either.
        for claim in record["claims"]:
            unresolved_claims.setdefault(claim["id"], f"statement match unread: {reason}")
    else:
        for item in match["items"]:
            sid, claim_id, match_status = item["statement_id"], item["claim_id"], item["match"]
            if match_status == ma.MISMATCHED:
                mismatched[sid] = "; ".join(item["reasons"])[:1000]
                if claim_id:
                    blocked.setdefault(claim_id, f"statement {sid} does not match the claim: "
                                       + "; ".join(item["reasons"]))
            elif match_status == ma.UNRESOLVED:
                unresolved_statements[sid] = "; ".join(item["disputed"])[:1000]
                if claim_id:
                    unresolved_claims.setdefault(claim_id, f"statement {sid}: the matcher's objection is not "
                                                 "corroborated by the cross-check: " + "; ".join(item["disputed"]))
            report["statement_match"].append(item)

        for pid, finding in _one_per_id(match["premise_findings"], "premise_id",
                                        [p["id"] for p in premises]).items():
            supports = supported_claims(record, pid, new_edges)
            hit = sorted(set(finding["implies"]) & supports)
            if hit:
                circular[pid] = f"restates or implies {', '.join(hit)}, which it supports: {finding['why']}"[:1000]
            report["circularity"].append({"premise_id": pid, "implies": sorted(set(finding["implies"])),
                                          "supports": sorted(supports), "circular": bool(hit),
                                          "why": finding["why"]})
            ctx.emit("challenge_check", {"card_id": card_id, "check": "circularity", "premise_id": pid,
                                         "circular": bool(hit)})

    # FIX-21: a measurement object is settled only by a defining statement that still shows something here.
    shows_nothing = {**{sid: "refuted by a counterexample" for sid in refuted_statements},
                     **{sid: f"match unresolved: {why}" for sid, why in unresolved_statements.items()},
                     **{sid: f"mismatched: {why}" for sid, why in mismatched.items()}}
    for component in settle_measurements(components, shows_nothing):
        ctx.emit("challenge_check", component_role_event(card_id, component))

    breaches = families.violations({role: used for role, used in models.items() if used})
    if breaches:
        raise StageError("model family separation broken: " + "; ".join(breaches))

    _, ledger, records = m_math.evaluate(record, checks, extra_edges=new_edges, extra_premises=new_premises,
                                         cited=cited_ok, citation_problems=citation_problems, circular=circular,
                                         citations_checked=True, blocked=blocked,
                                         refuted_statements=refuted_statements, mismatched=mismatched,
                                         failed=failed_claims, unresolved=unresolved_statements,
                                         unresolved_claims=unresolved_claims)
    # FIX-9: a premise settled at M by a check that held is exempt from the magic-X code requirement and from the
    # circularity verdict. Decided on the final ledger, so a discharging statement that stage C found mismatched or
    # unresolved settles nothing and the premise keeps every check.
    settled = mathrecords.settled_at_m(ledger)
    magic_missing = settle_magic_x(report["magic_x"], settled)
    for item in report["magic_x"]:
        if item.get("settled"):
            ctx.emit("challenge_check", {"card_id": card_id, "check": "magic_x", "premise_id": item["premise_id"],
                                         "status": item["status"], "ran": item["ran"], "missing": False,
                                         "settled": item["settled"]})
    for pid in settle_circularity(report["circularity"], settled):
        ctx.emit("challenge_check", {"card_id": card_id, "check": "circularity", "premise_id": pid,
                                     "circular": False, "settled": settled[pid]})
    verdict, codes = mathrecords.challenge_gate(records, ledger, conservation=laws, magic_x=magic_failures,
                                                magic_x_missing=magic_missing, counterexamples=counterexample_claims,
                                                mismatched=sorted(mismatched))
    if any(f.get("resource_contradiction") and f["disposition"] == "violated" and f["used_by"] for f in findings):
        # FIX-17: the claims a confirmed resource contradiction names failed above; the code names the cause.
        codes = sorted(set(codes) | {RESOURCE_CODE})
        verdict = verdict if verdict == "infeasible_proved" else "fail"
    report["proposer_family_note"] = proposer_note(ctx, card, models["challenger"])
    profile_extra = {"profile": m_math.profile_block(record, records)} if record.get("profile") else {}
    ctx.put_json_artifact("challenge", ctx.artifact_name(card_id), {"card_id": card_id, **report,
                                                                    "challenger_call": calls[0]}, card_id)
    m_math.decide(ctx, card_id, STAGE, verdict, codes, records, record["definitions"], ledger, calls + job_ids, {
        "models": {role: sorted(used) for role, used in models.items()},
        "checks": m_verdict["checks"], "statements_artifact": m_math.frozen_event(ctx, card_id)["path"],
        "magic_x": report["magic_x"],
        "findings": [dict({k: f.get(k) for k in ("id", "disposition", "kind", "subject", "used_by", "sources",
                                                 "reason")}, **materiality_summary(f),
                          **({"resource_contradiction": True} if f.get("resource_contradiction") else {}))
                     for f in findings],
        "citations": [{k: c[k] for k in ("premise_id", "verified", "evidence", "reason")} for c in report["citations"]],
        "conservation": report["conservation"],
        "statement_match": [{k: i.get(k) for k in ("statement_id", "claim_id", "mismatch", "match", "reasons",
                                                   "disputed")} for i in report["statement_match"]],
        "crosscheck": report["crosscheck"], "matcher_unread": report["matcher_unread"],
        "tiebreak": report["tiebreak"], "disputes": report["disputes"],
        "definition_checks": report["definitions"], "circularity": report["circularity"],
        "counterexamples": report["counterexamples"],
        "component_roles": [{k: c.get(k) for k in ("name", "subject", "role", "disposition", "settled_by",
                                                   "role_correction", "reason")}
                            for c in components if c.get("role_correction") or c["role"] == MEASUREMENT_ROLE],
        **profile_extra})
    return "S4" if verdict == "pass" else None


async def _magic_x(ctx: StageContext, card_id: str, card: dict, pid: str, kind: str, premise_status: str,
                   cost_in_p1: bool, entry: dict | None, job_ids: list[str]) -> dict:
    """Baseline + X for one dependency. ``cost_in_p1`` is True only when a component covered by the verified P1
    implements it (FIX-4: the exemption also needs X proven or measured, which magic_x.py checks)."""
    item = {"premise_id": pid, "ran": False, "status": None, "evidence": "", "artifact": None,
            "cost_in_p1": cost_in_p1}
    if entry is None or not entry["code"]:
        item["evidence"] = "the challenger gave no runnable code"
    elif entry["claim"] is None:
        item["evidence"] = f"no construction task fits the card's claimed advantage: {entry['why']}"
    elif entry["quote"] and not quote_in_card(entry["quote"], card):
        item["evidence"] = "the advantage quote is not text of the card's claims; magic-X not run"
    else:
        claim = dict(entry["claim"], params=entry["claim"]["params"] or None)
        status = premise_status if premise_status in ("proven", "cited", "measured", "open") else "open"
        request = {"kind": "magic_x", "claim": claim,
                   "assumption": {"id": pid, "kind": kind, "source": entry["code"], "status": status,
                                  "cost_in_p1": cost_in_p1}}
        check = await _library(ctx, card_id, statement_for(pid), request, "magic_x")
        job_ids += check["job_ids"]
        decided = check["decided"] and check["status"] in ("violated", "discharged")
        item.update(ran=True, status=check["status"] if decided else "unknown",
                    artifact=check["artifact"], evidence=check["evidence"] or "; ".join(check["reasons"]))
    return item


def settle_magic_x(items: list[dict], settled: dict[str, str]) -> list[str]:
    """The premise ids whose required magic-X code is missing (``magic_x_missing``), after exempting the original
    premises a discharging check settled at M (FIX-9): such an item is marked ``settled`` ("settled at M by <statement
    id>"), not required and not missing. A magic-X run that reached the claim (``violated``) is not changed here: it
    still fails the card through ``magic_x``."""
    missing = []
    for item in items:
        if item["origin"] == "original" and item["premise_id"] in settled:
            item.update(required=False, missing=False, settled=settled[item["premise_id"]])
        elif item["missing"]:
            missing.append(item["premise_id"])
    return missing


def settle_circularity(findings: list[dict], settled: dict[str, str]) -> list[str]:
    """Marks the statement-match circularity findings on premises a discharging check settled at M (FIX-9): the
    finding is kept (``implies``, ``supports``, ``why``), ``circular`` is False and ``settled`` says by which check.
    Returns the premise ids whose finding would have made them circular. The ledger applies the same rule
    (mathrecords.premise_ledger)."""
    changed = []
    for finding in findings:
        if finding["premise_id"] in settled:
            if finding["circular"]:
                changed.append(finding["premise_id"])
            finding.update(circular=False, settled=settled[finding["premise_id"]])
    return changed


def _one_per_id(entries: list[dict], key: str, expected: list[str]) -> dict[str, dict]:
    """The statement-match role's finding for each expected id. The schema asks for exactly one each; a repeated
    id in place of a missing one means a finding is missing, which is a role failure (resumable), not a pass."""
    found: dict[str, dict] = {}
    for entry in entries:
        found.setdefault(entry[key], entry)
    missing = [i for i in expected if i not in found]
    if missing:
        raise StageError(f"statement_match gave no finding for {key} {', '.join(missing)}")
    return found


def statement_for(premise_id: str) -> str:
    return f"premise {premise_id}"


async def _library(ctx: StageContext, card_id: str, label: str, request: dict, purpose: str, *,
                   stage: str = STAGE, kind: str = "math_check") -> dict:
    """A library check that is not a frozen statement (magic-X, conservation, predicate agreement, a finding's
    verification), recorded as a ``kind`` artifact and event: math_check, or for the predicate check of a statement
    not yet frozen (stage M's match before the freeze, FIX-19) prefreeze_check, so every math_check of stage M stays
    a check of a frozen statement."""
    try:
        record = await asyncio.to_thread(ctx.math.check_library, request)
    except manifest.ManifestMismatch as exc:
        raise StageError(f"math check files differ from MANIFEST.json: {exc}") from exc
    printed = (record["runs"][0].get("printed") or {}) if record["runs"] else {}
    check = {"card_id": card_id, "stage": stage, "statement_id": label, "purpose": purpose, "form": request["kind"],
             "status": record["status"], "decided": record["decided"], "evidence": record["evidence"],
             "counterexample": record["counterexample"], "reasons": record["problems"], "runs": record["runs"],
             "job_ids": [run["job_id"] for run in record["runs"]], "check_hashes": record["check_hashes"],
             "request": request, "details": {k: printed[k] for k in ("kill_codes", "checks", "details", "checked",
                                                                     "confirmed", "prover_fault")
                                             if k in printed}}
    if any(run.get("status") == "infra_error" for run in record["runs"]):
        raise StageError(f"executor infra_error in {purpose} check {label}")
    name = ctx.artifact_name(card_id)
    ctx.put_json_artifact(kind, name, check, card_id)
    ctx.emit(kind, {"card_id": card_id, "stage": stage, "statement_id": label, "purpose": purpose,
                    "status": check["status"], "decided": check["decided"], "artifact": name,
                    "job_ids": check["job_ids"]})
    return dict(check, artifact=name)


def _verdicts(ctx: StageContext, card_id: str) -> list[dict]:
    out = []
    for data in ctx.card_events(card_id, "artifact"):
        if data["kind"] == "math_verdict":
            out.append(json.loads((ctx.run_dir / data["path"]).read_text(encoding="utf-8")))
    return out


def proposer_note(ctx: StageContext, card: dict, challenger_models: set) -> str:
    """Whether the challenger shares the proposer's family (reported, not enforced)."""
    call = card.get("created_by_call")
    proposer = next((e["data"]["model"] for e in ctx.ledger.iter_events("llm_call")
                     if e["data"].get("call_id") == call), None)
    if proposer is None:
        return "proposer model not in this run's ledger (copied card); not compared"
    same = {families.family(proposer)} & {families.family(m) for m in challenger_models}
    return (f"challenger shares the proposer's family ({', '.join(same)}); allowed where routing leaves no other"
            if same else f"challenger family differs from the proposer's ({proposer})")
