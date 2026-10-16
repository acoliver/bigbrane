"""Stage M, the math stage (DISCOVERY-PIPELINE-PLAN.md sections 4.1 to 4.3). Replaces S3 when s3.mode is "math".

Order of work, each step recorded before the next starts:

1. The check files are verified against hybrid/math/MANIFEST.json; a difference stops the stage (StageError)
   before any model call or job.
2. Statement fixer (role ``statement_fixer``, a family other than the prover's). For each card claim it writes an
   obligation with one to three formal statements (a Lean ``def Claim : Prop``, a bounded Z3 property, a SymPy
   relation, or an executed specification: a simulation, a construction task or a cost formula), the definitions
   (conventions and protocol choices) and premises (substantive claims a proof relies on but does not prove) each
   obligation uses, and the claims it cannot formalize. A premise says why it is needed and whether code (with a
   measurement plan) or a citation can settle it. Host checks (statement data, claim quotes, coverage, ids, a
   premise without its plan or citation) get one retry with the problems quoted; statements that still fail are
   dropped and recorded. FIX-18: a retry answer that still carries a Lean lint finding (a forbidden construct, a
   bound name that is a Lean 4 keyword) gets one more round with those findings, so a renameable identifier is
   repaired before freezing. The fixer copies the card's own P3 limits (CARD LIMITS); a limit the card does not
   state is a problem and is refused at freeze.
3. Freeze: the statements are hashed (Lean: the whitespace-normalized hash of Phase 1; others: sorted-key JSON), the
   definitions too, and all are stored as a ``math_statements`` artifact; a ``math_statements_frozen`` event
   records the artifact's sha256 and every statement and definition hash. Definitions are part of what is frozen:
   the prover and the statement-match role read them with the statements. No prover call happens before this
   event. A resumed M reuses the frozen record.
4. Checks without a prover: Z3 and SymPy statements run in the tools image as they are.
5. Prover (role ``prover``), at most ``math.max_prover_attempts`` rounds; each round counts as one attempt for every
   statement still open. A round is several prover calls made together (``prover_batches``): each hand-set
   construction obligation alone, the other statements grouped by form, at most ``PROVER_BATCH`` per call. A reply
   that is still not valid JSON for the schema after the client's retry is a failed attempt for the statements of
   its call, and the rounds go on (plan section 10: four attempts per obligation). A call whose every backend reply
   spent the whole output cap on reasoning without text (hybrid.llm.OutputExhausted, after the client's one retry at
   lower effort) is split in two and asked again once in the same round when it held several statements (event
   ``prover_split``); a statement whose call is still exhausted has a failed attempt (``output_exhausted``, FIX-5).
   Any other transport failure of every backend stops the stage. Each learnability statement's best measured attempt
   of the run (full source, recipe, per-seed accuracies, controls, training times and the learner's CURVE) is shown in
   later rounds, and a learner that changes a recipe which passed on every seed earlier in the run, without a
   ``recipe_change_reason``, is refused without a job (prover_memory.py). A round whose failures repeat the previous round's exactly (the same counterexamples, or
   the same proofs sent again) ends the rounds early (``stall:prover``, Phase 8). The prover returns only proofs
   (Lean, ``theorem claim_holds : Claim``) and sources (``run`` for a simulation, ``init``/``update``/``predict`` for
   a construction or a cost statement). A proof that redeclares ``Claim`` is rejected without running. Before every
   check the frozen artifact is re-read and its hash and the statement's hash compared with the freeze event
   (StageError on a difference), and the runner verifies MANIFEST.json again. A learnability check runs a timing
   probe first (MathRunner.check_learnability; event ``math_probe``, never deciding); a projected overrun or a full
   job stopped by the wall cap is a resource failure: status unknown, recorded under ``resource`` with the cap and
   the measured and projected cost, and shown to the prover in the next round. ``ResourceHistory`` keeps such
   failures by frozen experiment, so an unchanged source is not run again and, after a full timeout, an equivalent
   experiment runs its full job only when its probe shows a lower cost.
   P2 constructions (FIX-6) get the same within-run memory: each construction statement's best measured attempt
   (per-size accuracy and state, full source) and its ``frozen_contract`` (the rounds, widths, hidden width, positions
   and state words its check allows at each size), with the prompt section construction_contract: the frozen contract
   is the authority, and a claim that it is insufficient is argued as a finding, never acted on by submitting a
   witness known to fail.
5b. Conformance audits (FIX-6): after the rounds, every ``conformance`` statement runs the cost reference that held
   for its ``reference`` statement against the fixer's frozen equations (spec_checks.reference_conformance). Its
   obligation discharges the premise that the reference executes the card's own equations and buffers; a premise
   about the card's own frozen definitions is settled only here, never deferred to test design
   (mathrecords, Where a premise is settled). A reference that did not hold leaves the audit unrun (status unknown).
6. Tier records, the premise ledger and the gate (``hybrid.research.mathrecords``). Artifacts
   ``assumption_ledger`` (the definitions and the premise ledger) and ``math_verdict`` (stage M), event
   ``math_decision``, verdict M: ``pass`` (to C; conditional claims included when each open premise can still be
   settled: code with a measurement plan, or a citation stage C will verify), ``infeasible_proved`` (a Z3 or
   SymPy counterexample), ``fail`` (an executed check failed), ``undetermined`` (a claim not shown, kill code
   ``unproven:<claim>``, or conditional on a premise nothing can settle, ``unproven:<premise>``).

Transformer profile (Phase 4, ``math.profile``; hybrid/math/profile.py). The claims to formalize then include the
profile slots P1, P2.copy, P2.recall, P2.nesting, P2.composition, P3, P4.copy, P4.recall, P4.nesting,
P4.composition and P5, each with the forms it accepts (a statement of another form is dropped), its required sizes and
thresholds; the fixer also states the P3 limits. A slot holds only when every statement it has holds; after the
checks, a P3 limit that a P2 construction beat and a P2 size that P5 does not account for fail P3 and P5. The kill
codes name the slot (``obligation_failed:P2.nesting``, ``unproven:P3``). The verdict records each slot's status and
tier. With ``target.enabled`` the compute target gate (hybrid/math/target.py, decision D1) then runs on the verified P1
formula and a miss fails the card with ``obligation_failed:target``; calibration runs the profile with the target off
and applies the gate separately. FIX-17 (decision C17): the coupled measure (hybrid/math/coupled.py) then substitutes
the card's per-task resources into the verified P1 formulas and records each P2 slot's skill scale b (task size
s = n^b with total cost at most n^coupled_max_cost_exponent and state at most n^coupled_max_state_exponent, [target]);
when the target gate applies, a card with b > 0 on no P2 slot fails with ``obligation_failed:target_coupled``.
Calibration records it only.

Concurrency. The checks without a prover, and the checks of one prover round, run at once (``execute_checks``); the
executor's host-wide job slots (hybrid/slots.py, ``[math] max_parallel_jobs``) bound how many jobs run, and every job
keeps its own caps. Learnability statements of the same frozen experiment still run one after another, so the
ResourceHistory rules above see the earlier result. The checks are recorded afterwards in statement order
(``record_in_order``), so the artifacts, events and decisions do not depend on which job finished first.

Every check is a ``math_check`` artifact and event, every prover attempt a ``proof_attempt`` event, every role call a
``math_role_call`` event (the LLM client also writes its ``llm_call``). Executor infrastructure errors raise
StageError so the card can be resumed. The fixer and the prover must come from different model families
(``hybrid.research.families``); a breach raises StageError.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter

from hybrid.math import constructions, cost, coupled, lean_check, learnability, manifest, profile, target
from hybrid.math import runner as math_runner
from hybrid.research import families, lineage, mathrecords
from hybrid.research.schemas import validate
from hybrid.research.stages import card_resources
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.prover_memory import BestAttempts, frozen_contract
from hybrid.research.stages.common import StageContext, StageError, card_for_prompt, load_section
from hybrid.research.stages.falsify import claim_items
from hybrid.research.stages.s3_feasibility import known_open_issues

STAGE = "M"
# What a check record keeps from the printed report, beyond the contract fields (FIX-3 added the cost checks' first
# failures, the crash location and the learnability recipe, which the prover's next round reads).
# "wall_clock" is a learnability check's training times, kept apart from its deterministic "results" (FIX-5).
DETAIL_KEYS = ("results", "checked", "kill_codes", "checks", "failed_checks", "crash", "recipe", "architecture",
               "wall_clock")
LIBRARY_KIND = {"z3": "z3", "sympy": "sympy", "simulation": "simulation", "construction": "audited_construction",
                "cost": "cost", "conservation": "conservation", "learnability": "learnability",
                "conformance": "conformance"}


# ----- small helpers ---------------------------------------------------------------------------------------------

def require_math(ctx: StageContext) -> None:
    if ctx.math is None:
        raise RuntimeError("stages M and C need ctx.math (a hybrid.math.runner.MathRunner)")


def verify_manifest() -> dict:
    try:
        return manifest.verify()
    except manifest.ManifestMismatch as exc:
        raise StageError(f"math check files differ from MANIFEST.json; nothing was run: {exc}") from exc


def role_call(ctx: StageContext, card_id: str, stage: str, role: str, purpose: str, reply) -> None:
    ctx.emit("math_role_call", {"card_id": card_id, "stage": stage, "role": role, "purpose": purpose,
                                "call_id": reply.call_id, "model": reply.model})


def frozen_event(ctx: StageContext, card_id: str) -> dict | None:
    events = ctx.card_events(card_id, "math_statements_frozen")
    return events[-1] if events else None


def load_frozen(ctx: StageContext, card_id: str) -> dict | None:
    """The card's frozen statements, verified against the freeze event; None when none were frozen."""
    event = frozen_event(ctx, card_id)
    if event is None:
        return None
    raw = (ctx.run_dir / event["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != event["sha256"]:
        raise StageError(f"{event['path']} changed after it was frozen")
    record = json.loads(raw)
    for statement in all_statements(record):
        if statement_digest(statement) != event["hashes"].get(statement["id"]):
            raise StageError(f"frozen statement {statement['id']} does not match its recorded hash")
    for definition in record["definitions"]:
        if mc.definition_hash(definition) != event["hashes"].get(definition["id"]):
            raise StageError(f"frozen definition {definition['id']} does not match its recorded hash")
    return record


def all_statements(record: dict) -> list[dict]:
    return [s for o in record["obligations"] for s in o["statements"]]


def statement_digest(statement: dict) -> str:
    from hybrid.math import spec_checks
    return (lean_check.statement_hash(statement["spec"]) if statement["form"] == "lean"
            else spec_checks.spec_hash(statement["spec"]))


def obligation_of(record: dict, statement_id: str) -> dict:
    return next(o for o in record["obligations"] if any(s["id"] == statement_id for s in o["statements"]))


# ----- statement fixer and freezing -----------------------------------------------------------------------------------

def profile_required(ctx: StageContext) -> bool:
    return ctx.cfg["math"]["profile"]


def fixer_user(ctx: StageContext, card_id: str, card: dict, claims: list[dict]) -> str:
    parts = ["CARD:\n" + card_for_prompt(card),
             "CLAIMS TO FORMALIZE (claim_id: text):\n" + "\n".join(f"- {c['id']}: {c['text']}" for c in claims)]
    if any(profile.is_slot(c["id"]) for c in claims):
        parts.append(load_section("profile_obligations") + "\n\n" + profile.template_text(ctx.math.library_caps) + "\n"
                     + params_note())
        # FIX-16: the card's resources, which the statements copy.
        if card_resources.resources_text(card):
            parts.append(card_resources.resources_text(card))
        # FIX-18: the card's own P3 limits, which the limits and the P3 statements formalize.
        if card_resources.limits_text(card):
            parts.append(card_resources.limits_text(card))
    # FIX-6: new section file; a premise about the card's own frozen definitions is settled at stage M.
    parts.append(load_section("premise_settlement"))
    # FIX-7: new section file; a definition that restates the card's own mechanism has kind "mechanism".
    parts.append(load_section("mechanism_definitions"))
    issues = known_open_issues(ctx, card_id)
    if issues:
        parts.append(issues)
    dialogue = ctx.latest_json_artifact(card_id, "dialogue_assumptions")
    if dialogue is not None:
        # Phase 7: the refinement dialogue's definitions and premises, in this role's own vocabulary.
        parts.append(load_section("refine_premises") + "\n" + json.dumps(
            {key: dialogue[key] for key in ("definitions", "premises", "open_questions")}, indent=2,
            ensure_ascii=False))
    return "\n\n".join(parts)


def params_note() -> str:
    """The layout parameters each task accepts, read from the construction harness. In all three card-runs of the
    2026-10-13 calibration the fixer first put training settings (optimizer, learning rate, epochs, width) in a
    learnability statement's params and spent its one retry moving them."""
    from hybrid.math import constructions
    accepted = {task: sorted(set(defaults) | ({"n_keys"} if task == "recall" else set()))
                for task, defaults in constructions.DEFAULTS.items()}
    return ("Task parameters: `params` of a construction or learnability statement accepts only the task's layout "
            "parameters (" + "; ".join(f"{task}: {', '.join(names) or 'none'}" for task, names in accepted.items())
            + "). The mechanism's model class (its architecture, learning rule and the widths the card states) is a "
            "convention: state it in a definition. The training recipe (initialization, optimizer, learning rate, "
            "epochs or passes, batch size) is not frozen: the prover chooses it within the budget and the same "
            "recipe trains both controls (see P4 TRAINING RECIPE).")


def limits(cfg: dict, required: bool) -> tuple[int, int]:
    """Most obligations and statements the fixer may write: the [math] limits, plus room for the profile slots."""
    if not required:
        return cfg["max_obligations"], cfg["max_statements"]
    return cfg["max_obligations"] + len(profile.SLOTS), cfg["max_statements"] + PROFILE_STATEMENTS


# Statements the profile needs at most: one per slot, two for P2.composition, three for P3.
PROFILE_STATEMENTS = len(profile.SLOTS) + 3


async def fix_and_freeze(ctx: StageContext, card_id: str, card: dict, claims: list[dict]) -> dict:
    required = any(profile.is_slot(c["id"]) for c in claims)
    max_obligations, max_statements = limits(ctx.cfg["math"], required)
    claim_ids = [c["id"] for c in claims]
    texts = {c["id"]: c["text"] for c in claims}
    schema = mc.statement_fixer_schema(claim_ids, max_obligations, profile_required=required)
    user = fixer_user(ctx, card_id, card, claims)
    reply = await ctx.ask("statement_fixer", "statement_fixer", user, schema)
    role_call(ctx, card_id, STAGE, "statement_fixer", "write statements", reply)
    calls, first_problems = [reply.call_id], mc.fixer_problems(reply.parsed, card, claim_ids, max_statements, texts)
    lint_problems: list[str] = []
    # FIX-19 (decision M): each answer that has a round after it is matched with its claims before the freeze, by the
    # roles and rules of stage C; an upheld mismatch goes back to the fixer as a problem. Unchanged statements keep
    # their reading (``matches``), so each round matches only new or changed statements.
    matches: dict[str, dict] = {}
    match_rounds: list[dict] = []
    match_problems, entry = await prefreeze_match(ctx, card_id, card, claims, mc.freeze(
        reply.parsed, card, claim_ids, max_statements, texts), matches, 1)
    match_rounds.append(entry)
    if first_problems or match_problems:
        reply = await ask_again(ctx, card_id, user, schema, reply, first_problems + match_problems,
                                "rewrite statements after host problems")
        calls.append(reply.call_id)
        # FIX-18 (decision L): the Lean lint runs on the rewritten answer too. A statement that first appears in the
        # rewrite (in calibration 200147Z K1 run 3, P3 with a bound variable named `prefix`) would otherwise be
        # dropped at freeze for a renameable identifier with no round left; the findings get one repair round.
        lint_problems = mc.lint_problems(mc.fixer_problems(reply.parsed, card, claim_ids, max_statements, texts))
        match_problems, entry = await prefreeze_match(ctx, card_id, card, claims, mc.freeze(
            reply.parsed, card, claim_ids, max_statements, texts), matches, 2)
        match_rounds.append(entry)
        if lint_problems or match_problems:
            purpose = ("rename after Lean lint problems" if not match_problems
                       else "restate after statement-match problems" if not lint_problems
                       else "repair after Lean lint and statement-match problems")
            reply = await ask_again(ctx, card_id, user, schema, reply, lint_problems + match_problems, purpose)
            calls.append(reply.call_id)
        # The repair round is the last: its answer is frozen as written, and stage C's match decides on it.
    frozen = mc.freeze(reply.parsed, card, claim_ids, max_statements, texts)
    record = {"card_id": card_id, "card_version": card["version"], "fixer_calls": calls,
              "fixer_model": reply.model, "claims": claims, **frozen, "first_problems": first_problems,
              "prefreeze_match": match_rounds}
    if lint_problems:
        record["lint_problems"] = lint_problems
    validate("math_statement", record)
    stored = ctx.put_json_artifact("math_statements", ctx.artifact_name(card_id), record, card_id)
    hashes = {s["id"]: s["sha256"] for s in all_statements(record)}
    hashes.update({d["id"]: d["sha256"] for d in record["definitions"]})
    ctx.emit("math_statements_frozen", {"card_id": card_id, "path": stored["path"].relative_to(ctx.run_dir).as_posix(),
                                        "sha256": stored["sha256"], "hashes": hashes, "fixer_calls": calls,
                                        "fixer_model": reply.model, "dropped": len(record["dropped"])})
    return record


async def ask_again(ctx: StageContext, card_id: str, user: str, schema: dict, reply, problems: list[str],
                    purpose: str):
    """One revision round of the statement fixer: its previous answer and the host's problems, answered in full."""
    retry_user = (user + "\n\nYOUR PREVIOUS ANSWER (rejected by the host checks):\n"
                  + json.dumps(reply.parsed, indent=1, ensure_ascii=False)[:30000]
                  + "\n\nPROBLEMS TO FIX (answer again in full; statements that still fail are dropped):\n"
                  + "\n".join(f"- {p}" for p in problems))
    reply = await ctx.ask("statement_fixer", "statement_fixer", retry_user, schema)
    role_call(ctx, card_id, STAGE, "statement_fixer", purpose, reply)
    return reply


# ----- statement match before the freeze (FIX-19) -------------------------------------------------------------------

PREFREEZE_EVENT = "statement_prefreeze_check"
MATCH_ROLES = ("statement_match", "statement_crosscheck", "statement_tiebreak")


def match_key(statement: dict, obligation: dict, record: dict) -> str:
    """Everything a statement's match reads besides the card: the statement, its claim (or the premises it
    discharges), the definitions its obligation cites and, for P3, the limits it instantiates. A statement whose key
    is unchanged since an earlier round keeps that round's reading."""
    claim = next((c for c in record["claims"] if c["id"] == obligation["claim_id"]), None)
    definitions = {d["id"]: d for d in record["definitions"]}
    key = {"form": statement["form"], "spec": statement["spec"], "claim_id": obligation["claim_id"],
           "claim": claim["text"] if claim else None, "claim_quote": obligation["claim_quote"],
           "discharges": None if claim else obligation["discharges"],
           "definitions": [{k: definitions[did].get(k) for k in ("id", "kind", "text", "scope")}
                           for did in obligation["definitions"] if did in definitions]}
    if obligation["claim_id"] == "P3":
        key["limits"] = (record.get("profile") or {}).get("limits")
    return hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def match_subrecord(record: dict, statement_ids: set[str]) -> dict:
    """The part of a draft record the match reads for some of its statements: those statements in their
    obligations, the definitions those obligations cite, every claim, and the profile (the P3 limits)."""
    obligations = [dict(o, statements=[s for s in o["statements"] if s["id"] in statement_ids])
                   for o in record["obligations"]]
    obligations = [o for o in obligations if o["statements"]]
    cited = {did for o in obligations for did in o["definitions"]}
    return {"claims": record["claims"], "obligations": obligations, "premises": [],
            "definitions": [d for d in record["definitions"] if d["id"] in cited], "profile": record.get("profile")}


def mismatch_problem(sid: str, claim_id: str | None, reasons: list[str]) -> str:
    oid, letter = sid.split(".")
    what = f"claim {claim_id}" if claim_id else "the premises it discharges"
    return (f"{oid} statement {'abc'.index(letter) + 1} ({sid}): statement match before the freeze: it does not "
            f"formalize {what}. The statement-match checker raised the objection and it was upheld (cross-check, or "
            "the tiebreak on a dispute), with the same rules as the audit at stage C: " + "; ".join(reasons)[:1500]
            + ". Restate it over the claim's own objects (for P3, the card's update, state or readout that the limit "
              "names, as the card's mechanism definitions give them); a statement that still differs is blocked at "
              "stage C.")


async def prefreeze_match(ctx: StageContext, card_id: str, card: dict, claims: list[dict], draft: dict,
                          matches: dict[str, dict], round_index: int) -> tuple[list[str], dict]:
    """FIX-19 (decision M): the statement match of one fixer answer before the freeze. ``draft`` is what freezing the
    answer would give (math_contracts.freeze). Statements whose match key is in ``matches`` reuse that reading; the
    others go to stage C's match (c_challenge.match_statements: back-translation, comparison, cross-check, tiebreak,
    predicate check) without premises, since circularity is a question about the frozen premises and stays at C.
    Returns the problems (one per statement whose mismatch was upheld) and the round's record. A disputed or unread
    match is recorded and not fed back: stage C decides it. A reading that failed its schema is not kept, so the
    statement is matched again in the next round."""
    from hybrid.research.stages import c_challenge  # c_challenge imports this module
    from hybrid.research.stages import match_adjudication as ma
    record = dict(draft, claims=claims)
    keys = {s["id"]: match_key(s, o, record) for o in record["obligations"] for s in o["statements"]}
    todo = [sid for sid, key in keys.items() if key not in matches]
    entry = {"round": round_index, "matched": todo, "reused": [sid for sid in keys if sid not in todo],
             "mismatched": {}, "unresolved": {}, "calls": [], "models": {}}
    readings: dict[str, dict] = {}
    if todo:
        models: dict[str, set] = {role: set() for role in MATCH_ROLES}
        job_ids: list[str] = []
        result = await c_challenge.match_statements(ctx, card_id, match_subrecord(record, set(todo)), card, [], [],
                                                    stage=STAGE, event=PREFREEZE_EVENT, calls=entry["calls"],
                                                    models=models, job_ids=job_ids)
        entry["models"] = {role: sorted(used) for role, used in models.items() if used}
        if job_ids:
            entry["job_ids"] = job_ids
        if result["unread"] is not None:
            entry["unread"] = result["unread"]["call_id"]
        for item in result["items"]:
            reading = {"match": item["match"], "reasons": item["reasons"], "disputed": item["disputed"]}
            readings[item["statement_id"]] = reading
            if result["unread"] is None:
                matches[keys[item["statement_id"]]] = reading
    problems = []
    for sid, key in keys.items():
        reading = readings.get(sid) or matches[key]
        claim_id = obligation_of(record, sid)["claim_id"]
        if reading["match"] == ma.MISMATCHED:
            entry["mismatched"][sid] = reading["reasons"]
            problems.append(mismatch_problem(sid, claim_id, reading["reasons"]))
        elif reading["match"] == ma.UNRESOLVED:
            entry["unresolved"][sid] = reading["disputed"]
    ctx.emit("statement_prefreeze_match", {"card_id": card_id, "round": round_index, "matched": entry["matched"],
                                           "reused": entry["reused"], "mismatched": sorted(entry["mismatched"]),
                                           "unresolved": sorted(entry["unresolved"]), "calls": entry["calls"]})
    return problems, entry


def prefreeze_models(record: dict) -> dict[str, set]:
    """The models each statement-match role used before the freeze (FIX-19), for the family check."""
    used: dict[str, set] = {}
    for entry in record.get("prefreeze_match") or []:
        for role, models in entry.get("models", {}).items():
            used.setdefault(role, set()).update(models)
    return used


# ----- running checks ---------------------------------------------------------------------------------------------

def _raise_on_infra(check: dict) -> None:
    infra = [run for run in check["runs"] if run.get("status") == "infra_error"]
    if infra:
        raise StageError(f"executor infra_error in math check {check['statement_id']}: {infra[0].get('detail', '')}")


def architecture_contract(record: dict | None) -> dict | None:
    """The profile's shared architecture contract (FIX-4), None for a record frozen without one."""
    return ((record or {}).get("profile") or {}).get("architecture")


def library_request(statement: dict, text: str | None, contract: dict | None = None) -> dict:
    """The job request of a statement. With the profile's architecture contract (FIX-4) a cost, construction or
    learnability statement that declares its widths is also checked against the contract. FIX-20: a cost statement
    runs its reference under the contract's precision policy (``precision_bits``), the counter the conformance audit
    of that reference uses; a conformance statement carries its frozen ``precision_bits`` in its spec."""
    form, spec = statement["form"], statement["spec"]
    kind = LIBRARY_KIND[form]
    checked = contract is not None and isinstance(spec, dict) and spec.get("widths") is not None
    if form in ("z3", "sympy"):
        return {"kind": kind, "spec": spec}
    if form == "conservation":
        return {"kind": kind, "claims": spec}
    if form == "cost":
        claim = {k: spec.get(k) for k in cost.CLAIM_FIELDS}
        request = {"kind": "cost", "source": text, "claim": claim}
        if spec.get("ns"):
            request["ns"] = spec["ns"]
        if checked:
            request["architecture"] = {"shared": contract, "widths": spec["widths"]}
        if contract is not None:
            request["precision_bits"] = contract["precision_bits"]
        return request
    request = {"kind": kind, "spec": spec, "source": text}
    if checked and form in ("construction", "learnability"):
        request["architecture"] = contract
    return request


async def run_check(ctx: StageContext, card_id: str, stage: str, statement: dict, *, purpose: str,
                    text: str | None = None, attempt: int | None = None, prover_call: str | None = None,
                    spec_override: dict | None = None, history: "ResourceHistory | None" = None,
                    refused: str | None = None) -> dict:
    """Run one check of a frozen statement (``text`` is the prover's proof or source) and record it. The frozen
    artifact is re-verified first; ``spec_override`` is only for the challenger's wider Z3 bounds. ``history`` (the
    prover rounds') holds learnability experiments that already stopped for their cost: an unchanged source for one
    is refused without a job, and another source runs its full check only when its timing probe shows a lower
    cost. ``refused`` is a host reason not to run the source at all (FIX-5: a learner that changes a recipe that passed
    on every seed earlier in the run without saying why); the attempt is recorded as an unknown prover fault."""
    check = await execute_check(ctx, card_id, stage, statement, purpose=purpose, text=text, attempt=attempt,
                                prover_call=prover_call, spec_override=spec_override, history=history,
                                refused=refused)
    return record_check(ctx, card_id, stage, statement, check)


async def execute_check(ctx: StageContext, card_id: str, stage: str, statement: dict, *, purpose: str,
                        text: str | None = None, attempt: int | None = None, prover_call: str | None = None,
                        spec_override: dict | None = None, history: "ResourceHistory | None" = None,
                        refused: str | None = None, skipped: str | None = None) -> dict:
    """The check of ``run_check`` without recording it: the jobs run and the check record is returned, but no
    artifact or event is written (``record_check`` does that), so concurrent checks can be recorded in statement
    order. A resource failure is noted in ``history`` here, before any later check of the same experiment runs.
    ``skipped`` is a host reason the check has nothing to run (FIX-6: a conformance audit whose reference statement did
    not hold); it is recorded as unknown, not as a prover fault."""
    frozen = load_frozen(ctx, card_id)
    if frozen is None:
        raise RuntimeError(f"{card_id}: a check ran before any statement was frozen")
    base = {"card_id": card_id, "stage": stage, "statement_id": statement["id"], "form": statement["form"],
            "purpose": purpose, "attempt": attempt, "prover_call": prover_call, "statement_sha256": statement["sha256"],
            "text": text, "reasons": [], "counterexample": None, "job_ids": [], "check_hashes": {}, "runs": [],
            "tier": mathrecords.FORM_TIER[statement["form"]], "prover_fault": False}
    unchanged = (history.unchanged(statement, text) if history is not None and statement["form"] == "learnability"
                 else None)
    if skipped:
        check = dict(base, status="unknown", decided=False, reasons=[skipped],
                     evidence=f"not run (status unknown: not a pass and not a counterexample): {skipped}"[:2000])
    elif refused:
        check = dict(base, status="unknown", decided=False, prover_fault=True, reasons=[refused], precheck=[refused],
                     evidence=("refused before any executor job (status unknown: not a pass and not a counterexample): "
                               + refused)[:2000])
    elif statement["form"] == "lean":
        if mc.proof_redeclares_claim(text or ""):
            check = dict(base, status="reject", decided=True, evidence="the proof redeclares Claim; statements are "
                         "frozen and the proof may only supply theorem claim_holds : Claim",
                         reasons=["proof redeclares Claim"])
        else:
            try:
                result = await asyncio.to_thread(ctx.math.check_lean, lean_check.LeanCheck(
                    statement["spec"], statement["sha256"], text or ""))
            except manifest.ManifestMismatch as exc:
                raise StageError(f"math check files differ from MANIFEST.json: {exc}") from exc
            runs = [{"job_id": r.job_id, "status": r.receipt_status, "detail": "; ".join(r.reasons)[:300],
                     "exit_code": r.exit_code, "run_s": r.run_s, "facts": r.facts, "output_tail": r.stdout_tail}
                    for r in result.runs]
            infra = [r for r in result.runs if r.status == "infra_error"]
            if infra:
                raise StageError(f"executor infra_error in Lean check {statement['id']}: {infra[0].reasons}")
            evidence = (f"Lean: claim_holds : Claim checked twice, axioms {result.axioms}" if result.status == "pass"
                        else "Lean rejected: " + "; ".join(result.reasons))
            check = dict(base, status=result.status, decided=result.decided, evidence=evidence[:2000],
                         reasons=result.reasons, runs=runs, job_ids=[r.job_id for r in result.runs],
                         check_hashes=result.check_hashes, axioms=result.axioms)
    elif unchanged:
        evidence = ("resource failure (status unknown: not a pass and not a counterexample): the check was not run: "
                    + unchanged)
        check = dict(base, status="unknown", decided=False, evidence=evidence, reasons=[unchanged],
                     resource={"kind": "unchanged_source", "reason": unchanged})
    elif text is not None and source_problems(statement["form"], text):
        problems = source_problems(statement["form"], text)
        evidence = ("refused before any executor job (status unknown: not a pass and not a counterexample); fix these "
                    "and send the source again: " + "; ".join(problems))
        check = dict(base, status="unknown", decided=False, evidence=evidence[:2000], reasons=problems,
                     prover_fault=True, precheck=problems)
    else:
        request = library_request(dict(statement, spec=spec_override or statement["spec"]), text,
                                  architecture_contract(frozen))
        admit = history.admit(statement) if history is not None and statement["form"] == "learnability" else None
        try:
            record = await asyncio.to_thread(ctx.math.check_library, request, admit=admit)
        except manifest.ManifestMismatch as exc:
            raise StageError(f"math check files differ from MANIFEST.json: {exc}") from exc
        printed = (record["runs"][0].get("printed") or {}) if record["runs"] else {}
        check = dict(base, status=record["status"], decided=record["decided"], evidence=record["evidence"],
                     reasons=record["problems"], counterexample=record["counterexample"], runs=record["runs"],
                     job_ids=[run["job_id"] for run in record["runs"]], check_hashes=record["check_hashes"],
                     prover_fault=bool(printed.get("prover_fault") or record.get("prover_fault")),
                     bound=printed.get("bound"),
                     details={k: printed[k] for k in DETAIL_KEYS if k in printed})
        reported = printed or _reported_tail(record["runs"][0] if record["runs"] else {})
        if not check["evidence"] and reported.get("evidence"):
            # A job that exits non-zero (the driver reports a crash of the source as unknown and exits 1) has its
            # report only in the output tail; card k2's first cost attempt in the 2026-10-13 smoke read as an empty
            # evidence with "first run candidate_error".
            check.update(evidence=f"the check printed: {reported['evidence']}"[:2000],
                         prover_fault=check["prover_fault"] or bool(reported.get("prover_fault")))
        if record.get("probe") is not None:
            # Recorded apart from the deciding runs: the probe never discharges anything.
            check.update(probe={k: v for k, v in record["probe"].items() if k != "run"},
                         resource=record.get("resource"))
        _raise_on_infra(check)
        if record.get("probe") is not None and record["probe"]["status"] == "infra_error":
            raise StageError(f"executor infra_error in the timing probe of {statement['id']}: "
                             f"{record['probe']['detail']}")
    if history is not None and check.get("resource"):
        history.note(statement, check, text)
    return check


def record_check(ctx: StageContext, card_id: str, stage: str, statement: dict, check: dict) -> dict:
    """Store a check from ``execute_check`` as a ``math_check`` artifact with its ``math_probe`` and ``math_check``
    events; returns the check with its artifact name."""
    name = ctx.artifact_name(card_id)
    ctx.put_json_artifact("math_check", name, check, card_id)
    if check.get("probe"):
        probe = check["probe"]
        ctx.emit("math_probe", {"card_id": card_id, "stage": stage, "statement_id": statement["id"],
                                "attempt": check["attempt"], "job_id": probe["job_id"], "status": probe["status"],
                                "run_s": probe["run_s"], "projected_s": probe["projected_s"], "cap_s": probe["cap_s"],
                                "fit_s": probe["fit_s"], "artifact": name, "decides": False})
    ctx.emit("math_check", {"card_id": card_id, "stage": stage, "statement_id": statement["id"],
                            "purpose": check["purpose"], "status": check["status"], "decided": check["decided"], "artifact": name,
                            "job_ids": check["job_ids"],
                            **({"resource": check["resource"]["kind"]} if check.get("resource") else {})})
    return dict(check, artifact=name)


# ----- checks run together ----------------------------------------------------------------------------------------

def check_group(statement: dict, history: "ResourceHistory | None") -> str:
    """Checks with the same group run one after another, in statement order; different groups run at once. A
    learnability statement's group is its frozen experiment (ResourceHistory.key), so a second source for the same
    experiment still sees the first one's resource failure (the FIX-2 refusal and cost admission), as when every
    check ran in turn; any other statement is its own group."""
    if history is not None and statement["form"] == "learnability":
        return "experiment:" + history.key(statement)
    return "statement:" + statement["id"]


async def execute_checks(ctx: StageContext, card_id: str, stage: str, requests: list[dict],
                         history: "ResourceHistory | None" = None) -> list[dict | BaseException | None]:
    """Execute several checks concurrently, without recording them. ``requests`` are the keyword arguments of
    ``execute_check`` with ``statement``, in statement order. The executor's host-wide job slots bound how many jobs
    run at once. Returns, per request, the check, the exception it raised, or None when an earlier check of its group
    raised (that check was not run, as when the checks ran in turn)."""
    results: list[dict | BaseException | None] = [None] * len(requests)
    groups: dict[str, list[int]] = {}
    for index, request in enumerate(requests):
        groups.setdefault(check_group(request["statement"], history), []).append(index)

    async def run_group(indices: list[int]) -> None:
        for index in indices:
            request = dict(requests[index])
            statement = request.pop("statement")
            try:
                results[index] = await execute_check(ctx, card_id, stage, statement, history=history, **request)
            except Exception as exc:  # re-raised by the caller in statement order (record_in_order)
                results[index] = exc
                return

    await asyncio.gather(*(run_group(indices) for indices in groups.values()))
    return results


def record_in_order(ctx: StageContext, card_id: str, stage: str, statements: list[dict],
                    results: list[dict | BaseException | None], on_recorded=None) -> list[dict]:
    """Record executed checks in statement order (``on_recorded(index, check)`` after each, for the caller's events).
    The first failed check stops the recording and its exception is raised, so the ledger holds what the serial loop
    would have written before the same failure."""
    recorded = []
    for index, (statement, result) in enumerate(zip(statements, results)):
        if isinstance(result, BaseException):
            raise result
        if result is None:
            raise RuntimeError(f"{statement['id']}: check not run although no earlier check of its group failed")
        check = record_check(ctx, card_id, stage, statement, result)
        recorded.append(check)
        if on_recorded is not None:
            on_recorded(index, check)
    return recorded


def _reported_tail(run: dict) -> dict:
    """The driver's report from the last line of a run's output tail, when it is a whole JSON object."""
    lines = [line for line in (run.get("stdout_tail") or "").splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        report = json.loads(lines[-1])
    except ValueError:
        return {}
    return report if isinstance(report, dict) else {}


def source_problems(form: str, text: str) -> list[str]:
    """What the checkers would refuse or crash on, found on the host without a job: the source audit, names absent
    from the form's sandbox (cost.sandbox_problems) and, for a learner, a missing RECIPE. Lean proofs and statements
    checked without a prover are not sources."""
    if form not in cost.SANDBOX_GLOBALS:
        return []
    problems = cost.audit_source(text) + cost.sandbox_problems(text, form)
    if form == "learnability":
        problems += learnability.recipe_problems(text)
    return list(dict.fromkeys(problems))


def holds(check: dict | None) -> bool:
    return bool(check) and check["decided"] and check["status"] in ("pass", "discharged")


# ----- learnability experiments that stopped for their cost -------------------------------------------------------

# After an equivalent experiment ran out of its cap, a new source's probe must project at most this share of the
# projection that experiment's probe made: that projection fell short of the real cost, so the new one must beat it by
# the margin the probe leaves (learnability.PROBE_FIT) to be expected to fit.
COST_IMPROVEMENT = learnability.PROBE_FIT


def _sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class ResourceHistory:
    """Learnability experiments of one card that stopped for their cost in the prover rounds, by experiment: the form,
    the frozen statement hash and the hashes of the definitions it uses. Two claims can freeze the same experiment:
    in the 2026-10-13 calibration card k1's O9.a (sample efficiency) and O10.a (P4.copy) had the same statement and
    definitions, the prover wrote both learners in one reply, and the second, as slow as the first, was queued behind
    the first's 1800 s timeout (results/phase4-20261013T073748Z/p4-timing/REPORT.md)."""

    def __init__(self, record: dict):
        self.record = record
        self.definition_hashes = {d["id"]: d["sha256"] for d in record["definitions"]}
        self.stopped: dict[str, dict] = {}

    def key(self, statement: dict) -> str:
        used = obligation_of(self.record, statement["id"])["definitions"]
        return json.dumps([statement["form"], statement["sha256"], sorted(self.definition_hashes[d] for d in used)])

    def unchanged(self, statement: dict, text: str | None) -> str | None:
        """Why this exact source is refused without a job: it already stopped for its cost on this experiment."""
        entry = self.stopped.get(self.key(statement))
        if entry is None or _sha(text) not in entry["sources"]:
            return None
        return (f"this exact source already stopped for its cost on the same frozen experiment ({entry['first']}: "
                f"{entry['kind']}); it would cost the same again. Send a source with a lower measured cost.")

    def admit(self, statement: dict):
        """The runner's ``admit`` for a new source: after a full job of this experiment ran out of its cap, the new
        source's timing probe must project at most COST_IMPROVEMENT of what that job's probe projected."""
        entry = self.stopped.get(self.key(statement))
        if entry is None or entry["timeout_projected_s"] is None:
            return None
        old = entry["timeout_projected_s"]

        def admit(probe: dict) -> str | None:
            if probe["projected_s"] <= COST_IMPROVEMENT * old:
                return None
            return (f"an equivalent frozen experiment ({entry['timeout_statement']}: the same statement and "
                    f"definitions) ran out of its {probe['cap_s']:.0f} s cap although its timing probe projected "
                    f"{old:.0f} s; this source's probe projects {probe['projected_s']:.0f} s, not below "
                    f"{COST_IMPROVEMENT:.0%} of that, so the same timeout is expected")
        return admit

    def note(self, statement: dict, check: dict, text: str | None) -> None:
        resource = check["resource"]
        entry = self.stopped.setdefault(self.key(statement), {
            "first": f"{statement['id']} attempt {check['attempt']}", "kind": resource["kind"], "sources": set(),
            "timeout_projected_s": None, "timeout_statement": None})
        entry["sources"].add(_sha(text))
        if resource["kind"] == "timeout" and resource.get("projected_s") is not None:
            if entry["timeout_projected_s"] is None or resource["projected_s"] < entry["timeout_projected_s"]:
                entry.update(timeout_projected_s=resource["projected_s"],
                             timeout_statement=f"{statement['id']} attempt {check['attempt']}")


# ----- prover -----------------------------------------------------------------------------------------------------

def definitions_of(record: dict, obligation: dict) -> list[dict]:
    """The frozen definitions an obligation uses, as shown to the prover and the statement-match role."""
    by_id = {d["id"]: d for d in record["definitions"]}
    return [{"definition_id": did, "kind": by_id[did]["kind"], "text": by_id[did]["text"]}
            for did in obligation["definitions"]]


def prover_user(card: dict, record: dict, todo: list[dict], failures: dict[str, str], caps=None,
                sources: dict[str, str] | None = None, history: dict[str, list[str]] | None = None,
                best: BestAttempts | None = None) -> str:
    """The prover's message for one call. ``caps`` are the runner's library caps (MathRunner.library_caps; the
    default is LIBRARY_CAPS). A learnability statement always brings the learner budget, whether or not it is a
    profile slot: in the 2026-10-13 calibration the learner contract reached the prover only through the profile
    section. ``sources`` holds each statement's last submitted source or proof and ``history`` its compact attempt
    history across rounds: in the 2026-10-13 smoke card k1's P1 alternated between an operation fix and a state fix
    for four attempts because each round saw only the last failure. ``best`` (FIX-5) adds each learnability
    statement's best measured attempt of this run with its full source (prover_memory.BestAttempts)."""
    text = _prover_user(card, record, todo, failures, sources or {}, history or {}, best)
    if any(profile.is_slot(obligation_of(record, s["id"])["claim_id"]) for s in todo):
        text += "\n\n" + load_section("profile_prover")
    contracts = task_contracts(todo)
    if contracts:
        text += "\n\n" + contracts
    if any(s["form"] == "learnability" for s in todo):
        caps = caps or math_runner.LIBRARY_CAPS
        text += "\n\n" + learnability.budget_text(caps.wall_s, caps.memory_mb, caps.cpus, profile.P4_SEEDS)
        text += "\n\n" + RECIPE_NOTE
        # FIX-5: new section file; the best attempt, the passing-recipe rule and the learner's CURVE.
        text += "\n\n" + load_section("recipe_carry")
    if any(s["form"] == "construction" for s in todo):
        # FIX-6: new section file; the frozen contract is the authority, and the construction memory.
        text += "\n\n" + load_section("construction_contract")
        # FIX-7: new section file; attempts are ranked against the query-independent baseline.
        text += "\n\n" + load_section("construction_baseline")
    if any(audits_of(record, s["id"]) for s in todo):
        # FIX-6: new section file; the conformance audit the reference will meet.
        text += "\n\n" + load_section("conformance_audit")
    if any(s["form"] != "lean" for s in todo):
        text += "\n\n" + sandbox_note()
    contract = architecture_contract(record)
    if contract is not None and any(isinstance(s["spec"], dict) and s["spec"].get("widths") for s in todo):
        # FIX-4: new section file; the shared contract the checks apply, as frozen.
        text += ("\n\n" + load_section("profile_architecture") + "\nFROZEN ARCHITECTURE CONTRACT: "
                 + json.dumps(contract, sort_keys=True))
    return text


RECIPE_NOTE = ("TRAINING RECIPE. A learnability source sets RECIPE = {...} at module level: the training recipe you "
               "choose (initialization, optimizer, learning rate, epochs or passes, batch size), within the card's "
               "learning rule, the frozen model class and the budget above. The frozen statement fixes m, the seeds, "
               "the threshold and both controls; the harness applies the same learn and the same RECIPE to the true "
               "data and to both controls, records the recipe with the result, and stops the check if RECIPE changes "
               "during a training. Choose a recipe that you expect to reach the threshold, and change it between "
               "attempts when the reported accuracy says it does not learn.")


def task_contracts(todo: list[dict]) -> str:
    """The host-generated contracts of the batch's executed statements: the counting convention for cost statements
    and, for each construction or learnability statement, its task measured on the instances its check runs."""
    parts = []
    if any(s["form"] == "cost" for s in todo):
        parts.append(cost.counting_text())
    for statement in todo:
        spec = statement["spec"]
        try:
            if statement["form"] == "construction":
                parts.append(f"For {statement['id']}: " + constructions.task_contract(
                    spec["task"], spec["sizes"], spec.get("params"),
                    count=spec.get("count") or constructions.DEFAULT_COUNT, seed=spec.get("seed") or 0))
            elif statement["form"] == "learnability":
                parts.append(f"For {statement['id']}: " + learnability.task_contract(spec))
        except ValueError as exc:
            # A statement frozen before the host checked its layout (FIX-3); the check will refuse it the same way.
            parts.append(f"For {statement['id']}: the harness cannot build this task's layout: {exc}")
    return "\n\n".join(parts)


def sandbox_note() -> str:
    """What the checkers' namespaces hold, read from the check files so the prover is told what actually runs."""
    names = ", ".join(sorted(name for name in cost.SAFE_BUILTINS if not name.startswith("_")))
    return ("SANDBOX. Simulation, cost and learnability sources run with only these Python builtins: " + names
            + ". No other builtin exists there: print, str, set and map are absent, and class statements do not "
            "work. Every source is checked for absent names before a job runs, and a source that uses one comes "
            "back as a failed attempt naming the line. Construction sources run with the standard builtins. `np` is NumPy in simulation, construction and learnability "
            "sources. A cost source gets only `ops.np`, the counted subset: it has no `random`; use "
            "`ops.np.normal(shape, seed, scale)` for seeded random values or set parameters to fixed values.")


# Most characters of a previous source shown again to the prover.
PREVIOUS_SOURCE_CHARS = 16000


def _prover_user(card: dict, record: dict, todo: list[dict], failures: dict[str, str], sources: dict[str, str],
                 history: dict[str, list[str]], best: BestAttempts | None = None) -> str:
    shown = []
    for statement in todo:
        obligation = obligation_of(record, statement["id"])
        item = {"statement_id": statement["id"], "form": statement["form"], "claim_id": obligation["claim_id"],
                "claim_quote": obligation["claim_quote"], "frozen_statement": statement["spec"],
                "frozen_definitions": definitions_of(record, obligation)}
        if statement["id"] in failures:
            item["previous_attempt_failed"] = failures[statement["id"]]
        if history.get(statement["id"]):
            item["attempt_history"] = history[statement["id"]]
        block = best.prompt_block(statement["id"]) if best is not None else None
        if block is not None:
            item["best_attempt"] = block
        if sources.get(statement["id"]) and (block is None or block["source"] != sources[statement["id"]]):
            item["previous_source"] = sources[statement["id"]][:PREVIOUS_SOURCE_CHARS]
        if statement["form"] == "construction":
            item["frozen_contract"] = frozen_contract(statement, architecture_contract(record))
            ranking = best.ranking_block(statement["id"]) if best is not None else None
            if ranking is not None:
                # FIX-7: the top attempts against the query-independent baseline; a shortcut is never best_attempt.
                item["attempts_vs_baseline"] = ranking
        audits = audits_of(record, statement["id"])
        if audits:
            item["conformance_audit"] = [{"statement_id": a["id"], **{k: a["spec"].get(k) for k in (
                "equations", "points", "streams", "tolerance", "precision_bits")}} for a in audits]
        shown.append(item)
    return ("CARD (for the mechanism; the statements below are frozen and are what you must prove or run):\n"
            + card_for_prompt(card) + "\n\nFROZEN STATEMENTS:\n" + json.dumps(shown, indent=1, ensure_ascii=False))


def audits_of(record: dict, statement_id: str) -> list[dict]:
    """The frozen conformance statements whose reference is ``statement_id`` (FIX-6)."""
    return [s for s in all_statements(record) if s["form"] == "conformance" and s["spec"]["reference"] == statement_id]


async def audit_conformance(ctx: StageContext, card_id: str, record: dict, checks: dict[str, dict]) -> None:
    """Run every conformance statement against the source that held for its reference statement and add the checks
    to ``checks`` (step 5b). A reference that did not hold leaves its audit unrun and unknown."""
    audits = [s for s in all_statements(record) if s["form"] == "conformance"]
    if not audits:
        return
    requests = []
    for statement in audits:
        reference = statement["spec"]["reference"]
        held = checks.get(reference)
        if holds(held):
            requests.append({"statement": statement, "purpose": "conformance", "text": held["text"]})
        else:
            requests.append({"statement": statement, "purpose": "conformance",
                             "skipped": f"the reference statement {reference} did not hold, so there is no reference "
                                        "source to audit"})
    results = await execute_checks(ctx, card_id, STAGE, requests)
    for check in record_in_order(ctx, card_id, STAGE, audits, results):
        checks[check["statement_id"]] = check


# Receipt statuses of a run that stopped for its resources or the executor, named in the prover's feedback.
STOPPED_RUN = ("timeout", "oom", "pids", "infra_error")


def failure_text(check: dict) -> str:
    """What the prover reads about its failed attempt: the evidence, the checker's reasons when the evidence does not
    hold them, the structured counterexample (the failing size or point, the input, expected against got) before any
    output tail, the first failure of every failing cost check, where a crash was raised, a run the executor stopped
    (status, time, receipt detail) and the timing probe's projection. In the 2026-10-13 calibration a learnability
    timeout had empty evidence and its 1800 s reason only in ``reasons``; in the smoke that followed the cost feedback
    was the tail of a long JSON report, which named ops_exact but not the 357696 against 672768 at the first point."""
    evidence = check["evidence"] or "no report was printed"
    parts = [f"status {check['status']}: {evidence}"]
    reasons = [r for r in check.get("reasons") or [] if r and r not in evidence]
    if reasons:
        parts.append("checker reasons: " + "; ".join(reasons))
    counterexample = check.get("counterexample")
    if counterexample is not None:
        parts.append("COUNTEREXAMPLE: " + counterexample_text(check["form"], counterexample))
    details = check.get("details") or {}
    failed = details.get("failed_checks") or {}
    others = {name: f for name, f in failed.items() if f != counterexample}
    if others:
        parts.append("OTHER FAILING CHECKS (first failure of each): " + _compact(others, 1200))
    crash = details.get("crash")
    if crash:
        parts.append("CRASH: " + _compact(crash, 600))
    for run in check["runs"][:1]:
        if run.get("status") in STOPPED_RUN:
            took = f" after {run['run_s']:.1f} s" if run.get("run_s") is not None else ""
            parts.append(f"job {run['job_id']}: executor status {run['status']}{took}: {run.get('detail') or ''}")
        tail = run.get("output_tail") or run.get("stdout_tail") or ""
        if tail and counterexample is None:
            parts.append("output tail:\n" + tail[-1500:])
    probe = check.get("probe")
    if probe and probe.get("projected_s") is not None and not check.get("resource"):
        parts.append(f"timing probe (decides nothing): projected full check {probe['projected_s']:.0f} s against the "
                     f"{probe['cap_s']:.0f} s cap")
    return "\n".join(parts)[:6000]


def _compact(value, limit: int) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _first_error_text(error: dict) -> str:
    tokens = error.get("tokens")
    shown = ""
    if tokens is not None:
        more = error.get("tokens_total", len(tokens)) - len(tokens)
        shown = f"; input tokens {tokens}" + (f" (+{more} more)" if more > 0 else "")
    where = f"instance {error['instance']}, " if error.get("instance") is not None else ""
    return (f"{where}after reading the token at position {error.get('position')}, predict returned "
            f"{error.get('predicted')} but the answer is {error.get('expected')}{shown}")


def counterexample_text(form: str, counterexample: dict) -> str:
    """A counterexample in words, with its record: the failing size or point, the input, expected against got."""
    if not isinstance(counterexample, dict):
        return _compact(counterexample, 1500)
    record = _compact({k: v for k, v in counterexample.items() if k != "first_error"}, 800)
    if form == "cost" and counterexample.get("check") in ("ops_exact", "state_exact"):
        what = "operations" if counterexample["check"] == "ops_exact" else "peak state words"
        line = (f"{counterexample['check']} at {counterexample.get('point')}: counted {what} "
                f"{counterexample.get('counted')}, formula {counterexample.get('formula')} (difference "
                f"{counterexample.get('difference')})")
        if counterexample.get("counted_by_kind"):
            line += f"; counted by kind {counterexample['counted_by_kind']}"
        return line + f". Record: {record}"
    error = counterexample.get("first_error")
    if form == "construction":
        line = (f"{counterexample.get('task')} at size {counterexample.get('size')}: accuracy "
                f"{counterexample.get('accuracy')} against the threshold {counterexample.get('threshold')}")
        if counterexample.get("check") == "state_words":
            line = (f"{counterexample.get('task')} at size {counterexample.get('size')}: peak state words "
                    f"{counterexample.get('state_peak_words')} above the declared "
                    f"{counterexample.get('declared_state_words')}")
        return line + (f"; first error: {_first_error_text(error)}" if error else "") + f". Record: {record}"
    if form == "learnability":
        line = (f"seed {counterexample.get('seed')}: {counterexample.get('reason')}, accuracy "
                f"{counterexample.get('accuracy')}, control accuracy {counterexample.get('control_accuracy')} at "
                f"{counterexample.get('task')} size {counterexample.get('size')} from m={counterexample.get('examples')}")
        return line + (f"; first test error: {_first_error_text(error)}" if error else "") + f". Record: {record}"
    return _compact(counterexample, 1500)


def attempt_line(attempt: int, check: dict | None, note: str | None = None) -> str:
    """One line of a statement's attempt history: what the attempt measured, kept across rounds so a later crash does
    not erase an earlier decided measurement."""
    head = f"attempt {attempt}: "
    if check is None:
        return head + (note or "nothing was checked")
    status = check["status"] + (" (prover fault)" if check.get("prover_fault") else "")
    details = check.get("details") or {}
    ce = check.get("counterexample")
    if check.get("resource"):
        return head + f"{status}, resource failure {check['resource']['kind']}"
    if check["form"] == "cost" and isinstance(ce, dict):
        failed = details.get("failed_checks") or {ce.get("check"): ce}
        bits = []
        for name, f in failed.items():
            if name in ("ops_exact", "state_exact"):
                bits.append(f"{name} at {f.get('point')}: counted {f.get('counted')} vs formula {f.get('formula')}")
            else:
                bits.append(name)
        return head + status + ": " + "; ".join(bits)
    rows = details.get("results") or []
    if check["form"] == "construction" and rows:
        return head + status + ": " + ", ".join(
            f"size {r['size']} accuracy {r['accuracy']:.4f}, peak {r['state_peak_words']} words"
            + (f" (declared {r['declared_state_words']:g})" if r.get("declared_state_words") is not None else "")
            for r in rows)
    if check["form"] == "learnability" and rows:
        recipe = rows[0].get("recipe")
        return head + status + ": " + ", ".join(
            f"seed {r['seed']} accuracy {r['accuracy']:.4f} control {r['control_accuracy']:.4f}" for r in rows) + (
            f"; recipe {_compact(recipe, 300)}" if recipe else "") + (
            f"; then: {(check.get('evidence') or '')[:300]}" if details.get("crash") else "")
    return head + status + ": " + (check.get("evidence") or "")[:300]


# What each form needs from the prover, for the feedback after an attempt that sent nothing.
WANTED = {"lean": "a Lean proof defining theorem claim_holds : Claim",
          "simulation": "a source defining run(given)",
          "construction": "a source defining init(layout), update(state, token) and predict(state)",
          "cost": "a source defining init(sizes, ops), update(params, state, token, ops) and predict(params, state, ops)",
          "learnability": "a learner source defining learn(layout, examples, seed), init(layout, params), "
                          "update(state, token) and predict(state)"}


def no_text_failure(form: str, limit: int) -> str:
    """Feedback for an attempt with neither proof nor source. In the 2026-10-13 calibration (card k2) the prover left
    both learners null on all four attempts, saying it doubted the frozen SGD schedule would succeed, and was told
    only that nothing was given."""
    return (f"no proof or source was given for this statement, so nothing was checked and one of its {limit} attempts "
            f"was used. Send {WANTED[form]}. Write the most faithful version of the frozen statement and definitions "
            "even if you expect the check to fail: the check reports how far it gets, while a null attempt leaves the "
            "claim unproven.")


def round_failure(check: dict | None, text: str | None):
    """What a failed attempt is compared on for stall detection: its counterexample when it has one, else the proof or
    source it submitted. A prover that sends the same failing proof again repeats the round exactly; a new proof that
    also fails is progress, bounded by math.max_prover_attempts."""
    if check is not None and check.get("counterexample") is not None:
        return {"counterexample": check["counterexample"]}
    return {"submitted": text}


# Most statements per prover call. With the profile on, a card has 12 to 16 prover statements; one reply holding every
# proof and program ran Claude Opus past its 64000-token cap (finish_reason=length) in 7 of its 9 prover calls in the
# 2026-10-13 calibration, and the fallback model then declined every construction. Each hand-set construction
# obligation now gets a call of its own and other statements share a call with at most this many of their form.
PROVER_BATCH = 4


def prover_batches(record: dict, todo: list[dict]) -> list[list[dict]]:
    """One round's prover calls: each obligation with a construction alone, the other obligations grouped by the form
    of their first open statement, at most PROVER_BATCH statements per call; an obligation is never split."""
    by_obligation: dict[str, list[dict]] = {}
    for statement in todo:
        by_obligation.setdefault(obligation_of(record, statement["id"])["id"], []).append(statement)
    batches: list[list[dict]] = []
    open_batch: dict[str, list[dict]] = {}
    for statements in by_obligation.values():
        if any(s["form"] == "construction" for s in statements):
            batches.append(list(statements))
            continue
        form = statements[0]["form"]
        batch = open_batch.get(form)
        if batch is None or len(batch) + len(statements) > PROVER_BATCH:
            batch = []
            batches.append(batch)
            open_batch[form] = batch
        batch.extend(statements)
    return batches


class InvalidReply:
    """A prover call whose reply was not valid JSON for the schema after the client's retry."""

    def __init__(self, exc: StageError):
        self.call_id = getattr(exc.__cause__, "call_id", None)
        self.error = str(exc)


def _invalid_reply(exc: BaseException) -> bool:
    from hybrid.llm.errors import SchemaValidationError
    return isinstance(exc, StageError) and (isinstance(exc.__cause__, SchemaValidationError)
                                            or "no parsed JSON" in str(exc))


class Exhausted:
    """A prover call whose every backend reply spent the whole output cap on reasoning and returned no text, after the
    client's one retry at lower effort (hybrid.llm.OutputExhausted). In the 2026-10-13 smoke (k2-run1, O7.a) Claude
    Opus and Sonnet both did so at 64000 tokens on one construction statement and the card-run ended with a stage
    error (FIX-5); now the batch is split once, and a statement whose call is still exhausted has a failed attempt."""

    def __init__(self, exc: StageError):
        self.call_id = getattr(exc.__cause__, "call_id", None)
        self.error = str(exc)


def _exhausted(exc: BaseException) -> bool:
    from hybrid.llm.errors import OutputExhausted
    return isinstance(exc, StageError) and isinstance(exc.__cause__, OutputExhausted)


async def _ask_prover(ctx: StageContext, user: str, ids: list[str]):
    try:
        return await ctx.ask("prover", "prover", user, mc.prover_schema(ids))
    except StageError as exc:
        if _invalid_reply(exc):
            return InvalidReply(exc)
        if _exhausted(exc):
            return Exhausted(exc)
        raise


def split_batch(batch: list[dict]) -> list[list[dict]]:
    """A batch in two halves, in order; the retry after an exhausted call asks for fewer statements per call."""
    half = (len(batch) + 1) // 2
    return [batch[:half], batch[half:]]


async def _ask_round(ctx: StageContext, card_id: str, batches: list[list[dict]], user_of) -> list[tuple]:
    """One round's prover calls. A call of several statements that comes back Exhausted is split in two and each half
    asked once more in the same round (event ``prover_split``); the exhausted call is returned as (batch, reply,
    True) so its call id is recorded without charging an attempt. Any other failure stops the stage as before."""
    async def ask(group: list[list[dict]]) -> list:
        replies = await asyncio.gather(*(_ask_prover(ctx, user_of(batch), [s["id"] for s in batch])
                                         for batch in group), return_exceptions=True)
        errors = [r for r in replies if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        return replies

    pairs, halves = [], []
    for batch, reply in zip(batches, await ask(batches)):
        if isinstance(reply, Exhausted) and len(batch) > 1:
            parts = split_batch(batch)
            ctx.emit("prover_split", {"card_id": card_id, "stage": STAGE, "call_id": reply.call_id,
                                      "statements": [s["id"] for s in batch],
                                      "into": [[s["id"] for s in part] for part in parts], "error": reply.error[:1000]})
            pairs.append((batch, reply, True))
            halves.extend(parts)
        else:
            pairs.append((batch, reply, False))
    if halves:
        pairs.extend((batch, reply, False) for batch, reply in zip(halves, await ask(halves)))
    return pairs


def exhausted_failure(limit: int, split: bool) -> str:
    """Feedback for a statement whose prover call used the whole output limit on reasoning and returned no text."""
    return ("your previous reply used the whole output limit on reasoning and returned no text"
            + (", also when asked for fewer statements per call" if split else "") + ", so nothing was checked "
            f"and one of its {limit} attempts was used. Reason less before writing: send the most direct faithful "
            "proof or source first, keep notes brief, and leave refinements to the next round, which will show you "
            "what the check measured.")


async def prove(ctx: StageContext, card_id: str, card: dict, record: dict,
                checks: dict[str, dict]) -> tuple[list[str], set[str], list[lineage.LoopStop]]:
    """Prover rounds for the Lean and executed statements. Updates ``checks`` with the latest check of each.
    Returns the call ids, the models that answered, and the stop that ended the rounds while statements still failed
    (``stall:prover`` when a round repeated the previous round's failures exactly, ``budget:prover`` when the attempts
    ran out), recorded as ``loop_stop`` events."""
    limit = ctx.cfg["math"]["max_prover_attempts"]
    used = Counter(e["statement_id"] for e in ctx.card_events(card_id, "proof_attempt"))
    pending = [s for s in all_statements(record) if s["form"] in mathrecords.PROVER_FORMS
               and not holds(checks.get(s["id"]))]
    failures = {sid: failure_text(c) for sid, c in checks.items() if not holds(c)}
    calls, models, stops = [], set(), []
    history = ResourceHistory(record)
    # Per statement, across rounds: the last source or proof sent and one line per attempt.
    sources: dict[str, str] = {}
    attempts: dict[str, list[str]] = {}
    # Per learnability statement: the best measured attempt with its full source, and any that passed every seed.
    best = BestAttempts()
    while True:
        todo = [s for s in pending if used[s["id"]] < limit and not holds(checks.get(s["id"]))]
        if not todo:
            still = [s["id"] for s in pending if not holds(checks.get(s["id"]))]
            if still:
                stops.append(lineage.LoopStop(lineage.PROVER, "budget", f"{limit} attempt(s) used on "
                                              + ", ".join(still)))
            break
        batches = prover_batches(record, todo)
        pairs = await _ask_round(ctx, card_id, batches, lambda batch: prover_user(
            card, record, batch, failures, ctx.math.library_caps, sources, attempts, best))
        split_ids = {s["id"] for batch, _, was_split in pairs if was_split for s in batch}
        round_failures = {}
        # The round in two passes. The first walks the replies in order, counts the attempts and collects the checks
        # to run; the checks then run concurrently under the executor's job slots; the second walks the same order
        # and writes every artifact and event, so the ledger, the decisions and the next round's prompts are the same
        # whichever check finished first.
        plan: list[tuple] = []
        requests: list[dict] = []
        for batch, reply, was_split in pairs:
            ids = [s["id"] for s in batch]
            if was_split or isinstance(reply, (Exhausted, InvalidReply)):
                plan.append(("reply", batch, reply, was_split))
                if not was_split:
                    for statement in batch:
                        used[statement["id"]] += 1
                        plan.append(("no_check", statement, reply, used[statement["id"]]))
                continue
            plan.append(("reply", batch, reply, False))
            answered = {}
            for entry in reply.parsed["attempts"]:
                answered.setdefault(entry["statement_id"], entry)
            for statement in batch:
                sid = statement["id"]
                used[sid] += 1
                entry = answered.get(sid)
                text = None if entry is None else (entry["proof"] if statement["form"] == "lean" else entry["source"])
                if not text:
                    plan.append(("no_check", statement, reply, used[sid]))
                    continue
                reason = entry.get("recipe_change_reason")
                refused = (best.discarded_without_reason(statement, text, reason)
                           if statement["form"] == "learnability" else None)
                plan.append(("check", statement, reply, used[sid], text, reason, refused, len(requests)))
                requests.append({"statement": statement, "purpose": "prove", "text": text, "attempt": used[sid],
                                 "prover_call": reply.call_id, "refused": refused})
        results = await execute_checks(ctx, card_id, STAGE, requests, history)
        for item in plan:
            if item[0] == "reply":
                _, batch, reply, was_split = item
                ids = [s["id"] for s in batch]
                purpose = f"attempt for {', '.join(ids)}"
                if was_split or isinstance(reply, Exhausted):
                    if reply.call_id:
                        calls.append(reply.call_id)
                    ctx.emit("math_role_call", {"card_id": card_id, "stage": STAGE, "role": "prover",
                                                "purpose": purpose, "call_id": reply.call_id, "model": None,
                                                "output_exhausted": reply.error[:2000],
                                                **({"split": True} if was_split else {})})
                elif isinstance(reply, InvalidReply):
                    if reply.call_id:
                        calls.append(reply.call_id)
                    ctx.emit("math_role_call", {"card_id": card_id, "stage": STAGE, "role": "prover",
                                                "purpose": purpose, "call_id": reply.call_id, "model": None,
                                                "invalid_reply": reply.error[:2000]})
                else:
                    role_call(ctx, card_id, STAGE, "prover", purpose, reply)
                    calls.append(reply.call_id)
                    models.add(reply.model)
                continue
            if item[0] == "no_check":
                _, statement, reply, attempt = item
                sid = statement["id"]
                if isinstance(reply, Exhausted):
                    failures[sid] = exhausted_failure(limit, sid in split_ids)
                    note, status = "the reply used the whole output limit on reasoning; nothing was sent", \
                        "output_exhausted"
                    round_failures[sid] = {"output_exhausted": True}
                elif isinstance(reply, InvalidReply):
                    failures[sid] = ("your previous reply could not be used, so no proof or source was checked: "
                                     + reply.error[:1500])
                    note, status = "the reply could not be used", "invalid_reply"
                    round_failures[sid] = {"invalid_reply": reply.error}
                else:
                    failures[sid] = no_text_failure(statement["form"], limit)
                    note, status = "no proof or source was sent", "no_text"
                    round_failures[sid] = round_failure(None, None)
                attempts.setdefault(sid, []).append(attempt_line(attempt, None, note))
                ctx.emit("proof_attempt", {"card_id": card_id, "statement_id": sid, "attempt": attempt,
                                           "call_id": reply.call_id, "status": status, "artifact": None})
                continue
            _, statement, reply, attempt, text, reason, refused, index = item
            sid = statement["id"]
            check = record_in_order(ctx, card_id, STAGE, [statement], [results[index]])[0]
            checks[sid] = check
            sources[sid] = text
            best.note(statement, attempt, check, text)
            attempts.setdefault(sid, []).append(attempt_line(attempt, check))
            ctx.emit("proof_attempt", {"card_id": card_id, "statement_id": sid, "attempt": attempt,
                                       "call_id": reply.call_id, "status": check["status"],
                                       "decided": check["decided"], "artifact": check["artifact"],
                                       **({"recipe_change_reason": reason[:1000]} if reason else {}),
                                       **({"refused": "passing_recipe_changed"} if refused else {})})
            if not holds(check):
                failures[sid] = failure_text(check)
                round_failures[sid] = round_failure(check, text)
        round_id = "attempt-" + str(max(used[s["id"]] for s in todo))
        if lineage.observe_round(ctx.ledger, card_id=card_id, stage=STAGE, loop=lineage.PROVER, key=card_id,
                                 round_id=round_id, failures=round_failures):
            stops.append(lineage.LoopStop(lineage.PROVER, "stall", "the prover repeated the previous round's failing "
                                          "proofs or counterexamples for " + ", ".join(sorted(round_failures))))
            break
    for stop in stops:
        lineage.record_stop(ctx.ledger, card_id=card_id, stage=STAGE, stop=stop, terminal=False)
    return calls, models, stops


# ----- evaluation --------------------------------------------------------------------------------------------------

def evaluate(record: dict, checks: dict[str, dict], *, extra_edges=(), extra_premises=(), cited=None,
             citation_problems=None, circular=None, citations_checked=False, blocked=None, refuted_statements=None,
             mismatched=None, failed=None, unresolved=None,
             unresolved_claims=None) -> tuple[dict, list[dict], list[dict]]:
    """Obligation outcomes, the premise ledger and the claim records for a frozen record and its checks.
    ``refuted_statements`` maps a statement id to a counterexample check found by stage C; ``mismatched`` maps a
    statement id to the statement-match finding, and such a statement shows nothing whatever its check said (so a
    mismatched statement cannot discharge a premise either); ``failed`` maps a claim id to a confirmed witness
    violation stage C found (FIX-4). ``unresolved`` (FIX-7) maps a statement id to the disputed match objections: its
    claim keeps the check's result, but it cannot discharge a premise; ``unresolved_claims`` maps the claim ids to
    the dispute (given at stage C only: the claim records then carry ``match``). The other keywords are stage C's
    findings (``mathrecords.premise_ledger``)."""
    unresolved = unresolved or {}

    def outcomes_for(disputed_show_nothing: bool) -> dict:
        outcomes = {}
        for obligation in record["obligations"]:
            results = []
            for statement in obligation["statements"]:
                sid = statement["id"]
                check = (refuted_statements or {}).get(sid) or checks.get(sid)
                if sid in (mismatched or {}) and sid not in (refuted_statements or {}):
                    check = {"status": "unknown", "decided": False, "artifact": None,
                             "evidence": "does not match its claim: " + mismatched[sid]}
                elif disputed_show_nothing and sid in unresolved:
                    check = {"status": "unknown", "decided": False, "artifact": None,
                             "evidence": "its match with the claim is unresolved: " + unresolved[sid]}
                results.append(mathrecords.statement_result(statement, check))
            outcomes[obligation["id"]] = mathrecords.obligation_result(
                obligation, results, require_all=profile.is_slot(obligation["claim_id"]))
        return outcomes

    outcomes = outcomes_for(False)
    premises = list(record["premises"]) + list(extra_premises)
    ledger = mathrecords.premise_ledger(premises, record["obligations"],
                                        outcomes_for(True) if unresolved else outcomes, extra_edges=extra_edges,
                                        cited=cited, citation_problems=citation_problems, circular=circular,
                                        citations_checked=citations_checked)
    records = mathrecords.claim_records(record["claims"], record["obligations"], outcomes, ledger,
                                        not_formalized=record["not_formalized"], blocked=blocked,
                                        failed={**profile_failures(record, checks), **(failed or {})},
                                        unresolved=unresolved_claims)
    return outcomes, ledger, records


def slot_statements(record: dict, slots) -> list[dict]:
    return [s for o in record["obligations"] if o["claim_id"] in slots for s in o["statements"]]


def profile_failures(record: dict, checks: dict[str, dict]) -> dict[str, str]:
    """Profile slots that fail on consistency although their checks held (P3 against P2, P5 against P2)."""
    if not record.get("profile"):
        return {}
    p2 = profile.p2_sizes([(s, checks.get(s["id"])) for s in slot_statements(record, profile.P2_SLOTS)])
    p5 = [s["spec"] for s in slot_statements(record, ("P5",)) if s["form"] == "conservation"]
    return profile.consistency(record["profile"]["limits"], p2, p5)


def profile_block(record: dict, records: list[dict]) -> dict | None:
    """Each slot's status and tier, with the forms and tiers it accepts (None for a card without the profile)."""
    if not record.get("profile"):
        return None
    by_id = {r["claim_id"]: r for r in records}
    slots = {slot: {"status": by_id[slot]["status"], "tier": by_id[slot]["tier"], "label": by_id[slot]["label"],
                    "phrase": by_id[slot]["phrase"], "accepted_forms": list(profile.SLOT_FORMS[slot]),
                    "accepted_tiers": profile.accepted_tiers(slot)} for slot in profile.SLOTS if slot in by_id}
    passed = all(entry["status"] in mathrecords.CLAIM_HOLDS.values() for entry in slots.values())
    return {"version": record["profile"]["version"], "limits": record["profile"]["limits"], "slots": slots,
            "passed": passed and len(slots) == len(profile.SLOTS)}


def p1_cost_spec(record: dict, checks: dict[str, dict]) -> dict | None:
    """The frozen P1 cost statement, when its check held (the formula the target gate reads)."""
    for statement in slot_statements(record, ("P1",)):
        if statement["form"] == "cost" and holds(checks.get(statement["id"])):
            return statement["spec"]
    return None


def target_block(cfg: dict, record: dict, checks: dict[str, dict]) -> dict:
    """The compute target gate on the verified P1 formula; ``not_applicable`` when P1 did not hold."""
    rule = cfg["target"]
    spec = p1_cost_spec(record, checks)
    if spec is None:
        result = {"status": "not_applicable", "kill_code": None, "reasons": ["P1 did not hold: no verified formula"],
                  "rule": {"max_cost_exponent": rule["max_cost_exponent"], "state_growth": rule["state_growth"]}}
    else:
        result = target.evaluate(spec, max_cost_exponent=rule["max_cost_exponent"], state_rule=rule["state_growth"])
    return dict(result, applied=bool(rule["enabled"]))


# The [target] keys the target and coupled blocks read (calibration and planted pass them as the target rule).
TARGET_RULE_KEYS = ("max_cost_exponent", "state_growth", "coupled_max_cost_exponent", "coupled_max_state_exponent")


def coupled_block(cfg: dict, record: dict, checks: dict[str, dict]) -> dict:
    """FIX-17: the skill scale at target of every P2 slot, from the verified P1 formula and the card's frozen resources
    (hybrid/math/coupled.py); ``not_applicable`` when P1 did not hold or the record froze no resources. Applied as a
    gate exactly when the target gate is (live runs); calibration records it."""
    rule = cfg["target"]
    spec = p1_cost_spec(record, checks)
    resources = record.get("resources")
    shown = {"max_cost_exponent": rule["coupled_max_cost_exponent"],
             "max_state_exponent": rule["coupled_max_state_exponent"]}
    if spec is None:
        result = {"status": "not_applicable", "kill_code": None, "rule": shown,
                  "reasons": ["P1 did not hold: no verified formula"]}
    elif not isinstance(resources, dict):
        result = {"status": "not_applicable", "kill_code": None, "rule": shown,
                  "reasons": ["the frozen record holds no card resources (a record frozen before FIX-16)"]}
    else:
        result = coupled.evaluate(spec, resources, max_cost_exponent=rule["coupled_max_cost_exponent"],
                                  max_state_exponent=rule["coupled_max_state_exponent"])
    return dict(result, applied=bool(rule["enabled"]))


def coupled_summary(block: dict | None) -> dict | None:
    """The coupled measure in a line of events and run records: status, whether it decided, and each slot's scale."""
    if not block:
        return None
    return {"status": block["status"], "applied": block.get("applied", False),
            "slots": {slot: entry["scale"] for slot, entry in (block.get("slots") or {}).items()}}


def check_summary(check: dict) -> dict:
    return {k: check.get(k) for k in ("statement_id", "form", "purpose", "attempt", "status", "decided", "tier",
                                      "evidence", "counterexample", "artifact", "job_ids", "text", "bound",
                                      "details", "resource")}


def write_ledger(ctx: StageContext, card_id: str, stage: str, definitions: list[dict], ledger: list[dict]) -> str:
    """The ``assumption_ledger`` artifact: the frozen definitions (no status) and the premise ledger."""
    for entry in ledger:
        validate("assumption", entry)
    name = ctx.artifact_name(card_id)
    ctx.put_json_artifact("assumption_ledger", name, {"card_id": card_id, "stage": stage,
                                                      "definitions": definitions, "premises": ledger}, card_id)
    return name


def premise_line(entry: dict) -> str:
    how = entry["testable_by"] + (", measurement plan" if entry["measurement_plan"] else "")
    return (f"premise {entry['id']} ({entry['added_by']}, testable by {how}): {entry['status']}; used by "
            f"{', '.join(entry['used_by']) or 'none'}")


def decide(ctx: StageContext, card_id: str, stage: str, verdict: str, codes: list[str], records: list[dict],
           definitions: list[dict], ledger: list[dict], calls: list[str], extra: dict,
           notes: tuple[str, ...] = ()) -> None:
    by_calls = list(dict.fromkeys(c for c in calls if c))
    reasons = [f"math gate ({stage}): {verdict}" + (f"; kill codes {', '.join(codes)}" if codes else ""), *notes]
    reasons += [mathrecords.claim_line(r) for r in records]
    reasons += [premise_line(entry) for entry in ledger]
    reasons += [f"definition {d['id']} ({d['kind']}): part of the frozen statements" for d in definitions]
    measurements = mathrecords.scheduled_measurements(records, ledger)
    reasons += [f"measurement for premise {m['premise_id']} deferred to test design (claims "
                f"{', '.join(m['claims'])})" for m in measurements]
    ledger_name = write_ledger(ctx, card_id, stage, definitions, ledger)
    verdict_record = {"card_id": card_id, "stage": stage, "verdict": verdict, "kill_codes": codes, "claims": records,
                      "assumption_ledger": ledger_name, "manifest": verify_manifest(),
                      "definitions": [d["id"] for d in definitions],
                      "conditional": [r["claim_id"] for r in records if r["status"] == "conditional"],
                      "scheduled_measurements": measurements, **extra}
    validate("math_verdict", verdict_record)
    name = ctx.artifact_name(card_id)
    ctx.put_json_artifact("math_verdict", name, verdict_record, card_id)
    ctx.emit("math_decision", {"card_id": card_id, "stage": stage, "verdict": verdict, "kill_codes": codes,
                               "artifact": name, "tiers": {r["claim_id"]: r["label"] for r in records},
                               "statuses": {r["claim_id"]: r["status"] for r in records},
                               "premises": {a["id"]: a["status"] for a in ledger},
                               "definitions": [d["id"] for d in definitions],
                               **({"target_coupled": coupled_summary(extra["target_coupled"])}
                                  if extra.get("target_coupled") else {})})
    ctx.verdict(card_id, stage, verdict, reasons, by_calls[:200])


# ----- card completeness (FIX-16) -----------------------------------------------------------------------------------

REVISION_LOOP = "math_revision"


async def complete_card(ctx: StageContext, card_id: str, card: dict) -> dict | None:
    """The completeness gate, before the statement fixer runs: the card returned when its ``resources`` state every
    item the profile statements need (card_resources.missing), else None after the stage verdict was written.

    Stage V (revision) has no stage of its own and the store refuses a verdict for an earlier stage, so the gate runs
    V's step here: each round names the missing items and is charged to the ``math_revision`` loop
    (budgets.idea.math_revision_rounds); when granted, the card's author states them (card_resources.
    request_resources, a new card version) and the gate runs again. A round that leaves the same items missing
    stalls the loop. Without a budget book (calibration, the planted override, plumb) there is no author to ask and
    the gate ends at once. The verdict is ``undetermined`` with kill code ``card_incomplete``: the card was not
    formalized, which is not a math failure."""
    items = card_resources.missing(card)
    while items:
        ctx.emit("card_incomplete", {"card_id": card_id, "card_version": card["version"], "missing": items})
        stop = None
        if ctx.book is not None:
            stop = ctx.book.after_round(card_id=card_id, stage=STAGE, loop=REVISION_LOOP,
                                        round_id=f"resources-v{card['version']}",
                                        failures={item: None for item in items})
            if stop is None:
                card = await card_resources.request_resources(ctx, card_id, items, STAGE)
                items = card_resources.missing(card)
                continue
        incomplete_verdict(ctx, card_id, card, items, stop)
        return None
    return card


def incomplete_verdict(ctx: StageContext, card_id: str, card: dict, items: list[str], stop) -> None:
    codes = [card_resources.KILL_CODE] + ([stop.kill_code] if stop is not None else [])
    how = (f"the {REVISION_LOOP} loop ended: {stop.kill_code}: {stop.detail}" if stop is not None
           else "no author can be asked in this run (calibration or an operator override): a fixture error")
    notes = (f"gate: {card_resources.KILL_CODE}: the card's resources leave {len(items)} item(s) unstated, so no "
             f"statement was written ({how})",
             *(f"missing resource: {item}" for item in items))
    decide(ctx, card_id, STAGE, "undetermined", codes, [], [], [], [card["created_by_call"]],
           {"models": {}, "checks": {}, "missing_resources": items}, notes=notes)


# ----- the stage ---------------------------------------------------------------------------------------------------

async def run(ctx: StageContext, card_id: str) -> str | None:
    require_math(ctx)
    verify_manifest()
    card = ctx.card(card_id)
    claims = mathrecords.formal_claims(claim_items(card))
    if profile_required(ctx):
        claims += profile.profile_claims()
    record = load_frozen(ctx, card_id)
    if record is None:
        if profile_required(ctx):
            card = await complete_card(ctx, card_id, card)
            if card is None:
                return None
        record = await fix_and_freeze(ctx, card_id, card, claims)
    calls = list(record["fixer_calls"])

    checks: dict[str, dict] = {}
    auto = [s for s in all_statements(record) if s["form"] in mathrecords.AUTO_FORMS]
    results = await execute_checks(ctx, card_id, STAGE, [{"statement": s, "purpose": "auto"} for s in auto])
    for check in record_in_order(ctx, card_id, STAGE, auto, results):
        checks[check["statement_id"]] = check
    prover_calls, prover_models, prover_stops = await prove(ctx, card_id, card, record, checks)
    await audit_conformance(ctx, card_id, record, checks)
    calls += prover_calls
    breaches = families.violations({"statement_fixer": record["fixer_model"], "prover": prover_models,
                                    **prefreeze_models(record)})
    if breaches:
        raise StageError("model family separation broken: " + "; ".join(breaches))

    outcomes, ledger, records = evaluate(record, checks)
    verdict, codes = mathrecords.math_gate(records, ledger)
    extra = {}
    notes = tuple(stop.line() for stop in prover_stops)
    if record.get("profile"):
        extra["profile"] = profile_block(record, records)
        extra["target"] = target_block(ctx.cfg, record, checks)
        if extra["target"]["applied"] and extra["target"]["status"] == target.FAIL:
            codes = sorted(set(codes) | {target.KILL_CODE})
            verdict = "infeasible_proved" if verdict == "infeasible_proved" else "fail"
            notes += ("target gate: " + "; ".join(extra["target"]["reasons"]),)
        extra["target_coupled"] = coupled_block(ctx.cfg, record, checks)
        notes += ("skill scale at target (FIX-17): " + "; ".join(
            f"{slot} {entry['scale']}" for slot, entry in (extra["target_coupled"].get("slots") or {}).items())
            if extra["target_coupled"]["status"] != "not_applicable"
            else "skill scale at target (FIX-17): not applicable: " + "; ".join(extra["target_coupled"]["reasons"]),)
        if extra["target_coupled"]["applied"] and extra["target_coupled"]["status"] == "fail":
            codes = sorted(set(codes) | {coupled.KILL_CODE})
            verdict = "infeasible_proved" if verdict == "infeasible_proved" else "fail"
            notes += ("coupled target gate: " + "; ".join(extra["target_coupled"]["reasons"]),)
    job_ids = [j for c in checks.values() for j in c["job_ids"]]
    decide(ctx, card_id, STAGE, verdict, codes, records, record["definitions"], ledger, calls + job_ids, {
        "statements_artifact": frozen_event(ctx, card_id)["path"],
        "statement_hashes": frozen_event(ctx, card_id)["hashes"],
        "models": {"statement_fixer": [record["fixer_model"]], "prover": sorted(prover_models)},
        "checks": {sid: check_summary(c) for sid, c in checks.items()},
        "obligations": {oid: {k: v for k, v in o.items() if k != "statements"} for oid, o in outcomes.items()},
        "dropped": record["dropped"], "not_formalized": record["not_formalized"], **extra},
        notes=notes)
    return "C" if verdict == "pass" else None
