"""Stage R: the refinement dialogue (DISCOVERY-PIPELINE-PLAN.md section 7, step 2; Phase 7).

Runs between S1 and S2 when `refine.enabled`. S1 is its entry check (stages/s1_wellformed.py `entry_check`): a fatal
finding or a toy example that does not reproduce ends the card there, and S1 makes no revision, because the
dialogue's challenger replaces the revision loop.

Rounds. A challenger asks at most `refine.max_questions` questions about the card; the proposer answers by revising
the card and by writing the definitions and premises the card relies on. Each question must target one of five
kinds (QUESTION_TARGETS): an undefined object, an unstated assumption, a claim without a test, a mechanism step whose
cost is not stated, or a commitment that looks like decoration (R1). The schema accepts nothing else, so free-form
critique cannot be returned. The dialogue ends when the challenger asks nothing (`converged`), when the idea has used
its rounds (`budgets.idea.refinement_rounds`, 3; `budget:refinement`) or when a round asks the same questions at the
same places as the round before (`stall:refinement`). Rounds are charged through `BudgetBook.after_round` with loop
`refinement`, so a renamed copy of an idea shares the idea's rounds. When the rounds are used up, no further
challenger call is made: the stop is recorded and the dialogue ends with the last revision unchecked by the
challenger (the plan counts at most three dialogue calls on the GPT route per card).

Roles. The challenger must come from a family other than the proposer's (plan section 16). The proposer's family is
that of the model that drafted the card (`card_created`): a card drafted by a GPT model gets `refine_challenger_alt`
(Claude first), every other card `refine_challenger` (GPT first, local Qwen as fallback). When a fallback puts the
challenger in the proposer's family the round records `family_overlap`; it is reported, not enforced, as for the
stage C challenger (hybrid/research/families.py). The proposer is the card's explorer role, as for S1 revisions.

Growth limit. The parsimony counter (state objects plus update and readout equations, `common.parsimony`) may not
grow by more than `refine.growth_limit` (0.25) over the dialogue. A revision past the limit is rejected
(`refine_answer` with `accepted: false`); the card keeps its previous version, and the round still counts.

Output. The revised card, plus the `dialogue_assumptions` artifact: definitions (D1, ...; conventions and protocol
choices) and premises (A1, ...; each with why_needed and testable_by code or citation) in the vocabulary of stage M's
statement fixer (stages/math_contracts.py), every entry tagged `added_by: dialogue`, and the questions still open
when the dialogue ended, each with a `status`: a stall's questions are `unanswered`; at the round cap the last round's
questions are `answered_unchecked` (the last accepted revision answered them, but the challenger never saw it) or
`unanswered` (FIX-10). A converged dialogue has none. Stage M gives this list to the statement fixer (prompt_sections/refine_premises.md).

Resources (FIX-16). In a run that holds cards to the profile, a card whose `resources` section leaves items unstated
(card_resources.missing) is asked once after the dialogue to state them, so it reaches stage M complete; M gates
again before its statement fixer.

Gate. R passes whenever the dialogue ends, with the end, the open questions and the core sizes in the reasons. It
fails only when an accepted revision broke the toy example (`gate: ill_formed:toy: ...`), which is rerun after the
last accepted revision.

Resume. Every count comes from the ledger. A round that was granted and charged but whose answer was lost when the
run stopped is answered from its recorded questions, without a new challenger call or a second charge.

Ledger events: `refine_start` (once per card: the core size the growth limit is measured from), `refine_round` (the
challenger's questions, its model and any family overlap), `refine_answer` (the proposer's answer: accepted or not,
core size, problems), plus `card_revised` with `stage: "R"` for each accepted revision. Artifacts: `refine_round`,
`refine_answer`, `dialogue_assumptions`, `refinement` (the whole dialogue).
"""
from __future__ import annotations

import copy
import json

from hybrid.research import lineage
from hybrid.research.families import family
from hybrid.research.schemas import validate
from hybrid.research.stages import card_resources
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import StageContext, StageError, card_for_prompt, load_prompt, parsimony
from hybrid.research.stages.s1_wellformed import (check_toy, explorer_role, revised_card, revision_schema,
                                                  store_revision)

STAGE = "R"
LOOP = "refinement"
CHALLENGER = "refine_challenger"
CHALLENGER_ALT = "refine_challenger_alt"
PROPOSER_PROMPT = "refine_proposer"
QUESTION_TARGETS = ("undefined_object", "unstated_assumption", "untested_claim", "uncosted_step",
                    "decorative_commitment")
CONVERGED = "converged"
# Status of an open question when the dialogue ends (FIX-10): a stall leaves its questions unanswered; at the round cap
# the last round's questions were answered by a revision the challenger never checked, or not answered at all.
ANSWERED_UNCHECKED = "answered_unchecked"
UNANSWERED = "unanswered"
PREMISE_BASIS = "stated by the proposer in the refinement dialogue; nothing has checked it yet"


# ----- schemas -------------------------------------------------------------------------------------------------------

_TEXT = {"type": "string", "minLength": 1}


def challenger_schema(max_questions: int) -> dict:
    question = {"type": "object", "additionalProperties": False, "required": ["id", "target", "where", "question"],
                "properties": {"id": {"type": "string", "pattern": r"^Q[0-9]{1,2}$"},
                               "target": {"enum": list(QUESTION_TARGETS)}, "where": _TEXT, "question": _TEXT}}
    return {"type": "object", "additionalProperties": False, "required": ["questions", "notes"],
            "properties": {"questions": {"type": "array", "maxItems": max_questions, "items": question},
                           "notes": {"type": "string"}}}


def _fixer_items() -> tuple[dict, dict]:
    """The definition and premise item schemas of the statement fixer (the Phase 3 vocabulary)."""
    props = mc.statement_fixer_schema(["claim"], 1)["properties"]
    premise = copy.deepcopy(props["premises"]["items"])
    # FIX-6: ``subject`` (where a premise is settled) is the statement fixer's call, made against the frozen
    # definitions; the dialogue's premises keep the Phase 3 vocabulary.
    premise["required"] = [name for name in premise["required"] if name != "subject"]
    premise["properties"].pop("subject")
    return props["definitions"]["items"], premise


def proposer_schema(card: dict) -> dict:
    definition, premise = _fixer_items()
    answer = {"type": "object", "additionalProperties": False, "required": ["question_id", "answer", "change"],
              "properties": {"question_id": _TEXT, "answer": _TEXT, "change": {"type": "string"}}}
    draft = revision_schema(card)
    # The draft's "#/$defs/..." references must resolve from the root of this schema.
    defs = draft.pop("$defs")
    draft.pop("$schema", None)
    return {"type": "object", "additionalProperties": False, "required": ["card", "answers", "definitions", "premises"],
            "$defs": defs,
            "properties": {"card": draft,
                           "answers": {"type": "array", "items": answer},
                           "definitions": {"type": "array", "maxItems": 20, "items": definition},
                           "premises": {"type": "array", "maxItems": 20, "items": premise}}}


# ----- helpers ------------------------------------------------------------------------------------------------------


def core_size(card: dict) -> int:
    counts = parsimony(card)
    return counts["state_objects"] + counts["update_equations"] + counts["readout_equations"]


def question_key(question: dict) -> str:
    """A question's identity for stall detection: its target kind at its place in the card (wording ignored)."""
    return f"{question['target']}@{' '.join(question['where'].lower().split())}"


def proposer_model(ctx: StageContext, card_id: str) -> str | None:
    created = ctx.card_events(card_id, "card_created")
    if not created:
        raise StageError(f"{card_id} has no card_created event")
    return created[0].get("model")


def challenger_role(proposer: str | None) -> str:
    """refine_challenger is GPT first; a card a GPT model drafted gets the Claude-first role instead."""
    return CHALLENGER_ALT if family(proposer) == "openai" else CHALLENGER


def dialogue_entries(parsed: dict) -> tuple[list[dict], list[dict]]:
    """The proposer's definitions and premises as ledger entries tagged added_by dialogue. Premises are open: the
    dialogue states them; stages M and C settle them."""
    definitions = [dict(item, added_by="dialogue") for item in parsed["definitions"]]
    premises = []
    for item in parsed["premises"]:
        entry = dict(item, added_by="dialogue", status="open", evidence=None, used_by=[], basis=PREMISE_BASIS)
        validate("assumption", entry)
        premises.append(entry)
    return definitions, premises


def answer_problems(parsed: dict, questions: list[dict]) -> list[str]:
    problems = []
    answered = {a["question_id"] for a in parsed["answers"]}
    missing = [q["id"] for q in questions if q["id"] not in answered]
    if missing:
        problems.append(f"no answer to {', '.join(missing)}")
    for kind, items in (("definition", parsed["definitions"]), ("premise", parsed["premises"])):
        ids = [item["id"] for item in items]
        if len(set(ids)) != len(ids):
            problems.append(f"repeated {kind} ids: {ids}")
    for premise in parsed["premises"]:
        problems += mc.premise_problems(premise)
    return problems


def start_size(ctx: StageContext, card_id: str) -> dict:
    """The core size the growth limit is measured from: recorded once, when the dialogue first starts on the card, so a
    resumed dialogue keeps the same limit."""
    recorded = ctx.card_events(card_id, "refine_start")
    if recorded:
        return recorded[0]
    card = ctx.card(card_id)
    base = core_size(card)
    limit = int(base * (1 + ctx.cfg["refine"]["growth_limit"]))
    return ctx.emit("refine_start", {"card_id": card_id, "card_version": card["version"], "core_size": base,
                                     "core_limit": limit, "proposer_model": proposer_model(ctx, card_id)})["data"]


def history_block(ctx: StageContext, card_id: str) -> str:
    rounds = ctx.card_events(card_id, "refine_round")
    answers = ctx.card_events(card_id, "refine_answer")
    if not rounds:
        return "PREVIOUS ROUNDS: none (this is the first round)."
    lines = ["PREVIOUS ROUNDS (your questions and the proposer's answers):"]
    for index, asked in enumerate(rounds, start=1):
        lines.append(f"Round {index}:")
        lines += [f"- {q['id']} [{q['target']} @ {q['where']}] {q['question']}" for q in asked["questions"]]
        answer = next((a for a in answers if a["round"] == asked["round"]), None)
        if answer is not None:
            state = "accepted" if answer["accepted"] else f"REJECTED ({answer['rejected_because']})"
            lines.append(f"  proposer's revision {state}; answers:")
            lines += [f"  - {a['question_id']}: {a['answer'][:500]}" for a in answer["answers"]]
    return "\n".join(lines)


def residual_block(ctx: StageContext, card_id: str) -> str:
    residual = ctx.latest_json_artifact(card_id, "s1_residual")
    if not residual or not residual["findings"]:
        return "FINDINGS OF THE S1 ENTRY CHECK: none."
    rows = [f"- {f['kind']} ({f['where']}): {f['problem']}" for f in residual["findings"]]
    return "FINDINGS OF THE S1 ENTRY CHECK (fixable; ask about any the card still leaves open):\n" + "\n".join(rows)


def current_list(ctx: StageContext, card_id: str) -> tuple[list[dict], list[dict]]:
    """Definitions and premises of the latest accepted answer (empty before the first one)."""
    accepted = [a for a in ctx.card_events(card_id, "refine_answer") if a["accepted"]]
    if not accepted:
        return [], []
    record = json.loads((ctx.run_dir / accepted[-1]["artifact"]).read_text(encoding="utf-8"))
    return record["definitions"], record["premises"]


def stored_path(ctx: StageContext, stored: dict) -> str:
    return stored["path"].relative_to(ctx.run_dir).as_posix()


def list_block(definitions: list[dict], premises: list[dict]) -> str:
    if not definitions and not premises:
        return "DEFINITIONS AND PREMISES SO FAR: none written yet."
    return "DEFINITIONS AND PREMISES SO FAR:\n" + json.dumps({"definitions": definitions, "premises": premises},
                                                               indent=2, ensure_ascii=False)


def last_round_questions(ctx: StageContext, card_id: str) -> list[dict]:
    """FIX-10: when the rounds run out, the questions of the last challenger round stay open, because the challenger
    never saw the revision that answered them. Each is `answered_unchecked` when the last accepted answer of that round
    answered it, else `unanswered` (no answer, or a revision rejected for growth). Empty when no round was asked."""
    rounds = ctx.card_events(card_id, "refine_round")
    if not rounds:
        return []
    last = rounds[-1]
    answer = next((a for a in ctx.card_events(card_id, "refine_answer") if a["round"] == last["round"]), None)
    answered = {a["question_id"] for a in answer["answers"]} if answer is not None and answer["accepted"] else set()
    return [dict(q, status=ANSWERED_UNCHECKED if q["id"] in answered else UNANSWERED) for q in last["questions"]]


def proposer_system() -> str:
    shared = load_prompt("explorer").split("\n", 1)[1].strip()
    return (load_prompt(PROPOSER_PROMPT).rstrip() + "\n\n## General card rules (shared with every explorer)\n\n"
            + shared + "\n")


# ----- the stage ----------------------------------------------------------------------------------------------------


async def run(ctx: StageContext, card_id: str) -> str | None:
    rcfg = ctx.cfg["refine"]
    start = start_size(ctx, card_id)
    proposer = start["proposer_model"]
    role = challenger_role(proposer)
    calls: list[str] = []
    ended, open_questions, overlaps = None, [], []
    while True:
        card = ctx.card(card_id)
        applied = len(ctx.card_events(card_id, "refine_answer"))
        rounds = ctx.card_events(card_id, "refine_round")
        charged = sum(1 for e in ctx.card_events(card_id, "budget_charge") if e["loop"] == LOOP)
        if rounds and rounds[-1]["questions"] and applied < len(rounds) and charged > applied:
            # A resumed run whose last round was granted (and charged) but whose answer was lost: answer the recorded
            # questions now instead of asking again and charging the round twice.
            await answer_round(ctx, card, rounds[-1]["questions"], rounds[-1]["round"], start, calls)
            continue
        used, cap = ctx.book.used(card_id, LOOP), ctx.book.cap(LOOP)
        if used >= cap:
            stop = lineage.LoopStop(LOOP, "budget", f"{used} of {cap} round(s) used")
            lineage.record_stop(ctx.ledger, card_id=card_id, stage=STAGE, stop=stop, terminal=False)
            ended, open_questions = stop.kill_code, last_round_questions(ctx, card_id)
            break
        definitions, premises = current_list(ctx, card_id)
        user = "\n\n".join(["CARD:\n" + card_for_prompt(card), residual_block(ctx, card_id),
                            history_block(ctx, card_id), list_block(definitions, premises),
                            f"Ask at most {rcfg['max_questions']} questions, each with one target kind from the "
                            f"schema, or none when nothing of those kinds is left open. Return JSON."])
        asked = await ctx.call(role, load_prompt(CHALLENGER), user, challenger_schema(rcfg["max_questions"]))
        calls.append(asked.call_id)
        questions = asked.parsed["questions"]
        overlap = family(asked.model) == family(proposer)
        if overlap:
            overlaps.append(asked.call_id)
        round_no = len(ctx.card_events(card_id, "refine_round")) + 1
        stored = ctx.put_json_artifact("refine_round", ctx.artifact_name(card_id),
                                       {"card_id": card_id, "round": round_no, "call_id": asked.call_id,
                                        "model": asked.model, "questions": questions, "notes": asked.parsed["notes"]},
                                       card_id)
        ctx.emit("refine_round", {"card_id": card_id, "round": round_no, "role": role, "call_id": asked.call_id,
                                  "model": asked.model, "proposer_model": proposer, "family_overlap": overlap,
                                  "questions": questions, "artifact": stored_path(ctx, stored)})
        if not questions:
            ended = CONVERGED
            break
        stop = ctx.book.after_round(card_id=card_id, stage=STAGE, loop=LOOP, round_id=f"after-{applied}-answers",
                                    failures={question_key(q): None for q in questions})
        if stop is not None:
            ended, open_questions = stop.kill_code, [dict(q, status=UNANSWERED) for q in questions]
            break
        await answer_round(ctx, card, questions, round_no, start, calls)
    resources = await resources_round(ctx, card_id, calls)
    return await finish(ctx, card_id, start, ended, open_questions, overlaps, calls, resources)


def profile_run(cfg: dict) -> bool:
    return cfg["s3"]["mode"] == "math" and bool(cfg.get("math", {}).get("profile"))


async def resources_round(ctx: StageContext, card_id: str, calls: list[str]) -> list[str]:
    """FIX-16: the completeness gate at R. In a run that holds cards to the profile, a card whose ``resources`` leave
    items unstated is asked once, after the dialogue, to state them (card_resources.request_resources), so it reaches
    stage M complete. Returns the verdict lines; stage M gates again and may ask once more under its own budget."""
    if not profile_run(ctx.cfg):
        return []
    items = card_resources.missing(ctx.card(card_id))
    if not items:
        return ["card resources: complete"]
    revised = await card_resources.request_resources(ctx, card_id, items, STAGE)
    calls.append(revised["created_by_call"])
    left = card_resources.missing(revised)
    return [f"card resources: {len(items)} item(s) were unstated; the author was asked once; "
            + (f"{len(left)} still unstated: {', '.join(left)[:600]}" if left else "now complete")]


async def answer_round(ctx: StageContext, card: dict, questions: list[dict], round_no: int, start: dict,
                       calls: list[str]) -> None:
    card_id = card["id"]
    definitions, premises = current_list(ctx, card_id)
    user = "\n\n".join([
        "YOUR CARD:\n" + card_for_prompt(card),
        "THE CHALLENGER'S QUESTIONS:\n" + "\n".join(f"- {q['id']} [{q['target']} @ {q['where']}] {q['question']}"
                                                    for q in questions),
        list_block(definitions, premises),
        f"Answer every question by id. Revise the card where an answer changes it, by tightening: the core (state "
        f"objects plus update and readout equations, {start['core_size']} at the start of the dialogue) may not grow "
        f"past {start['core_limit']}; a revision past that is rejected. Return the full revised card draft, the "
        f"answers, and the complete current definitions and premises lists as JSON."])
    reply = await ctx.call(explorer_role(ctx, card_id), proposer_system(), user, proposer_schema(card))
    calls.append(reply.call_id)
    revised = revised_card(ctx, card_id, reply.parsed["card"], reply.call_id)
    size = core_size(revised)
    definitions, premises = dialogue_entries(reply.parsed)
    problems = answer_problems(reply.parsed, questions)
    accepted = size <= start["core_limit"]
    rejected_because = None if accepted else (f"core grew to {size}, past the limit {start['core_limit']} "
                                              f"({start['core_size']} at the start)")
    record = {"card_id": card_id, "round": round_no, "call_id": reply.call_id, "model": reply.model,
              "accepted": accepted, "rejected_because": rejected_because, "core_size": size,
              "answers": reply.parsed["answers"], "definitions": definitions, "premises": premises,
              "problems": problems}
    stored = ctx.put_json_artifact("refine_answer", ctx.artifact_name(card_id), record, card_id)
    if accepted:
        store_revision(ctx, revised, reply.call_id, stage=STAGE)
    ctx.emit("refine_answer", {"card_id": card_id, "round": round_no, "call_id": reply.call_id,
                               "model": reply.model, "accepted": accepted, "rejected_because": rejected_because,
                               "core_size": size, "problems": problems, "artifact": stored_path(ctx, stored),
                               "answers": [{"question_id": a["question_id"], "answer": a["answer"][:600]}
                                           for a in reply.parsed["answers"]],
                               "version": revised["version"] if accepted else None})


async def finish(ctx: StageContext, card_id: str, start: dict, ended: str, open_questions: list[dict],
                 overlaps: list[str], calls: list[str], resources: list[str] = ()) -> str | None:
    card = ctx.card(card_id)
    answers = ctx.card_events(card_id, "refine_answer")
    accepted = [a for a in answers if a["accepted"]]
    definitions, premises = current_list(ctx, card_id)
    rounds = len(ctx.card_events(card_id, "refine_round"))
    unchecked = ended.startswith("budget") and bool(accepted)
    summary = {"card_id": card_id, "card_version": card["version"], "ended": ended, "rounds": rounds,
               "answers": len(answers), "accepted": len(accepted), "core_size_start": start["core_size"],
               "core_size_end": core_size(card), "core_limit": start["core_limit"],
               "open_questions": open_questions, "last_revision_unchecked": unchecked,
               "family_overlap_calls": overlaps, "definitions": definitions, "premises": premises}
    ctx.put_json_artifact("dialogue_assumptions", ctx.artifact_name(card_id),
                          {"card_id": card_id, "card_version": card["version"], "definitions": definitions,
                           "premises": premises, "open_questions": open_questions, "ended": ended}, card_id)
    ctx.put_json_artifact("refinement", ctx.artifact_name(card_id), summary, card_id)
    reasons = [f"dialogue ended: {ended} after {rounds} challenger round(s), {len(accepted)} of {len(answers)} "
               f"revision(s) accepted",
               f"core size {start['core_size']} -> {core_size(card)} (limit {start['core_limit']})",
               f"assumption list: {len(definitions)} definition(s), {len(premises)} premise(s), added_by dialogue"]
    reasons += [f"open question {q['id']} ({q['status'].replace('_', ', ')}) [{q['target']} @ {q['where']}]: "
                f"{q['question'][:300]}" for q in open_questions]
    reasons += list(resources)
    if unchecked:
        reasons.append("the last revision used the last round and was not shown to the challenger again")
    if overlaps:
        reasons.append(f"family overlap: the challenger answered from the proposer's family in {len(overlaps)} "
                       "call(s) (a fallback route); recorded, not enforced")
    if accepted:
        toy_ok, toy_reasons, job_id = await check_toy(ctx, card)
        calls.append(job_id)
        if not toy_ok:
            ctx.verdict(card_id, STAGE, "fail", ["gate: ill_formed:toy: the revised card's toy example does not "
                                                 "reproduce", *toy_reasons, *reasons], calls)
            return None
        reasons += toy_reasons
    # A dialogue the idea's used-up rounds skipped made no call: the verdict cites the call that wrote the card version.
    ctx.verdict(card_id, STAGE, "pass", reasons, calls or [card["created_by_call"]])
    return "S2"
