"""S1 well-formed: salad critic plus the toy example run in the executor.

Gate, per critic pass:
1. Any fatal finding (dimension error, contradictory definition, unfalsifiable central claim) fails the card.
2. No findings and a reproduced toy example pass it.
3. Otherwise (fixable findings or a toy mismatch) the card is revised while s1.max_revisions remain.
4. Once the revisions are used up: with s1.pass_with_residual_fixable (default true), no fatal finding and
   a reproduced toy example pass the card; the remaining fixable findings are listed in the reasons with a
   `residual:` prefix and stored as the `s1_residual` artifact for S3. A toy mismatch fails the card, and
   so do fixable findings when the option is false.

Budget and stall (hybrid.research.lineage, Phase 8). Revisions are counted per idea: a card that renames the
mechanism of an earlier card shares that card's count, so it does not get fresh revisions. The loop also ends early
when a revision leaves the same failures as the round before (the same fixable findings at the same places, and the
same toy-example mismatch): a `stall`. Either way the loop's end is recorded (`loop_stop`, `budget:s1_revision` or
`stall:s1_revision`) and rule 4 decides; a card that then fails carries the code with its `ill_formed` kill code.

Entry check of the refinement dialogue (refine.enabled, Phase 7, DISCOVERY-PIPELINE-PLAN.md section 7). The dialogue's
challenger replaces the revision loop, so S1 makes one pass and no revision: a fatal finding fails the card, a toy
example that does not reproduce fails it, and otherwise the card passes to stage R with its fixable findings stored as
the `s1_residual` artifact, which the challenger reads in its first round.
"""
from __future__ import annotations

import json

from hybrid.research.schemas import SchemaViolation, load_schema, validate
from hybrid.research.stages.common import (ESCAPE_FIELD, GENERATION_FIELDS, ORCHESTRATOR_FIELDS, STAGE_FIELDS,
                                           TRANSPLANT_FIELD, StageContext, StageError, after_s1,
                                           card_for_prompt, compare_expected, draft_schema, ground_draft,
                                           parsimony, refine_enabled)


LOOP = "s1_revision"


def failures(critic: dict, toy_ok: bool, toy_problems: list[str]) -> dict:
    """The round's failures for stall detection: each fixable finding by kind and place, and the toy-example
    mismatch with its text (the toy program is deterministic, so the same mismatch prints the same text)."""
    out = {f"{item['kind']}@{' '.join(item['where'].lower().split())}": None for item in critic["fixable"]}
    if not toy_ok:
        out["toy_example"] = toy_problems
    return out


def explorer_role(ctx: StageContext, card_id: str) -> str:
    created = ctx.card_events(card_id, "card_created")
    if not created:
        raise StageError(f"{card_id} has no card_created event")
    return created[0]["role"]


def finding_lines(critic: dict) -> tuple[list[str], list[str]]:
    fatal = [f"fatal {item['kind']} ({item['where']}): {item['problem']}" for item in critic["fatal"]]
    fixable = [f"fixable {item['kind']} ({item['where']}): {item['problem']}" for item in critic["fixable"]]
    return fatal, fixable


def residual_lines(critic: dict) -> list[str]:
    return [f"residual: {item['kind']} ({item['where']}): {item['problem']}" for item in critic["fixable"]]


async def check_toy(ctx: StageContext, card: dict) -> tuple[bool, list[str], str]:
    s1 = ctx.cfg["s1"]
    result = await ctx.runner.run(card["id"], "toy_example", {"main.py": card["toy_example"]["python"]})
    if not result.ok:
        return False, [f"toy example did not run: {result.status} {result.detail}; stderr tail: "
                       f"{result.stderr[-1500:]}"], result.job_id
    printed = result.printed_json()
    if printed is None:
        return False, [f"toy example printed no JSON object; stdout tail: {result.stdout[-800:]}"], result.job_id
    problems = compare_expected(card["toy_example"]["expected"], printed, s1["toy_rtol"], s1["toy_atol"])
    if problems:
        return False, ["toy example output differs from expected: " + "; ".join(problems[:12])], result.job_id
    return True, [f"toy example reproduced {len(card['toy_example']['expected'])} expected value(s)"], result.job_id


async def revise(ctx: StageContext, card: dict, critic: dict, toy_problems: list[str]) -> dict:
    user = "\n\n".join([
        "OPERATOR: revise",
        "YOUR CARD:\n" + card_for_prompt(card),
        "SALAD CRITIC OUTPUT:\n" + json.dumps(critic, indent=2),
        "TOY EXAMPLE CHECK:\n" + ("\n".join(toy_problems) or "passed"),
        "Fix every fixable finding and the toy example if it failed. Keep the core idea. Fix by tightening: "
        "define a symbol where it is used, drop or merge objects, or move a secondary part into extensions. "
        "Do not grow the core past the parsimony limits in the system prompt; a revision that adds state "
        "objects to answer a critique usually creates new gaps. Return the full revised card draft as JSON "
        "(same fields as before, without id, version, parent_ids, created_by_call, lens or status).",
    ])
    result = await ctx.ask(explorer_role(ctx, card["id"]), "explorer", user, revision_schema(card))
    revised = revised_card(ctx, card["id"], result.parsed, result.call_id)
    store_revision(ctx, revised, result.call_id)
    return revised


def revision_schema(card: dict) -> dict:
    """The draft schema a revision of this card answers in: an escape card keeps its non_reduction_argument, and a card
    drafted by a generation arm without papers is not asked for a source_transplant."""
    return draft_schema(escape=ESCAPE_FIELD in card, transplant=TRANSPLANT_FIELD in card or "arm" not in card)


def revised_card(ctx: StageContext, card_id: str, draft: dict, call_id: str) -> dict:
    """The next version of a card from a revised draft: references limited to the papers the card already had, the
    orchestrator, generation and stage fields carried over unchanged. Validated, not stored."""
    current = ctx.store.get_card(card_id)
    allowed = {ref["source_id"] for ref in current["references"]}
    revised, _, _ = ground_draft(draft, allowed)
    revised.update({name: current[name] for name in ORCHESTRATOR_FIELDS})
    revised.update({name: current[name] for name in GENERATION_FIELDS + STAGE_FIELDS if name in current})
    revised["version"] = ctx.store.next_version(card_id)
    revised["created_by_call"] = call_id
    try:
        validate("theory_card", revised)
    except SchemaViolation as exc:
        raise StageError(f"revised card fails theory_card schema: {exc}") from exc
    return revised


def store_revision(ctx: StageContext, revised: dict, call_id: str, **extra) -> None:
    ctx.store.put_card(revised)
    ctx.emit("card_revised", {"card_id": revised["id"], "version": revised["version"], "call_id": call_id,
                              **parsimony(revised), **extra})


async def run(ctx: StageContext, card_id: str) -> str | None:
    max_revisions = ctx.cfg["s1"]["max_revisions"]
    while True:
        card = ctx.card(card_id)
        critic = await ctx.ask("salad_critic", "salad_critic", card_for_prompt(card), load_schema("salad_critic"))
        toy_ok, toy_reasons, job_id = await check_toy(ctx, card)
        parsed = critic.parsed
        calls = [critic.call_id, job_id]
        ctx.put_json_artifact("salad_critic", ctx.artifact_name(card_id), parsed, card_id)
        fatal, fixable = finding_lines(parsed)
        if fatal:
            ctx.verdict(card_id, "S1", "fail", ["gate: fatal salad-critic finding", *fatal, *fixable,
                                                *([] if toy_ok else toy_reasons)], calls)
            return None
        if parsed["ok"] and toy_ok:
            ctx.verdict(card_id, "S1", "pass", ["salad critic: ok. " + parsed["notes"][:400], *toy_reasons], calls)
            return after_s1(ctx.cfg)
        if refine_enabled(ctx.cfg):
            return entry_check(ctx, card, parsed, toy_ok, toy_reasons, calls)
        reasons = fixable or ["salad critic: ok"]
        reasons += [] if toy_ok else toy_reasons
        failing = failures(parsed, toy_ok, toy_reasons)
        # A round is the critic pass after the n-th applied revision (card versions also change with every verdict, so
        # they do not name a round). A resume that repeats the pass on unchanged content repeats the same round.
        applied = len(ctx.card_events(card_id, "card_revised"))
        stop = ctx.book.after_round(card_id=card_id, stage="S1", loop=LOOP, round_id=f"after-{applied}-revisions",
                                    failures=failing)
        used = ctx.book.used(card_id, LOOP)
        if stop is None:
            ctx.verdict(card_id, "S1", "revise", [f"revision {used} of {max_revisions}", *reasons], calls)
            await revise(ctx, ctx.card(card_id), parsed, [] if toy_ok else toy_reasons)
            continue
        if toy_ok and ctx.cfg["s1"]["pass_with_residual_fixable"]:
            residual = residual_lines(parsed)
            if parsed["fixable"]:
                ctx.put_json_artifact("s1_residual", ctx.artifact_name(card_id),
                                      {"card_id": card_id, "card_version": card["version"],
                                       "revisions_used": used, "findings": parsed["fixable"]}, card_id)
            ctx.verdict(card_id, "S1", "pass",
                        [f"gate: passed with {len(residual)} residual fixable finding(s) after {used} "
                         "revision(s); no fatal finding and the toy example reproduced", stop.line(),
                         *residual, *toy_reasons], calls)
            return after_s1(ctx.cfg)
        ctx.verdict(card_id, "S1", "fail", [f"gate: fixable findings remain after {used} revision(s)", stop.line(),
                                            *reasons], calls)
        return None


def entry_check(ctx: StageContext, card: dict, critic: dict, toy_ok: bool, toy_reasons: list[str],
                calls: list[str]) -> str | None:
    """S1 as the refinement dialogue's entry check: no revision here. A toy example that does not reproduce fails the
    card; fixable findings go to stage R as the s1_residual artifact."""
    card_id = card["id"]
    if not toy_ok:
        ctx.verdict(card_id, "S1", "fail", ["gate: the toy example does not reproduce at the refinement dialogue's "
                                            "entry check", *toy_reasons, *finding_lines(critic)[1]], calls)
        return None
    ctx.put_json_artifact("s1_residual", ctx.artifact_name(card_id),
                          {"card_id": card_id, "card_version": card["version"], "revisions_used": 0,
                           "findings": critic["fixable"]}, card_id)
    ctx.verdict(card_id, "S1", "pass",
                [f"gate: entry check passed; {len(critic['fixable'])} fixable finding(s) go to the refinement "
                 "dialogue", *residual_lines(critic), *toy_reasons], calls)
    return after_s1(ctx.cfg)
