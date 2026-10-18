"""S2 prior art and reduction: literature classification, reduction critic, numeric equivalence check.

Gate (RESEARCH-SYSTEM-SPEC.md section 3), applied in this order:
1. `known_disproven` fails.
2. A verified reduction gives `reduces_to_known` unless the critic states that the difference is
   in the learning rule (difference_if_any starts with "LEARNING_RULE:"), which passes this rule.
   Verified is decided on the host (FIX-10, `judge_comparisons`). The critic declares the comparators the
   check must run (FIX-12, `comparators`: id, base family and form, the repaired update and read, the
   transformation types from the repair policy contracts.REPAIR_TYPES, the parameter mapping and whether
   the equality holds only in a limit). The check program prints its comparisons as data
   (contracts.REDUCTION_CHECK_REPORT), each under a declared id with that declaration's family,
   transformation types and limit_only; an omitted, undeclared or repeated id, or metadata that disagrees
   with the declaration, makes the check unverified (`report_problem`). Eligibility comes from the
   declaration, never from a label the program prints (`eligibility`): a comparator of the stated
   `reduces_to` whose types are all in the policy verifies the reduction within EQUIVALENCE_TOL; one with
   a type outside it (`other`) is unresolved and cannot verify; a limit-only one never verifies; a
   comparison under no declared id never counts. When every counted comparison is outside the tolerance
   the claim is refuted. No flag the program computes is read. When the critic gives no check code, or the
   program prints no JSON object at all, a confidence at or above s2.reduction_confidence counts as
   verified; a printed object outside the contract is unverified.
3. `prior_art_tested` fails unless the card adds a specific untested element (one prior_art call).
4. `novelty_unassessed` gives `undetermined`. A `novel` label whose searches returned no paper, even after
   one retry with broader queries, is not a finding of novelty (s2.empty_search_counts_as_novel = false).
5. With s2.require_radical, `radical_enough: false` fails.
6. Otherwise (`novel` with papers, `prior_art_undertested`, or tested-plus-new-element) passes.

The reduction critic also says whether the core idea is a textbook construction from its source field
applied without modification (`textbook_transplant`). That answer is recorded in the artifact and the
verdict reasons and counted by `inspect`; it does not gate.

A card with a `non_reduction_argument` (an escape card) gets prompts/reduction_critic_escape.md and
prompts/reduction_check_escape.md appended to the critic and check user messages, with the argument and
the parent's reduction: the critic tests the argument and the check tries the strongest repaired mapping.
The gate is unchanged. A `reduces_to_known` verdict calls ctx.on_reduction, which queues an escape task.

Own history (outcomes.enabled, DISCOVERY-PIPELINE-PLAN.md sections 8 and 9): before the reduction critic, the
outcome store is searched by the card's mechanism signature and its nearest neighbours (`outcomes.own_history`).
The matches, with their recorded kill codes and cause ids, are stored as an `own_history` artifact and event, listed
in the verdict reasons, and appended to the reduction critic's user message (prompt_sections/own_history.md), each
cause whole as stored (FIX-12, F1), followed by the cause-bearing equations before stage R and now, each marked
unchanged, attempted_repair or validated_repair (FIX-12, F3; `equation_changes`, also stored in the artifact). With
outcomes.enforce_r3, an exact-signature match with a dead idea fails the card with `gate: not_radical:R3` unless
the card's difference statement cites every cause of that record (rule R3, plan section 1.3), checked right after
the reduction rule. This path applies when s2.difference_statement is off; cards then carry no difference statement,
so every exact match fails.

Phase 7 (stages/difference.py; DISCOVERY-PIPELINE-PLAN.md sections 1.2, 1.3 and 8):
- s2.family_exclusion: the reduction critic also names the card's family from hybrid/config/families.toml (recorded,
  and it picks between sibling families, see difference.excluded_reduction), and a verified reduction whose counted
  comparator (`counted_comparisons`, KD4) is in an out-of-scope family gives `reduces_to_known` with the gate reason
  `gate: excluded_family:<id>: ...` (kill code excluded_family:<id>), whatever `difference_if_any` says. The critic's
  label alone never kills: a counted comparator in an in-scope family takes the ordinary verified-reduction path.
- s2.difference_statement: after the other gates pass, the proposer writes a difference statement, the host and the
  challenger role check it (R3 against exact and near signature matches of dead ideas when outcomes.enforce_r3), and a
  failure gets one send-back (loop send_back) before the card is dropped with `gate: not_radical:R3: ...` or
  `gate: no_difference: ...`. Both switches default to false, so runs started before Phase 7 resume as they started.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import httpx
import jsonschema

from hybrid.literature import LOCAL_SOURCE, SourceError, classify_prior_art, resolve_index_path
from hybrid.research import outcomes as outcome_store
from hybrid.research.signature import card_signature
from hybrid.research.schemas import SchemaViolation, load_schema, validate
from hybrid.research.stages import difference
from hybrid.research.stages.common import StageContext, StageError, card_for_prompt, load_section
from hybrid.research.stages.contracts import (PRIOR_ART_DELTA, REDUCTION_CHECK, REDUCTION_CHECK_REPORT, REPAIR_TYPES,
                                              repair_policy_text)
from hybrid.research.stages.escape import escape_user_suffix
from hybrid.research.stages.r_refine import challenger_role, proposer_model
from hybrid.research.stages.s1_wellformed import explorer_role

RECEIPT_SOURCES = (LOCAL_SOURCE, "arxiv", "openalex", "semantic_scholar")
# Literature events name Semantic Scholar "s2" (hybrid.literature.paper.SOURCES); receipts use the full name.
RECEIPT_SOURCE_NAMES = {"s2": "semantic_scholar"}
LEARNING_RULE_MARK = "LEARNING_RULE:"
NOVELTY_UNASSESSED = "novelty_unassessed"
BROADER_SEARCH_NOTE = ("previous queries returned nothing; broaden to the parent fields and to the "
                       "known-alternatives list")
# FIX-10: the host's equivalence tolerance on a comparison's max_abs_diff.
EQUIVALENCE_TOL = 1e-6


class CardLiteratureSink:
    """Forwards literature events to the ledger and adds one literature_receipt per source,
    tagged with the card, in the shape the inspect report validates. The arxiv_local receipt also
    states the index snapshot (date, record count, whether the harvest finished), or why the index
    was skipped."""

    def __init__(self, ctx: StageContext, card_id: str):
        self.ctx, self.card_id = ctx, card_id

    def append(self, type: str, actor: str, data: dict) -> dict:
        event = self.ctx.ledger.append(type, actor, dict(data, card_id=self.card_id))
        if type == "literature_search":
            for event_source, ids in data["per_source_ids"].items():
                source = RECEIPT_SOURCE_NAMES.get(event_source, event_source)
                if source in RECEIPT_SOURCES:
                    receipt = {"query": data["query"], "source": source, "ts": data["ts"],
                               "result_ids": list(ids), "card_id": self.card_id}
                    if source == LOCAL_SOURCE:
                        receipt.update(local_index_fields(data))
                    self.ctx.ledger.append("literature_receipt", actor, receipt)
        return event


def local_index_fields(search_event: dict) -> dict:
    """Snapshot facts for the arxiv_local receipt, and the reason when the index was not searched."""
    index = search_event.get("local_index") or {}
    fields: dict = {}
    if index.get("status") == "ok":
        fields.update(record_count=index["record_count"], index_complete=index["complete"])
        if index["snapshot_date"]:
            fields["snapshot_date"] = index["snapshot_date"]
    note = search_event["errors"].get(LOCAL_SOURCE)
    if note:
        fields["note"] = note
    return fields


async def prior_art(ctx: StageContext, card: dict, *, broaden: bool = False) -> tuple[dict, list[str]]:
    calls: list[str] = []

    async def llm_complete(role: str, system: str, user: str, json_schema: dict) -> Any:
        result = await ctx.call(role, system, user, json_schema, allow_web=False)
        calls.append(result.call_id)
        return result

    card_text = card_for_prompt(card)
    if broaden:
        card_text += "\n\n" + BROADER_SEARCH_NOTE
    try:
        classify = ctx.classify_prior_art or partial(
            classify_prior_art, local_index_path=resolve_index_path(ctx.cfg["literature"]["local_index_path"]))
        result = await classify(card_text, llm_complete, CardLiteratureSink(ctx, card["id"]))
    except (ValueError, jsonschema.ValidationError, SourceError, httpx.HTTPError) as exc:
        # Model output that fails the literature schemas, or a literature transport failure.
        raise StageError(f"classify_prior_art failed: {type(exc).__name__}: {exc}") from exc
    return result, calls


async def prior_art_checked(ctx: StageContext, card: dict) -> tuple[dict, list[str], bool]:
    """Returns (result, calls, novelty_unassessed).

    A `novel` label with no paper from any query says nothing about novelty. Unless
    s2.empty_search_counts_as_novel is set, the classifier runs once more with broader queries. If that
    search returns no paper either, the result is novelty_unassessed.
    """
    result, calls = await prior_art(ctx, card)
    if result["label"] != "novel" or result["candidates"] or ctx.cfg["s2"]["empty_search_counts_as_novel"]:
        return result, calls, False
    retry, retry_calls = await prior_art(ctx, card, broaden=True)
    calls += retry_calls
    unassessed = retry["label"] == "novel" and not retry["candidates"]
    ctx.emit("novelty_unassessed", {"card_id": card["id"], "first_queries": result["queries"],
                                    "retry_queries": retry["queries"], "retry_papers": len(retry["candidates"]),
                                    "still_unassessed": unassessed})
    return retry, calls, unassessed


def prior_art_record(result: dict) -> dict:
    record = {"label": result["label"],
              "evidence": [{key: item[key] for key in ("source_id", "title", "url", "relation")}
                           for item in result["evidence"]],
              "notes": result["notes"]}
    return record


@dataclass(frozen=True)
class ReductionCheck:
    outcome: str                # verified, refuted or unverified
    reasons: list[str]
    ids: list[str]
    report: dict | None         # the host's reading of the printed comparisons (None when none was read)


def types_of(comparator: dict) -> list[str]:
    """The transformation types of a declared comparator, sorted and without repeats (the form the report repeats)."""
    return sorted({t["type"] for t in comparator["transformations"]})


def eligibility(comparator: dict, reduces_to: str) -> str:
    """FIX-12: how a declared comparator's comparison counts, derived from the declaration and the repair policy
    (contracts.REPAIR_TYPES), never from a label the check program prints:
    - `another_family`: its base family is not the claimed `reduces_to`; it cannot decide that claim;
    - `unresolved`: a transformation type outside the policy (`other`); its family membership is not established, so it
      never verifies a reduction, and a match through it keeps the claim from being refuted;
    - `limit_only`: the declared equality holds only in a limit; it never verifies, and a measured difference at the
      card's parameters counts against the claim;
    - `counted`: the published form or a repair built only from policy types; it verifies within the tolerance and
      counts against the claim outside it."""
    if comparator["base_family"] != reduces_to:
        return "another_family"
    if any(t not in REPAIR_TYPES for t in types_of(comparator)):
        return "unresolved"
    if comparator["limit_only"]:
        return "limit_only"
    return "counted"


def declaration_problem(critic: dict) -> str | None:
    """Why the critic's declared comparators cannot bind a check (repeated ids), or None."""
    ids = [c["id"] for c in critic["comparators"]]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        return f"the critic declared comparator id(s) {', '.join(repeated)} more than once"
    return None


def _comparison_line(c: dict) -> str:
    types = ", ".join(c["transformations"]) or "as published"
    return (f"{c['declared_id'] or 'undeclared'}: {c['comparator'][:200]} (family {c['family']}; {types}"
            f"{'; limit only' if c['limit_only'] else ''}; {c['inputs_tested']} input(s)): "
            f"max_abs_diff {c['max_abs_diff']:.3g}")


def judge_comparisons(comparisons: list[dict], declared: list[dict],
                      reduces_to: str) -> tuple[str, list[str], list[dict]]:
    """FIX-10, FIX-12: equivalence is decided here, from the numbers the check program printed against the comparators
    the critic declared, never by the program. The report must already match the declaration (report_problem).

    Each comparison takes the eligibility of its declared comparator (`eligibility`); a comparison with declared_id
    null (a rewrite of the card, a sanity check) never counts. Verified: a `counted` comparison has a finite
    max_abs_diff at or below EQUIVALENCE_TOL. Otherwise unverified when an `unresolved` or `limit_only` comparison is
    within the tolerance (the match rests on a repair outside the policy, or on a limit), refuted when some `counted`
    or `limit_only` comparison was measured, and unverified when none was. Returns (outcome, reasons, comparisons
    with `eligibility`, `counted` and `within_tolerance` added)."""
    by_id = {c["id"]: c for c in declared}
    judged = []
    for c in comparisons:
        status = eligibility(by_id[c["declared_id"]], reduces_to) if c["declared_id"] is not None else "undeclared"
        diff = float(c["max_abs_diff"])
        judged.append(dict(c, eligibility=status, counted=status in ("counted", "limit_only"),
                           within_tolerance=math.isfinite(diff) and diff <= EQUIVALENCE_TOL))
    groups = {status: [c for c in judged if c["eligibility"] == status]
              for status in ("counted", "limit_only", "unresolved", "another_family", "undeclared")}
    tail = []
    if groups["unresolved"]:
        tail.append("unresolved (a transformation type outside the family policy, so family membership is not "
                    "established and the comparison cannot verify the reduction): "
                    + "; ".join(_comparison_line(c) for c in groups["unresolved"]))
    if groups["another_family"]:
        tail.append(f"not counted (declared in another family than {reduces_to}): "
                    + "; ".join(_comparison_line(c) for c in groups["another_family"]))
    if groups["undeclared"]:
        tail.append("not counted (no declared comparator; a rewrite of the card proves an identity, not a reduction): "
                    + "; ".join(_comparison_line(c) for c in groups["undeclared"]))
    within = [c for c in groups["counted"] if c["within_tolerance"]]
    if within:
        others = [c for c in judged if c["counted"] and c not in within]
        lines = [f"numeric check confirms equivalence to {reduces_to}: {_comparison_line(within[0])}, within the host "
                 f"tolerance {EQUIVALENCE_TOL:g}; its transformations are all in the family policy"]
        if others:
            lines.append(f"other comparisons with {reduces_to}: " + "; ".join(_comparison_line(c) for c in others))
        return "verified", lines + tail, judged
    limit_within = [c for c in groups["limit_only"] if c["within_tolerance"]]
    if limit_within:
        return "unverified", [f"only a limit-only comparator is within the tolerance, and equality in a limit does not "
                              f"verify a reduction: " + "; ".join(_comparison_line(c) for c in limit_within)] + tail, \
            judged
    if any(c["within_tolerance"] for c in groups["unresolved"]):
        return "unverified", [f"no comparator of {reduces_to} inside the family policy is within the tolerance; a "
                              f"repair outside the policy matches, so the reduction is unresolved"] + tail, judged
    against = groups["counted"] + groups["limit_only"]
    if against:
        return "refuted", [f"numeric check rejects the claimed reduction: every declared comparator of {reduces_to} "
                           f"differs by more than the host tolerance {EQUIVALENCE_TOL:g}: "
                           + "; ".join(_comparison_line(c) for c in against)] + tail, judged
    return "unverified", [f"the check compared no declared comparator of {reduces_to} inside the family policy, so "
                          f"it does not verify the reduction"] + tail, judged


def counted_comparisons(critic: dict, check_report: dict | None) -> list[dict]:
    """KD4: the comparators that carried a verified reduction, for the excluded-family gate. Each has the `family` its
    declaration stated (the report repeats it, report_problem), its `declared_id` and `comparator` text: the
    comparisons of the host's report whose eligibility is `counted` and that are within the tolerance. A comparison
    that was not counted (another family, unresolved, limit-only, undeclared) or that missed the tolerance never
    decides. With no report (verified on the critic's confidence alone, see run_reduction_check) nothing was compared,
    and the claimed `reduces_to` stands as the one counted family."""
    if check_report is None:
        return [{"declared_id": None, "family": critic["reduces_to"], "comparator": ""}]
    return [{"declared_id": c["declared_id"], "family": c["family"], "comparator": c["comparator"]}
            for c in check_report["comparisons"] if c["eligibility"] == "counted" and c["within_tolerance"]]


def report_problem(printed: dict, declared: list[dict]) -> str | None:
    """Why a printed object is not a comparison report bound to the declared comparators, or None.

    It must match contracts.REDUCTION_CHECK_REPORT; every declared comparator must be reported exactly once under its
    id (an omitted, repeated or undeclared id is refused); and each reported comparison must repeat its declaration's
    family, transformation types and limit_only (FIX-12)."""
    error = next(iter(jsonschema.Draft202012Validator(REDUCTION_CHECK_REPORT).iter_errors(printed)), None)
    if error is not None:
        return f"{'/'.join(map(str, error.absolute_path)) or 'report'}: {error.message[:300]}"
    by_id = {c["id"]: c for c in declared}
    reported = [c["declared_id"] for c in printed["comparisons"] if c["declared_id"] is not None]
    unknown = [i for i in reported if i not in by_id]
    if unknown:
        return f"comparison(s) under undeclared id(s) {', '.join(unknown)}; only the critic's declared comparators count"
    repeated = sorted({i for i in reported if reported.count(i) > 1})
    if repeated:
        return f"declared comparator(s) {', '.join(repeated)} reported more than once"
    omitted = [i for i in by_id if i not in reported]
    if omitted:
        return f"declared comparator(s) {', '.join(omitted)} not reported"
    for c in printed["comparisons"]:
        if c["declared_id"] is None:
            continue
        decl = by_id[c["declared_id"]]
        shown = {"family": c["family"], "transformations": sorted(set(c["transformations"])),
                 "limit_only": c["limit_only"]}
        expected = {"family": decl["base_family"], "transformations": types_of(decl), "limit_only": decl["limit_only"]}
        wrong = [key for key in expected if shown[key] != expected[key]]
        if wrong:
            return (f"comparison {c['declared_id']} disagrees with its declaration on "
                    + ", ".join(f"{key} (reported {shown[key]!r}, declared {expected[key]!r})" for key in wrong))
    return None


def family_policy_block(ctx: StageContext, reduces_to: str) -> str:
    """FIX-12: the family definition the check is judged by, for the check request: the families.toml forms that
    `reduces_to` maps to and the repair policy."""
    families, _ = difference.families_for(ctx)
    forms = [f for f in (*families.excluded, *families.known) if reduces_to in f.reduces_to]
    lines = [f"FAMILY POLICY (hybrid/config/families.toml; a family includes its published form and every repair built "
             f"only from the types of the repair policy). Families of {reduces_to}:"]
    lines += [f"- {f.id}: {f.name}. Form: {f.form}" for f in forms] or [f"- {reduces_to}: no families.toml entry"]
    return "\n".join(lines) + "\n\n" + repair_policy_text()


async def run_reduction_check(ctx: StageContext, card: dict, critic: dict) -> ReductionCheck:
    """The executed check of the critic's reduction claim. The program prints its comparisons as data
    (contracts.REDUCTION_CHECK_REPORT), each under the id of a comparator the critic declared; report_problem binds the
    report to the declaration and judge_comparisons decides. The critic's confidence stands in for a check only when
    there is none: no check code, or a program that printed no JSON object (a crash or a timeout). A printed object
    outside the contract or the declaration is unverified, whatever the confidence."""
    s2 = ctx.cfg["s2"]
    problem = declaration_problem(critic)
    if problem is not None:
        return ReductionCheck("unverified", [f"no check was run: {problem}"], [], {"problem": problem})
    user = "\n\n".join([
        "CARD:\n" + card_for_prompt(card),
        "YOUR REDUCTION CLAIM:\n" + json.dumps(critic, indent=2),
        family_policy_block(ctx, critic["reduces_to"]),
        "Write check_code now, or null if no numeric check is possible.",
    ]) + escape_user_suffix(ctx, card, "reduction_check_escape")
    request = await ctx.ask("reduction_critic", "reduction_check", user, REDUCTION_CHECK)
    ids = [request.call_id]
    code = request.parsed["check_code"]
    confidence = critic["confidence"]
    if not code:
        if confidence >= s2["reduction_confidence"]:
            return ReductionCheck("verified", [f"no numeric check possible ({request.parsed['explanation'][:300]}); "
                                               f"confidence {confidence} >= {s2['reduction_confidence']}"], ids, None)
        return ReductionCheck("unverified", [f"no numeric check and confidence {confidence} below "
                                             f"{s2['reduction_confidence']}"], ids, None)
    result = await ctx.runner.run(card["id"], "reduction_check", {"main.py": code})
    ids.append(result.job_id)
    printed = result.printed_json() if result.ok else None
    if printed is None:
        detail = (f"check printed no comparison report ({result.status}; "
                  f"{result.stderr[-400:] or result.stdout[-400:]})")
        if confidence >= s2["reduction_confidence"]:
            return ReductionCheck("verified", [detail + f"; confidence {confidence} counts as verified"], ids, None)
        return ReductionCheck("unverified", [detail], ids, None)
    declared = critic["comparators"]
    problem = report_problem(printed, declared)
    if problem is not None:
        return ReductionCheck("unverified", [f"check printed a result outside the comparison contract ({problem}); "
                                             f"the host decides equivalence only from comparisons with the declared "
                                             f"comparators"], ids, {"job_id": result.job_id, "problem": problem})
    outcome, lines, judged = judge_comparisons(printed["comparisons"], declared, critic["reduces_to"])
    report = {"job_id": result.job_id, "reduces_to": critic["reduces_to"], "tolerance": EQUIVALENCE_TOL,
              "outcome": outcome,
              "declared": [{"id": c["id"], "base_family": c["base_family"], "transformations": types_of(c),
                            "limit_only": c["limit_only"], "eligibility": eligibility(c, critic["reduces_to"])}
                           for c in declared],
              "comparisons": judged, "notes": printed["notes"]}
    return ReductionCheck(outcome, lines, ids, report)


# ----- F3 (FIX-12): what the refinement dialogue changed in the cause-bearing equations ---------------------------------

UNCHANGED, ATTEMPTED, VALIDATED = "unchanged", "attempted_repair", "validated_repair"


def equation_slots(card: dict) -> dict[str, str]:
    """The card's cause-bearing equations by location (`mechanism.update[i]`, `mechanism.readout[i]`,
    `mechanism.learning`), each as the text that decides whether it changed: the equation itself (plain and latex),
    not its name."""
    mechanism = card["mechanism"]
    slots = {}
    for kind in ("update", "readout"):
        for index, entry in enumerate(mechanism[kind]):
            slots[f"mechanism.{kind}[{index}]"] = (f"{entry.get('plain') or ''}\n{entry.get('latex') or ''}".strip()
                                                   if isinstance(entry, dict) else str(entry))
    learning = mechanism.get("learning")
    slots["mechanism.learning"] = learning if isinstance(learning, str) else json.dumps(learning, sort_keys=True)
    return slots


def _shown(card: dict, slot: str) -> str:
    kind, _, rest = slot.removeprefix("mechanism.").partition("[")
    if kind == "learning":
        return equation_slots(card)[slot]
    entry = card["mechanism"][kind][int(rest.rstrip("]"))]
    return (entry.get("plain") or entry.get("latex") or "") if isinstance(entry, dict) else str(entry)


def equation_changes(ctx: StageContext, card: dict) -> dict | None:
    """The cause-bearing equations before stage R (the version recorded at `refine_start`) and now, each with a status:
    `unchanged`; `attempted_repair` (changed by an accepted R revision that no later challenger round examined);
    `validated_repair` (changed, and a later challenger round examined the revised card and asked nothing more at that
    equation). None when the card had no refinement dialogue."""
    starts = ctx.card_events(card["id"], "refine_start")
    if not starts:
        return None
    before_version = starts[0]["card_version"]
    accepted = sorted((a for a in ctx.card_events(card["id"], "refine_answer") if a["accepted"]),
                      key=lambda a: a["round"])
    rounds = ctx.card_events(card["id"], "refine_round")
    chain = [(0, ctx.store.get_card(card["id"], before_version))]
    chain += [(a["round"], ctx.store.get_card(card["id"], a["version"])) for a in accepted]
    before, now = equation_slots(chain[0][1]), equation_slots(card)
    rows = []
    for slot in dict.fromkeys([*before, *now]):
        if before.get(slot) == now.get(slot):
            rows.append({"where": slot, "status": UNCHANGED})
            continue
        changed_in = [r for (r, older), (_, newer) in zip(chain, chain[1:])
                      if equation_slots(older).get(slot) != equation_slots(newer).get(slot)]
        last = changed_in[-1] if changed_in else None
        examined = [r["round"] for r in rounds if last is not None and r["round"] > last
                    and not any(str(q.get("where", "")).startswith(slot) for q in r["questions"])]
        rows.append({"where": slot, "status": VALIDATED if examined else ATTEMPTED, "changed_in_round": last,
                     "examined_in_round": examined[0] if examined else None,
                     "before": _shown(chain[0][1], slot) if slot in before else None,
                     "now": _shown(card, slot) if slot in now else None})
    return {"before_version": before_version, "now_version": card["version"], "equations": rows}


def render_changes(changes: dict | None) -> str:
    if changes is None:
        return "EQUATION CHANGES IN STAGE R: the card had no refinement dialogue."
    lines = [f"EQUATION CHANGES IN STAGE R (card version {changes['before_version']} before the dialogue, version "
             f"{changes['now_version']} now). {UNCHANGED}: the same equation before and after R. {ATTEMPTED}: changed by "
             f"an accepted R revision that no later challenger round examined. {VALIDATED}: changed, and a later "
             f"challenger round examined the revised card and asked nothing more about it. No status shows that a "
             f"recorded cause is resolved; only an executed test of that cause would."]
    for row in changes["equations"]:
        if row["status"] == UNCHANGED:
            lines.append(f"- {row['where']}: {UNCHANGED}")
            continue
        when = (f"changed in round {row['changed_in_round']}" if row["changed_in_round"] is not None
                else "changed during R")
        if row["examined_in_round"] is not None:
            when += f", examined in round {row['examined_in_round']}"
        lines.append(f"- {row['where']}: {row['status']} ({when})\n    before R: {row['before'] or '(absent)'}\n"
                     f"    now: {row['now'] or '(removed)'}")
    return "\n".join(lines)


def own_history(ctx: StageContext, card: dict) -> tuple[list, str]:
    """Step 3 query: earlier outcomes with this card's signature and its nearest neighbours. Records them (artifact
    and event) and returns (matches, the block for the reduction critic). The block also shows what stage R changed in
    the cause-bearing equations (FIX-12, F3), so the critic's history judgment can say which recorded cause an
    equation change answers."""
    cfg = ctx.cfg["outcomes"]
    signature = card_signature(card)
    matches = outcome_store.own_history(ctx.outcomes, card, signature, current_run=ctx.run_id,
                                        limit=cfg["history_max"], jaccard_min=cfg["jaccard_min"],
                                        tfidf_min=cfg["tfidf_min"])
    found = [match.as_dict() for match in matches]
    changes = equation_changes(ctx, card)
    ctx.put_json_artifact("own_history", ctx.artifact_name(card["id"]),
                          {"card_id": card["id"], "signature": signature, "matches": found,
                           "equation_changes": changes}, card["id"])
    ctx.emit("own_history", {"card_id": card["id"], "signature": signature["signature"],
                             "exact": [m["record_id"] for m in found if m["exact_signature"]],
                             "near": [m["record_id"] for m in found if not m["exact_signature"]]})
    return matches, ("\n\n" + load_section("own_history") + "\n" + outcome_store.render_history(matches) + "\n\n"
                     + render_changes(changes))


def history_reason(matches: list) -> str:
    exact = [m for m in matches if m.exact]
    rows = [f"{m.stored.record['card_id']} ({', '.join(m.stored.record['kill_codes']) or m.stored.record['outcome']})"
            for m in matches[:5]]
    return (f"own history: {len(exact)} exact signature match(es), {len(matches) - len(exact)} near: "
            + ("; ".join(rows) or "none"))


async def run(ctx: StageContext, card_id: str) -> str | None:
    card = ctx.card(card_id)
    pa_result, pa_calls, unassessed = await prior_art_checked(ctx, card)
    record = prior_art_record(pa_result)
    if unassessed:
        record["label"] = NOVELTY_UNASSESSED
    calls = list(pa_calls)
    reasons: list[str] = [f"prior art: {record['label']}. {record['notes'][:400]}"]
    try:
        validate("prior_art", record)
    except SchemaViolation as exc:
        reasons.append(f"prior-art label has no grounded evidence ({exc}); label kept, evidence missing")
    ctx.put_json_artifact("prior_art", ctx.artifact_name(card_id),
                          dict(record, queries=pa_result["queries"], rejected_evidence=pa_result["rejected_evidence"],
                               candidates=pa_result["candidates"]), card_id)

    delta = None
    if record["label"] == "prior_art_tested":
        user = "\n\n".join(["CARD:\n" + card_for_prompt(card),
                            "PRIOR ART FOUND:\n" + json.dumps(record, indent=2)])
        delta_call = await ctx.ask("prior_art", "prior_art_delta", user, PRIOR_ART_DELTA)
        calls.append(delta_call.call_id)
        delta = delta_call.parsed

    matches, history_block = own_history(ctx, card) if ctx.outcomes is not None else ([], "")
    if ctx.outcomes is not None:
        reasons.append(history_reason(matches))
    user = "\n\n".join(["CARD:\n" + card_for_prompt(card),
                        "PRIOR ART CLASSIFICATION:\n" + json.dumps(record, indent=2), repair_policy_text()])
    user += escape_user_suffix(ctx, card, "reduction_critic_escape") + history_block
    s2 = ctx.cfg["s2"]
    exclusion, statement_on = s2["family_exclusion"], s2["difference_statement"]
    families, keywords = difference.families_for(ctx) if (exclusion or statement_on) else (None, {})
    schema = load_schema("reduction_critic")
    if exclusion:
        user += difference.critic_section(families)
        schema = difference.critic_schema(families)
    critic_call = await ctx.ask("reduction_critic", "reduction_critic", user, schema)
    critic = critic_call.parsed
    calls.append(critic_call.call_id)
    reasons.append(f"textbook transplant: {'yes' if critic['textbook_transplant'] else 'no'} (recorded, not gated): "
                   f"{critic['textbook_why'][:300]}")
    reduction, check_report = "none", None
    if critic["reduces_to"] != "none":
        checked = await run_reduction_check(ctx, card, critic)
        reduction, check_report = checked.outcome, checked.report
        calls += checked.ids
        reasons += [f"reduction claim {critic['reduces_to']} ({reduction}): " + r for r in checked.reasons]
    ctx.put_json_artifact("reduction_critic", ctx.artifact_name(card_id),
                          dict(critic, reduction_outcome=reduction, reduction_check=check_report,
                               prior_art_delta=delta), card_id)

    if record["label"] == "known_disproven":
        ctx.verdict(card_id, "S2", "fail", reasons + ["gate: known_disproven fails"], calls)
        return None
    learning_rule_difference = critic["difference_if_any"].strip().startswith(LEARNING_RULE_MARK)
    counted = counted_comparisons(critic, check_report) if reduction == "verified" else []
    excluded = difference.excluded_reduction(critic, reduction, families, keywords, counted) if exclusion else None
    if excluded is not None:
        ids = [c["declared_id"] for c in counted if c["declared_id"]]
        note = f"; counted comparator{'s' if len(ids) > 1 else ''}: {', '.join(ids)}" if ids else ""
        ctx.verdict(card_id, "S2", "reduces_to_known",
                    reasons + [f"gate: excluded_family:{excluded.id}: the update rule reduces to {critic['reduces_to']} "
                               f"(check {reduction}{note}), which is in the out-of-scope family {excluded.name} "
                               f"(families.toml, decision D8); a LEARNING_RULE: difference does not apply: "
                               f"{(critic['mapping'] or '')[:500]}"], calls)
        if ctx.on_reduction is not None:
            ctx.on_reduction(card_id)
        return None
    if reduction == "verified":
        if not learning_rule_difference:
            ctx.verdict(card_id, "S2", "reduces_to_known",
                        reasons + [f"gate: update rule reduces to {critic['reduces_to']}: {critic['mapping'][:600]}"],
                        calls)
            if ctx.on_reduction is not None:
                ctx.on_reduction(card_id)
            return None
        reasons.append(f"update rule reduces to {critic['reduces_to']}, difference is in the learning rule: "
                       f"{critic['difference_if_any'][:400]}")
    if ctx.outcomes is not None and ctx.cfg["outcomes"]["enforce_r3"] and not statement_on:
        problems = outcome_store.r3_problems(card, matches)
        if problems:
            ctx.verdict(card_id, "S2", "fail", reasons + ["gate: not_radical:R3: " + "; ".join(problems)[:1500]],
                        calls)
            return None
    if record["label"] == NOVELTY_UNASSESSED:
        ctx.verdict(card_id, "S2", "undetermined",
                    reasons + ["novelty unassessed: literature search returned no papers"], calls)
        return None
    if record["label"] == "prior_art_tested":
        if not delta["adds_untested_element"]:
            ctx.verdict(card_id, "S2", "fail",
                        reasons + [f"gate: prior_art_tested and no untested element ({delta['why'][:300]})"], calls)
            return None
        reasons.append(f"prior art tested, card adds an untested element: {delta['element'][:300]}")
    if ctx.cfg["s2"]["require_radical"] and not critic["radical_enough"]:
        ctx.verdict(card_id, "S2", "fail", reasons + [f"gate: not radical enough: {critic['why'][:500]}"], calls)
        return None
    reasons.append(f"reduction critic: {critic['reduces_to']}, radical_enough {critic['radical_enough']}: "
                   f"{critic['why'][:400]}")
    if exclusion:
        reasons.append(f"family (reduction critic): {critic['family']}: {critic['family_why'][:300]}")
    if statement_on:
        roles = (explorer_role(ctx, card_id), challenger_role(proposer_model(ctx, card_id)))
        code, more, ids = await difference.gate(ctx, card, record=record, pa_result=pa_result, matches=matches,
                                                critic=critic, families=families, roles=roles)
        reasons += more
        calls += ids
        if code is not None:
            ctx.verdict(card_id, "S2", "fail", [f"gate: {code}: " + (more[0] if more else "")[:1200], *reasons],
                        calls)
            return None
    ctx.verdict(card_id, "S2", "pass", reasons, calls)
    return "S3"
