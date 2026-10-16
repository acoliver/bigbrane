"""S6 verdict: the claims ledger. Falsifier rules decide claims mechanically; the analyst comments on every claim.

Each card claim gets a stable id (falsify.claim_items). For a claim named by one or more pre-registered
falsifier rules, the status is computed from the evaluator's rule decisions (bench_result `falsifiers`):
supported or refuted when the rules that reached a decision agree, inconclusive when none did (the candidate
did not train, or a named entrant has no valid predictions) or when they disagree. The analyst sees the claims,
the falsifier tasks with their rules and decisions, and the bench JSON verbatim; its text is kept as commentary
on every claim, and its own reading as `analyst_status`. The ledger records, per claim, whether a rule or the
analyst decided it.

A rule decides only the claims it names, and a card falsifier the designer called untestable is named by no
rule even when a task ran its protocol. The link from such a claim to the task is the analyst's own: every
analyst entry lists, in `falsifier_tasks`, the falsifier task ids whose numbers it uses (an enum of the task ids
S5 recorded). A task where the candidate decided nothing (no valid predictions, a protocol that does not fit
the candidate, or no training: `task_gate`) gates every claim whose entry uses it. The analyst's supported or
refuted on such a claim becomes inconclusive, decided by the rule, with the task's reason. Whenever a rule or a
gate overrides an analyst decision, the analyst's reading is kept as `analyst_override`, a note and not a
decision. The analyst decides only claims with no rule and no gated task.
"""
from __future__ import annotations

import json

from hybrid.research.schemas import validate
from hybrid.research.stages.common import StageContext, StageError
from hybrid.research.stages.contracts import analyst_schema
from hybrid.research.stages.falsify import claim_items

STATUSES = ("supported", "refuted", "inconclusive", "untested")
DECISIONS = ("supported", "refuted")

SOURCES_NOTE = (
    "FALSIFIER TASK SOURCES: in every entry, falsifier_tasks lists the ids of the falsifier tasks whose numbers the "
    "entry uses, and is empty when it uses only the standard bench tasks or no numbers. A task marked 'candidate "
    "gate' above decides nothing for this candidate (it did not train there, produced no valid predictions, or the "
    "protocol does not fit it): the ledger records any claim whose entry uses that task as inconclusive, and your "
    "status is kept as a note. Give your own reading anyway.")


def bench_raw_text(ctx: StageContext, card_id: str) -> str:
    raws = [a for a in ctx.card_events(card_id, "artifact") if a["kind"] == "bench_raw"]
    if not raws:
        raise StageError(f"{card_id} has no bench_raw artifact")
    return (ctx.run_dir / raws[-1]["path"]).read_text(encoding="utf-8")


def rule_records(falsifiers: dict) -> dict[str, list[dict]]:
    """{claim id: [decision of every rule that names it]} from the bench result's falsifiers block."""
    out: dict[str, list[dict]] = {}
    for task_id, block in falsifiers.items():
        for decision in block["decisions"]:
            record = {"task_id": task_id, "rule_index": decision["rule_index"], "outcome": decision["outcome"],
                      "reason": decision["reason"], "quote": decision["quote"],
                      "conditions": decision["clauses"], "if_met": decision["rule"]["if_met"]}
            for claim_id in decision["claims"]:
                out.setdefault(claim_id, []).append(record)
    return out


def task_gate(block: dict) -> str | None:
    """Why the candidate's numbers on this falsifier task decide nothing, or None.

    falsifier_eval.decide checks three things for the whole task before any rule: the candidate has valid
    predictions, the protocol fits the candidate (task_blocked), and the candidate trained (beats the better
    trivial reference). The first two return without a `training` record and the third records
    `training.trained` false; each gives every rule of the task the same inconclusive decision. A rule's own
    gates (control rule, a named entrant without predictions) come after training and do not gate the task."""
    for decision in block["decisions"]:
        training = decision.get("training")
        if decision["outcome"] == "inconclusive" and (training is None or not training["trained"]):
            return decision["reason"]
    return None


def task_gates(falsifiers: dict) -> dict[str, str]:
    """{task id: gate reason} for every falsifier task where the candidate decided nothing."""
    gates = {task_id: task_gate(block) for task_id, block in falsifiers.items()}
    return {task_id: reason for task_id, reason in gates.items() if reason is not None}


def combine(records: list[dict]) -> tuple[str, str]:
    """Status of a claim from its rules, and a note saying how it was reached."""
    decisive = sorted({r["outcome"] for r in records if r["outcome"] != "inconclusive"})
    names = ", ".join(f"{r['task_id']} rule {r['rule_index']}: {r['outcome']}" for r in records)
    if len(decisive) == 1:
        return decisive[0], f"decided by rule ({names})"
    if decisive:
        return "inconclusive", f"rules disagree ({names})"
    return "inconclusive", f"no rule reached a decision ({names})"


def reading(claim_id: str, status: str) -> str:
    """The analyst's status in the claim's own terms: a refuted falsifier is one that fired."""
    if claim_id.startswith("falsifier."):
        return {"refuted": "fired", "supported": "not fired"}[status]
    return status


def override_note(claim_id: str, analyst_status: str, by: str) -> str:
    return f"analyst would have marked {reading(claim_id, analyst_status)}; overridden by rule: {by}"


def claim_line(entry: dict) -> str:
    """One verdict line for a ledger entry. Falsifier claims read by meaning: supported is "did not fire"."""
    note = (entry.get("analyst_override") or entry.get("decision_note") or entry["evidence"])[:300]
    claim_id, text = entry["claim_id"], entry["claim"]
    if claim_id.startswith("falsifier."):
        number = claim_id.split(".", 1)[1]
        text = text.split(": ", 1)[1] if text.startswith(f"falsifier {number}: ") else text
        wording = {"supported": f"falsifier {number} did not fire (the card survives on this point)",
                   "refuted": f"falsifier {number} fired, card refuted on this point",
                   "inconclusive": f"falsifier {number} inconclusive (neither fired nor ruled out)",
                   "untested": f"falsifier {number} untested"}[entry["status"]]
        return f"{wording} [{entry['decided_by']}]: {text[:200]} ({note})"
    return f"{entry['status']} [{entry['decided_by']}]: {text[:200]} ({note})"


def falsified_by(ledger: list[dict]) -> list[dict]:
    """Falsifiers that a pre-registered rule found fired: {claim_id, task_id, rule_index} per deciding rule."""
    return [{"claim_id": c["claim_id"], "task_id": r["task_id"], "rule_index": r["rule_index"]}
            for c in ledger if c["claim_id"].startswith("falsifier.") and c["status"] == "refuted"
            and c["decided_by"] == "rule" for r in c["rules"] if r["outcome"] == "refuted"]


def falsifier_section(falsifiers: dict) -> str:
    if not falsifiers:
        return "No falsifier task ran for this card."
    parts = []
    for task_id, block in falsifiers.items():
        entrants = {name: (round(e["metrics"]["acc"], 4), round(e["metrics"]["kl"], 4)) if e["status"] == "ok"
                    else e["status"] for name, e in block["entrants"].items()}
        lines = [f"TASK {task_id}: {block['title']}", f"protocol: {block['protocol']}",
                 f"deviations from the card: {block['deviations']}",
                 f"overall (acc, kl) per entrant: {json.dumps(entrants)}"]
        gate = task_gate(block)
        if gate is not None:
            lines.append(f"candidate gate: {gate}; no claim can be decided from this task's numbers")
        for decision in block["decisions"]:
            lines.append(f"rule {decision['rule_index']} for {', '.join(decision['claims'])}, quoting "
                         f"{decision['quote']!r}: conditions {json.dumps(decision['clauses'])}, if met the claims are "
                         f"{decision['rule']['if_met']}; decision: {decision['outcome']} ({decision['reason']})")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def merge(card_id: str, claims: list[dict], analyst: dict, falsifiers: dict) -> dict:
    by_id: dict[str, dict] = {}
    for entry in analyst["claims"]:
        by_id.setdefault(entry["claim_id"], entry)
    rules = rule_records(falsifiers)
    gates = task_gates(falsifiers)
    ledger = []
    for item in claims:
        comment = by_id.get(item["id"])
        analyst_status = comment["status"] if comment else None
        cited = list(comment["falsifier_tasks"]) if comment else []
        unknown = [task_id for task_id in cited if task_id not in falsifiers]
        if unknown:
            raise ValueError(f"analyst entry for {item['id']} names falsifier tasks S5 did not run: {unknown}")
        entry = {"claim_id": item["id"], "claim": item["text"], "rules": rules.get(item["id"], []),
                 "analyst_status": analyst_status, "analyst_tasks": cited,
                 "evidence": comment["evidence"] if comment else "", "gated_by": []}
        gated = [{"task_id": task_id, "reason": gates[task_id]} for task_id in cited if task_id in gates]
        if entry["rules"]:
            entry["status"], entry["decision_note"] = combine(entry["rules"])
            entry["decided_by"] = "rule"
            by = "; ".join(f"{r['task_id']} rule {r['rule_index']}: {r['outcome']} ({r['reason']})"
                           for r in entry["rules"])
        elif gated and analyst_status in DECISIONS:
            entry["status"], entry["decided_by"], entry["gated_by"] = "inconclusive", "rule", gated
            by = "; ".join(f"{g['task_id']}: {g['reason']}" for g in gated)
            entry["decision_note"] = f"the analyst's reading uses falsifier tasks that decide nothing ({by})"
        else:
            entry["status"] = analyst_status or "untested"
            entry["decided_by"] = "analyst"
            if comment is None:
                entry["decision_note"] = "the analyst gave no entry for this claim"
        if entry["decided_by"] == "rule" and analyst_status in DECISIONS and analyst_status != entry["status"]:
            entry["analyst_override"] = override_note(item["id"], analyst_status, by)
        ledger.append(entry)
    counts = {status: sum(1 for c in ledger if c["status"] == status) for status in STATUSES}
    return validate("claims_ledger", {"card_id": card_id, "claims": ledger, "counts": counts,
                                      "falsifier_tasks": sorted(falsifiers), "falsified_by": falsified_by(ledger)})


async def run(ctx: StageContext, card_id: str) -> str | None:
    card = ctx.card(card_id)
    claims = claim_items(card)
    raw_text = bench_raw_text(ctx, card_id)
    falsifiers = json.loads(raw_text).get("falsifiers") or {}
    user = "\n\n".join([
        f"CARD {card_id}: {card['title']}\n{card['summary']}",
        "CLAIM IDS (write one ledger entry per id):\n" + "\n".join(f"- {c['id']}: {c['text']}" for c in claims),
        "CLAIMS AND FALSIFIERS:\n" + json.dumps({"claims": card["claims"], "falsifiers": card["falsifiers"]},
                                                indent=2),
        "FALSIFIER TASKS AND PRE-REGISTERED RULES (decided by the evaluator):\n" + falsifier_section(falsifiers),
        SOURCES_NOTE,
        "BENCH RESULT JSON (verbatim):\n" + raw_text,
    ])
    schema = analyst_schema([c["id"] for c in claims], sorted(falsifiers))
    result = await ctx.ask("analyst", "analyst", user, schema)
    ledger = merge(card_id, claims, result.parsed, falsifiers)
    ctx.put_json_artifact("claims_ledger", ctx.artifact_name(card_id), ledger, card_id)
    counts = ledger["counts"]
    reasons = [f"card falsified by {f['claim_id']} (task {f['task_id']} rule {f['rule_index']})"
               for f in ledger["falsified_by"]]
    reasons += [f"claims ledger: {counts['supported']} supported, {counts['refuted']} refuted, "
               f"{counts['inconclusive']} inconclusive, {counts['untested']} untested; "
               f"{sum(1 for c in ledger['claims'] if c['decided_by'] == 'rule')} decided by falsifier rules"]
    reasons += [claim_line(c) for c in ledger["claims"]]
    ctx.verdict(card_id, "S6", "pass", reasons, [result.call_id])
    mutable = sum(1 for c in ledger["claims"] if c["status"] == "supported" and not c["claim_id"].startswith("falsifier."))
    if mutable > 0 and ctx.on_mutation is not None:
        ctx.on_mutation(card_id)
    return None
