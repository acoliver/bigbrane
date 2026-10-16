"""The escape operator: context about a card's S2 reduction, for the escape explorer and the S2 critic.

When S2 gives `reduces_to_known`, the orchestrator queues an `escape` task. The escape explorer gets the
parent card, the reduction critic's mapping and the numeric check, and must return a card with a
`non_reduction_argument`. When such a card reaches S2, the reduction critic and its check are told to test
that argument against the strongest repaired version of the parent's mapping.

Prompt text lives in prompts/explorer_escape.md (appended to the explorer system prompt for escape tasks
only), prompts/reduction_critic_escape.md and prompts/reduction_check_escape.md (appended to the S2 user
messages for cards that carry the argument). The existing prompt files are not changed by this operator.
"""
from __future__ import annotations

import json

from hybrid.research.ledger import Ledger
from hybrid.research.stages.common import ESCAPE_FIELD, ExecResult, StageContext, load_prompt

# How much of the parent's reduction check program the escape explorer and the S2 critic see.
CHECK_CODE_CHARS = 6000
MAPPING_CHARS = 3000
CRITIC_FIELDS = ("reduces_to", "mapping", "confidence", "difference_if_any", "why", "reduction_outcome")


def escape_chain(ledger: Ledger, parent_id: str) -> list[str]:
    """The parent, then each ancestor reached by consecutive escape steps: a card made by escape from a card
    that was itself made by escape reuses that card's vocabulary as well. The walk stops at the first
    ancestor not made by escape, which is included; its own parents are not on the chain."""
    created = {event["data"]["card_id"]: event["data"] for event in ledger.iter_events("card_created")}
    chain = [parent_id]
    while created[chain[-1]]["op"] == "escape":
        chain.append(created[chain[-1]]["parents"][0])
    return chain


def reduction_record(ctx: StageContext, card_id: str) -> dict | None:
    """The card's latest S2 reduction critic output (with reduction_outcome), or None before S2."""
    return ctx.latest_json_artifact(card_id, "reduction_critic")


def reduction_check_record(ctx: StageContext, card_id: str) -> dict | None:
    """The card's latest executed reduction check as {status, printed, check_code}, or None if none ran."""
    for data in reversed(ctx.card_events(card_id, "artifact")):
        if data["kind"] != "executor":
            continue
        record = json.loads((ctx.run_dir / data["path"]).read_text(encoding="utf-8"))
        if record["purpose"] != "reduction_check":
            continue
        result = ExecResult(record["job_id"], record["status"], record["detail"], record["stdout"],
                            record["stderr"], record["run_s"])
        return {"job_id": record["job_id"], "status": record["status"],
                "printed": result.printed_json() if result.ok else None,
                "check_code": record["files"]["main.py"]}
    return None


def s2_reduction_reasons(card: dict) -> list[str]:
    verdict = card["status"]["verdicts"].get("S2")
    if verdict is None:
        return []
    return [reason for reason in verdict["reasons"]
            if reason.startswith("reduction claim") or reason.startswith("gate: update rule reduces to")]


def _critic_view(record: dict) -> dict:
    shown = {key: record[key] for key in CRITIC_FIELDS if key in record}
    if isinstance(shown.get("mapping"), str):
        shown["mapping"] = shown["mapping"][:MAPPING_CHARS]
    return shown


def _check_block(check: dict | None) -> str:
    if check is None:
        return "No numeric check program ran; the reduction was accepted on the critic's confidence."
    printed = json.dumps(check["printed"]) if check["printed"] is not None else "(no JSON verdict printed)"
    code = check["check_code"]
    clipped = code if len(code) <= CHECK_CODE_CHARS else code[:CHECK_CODE_CHARS] + "\n# ... (truncated)"
    return (f"job {check['job_id']}, status {check['status']}, printed: {printed}\n"
            "The program below implements the parent card and the known form built through the mapping, and "
            "compares them step by step. Read it to see exactly which property of the parent the mapping "
            f"uses.\n```python\n{clipped}\n```")


def parent_reduction_block(ctx: StageContext, parent_id: str) -> str:
    """The S0 user-message section for an escape task: the parent's reduction and its check result."""
    record = reduction_record(ctx, parent_id)
    if record is None:
        raise ValueError(f"escape parent {parent_id} has no reduction_critic artifact")
    parent = ctx.card(parent_id)
    reasons = s2_reduction_reasons(parent)
    return "\n\n".join([
        f"REDUCTION OF THE PARENT ({parent_id}, S2 reduction critic):\n"
        + json.dumps(_critic_view(record), indent=2, ensure_ascii=False),
        "S2 VERDICT ON THE REDUCTION:\n" + ("\n".join(f"- {reason[:900]}" for reason in reasons) or "(none)"),
        "NUMERIC CHECK OF THE REDUCTION:\n" + _check_block(reduction_check_record(ctx, parent_id)),
    ])


def escape_user_suffix(ctx: StageContext, card: dict, section: str) -> str:
    """Text appended to an S2 user message (critic or check) for a card with a non_reduction_argument:
    the section file, the argument, and the parent's reduction when the parent has one. Empty otherwise."""
    argument = card.get(ESCAPE_FIELD)
    if argument is None:
        return ""
    parts = [load_prompt(section).strip(),
             "THE CARD'S NON-REDUCTION ARGUMENT:\n" + json.dumps(argument, indent=2, ensure_ascii=False)]
    for parent_id in card["parent_ids"]:
        record = reduction_record(ctx, parent_id)
        if record is not None:
            parts.append(f"THE PARENT'S REDUCTION ({parent_id}), THE MAPPING TO REPAIR:\n"
                         + json.dumps(_critic_view(record), indent=2, ensure_ascii=False))
    return "\n\n" + "\n\n".join(parts)
