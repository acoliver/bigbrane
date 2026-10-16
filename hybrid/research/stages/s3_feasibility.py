"""S3 feasibility: formalizer obligations, disprover attacks with executor checks, executed obligation checks,
two judges.

Gate (RESEARCH-SYSTEM-SPEC.md section 3, plus the obligation checks of `obligation_checks.py`). The rule that
decided is recorded as `gate rule: <rule>` in the verdict reasons and in an `s3_gate` artifact:
- `disproof`: an attack has status "succeeds" and its check code ran and printed {"disproved": true}. Gives
  `infeasible_proved`.
- `check_violation`: an obligation check printed `violated` with a counterexample, the replay printed the same,
  and the formalizer did not mark the obligation optional. Gives `infeasible_proved`.
- `check_settles_split`: the judges vote differently and disagree about an obligation a check decided. Every
  check-decided obligation takes the check's status (discharged, or refuted for a violation) in both judges'
  lists, and each judge's vote is derived again from its list (refuted: infeasible, open: undetermined, else
  feasible). A judge whose vote does not follow from its own list keeps its vote. The derived votes then go
  through the judge rules below.
- `judges`: both feasible gives `pass`; both infeasible gives `infeasible_proved` when an attack survived or an
  obligation was refuted, `fail` otherwise; anything else gets s3.extra_rounds more rounds, then `undetermined`.
  The `s3_gate` artifact says whether every checkable obligation was discharged by a check.
Judges see the card, the obligations, the attack results and the obligation checks, never each other's verdict.
A card that passed S1 with residual fixable findings (artifact `s1_residual`) has them appended to the
formalizer and disprover user messages under "known open issues from S1".
"""
from __future__ import annotations

import asyncio
import json

from hybrid.research.schemas import load_schema
from hybrid.research.stages import obligation_checks as oc
from hybrid.research.stages.common import StageContext, card_for_prompt
from hybrid.research.stages.contracts import FORMALIZER

JUDGES = ("feasibility_judge_a", "feasibility_judge_b")


async def run_attack_checks(ctx: StageContext, card_id: str, attacks: list[dict]) -> tuple[list[dict], list[str]]:
    results, ids = [], []
    budget = ctx.cfg["s3"]["max_checks"]
    for attack in attacks:
        entry = dict(attack)
        code = attack.get("check_code")
        if code and budget > 0:
            budget -= 1
            result = await ctx.runner.run(card_id, "disproof_check", {"main.py": code})
            ids.append(result.job_id)
            printed = result.printed_json() if result.ok else None
            entry["check"] = {"job_id": result.job_id, "status": result.status,
                              "printed": printed, "stdout_tail": result.stdout[-1500:],
                              "stderr_tail": result.stderr[-800:]}
            disproved = printed.get("disproved") if printed else None
            entry["confirmed"] = attack["status"] == "succeeds" and disproved is True
            entry["check_refutes_attack"] = disproved is False
        else:
            entry["check"] = None if not code else {"status": "skipped", "reason": "max_checks reached"}
            entry["confirmed"] = False
            entry["check_refutes_attack"] = False
        results.append(entry)
    return results, ids


def disproof_reason(attack: dict) -> str:
    """The verdict reason of a confirmed disproof: the attacked claim and the counterexample, whole. The outcome record
    of an `infeasible:disproof` death rebuilds its cause with this function from the `disproof` artifact."""
    return f"confirmed disproof of '{attack['target_claim']}': {attack['counterexample']}"


def judge_prompt(card: dict, obligations: list[dict], attack_results: list[dict], round_no: int,
                 checks: list[dict] | None = None) -> str:
    parts = ["CARD:\n" + card_for_prompt(card),
             "PROOF OBLIGATIONS (from the formalizer):\n" + json.dumps(obligations, indent=2),
             "DISPROOF ATTEMPTS AND EXECUTED CHECKS:\n" + json.dumps(attack_results, indent=2)]
    if checks:
        parts.append(oc.judge_section(checks))
    if round_no > 1:
        parts.append(f"This is judging round {round_no}: an earlier round did not reach agreement. "
                     "Look again at every obligation you marked open and decide whether the card's "
                     "own text discharges it.")
    return "\n\n".join(parts)


def known_open_issues(ctx: StageContext, card_id: str) -> str:
    """Text for the formalizer and disprover user messages: the fixable findings S1 passed the card with.
    Empty when the card passed S1 clean. It goes in the user message, so the prompt files stay as they are."""
    residual = ctx.latest_json_artifact(card_id, "s1_residual")
    if not residual or not residual["findings"]:
        return ""
    lines = [f"- {item['kind']} ({item['where']}): {item['problem']}" for item in residual["findings"]]
    return ("known open issues from S1 (fixable findings the salad critic still had after the last revision; "
            "the card passed S1 with them, so treat each as an open point to check rather than as settled):\n"
            + "\n".join(lines))


# ----- the gate on the obligation checks ---------------------------------------------------------


def derived_vote(statuses: list[str]) -> str:
    """The vote the judge prompt's definitions give for a list of obligation statuses."""
    if "refuted" in statuses:
        return "infeasible"
    if "open" in statuses:
        return "undetermined"
    return "feasible"


def settle_split(judged: dict[str, dict], checks: list[dict]) -> dict:
    """Apply `check_settles_split` when the judges vote differently and disagree about a check-decided obligation.

    Returns {"applied", "contested", "votes" (effective, per judge), "obligations" (effective lists),
    "notes"}. When not applied, the effective votes and lists are the judges' own."""
    votes = {role: j["verdict"] for role, j in judged.items()}
    lists = {role: [dict(o) for o in j["obligations"]] for role, j in judged.items()}
    result = {"applied": False, "contested": [], "votes": votes, "obligations": lists, "notes": []}
    decided = oc.decided_statuses(checks)
    if len(set(votes.values())) == 1 or not decided:
        return result
    by_name = {role: {o["name"].strip(): o["status"] for o in j["obligations"]} for role, j in judged.items()}
    contested = [name for name in decided
                 if any(name in seen for seen in by_name.values())
                 and len({seen.get(name) for seen in by_name.values()}) > 1]
    if not contested:
        return result
    new_votes, new_lists, notes = {}, {}, []
    for role, j in judged.items():
        own = [o["status"] for o in j["obligations"]]
        if derived_vote(own) != j["verdict"]:
            new_votes[role], new_lists[role] = j["verdict"], lists[role]
            notes.append(f"{role} voted {j['verdict']}, which does not follow from its own obligation statuses "
                         f"({derived_vote(own)}); its vote is kept")
            continue
        updated = []
        for o in j["obligations"]:
            name = o["name"].strip()
            if name in decided and o["status"] != decided[name]:
                notes.append(f"{role}: {name} {o['status']} -> {decided[name]} (check)")
                o = dict(o, status=decided[name], argument=o["argument"] + " [status set by the executed check]")
            updated.append(dict(o))
        new_lists[role] = updated
        new_votes[role] = derived_vote([o["status"] for o in updated])
        if new_votes[role] != j["verdict"]:
            notes.append(f"{role}: vote {j['verdict']} -> {new_votes[role]}")
    return {"applied": True, "contested": contested, "votes": new_votes, "obligations": new_lists, "notes": notes}


# ----- the stage ---------------------------------------------------------------------------------


async def run(ctx: StageContext, card_id: str) -> str | None:
    card = ctx.card(card_id)
    open_issues = known_open_issues(ctx, card_id)
    formalizer_user = card_for_prompt(card) + f"\n\n{oc.OPTIONAL_NOTE}" + (f"\n\n{open_issues}" if open_issues else "")
    form = await ctx.ask("formalizer", "formalizer", formalizer_user, FORMALIZER)
    obligations = form.parsed["obligations"]
    calls = [form.call_id]
    ctx.put_json_artifact("obligations", ctx.artifact_name(card_id), form.parsed, card_id)

    user = "CARD:\n" + card_for_prompt(card) + "\n\nPROOF OBLIGATIONS:\n" + json.dumps(obligations, indent=2)
    if open_issues:
        user += f"\n\n{open_issues}"
    dis = await ctx.ask("disprover", "disprover", user, load_schema("disprover"))
    calls.append(dis.call_id)
    attack_results, job_ids = await run_attack_checks(ctx, card_id, dis.parsed["attacks"])
    calls += job_ids
    ctx.put_json_artifact("disproof", ctx.artifact_name(card_id), {"attacks": attack_results}, card_id)
    confirmed = [a for a in attack_results if a["confirmed"]]

    checks: list[dict] = []
    if ctx.cfg["s3"]["symbolic_checks"]:
        checks, check_ids = await oc.run_checks(ctx, card_id, card, obligations)
        calls += check_ids
    violations = oc.stated_violations(checks)
    check_lines = oc.summary_lines(checks)
    all_discharged = oc.all_checkable_discharged(checks)

    def decide(verdict: str, rule: str, reasons: list[str], round_no: int, gate: dict | None = None) -> None:
        ctx.put_json_artifact("s3_gate", ctx.artifact_name(card_id), {
            "rule": rule, "verdict": verdict, "round": round_no, "symbolic_checks": ctx.cfg["s3"]["symbolic_checks"],
            "all_checkable_discharged": all_discharged, "checks": check_lines, **(gate or {})}, card_id)
        ctx.verdict(card_id, "S3", verdict, [f"gate rule: {rule}"] + reasons + check_lines, calls)

    rounds = 1 + ctx.cfg["s3"]["extra_rounds"]
    votes: list[str] = []
    rule = "judges"
    for round_no in range(1, rounds + 1):
        prompt = judge_prompt(card, obligations, attack_results, round_no, checks)
        results = await asyncio.gather(*(ctx.ask(role, role, prompt, load_schema("feasibility_judge"))
                                         for role in JUDGES))
        calls += [r.call_id for r in results]
        judged = {role: r.parsed for role, r in zip(JUDGES, results)}
        settled = settle_split(judged, checks)
        gate = {"votes": {role: j["verdict"] for role, j in judged.items()}, "effective_votes": settled["votes"],
                "check_settles_split": settled["applied"], "contested": settled["contested"],
                "notes": settled["notes"]}
        ctx.put_json_artifact("judges", ctx.artifact_name(card_id),
                              {"round": round_no, "gate": gate,
                               "judges": {role: dict(j, call_id=r.call_id, model=r.model)
                                          for (role, j), r in zip(judged.items(), results)}},
                              card_id)
        votes = [settled["votes"][role] for role in JUDGES]
        rule = "check_settles_split" if settled["applied"] else "judges"
        summary = [f"round {round_no} {role} ({r.model}): {j['verdict']} (confidence {j['confidence']}); "
                   f"fatal: {'; '.join(j['fatal_issues'])[:400] or 'none'}"
                   for (role, j), r in zip(judged.items(), results)]
        if settled["applied"]:
            summary.append(f"check settles split on {', '.join(settled['contested'])}: effective votes {votes}; "
                           + "; ".join(settled["notes"])[:600])
        if confirmed:
            # The confirmed attack is the deciding explanation of the death, so its target and counterexample are kept
            # whole (FIX-12, F2: outcome records rebuilt their causes from reasons clipped here).
            reasons = [disproof_reason(a) for a in confirmed] + summary
            decide("infeasible_proved", "disproof", reasons, round_no, gate)
            return None
        if violations:
            reasons = [f"executed check violated stated obligation '{r['obligation']}' (replayed): "
                       f"{r['evidence'][:300]}; counterexample {json.dumps(r['counterexample'], default=str)[:400]}"
                       for r in violations] + summary
            decide("infeasible_proved", "check_violation", reasons, round_no, gate)
            return None
        if votes == ["feasible", "feasible"]:
            open_count = sum(1 for lst in settled["obligations"].values() for o in lst if o["status"] != "discharged")
            decide("pass", rule, summary + [f"both judges feasible; {len(obligations)} obligations, "
                                            f"{open_count} judge entries not marked discharged"], round_no, gate)
            return "S4"
        if votes == ["infeasible", "infeasible"]:
            succeeded = [a for a in attack_results if a["status"] == "succeeds" and not a["check_refutes_attack"]]
            refuted = [o["name"] for lst in settled["obligations"].values() for o in lst if o["status"] == "refuted"]
            if succeeded or refuted:
                reasons = summary + [f"surviving attack on '{a['target_claim']}': {a['counterexample']}"
                                     for a in succeeded] + [f"obligation refuted: {name}" for name in refuted]
                decide("infeasible_proved", rule, reasons, round_no, gate)
            else:
                decide("fail", rule, summary + ["both judges infeasible without a refuted "
                                                "obligation or surviving attack"], round_no, gate)
            return None
    ending = (f"both judges undetermined after {rounds} round(s)" if votes == ["undetermined", "undetermined"]
              else f"judges did not agree after {rounds} round(s): last votes {votes}")
    decide("undetermined", rule, [ending], rounds, gate)
    return None
