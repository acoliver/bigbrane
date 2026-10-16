"""Executed checks of the S3 proof obligations (s3.symbolic_checks).

After the disprover, the `obligation_checker` role sorts every formalizer obligation into a kind and writes a
sympy or numpy program for each obligation of a checkable kind (complexity, capacity, algebraic,
numerical_stability). Each program runs in the executor; a program whose first run decides the obligation
(discharged or violated) runs a second time, and the obligation counts as decided only when both runs print
the same status and counterexample. One `obligation_check` artifact is recorded per obligation.

Program contract: the last stdout line is one JSON object
{"status": "discharged" | "violated" | "unknown", "evidence": str, "counterexample": object | null},
with a non-null counterexample when the status is violated. A crash, a timeout, a contract breach or a replay
that disagrees leaves the obligation `unknown`.

The gate rules that use these results are in `s3_feasibility.py`.
"""
from __future__ import annotations

import json

from hybrid.research.stages.common import ExecResult, StageContext, card_for_prompt

CHECKABLE_KINDS = ("complexity", "capacity", "algebraic", "numerical_stability")
KINDS = CHECKABLE_KINDS + ("learnability", "other")
REPORT_STATUSES = ("discharged", "violated", "unknown")
DECIDING = ("discharged", "violated")
# What a decided check means for an obligation in the judges' vocabulary.
JUDGE_STATUS = {"discharged": "discharged", "violated": "refuted"}
PROGRAM_SHOWN = 4000

# Appended to the formalizer user message (the prompt file stays as it is): the schema asks for `optional`.
OPTIONAL_NOTE = (
    "For each obligation also give optional (true or false). optional is true only for an obligation that no "
    "card claim rests on: a side condition, a robustness remark, or a property of an extension. Every "
    "obligation a claim needs, including the complexity, capacity and learnability obligations, has optional "
    "false. An executed check that violates an obligation with optional false makes the card infeasible.")


def checker_schema(names: list[str]) -> dict:
    """Output of the obligation_checker role: one entry per obligation, named by the formalizer's names."""
    entry = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "kind", "program", "why"],
        "properties": {
            "name": {"enum": list(names)},
            "kind": {"enum": list(KINDS)},
            "program": {"type": ["string", "null"]},
            "why": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["checks"],
        "properties": {"checks": {"type": "array", "minItems": len(names), "maxItems": len(names), "items": entry}},
    }


def checker_user(card: dict, obligations: list[dict]) -> str:
    return ("CARD:\n" + card_for_prompt(card) + "\n\nPROOF OBLIGATIONS (from the formalizer):\n"
            + json.dumps(obligations, indent=2, ensure_ascii=False))


def parse_report(printed: dict | None) -> tuple[dict | None, list[str]]:
    """The program's report if it follows the contract, else None and the breaches."""
    if printed is None:
        return None, ["no JSON object on stdout"]
    problems = []
    status = printed.get("status")
    if status not in REPORT_STATUSES:
        problems.append(f"status {status!r} is not one of {', '.join(REPORT_STATUSES)}")
    evidence = printed.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        problems.append("evidence is missing or empty")
    counterexample = printed.get("counterexample")
    if status == "violated" and counterexample in (None, "", {}, []):
        problems.append("violated without a counterexample")
    if problems:
        return None, problems
    return {"status": status, "evidence": evidence, "counterexample": counterexample}, []


def run_entry(result: ExecResult) -> dict:
    """One run as recorded in a check: the receipt status and detail, its wall time, and what it printed."""
    return {"job_id": result.job_id, "status": result.status, "detail": result.detail,
            "run_s": round(result.run_s, 3) if result.run_s is not None else None,
            "printed": result.printed_json() if result.ok else None,
            "stdout_tail": result.stdout[-1500:], "stderr_tail": result.stderr[-800:]}


def _fingerprint(report: dict) -> str:
    return json.dumps([report["status"], report["counterexample"]], sort_keys=True, default=str)


def outcome(first: ExecResult, second: ExecResult | None) -> dict:
    """Status of one obligation from its first run and, when the first run decided it, its replay."""
    runs = [run_entry(first)] + ([run_entry(second)] if second is not None else [])
    base = {"runs": runs, "replayed": False, "decided": False, "evidence": "", "counterexample": None}
    if not first.ok:
        return dict(base, status="unknown", problems=[f"first run {first.status}: {first.detail}"[:400]])
    report, problems = parse_report(runs[0]["printed"])
    if report is None:
        return dict(base, status="unknown", problems=["first run breaks the contract: " + "; ".join(problems)])
    base.update(evidence=report["evidence"], counterexample=report["counterexample"])
    if report["status"] not in DECIDING:
        return dict(base, status="unknown", problems=[])
    if second is None:
        raise AssertionError("a deciding first run needs its replay")
    if not second.ok:
        return dict(base, status="unknown", problems=[f"replay {second.status}: {second.detail}"[:400]])
    replay, problems = parse_report(runs[1]["printed"])
    if replay is None:
        return dict(base, status="unknown", problems=["replay breaks the contract: " + "; ".join(problems)])
    if _fingerprint(replay) != _fingerprint(report):
        return dict(base, status="unknown",
                    problems=[f"replay disagreed: first {_fingerprint(report)[:300]}, replay {_fingerprint(replay)[:300]}"])
    return dict(base, status=report["status"], replayed=True, decided=True, problems=[])


async def run_checks(ctx: StageContext, card_id: str, card: dict, obligations: list[dict]) -> tuple[list[dict], list[str]]:
    """Classify, write and run the obligation checks. Returns one record per obligation (formalizer order) and
    the call and job ids behind them."""
    names = [o["name"] for o in obligations]
    reply = await ctx.ask("obligation_checker", "obligation_checker", checker_user(card, obligations),
                          checker_schema(names))
    ids = [reply.call_id]
    entries: dict[str, dict] = {}
    duplicates = []
    for entry in reply.parsed["checks"]:
        if entry["name"] in entries:
            duplicates.append(entry["name"])
        else:
            entries[entry["name"]] = entry
    budget = ctx.cfg["s3"]["max_obligation_checks"]
    records = []
    for obligation in obligations:
        name = obligation["name"]
        entry = entries.get(name)
        record = {"obligation": name, "optional": obligation["optional"], "formalizer_kind": obligation["kind"],
                  "checker_call": reply.call_id, "kind": entry["kind"] if entry else None,
                  "why": entry["why"] if entry else "", "program": entry["program"] if entry else None,
                  "status": "not_checked", "decided": False, "replayed": False, "evidence": "",
                  "counterexample": None, "runs": [], "problems": []}
        if entry is None:
            record["problems"] = ["the checker gave no entry for this obligation"
                                  + (f" (duplicate entries for {sorted(set(duplicates))})" if duplicates else "")]
        elif entry["kind"] not in CHECKABLE_KINDS:
            if entry["program"]:
                record["problems"] = [f"program ignored: kind {entry['kind']} is left to the judges"]
        elif not entry["program"]:
            record["problems"] = ["checkable kind, but the checker wrote no program"]
        elif budget <= 0:
            record["status"] = "skipped"
            record["problems"] = ["s3.max_obligation_checks reached"]
        else:
            budget -= 1
            files = {"main.py": entry["program"]}
            first = await ctx.runner.run(card_id, "obligation_check", files)
            ids.append(first.job_id)
            second = None
            report, _ = parse_report(first.printed_json() if first.ok else None)
            if report is not None and report["status"] in DECIDING:
                second = await ctx.runner.run(card_id, "obligation_check_replay", files)
                ids.append(second.job_id)
            record.update(outcome(first, second))
        ctx.put_json_artifact("obligation_check", ctx.artifact_name(card_id), record, card_id)
        records.append(record)
    return records, ids


def decided_statuses(records: list[dict]) -> dict[str, str]:
    """Judge-vocabulary status of every obligation a check decided, except violations of optional obligations
    (an optional obligation does not decide the card either way)."""
    return {r["obligation"]: JUDGE_STATUS[r["status"]] for r in records
            if r["decided"] and not (r["status"] == "violated" and r["optional"])}


def stated_violations(records: list[dict]) -> list[dict]:
    return [r for r in records if r["decided"] and r["status"] == "violated" and not r["optional"]]


def all_checkable_discharged(records: list[dict]) -> bool:
    checkable = [r for r in records if r["kind"] in CHECKABLE_KINDS]
    return bool(checkable) and all(r["decided"] and r["status"] == "discharged" for r in checkable)


def judge_section(records: list[dict]) -> str:
    """The block appended to both judges' user messages."""
    shown = []
    for r in records:
        item = {"obligation": r["obligation"], "optional": r["optional"], "kind": r["kind"], "status": r["status"],
                "reproduced_by_replay": r["replayed"], "evidence": r["evidence"],
                "counterexample": r["counterexample"], "problems": r["problems"]}
        if r["program"] and r["runs"]:
            item["program"] = r["program"][:PROGRAM_SHOWN]
            item["job_ids"] = [run["job_id"] for run in r["runs"]]
        shown.append(item)
    return ("EXECUTED OBLIGATION CHECKS (programs written by a separate checker and run in the sandbox before "
            "judging; a discharged or violated status counts only when a second run reproduced it):\n"
            + json.dumps(shown, indent=2, ensure_ascii=False, default=str)
            + "\nHow the gate uses them: a reproduced violation of an obligation that is not optional makes the "
              "card infeasible whatever the judges say. Where the two judges disagree about an obligation a "
              "check decided, the check's result is used. Treat a check as you treat the disproof checks: it "
              "outranks prose unless you can point to a specific bug in the program (name the line and the "
              "error) or show that it does not implement the card's equations or the obligation as stated.")


def summary_lines(records: list[dict]) -> list[str]:
    lines = []
    for r in records:
        tag = f"obligation check {r['obligation']} ({r['kind'] or 'unclassified'}"
        tag += ", optional)" if r["optional"] else ")"
        text = f"{tag}: {r['status']}"
        if r["decided"]:
            text += " (replayed)"
        if r["evidence"]:
            text += f": {r['evidence'][:300]}"
        if r["status"] == "violated" and r["counterexample"] is not None:
            text += f"; counterexample {json.dumps(r['counterexample'], default=str)[:300]}"
        if r["problems"]:
            text += f" [{'; '.join(r['problems'])[:200]}]"
        lines.append(text)
    return lines
