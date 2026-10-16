"""Scoring of a planted-card test against its sealed expectations (DISCOVERY-PIPELINE-PLAN.md section 13.2).

Input: for each run of the whole set, one result per card (hybrid/research/planted.py ``card_result``): the stop point,
the kill codes, whether the card ended, its cost, and for T5 cards the verdicts of M and C under the override. The
expected stop and kill codes per type come from ``RULES`` (plan section 13.1); the sealed file holds only each
card's type.

Stop points. The plan names stages by letter: G (generation admission, recorded as S0), R (the entry check S1 and the
dialogue R), P (prior art, S2), M, C, and T (test design: the card passed C and the target gate, so the driver stops
it before S4). A card whose line did not end (a stage error, the run budget) has stop ``open``; a card that never got
admitted has ``none``.

Section 13.1 predates generation stage G (Phase 6), which applies part of the P checks at admission: the family
exclusion, a known family without a real difference, and the mechanism-signature duplicate. For the types the plan
expects to stop at P (T1, T2, T5, T6) a stop at G counts as the expected stop, with the same kill codes, plus
``not_radical:R1`` (a closed archive cell, which is a cell of an out-of-scope family) for T1 and T5, ``no_difference``
for T1 (G's code for "the classifier places the core in a known family and finds no real difference", which is the
reduction to a known method that T1 tests) and ``duplicate:<card>`` for T6 (the signature match the plan names).

T3 (a valid proof resting on an assumption X that does all the work) is expected at C with ``magic_x`` or
``unproven``. Section 4.3 puts the open-premise gate in M: when the statement fixer finds that nothing (no code with a
measurement plan, no citation) can settle X, M already ends the card with ``unproven:<premise>``. That stop is right
for T3 only with an ``unproven`` code; a T3 card killed at M for any other reason has the wrong stop.

Metrics per run, with thresholds as rates; a count threshold is ``floor(rate * n)`` over the cards in the run, which
gives the plan's numbers on the 49 cards (9 of 10, 5 of 6, 7 of 8, 8 of 9, 5 of 6, at most 1 of 10 false kills, 3 of
4) and the plan's rounded-down numbers on the 21 held-back cards (2 of 3 per caught type, 0 of 6 false kills, 2 of 3
T7 survivors):

- catch rate per type T1, T2, T3, T4, T6: share killed (any stop, any code);
- stop-point accuracy: share of killed cards of T1 to T6 that stopped at an expected stop, at least 0.85;
- reason accuracy: share of killed cards of T1 to T6 with at least one kill code of the expected codes, at least 0.80;
- false kills: T5 and T7 cards killed at M or C (T5: in the main line or under the override), at most 0.1 of them.
  A valid card killed there only by target-gate codes (``TARGET_CODES``) is a target kill, reported apart from false
  kills per type and overall (DECISIONS T7c): K1 itself passes the profile and fails the target, so the gate is not
  evidence that the pipeline kills valid mechanisms. Any other code on the kill makes it a false kill;
- T7 survival: T7 cards that reached T, at least 0.75;
- loop termination: every card ended (dead or reached T; T5 also finished both override stages) with no budget
  overrun (model calls or revision rounds above their per-idea caps);
- cost per card: model calls, tokens and executor seconds, median and maximum; the maximum calls within the cap.

Across runs: repeatability is the share of cards whose verdict (stop, the set of kill-code families, and for T5 the
override verdicts) is the same in every run, at least 0.90 over at least two runs. Check integrity: zero differences in
the check-file hash audit. Acceptance needs every threshold in every run, the agreement, and the audit.
"""
from __future__ import annotations

import fnmatch
import json
import math
import statistics
from dataclasses import dataclass

from hybrid.research.outcomes import kill_family
from hybrid.research.planted_seal import TYPES

STAGE_STOP = {"S0": "G", "S1": "R", "R": "R", "S2": "P", "M": "M", "C": "C"}
KILL_STOPS = ("G", "R", "P", "M", "C")
TARGET_CODES = ("obligation_failed:target", "obligation_failed:target_coupled")


@dataclass(frozen=True)
class Rule:
    stops: tuple[str, ...]
    codes: tuple[str, ...]      # fnmatch patterns; empty for a card that must not die
    valid: bool = False         # counts for the false-kill rate
    override: bool = False      # M and C also run under the operator override and must both pass
    # Further stops that are right only with a code matching the pattern: (pattern, stops).
    code_stops: tuple[tuple[str, tuple[str, ...]], ...] = ()


RULES = {
    "T1": Rule(("G", "P"), ("reduces_to_known:*", "excluded_family:*", "not_radical:R1", "no_difference")),
    "T2": Rule(("G", "P"), ("no_difference", "not_radical:R3")),
    "T3": Rule(("C",), ("magic_x:*", "magic_x_missing:*", "unproven:*"), code_stops=(("unproven:*", ("M",)),)),
    "T4": Rule(("M", "C"), ("obligation_failed:*", "conservation:*")),
    "T5": Rule(("G", "P"), ("reduces_to_known:*", "excluded_family:*", "not_radical:R1", "no_difference"),
               valid=True, override=True),
    "T6": Rule(("G", "P"), ("no_difference", "not_radical:R3", "duplicate:*")),
    "T7": Rule(("T",), (), valid=True),
}
CATCH_TYPES = ("T1", "T2", "T3", "T4", "T6")
SCORED_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6")
THRESHOLDS = {"catch": {"T1": 9 / 10, "T2": 5 / 6, "T3": 7 / 8, "T4": 8 / 9, "T6": 5 / 6},
              "stop_accuracy": 0.85, "reason_accuracy": 0.80, "false_kill_rate": 1 / 10, "t7_survival": 3 / 4,
              "agreement": 0.90}


def floor_count(rate: float, n: int) -> int:
    """The count a rate asks for over n cards, rounded down (plan section 13.2: the same rates, rounded down)."""
    return math.floor(rate * n + 1e-9)


def code_matches(code: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(code, pattern) for pattern in patterns)


def target_only(codes) -> bool:
    """True for a kill whose every code is a target-gate code (a kill without codes is not)."""
    return bool(codes) and all(code in TARGET_CODES for code in codes)


def override_passed(result: dict) -> bool:
    """M and C passed under the override. The override records one list of kill codes for M and C together, and a
    failed stage always carries a code, so a list of target codes only means every failure was the target gate: those
    verdicts count as passed (DECISIONS T7c)."""
    override = result.get("override") or {}
    excused = target_only(override.get("kill_codes"))

    def passed(verdict) -> bool:
        return verdict == "pass" or (excused and verdict is not None)

    return passed(override.get("M")) and passed(override.get("C")) and not override.get("stopped")


def late_kills(result: dict, rule: Rule) -> list[list[str]]:
    """Kill codes of each kill of a valid card at M or C: the main line's, and the override's for T5."""
    kills = []
    if result["outcome"] == "dead" and result["stop"] in ("M", "C"):
        kills.append(list(result["kill_codes"]))
    override = result.get("override")
    if rule.override and override is not None and override.get("killed_at") in ("M", "C"):
        kills.append(list(override.get("kill_codes") or []))
    return kills


def judge(result: dict, kind: str) -> dict:
    """One card in one run against its type's rule."""
    rule = RULES[kind]
    killed = result["outcome"] == "dead"
    stop_ok = result["stop"] in rule.stops or any(
        result["stop"] in stops and code_matches(code, (pattern,))
        for pattern, stops in rule.code_stops for code in result["kill_codes"])
    reason_ok = (any(code_matches(code, rule.codes) for code in result["kill_codes"]) if rule.codes
                 else not result["kill_codes"])
    kills = late_kills(result, rule) if rule.valid else []
    false_kill = any(not target_only(codes) for codes in kills)
    target_kill = bool(kills) and not false_kill
    ended = result["outcome"] in ("dead", "reached") and (
        not rule.override or bool(result.get("override")) and not result["override"].get("stopped"))
    correct = stop_ok and reason_ok and not false_kill and (not rule.override or override_passed(result))
    return {"killed": killed, "stop_ok": stop_ok, "reason_ok": reason_ok, "false_kill": false_kill,
            "target_kill": target_kill, "target_codes": sorted({c for codes in kills for c in codes}) if target_kill
            else [], "survived": result["stop"] == "T",
            "terminated": ended and not result["over_budget"], "correct": correct}


def verdict_key(result: dict) -> list:
    override = result.get("override") or {}
    return [result["stop"], sorted({kill_family(code) for code in result["kill_codes"]}),
            override.get("M"), override.get("C")]


def _stats(values: list[float]) -> dict:
    return {"median": statistics.median(values) if values else None, "max": max(values) if values else None}


def score_run(results: dict[str, dict], types: dict[str, str], *, idea_calls_cap: int) -> dict:
    """Metrics of one run: per-type table, every section 13.2 metric of one run, and per-card judgments."""
    judged = {key: judge(result, types[key]) for key, result in results.items()}
    per_type = {}
    for kind in TYPES:
        keys = [key for key in results if types[key] == kind]
        if not keys:
            continue
        rows = [judged[k] for k in keys]
        per_type[kind] = {"n": len(keys), "killed": sum(r["killed"] for r in rows),
                          "stop_ok": sum(r["killed"] and r["stop_ok"] for r in rows),
                          "reason_ok": sum(r["killed"] and r["reason_ok"] for r in rows),
                          "false_kills": sum(r["false_kill"] for r in rows),
                          "target_kills": sum(r["target_kill"] for r in rows),
                          "reached_t": sum(r["survived"] for r in rows),
                          "terminated": sum(r["terminated"] for r in rows),
                          "correct": sum(r["correct"] for r in rows)}
    checks = []

    def check(name: str, value, needed, ok: bool, detail: str) -> None:
        checks.append({"metric": name, "value": value, "needed": needed, "ok": ok, "detail": detail})

    for kind in CATCH_TYPES:
        if kind in per_type:
            row = per_type[kind]
            need = floor_count(THRESHOLDS["catch"][kind], row["n"])
            check(f"catch {kind}", row["killed"], need, row["killed"] >= need, f"{row['killed']} of {row['n']} killed")
    caught = [k for k in results if types[k] in SCORED_TYPES and judged[k]["killed"]]
    for name, field in (("stop_accuracy", "stop_ok"), ("reason_accuracy", "reason_ok")):
        good = sum(judged[k][field] for k in caught)
        rate = good / len(caught) if caught else None
        check(name, rate, THRESHOLDS[name], rate is not None and rate >= THRESHOLDS[name] - 1e-9,
              f"{good} of {len(caught)} killed cards of T1 to T6")
    valid = [k for k in results if RULES[types[k]].valid]
    false = sum(judged[k]["false_kill"] for k in valid)
    target = sum(judged[k]["target_kill"] for k in valid)
    allowed = floor_count(THRESHOLDS["false_kill_rate"], len(valid))
    check("false_kills", false, allowed, false <= allowed,
          f"{false} of {len(valid)} valid cards killed at M or C; {target} more killed only by the target gate "
          "(reported apart, not counted)")
    t7 = [k for k in results if types[k] == "T7"]
    if t7:
        survived = sum(judged[k]["survived"] for k in t7)
        need = floor_count(THRESHOLDS["t7_survival"], len(t7))
        check("t7_survival", survived, need, survived >= need, f"{survived} of {len(t7)} reached T")
    ended = sum(judged[k]["terminated"] for k in results)
    overruns = sorted(k for k in results if results[k]["over_budget"])
    check("termination", ended, len(results), ended == len(results),
          f"{ended} of {len(results)} ended within budget; overruns: {', '.join(overruns) or 'none'}")
    costs = {name: _stats([float(r[name]) for r in results.values()]) for name in ("model_calls", "tokens",
                                                                                   "executor_s")}
    max_calls = costs["model_calls"]["max"]
    check("cost_cap", max_calls, idea_calls_cap, max_calls is None or max_calls <= idea_calls_cap,
          f"maximum model calls per card {max_calls} against the per-idea cap {idea_calls_cap}")
    target_kills = {"total": sum(row["target_kills"] for row in per_type.values()),
                    "per_type": {kind: row["target_kills"] for kind, row in per_type.items() if row["target_kills"]}}
    return {"per_type": per_type, "checks": checks, "costs": costs, "judged": judged, "target_kills": target_kills,
            "passed": all(c["ok"] for c in checks)}


def agreement(runs: list[dict[str, dict]]) -> dict:
    """Share of cards whose verdict is the same in every run (None with fewer than two runs)."""
    if len(runs) < 2:
        return {"rate": None, "split": [], "runs": len(runs)}
    keys = sorted(set.intersection(*(set(run) for run in runs)))
    split = [key for key in keys if any(verdict_key(run[key]) != verdict_key(runs[0][key]) for run in runs[1:])]
    return {"rate": (len(keys) - len(split)) / len(keys) if keys else None, "split": split, "runs": len(runs)}


def score(runs: list[dict[str, dict]], types: dict[str, str], *, idea_calls_cap: int, audit: dict,
          problems: list[str] = ()) -> dict:
    """The whole test: every run scored, agreement across runs, the audit, and acceptance."""
    scored = [score_run(run, types, idea_calls_cap=idea_calls_cap) for run in runs]
    agree = agreement(runs)
    agree_ok = agree["rate"] is not None and agree["rate"] >= THRESHOLDS["agreement"] - 1e-9
    audit_ok = not audit.get("differences")
    accepted = bool(runs) and all(s["passed"] for s in scored) and agree_ok and audit_ok and not problems
    return {"runs": scored, "agreement": dict(agree, needed=THRESHOLDS["agreement"], ok=agree_ok),
            "audit": dict(audit, ok=audit_ok), "accepted": accepted, "problems": list(problems),
            "thresholds": THRESHOLDS}


# ----- report -------------------------------------------------------------------------------------------------------

RATE_METRICS = ("stop_accuracy", "reason_accuracy")


def _value(value) -> str:
    """A count or a measured number (calls, tokens, seconds)."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.2f}"
    return str(value)


def _rate(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_markdown(report: dict, results: list[dict[str, dict]], sealed_entries: dict[str, dict],
                    meta: dict) -> str:
    types = {key: entry["type"] for key, entry in sealed_entries.items()}
    lines = ["# Planted-card test", "",
             f"Set {meta['set']}, {meta['cards']} cards, {meta['runs']} run(s). Started {meta['started']}, finished "
             f"{meta['finished']}. Accepted (plan section 13.2): {'yes' if report['accepted'] else 'no'}.", ""]
    for index, (run, scored) in enumerate(zip(results, report["runs"]), 1):
        lines += [f"## Run {index}", "",
                  "| Type | Cards | Killed | Right stop | Right reason | False kills | Target kills | Reached T | Ended "
                  "| Correct |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for kind, row in scored["per_type"].items():
            lines.append(f"| {kind} | {row['n']} | {row['killed']} | {row['stop_ok']} | {row['reason_ok']} | "
                         f"{row['false_kills']} | {row['target_kills']} | {row['reached_t']} | {row['terminated']} | {row['correct']} |")
        lines += ["", "| Metric | Value | Needed | Met | Detail |", "|---|---|---|---|---|"]
        for c in scored["checks"]:
            show = _rate if c["metric"] in RATE_METRICS else _value
            lines.append(f"| {c['metric']} | {show(c['value'])} | {show(c['needed'])} | "
                         f"{'yes' if c['ok'] else 'no'} | {c['detail']} |")
        costs = scored["costs"]
        lines += ["", "Cost per card (median / max): " + "; ".join(
            f"{name} {_value(v['median'])} / {_value(v['max'])}" for name, v in costs.items()), "",
            "Per card:", ""]
        for key in sorted(run, key=lambda k: (types[k], k)):
            r, j = run[key], scored["judged"][key]
            rule = RULES[types[key]]
            override = ""
            if r.get("override") is not None:
                o = r["override"]
                override = f"; override M {o.get('M')}, C {o.get('C')}" + (f" ({', '.join(o['kill_codes'])})"
                                                                         if o.get("kill_codes") else "")
            lines.append(f"- {key} {types[key]} ({r['card_id'] or '-'}): stop {r['stop']} (expected "
                         f"{'/'.join(rule.stops)}), codes {', '.join(r['kill_codes']) or 'none'}{override}; calls "
                         f"{r['model_calls']}; {'correct' if j['correct'] else 'WRONG'}"
                         + ("; target kill (" + ", ".join(j["target_codes"]) + ")" if j["target_kill"] else "")
                         + (f"; {'; '.join(r['notes'])}" if r.get("notes") else ""))
        lines.append("")
    agree = report["agreement"]
    lines += ["## Across runs", "",
              f"Repeatability: {_rate(agree['rate'])} of cards agree over {agree['runs']} run(s) (needed "
              f"{_rate(agree['needed'])}): {'met' if agree['ok'] else 'not met'}."
              + (f" Split: {', '.join(agree['split'])}." if agree["split"] else ""),
              f"Check integrity: {len(report['audit'].get('differences', []))} difference(s) in "
              f"{report['audit'].get('files', 0)} audited files and {report['audit'].get('math_checks', 0)} math "
              f"checks: {'met' if report['audit']['ok'] else 'not met'}."]
    for item in report["audit"].get("differences", []):
        lines.append(f"- {item}")
    if report["problems"]:
        lines += ["", "Problems:"] + [f"- {p}" for p in report["problems"]]
    return "\n".join(lines) + "\n"


def dumps(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
