"""Scores the r2-bar replay against the acceptance rules in STATUS.md. Reads runs/*/ledger.jsonl, writes SCORE.md."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_PUBLISHED = [f"card-1710000{i}" for i in range(1, 9)]
PROBE_CONTROLS = ["card-17100009", "card-1710000a"]
# Recheck cards whose earlier S2 had a host-counted reduction (verified check): RESULT.md and the audit table.
COUNTED_BEFORE = ["card-76ee768c", "card-440f0204", "card-62785c10", "card-83bd879a", "card-97baae43", "card-d6dfeac2"]
BEFORE = {  # earlier binding kill (audit section 1)
    "card-76ee768c": "reduces_to_known", "card-a4f684fb": "no_difference", "card-440f0204": "reduces_to_known",
    "card-7e993aab": "R2", "card-83bd879a": "reduces_to_known", "card-89248ee5": "R2", "card-8ad6004f": "R2",
    "card-d6dfeac2": "reduces_to_known", "card-62785c10": "excluded_family", "card-c5524960": "R2",
    "card-97baae43": "reduces_to_known", "card-0fe3943c": "R2", "card-3485a13f": "R2", "card-a775f78d": "R2",
    "card-ebf0e99d": "R2", "card-f5ee7edb": "R2", "card-8154d3b4": "R2", "card-084a2510": "R2", "card-909ad773": "R2"}


def final_s2(ledger):
    rows = {}
    for line in ledger.read_text().splitlines():
        e = json.loads(line)
        if e["type"] == "verdict" and e["data"].get("stage") == "S2":
            rows[e["data"]["card_id"]] = e["data"]
    return rows


def classify(e):
    if e["verdict"] == "pass":
        return "pass"
    text = " | ".join(e["reasons"])
    for tag, key in (("excluded_family", "gate: excluded_family"), ("reduces_to_known", "gate: update rule reduces"),
                     ("R2", "gate: not radical enough"), ("R3", "gate: not_radical:R3"),
                     ("no_difference", "gate: no_difference"), ("prior_art", "gate: prior_art_tested"),
                     ("known_disproven", "gate: known_disproven")):
        if key in text:
            return tag
    return e["verdict"]


def gate_line(e):
    for r in e["reasons"]:
        if r.startswith("gate:"):
            return r[:360]
    return "; ".join(e["reasons"])[:300]


results = {}
for run in sorted((ROOT / "runs").glob("*")):
    ledger = run / "ledger.jsonl"
    if ledger.exists():
        for cid, e in final_s2(ledger).items():
            results[cid] = (run.name, e)
lines = ["# Replay score", "", "| card | run | S2 verdict | class | earlier kill | gate reason |", "|---|---|---|---|---|---|"]
for cid, (run, e) in sorted(results.items()):
    lines.append(f"| {cid} | {run} | {e['verdict']} | {classify(e)} | {BEFORE.get(cid, 'probe')} | "
                 f"{gate_line(e).replace('|', '/')} |")
done = {c: classify(e) for c, (_, e) in results.items()}
r1 = [(c, done.get(c, "NOT RUN")) for c in PROBE_PUBLISHED + PROBE_CONTROLS]
ok1 = all(d == "reduces_to_known" for c, d in r1 if c in PROBE_PUBLISHED) and \
    all(d == "excluded_family" for c, d in r1 if c in PROBE_CONTROLS)
r2 = [(c, done.get(c, "NOT RUN")) for c in COUNTED_BEFORE]
ok2 = all(d in ("reduces_to_known", "excluded_family") for _, d in r2)
flips = [c for c, d in done.items() if d == "pass"]
lines += ["", f"## Rule 1 (novelty probe guard): {'PASS' if ok1 else 'FAIL or incomplete'}", ""]
lines += [f"- {c}: {d}" for c, d in r1]
lines += ["", f"## Rule 2 (counted reductions stay dead): {'PASS' if ok2 else 'FAIL or incomplete'}", ""]
lines += [f"- {c}: {d}" for c, d in r2]
lines += ["", f"## Rule 3 (flips to pass): {', '.join(flips) or 'none'}", ""]
(ROOT / "SCORE.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
