"""Phase 7 exit criterion, offline: replay the S2 reductions of runs/run-20261010T181300Z through the Phase 7 S2 gate
with the recorded critic answers and check outcomes (no live model call). The source run is read, never written.

    .venv-hybrid/bin/python scripts/phase7_replay.py --out results/phase7-<stamp> [--refresh-fixture]

Writes <out>/replay.json, <out>/replay.md and the replay run under <out>/replay-run/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid.tests.research import phase7_replay  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--refresh-fixture", action="store_true",
                        help="extract the fixture again from runs/run-20261010T181300Z (read-only)")
    args = parser.parse_args()
    if args.refresh_fixture:
        phase7_replay.write_fixture()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    work = out / "replay-run"
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"{work} is not empty")
    work.mkdir(parents=True, exist_ok=True)
    rows = phase7_replay.replay(phase7_replay.load_fixture(), work)
    died = [r for r in rows if r["expected_codes"]]
    matched = [r for r in died if r["replay_codes"] == r["expected_codes"]]
    summary = {"source_run": phase7_replay.SOURCE_RUN, "cards": len(rows), "s2_deaths": len(died),
               "expected_matched": len(matched),
               "original_codes": dict(Counter(c for r in died for c in r["original_codes"])),
               "replay_codes": dict(Counter(c for r in died for c in r["replay_codes"])),
               "passed_s2": [r["card_id"] for r in rows if not r["expected_codes"]], "rows": rows}
    (out / "replay.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# Phase 7 replay of {phase7_replay.SOURCE_RUN} S2 reductions (offline, recorded critic outputs)", "",
             f"{len(matched)} of {len(died)} S2 deaths get the expected Phase 7 kill code.", "",
             "| card | reduces_to | check | LEARNING_RULE: | original code | replay code | expected |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['card_id']} | {r['reduces_to']} | {r['check']} | {'yes' if r['learning_rule_prefix'] else 'no'}"
                     f" | {', '.join(r['original_codes']) or r['original_outcome']} | "
                     f"{', '.join(r['replay_codes']) or ('S2 ' + r['replay_s2'])} | "
                     f"{', '.join(r['expected_codes']) or 'passes S2 reduction gates'} |")
    (out / "replay.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(matched)}/{len(died)} expected; wrote {out / 'replay.json'} and {out / 'replay.md'}")
    return 0 if len(matched) == len(died) else 1


if __name__ == "__main__":
    raise SystemExit(main())
