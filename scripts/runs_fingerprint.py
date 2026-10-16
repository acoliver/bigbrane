"""Fingerprint the files the outcome backfill reads, to show it wrote nothing inside runs/.

    .venv-hybrid/bin/python scripts/runs_fingerprint.py runs > results/<dir>/runs-before.json
    ... backfill ...
    .venv-hybrid/bin/python scripts/runs_fingerprint.py runs > results/<dir>/runs-after.json
    .venv-hybrid/bin/python scripts/runs_fingerprint.py --compare results/<dir>/runs-before.json results/<dir>/runs-after.json

For every run directory with a ledger and a cards/ directory: sha256, size and mtime of ledger.jsonl and of every
file under cards/ and artifacts/, plus the names and mtimes of the run directory's top-level entries (a new file such
as a lock file would show there). Read-only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def fingerprint(runs_root: Path) -> dict:
    out = {}
    for run in sorted(p for p in runs_root.iterdir() if (p / "ledger.jsonl").is_file() and (p / "cards").is_dir()):
        files = [run / "ledger.jsonl"]
        for sub in ("cards", "artifacts"):
            if (run / sub).is_dir():
                files += sorted(p for p in (run / sub).rglob("*") if p.is_file())
        entry = {"top": {p.name: p.stat().st_mtime_ns for p in sorted(run.iterdir())}, "files": {}}
        for path in files:
            stat = path.stat()
            entry["files"][path.relative_to(run).as_posix()] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
        out[run.name] = entry
    return out


def compare(before: dict, after: dict) -> list[str]:
    problems = []
    for run in sorted(set(before) | set(after)):
        if run not in before or run not in after:
            problems.append(f"{run}: present only {'after' if run in after else 'before'}")
            continue
        if before[run]["top"] != after[run]["top"]:
            problems.append(f"{run}: top-level entries or their mtimes changed")
        if before[run]["files"] != after[run]["files"]:
            changed = [name for name in set(before[run]["files"]) | set(after[run]["files"])
                       if before[run]["files"].get(name) != after[run]["files"].get(name)]
            problems.append(f"{run}: {len(changed)} file(s) changed, e.g. {sorted(changed)[:3]}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args(argv)
    if args.compare:
        before, after = (json.loads(p.read_text()) for p in args.paths)
        problems = compare(before, after)
        files = sum(len(v["files"]) for v in before.values())
        print(f"runs: {len(before)}, files fingerprinted: {files}, differences: {len(problems)}")
        for line in problems:
            print("  " + line)
        return 1 if problems else 0
    json.dump(fingerprint(args.paths[0]), sys.stdout, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
