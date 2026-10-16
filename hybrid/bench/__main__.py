"""CLI.

    python -m hybrid.bench eval --candidate <dir> --config <toml> --run-dir <dir> [--local]
                                [--claimed-exponent X] [--baselines a,b] [--cache-dir <dir>]
    python -m hybrid.bench baselines --config <toml> --run-dir <dir> [--local] [--markdown <path>]

Without --local, jobs run in the Podman executor (the VM must be running) and the result is
trusted. --local runs jobs as host subprocesses for development and is labeled trusted: false.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .evaluate import evaluate_candidate
from .local import LocalExecutor
from .report import render

RUN_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,40}")


def _executor(run_dir, local):
    run_id = Path(run_dir).resolve().name.lower()
    if not RUN_ID.fullmatch(run_id):
        raise SystemExit("run directory name %r is not a valid run id ([a-z0-9][a-z0-9_.-]*)" % run_id)
    receipts = Path(run_dir).resolve() / "receipts"
    if local:
        return LocalExecutor(run_id, receipts)
    from hybrid.executor.executor import Executor
    return Executor(run_id, receipts)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hybrid.bench")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("eval", "baselines"):
        p = sub.add_parser(name)
        if name == "eval":
            p.add_argument("--candidate", required=True)
            p.add_argument("--claimed-exponent", type=float)
        p.add_argument("--config", required=True)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--local", action="store_true")
        p.add_argument("--baselines", help="comma-separated subset of baselines")
        p.add_argument("--cache-dir")
        p.add_argument("--markdown", help="also write the Markdown tables here")
    args = parser.parse_args(argv)
    if Path(args.run_dir).resolve().joinpath("bench").exists():
        raise SystemExit("%s/bench already exists; use a fresh run directory" % args.run_dir)
    executor = _executor(args.run_dir, args.local)
    baselines = args.baselines.split(",") if args.baselines else None
    result = evaluate_candidate(args.candidate if args.command == "eval" else None, args.config, args.run_dir,
                                executor, claimed_exponent=getattr(args, "claimed_exponent", None),
                                baselines=baselines, cache_dir=args.cache_dir)
    table = render(result)
    if args.markdown:
        Path(args.markdown).write_text(table)
    print(table)
    print(json.dumps({"trusted": result["trusted"], "bench_result": str(Path(args.run_dir).resolve() / "bench" /
                                                                       "bench_result.json"),
                      "timings_s": result["host"]["timings_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
