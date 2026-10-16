"""Phase 2 live run: the math library in the tools image through the executor.

    python -m hybrid.math.library_smoke --out results/phase2-<stamp>/live [--quick]

Runs the selfcheck battery (the plan's full grid unless --quick) and one request of every kind, each through
``MathRunner.check_library``: manifest verified on the host, files staged, re-hashed in the container, run with
``--network=none``, a deciding result replayed. Writes cases.json and prints one line per case. Exit 0 only when
every case gives its expected, replayed result and every job ran without network.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from hybrid.math import references as ref
from hybrid.math.smoke import network_modes


def library_cases(quick: bool) -> list[tuple[str, dict, str]]:
    return [
        ("selfcheck_battery", {"kind": "selfcheck", "quick": quick}, "discharged"),
        ("conservation_planted_capacity_overrun",
         {"kind": "conservation", "claims": {"state_bits": 256, "recall": [{"K": 64, "V": 256, "accuracy": 0.99}]}},
         "violated"),
        ("cost_planted_exponent_error",
         {"kind": "cost", "source": ref.PLANTED_QUADRATIC, "claim": ref.REFERENCES["planted_quadratic"][1],
          "ns": [64, 128, 256, 512]}, "violated"),
        ("cost_elman_rnn_plan_grid",
         {"kind": "cost", "source": ref.ELMAN_RNN, "claim": ref.REFERENCES["elman_rnn"][1]}, "discharged"),
        ("construction_recall_capacity_8",
         {"kind": "construction", "source": ref.CONSTRUCTIONS["recall_table_cap8"][0], "task": "recall",
          "sizes": [4, 8, 16], "params": {"V": 16}}, "violated"),
        ("magic_x_retrieval_oracle",
         {"kind": "magic_x", "claim": ref.MAGIC_X_CLAIM,
          "assumption": {"id": "retrieval_oracle", "kind": "store", "source": ref.RETRIEVAL_ORACLE}}, "violated"),
    ]


def run_cases(runner, receipts_dir: Path, quick: bool):
    """Yield one record per case as it finishes."""
    for name, request, expected in library_cases(quick):
        started = time.monotonic()
        record = runner.check_library(request)
        job_ids = [run["job_id"] for run in record["runs"]]
        network = network_modes(receipts_dir, job_ids)
        ok = record["status"] == expected and record["decided"] and set(network) == {"none"}
        yield {"case": name, "expected": expected, "status": record["status"], "decided": record["decided"],
               "ok": ok, "problems": record["problems"], "evidence": record["evidence"],
               "counterexample": record["counterexample"], "wall_s": round(time.monotonic() - started, 2),
               "job_ids": job_ids, "network": network, "check_hashes": record["check_hashes"]}


def main(argv: list[str] | None = None) -> int:
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.math.runner import MathRunner

    parser = argparse.ArgumentParser(prog="python -m hybrid.math.library_smoke")
    parser.add_argument("--out", type=Path, required=True, help="new directory for jobs, receipts and cases.json")
    parser.add_argument("--quick", action="store_true", help="selfcheck on the small grid")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipts = out / "receipts"
    runner = MathRunner(Executor(out.parent.name, receipts), read_lock("tools")["reference"], out)
    cases = []
    for case in run_cases(runner, receipts, args.quick):
        cases.append(case)
        print(f"{'OK  ' if case['ok'] else 'FAIL'} {case['case']}: {case['status']} (expected {case['expected']}) "
              f"decided={case['decided']} wall_s={case['wall_s']} jobs={len(case['job_ids'])} "
              f"{case['evidence'][:200]} {'; '.join(case['problems'])[:300]}", flush=True)
    (out / "cases.json").write_text(json.dumps(cases, indent=2, default=str) + "\n")
    return 0 if all(case["ok"] for case in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
