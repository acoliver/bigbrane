"""Phase 3 live run: checks of frozen statements (spec_checks.py) in the tools image through the executor.

    python -m hybrid.math.spec_smoke --out results/phase3-<stamp>/spec-live<n>

One request of every frozen-statement kind, each through ``MathRunner.check_library`` (manifest verified on the
host, files staged and re-hashed in the container, ``--network=none``, a deciding result replayed). Z3 runs only in
the image, so this is where the z3 and predicate-agreement paths are exercised for real. Writes cases.json and
prints one line per case. Exit 0 only when every case gives its expected, replayed result without network.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from hybrid.math import references as ref
from hybrid.math.smoke import network_modes

BOUNDED = {"variables": {"x": {"sort": "int", "lo": 0, "hi": 1000}}, "hypotheses": [],
           "property": "x * x != 2 * x + 3"}
HOLDS = {"variables": {"a": {"sort": "real", "lo": -100, "hi": 100}, "b": {"sort": "real", "lo": -100, "hi": 100}},
         "hypotheses": [], "property": "2 * a * b <= a * a + b * b"}
# The Navier-Stokes lesson in miniature: the formal statement asks for n**2 where the claim says n.
EXPONENT = {"variables": {"n": {"sort": "int", "lo": 1, "hi": 4096}, "c": {"sort": "int", "lo": 1, "hi": 16}},
            "hypotheses": [], "property": "c * n * n <= 16 * n * n"}
CLAIM_LINEAR = "def claim(n, c):\n    return c * n <= 16\n"
CLAIM_SAME = "def claim(n, c):\n    return c * n * n <= 16 * n * n\n"
SIMULATION = {
    "generator": "def generate(size, rng):\n    xs = [int(v) for v in rng.integers(-50, 50, size=size)]\n"
                 "    return {'input': xs, 'target': None}\n",
    "metric": "def metric(given, target, output):\n"
              "    pairs = list(zip(given, output['f'], output['ff']))\n"
              "    return sum(1.0 for x, fx, ffx in pairs if ffx == x) / max(1, len(pairs))\n",
    "sizes": [4, 16, 64], "count": 50, "threshold": 1.0, "seed": 3}
INVOLUTION = "def run(xs):\n    f = [7 - x for x in xs]\n    return {'f': f, 'ff': [7 - y for y in f]}\n"


def spec_cases() -> list[tuple[str, dict, str]]:
    return [
        ("z3_bounded_counterexample", {"kind": "z3", "spec": BOUNDED}, "violated"),
        ("z3_real_inequality_holds", {"kind": "z3", "spec": HOLDS}, "discharged"),
        ("sympy_identity", {"kind": "sympy", "spec": {"symbols": {"x": "real", "y": "real"}, "lhs": "(x + y)**3",
                                                      "rhs": "x**3 + 3*x**2*y + 3*x*y**2 + y**3",
                                                      "relation": "=="}}, "discharged"),
        ("sympy_false_identity", {"kind": "sympy", "spec": {"symbols": {"x": "real"}, "lhs": "(x + 1)**2",
                                                            "rhs": "x**2 + 1", "relation": "=="}}, "violated"),
        ("simulation_involution", {"kind": "simulation", "spec": SIMULATION, "source": INVOLUTION}, "discharged"),
        ("audited_construction_recall_cap8",
         {"kind": "audited_construction", "spec": {"task": "recall", "sizes": [4, 8, 16], "params": {"V": 16}},
          "source": ref.CONSTRUCTIONS["recall_table_cap8"][0]}, "violated"),
        ("statement_match_strengthened_exponent",
         {"kind": "predicate_agreement", "spec": EXPONENT, "predicate": CLAIM_LINEAR, "count": 1000}, "violated"),
        ("statement_match_agrees",
         {"kind": "predicate_agreement", "spec": EXPONENT, "predicate": CLAIM_SAME, "count": 1000}, "discharged"),
    ]


def main(argv: list[str] | None = None) -> int:
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.math.runner import MathRunner

    parser = argparse.ArgumentParser(prog="python -m hybrid.math.spec_smoke")
    parser.add_argument("--out", type=Path, required=True, help="new directory for jobs, receipts and cases.json")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipts = out / "receipts"
    runner = MathRunner(Executor(out.parent.name, receipts), read_lock("tools")["reference"], out)
    cases = []
    for name, request, expected in spec_cases():
        started = time.monotonic()
        record = runner.check_library(request)
        job_ids = [run["job_id"] for run in record["runs"]]
        network = network_modes(receipts, job_ids)
        printed = record["runs"][0]["printed"] or {}
        ok = record["status"] == expected and record["decided"] and set(network) == {"none"}
        case = {"case": name, "expected": expected, "status": record["status"], "decided": record["decided"],
                "ok": ok, "tier": printed.get("tier"), "problems": record["problems"], "evidence": record["evidence"],
                "counterexample": record["counterexample"], "wall_s": round(time.monotonic() - started, 2),
                "job_ids": job_ids, "network": network, "check_hashes": record["check_hashes"]}
        cases.append(case)
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {case['status']} (expected {expected}) decided={case['decided']} "
              f"tier={case['tier']} wall_s={case['wall_s']} {case['evidence'][:200]} "
              f"{'; '.join(case['problems'])[:300]}", flush=True)
    (out / "cases.json").write_text(json.dumps(cases, indent=2, default=str) + "\n")
    return 0 if all(case["ok"] for case in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
