"""Phase 1 smoke tests of the tools image, run live in Podman.

    python -m hybrid.math.smoke --out results/phase1-<stamp>/smoke

Every Lean case and every program case runs twice in Podman (check_lean always runs twice; a deciding program
run is replayed, and the smoke runs the whole check a second time for programs so each case is executed in
Podman twice even when the first run does not decide). The hash-mismatch case is decided on the host and never
reaches Podman. Writes cases.json and prints one line per case. Exit 0 only when every case gives its expected
result and every job ran with --network=none.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from hybrid.math.lean_check import LeanCheck, statement_hash

SMALL = "def Claim : Prop := ∀ n : Nat, n + 0 = n\n"
MATHLIB = "import Mathlib\n\ndef Claim : Prop := ∀ a b : ℝ, 2 * a * b ≤ a ^ 2 + b ^ 2\n"
NATIVE = "def Claim : Prop := 2 ^ 10 = 1024\n"

LEAN_CASES = [
    ("lean_small_lemma", SMALL, "theorem claim_holds : Claim := fun _ => rfl\n", "pass"),
    ("lean_mathlib_lemma", MATHLIB,
     "theorem claim_holds : Claim := by\n  intro a b\n  nlinarith [sq_nonneg (a - b)]\n", "pass"),
    ("lean_reject_sorry", SMALL, "theorem claim_holds : Claim := by\n  sorry\n", "reject"),
    ("lean_reject_user_axiom", SMALL,
     "axiom bigbrane_magic : Claim\n\ntheorem claim_holds : Claim := bigbrane_magic\n", "reject"),
    ("lean_reject_native_decide", NATIVE,
     "theorem claim_holds : Claim := by\n  show 2 ^ 10 = 1024\n  native_decide\n", "reject"),
    # The prover restates the claim as a weaker proposition instead of proving Claim.
    ("lean_reject_restated_in_proof", SMALL,
     "theorem claim_holds : ∀ n : Nat, n + 0 = n ∨ True := fun _ => Or.inr trivial\n", "reject"),
]
# The statement file itself is changed after its hash was fixed: rejected on the host.
HASH_CASE = ("lean_reject_restated_statement_file", SMALL, "def Claim : Prop := ∀ n : Nat, n + 0 = n ∨ True\n",
             "theorem claim_holds : Claim := fun _ => Or.inr trivial\n", "reject")

Z3_SAT = r'''
import json
import z3
x, y = z3.Ints("x y")
s = z3.Solver()
s.add(x + y == 10, x - y == 4, x > 0, y > 0)
result = s.check()
if result == z3.sat:
    m = s.model()
    print(json.dumps({"status": "discharged", "evidence": "sat: x=%s y=%s" % (m[x], m[y]), "counterexample": None}))
else:
    print(json.dumps({"status": "unknown", "evidence": "solver returned %s" % result, "counterexample": None}))
'''
# Claim: for every integer x in [0, 1000], x*x != 2*x + 3. The negation is satisfiable (x = 3): a counterexample.
Z3_COUNTEREXAMPLE = r'''
import json
import z3
x = z3.Int("x")
s = z3.Solver()
s.add(x >= 0, x <= 1000, x * x == 2 * x + 3)
result = s.check()
if result == z3.sat:
    value = s.model()[x].as_long()
    print(json.dumps({"status": "violated", "evidence": "negation sat within bound [0, 1000]",
                      "counterexample": {"x": value}}))
elif result == z3.unsat:
    print(json.dumps({"status": "discharged", "evidence": "negation unsat within bound [0, 1000]",
                      "counterexample": None}))
else:
    print(json.dumps({"status": "unknown", "evidence": str(result), "counterexample": None}))
'''
SYMPY_IDENTITY = r'''
import json
import random
import sympy as sp
x = sp.symbols("x")
lhs = (x + 1) ** 3
rhs = x ** 3 + 3 * x ** 2 + 3 * x + 1
symbolic = sp.simplify(lhs - rhs) == 0
rng = random.Random(0)
points = [sp.Rational(rng.randint(-1000, 1000), rng.randint(1, 1000)) for _ in range(20)]
spot = all((lhs - rhs).subs(x, p) == 0 for p in points)
status = "discharged" if symbolic and spot else "violated"
print(json.dumps({"status": status, "evidence": "simplify(lhs - rhs) == 0: %s; 20 rational spot checks: %s"
                  % (symbolic, spot), "counterexample": None if status == "discharged" else {"points": str(points)}}))
'''
PROGRAM_CASES = [
    ("z3_satisfiable", "z3", Z3_SAT, "discharged"),
    ("z3_counterexample", "z3", Z3_COUNTEREXAMPLE, "violated"),
    ("sympy_identity", "sympy", SYMPY_IDENTITY, "discharged"),
]


def network_modes(receipts_dir: Path, job_ids: list[str]) -> list[str]:
    modes = []
    for job_id in job_ids:
        receipt = json.loads((receipts_dir / f"{job_id}.receipt.json").read_text())
        modes.append(receipt["effective_policy"]["network_mode"])
    return modes


def run_smoke(runner, receipts_dir: Path) -> list[dict]:
    cases = []
    for name, statement, proof, expected in LEAN_CASES:
        check = LeanCheck(statement, statement_hash(statement), proof)
        started = time.monotonic()
        result = runner.check_lean(check)
        job_ids = [run.job_id for run in result.runs]
        cases.append({"case": name, "tool": "lean", "expected": expected, "status": result.status,
                      "decided": result.decided, "ok": result.status == expected and result.decided,
                      "reasons": result.reasons, "axioms": result.axioms,
                      "check_s": [round(run.run_s, 2) for run in result.runs],
                      "wall_s": round(time.monotonic() - started, 2), "job_ids": job_ids,
                      "network": network_modes(receipts_dir, job_ids),
                      "runs": [asdict(run) for run in result.runs]})
    name, statement, changed, proof, expected = HASH_CASE
    result = runner.check_lean(LeanCheck(changed, statement_hash(statement), proof))
    cases.append({"case": name, "tool": "lean", "expected": expected, "status": result.status,
                  "decided": result.decided, "ok": result.status == expected and not result.runs,
                  "reasons": result.reasons, "axioms": None, "check_s": [], "wall_s": 0.0, "job_ids": [],
                  "network": [], "runs": []})
    for name, tool, program, expected in PROGRAM_CASES:
        records = [runner.check_program(tool, program) for _ in range(2)]
        job_ids = [run["job_id"] for record in records for run in record["runs"]]
        statuses = [record["status"] for record in records]
        cases.append({"case": name, "tool": tool, "expected": expected, "status": statuses[0],
                      "decided": all(r["decided"] for r in records),
                      "ok": statuses == [expected, expected] and all(r["decided"] for r in records),
                      "reasons": [p for r in records for p in r["problems"]],
                      "evidence": records[0]["evidence"], "counterexample": records[0]["counterexample"],
                      "check_s": [], "wall_s": None, "job_ids": job_ids,
                      "network": network_modes(receipts_dir, job_ids), "runs": records})
    for case in cases:
        if case["network"] and set(case["network"]) != {"none"}:
            case["ok"] = False
            case["reasons"].append(f"a job ran with network {case['network']}")
    return cases


def main(argv: list[str] | None = None) -> int:
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.math.runner import MathRunner

    parser = argparse.ArgumentParser(prog="python -m hybrid.math.smoke")
    parser.add_argument("--out", type=Path, required=True, help="new directory for jobs, receipts and cases.json")
    parser.add_argument("--run-id", default=None, help="executor run id (labels); default: the out dir name")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipts = out / "receipts"
    executor = Executor(args.run_id or out.parent.name, receipts)
    runner = MathRunner(executor, read_lock("tools")["reference"], out)
    cases = run_smoke(runner, receipts)
    (out / "cases.json").write_text(json.dumps(cases, indent=2, default=str) + "\n")
    for case in cases:
        print(f"{'OK  ' if case['ok'] else 'FAIL'} {case['case']}: {case['status']} (expected {case['expected']}) "
              f"check_s={case['check_s']} jobs={len(case['job_ids'])} {'; '.join(case['reasons'])[:300]}",
              flush=True)
    return 0 if all(case["ok"] for case in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
