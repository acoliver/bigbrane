"""The math library's known-number battery: conservation laws against published or hand-derived numbers, the cost
verifier on reference implementations (including a planted exponent error), the construction harness on hand-set
mechanisms, and the magic-X harness on an assumption that alone gives the advantage.

The same battery runs on the host (hybrid/tests/math/test_library.py) and in the tools image through the executor
(``MathRunner.check_library({"kind": "selfcheck"})``), where pytest is not installed. ``run()`` returns the
obligation-check contract: discharged when every case gives its expected result. Its output is deterministic (no
timings), so the executor's replay rule applies.

Self-contained (stdlib, NumPy, SymPy) so it runs in the tools image as part of ``bbmath``.
"""
from __future__ import annotations

import hashlib
import json
import math

from . import conservation as cl
from . import constructions, cost, magic_x, references, tiers

QUICK_NS = (32, 64, 128, 256)
# The planted quadratic costs about n^2/2 counted Python-level operations, so its full grid stops at 1024.
PLANTED_FULL_NS = (64, 128, 256, 512, 1024)


def _case(name: str, expected, got, ok: bool | None = None, detail=None) -> dict:
    return {"case": name, "expected": expected, "got": got, "ok": (expected == got) if ok is None else ok,
            "detail": detail}


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def conservation_cases() -> list[dict]:
    cases = []
    # Storage: 100 facts of 10 bits need exactly 1000 bits (counting).
    ok = cl.storage_check(1000, 100, 10)
    over = cl.storage_check(999, 100, 10)
    cases.append(_case("storage_exact_fit", "discharged", ok.status))
    cases.append(_case("storage_planted_overrun", ["violated", 100],
                       [over.status, (over.counterexample or {}).get("smallest_failing_facts")]))
    # Jelassi et al. Theorem 2.7: 10 bits, binary strings of length 20: error > 1 - 2^10 / 2^20.
    bound = cl.copy_error_lower_bound(10, 2, 20)
    cases.append(_case("copy_jelassi_bound_10bits_D2_L20", 1 - 2 ** -10, bound, _close(bound, 1 - 2 ** -10, 1e-12)))
    cases.append(_case("copy_jelassi_bound_64bits_D16_L16", 0.0, cl.copy_error_lower_bound(64, 16, 16)))
    cases.append(_case("copy_jelassi_bound_64bits_D16_L17", 0.9375, cl.copy_error_lower_bound(64, 16, 17)))
    copy = cl.copy_check(10, 2, 20, 0.5)
    cases.append(_case("copy_claim_above_bound", ["violated", 12],
                       [copy.status, (copy.counterexample or {}).get("smallest_failing_L")]))
    # INDEX: n bits deterministic; n * (1 - h2(0.11)) with h2(0.11) = 0.499916 by hand.
    cases.append(_case("index_exact_1024", 1024.0, cl.streaming_index_required_bits(1024)))
    noisy = cl.streaming_index_required_bits(1024, 0.11)
    cases.append(_case("index_error_0.11_1024", 512.086, round(noisy, 3), _close(noisy, 512.086, 0.01)))
    # Planted capacity overrun: recall of K=64 values over V=256 at accuracy 0.99 from 256 bits.
    recall = cl.associative_recall_check(256, 64, 256, 0.99)
    cases.append(_case("recall_planted_capacity_overrun", ["violated", 33],
                       [recall.status, (recall.counterexample or {}).get("smallest_failing_K")]))
    cases.append(_case("recall_fits", "discharged", cl.associative_recall_check(512, 64, 256, 1.0).status))
    # Sample information: 1024 hypotheses, one bit per label: 10 examples (binary search attains it).
    cases.append(_case("sample_1024_hypotheses_binary", 10, cl.sample_bound(10, 0, 1)))
    # A permutation of 10 items, each example shows one (x, pi(x)) pair: ceil(log2(10!) / log2(10)) = 7.
    cases.append(_case("sample_permutation_10", 7, cl.sample_bound(math.log2(math.factorial(10)), 0, math.log2(10))))
    # Fano: 2^100 hypotheses at error 0.1 with one bit per example: (0.9 * 100 - 1) = 89.
    cases.append(_case("sample_fano_error_0.1", 89, cl.sample_bound(100, 0, 1, 0.1)))
    sample = cl.sample_check(10, 2, 1, 5)
    cases.append(_case("sample_claim_too_few", ["violated", 8],
                       [sample.status, sample.bound["required_examples"]]))
    cases.append(_case("dyck_depth_8_in_8_bits", "discharged", cl.dyck_check(8, 8, 1.0).status))
    dyck = cl.dyck_check(8, 9, 1.0)
    cases.append(_case("dyck_depth_9_in_8_bits", ["violated", 9],
                       [dyck.status, (dyck.counterexample or {}).get("smallest_failing_depth")]))
    cases.append(_case("read_bound", "violated", cl.read_check(100, 99).status))
    card = cl.report(cl.check_claims({"state_bits": 256, "recall": [{"K": 64, "V": 256, "accuracy": 0.99}],
                                      "copy": [{"D": 4, "L": 64, "accuracy": 0.99}]}))
    cases.append(_case("card_claims_report", ["violated", ["conservation:associative_recall"]],
                       [card["status"], card.get("kill_codes")]))
    return cases


def cost_cases(ns=QUICK_NS, planted_ns=QUICK_NS) -> list[dict]:
    cases = []
    for name, (source, claim, expected) in references.REFERENCES.items():
        grid = planted_ns if name.startswith("planted") else ns
        result = cost.verify_cost(source, cost.CostClaim(**claim), ns=grid)
        failed = sorted({f["check"] for f in result["failures"]})
        cases.append(_case(f"cost_{name}", expected, result["status"],
                           detail={"failed_checks": failed, "counterexample": result["counterexample"],
                                   "slopes_n": [s for s in result["slopes"] if s["parameter"] == "n"],
                                   "ns": list(grid)}))
    planted = next(c for c in cases if c["case"] == "cost_planted_quadratic")
    cases.append(_case("cost_planted_quadratic_caught_by_exponent", True,
                       "exponent_n" in planted["detail"]["failed_checks"]))
    return cases


def construction_cases() -> list[dict]:
    cases = []
    for task, size in (("copy", 16), ("recall", 16), ("dyck", 8), ("composition", 16), ("binding", 8)):
        cases.append(_case(f"oracle_self_score_{task}", 1.0, constructions.oracle_self_score(task, size, 50, 0)))
        first = constructions.generate(task, size, 20, 7)
        again = constructions.generate(task, size, 20, 7)
        digest = hashlib.sha256(json.dumps([list(i.tokens) for i in first]).encode()).hexdigest()
        cases.append(_case(f"generator_deterministic_{task}", True, first == again, detail=digest[:16]))
    for name, (source, task, sizes, params, expected) in references.CONSTRUCTIONS.items():
        result = constructions.run_construction(source, task, sizes, params=params, count=100)
        cases.append(_case(f"construction_{name}", expected, result["status"],
                           detail={"accuracy": [r["accuracy"] for r in result["results"]],
                                   "state_peak_words": [r["state_peak_words"] for r in result["results"]]}))
    return cases


def magic_x_cases() -> list[dict]:
    claim = magic_x.AdvantageClaim(**references.MAGIC_X_CLAIM)
    oracle = magic_x.magic_x_test(claim, magic_x.Assumption("retrieval_oracle", "store", references.RETRIEVAL_ORACLE))
    parity = magic_x.magic_x_test(claim, magic_x.Assumption("position_parity", "oracle", references.POSITION_PARITY))
    exempt = magic_x.magic_x_test(claim, magic_x.Assumption("retrieval_oracle_measured", "store",
                                                            references.RETRIEVAL_ORACLE, "measured", True))
    return [
        _case("magic_x_oracle_alone_gives_advantage", "violated", oracle["status"],
              detail=oracle["counterexample"]),
        _case("magic_x_uninformative_assumption", "discharged", parity["status"]),
        _case("magic_x_measured_with_cost_in_p1_is_exempt", "discharged", exempt["status"]),
    ]


def tier_cases() -> list[dict]:
    return [
        _case("tier_lean_proven", "proven", tiers.verdict_phrase(tiers.LEAN)),
        _case("tier_smt_bounded", "proven up to bound x <= 1000", tiers.verdict_phrase(tiers.SMT_BOUNDED,
                                                                                      bound="x <= 1000")),
        _case("tier_argued_never_satisfies", False, tiers.satisfies_obligation(tiers.ARGUED)),
    ]


def run(quick: bool = True) -> dict:
    """Run the whole battery. ``quick`` uses n in QUICK_NS for the cost references; otherwise the plan's grid
    (cost.DEFAULT_NS, n from 64 to 4096) and PLANTED_FULL_NS for the planted references."""
    grids = (QUICK_NS, QUICK_NS) if quick else (cost.DEFAULT_NS, PLANTED_FULL_NS)
    cases = (conservation_cases() + cost_cases(*grids) + construction_cases()
             + magic_x_cases() + tier_cases())
    failed = [c for c in cases if not c["ok"]]
    return {"status": "violated" if failed else "discharged",
            "evidence": f"{len(cases) - len(failed)} of {len(cases)} cases gave their expected result"
                        + (f"; failed: {[c['case'] for c in failed]}" if failed else ""),
            "counterexample": failed[0] if failed else None, "cases": cases}
