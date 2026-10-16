"""Harness and evaluator gates on card falsifier tasks, run through the local executor
(results/falsifier-20261011/recall-investigation.md, section 5)."""
from __future__ import annotations

import json

from hybrid.bench import falsifier as F
from hybrid.bench.evaluate import evaluate_candidate
from hybrid.bench.local import LocalExecutor
from hybrid.executor.job import Caps
from hybrid.executor.lock import read_lock

from .conftest import ROOT, TINY
from .falsifier_programs import MARKOV, MARKOV_RULE, SHIFT, design, variant

CAPS = Caps(wall_s=120, memory_mb=2048, pids=64, tmp_mb=64)
FIXTURES = ROOT / "hybrid" / "tests" / "research" / "fixtures" / "assoc_recall"
CODE_FILES = {"main.py", "falsifier_eval.py", "metrics.py", "falsifier_reference.py"}


def validate_local(work_dir, program, rules, **kwargs):
    executor = LocalExecutor("gates-test", work_dir / "receipts")
    return F.run_validation_job(executor, read_lock()["reference"], program, 0, rules, work_dir / "jobs", CAPS, 60.0,
                                **kwargs)[0]


def test_the_recorded_data_is_rejected_by_the_validation_job(work_dir):
    recorded = json.loads((FIXTURES / "design.json").read_text())
    report = validate_local(work_dir, recorded["program"], recorded["rules"], train_max_len=128)
    text = "\n".join(report["problems"])
    assert not report["ok"]
    assert ("group K=256 has test prefixes of 771 tokens but the longest training sequence has 148, so no model "
            "trains at this length; include training episodes from every group") in text
    assert "82 training sequences are longer than the 128 tokens the candidate trains on (longest 148)" in text
    controls = report["checks"]["controls"] if report.get("checks") else None
    assert controls is None or controls["lookup"]["K=256"] == 1.0      # no leak: lookup solves it, recency does not
    assert set(report["code_sha256"]) == CODE_FILES


# ----- harness gates on small programs ------------------------------------------------------------------

SHIFT_RULE_ONLY = {"claims": ["capability.1"], "quote": "Accuracy >= 0.9 at m = 2", "if_met": "supported",
                   "conditions": [{"entrant": "candidate", "minus": None, "metric": "acc", "group": "m=2", "op": ">=",
                                   "value": 0.9}]}


def test_group_coverage_and_guaranteed_training_length(work_dir):
    # SHIFT: test prefixes of 21 tokens, training sequences of 22: covered
    assert validate_local(work_dir, SHIFT, [SHIFT_RULE_ONLY], train_max_len=22)["ok"]
    short = validate_local(work_dir, SHIFT, [SHIFT_RULE_ONLY], train_max_len=20)
    assert not short["ok"] and "64 training sequences are longer than the 20 tokens the candidate trains on" in \
        "\n".join(short["problems"])
    # m=2 demonstrations only in the longest test group: shorter training sequences leave it uncovered
    uncovered = variant(SHIFT, "    for _ in range(64):\n        seq, target = episode(r, int(r.integers(1, 3)))",
                        "    for _ in range(64):\n        seq, target = episode(r, 1)\n        seq = seq[:-6]")
    report = validate_local(work_dir, uncovered, [SHIFT_RULE_ONLY])
    assert not report["ok"] and any("has test prefixes of 21 tokens but the longest training sequence has" in p
                                    for p in report["problems"])


def test_a_leak_control_that_meets_a_condition_rejects_the_task(work_dir):
    """The answer of this task is the token three back from the end: recency_3 scores 1.0."""
    leak = '''
import numpy as np


def generate(seed):
    r = np.random.default_rng(seed)

    def episode():
        seq = [int(x) for x in r.integers(1, 9, size=10)]
        return seq + [seq[-3]]
    train = [episode() for _ in range(64)]
    full = [episode() for _ in range(64)]
    return {"vocab_size": 9, "train": train, "prefixes": [s[:-1] for s in full], "next_tokens": [s[-1] for s in full],
            "oracle_kind": "answer", "oracle": [s[-1] for s in full], "group": [0] * 64, "group_labels": ["all"]}
'''
    rule = dict(SHIFT_RULE_ONLY, conditions=[dict(SHIFT_RULE_ONLY["conditions"][0], group="all")])
    report = validate_local(work_dir, leak, [rule])
    assert not report["ok"]
    assert "rule 0 is met by the recency_3 control, so the answer is predictable without the mechanism" in report["problems"]


def test_solvable_by_lookup_requires_lookup_to_solve_every_group(work_dir):
    report = validate_local(work_dir, SHIFT, [SHIFT_RULE_ONLY], solvable_by="lookup")
    assert not report["ok"] and any("declared solvable_by lookup" in p for p in report["problems"])
    assert validate_local(work_dir, SHIFT, [SHIFT_RULE_ONLY])["ok"]


def test_the_bench_result_flags_evaluator_code_that_differs_from_the_validated_code(work_dir, template_dir):
    report, out = F.run_validation_job(LocalExecutor("gates-test", work_dir / "r"), read_lock()["reference"], MARKOV, 0,
                                       [MARKOV_RULE], work_dir / "jobs", CAPS, 60.0)
    assert report["ok"]
    task = F.write_task(work_dir / "tasks" / "markov", out, design("markov", MARKOV, [MARKOV_RULE]), report)
    executor = LocalExecutor("gates-test", work_dir / "bench-receipts")
    result = evaluate_candidate(template_dir, TINY, work_dir / "run1", executor, baselines=["ngram"],
                                cache_dir=work_dir / "cache", falsifiers=[task.directory], log=lambda _: None)
    record = result["host"]["falsifier_code"]
    assert record["tasks"]["markov"]["same_code"] is True
    assert record["evaluator_code_sha256"]["falsifier_eval.py"] == report["code_sha256"]["falsifier_eval.py"]
    # the candidate's describe() reaches the evaluator, and the template reports the contract fields
    candidate = result["falsifiers"]["markov"]["entrants"]["candidate"]
    assert candidate["describe"]["train_max_len"] == 2048 and candidate["describe"]["hyper"] == {"context": 1}
    # a task validated under other code is flagged
    meta = json.loads((task.directory / "task.json").read_text())
    meta["validation"]["code_sha256"]["falsifier_eval.py"] = "0" * 64
    (task.directory / "task.json").write_text(json.dumps(meta))
    flagged = evaluate_candidate(template_dir, TINY, work_dir / "run2", executor, baselines=["ngram"],
                                 cache_dir=work_dir / "cache", falsifiers=[task.directory], log=lambda _: None)
    assert flagged["host"]["falsifier_code"]["tasks"]["markov"]["same_code"] is False


def test_the_evaluator_blocks_a_task_the_candidate_cannot_train_on_in_full(work_dir, template_dir):
    """A task with 24-token training sequences against a candidate that reports train_max_len 8."""
    report, out = F.run_validation_job(LocalExecutor("gates-test", work_dir / "r"), read_lock()["reference"], MARKOV, 0,
                                       [MARKOV_RULE], work_dir / "jobs", CAPS, 60.0)
    task = F.write_task(work_dir / "tasks" / "markov", out, design("markov", MARKOV, [MARKOV_RULE]), report)
    short = work_dir / "short"
    short.mkdir()
    (short / "candidate.py").write_text((template_dir / "candidate.py").read_text().replace(
        '"train_max_len": 2048', '"train_max_len": 8'))
    result = evaluate_candidate(short, TINY, work_dir / "run", LocalExecutor("gates-test", work_dir / "r2"),
                                baselines=["ngram"], cache_dir=work_dir / "cache", falsifiers=[task.directory],
                                log=lambda _: None)
    decision = result["falsifiers"]["markov"]["decisions"][0]
    assert decision["outcome"] == "inconclusive"
    assert "training sequences of 24 tokens but the candidate trains on at most 8 tokens" in decision["reason"]


# ----- reference-mechanism capacity control (falsifier_reference.py, harness.reference_problems) -------------

PAIR_RECALL = '''
import numpy as np

SEP, QRY, CONTENT = 1, 2, 64


def generate(seed):
    r = np.random.default_rng(seed)

    def episode(k):
        keys = [int(x) + 3 for x in r.choice(CONTENT, size=2 * k, replace=False)]
        pairs = [(keys[2 * i], keys[2 * i + 1]) for i in range(k)]
        seq = [0]
        for a, b in pairs:
            seq += [a, b, SEP]
        j = int(r.integers(k))
        return seq + [QRY, pairs[j][0], pairs[j][1]], k
    sizes = (2, 8)
    train = [episode(sizes[i % 2])[0] for i in range(96)]
    tests = [episode(sizes[i % 2]) for i in range(64)]
    return {"vocab_size": 3 + CONTENT, "train": train, "prefixes": [s[:-1] for s, _ in tests],
            "next_tokens": [s[-1] for s, _ in tests], "oracle_kind": "answer", "oracle": [s[-1] for s, _ in tests],
            "group": [sizes.index(k) for _, k in tests], "group_labels": ["K=2", "K=8"]}
'''
REFERENCE = {"kind": "pair_memory", "d": 16, "min_token": 3}


def acc_rule(if_met, op, value, group="K=8", **extra):
    return {"claims": ["capability.1"], "quote": "Accuracy at K = 8", "if_met": if_met,
            "conditions": [{"entrant": "candidate", "minus": None, "metric": "acc", "group": group, "op": op,
                            "value": value}], **extra}


def recorded_data():
    recorded = json.loads((FIXTURES / "design.json").read_text())
    namespace = {}
    exec(recorded["program"], namespace)
    return recorded, namespace["generate"](0)


def test_the_reference_reproduces_the_investigated_capacities_on_the_recorded_task():
    from hybrid.bench import falsifier_harness as H

    recorded, data = recorded_data()
    d64 = H.reference_accuracies(data, {"kind": "pair_memory", "d": 64, "min_token": 3})
    assert [round(d64["rls"][g], 2) for g in ("K=16", "K=32", "K=48", "K=256")] == [1.0, 1.0, 1.0, 0.70]
    assert round(d64["la"]["K=256"], 2) == 0.54
    d32 = H.reference_accuracies(data, {"kind": "pair_memory", "d": 32, "min_token": 3})
    assert round(d32["rls"]["K=256"], 2) == 0.09 and round(d32["rls"]["K=48"], 2) == 0.96
    # the recorded limit rule (accuracy at most 0.6 at K = 256) would refute the card's own d = 64 operator
    assert H.reference_problems(recorded["rules"], d64["rls"]) == [
        "the card's own mechanism violates limit rule 1 (reference acc 0.70 > 0.60), so the rule would refute a "
        "correct implementation"]


def test_reference_gates_on_falsifier_limit_and_control_rules():
    from hybrid.bench import falsifier_harness as H

    accuracies = {H.ALL: 0.8, "K=2": 0.95, "K=8": 0.70}
    falsifier = acc_rule("refuted", ">=", 0.6)
    limit = acc_rule("supported", "<=", 0.6)
    control = acc_rule("supported", ">=", 0.9, group="K=2")
    unmet_control = acc_rule("supported", ">=", 0.9)
    assert H.reference_problems([falsifier], accuracies) == [
        "the card's own d x d operator meets falsifier rule 0, so observing it cannot mean something other than the "
        "operator stores the bindings"]
    assert H.reference_problems([acc_rule("refuted", ">=", 0.9)], accuracies) == []
    assert H.reference_problems([limit], accuracies)[0].startswith("the card's own mechanism violates limit rule 0")
    assert H.reference_problems([acc_rule("supported", "<=", 0.9)], accuracies) == []
    assert H.reference_problems([control, dict(limit, control_rule=0)], accuracies)[0].startswith(
        "the card's own mechanism violates limit rule 1")
    loose = acc_rule("supported", "<=", 0.9)
    assert H.reference_problems([dict(loose, control_rule=1), control], accuracies) == []
    assert H.reference_problems([unmet_control, dict(limit, control_rule=0, conditions=[dict(
        limit["conditions"][0], value=0.9)])], accuracies) == [
        "the card's own mechanism does not meet control rule 0, so no build of the card can pass it"]
    # kl conditions and conditions naming another entrant cannot be scored from the reference's accuracy
    kl = acc_rule("refuted", ">=", 0.1)
    kl["conditions"][0]["metric"] = "kl"
    versus = acc_rule("refuted", ">=", 0.1)
    versus["conditions"][0]["minus"] = "ngram"
    assert H.reference_problems([kl, versus], accuracies) == []


def test_the_validation_job_scores_the_declared_reference(work_dir):
    fine = [acc_rule("supported", ">=", 0.9)]
    report = validate_local(work_dir, PAIR_RECALL, fine, reference=REFERENCE)
    assert report["ok"], report["problems"]
    accuracy = report["checks"]["reference"]["accuracy"]
    assert accuracy["rls"]["K=8"] > 0.9 and accuracy["la"]["K=8"] < accuracy["rls"]["K=8"]
    assert "reference" not in validate_local(work_dir, PAIR_RECALL, fine)["checks"]
    bad = [acc_rule("refuted", ">=", 0.9), acc_rule("supported", "<=", 0.6, control_rule=2),
           acc_rule("supported", ">=", 0.999)]
    report = validate_local(work_dir, PAIR_RECALL, bad, reference=REFERENCE)
    text = "\n".join(report["problems"])
    assert not report["ok"]
    assert "the card's own d x d operator meets falsifier rule 0" in text
    assert "the card's own mechanism violates limit rule 1 (reference acc 0.97 > 0.60)" in text
    assert "the card's own mechanism does not meet control rule 2" in text
    assert set(report["code_sha256"]) == CODE_FILES
