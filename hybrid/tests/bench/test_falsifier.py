"""Card falsifier tasks in the bench: static checks, the validation harness, scoring and rule decisions, and
falsifier jobs in evaluate_candidate (local executor; the Podman versions run with BIGBRANE_BENCH_PODMAN=1)."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from hybrid.bench import falsifier as F
from hybrid.bench import falsifier_eval as FE
from hybrid.bench.evaluate import evaluate_candidate
from hybrid.bench.local import LocalExecutor
from hybrid.executor.job import Caps
from hybrid.executor.lock import read_lock

from .conftest import STAMP, TINY, podman
from .falsifier_programs import MARKOV, MARKOV_RULE, SHIFT, SHIFT_RULE, design, variant

CAPS = Caps(wall_s=120, memory_mb=2048, pids=64, tmp_mb=64)


def validate_local(work_dir, program, rules, timeout=30.0):
    executor = LocalExecutor("falsifier-test", work_dir / "receipts")
    return F.run_validation_job(executor, read_lock()["reference"], program, 0, rules, work_dir / "jobs", CAPS,
                                timeout)


def validated_task(work_dir, task_id, program, rules):
    report, out = validate_local(work_dir, program, rules)
    assert report["ok"], report
    return F.write_task(work_dir / "tasks" / task_id, out, design(task_id, program, rules), report)


# ----- static checks and rule names -----------------------------------------------------------------

def test_static_checks_allow_the_fixture_programs_and_reject_escapes():
    assert F.static_problems(SHIFT) == [] and F.static_problems(MARKOV) == []
    bad = F.static_problems("import os\nimport numpy as np\ndef make(seed):\n    open('x')\n    np.save('y', 1)\n"
                            "    return ().__class__\n")
    text = "\n".join(bad)
    for phrase in ("forbidden import os", "forbidden call open()", "file access through save",
                   "dunder attribute access __class__", "no top-level function generate(seed)"):
        assert phrase in text, phrase
    assert F.static_problems("def generate(seed:\n")[0].startswith("syntax error")


def test_rule_names_must_be_known_and_involve_the_candidate():
    rule = {"claims": ["c"], "quote": "q", "if_met": "supported", "conditions": [
        {"entrant": "gru", "minus": "ngram", "metric": "acc", "group": None, "op": ">=", "value": 0.1},
        {"entrant": "candidate", "minus": "rwkv", "metric": "acc", "group": None, "op": ">=", "value": 0.1},
        {"entrant": "candidate", "minus": "candidate", "metric": "acc", "group": None, "op": ">=", "value": 0.1}]}
    problems = F.rule_problems([rule], ["ngram", "gru"])
    assert any("condition 0 does not involve the candidate" in p for p in problems)
    assert any("'rwkv'" in p for p in problems)
    assert any("subtracts an entrant from itself" in p for p in problems)
    assert F.rule_problems([SHIFT_RULE, MARKOV_RULE], ["ngram"]) == []


# ----- validation harness (local executor) ----------------------------------------------------------

def test_valid_programs_pass_with_the_oracle_self_score_and_analytic_levels(work_dir):
    for program, rule in ((SHIFT, SHIFT_RULE), (MARKOV, MARKOV_RULE)):
        report, out = validate_local(work_dir, program, [rule])
        assert report["ok"] and report["deterministic"] and report["violations"] == [], report
        checks = report["checks"]
        assert checks["oracle_self_score"] == {"kl": 0.0, "acc": 1.0}
        for name in ("uniform", "unigram"):
            for metric in ("kl", "acc"):
                assert math.isclose(checks["references"][name][metric], checks["analytic"][name][metric],
                                    rel_tol=1e-9, abs_tol=1e-12)
        assert {"train.json", "prefixes.json", "oracle.npz", "meta.json"} <= {p.name for p in out.iterdir()}
    assert report["checks"]["analytic"]["uniform"]["acc"] == 0.5   # MARKOV: two tokens


def test_a_validated_task_directory_holds_inputs_oracle_and_unigram(work_dir):
    task = validated_task(work_dir, "shift", SHIFT, [SHIFT_RULE])
    assert (task.vocab_size, task.oracle_kind, task.group_labels, task.n_prefixes) == (20, "answer", ["m=1", "m=2"], 64)
    with np.load(task.directory / "oracle.npz") as data:
        assert data["unigram"].shape == (20,) and math.isclose(data["unigram"].sum(), 1.0)
        assert (data["oracle"] == data["next_tokens"]).all()
    assert json.loads((task.directory / "task.json").read_text())["validation"]["ok"] is True


FAILURES = {
    "clock or entropy": (variant(SHIFT, "r = np.random.default_rng(seed)",
                                 "import random\n    r = np.random.default_rng(seed + random.randrange(10**6))"),
                         "not deterministic on replay"),
    "set order": (variant(SHIFT, "rest = [q for q in range(NC) if q not in demos]",
                          "rest = [int(x[:-1]) for x in {str(q) + 'x' for q in range(NC) if q not in demos}]"),
                  "not deterministic on replay"),
    "oracle off by one": (variant(SHIFT, '"oracle": list(nxt)', '"oracle": [(t + 1) % NC for t in nxt]'),
                          "next tokens disagree with the oracle"),
    "dist oracle misses the sampled token": (variant(MARKOV, "oracle.append(P[seq[11]].tolist())",
                                                     "oracle.append([1.0, 0.0] if seq[12] == 1 else [0.0, 1.0])"),
                                             "zero oracle probability"),
    "file write": (variant(SHIFT, "def generate(seed):\n",
                           "def generate(seed):\n    open('bb-escape-test', 'w').write('x')\n"), "escape"),
    "file read outside": (variant(SHIFT, "def generate(seed):\n",
                                  "def generate(seed):\n    open('/etc/hosts').read()\n"), "escape"),
    "network": (variant(SHIFT, "def generate(seed):\n", "def generate(seed):\n    import socket\n    socket.socket()\n"),
                "escape"),
    "subprocess": (variant(SHIFT, "def generate(seed):\n",
                           "def generate(seed):\n    import subprocess\n    subprocess.run(['true'])\n"), "escape"),
    "caps": (variant(SHIFT, "for _ in range(64):", "for _ in range(4000):"), "cap 2048"),
    "crash": (variant(SHIFT, "def generate(seed):\n", "def generate(seed):\n    raise RuntimeError('boom')\n"),
              "raised RuntimeError: boom"),
    "hang": (variant(SHIFT, "def generate(seed):\n", "def generate(seed):\n    while True:\n        pass\n"),
             "did not finish"),
    "small group": (variant(SHIFT, "for _ in range(32):", "for _ in range(8):"), "at least"),
    "missing key": (variant(SHIFT, '"next_tokens": nxt, ', ""), "missing ['next_tokens']"),
}


@pytest.mark.parametrize("case", sorted(FAILURES))
def test_validation_rejects(work_dir, case):
    program, phrase = FAILURES[case]
    report, out = validate_local(work_dir, program, [SHIFT_RULE], timeout=5.0)
    assert not report["ok"]
    assert any(phrase in problem for problem in report["problems"]), report["problems"]
    assert not (out / "train.json").exists() or case in ("clock or entropy", "set order")


def test_validation_rejects_a_rule_a_trivial_predictor_meets_or_that_names_a_missing_group(work_dir):
    vacuous = dict(SHIFT_RULE, conditions=[dict(SHIFT_RULE["conditions"][0], value=0.01, group="m=1")])
    report, _ = validate_local(work_dir, SHIFT, [vacuous])
    assert not report["ok"] and "replaced by uniform" in report["problems"][0]
    refuting = dict(SHIFT_RULE, if_met="refuted")    # the vacuity check is for rules that support the card
    assert validate_local(work_dir, SHIFT, [refuting])[0]["ok"]
    report, _ = validate_local(work_dir, SHIFT, [dict(vacuous, if_met="refuted")])   # but a leak control meets it
    assert not report["ok"] and any("control, so the answer is predictable" in p for p in report["problems"])
    unknown = dict(SHIFT_RULE, conditions=[dict(SHIFT_RULE["conditions"][0], group="m=7")])
    report, _ = validate_local(work_dir, SHIFT, [unknown])
    assert not report["ok"] and "group 'm=7'" in report["problems"][0]


def test_a_job_without_a_report_is_a_failed_report():
    report = F.parse_report("Traceback ...\nKilled\n", "timeout", "wall cap 120s exceeded")
    assert report["ok"] is False and "timeout" in report["problems"][0]


# ----- scoring and decisions -------------------------------------------------------------------------

def test_analytic_reference_levels_equal_the_metrics_code():
    rng = np.random.default_rng(3)
    vocab, n = 7, 50
    unigram = rng.dirichlet(np.ones(vocab))
    answers = rng.integers(0, vocab, size=n)
    group = np.zeros(n, dtype=np.int64)
    dist = rng.dirichlet(np.ones(vocab) * 0.3, size=n)
    dist[:5] = np.eye(vocab)[[0, 1, 2, 3, 4]]                     # rows with zeros and ties
    for kind, oracle in (("answer", answers), ("dist", dist)):
        refs = FE.references(kind, oracle, group, ["all"], unigram)
        analytic = FE.analytic_reference_levels(kind, oracle, unigram)
        for name in ("uniform", "unigram"):
            for metric in ("kl", "acc"):
                assert math.isclose(refs[name][metric], analytic[name][metric], rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(FE.references("answer", answers, group, ["all"], unigram)["uniform"]["kl"], math.log(vocab))


def scores(acc, kl, labels=("m=1", "m=2")):
    block = {"acc": acc, "kl": kl, "n": 32}
    return {**block, "groups": {label: dict(block) for label in labels}}


REFS = {"uniform": scores(0.05, 3.0), "unigram": scores(0.06, 2.9)}


def test_decide_supports_refutes_and_is_inconclusive_when_the_candidate_did_not_train():
    rule = SHIFT_RULE
    assert FE.decide(rule, {"candidate": scores(0.95, 0.2), **REFS})["outcome"] == "supported"
    refuted = FE.decide(rule, {"candidate": scores(0.5, 1.0), **REFS})
    assert refuted["outcome"] == "refuted" and refuted["clauses"][0]["lhs"] == 0.5
    untrained = FE.decide(rule, {"candidate": scores(0.07, 2.88), **REFS})
    assert untrained["outcome"] == "inconclusive" and "did not train" in untrained["reason"]
    assert untrained["training"]["trained"] is False
    assert FE.decide(rule, {"candidate": None, **REFS})["outcome"] == "inconclusive"


def test_decide_with_a_baseline_comparison_and_a_falsifying_condition():
    rule = {"claims": ["falsifier.1"], "quote": "q", "if_met": "refuted", "conditions": [
        {"entrant": "candidate", "minus": "linear_attention", "metric": "acc", "group": "m=2", "op": "<",
         "value": 0.3}]}
    assert FE.decide(rule, {"candidate": scores(0.9, 0.3), "linear_attention": scores(0.4, 1.0), **REFS})[
        "outcome"] == "supported"   # beat the baseline by 0.5: the falsifier was not observed
    assert FE.decide(rule, {"candidate": scores(0.5, 0.9), "linear_attention": scores(0.4, 1.0), **REFS})[
        "outcome"] == "refuted"
    missing = FE.decide(rule, {"candidate": scores(0.9, 0.3), "linear_attention": None, **REFS})
    assert missing["outcome"] == "inconclusive" and missing["missing"] == ["linear_attention"]


def test_trivially_met_only_checks_candidate_only_rules_that_support():
    easy = dict(SHIFT_RULE, conditions=[dict(SHIFT_RULE["conditions"][0], value=0.01)])
    assert FE.trivially_met(easy, REFS) == ["uniform", "unigram"]
    assert FE.trivially_met(SHIFT_RULE, REFS) == []
    assert FE.trivially_met(dict(easy, if_met="refuted"), REFS) == []
    limit = dict(SHIFT_RULE, conditions=[dict(SHIFT_RULE["conditions"][0], op="<=", value=0.6)])
    assert FE.trivially_met(limit, REFS) == []          # a predicted limit; decide() needs a trained candidate
    assert FE.decide(limit, {"candidate": scores(0.07, 2.88), **REFS})["outcome"] == "inconclusive"
    assert FE.decide(limit, {"candidate": scores(0.4, 1.5), **REFS})["outcome"] == "supported"


# ----- falsifier jobs in evaluate_candidate -----------------------------------------------------------

def _oracle_hashes(task):
    with np.load(task.directory / "oracle.npz") as data:
        return {hashlib.sha256(data[k].tobytes()).hexdigest() for k in ("oracle", "next_tokens")}


def check_falsifier_result(result, tasks, entrants, bench_dir):
    assert set(result["falsifiers"]) == {t.task_id for t in tasks}
    for task in tasks:
        block = result["falsifiers"][task.task_id]
        assert set(block["entrants"]) == set(entrants)
        for name in entrants:
            entry = block["entrants"][name]
            assert entry["status"] == "ok", entry
            assert 0 <= entry["metrics"]["acc"] <= 1 and entry["metrics"]["kl"] >= 0
            assert set(entry["metrics"]["groups"]) == set(task.group_labels)
        assert set(block["references"]) == {"uniform", "unigram"}
        assert [d["claims"] for d in block["decisions"]] == [r["claims"] for r in task.design["rules"]]
        assert all(d["outcome"] in ("supported", "refuted", "inconclusive") for d in block["decisions"])
    jobs = [d for d in (bench_dir / "jobs").iterdir() if (d / "in" / "falsifier_runner.py").is_file()]
    assert len(jobs) == len(tasks) * len(entrants)
    secrets = set().union(*(_oracle_hashes(t) for t in tasks))
    for job in jobs:
        names = {p.name for p in (job / "in").rglob("*") if p.is_file()}
        assert "oracle.npz" not in names and not any(n.endswith(".npz") for n in names)
        for path in (job / "in").rglob("*.json"):
            assert hashlib.sha256(path.read_bytes()).hexdigest() not in secrets


def test_falsifier_tasks_run_for_the_candidate_and_every_baseline_and_are_scored_in_the_evaluator(work_dir,
                                                                                                  template_dir):
    tasks = [validated_task(work_dir, "shift", SHIFT, [SHIFT_RULE]),
             validated_task(work_dir, "markov", MARKOV, [MARKOV_RULE])]
    executor = LocalExecutor("falsifier-test", work_dir / "bench-receipts")
    cache = work_dir / "cache"
    first = evaluate_candidate(template_dir, TINY, work_dir / "run1", executor, baselines=["ngram"], cache_dir=cache,
                               falsifiers=[t.directory for t in tasks], log=lambda _: None)
    check_falsifier_result(first, tasks, ["candidate", "ngram"], work_dir / "run1" / "bench")
    markov = first["falsifiers"]["markov"]["decisions"][0]
    assert markov["outcome"] == "supported" and markov["training"]["trained"]   # the bigram template learns it
    shift = first["falsifiers"]["shift"]["decisions"][0]
    assert shift["outcome"] == "inconclusive"                                     # the bigram cannot learn it
    assert set(first["host"]["falsifier_receipts"]) == {"candidate", "ngram"}
    # the cached n-gram still runs the falsifier tasks
    second = evaluate_candidate(template_dir, TINY, work_dir / "run2", executor, baselines=["ngram"], cache_dir=cache,
                                falsifiers=[t.directory for t in tasks], log=lambda _: None)
    assert second["entrants"]["ngram"]["job_id"] == first["entrants"]["ngram"]["job_id"]
    assert second["falsifiers"]["markov"]["entrants"]["ngram"]["metrics"] == \
        first["falsifiers"]["markov"]["entrants"]["ngram"]["metrics"]
    without = evaluate_candidate(template_dir, TINY, work_dir / "run3", executor, baselines=["ngram"],
                                 cache_dir=cache, log=lambda _: None)
    assert "falsifiers" not in without


def test_a_candidate_that_crashes_on_a_falsifier_task_is_job_failed_and_inconclusive(work_dir, template_dir):
    task = validated_task(work_dir, "markov", MARKOV, [MARKOV_RULE])
    crashing = work_dir / "crashing"
    crashing.mkdir()
    (crashing / "candidate.py").write_text(template_dir.joinpath("candidate.py").read_text().replace(
        "    def predict_proba(self, prefixes: list[list[int]]) -> np.ndarray:\n",
        "    def predict_proba(self, prefixes: list[list[int]]) -> np.ndarray:\n"
        "        if self.vocab_size == 2:\n            raise RuntimeError('cannot do two tokens')\n"))
    result = evaluate_candidate(crashing, TINY, work_dir / "run", LocalExecutor("falsifier-test", work_dir / "r"),
                                baselines=["ngram"], cache_dir=work_dir / "cache", falsifiers=[task.directory],
                                log=lambda _: None)
    block = result["falsifiers"]["markov"]
    assert block["entrants"]["candidate"]["status"] == "job_failed"
    assert block["decisions"][0]["outcome"] == "inconclusive"
    assert result["entrants"]["candidate"]["status"] == "ok"      # the standard tasks are unaffected


def test_duplicate_task_ids_are_refused(work_dir, template_dir):
    task = validated_task(work_dir, "markov", MARKOV, [MARKOV_RULE])
    with pytest.raises(ValueError, match="distinct"):
        evaluate_candidate(template_dir, TINY, work_dir / "run", LocalExecutor("t", work_dir / "r"),
                           baselines=["ngram"], falsifiers=[task.directory, task.directory], log=lambda _: None)


# ----- through Podman ----------------------------------------------------------------------------------

def _labeled(run_id):
    from hybrid.executor.podman import Podman
    found = []
    for argv in (["ps", "-a", "--format", "{{.Names}}"], ["volume", "ls", "--format", "{{.Name}}"]):
        found += Podman().run([*argv, "--filter", "label=bigbrane.owner=bigbrane",
                               "--filter", "label=bigbrane.run=" + run_id]).stdout.split()
    return found


@podman
def test_validation_and_falsifier_jobs_through_podman(work_dir, template_dir):
    from hybrid.executor.executor import Executor
    run_id = "falsifier-test-" + STAMP.lower()
    executor = Executor(run_id, work_dir / "receipts")
    image = read_lock()["reference"]
    report, out = F.run_validation_job(executor, image, MARKOV, 0, [MARKOV_RULE], work_dir / "jobs", CAPS, 40)
    assert report["ok"] and report["deterministic"], report
    task = F.write_task(work_dir / "tasks" / "markov", out, design("markov", MARKOV, [MARKOV_RULE]), report)
    for case in ("network", "file write", "file read outside"):
        bad, _ = F.run_validation_job(executor, image, FAILURES[case][0], 0, [SHIFT_RULE], work_dir / "jobs", CAPS, 40)
        assert not bad["ok"] and any("escape" in p for p in bad["problems"]), (case, bad)
    result = evaluate_candidate(template_dir, TINY, work_dir / "run", executor, baselines=["ngram"],
                                cache_dir=work_dir / "cache", falsifiers=[task.directory], log=lambda _: None)
    assert result["trusted"] is True
    check_falsifier_result(result, [task], ["candidate", "ngram"], work_dir / "run" / "bench")
    assert result["falsifiers"]["markov"]["decisions"][0]["outcome"] == "supported"
    assert _labeled(run_id) == []
