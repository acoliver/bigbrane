"""The evaluator reproduces known values: a perfect-oracle candidate scores KL 0 and accuracy 1,
a uniform candidate scores the closed-form KL and accuracy. Integrity and validity checks."""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pytest

from hybrid.bench import config as bench_config, evaluator, metrics
from hybrid.bench.evaluate import EVALUATOR_CODE, build_data
from hybrid.bench.local import output_hashes

from .conftest import TINY


def synthetic_scaling(exponent=1.0):
    return {"batch": 2, "repeats": 3, "rss_method": "synthetic", "stopped": None, "warmup": 1,
            "lengths": [{"n": n, "times_s": [1e-6 * n ** exponent] * 3, "peak_rss_delta_bytes": 1000 * n}
                        for n in (64, 128, 256, 512)]}


def write_entrant(root, name, eval_tasks, oracle_dir, mode, status="ok"):
    out = root / "entrants" / name / "out"
    out.mkdir(parents=True)
    for task, spec in eval_tasks.items():
        with np.load(oracle_dir / ("%s.npz" % task)) as data:
            oracle = data["oracle"]
        V = spec["vocab_size"]
        if mode == "perfect":
            pred = oracle if spec["oracle_kind"] == "dist" else np.eye(V)[oracle]
        else:
            pred = np.full((spec["n_prefixes"], V), 1.0 / V)
        (out / "pred" / task).mkdir(parents=True)
        for n in spec["train_sizes"]:
            np.save(out / "pred" / task / ("n%d.npy" % n), pred)
        for factor in spec["length_probes"]:
            with np.load(oracle_dir / ("%s__x%s.npz" % (task, factor))) as data:
                longer = data["oracle"]
            np.save(out / "pred" / task / ("n%d_x%s.npy" % (spec["train_sizes"][-1], factor)),
                    longer if mode == "perfect" else np.full(longer.shape, 1.0 / V))
    (out / "describe.json").write_text(json.dumps({
        task: {str(n): {"params": 1, "state_floats_per_sequence": 1, "notes": mode} for n in spec["train_sizes"]}
        for task, spec in eval_tasks.items()}))
    (out / "scaling.json").write_text(json.dumps(synthetic_scaling()))
    receipt = {"job_id": "job-" + name, "status": status, "detail": "", "outputs": output_hashes(out)}
    (root / "entrants" / name / "receipt.json").write_text(json.dumps(receipt))
    return out


@pytest.fixture
def staged(work_dir):
    cfg = bench_config.load_config(TINY)
    _, oracle_dir, eval_tasks = build_data(cfg, work_dir / "data")
    eval_in = work_dir / "eval_in"
    (eval_in / "code" / "hybrid" / "bench").mkdir(parents=True)
    for file in EVALUATOR_CODE:
        shutil.copyfile(metrics.__file__.replace("metrics.py", file), eval_in / "code" / "hybrid" / "bench" / file)
    shutil.copytree(oracle_dir, eval_in / "oracle", ignore=shutil.ignore_patterns("*.meta.json"))
    write_entrant(eval_in, "perfect", eval_tasks, oracle_dir, "perfect")
    write_entrant(eval_in, "uniform", eval_tasks, oracle_dir, "uniform")
    spec = {"tasks": eval_tasks, "task_config_hash": "x", "config_hash": "y",
            "fingerprint": bench_config.load_config(TINY)["fingerprint"],
            "entrants": {"perfect": {"role": "candidate", "claimed_exponent": 1.0},
                         "uniform": {"role": "baseline", "claimed_exponent": 2.0, "state": "fixed"}},
            "scaling": {"tolerance": 0.3, "tail_points": 3}}
    (eval_in / "eval_spec.json").write_text(json.dumps(spec))
    return eval_in, oracle_dir, eval_tasks, work_dir / "eval_out"


def run_eval(eval_in, eval_out):
    eval_out.mkdir(exist_ok=True)
    evaluator.main(eval_in, eval_out)
    return json.loads((eval_out / "bench_result.json").read_text())


def test_perfect_candidate_scores_kl_zero_and_full_accuracy(staged):
    eval_in, _, eval_tasks, eval_out = staged
    result = run_eval(eval_in, eval_out)
    perfect = result["entrants"]["perfect"]
    assert perfect["status"] == "ok"
    for task, spec in eval_tasks.items():
        for n in spec["train_sizes"]:
            m = perfect["tasks"][task]["sizes"][str(n)]
            assert m["valid"] and m["kl"] == 0.0 and m["acc"] == 1.0, (task, n, m)
            for g in m.get("groups", {}).values():
                assert g["kl"] == 0.0 and g["acc"] == 1.0
    assert perfect["tasks"]["compose"]["sizes"]["32"]["close_acc"] == 1.0
    assert perfect["tasks"]["hmm"]["tokens_to_target"]["first_tokens"] == eval_tasks["hmm"]["train_tokens"]["8"]


def test_uniform_candidate_scores_closed_form_values(staged):
    eval_in, oracle_dir, eval_tasks, eval_out = staged
    result = run_eval(eval_in, eval_out)
    uniform = result["entrants"]["uniform"]
    for task, spec in eval_tasks.items():
        V = spec["vocab_size"]
        with np.load(oracle_dir / ("%s.npz" % task)) as data:
            oracle = data["oracle"]
        if spec["oracle_kind"] == "dist":
            entropy = -np.where(oracle > 0, oracle * np.log(np.where(oracle > 0, oracle, 1)), 0).sum(1)
            kl = float(np.mean(math.log(V) - entropy))
            acc = float(np.mean((oracle >= oracle.max(1, keepdims=True) - 1e-12).sum(1) / V))
        else:
            kl, acc = math.log(V), 1.0 / V
        m = uniform["tasks"][task]["sizes"]["32"]
        assert m["kl"] == pytest.approx(kl, rel=1e-12, abs=1e-15), task
        assert m["acc"] == pytest.approx(acc, rel=1e-12), task
    close = uniform["tasks"]["compose"]["sizes"]["32"]["close_acc"]
    assert close == pytest.approx(1.0 / 2)   # two bracket types in the tiny config
    comp = result["comparison"]["perfect"]["uniform"]["hmm"]["sizes"]["32"]
    assert comp["kl_delta"] < 0 and comp["acc_delta"] > 0


def test_scaling_verdicts_from_raw_timings(staged):
    eval_in, _, _, eval_out = staged
    result = run_eval(eval_in, eval_out)
    perfect, uniform = result["entrants"]["perfect"]["scaling"], result["entrants"]["uniform"]["scaling"]
    assert perfect["time_fit_tail"]["slope"] == pytest.approx(1.0, abs=1e-9) and perfect["verdict"] == "match"
    assert uniform["verdict"] == "mismatch"   # timings are O(n), claim is 2
    assert perfect["memory_fit_tail"]["slope"] == pytest.approx(1.0, abs=1e-9)


def test_tampered_prediction_is_rejected(staged):
    eval_in, _, _, eval_out = staged
    path = eval_in / "entrants" / "uniform" / "out" / "pred" / "hmm" / "n8.npy"
    arr = np.load(path)
    arr[0, 0] *= 0.5
    np.save(path, arr)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        run_eval(eval_in, eval_out)


def test_invalid_predictions_get_no_metrics(staged):
    eval_in, _, eval_tasks, eval_out = staged
    out = eval_in / "entrants" / "uniform" / "out"
    bad = {"hmm": np.full((eval_tasks["hmm"]["n_prefixes"], eval_tasks["hmm"]["vocab_size"]), 0.5),
           "vom": np.full((3, eval_tasks["vom"]["vocab_size"]), 0.25)}
    for task, arr in bad.items():
        np.save(out / "pred" / task / "n8.npy", arr)
    nan = np.load(out / "pred" / "recall" / "n8.npy")
    nan[0, 0] = np.nan
    np.save(out / "pred" / "recall" / "n8.npy", nan)
    receipt = json.loads((eval_in / "entrants" / "uniform" / "receipt.json").read_text())
    receipt["outputs"] = output_hashes(out)
    (eval_in / "entrants" / "uniform" / "receipt.json").write_text(json.dumps(receipt))
    result = run_eval(eval_in, eval_out)
    sizes = {t: result["entrants"]["uniform"]["tasks"][t]["sizes"]["8"] for t in ("hmm", "vom", "recall")}
    assert not any(m["valid"] for m in sizes.values())
    assert "row sums" in sizes["hmm"]["problems"][0]
    assert "shape" in sizes["vom"]["problems"][0]
    assert "non-finite" in sizes["recall"]["problems"][0]
    assert result["entrants"]["uniform"]["tasks"]["hmm"]["sizes"]["32"]["valid"]


def test_failed_job_entrant_has_no_metrics(staged):
    eval_in, _, _, eval_out = staged
    receipt_path = eval_in / "entrants" / "uniform" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "timeout"
    receipt_path.write_text(json.dumps(receipt))
    result = run_eval(eval_in, eval_out)
    assert result["entrants"]["uniform"]["status"] == "job_failed"
    assert "tasks" not in result["entrants"]["uniform"]
    assert result["comparison"]["perfect"] == {}


def test_tokens_to_target():
    tokens = [100, 400, 1600]
    assert metrics.tokens_to_target(tokens, [0.5, 0.3, 0.1], "<=", 0.6)["interp_tokens"] == 100
    hit = metrics.tokens_to_target(tokens, [0.5, 0.3, 0.1], "<=", 0.2)
    assert hit["first_tokens"] == 1600 and hit["interp_tokens"] == pytest.approx(800.0)
    assert not metrics.tokens_to_target(tokens, [0.1, 0.2, None], ">=", 0.5)["reached"]


def test_metric_floor_keeps_zero_probability_finite():
    q = np.array([[1.0, 0.0]])
    assert metrics.ce_rows(np.array([1]), q)[0] == pytest.approx(-math.log(metrics.FLOOR))
    assert metrics.kl_rows(np.array([[0.5, 0.5]]), q)[0] == pytest.approx(0.5 * math.log(0.5 / metrics.FLOOR) +
                                                                          0.5 * math.log(0.5))
