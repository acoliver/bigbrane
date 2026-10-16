"""Mechanism fingerprint: unit checks on synthetic numbers, whole-evaluator checks on synthetic
predictions (flags trigger exactly when they should), the linear-attention baseline, and the
GRU-submitted-as-candidate case through the local and Podman executors."""
from __future__ import annotations

import json
import math
import shutil
import sys
import time

import numpy as np
import pytest
import torch

from hybrid.bench import config as bench_config, evaluator, fingerprint, scaling
from hybrid.bench.baselines import HERE as BASELINE_DIR, stage_baseline
from hybrid.bench.evaluate import EVALUATOR_CODE, build_data, evaluate_candidate
from hybrid.bench.local import LocalExecutor, output_hashes

from .conftest import ROOT, STAMP, TINY, podman
from .test_evaluator import run_eval, synthetic_scaling

sys.path.insert(0, str(BASELINE_DIR))
import bb_neural  # noqa: E402

CFG = bench_config.load_config(TINY)["fingerprint"]


# ---------------------------------------------------------------- divergences and the capacity fit

def test_js_is_zero_for_equal_ln2_for_disjoint_and_symmetric():
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    q = np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    js = fingerprint.js_rows(p, q)
    assert js[0] == pytest.approx(math.log(2)) and js[1] == 0.0
    np.testing.assert_allclose(fingerprint.js_rows(p, q), fingerprint.js_rows(q, p))


def test_half_capacity_k_interpolates_in_log_k():
    assert fingerprint.half_capacity_k([2, 4, 8, 16], [0.8, 0.6, 0.4, 0.2]) == pytest.approx(8.0)
    # half of 1.0 is 0.5; accuracy 0.8 at K=8 and 0.3 at K=32 puts the crossing at 8 * 4**0.6
    assert fingerprint.half_capacity_k([2, 8, 32], [1.0, 0.8, 0.3]) == pytest.approx(8 * 4 ** 0.6)
    assert fingerprint.half_capacity_k([2, 8, 32], [0.8, 0.7, 0.6]) is None   # no knee in range


def recall_spec(chance=0.0625):
    return {"tasks": {"recall": {"train_sizes": [32], "group_labels": [2, 8, 32], "chance_acc": chance,
                                 "value_bits": 4.0}}}


def recall_entry(accs, state_floats):
    return {"tasks": {"recall": {"sizes": {"32": {"valid": True, "groups": {
        str(k): {"acc": a, "n": 100} for k, a in zip((2, 8, 32), accs)}}}}},
        "describe": {"recall": {"32": {"state_floats_per_sequence": state_floats}}}}


def test_recurrent_state_shows_a_capacity_knee_and_attention_does_not():
    fixed = fingerprint.recall_capacity(recall_spec(), recall_entry([0.8, 0.4, 0.1], 128), CFG)
    assert fixed["status"] == "knee" and fixed["k_half"] == pytest.approx(8.0)
    assert fixed["state_floats_per_pair"] == pytest.approx(16.0)
    assert fixed["capacity_per_state_float"] == pytest.approx(8 / 128)
    assert fixed["value_bits_per_state_float"] == pytest.approx(8 * 4 / 128)
    attention = fingerprint.recall_capacity(recall_spec(), recall_entry([0.8, 0.78, 0.75], 65536), CFG)
    assert attention["status"] == "no_knee_in_range" and attention["pairs"] == 32 and attention["pairs_is_lower_bound"]
    unlearned = fingerprint.recall_capacity(recall_spec(), recall_entry([0.07, 0.06, 0.06], 128), CFG)
    assert unlearned["status"] == "not_learned" and unlearned["capacity_per_state_float"] == 0.0


def test_fewshot_slope_is_accuracy_gained_per_doubling_of_shots():
    spec = {"tasks": {"fewshot": {"train_sizes": [32], "group_labels": [2, 4, 8], "chance_acc": 0.077}}}

    def entry(accs):
        return {"tasks": {"fewshot": {"sizes": {"32": {"valid": True, "groups": {
            str(s): {"acc": a, "n": 100} for s, a in zip((2, 4, 8), accs)}}}}}}
    rising = fingerprint.fewshot_block(spec, entry([0.1, 0.2, 0.3]))
    assert rising["slope_per_doubling"] == pytest.approx(0.1)
    # binomial noise: weights -0.5, 0, 0.5 on accuracies 0.1, 0.2, 0.3 with 100 prefixes each
    assert rising["slope_se"] == pytest.approx(math.sqrt(0.25 * 0.09 / 100 + 0.25 * 0.21 / 100))
    assert fingerprint.fewshot_block(spec, entry([0.1, 0.1, 0.1]))["slope_per_doubling"] == pytest.approx(0.0, abs=1e-12)


def test_memory_growth_is_flat_below_the_floor_or_the_slope_and_growing_otherwise():
    def block(peak_per_n, base=0, missing=False):
        raw = {"batch": 1, "repeats": 1, "stopped": None, "rss_method": "synthetic",
               "lengths": [{"n": n, "times_s": [1e-3 * n],
                            "peak_rss_delta_bytes": None if missing else base + int(peak_per_n * n)}
                           for n in (64, 128, 256, 512, 1024)]}
        return fingerprint.scaling_block({"scaling": scaling.summarize(raw, 1.0, 0.3, 4)}, CFG)
    assert block(40_000)["memory_growth"] == "growing"           # 41 MB at n=1024, slope 1
    assert block(100)["memory_growth"] == "flat"                 # 0.1 MB at n=1024: under the floor
    assert block(0, base=40 << 20)["memory_growth"] == "flat"    # large but constant: slope 0
    assert block(0, missing=True)["memory_growth"] == "unknown"


# ---------------------------------------------------------------- the whole evaluator on synthetic predictions

def answer_preds(oracle, group, vocab, accs):
    """Peaked predictions whose top-1 accuracy per group is exactly accs[g] (first rows of each group are right)."""
    pred = np.full((len(oracle), vocab), 0.1 / (vocab - 1))
    target = (oracle + 1) % vocab
    for g, acc in enumerate(accs):
        rows = np.flatnonzero(group == g)
        right = rows[:int(round(acc * len(rows)))]
        target[right] = oracle[right]
    pred[np.arange(len(oracle)), target] = 0.9
    return pred


def dist_preds(oracle, noise, seed):
    mix = np.random.default_rng(seed).dirichlet(np.ones(oracle.shape[1]), size=len(oracle))
    return (1 - noise) * oracle + noise * mix


def write_synthetic(root, name, eval_tasks, oracle_dir, recipe):
    """An entrant whose predictions are built from the oracle by `recipe`:
    noise/seed for distribution tasks, recall/fewshot accuracy per group for answer tasks."""
    out = root / "entrants" / name / "out"
    out.mkdir(parents=True)
    describe = {}
    for task, spec in eval_tasks.items():
        with np.load(oracle_dir / ("%s.npz" % task)) as data:
            oracle, group = data["oracle"], data["group"]
        V = spec["vocab_size"]
        if spec["oracle_kind"] == "dist":
            pred = dist_preds(oracle, recipe["noise"], recipe["seed"])
        else:
            pred = answer_preds(oracle, group, V, recipe[task])
        (out / "pred" / task).mkdir(parents=True)
        for n in spec["train_sizes"]:
            np.save(out / "pred" / task / ("n%d.npy" % n), pred)
        for factor in spec["length_probes"]:
            with np.load(oracle_dir / ("%s__x%s.npz" % (task, factor))) as data:
                longer = dist_preds(data["oracle"], recipe["noise"] + 0.02 * int(factor), recipe["seed"])
            np.save(out / "pred" / task / ("n%d_x%s.npy" % (spec["train_sizes"][-1], factor)), longer)
        describe[task] = {str(n): {"params": 1, "state_floats_per_sequence": recipe["state"], "notes": name}
                          for n in spec["train_sizes"]}
    (out / "describe.json").write_text(json.dumps(describe))
    (out / "scaling.json").write_text(json.dumps(synthetic_scaling(recipe.get("exponent", 1.0))))
    receipt = {"job_id": "job-" + name, "status": "ok", "detail": "", "outputs": output_hashes(out)}
    (root / "entrants" / name / "receipt.json").write_text(json.dumps(receipt))


# tiny.toml: recall has K in (2, 8), 8 values (chance 0.125), 32 prefixes per K; fewshot has shots (2, 4), 32 per group
GRU = {"noise": 0.05, "seed": 1, "state": 100, "recall": [0.75, 0.25], "fewshot": [0.125, 0.25]}
SSM = {"noise": 0.15, "seed": 2, "state": 400, "recall": [0.75, 0.5], "fewshot": [0.125, 0.125]}
TRANSFORMER = {"noise": 0.10, "seed": 3, "state": 5000, "recall": [0.875, 0.875], "fewshot": [0.125, 0.5],
               "exponent": 2.0}
BASELINES = {"gru": (GRU, "fixed"), "ssm": (SSM, "fixed"), "transformer": (TRANSFORMER, "growing")}
# Different mechanism: exact distributions, recall without a knee, in-context gain, linear time.
NOVEL = {"noise": 0.0, "seed": 9, "state": 100, "recall": [0.875, 0.875], "fewshot": [0.125, 0.75]}


def evaluate_synthetic(work_dir, candidate):
    cfg = bench_config.load_config(TINY)
    _, oracle_dir, eval_tasks = build_data(cfg, work_dir / "data")
    eval_in = work_dir / "eval_in"
    (eval_in / "code" / "hybrid" / "bench").mkdir(parents=True)
    for file in EVALUATOR_CODE:
        shutil.copyfile(ROOT / "hybrid" / "bench" / file, eval_in / "code" / "hybrid" / "bench" / file)
    shutil.copytree(oracle_dir, eval_in / "oracle", ignore=shutil.ignore_patterns("*.meta.json"))
    entrants = {"candidate": {"role": "candidate", "claimed_exponent": None}}
    write_synthetic(eval_in, "candidate", eval_tasks, oracle_dir, candidate)
    for name, (recipe, state) in BASELINES.items():
        write_synthetic(eval_in, name, eval_tasks, oracle_dir, recipe)
        entrants[name] = {"role": "baseline", "claimed_exponent": 1.0, "state": state}
    (eval_in / "eval_spec.json").write_text(json.dumps({
        "tasks": eval_tasks, "task_config_hash": "x", "config_hash": "y", "entrants": entrants, "fingerprint": cfg["fingerprint"],
        "scaling": {"tolerance": 0.3, "tail_points": 3}}))
    return run_eval(eval_in, work_dir / "eval_out")


def test_baseline_fingerprints_follow_their_recipes(work_dir):
    result = evaluate_synthetic(work_dir, NOVEL)
    gru, ssm = (result["entrants"][n]["fingerprint"] for n in ("gru", "ssm"))
    assert gru["recall_capacity"]["status"] == "knee"
    assert gru["recall_capacity"]["k_half"] == pytest.approx(2 * 4 ** 0.75)    # 0.75 -> 0.25 puts half (0.375) 75% of the way
    assert gru["recall_capacity"]["state_floats"] == 100
    assert ssm["recall_capacity"]["status"] == "no_knee_in_range"
    assert gru["fewshot"]["slope_per_doubling"] == pytest.approx(0.125)
    assert ssm["fewshot"]["slope_per_doubling"] == pytest.approx(0.0)
    assert "no_fewshot_gain" in ssm["flags"] and "no_fewshot_gain" not in gru["flags"]
    assert "quadratic_or_worse" not in gru["flags"]
    assert result["entrants"]["transformer"]["fingerprint"]["scaling"]["time_slope_tail"] == pytest.approx(2.0)
    assert "quadratic_or_worse" in result["entrants"]["transformer"]["fingerprint"]["flags"]
    assert "candidate" in result["fingerprint"] and result["fingerprint"]["config"] == CFG
    for baseline in ("gru", "ssm", "transformer"):
        assert "behaves_like:" + baseline not in result["entrants"][baseline]["fingerprint"]["flags"]
        assert baseline not in result["entrants"][baseline]["fingerprint"]["similarity"]   # never compared with itself


def test_length_generalization_reports_kl_at_two_and_four_times_the_training_length(work_dir):
    result = evaluate_synthetic(work_dir, NOVEL)
    for task in ("hmm", "vom", "compose"):
        assert set(result["tasks"][task]["length_probes"]) == {"2", "4"}
        block = result["entrants"]["gru"]["fingerprint"]["length_generalization"][task]
        assert block["status"] == "ok" and set(block["factors"]) == {"2", "4"}
        for factor in ("2", "4"):
            cell = block["factors"][factor]
            assert cell["valid"] and cell["kl_delta"] == pytest.approx(cell["kl"] - block["base"]["kl"])
        # synthetic predictions get 0.02 noisier per factor, so KL rises with length
        assert block["factors"]["4"]["kl"] > block["factors"]["2"]["kl"]
    assert "recall" not in result["entrants"]["gru"]["fingerprint"]["length_generalization"]


def test_a_copy_of_a_baseline_is_flagged_behaves_like_it(work_dir):
    result = evaluate_synthetic(work_dir, dict(GRU))
    fp = result["entrants"]["candidate"]["fingerprint"]
    assert "behaves_like:gru" in fp["flags"]
    assert fp["nearest_baseline"]["name"] == "gru" and fp["nearest_baseline"]["mean_js"] == 0.0
    assert fp["similarity"]["gru"]["close_tasks"] == 5
    assert "no_capacity_advantage" in fp["flags"]   # same capacity per state float as the best fixed-state baseline
    assert result["fingerprint"]["candidate"]["flags"] == fp["flags"]
    assert result["fingerprint"]["candidate"]["nearest_baseline"]["name"] == "gru"


def test_a_different_mechanism_raises_no_flags(work_dir):
    result = evaluate_synthetic(work_dir, NOVEL)
    fp = result["entrants"]["candidate"]["fingerprint"]
    assert fp["flags"] == []
    # exact distributions are far from every noisy baseline once their own distance to the oracle is factored in
    for baseline in ("gru", "ssm", "transformer"):
        assert fp["similarity"][baseline]["close_tasks"] <= 1
    assert fp["recall_capacity"]["status"] == "no_knee_in_range"   # capacity is a lower bound: K=8 at 100 floats


@pytest.mark.parametrize("change,flag", [
    ({"exponent": 2.0}, "quadratic_or_worse"),
    ({"fewshot": [0.25, 0.25]}, "no_fewshot_gain"),
    ({"recall": [0.875, 0.25]}, "no_capacity_advantage"),          # knee at K=5.3 with 100 floats: worse than the GRU
    ({"recall": [0.125, 0.125]}, "no_capacity_advantage"),         # never learned recall
])
def test_each_flag_triggers_on_its_own(work_dir, change, flag):
    result = evaluate_synthetic(work_dir, {**NOVEL, **change})
    assert result["entrants"]["candidate"]["fingerprint"]["flags"] == [flag]


def test_capacity_flag_depends_on_capacity_per_state_float(work_dir):
    knee_recipe = {**NOVEL, "recall": [0.875, 0.25]}
    smaller = evaluate_synthetic(work_dir, {**knee_recipe, "state": 50})
    cap = smaller["entrants"]["candidate"]["fingerprint"]
    assert "no_capacity_advantage" not in cap["flags"]   # same knee as 100 floats, half the state: better per float
    assert cap["flag_details"]["capacity"]["best_fixed_baseline"] == "gru"


def test_growing_state_baselines_do_not_set_the_capacity_bar(work_dir):
    result = evaluate_synthetic(work_dir, NOVEL)
    details = result["entrants"]["candidate"]["fingerprint"]["flag_details"]["capacity"]
    assert details["best_fixed_baseline"] in ("gru", "ssm")   # never the transformer, whose KV cache grows


def test_similarity_normalizes_by_distance_to_the_oracle():
    """Two models equally near the oracle in unrelated directions are not 'the same' however small their JS is."""
    rng = np.random.default_rng(0)
    oracle = rng.dirichlet(np.ones(6), size=400)
    a = 0.98 * oracle + 0.02 * np.eye(6)[rng.integers(0, 6, 400)]   # each model strays toward its own random token
    b = 0.98 * oracle + 0.02 * np.eye(6)[rng.integers(0, 6, 400)]
    d = lambda x, y: math.sqrt(float(fingerprint.js_rows(x, y).mean()))
    unrelated = fingerprint.pair_similarity(d(a, oracle), a, b, d(b, oracle), CFG)
    assert unrelated["js"] < CFG["js_close"] and unrelated["ratio"] > CFG["ratio_close"] and not unrelated["close"]
    assert 0 <= unrelated["ratio"] <= 1
    same = fingerprint.pair_similarity(d(a, oracle), a, a, d(a, oracle), CFG)
    assert same["js"] == 0.0 and same["ratio"] == 0.0 and same["close"]
    perfect = fingerprint.pair_similarity(0.0, oracle, oracle, 0.0, CFG)
    assert perfect["ratio"] is None and not perfect["close"]   # indistinguishable from the oracle, so no evidence either way


# ---------------------------------------------------------------- length probe data

def test_length_probe_sets_are_fresh_and_beyond_the_training_length(work_dir):
    cfg = bench_config.load_config(TINY)
    inputs, oracle_dir, eval_tasks = build_data(cfg, work_dir / "data")
    for task, base_len in (("hmm", 32), ("vom", 48), ("compose", 40)):
        base = json.loads((inputs / "data" / task / "prefixes.json").read_text())["prefixes"]
        assert set(eval_tasks[task]["length_probes"]) == {"2", "4"}
        for factor in (2, 4):
            longer = json.loads((inputs / "data" / task / ("prefixes_x%d.json" % factor)).read_text())["prefixes"]
            lengths = [len(p) for p in longer]
            assert min(lengths) >= base_len * factor // 2 and max(lengths) < base_len * factor
            info = eval_tasks[task]["length_probes"][str(factor)]
            assert info["length"] == base_len * factor and info["n_prefixes"] == len(longer) == len(base)
            # the same stream would make the first longer sequence start with the first base sequence
            assert len(base[0]) < 12 or longer[0][:len(base[0])] != base[0]
            with np.load(oracle_dir / ("%s__x%d.npz" % (task, factor))) as data:
                assert data["oracle"].shape == (len(longer), eval_tasks[task]["vocab_size"])
    assert eval_tasks["recall"]["length_probes"] == {} and eval_tasks["recall"]["chance_acc"] == 1 / 8


def test_fingerprint_thresholds_are_validated_and_recorded():
    base = json.loads(json.dumps(bench_config.load_config(TINY)))
    base["fingerprint"] = {"js_close": 0.05}
    cfg = bench_config.load_config(base)
    assert cfg["fingerprint"]["js_close"] == 0.05 and cfg["fingerprint"]["quadratic_slope"] == 1.5
    for bad in ({"nope": 1}, {"length_factors": [1]}, {"length_tasks": ["recall"]}):
        broken = json.loads(json.dumps(base))
        broken["fingerprint"] = bad
        with pytest.raises(ValueError):
            bench_config.load_config(broken)
    changed = json.loads(json.dumps(base))
    changed["fingerprint"] = {"length_factors": [3]}
    assert bench_config.task_config_hash(bench_config.load_config(changed)) != bench_config.task_config_hash(cfg)
    thresholds_only = json.loads(json.dumps(base))
    thresholds_only["fingerprint"] = {"js_close": 0.5}
    assert bench_config.task_config_hash(bench_config.load_config(thresholds_only)) == bench_config.task_config_hash(cfg)


# ---------------------------------------------------------------- the linear-attention baseline

def test_linear_attention_matches_the_quadratic_reference_across_chunk_boundaries():
    torch.manual_seed(0)
    att = bb_neural.CausalLinearAttention(16).eval()
    for n in (1, 63, 64, 65, 150):
        x = torch.randn(2, n, 16)
        q, k, v = att.qkv(x).chunk(3, dim=-1)
        q, k = torch.nn.functional.elu(q) + 1, torch.nn.functional.elu(k) + 1
        scores = (q @ k.transpose(1, 2)).tril()
        reference = att.proj((scores @ v) / (scores.sum(-1, keepdim=True) + 1e-6))
        torch.testing.assert_close(att(x), reference, atol=1e-5, rtol=1e-4)


def test_linear_attention_state_is_a_fixed_d_by_d_matrix():
    """Reading the sequence one token at a time with only (S: d x d, z: d) reproduces the full forward pass."""
    torch.manual_seed(1)
    d = 16
    att = bb_neural.CausalLinearAttention(d).eval()
    x = torch.randn(1, 200, d)
    S, z = torch.zeros(d, d), torch.zeros(d)
    steps = []
    with torch.no_grad():
        q, k, v = att.qkv(x)[0].chunk(3, dim=-1)
        q, k = torch.nn.functional.elu(q) + 1, torch.nn.functional.elu(k) + 1
        for t in range(x.shape[1]):
            S, z = S + torch.outer(k[t], v[t]), z + k[t]
            steps.append((q[t] @ S) / (q[t] @ z + 1e-6))
        recurrent = att.proj(torch.stack(steps))
        torch.testing.assert_close(att(x)[0], recurrent, atol=1e-5, rtol=1e-4)
    assert S.shape == (d, d) and att.state_floats() == d * d + d
    net = bb_neural.LinearAttentionNet(8)
    assert net.state_floats() == 2 * (64 * 64 + 64)


def test_linear_attention_time_grows_about_linearly_with_context():
    net = bb_neural.LinearAttentionNet(8).eval()
    lengths = [1024, 2048, 4096, 8192]
    times = []
    with torch.no_grad():
        for n in lengths:
            x = torch.randint(0, 8, (4, n))
            net(x)
            runs = []
            for _ in range(7):
                start = time.perf_counter()
                net(x)
                runs.append(time.perf_counter() - start)
            times.append(min(runs))
    slope = scaling.fit_slope(lengths, times)["slope"]
    assert 0.6 < slope < 1.45, (slope, times)   # a quadratic model would read about 2


# ---------------------------------------------------------------- real code through executors

def candidate_result_checks(result):
    fp = result["entrants"]["candidate"]["fingerprint"]
    assert "behaves_like:gru" in fp["flags"], fp["similarity"]
    assert fp["nearest_baseline"]["name"] == "gru" and fp["nearest_baseline"]["mean_js"] < 1e-9
    assert fp["similarity"]["gru"]["close_tasks"] == fp["similarity"]["gru"]["compared_tasks"] == 5
    assert result["fingerprint"]["candidate"]["flags"] == fp["flags"]
    assert "behaves_like:gru" not in result["entrants"]["gru"]["fingerprint"]["flags"]
    for task, spec in result["tasks"].items():
        for factor in spec["length_probes"]:
            assert result["entrants"]["candidate"]["tasks"][task]["length_probes"][factor]["valid"]


def test_the_gru_submitted_as_a_candidate_is_flagged_behaves_like_gru(work_dir):
    candidate = stage_baseline("gru", work_dir / "gru_candidate")
    executor = LocalExecutor("fp-local", work_dir / "receipts")
    result = evaluate_candidate(candidate, TINY, work_dir / "run", executor, baselines=["ngram", "gru", "linear_attention"],
                                cache_dir=work_dir / "cache", log=lambda _: None)
    candidate_result_checks(result)
    assert result["entrants"]["linear_attention"]["fingerprint"]["recall_capacity"]["state_floats"] == 8320


@podman
def test_the_gru_submitted_as_a_candidate_is_flagged_through_podman(work_dir):
    from hybrid.executor.executor import Executor
    from .test_e2e_podman import labeled
    run_id = "bench-test-" + STAMP.lower() + "-fp"
    executor = Executor(run_id, work_dir / "receipts")
    candidate = stage_baseline("gru", work_dir / "gru_candidate")
    result = evaluate_candidate(candidate, TINY, work_dir / "run", executor, baselines=["ngram", "gru", "linear_attention"],
                                cache_dir=work_dir / "cache", log=lambda _: None)
    assert result["trusted"] is True
    candidate_result_checks(result)
    assert result["entrants"]["linear_attention"]["scaling"]["rss_method"] == "linux_fork_vmhwm"
    assert labeled("containers", run_id) == [] and labeled("volumes", run_id) == []
