"""Ladder slices, run ids and execution modes (hybrid/bench/ladder.py, ladder_host.py, ladder_runner.py).

The host_mps mode trains the trusted rung 1 baselines on the Mac's GPU. These tests pin down that the first
slice's run-id formula is unchanged, that the diagnostic slice is the one summary.md recommends, that a host job
runs only the trusted files with no network, and that the runner's device option leaves CPU training as it was.
Tests that need Apple MPS or sandbox-exec skip elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hybrid.bench import ladder, ladder_host

CONFIG = ladder.BENCH_DIR / "configs" / "ladder.toml"
mps = pytest.mark.skipif(not (torch.backends.mps.is_available() and Path(ladder_host.SANDBOX_EXEC).exists()),
                         reason="needs Apple MPS and sandbox-exec")


@pytest.fixture(scope="module")
def cfg():
    return ladder.load_config(CONFIG)


def key(task="mqar256", model="gdn-d64", lr=0.001, seed=0, steps=18000):
    return ladder.run_key(task, model, lr, seed, steps)


# ------------------------------------------------------------------------------------------------ slices and ids

def test_first_slice_settings_are_the_plain_config(cfg):
    assert ladder.slice_settings(cfg, "first") == {"execution": "container", "train": cfg["train"],
                                                  "caps": cfg["caps"], "seeds": [0, 1, 2, 3]}


def test_a_container_slice_without_overrides_keeps_the_first_slice_run_id_formula(cfg):
    k = key(steps=6000)
    legacy = json.dumps({"key": k, "task": cfg["tasks"]["mqar256"], "model": cfg["models"]["gdn-d64"],
                         "train": cfg["train"], "test_seed": cfg["test_seed"], "code": ladder.code_hash()},
                        sort_keys=True)
    expected = "r-" + hashlib.sha256(legacy.encode()).hexdigest()[:12]
    assert ladder.run_id(cfg, k, ladder.slice_settings(cfg, "first")) == expected


def test_run_ids_separate_execution_modes_and_training_overrides(cfg):
    first = ladder.slice_settings(cfg, "first")
    on_mps = dict(first, execution="host_mps")
    longer = dict(first, train={**first["train"], "time_budget_s": 15000})
    ids = {ladder.run_id(cfg, key(), s) for s in (first, on_mps, longer)}
    assert len(ids) == 3


def test_diag1_is_the_recommended_diagnostic_slice(cfg):
    s = ladder.slice_settings(cfg, "diag1")
    assert s["execution"] == "host_mps" and s["seeds"] == [0, 1, 2, 3]
    assert s["train"] == {**cfg["train"], "stop_val_acc": 1.01, "time_budget_s": 15000}
    assert s["caps"] == {**cfg["caps"], "wall_s": 18000}
    cells = {(c["task"], m): (c["lrs"], c["steps"]) for c in cfg["slices"]["diag1"]["cells"] for m in c["models"]}
    assert cells == {
        ("mqar256", "gdn-d64"): ([0.0001, 0.0003, 0.001], 18000),
        ("mqar256", "m2-s32-d64"): ([0.0001, 0.0003, 0.001], 18000),
        ("mqar512", "gdn-d32"): ([0.0003, 0.0006, 0.001], 18000),
        ("mqar512", "la-d32"): ([0.001, 0.003, 0.01], 18000),
        ("mqar512", "m2-s16-d32"): ([0.0001, 0.0003, 0.001], 18000),
        ("copy", "tf-d64"): ([0.0003, 0.001, 0.003], 18000),
        ("copy", "tf-halibi-d64"): ([0.0003, 0.001, 0.003], 18000)}


def _cells(cfg, slice_name):
    return {(c["task"], m): (c["lrs"], c["steps"]) for c in cfg["slices"][slice_name]["cells"] for m in c["models"]}


def test_diag2_reruns_diag1_on_the_new_recipe_with_same_device_attention_controls(cfg, work_dir):
    assert ladder.slice_settings(cfg, "diag2") == ladder.slice_settings(cfg, "diag1")
    cells = _cells(cfg, "diag2")
    diag1 = _cells(cfg, "diag1")
    assert {k: v for k, v in cells.items() if k in diag1} == diag1 and set(diag1) <= set(cells)
    assert {k: v for k, v in cells.items() if k not in diag1} == {
        ("mqar256", "tf-d64"): ([0.0003, 0.001, 0.003], 18000),
        ("mqar512", "tf-d32"): ([0.0003, 0.001, 0.003], 18000)}
    assert ladder.slice_workers(cfg, "diag2", None) == 3
    sched = ladder.Scheduler(cfg, None, work_dir, "diag2", 3, log=lambda m: None)
    assert len(sched.queue) == 27 and {k["seed"] for k in sched.queue} == {0}
    assert sum(len(c["seeds"]) for c in sched.pending.values()) == 27                    # 54 runs in all


DIAG3_REPLAYS = {"gdn-d32": ("mqar512", 0.001, 1), "gdn-d64": ("mqar256", 0.0003, 3)}


def test_diag3_replays_diag2_failures_and_sweeps_four_cells_on_diag2_settings(cfg, work_dir):
    settings = ladder.slice_settings(cfg, "diag3")
    assert settings == ladder.slice_settings(cfg, "diag2")
    assert settings["train"]["stop_val_acc"] == 1.01 and settings["train"]["time_budget_s"] == 15000
    assert settings["caps"]["wall_s"] == 18000 and ladder.slice_workers(cfg, "diag3", None) == 3
    cells = cfg["slices"]["diag3"]["cells"]
    assert [(c["task"], c["models"], c["lrs"], c.get("seeds"), c["steps"]) for c in cells] == [
        ("mqar512", ["gdn-d32"], [0.0001, 0.0003, 0.0006], None, 18000),
        ("mqar512", ["gdn-d32-replay1", "gdn-d32-replay2"], [0.001], [1], 18000),
        ("mqar256", ["gdn-d64"], [0.0001, 0.0002, 0.0003], None, 18000),
        ("mqar256", ["gdn-d64-replay1", "gdn-d64-replay2"], [0.0003], [3], 18000),
        ("mqar512", ["m2-s16-d32"], [0.0002, 0.0003, 0.0006], None, 18000),
        ("mqar512", ["la-d32"], [0.001, 0.003, 0.01], None, 18000)]
    sched = ladder.Scheduler(cfg, None, work_dir, "diag3", 3, log=lambda m: None)
    assert sched.queue[:3] == [key("mqar512", "gdn-d32", lr, 0) for lr in (0.0001, 0.0003, 0.0006)]
    assert len(sched.queue) == 16 and sum(len(c["seeds"]) for c in sched.pending.values()) == 12  # 28 runs in all
    assert {c: p["seeds"] for c, p in sched.pending.items()} == {
        ("mqar512", "gdn-d32"): [1, 2, 3], ("mqar256", "gdn-d64"): [1, 2, 3],
        ("mqar512", "m2-s16-d32"): [1, 2, 3], ("mqar512", "la-d32"): [1, 2, 3]}


@pytest.mark.parametrize("base", sorted(DIAG3_REPLAYS))
def test_diag3_replay_attempts_train_exactly_the_diag2_job_under_distinct_run_ids(cfg, base):
    task, lr, seed = DIAG3_REPLAYS[base]
    settings = ladder.slice_settings(cfg, "diag3")
    original = key(task, base, lr, seed)
    ids = {ladder.run_id(cfg, original, ladder.slice_settings(cfg, "diag2"))}
    for attempt in (1, 2):
        name = "%s-replay%d" % (base, attempt)
        assert cfg["models"][name] == {**cfg["models"][base], "attempt": attempt}
        replay = key(task, name, lr, seed)
        assert ladder.build_spec(cfg, replay, ["K4"], settings, "mps") == ladder.build_spec(
            cfg, original, ["K4"], ladder.slice_settings(cfg, "diag2"), "mps")
        ids.add(ladder.run_id(cfg, replay, settings))
    assert len(ids) == 3


DIAG4_REPEATS = {"gdn-d64-repeat1": ("gdn-d64", "mqar256", 0.0002), "m2-s16-d32-repeat1": ("m2-s16-d32", "mqar512", 0.0006)}


def test_diag4_runs_nine_fixed_rate_post_selection_seeds_on_diag3_settings(cfg, work_dir):
    settings = ladder.slice_settings(cfg, "diag4")
    assert settings == ladder.slice_settings(cfg, "diag3")
    assert settings["train"]["stop_val_acc"] == 1.01 and settings["train"]["time_budget_s"] == 15000
    assert settings["caps"]["wall_s"] == 18000 and ladder.slice_workers(cfg, "diag4", None) == 3
    cells = cfg["slices"]["diag4"]["cells"]
    assert [(c["task"], c["models"], c["lrs"], c["seeds"], c["steps"]) for c in cells] == [
        ("mqar512", ["gdn-d32"], [0.0003], [1, 2, 3], 18000),
        ("mqar256", ["gdn-d64-repeat1"], [0.0002], [1, 2, 3], 18000),
        ("mqar512", ["m2-s16-d32-repeat1"], [0.0006], [1, 2, 3], 18000)]
    sched = ladder.Scheduler(cfg, None, work_dir, "diag4", 3, log=lambda m: None)
    assert sched.pending == {}                                                          # no sweep, no selection
    assert sched.queue == (
        [key("mqar512", "gdn-d32", 0.0003, s) for s in (1, 2, 3)]
        + [key("mqar256", "gdn-d64-repeat1", 0.0002, s) for s in (1, 2, 3)]
        + [key("mqar512", "m2-s16-d32-repeat1", 0.0006, s) for s in (1, 2, 3)])


@pytest.mark.parametrize("name", sorted(DIAG4_REPEATS))
def test_diag4_repeats_train_the_diag3_job_under_distinct_run_ids(cfg, name):
    base, task, lr = DIAG4_REPEATS[name]
    assert cfg["models"][name] == {**cfg["models"][base], "attempt": 1}
    settings = ladder.slice_settings(cfg, "diag4")
    for seed in (1, 2, 3):
        original, repeat = key(task, base, lr, seed), key(task, name, lr, seed)
        assert ladder.build_spec(cfg, repeat, ["K4"], settings, "mps") == ladder.build_spec(
            cfg, original, ["K4"], ladder.slice_settings(cfg, "diag3"), "mps")
        assert ladder.run_id(cfg, repeat, settings) != ladder.run_id(cfg, original, ladder.slice_settings(cfg, "diag3"))


INST_FAIL_MODELS = ["gdn-d32-inst-f1", "gdn-d32-inst-f2", "gdn-d32-inst-f3"]
DIAG3_GDN_D32_RUN = Path(__file__).resolve().parents[3] / "results" / "phase10-diag3-20261013T2009Z" / "runs" / "r-f6e55f827701"


def test_inst_fail_mps_is_three_instrumented_attempts_of_the_configuration_that_stayed_at_chance(cfg, work_dir):
    settings = ladder.slice_settings(cfg, "inst-fail-mps")
    diag3 = ladder.slice_settings(cfg, "diag3")
    assert {k: v for k, v in settings.items() if k not in ("instrument", "seeds")} == {
        k: v for k, v in diag3.items() if k != "seeds"}                      # seeds never enter the job spec or id
    assert settings["seeds"] == [1]
    assert settings["instrument"] == ladder.slice_settings(cfg, "inst-mps")["instrument"]
    assert settings["instrument"] == {**ladder.INSTRUMENT_DEFAULTS, "enabled": True, "save_every": 1800}
    assert settings["train"]["stop_val_acc"] == 1.01 and settings["train"]["time_budget_s"] == 15000
    assert settings["caps"]["wall_s"] == 18000
    cells = cfg["slices"]["inst-fail-mps"]["cells"]
    assert [(c["task"], c["models"], c["lrs"], c["seeds"], c["steps"]) for c in cells] == [
        ("mqar512", INST_FAIL_MODELS, [0.001], [1], 18000)]
    sched = ladder.Scheduler(cfg, None, work_dir, "inst-fail-mps", 2, log=lambda m: None)
    assert sched.pending == {}
    assert sched.queue == [key("mqar512", name, 0.001, 1) for name in INST_FAIL_MODELS]
    assert ladder.slice_workers(cfg, "inst-fail-mps", 2) == 2


def test_inst_fail_attempts_train_the_diag3_gdn_d32_job_apart_from_instrumentation_and_label(cfg):
    settings = ladder.slice_settings(cfg, "inst-fail-mps")
    diag3 = ladder.slice_settings(cfg, "diag3")
    replay = key("mqar512", "gdn-d32-replay1", 0.001, 1)
    expected = ladder.build_spec(cfg, replay, ["K4"], diag3, "mps")
    ids = {ladder.run_id(cfg, replay, diag3), ladder.run_id(cfg, key("mqar512", "gdn-d32-replay2", 0.001, 1), diag3),
           ladder.run_id(cfg, key("mqar512", "gdn-d32", 0.001, 1), ladder.slice_settings(cfg, "diag2"))}
    for attempt, name in enumerate(INST_FAIL_MODELS, 1):
        assert cfg["models"][name] == {**cfg["models"]["gdn-d32"], "attempt": "inst-f%d" % attempt}
        spec = ladder.build_spec(cfg, key("mqar512", name, 0.001, 1), ["K4"], settings, "mps")
        assert spec.pop("instrument") == settings["instrument"]
        assert spec == expected
        ids.add(ladder.run_id(cfg, key("mqar512", name, 0.001, 1), settings))
    assert len(ids) == 6


@pytest.mark.skipif(not (DIAG3_GDN_D32_RUN / "result.json").exists(), reason="needs the diag3 run artifacts")
def test_inst_fail_attempt_specs_equal_the_recorded_diag3_run_spec_apart_from_instrumentation(cfg):
    recorded = json.loads((DIAG3_GDN_D32_RUN / "result.json").read_text())
    settings = ladder.slice_settings(cfg, "inst-fail-mps")
    names = [e["name"] for e in recorded["spec"]["eval_sets"]]
    for name in INST_FAIL_MODELS:
        spec = ladder.build_spec(cfg, key("mqar512", name, 0.001, 1), names, settings, "mps")
        del spec["instrument"]
        assert spec == recorded["spec"]


INST_LR_MODELS = ["gdn-d32-inst-l1", "gdn-d32-inst-l2"]


def test_inst_lr1e4_mps_is_two_instrumented_attempts_of_the_inst_fail_job_at_peak_lr_1e4(cfg, work_dir):
    settings = ladder.slice_settings(cfg, "inst-lr1e4-mps")
    assert settings == ladder.slice_settings(cfg, "inst-fail-mps")
    assert settings["seeds"] == [1]
    assert settings["instrument"] == {**ladder.INSTRUMENT_DEFAULTS, "enabled": True, "save_every": 1800}
    assert settings["train"]["stop_val_acc"] == 1.01 and settings["train"]["time_budget_s"] == 15000
    assert settings["caps"]["wall_s"] == 18000
    cells = cfg["slices"]["inst-lr1e4-mps"]["cells"]
    assert [(c["task"], c["models"], c["lrs"], c["seeds"], c["steps"]) for c in cells] == [
        ("mqar512", INST_LR_MODELS, [0.0001], [1], 18000)]
    sched = ladder.Scheduler(cfg, None, work_dir, "inst-lr1e4-mps", 2, log=lambda m: None)
    assert sched.pending == {}
    assert sched.queue == [key("mqar512", name, 0.0001, 1) for name in INST_LR_MODELS]
    assert ladder.slice_workers(cfg, "inst-lr1e4-mps", 2) == 2


def test_inst_lr1e4_attempt_specs_differ_from_inst_fail_specs_only_in_peak_lr(cfg):
    settings = ladder.slice_settings(cfg, "inst-lr1e4-mps")
    fail_settings = ladder.slice_settings(cfg, "inst-fail-mps")
    baseline = ladder.build_spec(cfg, key("mqar512", INST_FAIL_MODELS[0], 0.001, 1), ["K4"], fail_settings, "mps")
    ids = {ladder.run_id(cfg, key("mqar512", name, 0.001, 1), fail_settings) for name in INST_FAIL_MODELS}
    for attempt, name in enumerate(INST_LR_MODELS, 1):
        assert cfg["models"][name] == {**cfg["models"]["gdn-d32"], "attempt": "inst-l%d" % attempt}
        spec = ladder.build_spec(cfg, key("mqar512", name, 0.0001, 1), ["K4"], settings, "mps")
        assert spec == {**baseline, "train": {**baseline["train"], "lr": 0.0001}}
        assert baseline["train"]["lr"] == 0.001
        ids.add(ladder.run_id(cfg, key("mqar512", name, 0.0001, 1), settings))
    assert len(ids) == 5


def test_scheduler_sweeps_seed_0_then_runs_three_post_selection_seeds(cfg, work_dir):
    sched = ladder.Scheduler(cfg, None, work_dir, "diag1", 1, log=lambda m: None)
    assert len(sched.queue) == 21 and {k["seed"] for k in sched.queue} == {0}
    assert sum(len(c["seeds"]) for c in sched.pending.values()) == 21                    # 42 runs in all
    cell = [k for k in sched.queue if (k["task"], k["model"]) == ("mqar512", "la-d32")]
    for k, acc in zip(cell, (0.2, 0.6, 0.4)):
        sched._finished(k, {"train": {"best_val_acc": acc}})
    assert sched.queue[:3] == [key("mqar512", "la-d32", 0.003, s) for s in (1, 2, 3)]
    choice = json.loads((work_dir / "lr_choice" / "mqar512__la-d32.json").read_text())
    assert choice["chosen_lr"] == 0.003


def _config_with(work_dir, extra):
    path = work_dir / "ladder.toml"
    path.write_text(CONFIG.read_text() + "\n" + extra + "\n")
    return path


@pytest.mark.parametrize("extra,message", [
    ("[slices.bad]\ncells = []\nworkers = 3", "unknown keys"),
    ("[slices.bad]\ncells = []\nexecution = \"gpu\"", "unknown execution"),
    ("[slices.bad]\ncells = []\ntrain = { stop_val = 1.0 }", "not keys of \\[train\\]"),
    ("[slices.bad]\ncells = []\ncaps = { wall = 1 }", "not keys of \\[caps\\]"),
], ids=["key", "mode", "train", "caps"])
def test_config_rejects_malformed_slices(work_dir, extra, message):
    with pytest.raises(ValueError, match=message):
        ladder.load_config(_config_with(work_dir, extra))


def test_config_rejects_host_mps_without_its_table(work_dir):
    text = CONFIG.read_text()
    start = text.index("[execution.host_mps]")
    end = text.index("\n\n", start)
    path = work_dir / "ladder.toml"
    path.write_text(text[:start] + text[end:])
    with pytest.raises(ValueError, match="no \\[execution.host_mps\\] table"):
        ladder.load_config(path)


def test_host_mps_workers_never_exceed_the_measured_limit(cfg):
    limit = cfg["execution"]["host_mps"]["workers"]
    assert ladder.slice_workers(cfg, "diag1", None) == limit
    assert ladder.slice_workers(cfg, "diag1", 1) == 1
    with pytest.raises(ValueError, match="at most"):
        ladder.slice_workers(cfg, "diag1", limit + 1)
    assert ladder.slice_workers(cfg, "first", None) == ladder.DEFAULT_CONTAINER_WORKERS
    assert ladder.slice_workers(cfg, "first", 7) == 7


# ------------------------------------------------------------------------------------------------ host backend

def _stage_trusted(job_in):
    (job_in / "code").mkdir(parents=True)
    shutil.copyfile(ladder.RUNNER, job_in / "ladder_runner.py")
    for path in ladder.CODE_FILES:
        shutil.copyfile(path, job_in / "code" / path.name)


def _bare_backend():
    backend = object.__new__(ladder_host.HostMpsBackend)
    backend.trusted = ladder.trusted_files()
    return backend


def test_host_backend_accepts_exactly_the_trusted_files(work_dir):
    job_in = work_dir / "in"
    _stage_trusted(job_in)
    backend = _bare_backend()
    backend.check_trusted(job_in)
    (job_in / "code" / "candidate.py").write_text("print('hi')\n")
    with pytest.raises(PermissionError, match="trusted"):
        backend.check_trusted(job_in)
    (job_in / "code" / "candidate.py").unlink()
    with open(job_in / "code" / "bb_recall.py", "a") as handle:
        handle.write("\n# edited\n")
    with pytest.raises(PermissionError, match="trusted"):
        backend.check_trusted(job_in)


def test_trusted_files_are_the_files_the_code_hash_covers():
    assert set(ladder.trusted_files()) == {"ladder_runner.py", *("code/%s" % p.name for p in ladder.CODE_FILES)}


@mps
def test_sandbox_profile_denies_network_and_writes_outside_the_job(work_dir):
    out = work_dir / "out"
    out.mkdir()
    probe = ("import socket, sys\n"
             "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3); print('net-open')\n"
             "except OSError: print('net-denied')\n"
             "try:\n    open(sys.argv[1], 'w'); print('outside-open')\n"
             "except OSError: print('outside-denied')\n"
             "open(sys.argv[2], 'w').write('x'); print('inside-ok')\n")
    profile = ladder_host.sandbox_profile([out])
    done = subprocess.run([ladder_host.SANDBOX_EXEC, "-p", profile, sys.executable, "-I", "-c", probe,
                           str(work_dir / "outside.txt"), str(out / "inside.txt")],
                          capture_output=True, text=True, timeout=60)
    assert done.stdout.split() == ["net-denied", "outside-denied", "inside-ok"], done.stderr


# ------------------------------------------------------------------------------------------------ runner device

def _runner_job(work_dir, cfg, device, name):
    settings = ladder.slice_settings(cfg, "smoke")
    k = ladder.run_key("mqar64", "tf-d32", 0.003, 0, 40)
    set_names = ladder.build_eval_sets(cfg, "mqar64", work_dir / "data")
    job_in, job_out = work_dir / name / "in", work_dir / name / "out"
    _stage_trusted(job_in)
    (job_in / "data").mkdir()
    job_out.mkdir(parents=True)
    for n in set_names:
        shutil.copyfile(work_dir / "data" / "mqar64" / ("%s.npz" % n), job_in / "data" / ("%s.npz" % n))
    spec = ladder.build_spec(cfg, k, set_names, settings, device)
    (job_in / "spec.json").write_text(json.dumps(spec))
    return job_in, job_out, set_names


def _run_runner(job_in, job_out):
    subprocess.run([sys.executable, str(job_in / "ladder_runner.py"), str(job_in), str(job_out)], check=True,
                   capture_output=True, timeout=300)
    return json.loads((job_out / "train.json").read_text())


def test_runner_records_cpu_as_its_device(cfg, work_dir):
    job_in, job_out, _ = _runner_job(work_dir, cfg, "cpu", "cpu")
    train = _run_runner(job_in, job_out)
    assert train["device"] == "cpu" and train["dtype"] == "float32" and train["steps_run"] == 40


def test_runner_rejects_an_unknown_device(cfg, work_dir):
    job_in, job_out, _ = _runner_job(work_dir, cfg, "cuda", "cuda")
    with pytest.raises(subprocess.CalledProcessError):
        _run_runner(job_in, job_out)


@mps
def test_runner_on_mps_tracks_cpu_losses_and_refuses_cpu_fallback(cfg, work_dir):
    cpu = _run_runner(*_runner_job(work_dir, cfg, "cpu", "cpu")[:2])
    job_in, job_out, set_names = _runner_job(work_dir, cfg, "mps", "mps")
    on_mps = _run_runner(job_in, job_out)
    assert on_mps["device"] == "mps"
    for a, b in zip(cpu["curve"], on_mps["curve"]):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-3)
    for n in set_names:
        assert np.load(job_out / "pred" / ("%s.npy" % n)).dtype == np.int16
    job_in, job_out, _ = _runner_job(work_dir, cfg, "mps", "fallback")
    env = {"PATH": "/usr/bin:/bin", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    done = subprocess.run([sys.executable, str(job_in / "ladder_runner.py"), str(job_in), str(job_out)],
                          capture_output=True, text=True, timeout=300, env=env)
    assert done.returncode != 0 and "PYTORCH_ENABLE_MPS_FALLBACK" in done.stderr


@mps
def test_rope_tables_are_the_same_on_mps():
    from hybrid.bench.baselines import bb_recall
    cos, sin = bb_recall.rope_tables(torch.arange(300), 32)
    mcos, msin = bb_recall.rope_tables(torch.arange(300), 32, device=torch.device("mps"))
    assert mcos.device.type == "mps"
    assert torch.equal(cos, mcos.cpu()) and torch.equal(sin, msin.cpu())


@mps
def test_host_mps_execute_end_to_end(cfg, work_dir):
    settings = dict(ladder.slice_settings(cfg, "smoke"), execution="host_mps")
    backend = ladder.make_backend(cfg, "host_mps", work_dir)
    k = ladder.run_key("mqar64", "gdn-d32", 0.003, 0, 40)
    result = ladder.execute(cfg, backend, work_dir, k, settings, log=lambda m: None)
    assert result["status"] == "ok", result["detail"]
    assert result["device"] == "mps" and result["train"]["device"] == "mps"
    prov = result["execution"]
    assert prov["mode"] == "host_mps" and prov["torch"] == torch.__version__
    assert prov["code_hash"] == ladder.code_hash() == result["code_hash"]
    assert prov["network"].startswith("denied")
    assert set(result["sets"]) == {"K4", "K8", "K16"}
    assert result["run_id"] != ladder.run_id(cfg, k, ladder.slice_settings(cfg, "smoke"))
    container = ladder.ContainerBackend(cfg, executor=None)
    with pytest.raises(ValueError, match="cannot run"):
        ladder.execute(cfg, container, work_dir, k, settings, log=lambda m: None)
