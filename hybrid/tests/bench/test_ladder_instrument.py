"""Training instrumentation for ladder jobs (hybrid/bench/ladder_instrument.py, opt-in per slice).

Pins down that instrumentation is opt-in (an absent or disabled table changes neither the job spec nor the run id
formula), that the investigation slices train diag4's gdn-d64 seed-3 job, that an instrumented CPU job trains
exactly like an uninstrumented one (every per-step loss, every validation accuracy and every prediction equal),
and that the records it writes can be checked independently: the initial-weight digest against a freshly built
model, the first batch hash against the train stream, saved checkpoints against their digests, and the per-K
null excess against its definition.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from hybrid.bench import ladder, ladder_host
from hybrid.bench.tasks import mqar

from .conftest import SESSION_DIR

sys.path.insert(0, str(ladder.BENCH_DIR / "baselines"))
from hybrid.bench import ladder_instrument  # noqa: E402  (imports bb_recall as the job does)

CONFIG = ladder.BENCH_DIR / "configs" / "ladder.toml"
mps = pytest.mark.skipif(not (torch.backends.mps.is_available() and Path(ladder_host.SANDBOX_EXEC).exists()),
                         reason="needs Apple MPS and sandbox-exec")


@pytest.fixture(scope="module")
def cfg():
    return ladder.load_config(CONFIG)


def _config_with(work_dir, extra):
    path = work_dir / "ladder.toml"
    path.write_text(CONFIG.read_text() + "\n" + extra + "\n")
    return ladder.load_config(path)


# ------------------------------------------------------------------------------------------------ config

def test_slices_without_an_enabled_instrument_table_keep_their_spec_and_run_id_formula(cfg, work_dir):
    for name in cfg["slices"]:
        if name.startswith("inst-"):
            continue
        assert "instrument" not in ladder.slice_settings(cfg, name)
    extended = _config_with(work_dir, "[slices.off]\nexecution = \"host_mps\"\nseeds = [0, 1, 2, 3]\n"
                                      "train = { stop_val_acc = 1.01, time_budget_s = 15000 }\n"
                                      "caps = { wall_s = 18000 }\ninstrument = { enabled = false, probe_every = 7 }\n"
                                      "cells = []")
    off, diag4 = ladder.slice_settings(extended, "off"), ladder.slice_settings(extended, "diag4")
    assert off == diag4
    k = ladder.run_key("mqar256", "gdn-d64", 0.0002, 3, 18000)
    assert ladder.run_id(extended, k, off) == ladder.run_id(extended, k, diag4)
    spec = ladder.build_spec(extended, k, ["K4"], off, "mps")
    assert "instrument" not in spec and spec == ladder.build_spec(extended, k, ["K4"], diag4, "mps")


@pytest.mark.parametrize("extra,message", [
    ("[slices.bad]\ncells = []\ninstrument = { enabled = true, every = 5 }", "unknown instrument keys"),
    ("[slices.bad]\ncells = []\ninstrument = { probe_every = 5 }", "enabled = true or false"),
    ("[slices.bad]\ncells = []\ninstrument = { enabled = 1 }", "enabled = true or false"),
], ids=["key", "missing", "type"])
def test_config_rejects_malformed_instrument_tables(work_dir, extra, message):
    with pytest.raises(ValueError, match=message):
        _config_with(work_dir, extra)


def test_inst_slices_train_the_diag4_gdn_d64_seed_3_job_with_instrumentation(cfg, work_dir):
    mps_settings, cpu_settings = ladder.slice_settings(cfg, "inst-mps"), ladder.slice_settings(cfg, "inst-cpu")
    diag4 = ladder.slice_settings(cfg, "diag4")
    assert mps_settings["execution"] == "host_mps" and cpu_settings["execution"] == "container"
    expected = {**ladder.INSTRUMENT_DEFAULTS, "enabled": True, "save_every": 1800}
    assert mps_settings["instrument"] == cpu_settings["instrument"] == expected
    assert {k: v for k, v in mps_settings.items() if k != "instrument"} == diag4 | {"seeds": [3]}
    for name in ("inst-mps", "inst-cpu"):
        cells = cfg["slices"][name]["cells"]
        assert [(c["task"], c["models"], c["lrs"], c["seeds"], c["steps"]) for c in cells] == [
            ("mqar256", ["gdn-d64-inst1", "gdn-d64-inst2"], [0.0002], [3], 18000)]
        sched = ladder.Scheduler(cfg, None, work_dir, name, 1, log=lambda m: None)
        assert sched.pending == {} and sched.queue == [
            ladder.run_key("mqar256", m, 0.0002, 3, 18000) for m in ("gdn-d64-inst1", "gdn-d64-inst2")]
    base = ladder.run_key("mqar256", "gdn-d64-repeat1", 0.0002, 3, 18000)
    diag4_spec = ladder.build_spec(cfg, base, ["K4"], diag4, "mps")
    ids = {ladder.run_id(cfg, base, diag4)}
    for attempt in (1, 2):
        name = "gdn-d64-inst%d" % attempt
        assert cfg["models"][name] == {**cfg["models"]["gdn-d64"], "attempt": attempt}
        k = ladder.run_key("mqar256", name, 0.0002, 3, 18000)
        on_mps = ladder.build_spec(cfg, k, ["K4"], mps_settings, "mps")
        assert {key: v for key, v in on_mps.items() if key != "instrument"} == diag4_spec
        assert on_mps["instrument"] == expected
        on_cpu = ladder.build_spec(cfg, k, ["K4"], cpu_settings, "cpu")
        assert on_cpu["train"] == {**on_mps["train"], "device": "cpu", "time_budget_s": 30000}
        assert {key: v for key, v in on_cpu.items() if key != "train"} == {
            key: v for key, v in on_mps.items() if key != "train"}
        ids |= {ladder.run_id(cfg, k, mps_settings), ladder.run_id(cfg, k, cpu_settings)}
    assert len(ids) == 5
    assert ladder.slice_workers(cfg, "inst-mps", 1) == 1


def test_the_instrument_module_is_staged_and_hashed(cfg):
    assert ladder.BENCH_DIR / "ladder_instrument.py" in ladder.CODE_FILES
    assert "code/ladder_instrument.py" in ladder.trusted_files()


# ------------------------------------------------------------------------------------------------ helpers

def test_context_null_is_membership_over_k():
    target = np.array([[10, 11, 12, 13], [20, 21, 22, 23]])
    keyed = target.copy()
    retained = np.array([[11, 11, 11, 11], [99, 99, 99, 99]])
    assert ladder_instrument.context_null(keyed, target) == pytest.approx(0.25)
    assert ladder_instrument.context_null(retained, target) == pytest.approx(0.5 * 0.25)
    assert ladder_instrument.context_null(np.full_like(target, 5), target) == 0.0


def test_tensor_digest_sees_one_bit():
    a = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b": torch.zeros(3)}
    b = {k: v.clone() for k, v in a.items()}
    assert ladder_instrument.tensors_digest(a) == ladder_instrument.tensors_digest(b)
    b["w"].view(torch.int32)[0, 0] ^= 1
    assert ladder_instrument.tensors_digest(a) != ladder_instrument.tensors_digest(b)


def test_delta_recurrence_matches_the_reference_rule():
    from hybrid.bench.baselines import bb_recall
    torch.manual_seed(0)
    b, t, h, dh = 2, 9, 2, 4
    k = torch.nn.functional.normalize(torch.randn(b, t, h, dh), dim=-1)
    v, g, beta = torch.randn(b, t, h, dh), -torch.rand(b, t, h), torch.rand(b, t, h)
    s_norm, r_norm, u_norm = ladder_instrument.delta_recurrence(k, v, g, beta)
    _, final = bb_recall.recurrent_delta(k.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), g.transpose(1, 2),
                                         beta.transpose(1, 2))
    assert torch.allclose(s_norm[:, -1], final.flatten(2).norm(dim=-1), atol=1e-6)
    assert torch.allclose(u_norm, beta * r_norm, atol=1e-6)
    assert torch.allclose(r_norm[:, 0], v[:, 0].norm(dim=-1))                    # empty state: residual is v


# ------------------------------------------------------------------------------------------------ runner

STEPS = 240
DENSE = {**ladder.INSTRUMENT_DEFAULTS, "enabled": True, "probe_every": 7, "early_steps": [1, 2, 3],
         "checkpoint_every": 50, "save_every": 100, "hash_batches": 4, "probe_rows": 8, "perk_rows": 32}


def _job(cfg, work_dir, name, instrument, device="cpu"):
    settings = dict(ladder.slice_settings(cfg, "smoke"), train={**cfg["train"], "stop_val_acc": 1.01, "val_rows": 64})
    if instrument:
        settings["instrument"] = instrument
    k = ladder.run_key("mqar64", "gdn-d32", 0.003, 1, STEPS)
    set_names = ladder.build_eval_sets(cfg, "mqar64", work_dir / "data")
    job_in, job_out = work_dir / name / "in", work_dir / name / "out"
    (job_in / "code").mkdir(parents=True)
    (job_in / "data").mkdir()
    job_out.mkdir(parents=True)
    shutil.copyfile(ladder.RUNNER, job_in / "ladder_runner.py")
    for path in ladder.CODE_FILES:
        shutil.copyfile(path, job_in / "code" / path.name)
    for n in set_names:
        shutil.copyfile(work_dir / "data" / "mqar64" / ("%s.npz" % n), job_in / "data" / ("%s.npz" % n))
    spec = ladder.build_spec(cfg, k, set_names, settings, device)
    spec["train"]["eval_every"] = 1                                             # every step's loss in the curve
    (job_in / "spec.json").write_text(json.dumps(spec))
    subprocess.run([sys.executable, str(job_in / "ladder_runner.py"), str(job_in), str(job_out)], check=True,
                   capture_output=True, timeout=600)
    return spec, set_names, job_out


def _records(job_out):
    return [json.loads(line) for line in (job_out / "instrument.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def pair(cfg):
    base = SESSION_DIR / "instrument-pair"
    base.mkdir(parents=True, exist_ok=False)
    off = _job(cfg, base, "off", None)
    on = _job(cfg, base, "on", DENSE)
    return off, on


def test_instrumented_cpu_training_equals_uninstrumented_training_exactly(pair):
    (spec_off, set_names, out_off), (spec_on, _, out_on) = pair
    assert {k: v for k, v in spec_on.items() if k != "instrument"} == spec_off
    off, on = (json.loads((o / "train.json").read_text()) for o in (out_off, out_on))
    assert len(off["curve"]) == len(on["curve"]) == STEPS
    assert [c["train_loss"] for c in off["curve"]] == [c["train_loss"] for c in on["curve"]]
    assert [c["val_acc"] for c in off["curve"]] == [c["val_acc"] for c in on["curve"]]
    assert (off["best_step"], off["best_val_acc"]) == (on["best_step"], on["best_val_acc"])
    for n in set_names:
        assert np.array_equal(np.load(out_off / "pred" / ("%s.npy" % n)), np.load(out_on / "pred" / ("%s.npy" % n)))
    assert "instrument" not in off and not (out_off / "instrument.jsonl").exists()
    assert on["instrument"]["file"] == "instrument.jsonl" and on["instrument"]["overhead_s"] > 0


def test_instrument_records_are_complete_and_independently_checkable(pair):
    from hybrid.bench.baselines import bb_recall
    from hybrid.bench import ladder_runner
    _, (spec, _, out) = pair
    records = _records(out)
    kinds = [r["kind"] for r in records]
    assert kinds[0] == "header" and kinds[-1] == "end"
    header = records[0]
    assert header["seeds"]["model_seed"] == header["seeds"]["data_seed"] == 1
    assert header["seeds"]["streams"]["train"] == [1, 7, ladder_runner.TRAIN_PART]

    torch.manual_seed(1)
    fresh = bb_recall.LadderNet(spec["task"]["vocab"], "gated_deltanet", 32)
    assert header["init_weights_sha256"] == ladder_instrument.tensors_digest(fresh.state_dict())
    saved = np.load(out / "init_weights.npz")
    assert all(np.array_equal(saved[k], v.numpy()) for k, v in fresh.state_dict().items())

    batches = [r for r in records if r["kind"] == "batch"]
    assert [r["step"] for r in batches] == [1, 2, 3, 4]
    first = mqar.generate(ladder_runner.stream(1, ladder_runner.TRAIN_PART), spec["train"]["batch"],
                          **spec["task"]["train_params"])
    assert batches[0]["tokens_sha256"] == ladder_instrument.arrays_digest(first["tokens"])

    probes = [r for r in records if r["kind"] == "probe"]
    steps = [r["step"] for r in probes]
    assert steps == sorted({0, 1, 2, 3, STEPS} | set(range(7, STEPS + 1, 7)))
    losses = [loss for r in probes[1:] for loss in r["window"]["losses"]]
    curve = json.loads((out / "train.json").read_text())["curve"]
    assert losses == [c["train_loss"] for c in curve]
    assert sum(r["window"]["steps"] for r in probes[1:]) == STEPS
    assert records[-1]["clipped_steps"] == sum(r["window"]["clipped"] for r in probes[1:])
    last = probes[-1]
    names = [n for n, _ in fresh.named_parameters()]
    assert sorted(last["grad_norms"]) == sorted(names) == sorted(last["update_ratios"])
    assert last["grad_norm_global"] == pytest.approx(last["window"]["grad_norm_last"], rel=1e-5)
    assert set(last["perk"]) == {"K4", "K8", "K16"}
    for entry in last["perk"].values():
        assert entry["excess"] == pytest.approx(entry["acc"] - entry["null"])
    assert [layer["layer"] for layer in last["gdn"]] == [0, 1]
    for layer in last["gdn"]:
        assert 0 < layer["decay"]["q0"] <= layer["decay"]["q1"] <= 1
        assert 0 < layer["beta"]["q0"] < layer["beta"]["q1"] < 1
        assert layer["out_gate"]["q0"] >= -0.2785 and len(layer["A"]) == len(layer["dt_bias"]) == 1
        assert all(0 <= layer[key]["q0"] <= layer[key]["q1"] for key in ("state_norm", "residual_norm", "update_norm"))

    evals = [r for r in records if r["kind"] == "eval"]
    assert [r["step"] for r in evals] == list(range(1, STEPS + 1))
    assert [r["val_acc"] for r in evals] == [c["val_acc"] for c in curve]
    assert all(set(r["perk"]) == {"K4", "K8", "K16"} for r in evals)

    checkpoints = {r["step"]: r for r in records if r["kind"] == "checkpoint"}
    assert sorted(checkpoints) == [0, 50, 100, 150, 200, STEPS]
    assert checkpoints[0]["weights_sha256"] == header["init_weights_sha256"]
    for step in (100, 200):
        snapshot = torch.load(out / checkpoints[step]["saved"], weights_only=False)
        assert ladder_instrument.tensors_digest(snapshot["model"]) == checkpoints[step]["weights_sha256"]
    assert sorted(p.name for p in (out / "ckpt").iterdir()) == ["step-000100.pt", "step-000200.pt"]
    assert records[-1]["steps_run"] == STEPS


@mps
def test_host_mps_instrumented_job_end_to_end(cfg, work_dir):
    settings = dict(ladder.slice_settings(cfg, "smoke"), execution="host_mps", instrument=DENSE)
    backend = ladder.make_backend(cfg, "host_mps", work_dir)
    k = ladder.run_key("mqar64", "gdn-d32", 0.003, 0, 40)
    result = ladder.execute(cfg, backend, work_dir, k, settings, log=lambda m: None)
    assert result["status"] == "ok", result["detail"]
    assert result["spec"]["instrument"] == DENSE and result["train"]["instrument"]["file"] == "instrument.jsonl"
    records = _records(work_dir / "runs" / result["run_id"] / "out")
    assert records[0]["env"]["device"] == "mps" and records[-1]["steps_run"] == 40
    assert any(r["kind"] == "probe" and "gdn" in r for r in records)
