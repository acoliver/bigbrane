"""Test ladder, rung 1: host-side orchestration of calibration runs (DISCOVERY-PIPELINE-PLAN.md section 11,
IMPROVEMENT-PLAN.md B.3 to B.5).

One executor job per run: a (task, model, learning rate, seed) point. The host builds the evaluation sets
(tokens and positions to report go to the job; answers and group labels stay on the host), the job trains with
answer-only loss on batches it generates from the train seed and returns argmax predictions, and the host scores
them. Every run writes <out>/runs/<run id>/result.json; a run whose result exists is skipped, so an interrupted
grid resumes where it stopped.

Learning-rate protocol (B.4): for each (task, model) cell, the first seed (SWEEP_SEED, 0) at every learning rate in
the slice, then the learning rate with the best validation accuracy (validation drawn from the training
distribution, never the test sets) is run at the remaining seeds (ACCEPTANCE_SEEDS, 1, 2 and 3, by default). A slice
may instead fix one learning rate per cell, or name `lr_from = "task/model"` (same-seed repeat attempts, decision
B10t): the cell then runs at the rate chosen for that cell of the same slice.

A slice may override [train] and [caps] keys and name an execution mode (`execution`, default "container"):
- "container": one executor job per run in the pinned image, on CPU. Candidate code only ever runs this way.
- "host_mps": the trusted baselines on the host's Apple GPU (ladder_host.py), for diagnostics only (decision
  B10: the calibration gate refuses results that did not run in the container on CPU),
  configured by [execution.host_mps] (host python, worker count). Results carry the device and the backend's
  provenance (torch version, code hash, backend hash, sandbox).
The run id hashes the effective training settings and, for a mode other than "container", the mode; a slice with
no overrides in the container gets the same id formula as before these options existed.

A slice may also enable logging-only training instrumentation (`instrument = { enabled = true, ... }`, keys and
defaults in INSTRUMENT_DEFAULTS, recorder in ladder_instrument.py). Its settings then enter the job spec and the
run id; without it, neither changes. Each instrumented job writes runs/<id>/out/instrument.jsonl.

    python -m hybrid.bench.ladder microbench --config hybrid/bench/configs/ladder.toml --out results/phase10-<stamp>
    python -m hybrid.bench.ladder run --config ... --slice first --out ... [--workers 6]
    python -m hybrid.bench.ladder report --config ... --out ...
    python -m hybrid.bench.ladder gate [--report hybrid/bench/calibration/rung1.json]

`gate` reads the stored rung 1 calibration report and exits 0 only when every required criterion passed on the
current code hash; otherwise it prints why and exits 1.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
import tempfile
import threading
import time
import tomllib
import traceback
from pathlib import Path

import numpy as np

from hybrid.executor.executor import Executor
from hybrid.executor.job import Caps, Job

from .baselines import bb_recall
from .tasks import copy_task, mqar, ngram_lookup

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
TASK_MODULES = {"mqar": mqar, "copy_task": copy_task, "ngram_lookup": ngram_lookup}
CODE_FILES = [BENCH_DIR / "baselines" / "bb_recall.py", BENCH_DIR / "tasks" / "mqar.py",
              BENCH_DIR / "tasks" / "copy_task.py", BENCH_DIR / "tasks" / "ngram_lookup.py",
              BENCH_DIR / "ladder_instrument.py"]
RUNNER = BENCH_DIR / "ladder_runner.py"
STORED_CALIBRATION = BENCH_DIR / "calibration" / "rung1.json"
IGNORE = -100
TEST_PART = 3
CONTAINER = "container"
EXECUTION_MODES = (CONTAINER, "host_mps")
SLICE_KEYS = {"cells", "execution", "train", "caps", "seeds", "instrument"}
# Training instrumentation (ladder_instrument.py), opt-in per slice with `instrument = { enabled = true, ... }`.
# A slice whose table is absent or has enabled = false gets no instrument settings, so its job spec and run id
# formula are those of an uninstrumented slice. probe_every / early_steps: probe schedule (gradients, update
# ratios, gates, per-K accuracy); checkpoint_every: digest interval; save_every: weight and optimizer snapshot
# interval (0 = none); hash_batches: training batches hashed one by one; probe_rows / perk_rows: rows in the probe
# batch and in each per-K set.
INSTRUMENT_DEFAULTS = {"enabled": False, "probe_every": 100, "early_steps": [1, 10, 50], "checkpoint_every": 900,
                       "save_every": 0, "hash_batches": 16, "probe_rows": 16, "perk_rows": 128, "save_init": True}
# Seed policy (decision B10): seed 0 sweeps the learning rate, then the acceptance seeds 1, 2 and 3 run at the
# chosen rate. ladder_criteria judges calibration on the acceptance seeds only.
SWEEP_SEED = 0
ACCEPTANCE_SEEDS = (1, 2, 3)
DEFAULT_SEEDS = [SWEEP_SEED, *ACCEPTANCE_SEEDS]
DEFAULT_CONTAINER_WORKERS = 6


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _log(message):
    print("[ladder %s] %s" % (_utc(), message), file=sys.stderr, flush=True)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def load_config(path):
    with open(path, "rb") as handle:
        cfg = tomllib.load(handle)
    for key in ("image", "test_seed", "caps", "train", "tasks", "models", "slices"):
        if key not in cfg:
            raise ValueError("ladder config %s: missing %r" % (path, key))
    for name, task in cfg["tasks"].items():
        if task["module"] not in TASK_MODULES:
            raise ValueError("task %s: unknown module %r" % (name, task["module"]))
    for name, model in cfg["models"].items():
        if model["family"] not in bb_recall.FAMILIES:
            raise ValueError("model %s: unknown family %r" % (name, model["family"]))
    for name in cfg.get("execution", {}):
        if name not in EXECUTION_MODES or name == CONTAINER:
            raise ValueError("ladder config %s: unknown [execution.%s] (configurable: host_mps)" % (path, name))
    if "host_mps" in cfg.get("execution", {}):
        host = cfg["execution"]["host_mps"]
        if set(host) != {"python", "workers"} or int(host["workers"]) < 1:
            raise ValueError("[execution.host_mps] needs exactly python and a positive workers, got %s" % host)
    for name, slice_ in cfg["slices"].items():
        unknown = set(slice_) - SLICE_KEYS
        if unknown:
            raise ValueError("slice %s: unknown keys %s" % (name, sorted(unknown)))
        mode = slice_.get("execution", CONTAINER)
        if mode not in EXECUTION_MODES:
            raise ValueError("slice %s: unknown execution %r" % (name, mode))
        if mode != CONTAINER and mode not in cfg.get("execution", {}):
            raise ValueError("slice %s: execution %r has no [execution.%s] table" % (name, mode, mode))
        for table in ("train", "caps"):
            extra = set(slice_.get(table, {})) - set(cfg[table])
            if extra:
                raise ValueError("slice %s: %s overrides %s are not keys of [%s]" % (name, table, sorted(extra), table))
        instrument = slice_.get("instrument", {})
        extra = set(instrument) - set(INSTRUMENT_DEFAULTS)
        if extra:
            raise ValueError("slice %s: unknown instrument keys %s" % (name, sorted(extra)))
        if "instrument" in slice_ and not isinstance(instrument.get("enabled"), bool):
            raise ValueError("slice %s: instrument needs enabled = true or false" % name)
        _check_followers(name, slice_["cells"])
    return cfg


def parse_cell_ref(text):
    """('task', 'model') of an `lr_from = "task/model"` reference."""
    task, _, model = text.partition("/")
    if not task or not model or "/" in model:
        raise ValueError("lr_from %r is not task/model" % (text,))
    return task, model


def _check_followers(slice_name, cells):
    """A cell with `lr_from = "task/model"` has no learning-rate list of its own: it runs at the rate chosen for the
    named cell of the same slice (a sweep cell or a cell with one rate), which must itself not be a follower."""
    own = {(cell["task"], model) for cell in cells if "lr_from" not in cell for model in cell["models"]}
    for cell in cells:
        if "lr_from" not in cell:
            continue
        if "lrs" in cell:
            raise ValueError("slice %s: a cell with lr_from takes no lrs" % slice_name)
        if parse_cell_ref(cell["lr_from"]) not in own:
            raise ValueError("slice %s: lr_from %r names no cell of the slice with its own learning rates"
                             % (slice_name, cell["lr_from"]))


def slice_settings(cfg, slice_name):
    """Effective execution mode, [train], [caps] and seeds of a slice (its overrides on top of the config), plus
    its instrument settings when the slice enables instrumentation."""
    slice_ = cfg["slices"][slice_name]
    settings = {"execution": slice_.get("execution", CONTAINER),
                "train": {**cfg["train"], **slice_.get("train", {})},
                "caps": {**cfg["caps"], **slice_.get("caps", {})},
                "seeds": list(slice_.get("seeds", DEFAULT_SEEDS))}
    if slice_.get("instrument", {}).get("enabled"):
        settings["instrument"] = {**INSTRUMENT_DEFAULTS, **slice_["instrument"]}
    return settings


def code_hash():
    digest = hashlib.sha256()
    for path in [RUNNER, *CODE_FILES]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def trusted_files():
    """Job-relative path -> SHA-256 of every file a ladder job stages: the files code_hash covers."""
    files = {"ladder_runner.py": RUNNER, **{"code/%s" % p.name: p for p in CODE_FILES}}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


# --------------------------------------------------------------------------------------------- tasks

def train_params(task):
    """Generator keyword arguments for training batches, from a [tasks.<name>] table."""
    module = task["module"]
    if module == "mqar":
        return {"seq_len": task["seq_len"], "k_values": task["train_k"], "vocab": task["vocab"],
                "power_a": task.get("power_a", 0.01)}
    if module == "copy_task":
        return {"lengths": list(range(1, task["train_max_len"] + 1)), "alphabet": task["alphabet"]}
    n, k = task.get("n", 3), task.get("k", 3)
    return {"lengths": list(range(max(n + k, task.get("train_min_len", n + k)), task["train_max_len"] + 1)),
            "variant": task["variant"], "n": n, "k": k, "alphabet": task["alphabet"]}


def task_vocab(task):
    module = TASK_MODULES[task["module"]]
    return module.vocab_size(**{k: task[k] for k in ("vocab", "alphabet") if k in task})


def eval_set_defs(task):
    """[(set name, generator kwargs, label)] for a task's evaluation sets."""
    module = task["module"]
    if module == "mqar":
        return [("K%d" % k, {"seq_len": task["seq_len"], "k_values": [k], "vocab": task["vocab"],
                             "power_a": task.get("power_a", 0.01)}, {"K": k}) for k in task["eval_k"]]
    if module == "copy_task":
        return [("L%d" % length, {"lengths": [length], "alphabet": task["alphabet"]}, {"L": length})
                for length in task["eval_lengths"]]
    return [("L%d" % length, {"lengths": [length], "variant": task["variant"], "n": task.get("n", 3),
                              "k": task.get("k", 3), "alphabet": task["alphabet"]}, {"L": length})
            for length in task["eval_lengths"]]


def _write_npz(path, **arrays):
    # Workers may regenerate the same set after another worker starts staging it. Never truncate
    # a published archive: use a unique file in the same directory, then replace its name atomically.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            np.savez(handle, **arrays)
            handle.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def build_eval_sets(cfg, task_name, data_dir):
    """Write <data_dir>/<task>/<set>.npz (tokens, positions: staged to jobs) and <set>.oracle.npz (targets and
    annotations: host only). Returns the list of set names. Idempotent: existing files are reused. Both archives
    are published atomically, with the oracle last so its presence marks a complete set."""
    task = cfg["tasks"][task_name]
    module = TASK_MODULES[task["module"]]
    out = data_dir / task_name
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for index, (name, kwargs, _label) in enumerate(eval_set_defs(task)):
        names.append(name)
        if (out / ("%s.oracle.npz" % name)).exists():
            continue
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((int(cfg["test_seed"]), 7, TEST_PART,
                                                                           index))))
        data = module.generate(rng, int(task["n_eval"]), **kwargs)
        positions = data["targets"] != IGNORE
        _write_npz(out / ("%s.npz" % name), tokens=data["tokens"].astype(np.int16), positions=positions)
        extras = {k: v for k, v in data.items() if k not in ("tokens", "targets")}
        _write_npz(out / ("%s.oracle.npz" % name), targets=data["targets"], positions=positions, **extras)
    return names


def score_set(module_name, oracle, pred):
    """Accuracy at the listed positions, plus per-sequence and per-distance-bucket breakdowns."""
    positions = oracle["positions"]
    targets = oracle["targets"][positions]
    if pred.shape != targets.shape:
        raise ValueError("prediction has %s entries, expected %s" % (pred.shape, targets.shape))
    correct = pred.astype(np.int64) == targets
    row_of = np.nonzero(positions)[0]
    per_row = np.bincount(row_of, weights=correct, minlength=positions.shape[0])
    count = np.bincount(row_of, minlength=positions.shape[0])
    result = {"acc": float(correct.mean()), "n": int(correct.size), "rows": int(positions.shape[0]),
              "seq_acc": float(np.mean(per_row == count))}
    if module_name == "mqar":
        dist = oracle["distance"][positions]
        buckets = {}
        edges = [0, 8, 16, 32, 64, 128, 256, 512, 1 << 20]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (dist > lo) & (dist <= hi)
            if sel.any():
                buckets["%d-%d" % (lo + 1, hi)] = {"acc": float(correct[sel].mean()), "n": int(sel.sum())}
        result["distance"] = buckets
    return result


# --------------------------------------------------------------------------------------------- runs

def run_key(task_name, model_name, lr, seed, steps):
    return {"task": task_name, "model": model_name, "lr": float(lr), "seed": int(seed), "steps": int(steps)}


def run_id(cfg, key, settings):
    """Content id of a run. `settings` is slice_settings(); a container slice without [train] overrides hashes
    exactly what the first slice hashed (key, task, model, [train], test seed, code hash)."""
    task, model = cfg["tasks"][key["task"]], cfg["models"][key["model"]]
    payload = {"key": key, "task": task, "model": model, "train": settings["train"], "test_seed": cfg["test_seed"],
               "code": code_hash()}
    if settings["execution"] != CONTAINER:
        payload["execution"] = settings["execution"]
    if "instrument" in settings:
        payload["instrument"] = settings["instrument"]
    blob = json.dumps(payload, sort_keys=True)
    return "r-" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def build_spec(cfg, key, set_names, settings, device):
    task, model = cfg["tasks"][key["task"]], cfg["models"][key["model"]]
    tr = dict(settings["train"])
    tr.update({"steps": key["steps"], "lr": key["lr"], "model_seed": key["seed"], "data_seed": key["seed"],
               "threads": settings["caps"].get("threads", 1), "device": device})
    tr.update(task.get("train_overrides", {}))
    spec = {"mode": "train", "family": model["family"], "d": model["d"], "layers": model.get("layers", 2),
            "hyper": model.get("hyper", {}),
            "task": {"module": task["module"], "train_params": train_params(task), "vocab": task_vocab(task)},
            "train": tr, "eval_sets": [{"name": n, "file": "data/%s.npz" % n} for n in set_names],
            "state_lengths": cfg.get("state_lengths", [64, 256, 512, 4096])}
    if "instrument" in settings:
        spec["instrument"] = dict(settings["instrument"])
    return spec


def caps(settings):
    c = settings["caps"]
    return Caps(wall_s=c["wall_s"], memory_mb=c["memory_mb"], pids=c["pids"], cpus=float(c.get("threads", 1)),
                tmp_mb=c["tmp_mb"])


class ContainerBackend:
    """Runs a ladder job as one executor job in the pinned image, on CPU."""

    mode, device = CONTAINER, "cpu"

    def __init__(self, cfg, executor):
        self.image, self.executor = cfg["image"], executor

    def provenance(self):
        return {"mode": CONTAINER, "device": "cpu", "image": self.image}

    def run(self, job_id, job_in, job_out, job_caps):
        job = Job(job_id, self.image, job_in, job_out, ("python3", "/in/ladder_runner.py", "/in", "/out"), job_caps)
        return self.executor.run(job)


def make_backend(cfg, mode, out_dir):
    if mode == CONTAINER:
        return ContainerBackend(cfg, Executor(out_dir.name.lower(), out_dir / "receipts"))
    from .ladder_host import HostMpsBackend
    return HostMpsBackend(ROOT / cfg["execution"][mode]["python"], out_dir / "receipts", trusted_files(),
                          code_hash(), out_dir.name.lower())


def execute(cfg, backend, out_dir, key, settings, log=_log):
    """Run one calibration point unless its result exists. Returns the result dict."""
    if backend.mode != settings["execution"]:
        raise ValueError("backend %s cannot run a %s slice" % (backend.mode, settings["execution"]))
    rid = run_id(cfg, key, settings)
    run_dir = out_dir / "runs" / rid
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    set_names = build_eval_sets(cfg, key["task"], out_dir / "data")
    spec = build_spec(cfg, key, set_names, settings, backend.device)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    job_in, job_out = run_dir / "in", run_dir / "out"
    (job_in / "code").mkdir(parents=True)
    (job_in / "data").mkdir()
    job_out.mkdir()
    shutil.copyfile(RUNNER, job_in / "ladder_runner.py")
    for path in CODE_FILES:
        shutil.copyfile(path, job_in / "code" / path.name)
    for name in set_names:
        shutil.copyfile(out_dir / "data" / key["task"] / ("%s.npz" % name), job_in / "data" / ("%s.npz" % name))
    _write_json(job_in / "spec.json", spec)
    log("start %s %s on %s" % (rid, key, backend.device))
    began = time.monotonic()
    receipt = backend.run(rid, job_in, job_out, caps(settings))
    wall = time.monotonic() - began
    result = {"run_id": rid, "key": key, "spec": spec, "status": receipt.status, "detail": receipt.detail,
              "receipt": {"job_id": receipt.job_id, "run_id": receipt.run_id, "run_s": receipt.run_s,
                          "image_digest": receipt.image_digest, "started_utc": receipt.started_utc,
                          "stderr_path": receipt.stderr_path, "exit_code": receipt.exit_code},
              "device": backend.device, "execution": backend.provenance(),
              "wall_s": wall, "code_hash": code_hash(), "finished_utc": _utc()}
    if receipt.status == "ok":
        result["train"] = json.loads((job_out / "train.json").read_text())
        if result["train"]["device"] != backend.device:
            raise RuntimeError("%s: job trained on %s, backend %s" % (rid, result["train"]["device"], backend.device))
        task, model = cfg["tasks"][key["task"]], cfg["models"][key["model"]]
        result["sets"] = {}
        for name in set_names:
            oracle = np.load(out_dir / "data" / key["task"] / ("%s.oracle.npz" % name), allow_pickle=False)
            pred = np.load(job_out / "pred" / ("%s.npy" % name), allow_pickle=False)
            result["sets"][name] = score_set(task["module"], oracle, pred)
        lengths = cfg.get("state_lengths", [64, 256, 512, 4096])
        result["state_floats"] = {str(n): bb_recall.state_formula(model["family"], model["d"], n,
                                                                  layers=model.get("layers", 2),
                                                                  hyper=model.get("hyper", {})) for n in lengths}
        result["mixer_state_floats"] = {str(n): bb_recall.state_formula(
            model["family"], model["d"], n, layers=model.get("layers", 2), hyper=model.get("hyper", {}),
            include_conv=False) for n in lengths}
        if result["state_floats"] != {k: int(v) for k, v in result["train"]["state_floats_at"].items()}:
            raise RuntimeError("%s: host state formula %s disagrees with the job's model %s"
                               % (rid, result["state_floats"], result["train"]["state_floats_at"]))
    shutil.rmtree(job_in)
    _write_json(result_path, result)
    accs = {n: round(s["acc"], 3) for n, s in result.get("sets", {}).items()}
    log("done %s %s status %s %.0fs %s" % (rid, key, receipt.status, wall, accs))
    with open(out_dir / "runs.jsonl", "a") as handle:
        handle.write(json.dumps({"run_id": rid, "key": key, "status": receipt.status, "wall_s": wall,
                                 "device": backend.device, "accs": accs,
                                 "finished_utc": result["finished_utc"]}) + "\n")
    return result


class Scheduler:
    """Work queue over cells. A cell with a learning-rate sweep queues its first-seed sweep first; when every
    sweep run of a cell has finished, the best learning rate's remaining seeds go to the front of the queue.

    A cell with `lr_from = "task/model"` (same-seed repeat attempts, or another task trained the same way) has no
    sweep: its runs queue at the learning rate of the named cell, once that rate is known, right behind the named
    cell's own post-selection seeds."""

    def __init__(self, cfg, backend, out_dir, slice_name, workers, log=_log):
        self.cfg, self.backend, self.out_dir, self.log = cfg, backend, out_dir, log
        self.settings = slice_settings(cfg, slice_name)
        self.workers = workers
        self.lock = threading.Lock()
        self.queue, self.pending, self.failures = [], {}, []
        self.followers, fixed_lr = {}, {}
        for cell in cfg["slices"][slice_name]["cells"]:
            for model in cell["models"]:
                if "lr_from" in cell:
                    self.followers.setdefault(parse_cell_ref(cell["lr_from"]), []).append((cell, model))
                else:
                    self._add_cell(cell, model)
                    if len(cell["lrs"]) == 1:
                        fixed_lr[(cell["task"], model)] = cell["lrs"][0]
        for base, lr in fixed_lr.items():
            self.queue.extend(self._follower_runs(base, lr))

    def _follower_runs(self, base, lr):
        return [run_key(cell["task"], model, lr, seed, int(cell["steps"]))
                for cell, model in self.followers.get(base, []) for seed in cell.get("seeds", self.settings["seeds"])]

    def _add_cell(self, cell, model):
        seeds, steps = cell.get("seeds", self.settings["seeds"]), int(cell["steps"])
        lrs = cell["lrs"]
        cell_id = (cell["task"], model)
        if len(lrs) == 1:
            for seed in seeds:
                self.queue.append(run_key(cell["task"], model, lrs[0], seed, steps))
            return
        self.pending[cell_id] = {"left": len(lrs), "results": {}, "sweep_seed": seeds[0], "seeds": seeds[1:],
                                 "steps": steps}
        for lr in lrs:
            self.queue.append(run_key(cell["task"], model, lr, seeds[0], steps))

    def _finished(self, key, result):
        cell_id = (key["task"], key["model"])
        with self.lock:
            cell = self.pending.get(cell_id)
            if cell is None or key["seed"] != cell["sweep_seed"] or key["lr"] in cell["results"]:
                return
            cell["results"][key["lr"]] = result
            cell["left"] -= 1
            if cell["left"]:
                return
            scored = {lr: (r.get("train", {}).get("best_val_acc", -1.0), lr) for lr, r in cell["results"].items()}
            best = max(scored, key=lambda lr: scored[lr])
            self.log("cell %s: learning rate %g chosen (validation %s)" % (
                cell_id, best, {lr: round(v[0], 3) for lr, v in scored.items()}))
            _write_json(self.out_dir / "lr_choice" / ("%s__%s.json" % cell_id),
                        {"task": cell_id[0], "model": cell_id[1], "chosen_lr": best,
                         "val_acc": {str(lr): v[0] for lr, v in scored.items()}})
            own = [run_key(cell_id[0], cell_id[1], best, seed, cell["steps"]) for seed in cell["seeds"]]
            self.queue[0:0] = own + self._follower_runs(cell_id, best)
            del self.pending[cell_id]

    def _worker(self):
        while True:
            with self.lock:
                if not self.queue:
                    if not self.pending:
                        return
                    key = None
                else:
                    key = self.queue.pop(0)
            if key is None:
                time.sleep(5)
                continue
            try:
                result = execute(self.cfg, self.backend, self.out_dir, key, self.settings, self.log)
            except Exception as error:  # recorded and reported; the rest of the grid continues
                self.log("FAILED %s: %s" % (key, error))
                self.failures.append({"key": key, "error": "%s\n%s" % (error, traceback.format_exc())})
                result = {"status": "host_error"}
            self._finished(key, result)

    def run(self):
        threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(self.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return self.failures


# --------------------------------------------------------------------------------------------- microbench

def microbench(cfg, executor, out_dir, threads_list, log=_log):
    mb = cfg["microbench"]
    configs = []
    for entry in mb["configs"]:
        model = cfg["models"][entry["model"]]
        configs.append({"model": entry["model"], "family": model["family"], "d": model["d"],
                        "hyper": model.get("hyper", {}), "vocab": entry["vocab"], "seq_len": entry["seq_len"],
                        "batch": entry["batch"]})
    spec = {"mode": "microbench", "configs": configs, "threads": threads_list, "warmup": mb.get("warmup", 2),
            "steps": mb.get("steps", 5), "infer_lengths": mb.get("infer_lengths", [256, 512, 1024, 2048, 4096])}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    job_id = "mb-%s-t%s" % (stamp, "-".join(str(t) for t in threads_list))
    base = out_dir / "microbench" / job_id
    job_in, job_out = base / "in", base / "out"
    (job_in / "code").mkdir(parents=True)
    job_out.mkdir()
    shutil.copyfile(RUNNER, job_in / "ladder_runner.py")
    for path in CODE_FILES:
        shutil.copyfile(path, job_in / "code" / path.name)
    _write_json(job_in / "spec.json", spec)
    c = cfg["caps"]
    job = Job(job_id, cfg["image"], job_in, job_out, ("python3", "/in/ladder_runner.py", "/in", "/out"),
              Caps(wall_s=mb.get("wall_s", 3600), memory_mb=c["memory_mb"], pids=c["pids"],
                   cpus=float(max(threads_list)), tmp_mb=c["tmp_mb"]))
    log("microbench %s threads %s, %d configs" % (job_id, threads_list, len(configs)))
    receipt = executor.run(job)
    log("microbench %s: %s %s (%.0fs)" % (job_id, receipt.status, receipt.detail, receipt.run_s))
    if receipt.status != "ok":
        raise RuntimeError("microbench job failed: %s %s; stderr %s" % (receipt.status, receipt.detail,
                                                                         receipt.stderr_path))
    report = json.loads((job_out / "microbench.json").read_text())
    report["receipt"] = {"job_id": receipt.job_id, "run_s": receipt.run_s, "image_digest": receipt.image_digest,
                         "started_utc": receipt.started_utc, "cpus": float(max(threads_list))}
    _write_json(base / "microbench.json", report)
    return report


# --------------------------------------------------------------------------------------------- report

def load_results(out_dir):
    results = []
    for path in sorted((out_dir / "runs").glob("*/result.json")):
        results.append(json.loads(path.read_text()))
    return results


def chosen_points(results, out_dir):
    """Results at the chosen learning rate of each cell (all seeds), keyed (task, model) -> {seed: result}."""
    choice = {}
    for path in (out_dir / "lr_choice").glob("*.json"):
        entry = json.loads(path.read_text())
        choice[(entry["task"], entry["model"])] = entry["chosen_lr"]
    points = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        key = result["key"]
        cell = (key["task"], key["model"])
        lr = choice.get(cell)
        if lr is not None and abs(key["lr"] - lr) > 1e-15:
            continue
        points.setdefault(cell, {})[key["seed"]] = result
    return points


def slice_workers(cfg, slice_name, requested):
    """Worker count for a slice: the request, else the mode's default. host_mps never exceeds its configured
    workers, which were measured so the GPU is not oversubscribed."""
    mode = slice_settings(cfg, slice_name)["execution"]
    if mode == CONTAINER:
        return DEFAULT_CONTAINER_WORKERS if requested is None else requested
    limit = int(cfg["execution"][mode]["workers"])
    if requested is not None and requested > limit:
        raise ValueError("slice %s runs %s with at most %d workers (requested %d)" % (slice_name, mode, limit,
                                                                                      requested))
    return limit if requested is None else requested


def gate(report_path):
    """Exit status of `ladder gate`: 0 when the stored rung 1 report is calibrated on the current code."""
    from . import ladder_criteria
    report_path = Path(report_path)
    if not report_path.exists():
        print("rung 1 NOT calibrated: no stored calibration report at %s" % report_path)
        return 1
    verdict = ladder_criteria.gate(json.loads(report_path.read_text()), code_hash())
    if verdict["calibrated"]:
        print("rung 1 calibrated: %s (code %s, devices %s)" % (report_path, verdict["code_hash"][:12],
                                                               ", ".join(verdict["devices"])))
        return 0
    print("rung 1 NOT calibrated: %s" % report_path)
    for reason in verdict["reasons"]:
        print("- %s" % reason)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hybrid.bench.ladder")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("microbench", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--slice", required=True)
            p.add_argument("--workers", type=int, default=None,
                           help="default 6 in the container; host_mps uses its configured workers")
        if name == "microbench":
            p.add_argument("--threads", default="1")
    p = sub.add_parser("gate")
    p.add_argument("--report", default=str(STORED_CALIBRATION))
    args = parser.parse_args(argv)
    if args.command == "gate":
        return gate(args.report)
    cfg = load_config(args.config)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        from . import ladder_criteria
        report = ladder_criteria.build_report(cfg, out_dir)
        _write_json(out_dir / "calibration.json", report)
        print(ladder_criteria.render_markdown(report))
        return 0
    if args.command == "microbench":
        executor = Executor(out_dir.name.lower(), out_dir / "receipts")
        threads = [int(t) for t in args.threads.split(",")]
        report = microbench(cfg, executor, out_dir, threads)
        for row in report["rows"]:
            print("%-28s N %4d B %3d threads %d: %.3f s/step, %.0f tok/s" % (
                row["model"], row["seq_len"], row["batch"], row["threads"], row["median_step_s"],
                row["tokens_per_s"]))
        return 0
    if args.slice not in cfg["slices"]:
        raise SystemExit("unknown slice %r (have %s)" % (args.slice, ", ".join(cfg["slices"])))
    workers = slice_workers(cfg, args.slice, args.workers)
    backend = make_backend(cfg, slice_settings(cfg, args.slice)["execution"], out_dir)
    _log("slice %s: %s on %s, %d workers, code %s" % (args.slice, backend.mode, backend.device, workers,
                                                      code_hash()[:12]))
    _write_json(out_dir / ("execution-%s.json" % args.slice),
                {"slice": args.slice, "workers": workers, "settings": slice_settings(cfg, args.slice),
                 "execution": backend.provenance(), "code_hash": code_hash(), "started_utc": _utc()})
    failures = Scheduler(cfg, backend, out_dir, args.slice, workers).run()
    _write_json(out_dir / ("failures-%s.json" % args.slice), failures)
    _log("slice %s finished, %d failures" % (args.slice, len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
