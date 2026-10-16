"""Host-side orchestration of one bench evaluation: two kinds of executor job.

1. Candidate jobs (the candidate and every uncached baseline): /in holds runner.py,
   candidate/, spec.json and data/ (train pools and test prefixes). No oracle file is ever
   written into a candidate job's input directory.
2. One evaluator job: /in holds the trusted evaluator code, oracle/ and entrants/ (each
   entrant's receipt and outputs). It writes bench_result.json.
3. With card falsifier tasks (`falsifiers=[task dir, ...]`, see falsifier.py): one more candidate job per
   (entrant, task), running falsifier_runner.py on that task's train sequences and test prefixes. Every
   entrant, cached baselines included, runs them. Their oracles go only to the evaluator job, which writes
   the per-task results and rule decisions under `falsifiers` in bench_result.json.

Baseline outputs are cached under runs/_bench_cache/<task config hash>/<mode>/<name>-s<seed>/
so they are computed once per (task config, code, image, baseline, seed).

Layout written under <run_dir>/bench/ (which must not exist yet):
    data/candidate_inputs/   spec.json + data/<task>/{train,prefixes}.json
    data/oracle/<task>.npz   oracle arrays (only ever staged into the evaluator job)
    data/falsifier_inputs/<task id>/   spec.json + data/{train,prefixes}.json per falsifier task
    data/falsifier_oracle/<task id>.npz   falsifier oracles (only ever staged into the evaluator job)
    jobs/<job-id>/{in,out}   executor job directories
    (receipts go to the executor's receipts_dir; the CLI uses <run_dir>/receipts/)
    bench_result.json        evaluator output plus the host block (trusted flag, receipts, timings)
    manifest.json            sha256 of every file under bench/
"""
from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import math
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hybrid.executor.executor import Executor
from hybrid.executor.job import Caps, Job
from hybrid.executor.lock import read_lock

from . import config as bench_config_mod
from . import falsifier as falsifier_mod
from .baselines import BASELINES, stage_baseline
from .local import LocalExecutor
from .tasks import BUILDERS, build_task
from .tasks.common import length_part

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
DEFAULT_CACHE = ROOT / "runs" / "_bench_cache"
EVAL_ENTRY = ("import sys; sys.path.insert(0, sys.argv[1] + '/code'); "
              "from hybrid.bench.evaluator import main; main(sys.argv[1], sys.argv[2])")
EVALUATOR_CODE = ["__init__.py", "metrics.py", "scaling.py", "fingerprint.py", "falsifier_eval.py", "evaluator.py"]
CANDIDATE_DIR_LIMIT = 16 << 20


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _job_id():
    return "job-" + uuid.uuid4().hex[:12]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _log(message):
    print("[bench %s] %s" % (_utc(), message), file=sys.stderr, flush=True)


def copy_candidate_dir(source, destination):
    """Copy a candidate directory, refusing symlinks and special files; skip bytecode caches."""
    source = Path(source).resolve()
    if not (source / "candidate.py").is_file():
        raise ValueError("candidate directory %s has no candidate.py" % source)
    total = 0
    for path in source.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("candidate directory holds a symlink or special file: %s" % path)
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            total += path.stat().st_size
    destination.mkdir(parents=True, exist_ok=True)
    if total > CANDIDATE_DIR_LIMIT:
        raise ValueError("candidate directory exceeds %d bytes" % CANDIDATE_DIR_LIMIT)


def _base_length(name, params):
    if "length" in params:
        return int(params["length"])
    return int(inspect.signature(BUILDERS[name]).parameters["length"].default)


def build_length_probes(cfg, name, task_cfg, inputs, oracle_dir):
    """Longer test sets for the length-generalization fingerprint. Sequences are `factor` times the training
    length and the scored position lies in the upper half, so every scored position is beyond what the
    training sequences ever showed. Each set draws from its own stream, so it shares nothing with the base
    test set. Returns (spec entries for the candidate job, summary for the evaluator spec)."""
    base = _base_length(name, task_cfg["params"])
    probes, summary = [], {}
    for factor in cfg["fingerprint"]["length_factors"]:
        params = {**task_cfg["params"], "length": base * factor, "min_prefix": base * factor // 2,
                  "test_part": length_part(factor)}
        ext = build_task(name, cfg["seed"], 1, task_cfg["n_test"], params)
        rel = "data/%s/prefixes_x%d.json" % (name, factor)
        _write_json(inputs / rel, {"prefixes": ext.prefixes})
        np.savez(oracle_dir / ("%s__x%d.npz" % (name, factor)), oracle=ext.oracle)
        probes.append({"factor": factor, "prefix_file": rel})
        summary[str(factor)] = {"length": base * factor, "min_prefix": base * factor // 2,
                                "n_prefixes": len(ext.prefixes)}
    return probes, summary


def build_data(cfg, data_dir):
    """Generate every task on the host, write candidate inputs and oracle files separately."""
    inputs, oracle_dir = data_dir / "candidate_inputs", data_dir / "oracle"
    oracle_dir.mkdir(parents=True)
    spec_tasks, eval_tasks = [], {}
    for name, task_cfg in cfg["tasks"].items():
        sizes = task_cfg["train_sizes"]
        task = build_task(name, cfg["seed"], max(sizes), task_cfg["n_test"], task_cfg["params"])
        _write_json(inputs / "data" / name / "train.json", {"sequences": task.train})
        _write_json(inputs / "data" / name / "prefixes.json", {"prefixes": task.prefixes})
        arrays = {"oracle": task.oracle, "group": task.group,
                  **{"extra_" + k: v for k, v in task.extras.items()}}
        np.savez(oracle_dir / ("%s.npz" % name), **arrays)
        _write_json(oracle_dir / ("%s.meta.json" % name), task.meta)
        spec_tasks.append({"name": name, "vocab_size": task.vocab_size, "train_sizes": sizes,
                           "train_file": "data/%s/train.json" % name, "prefix_file": "data/%s/prefixes.json" % name})
        eval_meta = {k: task.meta[k] for k in ("close_lo", "close_hi") if k in task.meta}
        if name in cfg["fingerprint"]["length_tasks"]:
            probes, probe_summary = build_length_probes(cfg, name, task_cfg, inputs, oracle_dir)
            spec_tasks[-1]["length_probes"] = probes
        else:
            probe_summary = {}
        eval_tasks[name] = {"vocab_size": task.vocab_size, "train_sizes": sizes, "oracle_kind": task.oracle_kind,
                            "train_tokens": {str(n): task.train_tokens(n) for n in sizes},
                            "group_name": task.group_name, "group_labels": task.group_labels,
                            "n_prefixes": len(task.prefixes), "target": task_cfg.get("target"), "meta": eval_meta,
                            "length_probes": probe_summary}
        if name == "recall":   # chance level and bits per stored value, for the capacity fingerprint
            eval_tasks[name].update(chance_acc=1.0 / task.meta["n_values"], value_bits=math.log2(task.meta["n_values"]))
        if name == "fewshot":
            eval_tasks[name].update(chance_acc=1.0 / task.meta["modulus"])
    sc = cfg["scaling"]
    lo, hi = sc["lengths_log2"]
    spec = {"seed": cfg["model_seed"], "budget": cfg["budget"], "tasks": spec_tasks,
            "scaling": {"task": sc["task"], "train_size": sc["train_size"], "lengths": [2 ** k for k in range(lo, hi + 1)],
                        "batch": sc["batch"], "repeats": sc["repeats"], "warmup": sc["warmup"], "seed": sc["seed"],
                        "stop_after_s": sc["stop_after_s"]}}
    _write_json(inputs / "spec.json", spec)
    return inputs, oracle_dir, eval_tasks


def _caps(cfg, role):
    c = cfg["caps"][role]
    return Caps(wall_s=c["wall_s"], memory_mb=c["memory_mb"], pids=c["pids"], cpus=1.0, tmp_mb=c["tmp_mb"])


def _receipt_summary(receipt):
    data = asdict(receipt)
    keep = ("job_id", "run_id", "status", "detail", "exit_code", "run_s", "outputs", "image", "image_digest",
            "stdout_path", "stderr_path", "started_utc")
    return {k: data[k] for k in keep if k in data}


def run_candidate_job(executor, image, cfg, inputs, candidate_dir, jobs_dir):
    job_id = _job_id()
    job_in, job_out = jobs_dir / job_id / "in", jobs_dir / job_id / "out"
    shutil.copytree(inputs, job_in)
    shutil.copyfile(BENCH_DIR / "runner.py", job_in / "runner.py")
    copy_candidate_dir(candidate_dir, job_in / "candidate")
    job_out.mkdir(parents=True)
    job = Job(job_id, image, job_in, job_out, ("python3", "/in/runner.py", "/in", "/out"), _caps(cfg, "candidate"))
    receipt = executor.run(job)
    return _receipt_summary(receipt), job_out


def run_falsifier_job(executor, image, cfg, inputs, candidate_dir, jobs_dir):
    job_id = _job_id()
    job_in, job_out = jobs_dir / job_id / "in", jobs_dir / job_id / "out"
    shutil.copytree(inputs, job_in)
    shutil.copyfile(falsifier_mod.RUNNER, job_in / "falsifier_runner.py")
    copy_candidate_dir(candidate_dir, job_in / "candidate")
    job_out.mkdir(parents=True)
    job = Job(job_id, image, job_in, job_out, ("python3", "/in/falsifier_runner.py", "/in", "/out"),
              _caps(cfg, "candidate"))
    receipt = executor.run(job)
    return _receipt_summary(receipt), job_out


def stage_falsifier_data(tasks, cfg, data_dir):
    """Candidate inputs per falsifier task, and the oracle files for the evaluator. Returns {task id: inputs dir}."""
    inputs = {}
    if tasks:
        (data_dir / "falsifier_oracle").mkdir(parents=True)
    for task in tasks:
        inputs[task.task_id] = data_dir / "falsifier_inputs" / task.task_id
        falsifier_mod.stage_inputs(task, inputs[task.task_id], cfg["model_seed"], cfg["budget"])
        shutil.copyfile(task.directory / "oracle.npz", data_dir / "falsifier_oracle" / ("%s.npz" % task.task_id))
    return inputs


def _cache_dir(cache_root, tch, mode_tag, name, seed):
    return Path(cache_root) / tch[:24] / mode_tag / ("%s-s%d" % (name, seed))


def evaluate_candidate(candidate_dir, bench_config, run_dir, executor, *, claimed_exponent=None, baselines=None,
                       cache_dir=None, log=_log, falsifiers=None):
    """Evaluate one candidate directory (or only the baselines when candidate_dir is None).

    executor: hybrid.executor.executor.Executor (trusted) or hybrid.bench.local.LocalExecutor
    (trusted: false). falsifiers: validated falsifier task directories (falsifier.write_task) run next to
    the standard tasks. Returns the bench_result dict, also written to <run_dir>/bench/bench_result.json."""
    began = time.monotonic()
    local = isinstance(executor, LocalExecutor)
    if not local and not isinstance(executor, Executor):
        raise TypeError("executor must be an Executor or a LocalExecutor, got %r" % type(executor))
    cfg = bench_config_mod.load_config(bench_config)
    tasks = [falsifier_mod.load_task(Path(d)) for d in (falsifiers or [])]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("falsifier task ids must be distinct: %s" % [task.task_id for task in tasks])
    image = cfg.get("image") or read_lock()["reference"]
    names = list(cfg["baselines"] if baselines is None else baselines)
    for name in names:
        if name not in BASELINES:
            raise ValueError("unknown baseline %r" % name)
    bench_dir = Path(run_dir).resolve() / "bench"
    if bench_dir.exists():
        raise FileExistsError("%s already exists; use a fresh run directory" % bench_dir)
    jobs_dir = bench_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    tch = bench_config_mod.task_config_hash(cfg)
    mode_tag = "local" if local else image.rpartition("@sha256:")[2][:16]
    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    log("task config %s, mode %s, baselines %s" % (tch[:12], "local" if local else "podman", names))

    t0 = time.monotonic()
    inputs, oracle_dir, eval_tasks = build_data(cfg, bench_dir / "data")
    falsifier_inputs = stage_falsifier_data(tasks, cfg, bench_dir / "data")
    data_s = time.monotonic() - t0

    entrants, pending, sources = {}, [], {}
    if candidate_dir is not None:
        entrants["candidate"] = {"role": "candidate", "claimed_exponent": claimed_exponent, "source": str(candidate_dir)}
        sources["candidate"] = Path(candidate_dir)
        pending.append(("main", "candidate", None, Path(candidate_dir)))
    for name in names:
        if name == "candidate":
            raise ValueError("a baseline cannot be named 'candidate'")
        entrants[name] = {"role": "baseline", "claimed_exponent": BASELINES[name]["claimed_exponent"],
                          "state": BASELINES[name]["state"]}
        cached = _cache_dir(cache_root, tch, mode_tag, name, cfg["model_seed"])
        if (cached / "receipt.json").is_file():
            entrants[name].update(source="cache", cache_dir=str(cached))
            log("baseline %s from cache %s" % (name, cached))
            if tasks:
                sources[name] = stage_baseline(name, bench_dir / "baseline_src" / name)
        else:
            staged = stage_baseline(name, bench_dir / "baseline_src" / name)
            entrants[name].update(source="job")
            sources[name] = staged
            pending.append(("main", name, None, staged))
    for name, directory in sources.items():
        for task in tasks:
            pending.append(("falsifier", name, task.task_id, directory))

    def work(item):
        kind, name, task_id, directory = item
        label = name if task_id is None else "%s on falsifier %s" % (name, task_id)
        log("candidate job for %s started" % label)
        if kind == "main":
            summary, out = run_candidate_job(executor, image, cfg, inputs, directory, jobs_dir)
        else:
            summary, out = run_falsifier_job(executor, image, cfg, falsifier_inputs[task_id], directory, jobs_dir)
        log("candidate job for %s: %s in %.1fs" % (label, summary["status"], summary["run_s"]))
        return kind, name, task_id, summary, out

    t0 = time.monotonic()
    results, falsifier_results = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, int(cfg["parallel_jobs"]))) as pool:
        for kind, name, task_id, summary, out in pool.map(work, pending):
            if kind == "main":
                results[name] = (summary, out)
            else:
                falsifier_results[(name, task_id)] = (summary, out)
    jobs_s = time.monotonic() - t0

    eval_id = _job_id()
    eval_in, eval_out = jobs_dir / eval_id / "in", jobs_dir / eval_id / "out"
    (eval_in / "code" / "hybrid" / "bench").mkdir(parents=True)
    shutil.copyfile(ROOT / "hybrid" / "__init__.py", eval_in / "code" / "hybrid" / "__init__.py")
    for file in EVALUATOR_CODE:
        shutil.copyfile(BENCH_DIR / file, eval_in / "code" / "hybrid" / "bench" / file)
    shutil.copytree(oracle_dir, eval_in / "oracle", ignore=shutil.ignore_patterns("*.meta.json"))
    receipts = {}
    for name, entrant in entrants.items():
        target = eval_in / "entrants" / name
        if entrant["source"] == "cache":
            cached = Path(entrant["cache_dir"])
            summary = json.loads((cached / "receipt.json").read_text())
            shutil.copytree(cached / "out", target / "out")
        else:
            summary, out = results[name]
            if summary["status"] == "ok":
                shutil.copytree(out, target / "out")
                if entrant["role"] == "baseline":
                    cached = _cache_dir(cache_root, tch, mode_tag, name, cfg["model_seed"])
                    staging = cached.with_name(cached.name + ".tmp-" + uuid.uuid4().hex[:6])
                    shutil.copytree(out, staging / "out")
                    _write_json(staging / "receipt.json", summary)
                    _write_json(staging / "meta.json", {"task_config_hash": tch, "mode": mode_tag, "image": image,
                                                        "baseline": name, "seed": cfg["model_seed"],
                                                        "created_utc": _utc(), "trusted": not local})
                    staging.rename(cached)
            else:
                target.mkdir(parents=True)
        _write_json(target / "receipt.json", summary)
        receipts[name] = summary
    falsifier_receipts = {}
    if tasks:
        shutil.copytree(bench_dir / "data" / "falsifier_oracle", eval_in / "falsifier_oracle")
    for (name, task_id), (summary, out) in falsifier_results.items():
        target = eval_in / "falsifier_entrants" / name / task_id
        if summary["status"] == "ok":
            shutil.copytree(out, target / "out")
        else:
            target.mkdir(parents=True)
        _write_json(target / "receipt.json", summary)
        falsifier_receipts.setdefault(name, {})[task_id] = summary
    eval_spec = {"tasks": eval_tasks, "entrants": entrants, "task_config_hash": tch,
                 "config_hash": bench_config_mod.config_hash(cfg), "fingerprint": cfg["fingerprint"],
                 "scaling": {"tolerance": cfg["scaling"]["tolerance"], "tail_points": cfg["scaling"]["tail_points"]}}
    if tasks:
        eval_spec["falsifiers"] = {task.task_id: falsifier_mod.eval_entry(task) for task in tasks}
    _write_json(eval_in / "eval_spec.json", eval_spec)
    eval_out.mkdir(parents=True)
    t0 = time.monotonic()
    job = Job(eval_id, image, eval_in, eval_out, ("python3", "-c", EVAL_ENTRY, "/in", "/out"), _caps(cfg, "evaluator"))
    eval_receipt = _receipt_summary(executor.run(job))
    eval_s = time.monotonic() - t0
    log("evaluator job: %s in %.1fs" % (eval_receipt["status"], eval_receipt["run_s"]))
    if eval_receipt["status"] != "ok":
        raise RuntimeError("evaluator job failed: %s (stderr %s)" % (eval_receipt["detail"], eval_receipt["stderr_path"]))
    result = json.loads((eval_out / "bench_result.json").read_text())
    result["host"] = {
        "trusted": not local, "mode": "local" if local else "podman", "image": None if local else image,
        "created_utc": _utc(), "candidate_dir": None if candidate_dir is None else str(Path(candidate_dir).resolve()),
        "config": cfg, "receipts": receipts, "evaluator_receipt": eval_receipt,
        "falsifier_receipts": falsifier_receipts,
        "falsifier_dirs": [str(task.directory) for task in tasks],
        "falsifier_code": falsifier_code_record(eval_in, tasks),
        "evaluator_output_sha256": hashlib.sha256((eval_out / "bench_result.json").read_bytes()).hexdigest(),
        "timings_s": {"data": data_s, "candidate_jobs": jobs_s, "evaluator_job": eval_s,
                      "total": time.monotonic() - began}}
    result["trusted"] = not local
    _write_json(bench_dir / "bench_result.json", result)
    write_manifest(bench_dir)
    return result


def falsifier_code_record(eval_in, tasks):
    """sha256 of the evaluator code staged for this run next to the hashes the validation job of each task ran,
    so a code edit between validation and evaluation is visible in the result. same_code is None for a task
    validated before hashes were recorded."""
    staged = {name: hashlib.sha256((eval_in / "code" / "hybrid" / "bench" / name).read_bytes()).hexdigest()
              for name in ("falsifier_eval.py", "metrics.py")}
    per_task = {}
    for task in tasks:
        validated = (task.design.get("validation") or {}).get("code_sha256")
        per_task[task.task_id] = {
            "validated_with": validated,
            "same_code": None if validated is None else all(validated.get(k) == v for k, v in staged.items())}
    return {"evaluator_code_sha256": staged, "tasks": per_task}


def write_manifest(bench_dir):
    files = {}
    for path in sorted(bench_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(bench_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(bench_dir / "manifest.json", {"created_utc": _utc(), "files": files})
