"""Bench config: TOML loading, validation and the hashes used for baseline caching."""
from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

from .tasks import TASK_NAMES

BENCH_DIR = Path(__file__).resolve().parent
TOP_KEYS = {"seed", "model_seed", "n_test", "train_sizes", "baselines", "parallel_jobs", "image",
            "budget", "caps", "scaling", "tasks", "fingerprint"}
TASK_KEYS = {"train_sizes", "n_test", "params", "target"}
SCALING_KEYS = {"task", "train_size", "lengths_log2", "batch", "repeats", "warmup", "seed", "stop_after_s",
                "tolerance", "tail_points"}
BUDGET_KEYS = {"max_steps", "max_epochs", "batch_size", "lr"}
CAPS_KEYS = {"wall_s", "memory_mb", "pids", "tmp_mb"}
# Mechanism fingerprint settings. length_factors and length_tasks decide what the candidate job predicts
# (so they enter the task config hash); every other key is an evaluator-only threshold.
FINGERPRINT_DEFAULTS = {
    "length_factors": [2, 4],       # test sequences this many times longer than the training length
    "length_tasks": ["hmm", "vom", "compose"],
    "learned_ratio": 2.0,           # recall counts as learned if accuracy at the smallest K >= ratio * chance
    "capacity_advantage_ratio": 1.0,  # no_capacity_advantage if capacity per state float <= ratio * best fixed baseline
    "quadratic_slope": 1.5,         # quadratic_or_worse if the tail time slope is above this
    "fewshot_slope_min": 0.03,      # no_fewshot_gain if accuracy gained per doubling of shots is below this
    "js_close": 0.02,               # a task counts as "same behavior" if mean JS (nats) is at most this ...
    "ratio_close": 0.5,             # ... and the oracle-normalized distance is at most this
    "close_task_fraction": 0.5,     # behaves_like needs close tasks strictly above this fraction of compared tasks
    "memory_flat_slope": 0.5,       # memory growth is "flat" below this log-log slope of peak RSS
    "memory_floor_mb": 16.0,        # or when peak RSS delta at the longest length is below this
}
FINGERPRINT_HASHED = ("length_factors", "length_tasks")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _unknown(where, given, allowed):
    extra = set(given) - allowed
    if extra:
        raise ValueError("unknown %s keys: %s" % (where, sorted(extra)))


def load_config(source):
    """source: path to a TOML file or an already parsed dict. Returns a validated, filled-in copy."""
    if isinstance(source, dict):
        cfg = copy.deepcopy(source)
    else:
        with open(source, "rb") as handle:
            cfg = tomllib.load(handle)
    _unknown("top-level", cfg, TOP_KEYS)
    for key in ("seed", "model_seed", "n_test", "train_sizes", "budget", "caps", "scaling", "tasks"):
        if key not in cfg:
            raise ValueError("config is missing %r" % key)
    _unknown("budget", cfg["budget"], BUDGET_KEYS)
    if set(cfg["budget"]) != BUDGET_KEYS:
        raise ValueError("budget needs exactly %s" % sorted(BUDGET_KEYS))
    for role in ("candidate", "evaluator"):
        _unknown("caps." + role, cfg["caps"][role], CAPS_KEYS)
    _unknown("scaling", cfg["scaling"], SCALING_KEYS)
    cfg.setdefault("baselines", ["ngram", "gru", "transformer", "ssm", "linear_attention"])
    cfg.setdefault("parallel_jobs", 4)
    fingerprint = cfg.setdefault("fingerprint", {})
    _unknown("fingerprint", fingerprint, set(FINGERPRINT_DEFAULTS))
    for key, value in FINGERPRINT_DEFAULTS.items():
        fingerprint.setdefault(key, copy.deepcopy(value))
    factors = fingerprint["length_factors"]
    if factors != sorted(set(factors)) or any(not isinstance(f, int) or f < 2 for f in factors):
        raise ValueError("fingerprint.length_factors must be strictly increasing ints of at least 2")
    for name in fingerprint["length_tasks"]:
        if name not in ("hmm", "vom", "compose"):
            raise ValueError("fingerprint.length_tasks may only name hmm, vom or compose, got %r" % name)
        if name not in cfg["tasks"]:
            raise ValueError("fingerprint.length_tasks names %r, which is not an evaluated task" % name)
    for name, task in cfg["tasks"].items():
        if name not in TASK_NAMES:
            raise ValueError("unknown task %r" % name)
        _unknown("tasks." + name, task, TASK_KEYS)
        task.setdefault("train_sizes", list(cfg["train_sizes"]))
        task.setdefault("n_test", cfg["n_test"])
        task.setdefault("params", {})
        sizes = task["train_sizes"]
        if sizes != sorted(set(sizes)) or sizes[0] < 1:
            raise ValueError("tasks.%s.train_sizes must be strictly increasing positive ints" % name)
        target = task.get("target")
        if target is not None and (target.get("op") not in ("<=", ">=") or target.get("metric") not in ("kl", "acc")):
            raise ValueError("tasks.%s.target needs metric kl|acc and op <=|>=" % name)
    scaling = cfg["scaling"]
    if scaling["task"] not in cfg["tasks"] or scaling["train_size"] not in cfg["tasks"][scaling["task"]]["train_sizes"]:
        raise ValueError("scaling.task/train_size must name an evaluated task and one of its train sizes")
    return cfg


def code_files():
    files = sorted((BENCH_DIR / "tasks").glob("*.py")) + [BENCH_DIR / "runner.py"]
    return files + sorted((BENCH_DIR / "baselines").glob("*.py"))


def code_hash():
    digest = hashlib.sha256()
    for path in code_files():
        digest.update(path.relative_to(BENCH_DIR).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def task_config_hash(cfg):
    """Everything that changes what a candidate job sees or does (not evaluator-only settings)."""
    scaling = {k: v for k, v in cfg["scaling"].items() if k not in ("tolerance", "tail_points")}
    tasks = {name: {k: v for k, v in t.items() if k != "target"} for name, t in cfg["tasks"].items()}
    payload = {"seed": cfg["seed"], "model_seed": cfg["model_seed"], "budget": cfg["budget"], "scaling": scaling,
               "tasks": tasks, "code": code_hash(),
               "fingerprint": {k: cfg["fingerprint"][k] for k in FINGERPRINT_HASHED}}
    return sha256_text(canonical_json(payload))


def config_hash(cfg):
    return sha256_text(canonical_json(cfg))
