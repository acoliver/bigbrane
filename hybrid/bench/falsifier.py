"""Card falsifier tasks, host side: the task program contract, static checks, the validation job's files,
validated task directories, and the staging used by evaluate_candidate.

Task program contract (written by the `falsifier_designer` role, run only inside the executor):

    def generate(seed: int) -> dict:
        return {"vocab_size": V,                    # int, 2..CAPS["max_vocab"]
                "train": [[int, ...], ...],         # training sequences (the candidate fits on all of them)
                "prefixes": [[int, ...], ...],      # one scored test prefix per independent test sequence
                "next_tokens": [int, ...],          # the token that followed each prefix in the generated sequence
                "oracle_kind": "answer" | "dist",
                "oracle": [int, ...] | [[float] * V, ...],   # exact target token, or exact next-token distribution
                "group": [int, ...],                # index into group_labels per prefix
                "group_labels": [str, ...]}         # e.g. ["m=1", "m=2", "m=3"]

Standard library (ALLOWED_MODULES) and NumPy only; no files, no network, deterministic in `seed`.

A validated task directory holds task.json (the design: task_id, title, protocol, deviations, rules, program,
vocabulary and groups, validation report), train.json and prefixes.json (candidate inputs) and oracle.npz
(oracle, group, next_tokens, unigram; staged into the evaluator job only).
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .falsifier_eval import REFERENCES, unigram_dist

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
HARNESS = BENCH_DIR / "falsifier_harness.py"
RUNNER = BENCH_DIR / "falsifier_runner.py"
CAPS = {
    "max_vocab": 2048,
    "max_train_sequences": 2048,
    "max_train_tokens": 300_000,
    "max_prefixes": 4096,
    "max_prefix_tokens": 600_000,
    "max_sequence_length": 2048,
    "max_groups": 12,
    "min_prefixes": 32,
    "min_group_size": 16,
    "max_oracle_floats": 2_000_000,
}
ALLOWED_MODULES = frozenset({"numpy", "math", "itertools", "functools", "collections", "dataclasses", "typing",
                             "__future__", "heapq", "bisect", "random", "operator", "fractions", "statistics",
                             "copy", "enum"})
FORBIDDEN_CALLS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals",
                             "locals", "vars", "setattr", "delattr", "getattr", "memoryview"})
TASK_FILES = ("task.json", "train.json", "prefixes.json", "oracle.npz")


def static_problems(code: str) -> list[str]:
    """AST checks on a task program before it is ever run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ["syntax error: %s" % exc]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    problems.append("forbidden import %s (line %d)" % (alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                problems.append("forbidden import from %s (line %d)" % (node.module, node.lineno))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append("forbidden call %s() (line %d)" % (node.func.id, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__name__":
            problems.append("dunder attribute access %s (line %d)" % (node.attr, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in ("load", "save", "savez", "loadtxt", "fromfile",
                                                               "tofile", "genfromtxt", "memmap"):
            problems.append("file access through %s (line %d)" % (node.attr, node.lineno))
    if not any(isinstance(node, ast.FunctionDef) and node.name == "generate" for node in tree.body):
        problems.append("no top-level function generate(seed)")
    return problems


def rule_problems(rules: list[dict], baselines: list[str]) -> list[str]:
    """Names in a pass rule must be the candidate, a configured baseline or a trivial reference, and every
    condition must involve the candidate."""
    allowed = {"candidate", *baselines, *REFERENCES}
    problems = []
    for r, rule in enumerate(rules):
        for index, cond in enumerate(rule["conditions"]):
            where = "rule %d condition %d" % (r, index)
            for key in ("entrant", "minus"):
                name = cond.get(key)
                if name is not None and name not in allowed:
                    problems.append("%s names %s %r; allowed: %s" % (where, key, name, sorted(allowed)))
            if "candidate" not in (cond["entrant"], cond.get("minus")):
                problems.append("%s does not involve the candidate" % where)
            if cond.get("minus") == cond["entrant"]:
                problems.append("%s subtracts an entrant from itself" % where)
    return problems


def staged_code() -> dict[str, str]:
    """The harness and the bench scoring code a validation job runs, read from disk now, keyed by path under /in."""
    code = {"main.py": HARNESS.read_text(), "code/hybrid/__init__.py": (ROOT / "hybrid" / "__init__.py").read_text(),
            "code/hybrid/bench/__init__.py": (BENCH_DIR / "__init__.py").read_text()}
    for name in ("metrics.py", "falsifier_eval.py", "falsifier_reference.py"):
        code["code/hybrid/bench/" + name] = (BENCH_DIR / name).read_text()
    return code


def harness_files(program: str, seed: int, rules: list[dict], child_timeout_s: float, *,
                  train_max_len: int | None = None, solvable_by: str | None = None,
                  reference: dict | None = None) -> dict[str, str]:
    """Input files of a validation job, keyed by path relative to /in. The entry file is main.py.

    train_max_len: the candidate's guaranteed training length (describe()["train_max_len"]); when given, no
    training sequence may be longer. solvable_by: "lookup" when the task's answer is determined by an earlier
    occurrence of the last prefix token (the harness then requires that lookup solve every group).
    reference: the design's reference mechanism, {"kind": "pair_memory", "d": int, "min_token": int}; the harness
    then checks the rules against that mechanism's accuracy (falsifier_reference)."""
    code = staged_code()
    config = {"seed": int(seed), "caps": CAPS, "rules": rules, "child_timeout_s": float(child_timeout_s),
              "train_max_len": train_max_len, "solvable_by": solvable_by, "reference": reference}
    return {"task.py": program, "harness.json": json.dumps(config, indent=2), **code}


CODE_FILES = {"main.py": "main.py", "falsifier_eval.py": "code/hybrid/bench/falsifier_eval.py",
              "metrics.py": "code/hybrid/bench/metrics.py",
              "falsifier_reference.py": "code/hybrid/bench/falsifier_reference.py"}


def code_hashes(files: dict[str, str]) -> dict[str, str]:
    """sha256 of the harness and evaluator code staged in a validation job (harness_files output). Recorded with
    every validation attempt: an edit to this code between two attempts can change the outcome."""
    return {name: hashlib.sha256(files[path].encode("utf-8")).hexdigest() for name, path in CODE_FILES.items()}


def current_code_hashes() -> dict[str, str]:
    return code_hashes(staged_code())


def parse_report(stdout: str, status: str, detail: str) -> dict:
    """The validation report printed by the harness (last JSON line with an "ok" key), or a failed report
    describing the job when it printed none (timeout, crash, memory cap)."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                report = json.loads(line)
            except ValueError:
                continue
            if isinstance(report, dict) and "ok" in report and isinstance(report.get("problems"), list):
                return report
    return {"ok": False, "problems": ["validation job %s %s printed no report; stdout tail: %s"
                                      % (status, detail, stdout[-800:])]}


def run_validation_job(executor, image: str, program: str, seed: int, rules: list[dict], jobs_dir: Path, caps,
                       child_timeout_s: float, *, train_max_len: int | None = None, solvable_by: str | None = None,
                       reference: dict | None = None):
    """Run the harness on a task program through an executor. Returns (report, job output directory).
    The research loop runs the same files through its CodeRunner; this is the bench-side entry point."""
    from hybrid.executor.job import Job

    job_id = "job-" + uuid.uuid4().hex[:12]
    job_in, job_out = Path(jobs_dir) / job_id / "in", Path(jobs_dir) / job_id / "out"
    files = harness_files(program, seed, rules, child_timeout_s, train_max_len=train_max_len, solvable_by=solvable_by,
                          reference=reference)
    for name, text in files.items():
        (job_in / name).parent.mkdir(parents=True, exist_ok=True)
        (job_in / name).write_text(text, encoding="utf-8")
    job_out.mkdir(parents=True)
    receipt = executor.run(Job(job_id, image, job_in, job_out, ("python3", "/in/main.py"), caps))
    stdout = Path(receipt.stdout_path).read_text(encoding="utf-8", errors="replace")
    report = parse_report(stdout, receipt.status, receipt.detail)
    return dict(report, code_sha256=code_hashes(files)), job_out


@dataclass(frozen=True)
class FalsifierTask:
    task_id: str
    directory: Path
    design: dict             # task.json
    vocab_size: int
    oracle_kind: str
    group_labels: list
    n_prefixes: int
    n_train: int
    train_tokens: int
    max_train_len: int = 0


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_task(directory: Path) -> FalsifierTask:
    """A validated task directory, checked for internal consistency (it was produced in the sandbox)."""
    directory = Path(directory)
    for name in TASK_FILES:
        if not (directory / name).is_file():
            raise ValueError("falsifier task %s has no %s" % (directory, name))
    design = _read_json(directory / "task.json")
    train = _read_json(directory / "train.json")["sequences"]
    prefixes = _read_json(directory / "prefixes.json")["prefixes"]
    with np.load(directory / "oracle.npz", allow_pickle=False) as data:
        oracle, group, unigram = data["oracle"], data["group"], data["unigram"]
    vocab, kind, labels = design["vocab_size"], design["oracle_kind"], design["group_labels"]
    n = len(prefixes)
    if group.shape != (n,) or unigram.shape != (vocab,):
        raise ValueError("falsifier task %s: group or unigram shape does not match" % directory)
    if (kind == "answer" and oracle.shape != (n,)) or (kind == "dist" and oracle.shape != (n, vocab)):
        raise ValueError("falsifier task %s: oracle shape %s" % (directory, oracle.shape))
    if group.min() < 0 or group.max() >= len(labels):
        raise ValueError("falsifier task %s: group index out of range" % directory)
    return FalsifierTask(design["task_id"], directory, design, vocab, kind, labels, n, len(train),
                         int(sum(len(s) for s in train)), max(len(s) for s in train))


def write_task(destination: Path, job_out: Path, design: dict, report: dict) -> FalsifierTask:
    """Copy a passed validation job's outputs into a task directory and add the unigram reference."""
    destination = Path(destination)
    destination.mkdir(parents=True)
    meta = _read_json(job_out / "meta.json")
    for name in ("train.json", "prefixes.json"):
        shutil.copyfile(job_out / name, destination / name)
    train = _read_json(destination / "train.json")["sequences"]
    with np.load(job_out / "oracle.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("oracle", "group", "next_tokens")}
    np.savez(destination / "oracle.npz", unigram=unigram_dist(train, meta["vocab_size"]), **arrays)
    record = dict(design, vocab_size=meta["vocab_size"], oracle_kind=meta["oracle_kind"],
                  group_labels=meta["group_labels"], validation=report)
    (destination / "task.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return load_task(destination)


def stage_inputs(task: FalsifierTask, destination: Path, seed: int, budget: dict) -> None:
    """Candidate-job inputs for one task: spec.json and the data. No oracle file is ever written here."""
    destination.mkdir(parents=True)
    (destination / "data").mkdir()
    for name in ("train.json", "prefixes.json"):
        shutil.copyfile(task.directory / name, destination / "data" / name)
    spec = {"seed": seed, "budget": budget, "task_id": task.task_id, "vocab_size": task.vocab_size,
            "train_file": "data/train.json", "prefix_file": "data/prefixes.json"}
    (destination / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True))


def eval_entry(task: FalsifierTask) -> dict:
    """What the evaluator job needs to know about a task besides its oracle file."""
    keep = ("title", "rules", "protocol", "deviations")
    return {**{key: task.design[key] for key in keep}, "vocab_size": task.vocab_size, "oracle_kind": task.oracle_kind,
            "group_labels": task.group_labels, "n_prefixes": task.n_prefixes, "n_train": task.n_train,
            "train_tokens": task.train_tokens, "max_train_len": task.max_train_len,
            "protocol_params": task.design["protocol_params"]}
