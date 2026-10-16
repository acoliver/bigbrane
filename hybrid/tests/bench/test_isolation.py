"""Path audit: a candidate job never receives oracle data and cannot reach it.

The static test inspects every candidate job's input directory. The Podman test runs a
snooping candidate that searches the container filesystem and probes the host paths of the
oracle files, then reports what it found through its own output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hybrid.bench import config as bench_config
from hybrid.bench.evaluate import evaluate_candidate
from hybrid.bench.local import LocalExecutor
from hybrid.bench.tasks import build_task

from .conftest import STAMP, TINY, podman


class RecordingExecutor(LocalExecutor):
    def __init__(self, *args):
        super().__init__(*args)
        self.inputs = []

    def run(self, job):
        files = {p.relative_to(job.input_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in job.input_dir.rglob("*") if p.is_file()}
        self.inputs.append((job.command, files, job.input_dir))
        return super().run(job)


def test_candidate_jobs_receive_no_oracle_data(work_dir, template_dir):
    executor = RecordingExecutor("bench-audit", work_dir / "receipts")
    evaluate_candidate(template_dir, TINY, work_dir / "run", executor, baselines=["ngram"],
                       cache_dir=work_dir / "cache", log=lambda _: None)
    oracle_dir = work_dir / "run" / "bench" / "data" / "oracle"
    oracle_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in oracle_dir.iterdir()}
    cfg = bench_config.load_config(TINY)
    candidate_jobs = [x for x in executor.inputs if "/in/runner.py" in x[0]]
    eval_jobs = [x for x in executor.inputs if "/in/runner.py" not in x[0]]
    assert len(candidate_jobs) == 2 and len(eval_jobs) == 1
    allowed = {"runner.py", "spec.json"} | {"data/%s/%s" % (t, f) for t in cfg["tasks"]
                                             for f in ("train.json", "prefixes.json")}
    allowed |= {"data/%s/prefixes_x%d.json" % (t, f) for t in cfg["fingerprint"]["length_tasks"]
                for f in cfg["fingerprint"]["length_factors"]}
    for _, files, input_dir in candidate_jobs:
        assert not set(files.values()) & oracle_hashes
        assert {f for f in files if not f.startswith("candidate/")} == allowed
        assert not any("oracle" in f or f.endswith((".npz", ".npy")) for f in files)
        spec = json.loads((input_dir / "spec.json").read_text())
        assert set(spec) == {"seed", "budget", "tasks", "scaling"}
        for task in spec["tasks"]:
            assert set(task) - {"length_probes"} == {"name", "vocab_size", "train_sizes", "train_file", "prefix_file"}
            assert ("length_probes" in task) == (task["name"] in cfg["fingerprint"]["length_tasks"])
            for probe in task.get("length_probes", []):   # prefixes only; the oracle for them is evaluator-only
                assert set(probe) == {"factor", "prefix_file"}
                longer = json.loads((input_dir / probe["prefix_file"]).read_text())
                assert set(longer) == {"prefixes"} and len(longer["prefixes"]) == cfg["tasks"][task["name"]]["n_test"]
            built = build_task(task["name"], cfg["seed"], max(task["train_sizes"]),
                               cfg["tasks"][task["name"]]["n_test"], cfg["tasks"][task["name"]]["params"])
            given = json.loads((input_dir / task["prefix_file"]).read_text())
            assert given == {"prefixes": built.prefixes}   # prefixes only, nothing past the scored position
    npz_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in oracle_dir.glob("*.npz")}
    n_oracles = 5 + len(cfg["fingerprint"]["length_tasks"]) * len(cfg["fingerprint"]["length_factors"])
    assert len(npz_hashes) == n_oracles and npz_hashes <= set(eval_jobs[0][1].values())


SNOOP = r'''
import json, os
from pathlib import Path
import numpy as np

SUSPECT = ("oracle", ".npz", "meta.json", "bench_result", "eval_spec")


def snoop():
    found, walked, errors = [], 0, 0
    for root, dirs, files in os.walk("/", onerror=lambda e: None):
        if root.startswith(("/proc", "/sys", "/dev")):
            dirs[:] = []
            continue
        for name in files + dirs:
            walked += 1
            if any(s in name for s in SUSPECT) and not root.startswith("/usr/local/lib/python"):
                found.append(os.path.join(root, name))
    targets = json.loads(Path(__file__).with_name("targets.json").read_text())
    reachable = [t for t in targets if os.path.exists(t)]
    readable = []
    for t in targets:
        try:
            with open(t, "rb") as handle:
                handle.read(1)
            readable.append(t)
        except OSError:
            pass
    listing = sorted(str(p.relative_to("/in")) for p in Path("/in").rglob("*"))
    Path("/out/snoop.json").write_text(json.dumps({"found": found, "walked": walked, "reachable": reachable,
                                                   "readable": readable, "in_listing": listing}))


class Candidate:
    done = False

    def __init__(self, vocab_size, seed, budget):
        self.V = vocab_size
        if not Candidate.done:
            snoop()
            Candidate.done = True

    def fit(self, train):
        pass

    def predict_proba(self, prefixes):
        return np.full((len(prefixes), self.V), 1.0 / self.V)

    def describe(self):
        return {"params": 0, "state_floats_per_sequence": 0, "notes": "snoop"}
'''


@podman
def test_snooping_candidate_finds_no_oracle_in_container(work_dir):
    from hybrid.executor.executor import Executor
    run_dir = work_dir / "run"
    oracle_dir = run_dir / "bench" / "data" / "oracle"
    candidate = work_dir / "snoop"
    candidate.mkdir()
    (candidate / "candidate.py").write_text(SNOOP)
    targets = [str(oracle_dir / ("%s.npz" % t)) for t in ("hmm", "vom", "recall", "compose", "fewshot")]
    targets += [str(oracle_dir), str(Path(__file__).resolve()), "/Volumes", str(work_dir)]
    (candidate / "targets.json").write_text(json.dumps(targets))
    executor = Executor("bench-test-" + STAMP.lower(), work_dir / "receipts")
    result = evaluate_candidate(candidate, TINY, run_dir, executor, baselines=[], cache_dir=work_dir / "cache",
                                log=lambda _: None)
    assert result["trusted"] and result["entrants"]["candidate"]["status"] == "ok"
    assert all(Path(t).exists() for t in targets[:5])        # the oracle files exist on the host
    job_id = result["host"]["receipts"]["candidate"]["job_id"]
    report = json.loads((run_dir / "bench" / "jobs" / job_id / "out" / "snoop.json").read_text())
    assert report["walked"] > 1000
    assert report["found"] == [] and report["reachable"] == [] and report["readable"] == []
    assert not any("oracle" in name or name.endswith(".npz") for name in report["in_listing"])
    assert "runner.py" in report["in_listing"]
