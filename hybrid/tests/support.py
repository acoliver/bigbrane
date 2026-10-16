"""Shared helpers for the P0 integration tests. Needs a running Podman VM and a built image."""
import json
import os
import shutil
from pathlib import Path

from hybrid.executor.executor import Executor
from hybrid.executor.job import Caps, Job
from hybrid.executor.lock import read_lock
from hybrid.executor.podman import Podman

ROOT = Path(__file__).resolve().parents[2]
PROBES = ROOT / "hybrid" / "probes"
FIXTURES = ROOT / "hybrid" / "fixtures"
RUN_ID = os.environ["BIGBRANE_P0_RUN"]
EVIDENCE = ROOT / "results" / RUN_ID / "probes"
LOCK = read_lock()
PODMAN = Podman()
EXECUTOR = Executor(RUN_ID, EVIDENCE / "receipts", PODMAN)
DEFAULT_CAPS = Caps(wall_s=120, memory_mb=512, pids=128)


def run_job(job_id, command, files=None, caps=DEFAULT_CAPS, executor=EXECUTOR):
    """Stage `files` (name -> bytes or Path) as /in, run command, return (receipt, output dir)."""
    job_dir = EVIDENCE / job_id
    shutil.rmtree(job_dir, ignore_errors=True)
    inputs, outputs = job_dir / "in", job_dir / "out"
    inputs.mkdir(parents=True)
    outputs.mkdir()
    for name, content in (files or {}).items():
        if isinstance(content, Path):
            shutil.copy(content, inputs / name)
        else:
            (inputs / name).write_bytes(content)
    job = Job(job_id, LOCK["reference"], inputs, outputs, command, caps)
    return executor.run(job), outputs


def run_probe(job_id, probe, args=(), extra_files=None, caps=DEFAULT_CAPS):
    files = {probe: PROBES / probe, "seed.txt": b"seed-data", **(extra_files or {})}
    receipt, outputs = run_job(job_id, ("python3", "/in/" + probe, *args), files, caps)
    result_path = outputs / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    return receipt, result, outputs


def labeled(kind, run_id=RUN_ID):
    argv = {"containers": ["ps", "-a", "--format", "{{.Names}}"],
            "volumes": ["volume", "ls", "--format", "{{.Name}}"],
            "pods": ["pod", "ls", "--format", "{{.Name}}"]}[kind]
    out = PODMAN.run([*argv, "--filter", "label=bigbrane.owner=bigbrane", "--filter", "label=bigbrane.run=" + run_id]).stdout
    return out.split()
