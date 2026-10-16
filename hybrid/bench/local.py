"""LocalExecutor: runs a Job as a host subprocess without Podman. Development and tests only.

No isolation of any kind: the job can read the host filesystem, including oracle files, and
has network access. Results produced through it are labeled trusted: false. Commands are
written for the container layout, so "/in" and "/out" (and paths under them) are rewritten
to the job's input and output directories and "python3" to the host interpreter.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LocalReceipt:
    job_id: str
    run_id: str
    status: str
    detail: str
    exit_code: int
    run_s: float
    command: list
    stdout_path: str
    stderr_path: str
    started_utc: str
    outputs: dict = field(default_factory=dict)
    trusted: bool = False
    image: str = "local"
    image_digest: str = "local"

    def write(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


def _map(arg, job):
    if arg == "python3":
        return sys.executable
    for mount, target in (("/in", job.input_dir), ("/out", job.output_dir)):
        if arg == mount:
            return str(target)
        if arg.startswith(mount + "/"):
            return str(target / arg[len(mount) + 1:])
    return arg


def output_hashes(output_dir):
    out = {}
    for path in sorted(Path(output_dir).rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            out[path.relative_to(output_dir).as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(),
                                                            "bytes": len(data)}
    return out


class LocalExecutor:
    trusted = False

    def __init__(self, run_id, receipts_dir):
        self.run_id = run_id
        self.receipts_dir = Path(receipts_dir)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)

    def run(self, job):
        argv = [_map(a, job) for a in job.command]
        base = self.receipts_dir / job.job_id
        stdout, stderr = base.with_suffix(".stdout"), base.with_suffix(".stderr")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
        begin = time.monotonic()
        with open(stdout, "wb") as out, open(stderr, "wb") as err:
            try:
                done = subprocess.run(argv, cwd=job.output_dir, stdout=out, stderr=err, env=env,
                                      timeout=job.caps.wall_s)
                code = done.returncode
                status, detail = ("ok", "") if code == 0 else ("candidate_error", "exit %d" % code)
            except subprocess.TimeoutExpired:
                code, status, detail = -1, "timeout", "wall cap %ss exceeded" % job.caps.wall_s
        receipt = LocalReceipt(job_id=job.job_id, run_id=self.run_id, status=status, detail=detail, exit_code=code,
                               run_s=time.monotonic() - begin, command=argv, stdout_path=str(stdout),
                               stderr_path=str(stderr), started_utc=started, outputs=output_hashes(job.output_dir))
        receipt.write(base.with_suffix(".receipt.json"))
        return receipt
