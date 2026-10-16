"""Host MPS backend for the test ladder: trains the trusted rung 1 baselines on the Mac's GPU (Apple MPS).

Scope. Andrew approved host GPU training on 2026-10-13 for calibration of our own baselines only. Candidate code
from the research loop never runs here: it stays in the executor container. The backend enforces that by
refusing any job whose staged files differ from the repository's trusted ladder files (the runner, bb_recall and
the task modules, the same files `ladder.code_hash` covers), and nothing outside hybrid/bench/ladder.py builds
these jobs.

Each job runs as one subprocess of the host venv's Python under `sandbox-exec` with a profile that denies all
network access and denies file writes outside the job's output directory, its private temporary directory and
the per-user Darwin cache and temporary directories that Metal and MPSGraph use for compiled shaders. The job
inherits no PYTORCH_ENABLE_MPS_FALLBACK, and the runner refuses to start if it is set, so an op without an MPS
kernel fails the job instead of running on CPU. Memory and process caps are not enforced on the host; the wall cap
is.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hybrid.executor.receipt import OK, TIMEOUT, classify

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
MODE, DEVICE = "host_mps", "mps"
PROBE = ("import json, platform, sys, torch; "
         "print(json.dumps({'torch': torch.__version__, 'python': sys.version.split()[0], "
         "'platform': platform.platform(), 'mps_available': torch.backends.mps.is_available()}))")


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _darwin_dir(name):
    out = subprocess.run(["/usr/bin/getconf", name], check=True, capture_output=True, text=True).stdout.strip()
    if not out:
        raise RuntimeError("getconf %s returned nothing" % name)
    return os.path.realpath(out)


def _quote(path):
    text = str(path)
    if '"' in text or "\\" in text:
        raise ValueError("path not usable in a sandbox profile: %r" % text)
    return '"%s"' % text


def sandbox_profile(writable):
    """Seatbelt profile: everything the default allows, minus network and minus writes outside `writable`."""
    paths = " ".join("(subpath %s)" % _quote(p) for p in writable)
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            "(allow file-write* %s (literal \"/dev/null\") (literal \"/dev/dtracehelper\"))" % paths)


@dataclass
class HostReceipt:
    """What ladder.execute reads from a receipt, plus the host-specific provenance of one job."""
    job_id: str
    run_id: str
    status: str
    detail: str
    exit_code: int
    started_utc: str
    finished_utc: str
    run_s: float
    command: list
    sandbox_profile: str
    stdout_path: str
    stderr_path: str
    device: str = DEVICE
    image_digest: None = None
    caps: dict = field(default_factory=dict)

    def write(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


class HostMpsBackend:
    """Runs one ladder job on the host GPU. `trusted` maps job-relative paths (ladder_runner.py, code/<file>) to
    their SHA-256; a job must stage exactly those files with exactly those contents."""

    mode, device = MODE, DEVICE

    def __init__(self, python, receipts_dir, trusted, code_hash, label):
        self.python = Path(python)
        if not self.python.exists():
            raise FileNotFoundError("host python %s does not exist" % self.python)
        if not Path(SANDBOX_EXEC).exists():
            raise FileNotFoundError("%s is missing; host_mps needs it to deny network access" % SANDBOX_EXEC)
        self.receipts_dir = Path(receipts_dir)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.trusted, self.code_hash, self.label = dict(trusted), code_hash, label
        self.darwin_dirs = [_darwin_dir("DARWIN_USER_CACHE_DIR"), _darwin_dir("DARWIN_USER_TEMP_DIR")]
        self.host = self._probe_host()
        if not self.host["mps_available"]:
            raise RuntimeError("torch.backends.mps.is_available() is False on this host")

    def _env(self, tmp_dir):
        return {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], "TMPDIR": str(tmp_dir),
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def _probe_host(self):
        probe_dir = self.receipts_dir / "host-probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        argv = [SANDBOX_EXEC, "-p", sandbox_profile([probe_dir, *self.darwin_dirs]), str(self.python), "-I", "-c",
                PROBE]
        out = subprocess.run(argv, check=True, capture_output=True, text=True, env=self._env(probe_dir), timeout=300)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def provenance(self):
        """Recorded in every result this backend produces."""
        return {"mode": MODE, "device": DEVICE, "dtype": "float32", "torch": self.host["torch"],
                "python": self.host["python"], "platform": self.host["platform"], "python_path": str(self.python),
                "code_hash": self.code_hash, "backend_sha256": _sha256(__file__), "network": "denied (sandbox-exec)",
                "cpu_fallback": "disabled", "caps_enforced": ["wall_s"]}

    def check_trusted(self, job_in):
        staged = {str(p.relative_to(job_in)): _sha256(p)
                  for p in [job_in / "ladder_runner.py", *sorted((job_in / "code").glob("*"))]}
        if staged != self.trusted:
            raise PermissionError("host_mps runs only the trusted ladder baselines; staged files %s differ from %s"
                                  % (sorted(staged), sorted(self.trusted)))

    def run(self, job_id, job_in, job_out, caps):
        job_in, job_out = Path(job_in).resolve(), Path(job_out).resolve()
        self.check_trusted(job_in)
        tmp_dir = job_out.parent / "tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        profile = sandbox_profile([job_out, tmp_dir, *self.darwin_dirs])
        argv = [SANDBOX_EXEC, "-p", profile, str(self.python), "-I", str(job_in / "ladder_runner.py"), str(job_in),
                str(job_out)]
        base = self.receipts_dir / job_id
        stdout_path, stderr_path = base.with_suffix(".stdout"), base.with_suffix(".stderr")
        started = _utc()
        begin = time.monotonic()
        timed_out = False
        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            proc = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=self._env(tmp_dir))
            try:
                proc.wait(timeout=caps.wall_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                proc.wait()
        run_s = time.monotonic() - begin
        shutil.rmtree(tmp_dir)
        status = classify(timed_out=timed_out, oom_killed=False, started=True, runtime_error="",
                          exit_code=proc.returncode)
        detail = "wall cap %ss exceeded" % caps.wall_s if status == TIMEOUT else (
            "" if status == OK else "exit %d; see %s" % (proc.returncode, stderr_path))
        receipt = HostReceipt(job_id=job_id, run_id=self.label, status=status, detail=detail,
                              exit_code=proc.returncode, started_utc=started, finished_utc=_utc(), run_s=run_s,
                              command=argv[:2] + ["<profile>"] + argv[3:], sandbox_profile=profile,
                              stdout_path=str(stdout_path), stderr_path=str(stderr_path),
                              caps={"wall_s": caps.wall_s})
        receipt.write(base.with_suffix(".receipt.json"))
        return receipt
