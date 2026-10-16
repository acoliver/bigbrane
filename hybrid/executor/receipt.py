"""Execution receipt and exit classification."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

OK = "ok"
CANDIDATE_ERROR = "candidate_error"
TIMEOUT = "timeout"
OOM = "oom"
PIDS = "pids"
INFRA_ERROR = "infra_error"
STATUSES = (OK, CANDIDATE_ERROR, TIMEOUT, OOM, PIDS, INFRA_ERROR)

# Reserved by hybrid/executor/supervisor.py.
SUPERVISOR_ERROR_EXIT = 253
PIDS_LIMIT_EXIT = 254


def classify(*, timed_out, oom_killed, started, runtime_error, exit_code):
    """Map host-observed facts to one status. Order is significant: a host wall kill
    or a kernel OOM kill explains the exit code that follows."""
    if not started or runtime_error:
        return INFRA_ERROR
    if timed_out:
        return TIMEOUT
    if oom_killed:
        return OOM
    if exit_code == SUPERVISOR_ERROR_EXIT:
        return INFRA_ERROR
    if exit_code == PIDS_LIMIT_EXIT:
        return PIDS
    if exit_code == 0:
        return OK
    return CANDIDATE_ERROR


@dataclass
class Receipt:
    job_id: str
    run_id: str
    status: str
    detail: str
    exit_code: int
    image: str
    image_digest: str
    image_id: str
    image_architecture: str
    container_name: str
    started_utc: str
    finished_utc: str
    stage_in_s: float
    run_s: float
    stage_out_s: float
    caps: dict
    command: list
    create_argv: list
    effective_policy: dict
    mounts: list
    stdout_path: str
    stderr_path: str
    stdout_bytes: int
    stderr_bytes: int
    outputs: dict = field(default_factory=dict)
    rejected_outputs: list = field(default_factory=list)
    cleanup_errors: list = field(default_factory=list)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError("unknown status %r" % (self.status,))

    def write(self, path):
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path
