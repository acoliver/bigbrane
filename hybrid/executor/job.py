"""Job description handed to the executor. Invalid jobs fail at construction."""
import re
from dataclasses import dataclass
from pathlib import Path

JOB_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}")
DIGEST_REF = re.compile(r"[a-z0-9][a-zA-Z0-9_./:-]*@sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class Caps:
    """Resource caps of one job. The executor enforces ``wall_s`` (it kills the container), memory, pids and cpus.
    ``cpu_s`` (decision CT) is the job's budget in process CPU seconds: the job enforces it on itself (the math
    library driver sets RLIMIT_CPU), so its verdict does not depend on how busy the host is. The executor only records
    it; with ``cpu_s`` set, ``wall_s`` is a hang guard, never a budget, and must be at least ``cpu_s``."""
    wall_s: float
    memory_mb: int
    pids: int
    cpus: float = 1.0
    tmp_mb: int = 64
    max_log_bytes: int = 1 << 20
    max_output_bytes: int = 64 << 20
    cpu_s: float | None = None

    def __post_init__(self):
        for name in ("wall_s", "memory_mb", "pids", "cpus", "tmp_mb", "max_log_bytes", "max_output_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError("cap %s must be positive, got %r" % (name, getattr(self, name)))
        if self.tmp_mb >= self.memory_mb:
            raise ValueError("tmpfs counts against the memory cap, so tmp_mb must be below memory_mb")
        if self.cpu_s is not None:
            if self.cpu_s <= 0:
                raise ValueError("cap cpu_s must be positive, got %r" % (self.cpu_s,))
            if self.wall_s < self.cpu_s:
                raise ValueError("the wall cap is the hang guard of a CPU-capped job, so wall_s must be at least cpu_s")


@dataclass(frozen=True)
class Job:
    job_id: str
    image: str
    input_dir: Path
    output_dir: Path
    command: tuple
    caps: Caps

    def __post_init__(self):
        if not JOB_ID.fullmatch(self.job_id):
            raise ValueError("bad job_id %r" % (self.job_id,))
        if not DIGEST_REF.fullmatch(self.image):
            raise ValueError("image must be pinned by digest (name@sha256:<64 hex>), got %r" % (self.image,))
        object.__setattr__(self, "input_dir", Path(self.input_dir).resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        object.__setattr__(self, "command", tuple(self.command))
        if not self.input_dir.is_dir():
            raise ValueError("input_dir is not a directory: %s" % self.input_dir)
        if not self.output_dir.is_dir() or any(self.output_dir.iterdir()):
            raise ValueError("output_dir must be an existing empty directory: %s" % self.output_dir)
        if not self.command or not all(isinstance(part, str) and part for part in self.command):
            raise ValueError("command must be a non-empty list of non-empty strings")
