"""Host-wide slots: a counting semaphore shared by every Bigbrane process on this machine.

A semaphore is a set of slot files ``<directory>/<name>.<i>.lock``. Holding a slot is holding an exclusive
``fcntl.flock`` on its file, so the operating system releases it when the holder exits or crashes, and two processes
(or two threads of one process, each with its own open file) never hold the same slot. A holder that needs several
slots takes them all or none and polls again, so no holder waits while holding part of what it needs.

Executor jobs (hybrid/executor/executor.py) take ``job_weight(caps)`` slots of the ``job`` semaphore: one per
SLOT_MB of their memory cap. The limit is ``[math] max_parallel_jobs`` of run.toml ("auto": half the Podman VM's
CPUs), clamped by ``job_slot_limit`` to the VM's CPUs minus one and to its memory less VM_RESERVE_MB in SLOT_MB
units, so the jobs that run at once fit in the VM. Every job keeps its own caps (1 CPU, its memory, its wall cap);
the wall cap starts when the job starts, after its slots are held.

The LLM client uses the same semaphores for its lanes (hybrid/llm/lanes.py).

The directory lies in the workspace (``var/slots``), never under /tmp. Processes that use different limits for the
same semaphore share the slots up to the larger limit.
"""
from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import math
import os
import subprocess
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SLOTS_DIR = ROOT / "var" / "slots"
JOB_SEMAPHORE = "job"
# One job slot is this much memory: the library and learnability cap (hybrid/math/runner.py LIBRARY_CAPS).
SLOT_MB = 4096
# VM memory kept back for the VM itself, podman and the stage-in and stage-out containers of running jobs.
VM_RESERVE_MB = 4096
MACHINE = "podman-machine-default"
POLL_S = 0.5


class SlotLimitError(ValueError):
    """The VM cannot run even one job under the caps (nothing is started)."""


class VMUnavailable(RuntimeError):
    """``podman machine inspect`` did not report the VM's size (nothing is started)."""


class FileSemaphore:
    """``limit`` slots named ``name`` under ``directory``; see the module docstring."""

    def __init__(self, directory: Path, name: str, limit: int, *, poll_s: float = POLL_S):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"semaphore {name}: limit must be an integer >= 1, got {limit!r}")
        self.directory = Path(directory)
        self.name = name
        self.limit = limit
        self.poll_s = poll_s
        self.directory.mkdir(parents=True, exist_ok=True)

    def slot_path(self, index: int) -> Path:
        return self.directory / f"{self.name}.{index}.lock"

    def try_acquire(self, weight: int = 1) -> list[int] | None:
        """Lock ``weight`` free slots and return their file descriptors, or lock nothing and return None."""
        if weight < 1 or weight > self.limit:
            raise ValueError(f"semaphore {self.name}: weight {weight} outside 1..{self.limit}")
        held: list[int] = []
        for index in range(self.limit):
            fd = os.open(self.slot_path(index), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    continue
                self.release(held)
                raise
            held.append(fd)
            if len(held) == weight:
                return held
        self.release(held)
        return None

    @staticmethod
    def release(fds: list[int]) -> None:
        for fd in fds:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def acquire(self, weight: int = 1, *, sleep: Callable[[float], None] = time.sleep) -> list[int]:
        """Block (polling) until ``weight`` slots are held."""
        while True:
            held = self.try_acquire(weight)
            if held is not None:
                return held
            sleep(self.poll_s)

    async def acquire_async(self, weight: int = 1) -> list[int]:
        while True:
            held = self.try_acquire(weight)
            if held is not None:
                return held
            await asyncio.sleep(self.poll_s)

    @contextmanager
    def hold(self, weight: int = 1):
        held = self.acquire(weight)
        try:
            yield
        finally:
            self.release(held)

    @asynccontextmanager
    async def hold_async(self, weight: int = 1):
        held = await self.acquire_async(weight)
        try:
            yield
        finally:
            self.release(held)

    def in_use(self) -> int:
        """Slots held right now by anyone (a probe: each free slot is locked and released at once)."""
        busy = 0
        for index in range(self.limit):
            fd = os.open(self.slot_path(index), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
                busy += 1
            finally:
                os.close(fd)
        return busy


# ----- executor jobs ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VMResources:
    cpus: int
    memory_mb: int


def vm_resources(run: Callable = subprocess.run, machine: str = MACHINE) -> VMResources:
    """The Podman VM's CPUs and memory from ``podman machine inspect`` (external output, checked)."""
    argv = ["podman", "machine", "inspect", machine]
    done = run(argv, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise VMUnavailable(f"cannot read the Podman VM size ({' '.join(argv)}): exit {done.returncode}: "
                           f"{done.stderr.strip()[:400]}")
    try:
        resources = json.loads(done.stdout)[0]["Resources"]
        cpus, memory = int(resources["CPUs"]), int(resources["Memory"])
    except (ValueError, LookupError, TypeError) as exc:
        raise VMUnavailable(f"unexpected output from {' '.join(argv)}: {done.stdout.strip()[:400]!r}") from exc
    if cpus < 1 or memory < 1:
        raise VMUnavailable(f"{' '.join(argv)} reports {cpus} CPUs and {memory} MiB")
    return VMResources(cpus, memory)


def job_slot_limit(requested: int | str, vm: VMResources) -> dict:
    """The job slot limit for ``[math] max_parallel_jobs`` on this VM, with what bounded it.

    "auto" asks for half the VM's CPUs. The limit is at most the VM's CPUs minus one (each job has a 1-CPU cap and
    the VM keeps one for podman and the stagers) and at most the VM memory less VM_RESERVE_MB in SLOT_MB slots (every
    slot is one job's 4096 MB cap). Raises SlotLimitError when not even one slot fits."""
    wanted = max(1, vm.cpus // 2) if requested == "auto" else requested
    if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted < 1:
        raise ValueError(f"math.max_parallel_jobs must be an integer >= 1 or \"auto\", got {requested!r}")
    cpu_cap = vm.cpus - 1
    memory_cap = (vm.memory_mb - VM_RESERVE_MB) // SLOT_MB
    limit = min(wanted, cpu_cap, memory_cap)
    if limit < 1:
        raise SlotLimitError(f"the Podman VM ({vm.cpus} CPUs, {vm.memory_mb} MiB) cannot run one {SLOT_MB} MB job "
                             f"with {VM_RESERVE_MB} MiB and one CPU kept back")
    clamped = []
    if wanted > cpu_cap:
        clamped.append(f"{wanted} jobs need more than the VM's {vm.cpus} CPUs less one")
    if wanted > memory_cap:
        clamped.append(f"{wanted} jobs of {SLOT_MB} MB need {wanted * SLOT_MB} MiB, above the VM's "
                       f"{vm.memory_mb} MiB less {VM_RESERVE_MB} MiB reserve")
    return {"requested": requested, "wanted": wanted, "limit": limit, "clamped": clamped,
            "vm": {"cpus": vm.cpus, "memory_mb": vm.memory_mb}, "slot_mb": SLOT_MB, "reserve_mb": VM_RESERVE_MB}


def job_weight(memory_mb: int, limit: int) -> int:
    """Slots a job with this memory cap takes: one per SLOT_MB, at most the limit (a job larger than every slot
    together still runs alone; the limit was set so that this much memory fits in the VM)."""
    return max(1, min(limit, math.ceil(memory_mb / SLOT_MB)))


class JobSlots:
    """The host-wide ``job`` semaphore as the executor uses it."""

    def __init__(self, limit: int, directory: Path = SLOTS_DIR, *, poll_s: float = POLL_S):
        self.semaphore = FileSemaphore(directory, JOB_SEMAPHORE, limit, poll_s=poll_s)

    @property
    def limit(self) -> int:
        return self.semaphore.limit

    def weight(self, memory_mb: int) -> int:
        return job_weight(memory_mb, self.limit)

    @contextmanager
    def hold(self, memory_mb: int):
        with self.semaphore.hold(self.weight(memory_mb)):
            yield


def job_slots_from_config(cfg: dict, *, run: Callable = subprocess.run,
                          directory: Path = SLOTS_DIR) -> tuple[JobSlots, dict]:
    """The job slots for a run config, sized against the Podman VM, and the record of how the limit was set."""
    record = job_slot_limit(cfg["math"].get("max_parallel_jobs", "auto"), vm_resources(run))
    record["directory"] = str(directory)
    return JobSlots(record["limit"], directory), record


def slot_line(record: dict) -> str:
    """The job slot settings in one line, for a command's first output."""
    vm = record["vm"]
    line = (f"job slots: {record['limit']} of {record['slot_mb']} MB under {record.get('directory', SLOTS_DIR)} "
            f"(max_parallel_jobs {record['requested']}; Podman VM {vm['cpus']} CPUs, {vm['memory_mb']} MiB)")
    if record["clamped"]:
        line += "; clamped: " + "; ".join(record["clamped"])
    return line
