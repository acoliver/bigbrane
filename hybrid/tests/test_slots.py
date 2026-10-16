"""Host-wide job slots (hybrid/slots.py): the bound across processes, weights, release on exit, the VM clamp, and the
executor holding its slots for each job."""
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid import slots
from hybrid.executor.executor import Executor
from hybrid.executor.job import Caps


def hold_and_log(directory: str, limit: int, log: str, seconds: float) -> None:
    """Child process: take one slot, append its enter and leave times to ``log``, release."""
    semaphore = slots.FileSemaphore(Path(directory), "job", limit, poll_s=0.01)
    with semaphore.hold():
        entered = time.time()
        time.sleep(seconds)
        left = time.time()
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps([entered, left]) + "\n")


def max_overlap(intervals: list[tuple[float, float]]) -> int:
    points = sorted([(t, 1) for t, _ in intervals] + [(t, -1) for _, t in intervals], key=lambda p: (p[0], p[1]))
    best = current = 0
    for _, step in points:
        current += step
        best = max(best, current)
    return best


def test_the_slot_bound_holds_across_processes(tmp_path):
    log = tmp_path / "intervals.jsonl"
    ctx = multiprocessing.get_context("spawn")
    workers = [ctx.Process(target=hold_and_log, args=(str(tmp_path / "slots"), 2, str(log), 0.4)) for _ in range(6)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    intervals = [tuple(json.loads(line)) for line in log.read_text().splitlines()]
    assert len(intervals) == 6
    # Never more than the limit at once, and the limit is used (the six holders do not run one by one).
    assert max_overlap(intervals) == 2


def test_two_open_files_in_one_process_never_share_a_slot(tmp_path):
    a = slots.FileSemaphore(tmp_path, "job", 1)
    b = slots.FileSemaphore(tmp_path, "job", 1)
    held = a.try_acquire()
    assert held is not None and b.try_acquire() is None and a.in_use() == 1
    a.release(held)
    again = b.try_acquire()
    assert again is not None
    b.release(again)
    assert a.in_use() == 0


def test_a_weighted_holder_takes_all_its_slots_or_none(tmp_path):
    semaphore = slots.FileSemaphore(tmp_path, "job", 3)
    two = semaphore.try_acquire(2)
    assert two is not None and semaphore.in_use() == 2
    assert semaphore.try_acquire(2) is None
    # The failed weight-2 attempt kept nothing: one slot is still free.
    assert semaphore.in_use() == 2
    one = semaphore.try_acquire(1)
    assert one is not None and semaphore.in_use() == 3
    semaphore.release(two + one)
    with pytest.raises(ValueError):
        semaphore.try_acquire(4)


def test_a_slot_is_released_when_its_holder_dies(tmp_path):
    code = ("import os, sys; from pathlib import Path; from hybrid.slots import FileSemaphore; "
            f"s = FileSemaphore(Path({str(tmp_path)!r}), 'job', 1); assert s.try_acquire() is not None; "
            "os._exit(0)")
    done = subprocess.run([sys.executable, "-c", code], cwd=slots.ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    semaphore = slots.FileSemaphore(tmp_path, "job", 1)
    held = semaphore.try_acquire()
    assert held is not None
    semaphore.release(held)


# ----- the VM clamp -----------------------------------------------------------------------------------------------

VM = slots.VMResources(cpus=8, memory_mb=24576)


def test_auto_is_half_the_vm_cpus():
    record = slots.job_slot_limit("auto", VM)
    assert record["limit"] == 4 and record["clamped"] == [] and record["wanted"] == 4


def test_a_request_above_the_vm_memory_is_clamped_with_the_reason():
    record = slots.job_slot_limit(6, VM)
    # (24576 - 4096) / 4096 = 5 slots of 4096 MB; 8 CPUs less one = 7.
    assert record["limit"] == 5
    assert any("need 24576 MiB" in reason for reason in record["clamped"])
    assert not any("CPUs" in reason for reason in record["clamped"])
    record = slots.job_slot_limit(12, slots.VMResources(cpus=4, memory_mb=65536))
    assert record["limit"] == 3 and any("CPUs" in reason for reason in record["clamped"])


def test_a_request_that_fits_is_kept():
    assert slots.job_slot_limit(3, VM)["limit"] == 3
    # The total memory of the jobs at once stays inside the VM after the reserve.
    for wanted in range(1, 20):
        limit = slots.job_slot_limit(wanted, VM)["limit"]
        assert limit * slots.SLOT_MB + slots.VM_RESERVE_MB <= VM.memory_mb and limit <= VM.cpus - 1


def test_a_vm_that_cannot_run_one_job_is_refused():
    with pytest.raises(slots.SlotLimitError, match="cannot run one 4096 MB job"):
        slots.job_slot_limit("auto", slots.VMResources(cpus=4, memory_mb=6000))
    with pytest.raises(slots.SlotLimitError):
        slots.job_slot_limit(2, slots.VMResources(cpus=1, memory_mb=65536))
    with pytest.raises(ValueError, match="max_parallel_jobs"):
        slots.job_slot_limit(0, VM)


def test_job_weights_follow_the_memory_cap():
    assert slots.job_weight(4096, 4) == 1 and slots.job_weight(2048, 4) == 1 and slots.job_weight(1024, 4) == 1
    assert slots.job_weight(8192, 4) == 2          # a Lean job
    assert slots.job_weight(32768, 3) == 3         # larger than every slot together: runs alone


def fake_inspect(stdout: str, code: int = 0):
    def run(argv, **kwargs):
        assert argv[:3] == ["podman", "machine", "inspect"]
        return SimpleNamespace(returncode=code, stdout=stdout, stderr="no machine")
    return run


def test_the_vm_size_is_read_from_podman_machine_inspect(tmp_path):
    out = json.dumps([{"Name": "podman-machine-default", "Resources": {"CPUs": 8, "Memory": 24576}}])
    assert slots.vm_resources(fake_inspect(out)) == VM
    with pytest.raises(slots.VMUnavailable, match="exit 125"):
        slots.vm_resources(fake_inspect("", 125))
    with pytest.raises(slots.VMUnavailable, match="unexpected output"):
        slots.vm_resources(fake_inspect("[{}]"))
    job_slots, record = slots.job_slots_from_config({"math": {"max_parallel_jobs": 8}}, run=fake_inspect(out),
                                                    directory=tmp_path)
    assert job_slots.limit == 5 and record["limit"] == 5 and record["directory"] == str(tmp_path)
    assert "clamped:" in slots.slot_line(record)


# ----- the executor holds its slots ---------------------------------------------------------------------------------

class CountingExecutor(Executor):
    """An executor whose jobs only count how many run at once."""

    def __init__(self, tmp_path, job_slots):
        super().__init__("run", tmp_path / "receipts", podman=object(), slots=job_slots)
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0
        self.ran = []

    def _run(self, job):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.05)
        with self.lock:
            self.now -= 1
            self.ran.append(job.job_id)
        return job.job_id


def test_the_executor_runs_at_most_the_slot_limit_of_jobs(tmp_path):
    executor = CountingExecutor(tmp_path, slots.JobSlots(2, tmp_path / "slots", poll_s=0.005))
    jobs = [SimpleNamespace(job_id=f"j{i}", caps=Caps(wall_s=10, memory_mb=4096, pids=8)) for i in range(8)]
    threads = [threading.Thread(target=executor.run, args=(job,)) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(executor.ran) == sorted(j.job_id for j in jobs)
    assert executor.peak == 2
    # A Lean-sized job takes both slots: it never runs beside another job.
    lean = SimpleNamespace(job_id="lean", caps=Caps(wall_s=10, memory_mb=8192, pids=8))
    small = [SimpleNamespace(job_id=f"s{i}", caps=Caps(wall_s=10, memory_mb=2048, pids=8)) for i in range(3)]
    executor.peak = 0
    threads = [threading.Thread(target=executor.run, args=(job,)) for job in [lean, *small]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert executor.peak <= 2


def test_without_slots_the_executor_runs_the_job_at_once(tmp_path):
    executor = CountingExecutor(tmp_path, None)
    assert executor.run(SimpleNamespace(job_id="x", caps=None)) == "x"
