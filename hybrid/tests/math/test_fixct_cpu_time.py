"""Decision CT (2026-10-17): learnability admission and library job caps use process CPU time, never wall time.

In calibration 20261017T030854Z (runs/calibration-math-20261017T030854Z/k2-run1) the wall-clock timing probe of K2's
recall and composition learners projected 3341 s and 3623 s against 1440 s admitted while the host ran at a load of 25
to 65; on 2026-10-15 the same kind of job projected 1007 s and passed. These tests pin the fix:

* a probe whose wall time is inflated (a fake wall clock, or real sleeps that cost no CPU) projects the same CPU time,
  and the runner admits it the same way;
* the job driver's RLIMIT_CPU cap stops a learner that burns CPU and reports a cap stop, which the runner records as
  a resource failure;
* the wall cap is a hang guard: its stop raises JobHangGuard (test_fix2_learnability_budget.py and
  hybrid/tests/research/test_fixct_hang_guard.py cover the runner and the stage).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid.executor.job import Caps
from hybrid.math import learnability, runner
from hybrid.math.runner import LIBRARY_CAPS, MathRunner, cpu_capped
from hybrid.tests.research import calibration_fakes as cf

IMAGE = "localhost/bigbrane/tools@sha256:" + "c" * 64
SPEC = {"task": "copy", "size": 3, "examples": 400, "seeds": [0, 1, 2], "test_count": 40}
BURNER = ('RECIPE = {"epochs": "until stopped"}\n'
          "def learn(layout, examples, seed):\n"
          "    total = 0\n"
          "    while True:\n"
          "        total = (total + 1) % 1000003\n"
          "def init(layout, params):\n    return {}\n"
          "def update(state, token):\n    return state\n"
          "def predict(state):\n    return 0\n")


class StepClocks:
    """Fake clocks: every reading advances the CPU clock by 1 s and the wall clock by ``stretch`` s, as a process that
    gets 1 / stretch of a core would see them."""

    def __init__(self, stretch: float):
        self.cpu = self.wall = 0.0
        self.stretch = stretch

    def cpu_now(self) -> float:
        self.cpu += 1.0
        return self.cpu

    def wall_now(self) -> float:
        self.wall += self.stretch
        return self.wall


def probe_with_clocks(monkeypatch, stretch: float) -> dict:
    clocks = StepClocks(stretch)
    monkeypatch.setattr(learnability, "_cpu_now", clocks.cpu_now)
    monkeypatch.setattr(learnability, "_wall_now", clocks.wall_now)
    return learnability.timing_probe(SPEC, cf.LEARNERS["copy"])


def test_a_fake_wall_clock_running_four_times_slower_leaves_the_cpu_projection_unchanged(monkeypatch):
    idle = probe_with_clocks(monkeypatch, 1.0)
    loaded = probe_with_clocks(monkeypatch, 4.0)
    assert loaded["projected_cpu_s"] == idle["projected_cpu_s"] > 0
    assert loaded["timing"]["phases_cpu"] == idle["timing"]["phases_cpu"]
    assert loaded["timing"]["total_wall_s"] == pytest.approx(4 * idle["timing"]["total_wall_s"])
    assert loaded["timing"]["phases_wall"]["train"] == pytest.approx(4 * idle["timing"]["phases_wall"]["train"])
    assert loaded["timing"]["clock"] == "process_cpu" and "CPU s" in loaded["evidence"]


def sleepy(monkeypatch, seconds: float) -> None:
    """Every training of the learner first sleeps ``seconds``: wall time passes, the process uses no CPU, as when the
    host gives the job's core to other work."""
    original = learnability.load_learner

    def load(source):
        namespace = original(source)
        learn = namespace["learn"]

        def slowed(*args):
            time.sleep(seconds)
            return learn(*args)
        namespace["learn"] = slowed
        return namespace
    monkeypatch.setattr(learnability, "load_learner", load)


def test_a_probe_under_real_contention_projects_the_same_cpu_time(monkeypatch):
    idle = learnability.timing_probe(SPEC, cf.LEARNERS["copy"])
    sleepy(monkeypatch, 0.4)
    loaded = learnability.timing_probe(SPEC, cf.LEARNERS["copy"])
    # Three trainings each slept 0.4 s: at least 1.2 s more wall time, about none more CPU time.
    assert loaded["timing"]["total_wall_s"] >= idle["timing"]["total_wall_s"] + 1.15
    assert loaded["timing"]["phases_wall"]["train"] >= 0.4
    assert loaded["timing"]["phases_cpu"]["train"] < 0.4
    a, b = idle["projected_cpu_s"], loaded["projected_cpu_s"]
    assert abs(a - b) <= 0.3 * max(a, b) + 0.05, (a, b)
    # The wall clock would have projected far more: the scaled sleeps alone add 3 seeds x (0.4 + 0.4) x 400/300 s.
    wall_projection = learnability.projected_cpu_seconds(
        SPEC, dict(loaded["timing"], phases_cpu=loaded["timing"]["phases_wall"]))
    assert wall_projection > b + 3.0


class PrintedProbe:
    """An executor that answers the probe with a recorded probe report (as the job prints it) and runs nothing
    else."""

    def __init__(self, tmp_path: Path, printed: dict):
        self.tmp_path, self.printed, self.jobs = Path(tmp_path), printed, []
        self.tmp_path.mkdir(parents=True, exist_ok=True)

    def run(self, job):
        self.jobs.append(job)
        request = json.loads((Path(job.input_dir) / "request.json").read_text())
        assert request["kind"] == "learnability_probe", "only the probe runs in these tests"
        out, err = self.tmp_path / f"{job.job_id}.stdout", self.tmp_path / f"{job.job_id}.stderr"
        clock = {"cpu_s": self.printed["timing"]["total_cpu_s"] + 0.2, "wall_s": 99.0,
                 "cap_cpu_s": request["cpu_cap_s"]}
        out.write_text(json.dumps(dict(self.printed, job_clock=clock)) + "\n")
        err.write_text("")
        # A busy host: the receipt's wall time is far above the job's CPU time. It must not matter.
        return SimpleNamespace(status="ok", detail="", exit_code=0, run_s=99.0, stdout_path=str(out),
                               stderr_path=str(err))


def test_admission_is_the_same_with_and_without_contention(tmp_path, monkeypatch):
    idle = learnability.timing_probe(SPEC, cf.LEARNERS["copy"])
    sleepy(monkeypatch, 0.4)
    loaded = learnability.timing_probe(SPEC, cf.LEARNERS["copy"])
    # A CPU cap whose admitted share sits between the CPU projection and what the wall clock would have projected.
    wall_projection = learnability.projected_cpu_seconds(
        SPEC, dict(loaded["timing"], phases_cpu=loaded["timing"]["phases_wall"]))
    fit = (max(idle["projected_cpu_s"], loaded["projected_cpu_s"]) + 0.5 + wall_projection) / 2
    caps = cpu_capped(fit / learnability.PROBE_FIT, memory_mb=4096, pids=64)
    records = []
    for name, printed in (("idle", idle), ("loaded", loaded)):
        probe = MathRunner(PrintedProbe(tmp_path / name, printed), IMAGE, tmp_path / f"work-{name}").probe_learnability(
            {"kind": "learnability", "spec": SPEC, "source": cf.LEARNERS["copy"]}, caps)
        records.append(probe)
        assert probe["run_s"] == 99.0 and probe["cpu_s"] == pytest.approx(printed["timing"]["total_cpu_s"] + 0.2)
        assert probe["projected_cpu_s"] < probe["fit_cpu_s"] < wall_projection
        assert runner.probe_refusal(probe) is None
    assert records[0]["fit_cpu_s"] == records[1]["fit_cpu_s"] == pytest.approx(fit)


def test_the_probe_cap_is_cpu_with_a_three_times_wall_hang_guard():
    probe = MathRunner(None, IMAGE, Path(".")).probe_caps(SPEC, LIBRARY_CAPS)
    assert (probe.cpu_s, probe.wall_s) == (480.0, 1530.0)
    assert (LIBRARY_CAPS.cpu_s, LIBRARY_CAPS.wall_s) == (1800, 5490)
    # The guard covers what RLIMIT_CPU lets the job use: the soft limit rounded up and the hard-limit grace.
    assert runner.hang_guard_s(1.1) == 3 * (2 + runner.CPU_GRACE_S)
    with pytest.raises(ValueError, match="wall_s must be at least cpu_s"):
        Caps(wall_s=10, cpu_s=20, memory_mb=128, pids=8)
    with pytest.raises(ValueError, match="CPU-capped"):
        MathRunner(None, IMAGE, Path(".")).library_files({"kind": "selfcheck"}, Caps(wall_s=10, memory_mb=128,
                                                                                      pids=8))


# ----- the CPU cap stops a learner that burns CPU ------------------------------------------------------------------

def staged(tmp_path: Path, request: dict, cpu_s: float) -> Path:
    files, _ = MathRunner(None, IMAGE, tmp_path).library_files(request, cpu_capped(cpu_s, memory_mb=4096, pids=64))
    root = tmp_path / "in"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


@pytest.mark.parametrize("kind", ["learnability_probe", "learnability"])
def test_the_driver_stops_a_cpu_burning_learner_at_its_cpu_cap(tmp_path, kind):
    root = staged(tmp_path, {"kind": kind, "spec": SPEC, "source": BURNER}, cpu_s=2)
    begin = time.monotonic()
    proc = subprocess.run([sys.executable, "-B", str(root / "main.py")], capture_output=True, text=True, cwd=root,
                          timeout=120)
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert proc.returncode == 0, proc.stderr
    assert report["status"] == "unknown" and report["counterexample"] is None
    cap = report["cpu_cap"]
    assert cap["cap_cpu_s"] == 2 and cap["kind"] == kind and 2.0 <= cap["cpu_s"] < 2.0 + 5
    assert cap["wall_s"] <= time.monotonic() - begin + 0.5
    assert "CPU cap reached" in report["evidence"] and "not a pass and not a counterexample" in report["evidence"]
    assert "prover_fault" not in report


class HostExecutor:
    """Runs each staged job's main.py with the host Python as the job's command, so the driver's RLIMIT_CPU applies;
    the receipt status follows the exit code as the executor's classify does."""

    def __init__(self, tmp_path: Path):
        self.tmp_path, self.jobs = Path(tmp_path), []
        self.tmp_path.mkdir(parents=True, exist_ok=True)

    def run(self, job):
        self.jobs.append(job)
        begin = time.monotonic()
        proc = subprocess.run([sys.executable, "-B", str(Path(job.input_dir) / "main.py")], capture_output=True,
                              text=True, cwd=job.input_dir, timeout=job.caps.wall_s)
        out, err = self.tmp_path / f"{job.job_id}.stdout", self.tmp_path / f"{job.job_id}.stderr"
        out.write_text(proc.stdout)
        err.write_text(proc.stderr)
        return SimpleNamespace(status="ok" if proc.returncode == 0 else "candidate_error", detail="",
                               exit_code=proc.returncode, run_s=time.monotonic() - begin, stdout_path=str(out),
                               stderr_path=str(err))


def test_the_runner_records_a_cpu_cap_stop_of_the_probe_as_a_resource_failure(tmp_path):
    executor = HostExecutor(tmp_path / "exec")
    caps = cpu_capped(4, memory_mb=4096, pids=64)   # the probe's CPU cap: 0.8 x 4 / 3 seeds, at least 1 s
    record = MathRunner(executor, IMAGE, tmp_path / "work").check_library(
        {"kind": "learnability", "spec": SPEC, "source": BURNER}, caps)
    assert [json.loads((Path(j.input_dir) / "request.json").read_text())["kind"] for j in executor.jobs] == [
        "learnability_probe"]
    assert executor.jobs[0].caps.cpu_s == pytest.approx(1.1) and executor.jobs[0].caps.wall_s == pytest.approx(96.0)
    assert (record["status"], record["decided"], record["counterexample"]) == ("unknown", False, None)
    resource = record["resource"]
    assert resource["kind"] == "probe_cpu_cap" and resource["probe_cap_cpu_s"] == pytest.approx(1.1)
    assert resource["probe_cpu_s"] >= 1.0 and resource["probe_run_s"] is not None
    assert "reached its own CPU cap of 1 s" in record["evidence"]
    assert not record.get("prover_fault")


def test_a_learner_that_fits_records_cpu_and_wall_times(tmp_path):
    root = staged(tmp_path, {"kind": "learnability", "spec": dict(SPEC, examples=60), "source": cf.LEARNERS["copy"]},
                  cpu_s=60)
    proc = subprocess.run([sys.executable, "-B", str(root / "main.py")], capture_output=True, text=True, cwd=root,
                          timeout=120)
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert proc.returncode == 0 and report["status"] == "discharged", report.get("evidence")
    clock = report["job_clock"]
    assert clock["cap_cpu_s"] == 60 and 0 < clock["cpu_s"] < 60 and clock["wall_s"] > 0
    times = report[learnability.WALL_CLOCK_KEY]
    assert [t["seed"] for t in times["train_cpu_s"]] == [t["seed"] for t in times["train_s"]] == SPEC["seeds"]
    assert set(times["train_cpu_s"][0]) == {"seed", "true", "empty", "wrong"}
