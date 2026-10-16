"""Free-space precheck (DISCOVERY-PIPELINE-PLAN.md section 20, decision D3).

Three places can fill up:

* the external volume holding the workspace (runs, results, ledgers): research runs and plumbing checks
  refuse to start below WORKSPACE_MIN_GB free;
* the internal disk holding Podman storage (``~/.local/share/containers``, where the VM disk image lives):
  image builds refuse below PODMAN_HOST_MIN_GB free;
* the Podman VM's own disk (``/var`` inside ``podman-machine-default``): image builds refuse below
  VM_MIN_GB free.

``python -m hybrid.diskguard`` prints all three. ``--require run`` or ``--require build`` exits 2 when the
matching check fails. A refusal always prints what was measured and the threshold.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

GB = 10**9
WORKSPACE_MIN_GB = 20
PODMAN_HOST_MIN_GB = 30
VM_MIN_GB = 20

WORKSPACE = Path(__file__).resolve().parents[1]
PODMAN_STORAGE = Path.home() / ".local" / "share" / "containers"
MACHINE = "podman-machine-default"
VM_PATH = "/var"


class InsufficientSpace(RuntimeError):
    """A measured free space is below its threshold. The message names every measurement that failed."""

    def __init__(self, failed: list["Measurement"]):
        super().__init__("not enough free disk space: " + "; ".join(m.describe() for m in failed))
        self.failed = failed


@dataclass(frozen=True)
class Measurement:
    name: str
    location: str
    free_bytes: int
    min_gb: int

    @property
    def ok(self) -> bool:
        return self.free_bytes >= self.min_gb * GB

    def describe(self) -> str:
        return (f"{self.name} ({self.location}) has {self.free_bytes / GB:.1f} GB free, "
                f"threshold {self.min_gb} GB: {'ok' if self.ok else 'BELOW THRESHOLD'}")


def path_free_bytes(path: Path, statvfs: Callable = os.statvfs) -> int:
    """Bytes available to an unprivileged user on the filesystem holding path."""
    st = statvfs(str(path))
    return st.f_bavail * st.f_frsize


def vm_free_bytes(run: Callable = subprocess.run, machine: str = MACHINE, path: str = VM_PATH) -> int:
    """Bytes available on the Podman VM's filesystem holding path, through ``podman machine ssh df``.
    Raises RuntimeError when the VM cannot be asked (an image build cannot run then either)."""
    argv = ["podman", "machine", "ssh", machine, "df", "-B1", "--output=avail", path]
    done = run(argv, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError(f"cannot measure the Podman VM disk ({' '.join(argv)}): exit {done.returncode}: "
                           f"{done.stderr.strip()[:400]}")
    values = [line.strip() for line in done.stdout.splitlines() if line.strip().isdigit()]
    if len(values) != 1:
        raise RuntimeError(f"unexpected df output from the Podman VM: {done.stdout.strip()[:400]!r}")
    return int(values[0])


def workspace_measurement(statvfs: Callable = os.statvfs) -> Measurement:
    return Measurement("workspace volume", str(WORKSPACE), path_free_bytes(WORKSPACE, statvfs), WORKSPACE_MIN_GB)


def podman_host_measurement(statvfs: Callable = os.statvfs) -> Measurement:
    return Measurement("Podman storage disk", str(PODMAN_STORAGE), path_free_bytes(PODMAN_STORAGE, statvfs),
                       PODMAN_HOST_MIN_GB)


def vm_measurement(run: Callable = subprocess.run) -> Measurement:
    return Measurement("Podman VM disk", f"{MACHINE}:{VM_PATH}", vm_free_bytes(run), VM_MIN_GB)


def _require(measurements: list[Measurement]) -> list[Measurement]:
    failed = [m for m in measurements if not m.ok]
    if failed:
        raise InsufficientSpace(failed)
    return measurements


def require_run_space(statvfs: Callable = os.statvfs) -> list[Measurement]:
    """Precheck for ``hybrid.research run`` and ``plumb``. Raises InsufficientSpace."""
    return _require([workspace_measurement(statvfs)])


def measure_build(statvfs: Callable = os.statvfs, run: Callable = subprocess.run) -> list[Measurement]:
    return [podman_host_measurement(statvfs), vm_measurement(run)]


def require_build_space(statvfs: Callable = os.statvfs, run: Callable = subprocess.run) -> list[Measurement]:
    """Precheck for image builds: internal disk and the VM disk. Raises InsufficientSpace."""
    return _require(measure_build(statvfs, run))


def guard_cli(check: Callable[[], list[Measurement]], what: str, *, report: bool = False) -> bool:
    """Run a precheck for a command-line entry point. On a refusal prints what was measured and the threshold to
    stderr and returns False (the caller exits with status 2). With report=True a pass is printed too (stdout);
    the research commands keep it off because their first output lines are part of their contract."""
    try:
        measurements = check()
    except InsufficientSpace as exc:
        print(f"refusing to start {what}: {exc}", file=sys.stderr)
        return False
    if report:
        for m in measurements:
            print(f"disk: {m.describe()}", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hybrid.diskguard", description=__doc__.split("\n\n")[0])
    parser.add_argument("--require", choices=("run", "build"),
                        help="exit 2 when the run (workspace) or build (Podman host and VM) check fails")
    args = parser.parse_args(argv)
    measurements = [workspace_measurement(), podman_host_measurement()]
    try:
        measurements.append(vm_measurement())
    except RuntimeError as exc:
        print(f"Podman VM disk: not measured: {exc}")
        if args.require == "build":
            return 2
    for m in measurements:
        print(m.describe())
    wanted = {"run": {"workspace volume"}, "build": {"Podman storage disk", "Podman VM disk"}}.get(args.require, set())
    return 2 if any(m.name in wanted and not m.ok for m in measurements) else 0


if __name__ == "__main__":
    sys.exit(main())
