"""Free-space precheck with fake statvfs and a fake `podman machine ssh df`. Nothing touches the real disks."""
import subprocess
from types import SimpleNamespace

import pytest

from hybrid import diskguard
from hybrid.diskguard import GB


def fake_statvfs(free_by_path):
    def statvfs(path):
        for prefix, free in free_by_path.items():
            if str(path).startswith(prefix):
                return SimpleNamespace(f_bavail=free // 1000, f_frsize=1000)
        raise AssertionError(f"unexpected statvfs path {path}")
    return statvfs


def fake_df(avail_bytes, returncode=0, stderr=""):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout=f"   Avail\n{avail_bytes}\n", stderr=stderr)
    run.calls = calls
    return run


ROOMY = {str(diskguard.WORKSPACE): 100 * GB, str(diskguard.PODMAN_STORAGE): 200 * GB}


def test_run_space_passes_at_or_above_threshold():
    statvfs = fake_statvfs({str(diskguard.WORKSPACE): 20 * GB})
    [measurement] = diskguard.require_run_space(statvfs)
    assert measurement.ok and measurement.min_gb == diskguard.WORKSPACE_MIN_GB == 20


def test_run_space_refuses_below_threshold_and_names_measurement_and_threshold():
    statvfs = fake_statvfs({str(diskguard.WORKSPACE): 19 * GB})
    with pytest.raises(diskguard.InsufficientSpace) as info:
        diskguard.require_run_space(statvfs)
    text = str(info.value)
    assert "19.0 GB free" in text and "threshold 20 GB" in text and str(diskguard.WORKSPACE) in text
    assert [m.name for m in info.value.failed] == ["workspace volume"]


def test_build_space_checks_host_disk_and_vm_disk():
    run = fake_df(80 * GB)
    measured = diskguard.require_build_space(fake_statvfs(ROOMY), run)
    assert [m.name for m in measured] == ["Podman storage disk", "Podman VM disk"]
    assert run.calls == [["podman", "machine", "ssh", diskguard.MACHINE, "df", "-B1", "--output=avail", "/var"]]


@pytest.mark.parametrize("host_gb, vm_gb, failing", [
    (29, 80, ["Podman storage disk"]),
    (200, 19, ["Podman VM disk"]),
    (10, 10, ["Podman storage disk", "Podman VM disk"]),
])
def test_build_space_refuses_each_low_disk(host_gb, vm_gb, failing):
    statvfs = fake_statvfs({str(diskguard.PODMAN_STORAGE): host_gb * GB})
    with pytest.raises(diskguard.InsufficientSpace) as info:
        diskguard.require_build_space(statvfs, fake_df(vm_gb * GB))
    assert [m.name for m in info.value.failed] == failing
    assert "threshold 30 GB" in str(info.value) or "threshold 20 GB" in str(info.value)


def test_vm_measurement_fails_fast_when_the_vm_cannot_be_asked():
    with pytest.raises(RuntimeError, match="cannot measure the Podman VM disk"):
        diskguard.vm_free_bytes(fake_df(0, returncode=125, stderr="machine not running"))


def test_vm_measurement_rejects_unexpected_df_output():
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="Avail\nlots\n", stderr="")
    with pytest.raises(RuntimeError, match="unexpected df output"):
        diskguard.vm_free_bytes(run)


def test_guard_cli_prints_measurements_or_refusal(capsys):
    assert diskguard.guard_cli(lambda: diskguard.require_run_space(fake_statvfs(ROOMY)), "x") is True
    assert capsys.readouterr() == ("", "")
    assert diskguard.guard_cli(lambda: diskguard.require_run_space(fake_statvfs(ROOMY)), "x", report=True) is True
    assert "workspace volume" in capsys.readouterr().out
    low = fake_statvfs({str(diskguard.WORKSPACE): 1 * GB})
    assert diskguard.guard_cli(lambda: diskguard.require_run_space(low), "the research run") is False
    err = capsys.readouterr().err
    assert "refusing to start the research run" in err and "1.0 GB free, threshold 20 GB" in err


def test_cli_prints_three_measurements(monkeypatch, capsys):
    monkeypatch.setattr(diskguard, "path_free_bytes", lambda path, statvfs=None: 50 * GB)
    monkeypatch.setattr(diskguard, "vm_free_bytes", lambda run=None: 70 * GB)
    assert diskguard.main([]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split(" (")[0] for line in out] == ["workspace volume", "Podman storage disk", "Podman VM disk"]
    assert all("threshold" in line for line in out)


def test_cli_require_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(diskguard, "path_free_bytes", lambda path, statvfs=None: 25 * GB)
    monkeypatch.setattr(diskguard, "vm_free_bytes", lambda run=None: 70 * GB)
    assert diskguard.main(["--require", "run"]) == 0
    assert diskguard.main(["--require", "build"]) == 2   # 25 GB on the Podman host disk is below 30


def test_cli_build_requirement_fails_when_vm_unreachable(monkeypatch, capsys):
    monkeypatch.setattr(diskguard, "path_free_bytes", lambda path, statvfs=None: 500 * GB)

    def unreachable(run=None):
        raise RuntimeError("machine down")
    monkeypatch.setattr(diskguard, "vm_free_bytes", unreachable)
    assert diskguard.main(["--require", "build"]) == 2
    assert "not measured" in capsys.readouterr().out
