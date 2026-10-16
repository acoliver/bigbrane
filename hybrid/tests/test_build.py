"""Image build entry point: labels, refusal on low disk, and the lock written after a build. No Podman."""
import subprocess

import pytest

from hybrid import diskguard
from hybrid.diskguard import GB, Measurement
from hybrid.executor import build, lock


def labels_of(argv):
    return {argv[i + 1] for i, arg in enumerate(argv) if arg == "--label"}


@pytest.mark.parametrize("kind", sorted(lock.IMAGES))
def test_build_argv_carries_owner_run_and_image_labels(kind):
    argv = build.build_argv(kind, "phase1-20261012T172040Z")
    assert labels_of(argv) == {"bigbrane.owner=bigbrane", "bigbrane.run=phase1-20261012T172040Z",
                               f"bigbrane.image={kind}"}
    assert "--layers=false" in argv and argv[argv.index("--platform") + 1] == "linux/arm64"
    assert argv[argv.index("-t") + 1] == f"localhost/bigbrane/{kind}:phase1-20261012T172040Z"
    assert argv[argv.index("-f") + 1] == str(lock.IMAGES[kind]["containerfile"])


def test_build_argv_rejects_bad_input():
    with pytest.raises(ValueError):
        build.build_argv("nope", "run-1")
    with pytest.raises(ValueError):
        build.build_argv("tools", "bad run id")


def test_build_refuses_on_low_disk_without_calling_podman(capsys):
    calls = []

    def low():
        raise diskguard.InsufficientSpace([Measurement("Podman VM disk", "vm:/var", 5 * GB, 20)])
    code = build.main(["tools", "--run", "r1"], check=low, measure=lambda: [],
                      runner=lambda argv: calls.append(argv))
    assert code == 2 and calls == []
    assert "5.0 GB free, threshold 20 GB" in capsys.readouterr().err


def test_build_runs_podman_then_writes_lock(monkeypatch, capsys):
    calls, locked = [], []
    ok = [Measurement("Podman storage disk", "/x", 100 * GB, 30)]
    monkeypatch.setattr(lock, "write_lock", lambda tag, kind: locked.append((tag, kind)) or {"reference": "r@sha"})

    def runner(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)
    assert build.main(["tools", "--run", "r1"], check=lambda: ok, measure=lambda: ok, runner=runner) == 0
    assert calls == [["podman", *build.build_argv("tools", "r1")]]
    assert locked == [("localhost/bigbrane/tools:r1", "tools")]
    out = capsys.readouterr().out
    assert "disk: Podman storage disk" in out and "disk after build" in out


def test_failed_build_writes_no_lock(monkeypatch):
    monkeypatch.setattr(lock, "write_lock", lambda *a, **k: pytest.fail("lock written after a failed build"))
    ok = [Measurement("Podman storage disk", "/x", 100 * GB, 30)]
    code = build.main(["tools", "--run", "r1"], check=lambda: ok, measure=lambda: ok,
                      runner=lambda argv: subprocess.CompletedProcess(argv, 125))
    assert code == 125


def test_base_digests_are_pinned_for_both_images():
    assert lock.base_digest("executor").startswith("sha256:")
    assert lock.base_digest("tools") == lock.base_digest("executor")
