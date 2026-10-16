"""Retention rules on a fake workspace under tmp_path, and the tools-image rule with a fake podman."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hybrid import retention
from hybrid.retention import Action

NOW = 2_000_000_000.0
ROOT_DIR = Path(__file__).resolve().parents[2]


def make(root, relative, age_days=0.0):
    path = root / relative
    path.mkdir(parents=True, exist_ok=True)
    (path / "file.txt").write_text("x")
    stamp = NOW - age_days * retention.DAY_S
    os.utime(path, (stamp, stamp))
    return path


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    make(root, "tmp/pytest-111", age_days=2)
    make(root, "tmp/pytest-222", age_days=0.5)
    make(root, "tmp/scratch/.lake", age_days=3)
    make(root, "tmp/fresh/.lake", age_days=0.1)
    for index in range(5):
        make(root, f"results/bench-tests-2026101{index}T000000Z", age_days=10 - index)
    protected = [make(root, p, age_days=400) for p in (
        "runs/run-1/__pycache__", "reports/r1", "data/d1", "results/experiment-001", "results/bench-final",
        "results/phase1-x/tmp/pytest-9", "bigbrane/__pycache__", "tests/__pycache__", ".venv-hybrid/lib/__pycache__")]
    make(root, "__pycache__")
    make(root, "hybrid/__pycache__")
    make(root, "hybrid/math/__pycache__")
    make(root, "scripts/__pycache__")
    return root, protected


def targets(actions, rule):
    return sorted(Path(a.target).relative_to(a.target.split("/ws/")[0] + "/ws").as_posix()
                  for a in actions if a.rule == rule)


def test_plan_follows_each_rule(workspace):
    root, _ = workspace
    actions = retention.plan_files(root, now=NOW)
    assert targets(actions, "tmp-pytest") == ["tmp/pytest-111"]
    assert targets(actions, "tmp-lean-cache") == ["tmp/scratch/.lake"]
    assert targets(actions, "bench-tests") == ["results/bench-tests-20261010T000000Z",
                                               "results/bench-tests-20261011T000000Z"]
    assert targets(actions, "pycache") == ["__pycache__", "hybrid/__pycache__", "hybrid/math/__pycache__",
                                           "scripts/__pycache__"]


def test_scan_does_not_descend_into_pruned_directories(workspace):
    root, _ = workspace
    make(root, "tmp/pytest-333/ws/tmp/stale/.lake", age_days=5)
    make(root, "tmp/.venv-x/lib/.lake", age_days=5)
    make(root, "tmp/nested/pytest-4/.lake", age_days=5)
    scanned = []

    def recording_scandir(path):
        scanned.append(Path(path))
        return os.scandir(path)

    actions = retention.plan_files(root, now=NOW, scandir=recording_scandir)
    listed = [Path(a.target) for a in actions]
    lake = [p for p in listed if p.name == ".lake"]
    assert sorted(lake) == [root / "tmp/scratch/.lake"]
    assert scanned and root / "tmp/scratch" in scanned
    for pruned in ("runs", "reports", "data", "tmp/pytest-111", "tmp/pytest-333", "tmp/.venv-x",
                   "tmp/nested/pytest-4", "tmp/scratch/.lake"):
        assert not any(p == root / pruned or root / pruned in p.parents for p in scanned), pruned
    assert not any(root / "runs" in p.parents for p in listed)


def test_nested_lake_is_planned_once_and_applies_without_error(workspace):
    root, _ = workspace
    make(root, "tmp/scratch/.lake/b/.lake", age_days=5)
    scanned = []

    def recording_scandir(path):
        scanned.append(Path(path))
        return os.scandir(path)

    actions = retention.plan_files(root, now=NOW, scandir=recording_scandir)
    assert targets(actions, "tmp-lean-cache") == ["tmp/scratch/.lake"]
    assert not any(p == root / "tmp/scratch/.lake" or root / "tmp/scratch/.lake" in p.parents for p in scanned)
    assert retention.apply_actions(actions, podman=lambda argv: pytest.fail("podman called")) == []
    assert not (root / "tmp/scratch/.lake").exists()


def test_nested_pytest_basetemps_are_planned_when_older_than_a_day(workspace):
    root, protected = workspace
    make(root, "tmp/fix1/pytest-tmp", age_days=2)
    make(root, "tmp/fix1/pytest-apply/deep/pytest-inner", age_days=3)
    os.utime(root / "tmp/fix1/pytest-apply", (NOW - 0.5 * retention.DAY_S,) * 2)
    make(root, "tmp/fix2/pytest-tmp", age_days=0.5)
    actions = retention.plan_files(root, now=NOW)
    assert targets(actions, "tmp-pytest") == ["tmp/fix1/pytest-tmp", "tmp/pytest-111"]
    scanned = []

    def recording_scandir(path):
        scanned.append(Path(path))
        return os.scandir(path)

    retention.plan_files(root, now=NOW, scandir=recording_scandir)
    assert not any(root / "tmp/fix2/pytest-tmp" in [p, *p.parents] for p in scanned)
    assert retention.apply_actions(actions, podman=lambda argv: pytest.fail("podman called")) == []
    assert (root / "tmp/fix2/pytest-tmp").exists() and (root / "tmp/fix1").exists()
    for path in protected:
        assert (path / "file.txt").exists()


def test_nested_planned_paths_are_dropped_for_every_rule():
    actions = [Action("tmp-pytest", "/w/tmp/a", ""), Action("tmp-lean-cache", "/w/tmp/a/.lake", ""),
               Action("pycache", "/w/hybrid/__pycache__", ""), Action("tmp-lean-cache", "/w/tmp/ab/.lake", "")]
    assert [a.target for a in retention._drop_nested(actions)] == ["/w/tmp/a", "/w/hybrid/__pycache__",
                                                                    "/w/tmp/ab/.lake"]


def test_apply_tolerates_a_missing_path_only_after_its_ancestor_was_removed(tmp_path):
    outer, inner, other = tmp_path / "a", tmp_path / "a" / "b", tmp_path / "c"
    outer.mkdir()
    (outer / "b").mkdir()

    def remove(path):
        if not Path(path).exists():
            raise FileNotFoundError(2, "No such file", path)
        shutil.rmtree(path)

    errors = retention.apply_actions(
        [Action("tmp-lean-cache", str(outer), ""), Action("tmp-lean-cache", str(inner), ""),
         Action("tmp-lean-cache", str(other), "")], podman=lambda argv: pytest.fail("podman"), remove_path=remove)
    assert len(errors) == 1 and str(other) in errors[0]


def test_find_named_matches_rglob_without_pruning(tmp_path):
    for relative in ("a/.lake/b/.lake", "a/c/.lake", ".lake", "d/e/f/.lake", "d/other"):
        (tmp_path / relative).mkdir(parents=True)
    (tmp_path / "g").mkdir()
    (tmp_path / "g/.lake").write_text("a file named like the directory")
    (tmp_path / "link").symlink_to(tmp_path / "a")
    assert sorted(retention.find_named(tmp_path, ".lake")) == sorted(tmp_path.rglob(".lake"))


def test_plan_never_names_protected_paths(workspace):
    root, protected = workspace
    planned = {Path(a.target).resolve() for a in retention.plan_files(root, now=NOW)}
    for path in protected:
        assert not any(p == path.resolve() or path.resolve() in p.parents or p in path.resolve().parents
                       for p in planned), path


def test_apply_removes_only_planned_and_leaves_protected(workspace):
    root, protected = workspace
    actions = retention.plan_files(root, now=NOW)
    assert retention.apply_actions(actions, podman=lambda argv: pytest.fail("podman called")) == []
    for action in actions:
        assert not Path(action.target).exists()
    for path in protected:
        assert (path / "file.txt").exists()
    kept = sorted(p.name for p in (root / "results").glob("bench-tests-*"))
    assert kept == ["bench-tests-20261012T000000Z", "bench-tests-20261013T000000Z", "bench-tests-20261014T000000Z"]
    assert (root / "tmp/pytest-222").exists() and (root / "tmp/fresh/.lake").exists()


@pytest.mark.parametrize("relative", ["runs/x", "reports/x", "data/x", "results/experiment-001",
                                      "results/other", "results", "bigbrane/x", "tests/x", ".venv-hybrid/x"])
def test_check_refuses_a_plan_naming_a_protected_path(tmp_path, relative):
    with pytest.raises(retention.ProtectedPath):
        retention._check(tmp_path, [Action("pycache", str(tmp_path / relative), "bug")])


def image(image_id, created, labels=None, names=None):
    return {"Id": image_id, "Created": created, "Names": names or [f"localhost/bigbrane/tools:{image_id}"],
            "Labels": labels if labels is not None else {"bigbrane.owner": "bigbrane", "bigbrane.image": "tools"}}


def fake_podman(listing):
    calls = []

    def podman(argv):
        calls.append(argv)
        if argv[0] == "images":
            return json.dumps(listing)
        if argv[0] == "rmi":
            return ""
        raise AssertionError(argv)
    podman.calls = calls
    return podman


def test_tools_images_keep_newest_two_and_the_pinned_one():
    podman = fake_podman([image("a", 100), image("b", 300), image("c", 200), image("d", 50)])
    actions = retention.plan_images(podman, pinned_id="d")
    assert [a.target for a in actions] == ["a"]
    assert podman.calls == [["images", "--filter", "label=bigbrane.owner=bigbrane",
                             "--filter", "label=bigbrane.image=tools", "--format", "json"]]
    retention.apply_actions(actions, podman)
    assert podman.calls[-1] == ["rmi", "a"]


def test_tools_image_rule_fails_fast_if_the_filter_leaks_an_unlabeled_image():
    podman = fake_podman([image("a", 1), image("x", 2, labels={}), image("b", 3), image("c", 4)])
    with pytest.raises(retention.ProtectedPath):
        retention.plan_images(podman, pinned_id=None)


def test_apply_reports_errors_and_continues(tmp_path):
    def broken_podman(argv):
        raise subprocess.CalledProcessError(1, argv)
    removed = []
    errors = retention.apply_actions(
        [Action("tools-image", "a", ""), Action("pycache", str(tmp_path / "p"), "")],
        broken_podman, remove_path=removed.append)
    assert len(errors) == 1 and removed == [str(tmp_path / "p")]


def test_shell_wrapper_dry_run_changes_nothing():
    script = ROOT_DIR / "scripts" / "retention.sh"
    done = subprocess.run([str(script), "--no-podman"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "mode: DRY-RUN" in done.stdout and "dry-run only" in done.stdout
    assert "remove [" not in done.stdout.replace("would remove [", "")
    assert all("/runs/" not in line and "/reports/" not in line and "/data/" not in line
               for line in done.stdout.splitlines())
