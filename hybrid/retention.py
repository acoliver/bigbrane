"""Workspace retention (DISCOVERY-PIPELINE-PLAN.md section 20, decision D3). Dry run unless apply=True.

Rules, and nothing else:

1. Directories named ``pytest-*`` at any depth under ``tmp/``, older than one day (conftest.py removes passing
   sessions; these are failures, plus basetemp dirs that fix sessions created inside their own folders, such as
   ``tmp/fixN/pytest-tmp``).
2. ``results/bench-tests-*``: keep the newest 3 by modification time, remove the rest.
3. ``__pycache__`` directories under the workspace root, ``hybrid/`` and ``scripts/``.
4. Lean caches (plan section 20): Bigbrane tools images beyond the newest 2, selected only through the
   ``bigbrane.owner=bigbrane`` and ``bigbrane.image=tools`` labels, never the image pinned in
   image.tools.lock.json; and ``.lake`` directories under ``tmp/`` older than one day.

The search under ``tmp/`` does not descend into ``pytest-*`` directories (rule 1 owns them whole, and they hold
about a million entries between them, whether or not they are old enough to remove yet), into ``.lake``
directories already planned for removal, or into ``.venv*`` directories. Nothing else is pruned.

A planned path that lies under another planned path is dropped, for every file rule, so each tree is removed
once. When applying, a missing path is an error unless an ancestor was removed earlier in the same run.

Every planned path is checked against the protected set before anything is removed: ``runs/``, ``reports/``,
``data/``, every ``results/`` entry other than ``bench-tests-*``, and the Experiment 1 files (``bigbrane/``,
``tests/``). A plan that names a protected path raises instead of deleting.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DAY_S = 86400
KEEP_BENCH_TESTS = 3
KEEP_TOOLS_IMAGES = 2
PYCACHE_ROOTS = ("hybrid", "scripts")
PROTECTED_TOP = ("runs", "reports", "data", "bigbrane", "tests")
TOOLS_FILTERS = ("--filter", "label=bigbrane.owner=bigbrane", "--filter", "label=bigbrane.image=tools")


class ProtectedPath(RuntimeError):
    """The plan named a path the retention rules must never touch. This is a bug in the rules."""


@dataclass(frozen=True)
class Action:
    rule: str
    target: str          # a workspace path, or an image id for the image rule
    detail: str


def _is_protected(root: Path, path: Path) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    parts = relative.parts
    if not parts:
        return True
    if parts[0] in PROTECTED_TOP or parts[0].startswith("."):
        return True
    if parts[0] == "results":
        return len(parts) < 2 or not parts[1].startswith("bench-tests-")
    return False


def _check(root: Path, actions: list[Action]) -> list[Action]:
    for action in actions:
        if action.rule != "tools-image" and _is_protected(root, Path(action.target)):
            raise ProtectedPath(f"retention planned to remove protected path {action.target} ({action.rule})")
    return actions


def _is_old(path: Path, now: float) -> bool:
    return now - path.stat().st_mtime > DAY_S


def _is_pytest_dir_name(name: str) -> bool:
    return name.startswith("pytest-")


def _tmp_candidates(tmp: Path, now: float, scandir: Callable) -> list[Path]:
    """``pytest-*`` and ``.lake`` entries below tmp. Never enters ``.venv*``, any ``pytest-*`` directory (rule 1
    owns it whole), or a ``.lake`` directory old enough to be planned (its contents go with it)."""
    def prune(directory: Path) -> bool:
        return directory.name.startswith(".venv")

    def stop(directory: Path) -> bool:
        return _is_pytest_dir_name(directory.name) or (directory.name == ".lake" and _is_old(directory, now))

    return find_named(tmp, lambda name: name == ".lake" or _is_pytest_dir_name(name), prune, scandir, stop)


def _drop_nested(actions: list[Action]) -> list[Action]:
    """Actions whose target does not lie under another action's target."""
    targets = {Path(action.target) for action in actions}
    return [action for action in actions if not any(parent in targets for parent in Path(action.target).parents)]


def find_named(base: Path, name: str | Callable[[str], bool], prune: Callable[[Path], bool] = lambda directory: False,
               scandir: Callable = os.scandir, stop: Callable[[Path], bool] = lambda directory: False) -> list[Path]:
    """Every file or directory called name (or whose name satisfies the predicate) below base, like
    ``base.rglob(name)`` (symlinks are not followed), except that a directory for which prune is true is neither
    listed nor entered, and a directory for which stop is true is listed but not entered."""
    matches = (lambda entry_name: entry_name == name) if isinstance(name, str) else name
    found = []
    pending = [base]
    while pending:
        directory = pending.pop()
        try:
            with scandir(directory) as entries:
                listing = list(entries)
        except OSError:
            continue  # vanished or unreadable while running jobs churn tmp/; rglob skips these the same way
        for entry in listing:
            path = directory / entry.name
            is_dir = entry.is_dir(follow_symlinks=False)
            if is_dir and prune(path):
                continue
            if matches(entry.name):
                found.append(path)
            if is_dir and not stop(path):
                pending.append(path)
    return found


def plan_files(root: Path, now: float | None = None, scandir: Callable = os.scandir) -> list[Action]:
    """The file-system actions for the workspace at root. Pure apart from reading directory metadata."""
    now = time.time() if now is None else now
    actions = []
    tmp = root / "tmp"
    if tmp.is_dir():
        candidates = sorted(_tmp_candidates(tmp, now, scandir))
        for rule, wanted in (("tmp-pytest", _is_pytest_dir_name), ("tmp-lean-cache", lambda name: name == ".lake")):
            for path in candidates:
                if not wanted(path.name):
                    continue
                age = now - path.stat().st_mtime
                if path.is_dir() and age > DAY_S:
                    actions.append(Action(rule, str(path), f"{age / DAY_S:.1f} days old"))
    bench = sorted((p for p in (root / "results").glob("bench-tests-*") if p.is_dir()),
                   key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    for path in bench[KEEP_BENCH_TESTS:]:
        actions.append(Action("bench-tests", str(path), f"older than the newest {KEEP_BENCH_TESTS}"))
    pycache = [root / "__pycache__"] if (root / "__pycache__").is_dir() else []
    for name in PYCACHE_ROOTS:
        if (root / name).is_dir():
            pycache += sorted(p for p in (root / name).rglob("__pycache__") if p.is_dir())
    actions += [Action("pycache", str(p), "bytecode cache") for p in pycache]
    return _check(root, _drop_nested(actions))


def plan_images(podman: Callable, pinned_id: str | None) -> list[Action]:
    """Tools images beyond the newest KEEP_TOOLS_IMAGES, from a label-filtered listing only."""
    listed = json.loads(podman(["images", *TOOLS_FILTERS, "--format", "json"]) or "[]")
    images = {}
    for image in listed:
        labels = image.get("Labels") or {}
        if labels.get("bigbrane.owner") != "bigbrane" or labels.get("bigbrane.image") != "tools":
            raise ProtectedPath(f"label-filtered listing returned an image without the labels: {image.get('Id')}")
        images[image["Id"]] = image
    ordered = sorted(images.values(), key=lambda image: image["Created"], reverse=True)
    actions = []
    for image in ordered[KEEP_TOOLS_IMAGES:]:
        if pinned_id and image["Id"] == pinned_id:
            continue
        names = ",".join(image.get("Names") or []) or "<untagged>"
        actions.append(Action("tools-image", image["Id"], f"{names}, older than the newest {KEEP_TOOLS_IMAGES}"))
    return actions


def _remove_path(path: str) -> None:
    shutil.rmtree(path)


def apply_actions(actions: list[Action], podman: Callable, remove_path: Callable = _remove_path) -> list[str]:
    """Carry out the actions. Returns error strings; one failure does not stop the others. A path that is already
    gone is not an error only when an ancestor was removed earlier in the same run."""
    errors = []
    removed = set()
    for action in actions:
        try:
            if action.rule == "tools-image":
                podman(["rmi", action.target])
            else:
                try:
                    remove_path(action.target)
                except FileNotFoundError:
                    if not any(parent in removed for parent in Path(action.target).parents):
                        raise
                removed.add(Path(action.target))
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{action.rule} {action.target}: {exc}")
    return errors


def podman_cli(argv: list[str]) -> str:
    return subprocess.run(["podman", *argv], check=True, capture_output=True, text=True, timeout=120).stdout


def main(argv: list[str] | None = None) -> int:
    import argparse

    from hybrid.diskguard import WORKSPACE
    from hybrid.executor.lock import IMAGES

    parser = argparse.ArgumentParser(prog="scripts/retention.sh")
    parser.add_argument("--apply", action="store_true", help="remove what the dry run lists")
    parser.add_argument("--no-podman", action="store_true", help="skip the tools image rule")
    args = parser.parse_args(argv)
    actions = plan_files(WORKSPACE)
    if not args.no_podman:
        lock_path = IMAGES["tools"]["lock"]
        pinned = json.loads(lock_path.read_text())["image_id"] if lock_path.exists() else None
        try:
            actions += plan_images(podman_cli, pinned)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"podman: not reachable, tools image rule skipped ({exc})")
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"mode: {mode}  workspace: {WORKSPACE}")
    if not actions:
        print("nothing to remove")
    for action in actions:
        print(f"{'remove' if args.apply else 'would remove'} [{action.rule}] {action.target} ({action.detail})")
    if not args.apply:
        print("dry-run only; re-run with --apply to execute.")
        return 0
    errors = apply_actions(actions, podman_cli)
    for error in errors:
        print(f"error: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
