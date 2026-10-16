"""Podman hygiene: the label audit with a fake podman, and a scan of our code for unfiltered removals."""
import json
import re
from pathlib import Path

import pytest

from hybrid.executor import audit, policy
from hybrid.executor.job import Caps, Job

ROOT = Path(__file__).resolve().parents[2]
BASE = "sha256:" + "b" * 64


def fake_podman(responses):
    calls = []

    def podman(argv):
        calls.append(argv)
        key = " ".join(argv)
        for prefix, value in responses.items():
            if key.startswith(prefix):
                return json.dumps(value)
        raise AssertionError(f"unexpected podman call {argv}")
    podman.calls = calls
    return podman


OWNED = {"bigbrane.owner": "bigbrane", "bigbrane.run": "r1"}
IMAGES = [
    {"Id": "i-tools", "Names": ["localhost/bigbrane/tools:r1"], "Labels": OWNED, "Digest": "sha256:1"},
    {"Id": "i-unlabeled", "Names": ["localhost/bigbrane/tools:hand"], "Labels": {}, "Digest": "sha256:2"},
    {"Id": "i-stray", "Names": [], "Labels": {"bigbrane.run": "r9"}, "Digest": "sha256:3"},
    {"Id": "i-other", "Names": ["ghcr.io/vybestack/llxprt-code/sandbox:0.11.0"], "Labels": {"x": "y"},
     "Digest": "sha256:4"},
    {"Id": "i-base", "Names": ["docker.io/library/python@" + BASE], "Labels": None, "Digest": BASE},
]
EXTERNAL = [
    {"Id": "c-ours", "Names": ["python-working-container"], "Image": "docker.io/library/python@" + BASE,
     "ImageID": "i-base", "Labels": None, "Status": "Storage"},
    {"Id": "c-node", "Names": ["node-working-container"], "Image": "docker.io/library/node:24-slim",
     "ImageID": "i-node", "Labels": None, "Status": "Storage"},
]


def responses():
    return {
        "ps -a --filter label=bigbrane.owner=bigbrane": [],
        "images --filter label=bigbrane.owner=bigbrane": [IMAGES[0]],
        "volume ls --filter label=bigbrane.owner=bigbrane": [],
        "pod ls --filter label=bigbrane.owner=bigbrane": [],
        "images --format json": IMAGES,
        "ps -a --external": EXTERNAL,
    }


def test_audit_lists_labeled_resources_and_our_unlabeled_ones():
    podman = fake_podman(responses())
    report = audit.audit(podman, {BASE})
    assert [i["Id"] for i in report["labeled"]["images"]] == ["i-tools"]
    assert [i["Id"] for i in report["unlabeled_ours"]] == ["i-unlabeled"]
    assert [i["Id"] for i in report["stray_bigbrane_labels"]] == ["i-stray"]
    assert [c["Id"] for c in report["build_leftovers"]] == ["c-ours"]
    assert [i["Id"] for i in report["base_images"]] == ["i-base"]
    assert audit.problems(report) == 3
    text = "\n".join(audit.render(report))
    assert "i-other" not in text and "llxprt" not in text and "c-node" not in text


def test_audit_is_read_only():
    podman = fake_podman(responses())
    audit.audit(podman, {BASE})
    verbs = {tuple(argv[:2]) for argv in podman.calls}
    assert verbs <= {("ps", "-a"), ("images", "--filter"), ("images", "--format"), ("volume", "ls"),
                     ("pod", "ls")}


def test_pinned_base_digests_come_from_both_containerfiles():
    assert audit.pinned_base_digests() == {
        "sha256:739ba32ae445e8d58f3d90feb85f83bebc8346f8dd280fa1eb5848f4ff1ed163"}


def test_every_policy_argv_carries_the_owner_label(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    job = Job("j1", "localhost/x@sha256:" + "a" * 64, tmp_path / "in", tmp_path / "out", ("true",),
              Caps(wall_s=1, memory_mb=128, pids=8))
    argvs = [policy.volume_create_argv("v", "r1", "j1"),
             policy.job_create_argv(job, "r1", "c", "vi", "vo"),
             policy.stager_run_argv(job, "r1", "s", "v", "/in", True, "pass")]
    for argv in argvs:
        assert "bigbrane.owner=bigbrane" in [argv[i + 1] for i, a in enumerate(argv) if a == "--label"]


CODE = [p for p in [*ROOT.joinpath("hybrid").rglob("*.py"), *ROOT.joinpath("scripts").glob("*")]
        if p.is_file() and "/tests/" not in p.as_posix() and p.suffix in (".py", ".sh", "")]
FORBIDDEN = re.compile(r"system\s+prune|system\s+reset|machine\s+(rm|reset)|prune\s+(-a|--all)")
# The only code allowed to remove images; both select ids through a bigbrane.owner label filter (tested above
# for retention, and in PODMAN-LIFECYCLE.md for the teardown script).
RMI_ALLOWED = {"scripts/podman-teardown.sh", "hybrid/retention.py"}


@pytest.mark.parametrize("path", CODE, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_unfiltered_podman_removal(path):
    text = path.read_text(errors="replace")
    relative = path.relative_to(ROOT).as_posix()
    assert not FORBIDDEN.search(text), f"{relative} has a forbidden podman command"
    for line in text.splitlines():
        if re.search(r"\bimage\s+prune\b", line):
            assert "label=bigbrane.owner=bigbrane" in line, f"{relative}: unfiltered image prune: {line}"
        if re.search(r"""(\brmi\b|"rmi")""", line) and not line.lstrip().startswith("#"):
            assert relative in RMI_ALLOWED, f"{relative} removes images: {line.strip()}"
