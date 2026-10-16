"""Bench test fixtures. Work directories live under results/bench-tests-<UTC>/, never /tmp.

Tests that need the Podman VM and the executor image run only when BIGBRANE_BENCH_PODMAN=1
(they are skipped with that reason otherwise)."""
from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SESSION_DIR = ROOT / "results" / ("bench-tests-" + STAMP)
TINY = ROOT / "hybrid" / "bench" / "configs" / "tiny.toml"
TEMPLATE = ROOT / "hybrid" / "bench" / "candidate_template.py"

podman = pytest.mark.skipif(os.environ.get("BIGBRANE_BENCH_PODMAN") != "1",
                            reason="needs the Podman VM and executor image; set BIGBRANE_BENCH_PODMAN=1")


@pytest.fixture
def work_dir(request):
    name = re.sub(r"[^a-z0-9_.-]", "-", request.node.name.lower())[:60]
    path = SESSION_DIR / name
    path.mkdir(parents=True, exist_ok=False)
    return path


@pytest.fixture
def template_dir(work_dir):
    path = work_dir / "template_candidate"
    path.mkdir()
    (path / "candidate.py").write_text(TEMPLATE.read_text())
    return path
