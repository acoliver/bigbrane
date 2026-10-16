"""The research CLI checks free space and the local LM Studio models before run and plumb. Tests must not depend on
the host disk or on what LM Studio has loaded, so the disk measurement reports a large volume and LM Studio reports
every local model of models.toml loaded with a large context, unless a test patches them (test_diskguard_wiring.py,
test_localmodels.py)."""
from pathlib import Path

import pytest

from hybrid import diskguard, localmodels


@pytest.fixture(autouse=True)
def roomy_workspace(monkeypatch):
    monkeypatch.setattr(diskguard, "path_free_bytes", lambda path, statvfs=None: 10**15)


@pytest.fixture(autouse=True)
def local_models_loaded(monkeypatch):
    from hybrid.llm import load_config
    from hybrid.research.orchestrator import ROOT

    models = load_config(ROOT / "hybrid" / "config" / "models.toml")
    ids = {entry.model for table in models.routes.values() for entries in table.values() for entry in entries
           if models.backends[entry.backend].kind == localmodels.LOCAL_KIND}
    listing = [{"id": model, "state": "loaded", "loaded_context_length": 10**7} for model in sorted(ids)]
    monkeypatch.setattr(localmodels, "fetch_models", lambda server, **kwargs: listing)


@pytest.fixture(autouse=True)
def no_real_sealed_sets(tmp_path, monkeypatch):
    """The real sealed planted sets live in ~/.bigbrane/planted (or BIGBRANE_PLANTED_DIR); no test may read or write
    them. A test that names no sealed directory resolves to one under its own temporary directory, which the root
    conftest.py keeps inside the workspace tmp/."""
    from hybrid.research import planted_seal

    monkeypatch.setattr(planted_seal, "DEFAULT_SEALED_DIR", tmp_path / "default-sealed")
    monkeypatch.delenv(planted_seal.SEALED_ENV, raising=False)


# FIX-16: the P2 count, the P4 test count and the P1 lengths are protocol values the host sets in every profile
# statement (hybrid/math/profile.PROTOCOL). Before FIX-16 the scripted fixers chose small ones for speed (count 20,
# test_count 40, ns 16 to 128); the research tests keep those sizes by setting the protocol to them.
# test_fix16_card_completeness.py checks the production values with this patch undone.
SCRIPTED_PROTOCOL = {"p2_count": 20, "p4_test_count": 40, "p1_ns": (16, 32, 64, 128)}


@pytest.fixture(autouse=True)
def scripted_protocol(monkeypatch):
    from hybrid.math import profile

    monkeypatch.setattr(profile, "PROTOCOL", dict(profile.PROTOCOL, **SCRIPTED_PROTOCOL))
