"""Phase 2 math library in the real tools image through Podman. Runs only with BIGBRANE_TOOLS_PODMAN=1:

    BIGBRANE_TOOLS_PODMAN=1 .venv-hybrid/bin/python -m pytest hybrid/tests/math/test_library_live.py -q -p no:cacheprovider
"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("BIGBRANE_TOOLS_PODMAN") != "1",
                                reason="needs Podman and the tools image (set BIGBRANE_TOOLS_PODMAN=1)")


def test_library_cases_in_podman(tmp_path):
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.math.library_smoke import run_cases
    from hybrid.math.runner import MathRunner

    receipts = tmp_path / "receipts"
    runner = MathRunner(Executor("mathlib-live-test", receipts), read_lock("tools")["reference"], tmp_path)
    cases = list(run_cases(runner, receipts, quick=True))
    failed = [(c["case"], c["status"], c["problems"]) for c in cases if not c["ok"]]
    assert failed == []
