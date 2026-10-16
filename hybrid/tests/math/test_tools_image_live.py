"""Phase 1 smoke cases against the real tools image in Podman. Needs the VM and the image in
hybrid/executor/image.tools.lock.json, so it runs only with BIGBRANE_TOOLS_PODMAN=1:

    BIGBRANE_TOOLS_PODMAN=1 .venv-hybrid/bin/python -m pytest hybrid/tests/math/test_tools_image_live.py -q -p no:cacheprovider
"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("BIGBRANE_TOOLS_PODMAN") != "1",
                                reason="needs Podman and the tools image (set BIGBRANE_TOOLS_PODMAN=1)")


def test_smoke_cases_in_podman(tmp_path):
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.math.runner import MathRunner
    from hybrid.math.smoke import run_smoke

    receipts = tmp_path / "receipts"
    runner = MathRunner(Executor("tools-live-test", receipts), read_lock("tools")["reference"], tmp_path)
    cases = run_smoke(runner, receipts)
    failed = [(c["case"], c["status"], c["reasons"]) for c in cases if not c["ok"]]
    assert failed == []
