"""Decision CT (2026-10-17): a CPU-capped library job stopped by the executor's wall-clock hang guard is an environment
failure. Stage M raises a stage error from the runner's JobHangGuard, the calibration records the card-run as
invalid_environment, and no check, verdict or resource failure is written for it. A job stopped at its CPU cap stays a
resource failure of the check (test_fix2_learnability_budget.py)."""
from __future__ import annotations

import asyncio

import pytest

from hybrid.executor.job import Caps
from hybrid.math.runner import LIBRARY_CAPS, JobHangGuard, MathRunner
from hybrid.research import calibration
from hybrid.research.stages.common import StageError
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_fix2_learnability_budget import ProbeExecutor, fitting_timing
from hybrid.tests.research.test_profile_stage import Harness


def hang_guard_harness(tmp_path, **statuses) -> Harness:
    h = Harness(tmp_path, "k1", protocols=False)
    h.ctx.math = MathRunner(ProbeExecutor(tmp_path / "exec2", timing=fitting_timing(), **statuses), mf.IMAGE,
                            h.run_dir)
    return h


@pytest.mark.parametrize("statuses", [{"full_status": "timeout"}, {"probe_status": "timeout"}],
                         ids=["full job", "probe"])
def test_a_hang_guard_stop_at_stage_m_is_an_invalid_environment_stop_and_no_verdict(tmp_path, statuses):
    h = hang_guard_harness(tmp_path, **statuses)
    stopped = asyncio.run(calibration.run_math_stages(h.ctx, h.card["id"]))
    assert stopped.startswith("stage_error at M: ") and "hang guard" in stopped
    [error] = h.ctx.card_events(h.card["id"], "stage_error")
    assert error["error_kinds"] == ["JobHangGuard"] and "environment failure" in error["error"]
    assert calibration.stop_validity(stopped, error["error_kinds"]) == calibration.INVALID_ENVIRONMENT
    # Nothing about the check was recorded as evidence: no copy-learner check, no M verdict.
    checks = [e["data"] for e in h.ledger.iter_events("math_check") if e["data"].get("resource")]
    assert checks == []
    assert not [e for e in h.ledger.iter_events("artifact") if e["data"]["kind"] == "math_verdict"
                and e["data"].get("card_id") == h.card["id"]]


def test_the_stage_error_carries_the_hang_guard_as_its_cause(tmp_path):
    h = hang_guard_harness(tmp_path, full_status="timeout")
    with pytest.raises(StageError) as exc:
        h.m()
    assert isinstance(exc.value.__cause__, JobHangGuard)
    assert calibration.error_kinds(exc.value.__cause__) == ["JobHangGuard"]


def test_error_kinds_and_validity_of_a_hang_guard():
    guard = JobHangGuard("mathlib-0123456789ab", "learnability", LIBRARY_CAPS, 5490.2, "wall cap 5490.0s exceeded")
    assert calibration.error_kinds(guard) == ["JobHangGuard"]
    assert calibration.stop_validity("stage_error at M: O7.a: " + str(guard), ["JobHangGuard"]) == \
        calibration.INVALID_ENVIRONMENT
    # A CPU cap stop is no stage error at all; any other cause stays valid as before.
    assert calibration.error_kinds(RuntimeError("x")) == []
    assert calibration.stop_validity("stage_error at M: boom", []) == calibration.VALID
    assert "hang guard of 5490 s (CPU cap 1800 s)" in str(guard)


def test_lean_and_program_jobs_keep_their_wall_caps():
    from hybrid.math.runner import LEAN_CAPS, PROGRAM_CAPS
    assert LEAN_CAPS.cpu_s is None and PROGRAM_CAPS.cpu_s is None
    assert isinstance(LIBRARY_CAPS, Caps) and LIBRARY_CAPS.cpu_s == 1800
