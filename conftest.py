"""Pytest temp directories live inside the workspace, never in the system temp directory.

Gateway tests write token files, so their directories must be under a path this project owns. A removed
basetemp is only cleaned up after a fully passing session, so failures leave their files for inspection.
"""
import os
import shutil
from pathlib import Path

import pytest

BASETEMP = Path(__file__).resolve().parent / "tmp" / f"pytest-{os.getpid()}"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if config.option.basetemp is None:
        BASETEMP.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = BASETEMP


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0 and session.config.option.basetemp == BASETEMP:
        shutil.rmtree(BASETEMP, ignore_errors=True)
