"""Retain rows from the real, unchanged Phase7 pytest fixture without another replay."""
import json
from pathlib import Path

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    result = yield
    if fixturedef.argname == 'rows' and request.module.__name__.endswith('test_phase7_replay'):
        rows = result.get_result()
        output = Path(__file__).parent / 'binding-phase7-rows.json'
        output.write_text(json.dumps(rows, indent=2) + '\n')
