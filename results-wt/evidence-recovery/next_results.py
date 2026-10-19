"""Save the rows returned by the unchanged Phase7 pytest fixture, not a second synthetic replay."""
import json
import os
from pathlib import Path

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    result = yield
    if fixturedef.argname == 'rows' and request.module.__name__.endswith('test_phase7_replay'):
        output = Path(os.environ['NEXT_ROWS_OUTPUT'])
        output.write_text(json.dumps(result.get_result(), indent=2) + '\n')
