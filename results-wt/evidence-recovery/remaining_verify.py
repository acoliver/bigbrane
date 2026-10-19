"""Run only owned historical targets and save the actual unchanged Phase7 fixture rows."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path.cwd()
REPORT = ROOT / 'results-wt/evidence-recovery'
LABEL = sys.argv[1] if len(sys.argv) > 1 else 'current'


class RecordRows:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef, request):
        result = yield
        if fixturedef.argname == 'rows' and request.module.__name__.endswith('test_phase7_replay'):
            rows = result.get_result()
            (REPORT / f'remaining-{LABEL}-phase7-rows.json').write_text(json.dumps(rows, indent=2)+'\n')


code = pytest.main([
    'hybrid/tests/research/test_remaining_historical_evidence.py',
    'hybrid/tests/research/test_reference_replays.py',
    'hybrid/tests/research/test_historical_binding.py',
    'hybrid/tests/research/test_phase7_replay.py', '-q',
    f'--basetemp=results-wt/evidence-recovery/remaining-pytest-{LABEL}',
    f'--junitxml=results-wt/evidence-recovery/remaining-{LABEL}-junit.xml',
], plugins=[RecordRows()])
(REPORT / f'remaining-{LABEL}.exit').write_text(str(code)+'\n')
raise SystemExit(code)
