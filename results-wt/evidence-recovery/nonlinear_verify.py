"""Run the new evidence tests and unchanged original Phase 7, preserving actual rows."""
import json
from pathlib import Path
import sys

import pytest

ROOT=Path.cwd();REPORT=ROOT/'results-wt/evidence-recovery';LABEL=sys.argv[1]
TARGETS=['hybrid/tests/research/test_nonlinear_historical_evidence.py','hybrid/tests/research/test_phase7_replay.py']
if len(sys.argv)>2 and sys.argv[2]=='controls':
    TARGETS[1:1]=['hybrid/tests/research/test_reference_replays.py','hybrid/tests/research/test_historical_binding.py',
                  'hybrid/tests/research/test_review_gate_evidence.py']

class RecordRows:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self,fixturedef,request):
        result=yield
        if fixturedef.argname=='rows' and request.module.__name__.endswith('test_phase7_replay'):
            (REPORT/f'nonlinear-{LABEL}-phase7-rows.json').write_text(json.dumps(result.get_result(),indent=2)+'\n')

code=pytest.main(TARGETS+['-q',f'--confcutdir={ROOT}',f'--basetemp={REPORT}/nonlinear-pytest-{LABEL}',
                        f'--junitxml={REPORT}/nonlinear-{LABEL}-junit.xml'],plugins=[RecordRows()])
(REPORT/f'nonlinear-{LABEL}.exit').write_text(str(code)+'\n')
raise SystemExit(code)
