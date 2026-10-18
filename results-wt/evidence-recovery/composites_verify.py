"""Run selected historical controls and the unchanged actual Phase7 assertions once per context."""
import json
from pathlib import Path
import sys

import pytest

ROOT=Path.cwd()
REPORT=ROOT/'results-wt/evidence-recovery'
LABEL=sys.argv[1]
TARGETS=[
 'hybrid/tests/research/test_composites_historical_evidence.py',
 'hybrid/tests/research/test_historical_references.py',
 'hybrid/tests/research/test_next_historical_references.py',
 'hybrid/tests/research/test_remaining_historical_evidence.py',
 'hybrid/tests/research/test_reference_replays.py',
 'hybrid/tests/research/test_historical_binding.py',
 'hybrid/tests/research/test_review_gate_evidence.py',
 'hybrid/tests/research/test_phase7_replay.py',
]

if len(sys.argv)>2 and sys.argv[2]=='focused':
    TARGETS=[TARGETS[0],TARGETS[-1]]

class RecordRows:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self,fixturedef,request):
        result=yield
        if fixturedef.argname=='rows' and request.module.__name__.endswith('test_phase7_replay'):
            (REPORT/f'composites-{LABEL}-phase7-rows.json').write_text(json.dumps(result.get_result(),indent=2)+'\n')

code=pytest.main(TARGETS+['-q',f'--confcutdir={ROOT}',
    f'--basetemp=results-wt/evidence-recovery/composites-pytest-{LABEL}',
    f'--junitxml=results-wt/evidence-recovery/composites-{LABEL}-junit.xml'],plugins=[RecordRows()])
(REPORT/f'composites-{LABEL}.exit').write_text(str(code)+'\n')
raise SystemExit(code)
