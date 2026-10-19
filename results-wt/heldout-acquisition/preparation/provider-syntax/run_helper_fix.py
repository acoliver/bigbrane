"""Run only held-out, provider-preflight, and helper-fix tests without network."""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import json

ROOT = Path(__file__).resolve().parent
HELDOUT = ROOT.parents[1]
FIX = ROOT / 'helper-fix-v2'
(FIX / 'temp').mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(FIX / 'temp')
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path[:0] = [str(ROOT), str(HELDOUT), str(HELDOUT.parents[1])]
from offline import install
install()
import pytest

args = [str(HELDOUT / 'test_harness.py'), str(ROOT / 'test_preflight.py'),
        str(ROOT / 'test_helper_fix.py'), '-c', str(HELDOUT / 'pytest.ini'),
        '--confcutdir=' + str(HELDOUT), '--basetemp=' + str(FIX / 'test-work'),
        '-o', 'cache_dir=' + str(FIX / 'pytest-cache'),
        '--junitxml=' + str(FIX / 'tests-v2.xml'), '-ra', '-q']
result = pytest.main(args)
cases = list(ET.parse(FIX / 'tests-v2.xml').getroot().iter('testcase'))
groups = {}
for case in cases:
    group = case.get('classname').rsplit('.', 1)[-1]
    count = groups.setdefault(group, {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0})
    count['total'] += 1
    status = ('failed' if case.find('failure') is not None or case.find('error') is not None
              else 'skipped' if case.find('skipped') is not None else 'passed')
    count[status] += 1
(FIX / 'TEST-SUMMARY-v2.json').write_text(json.dumps({
    'exit_code': int(result), 'total': len(cases), 'groups': groups,
    'network_prohibited': True, 'af_unix_allowed': True, 'pytest_args': args,
    'python': sys.executable, 'cwd': os.getcwd(), 'tmpdir': os.environ['TMPDIR'],
    'scope': 'held-out harness, provider preflight, helper fix only',
}, indent=2, sort_keys=True) + '\n')
raise SystemExit(result)
