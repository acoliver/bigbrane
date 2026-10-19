"""Only the focused preparation tests run, under the existing socket audit guard."""
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from offline import install
install()
import pytest
result = pytest.main([str(ROOT/'test_preflight.py'),'-c',str(ROOT/'pytest.ini'),
    '--confcutdir='+str(ROOT), '--basetemp='+str(ROOT/'test-work'),
    '-o','cache_dir='+str(ROOT/'pytest-cache'), '--junitxml='+str(ROOT/'tests.xml'),'-ra','-q'])
suite = ET.parse(ROOT/'tests.xml').getroot()
cases = list(suite.iter('testcase'))
failed = [c.get('name') for c in cases if c.find('failure') is not None or c.find('error') is not None]
summary = {'exit_code':int(result),'total':len(cases),'passed':sum(c.find('skipped') is None and c.find('failure') is None and c.find('error') is None for c in cases),
 'expected_failures':[c.get('name') for c in cases if c.find('skipped') is not None], 'failed':failed,
 'required_probe_builder_checks_passed':result==0,
 'network_prohibited':True,'scope':'focused provider contract only, not full suite'}
(ROOT/'OFFLINE-SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
raise SystemExit(result)
