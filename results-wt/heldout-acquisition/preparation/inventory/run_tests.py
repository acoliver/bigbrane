"""Run only preparation tests with sibling imports and process-wide network denial."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
SIBLING = ROOT.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1]), str(SIBLING)]
sys.dont_write_bytecode = True
os.environ['BIGBRANE_LIVE'] = '0'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
from offline import install
install()
import pytest
raise SystemExit(pytest.main(['-c',str(ROOT/'pytest.ini'),'--confcutdir='+str(ROOT),
    '--basetemp='+str(ROOT/'test-work'), '--junitxml='+str(ROOT/'tests.xml'),str(ROOT/'test_inventory.py'),'-q']))
