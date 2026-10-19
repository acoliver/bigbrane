"""Start the network prohibition before importing pytest or test modules."""
import os
from pathlib import Path
from offline import install

install()
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
import pytest

root = Path(__file__).resolve().parent
raise SystemExit(pytest.main(['-c', str(root / 'pytest.ini'), '--confcutdir', str(root),
    '--basetemp', str(root / 'test-work'), '-o', 'cache_dir=' + str(root / 'pytest-cache'),
    str(root / 'test_harness.py')]))
