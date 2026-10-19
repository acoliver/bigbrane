"""Run portable checks from a source/metadata-only copy with original data access denied."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MAIN = REPO.with_name('bigbrane')
WORK = REPO / 'tmp/preparation-checkpoint' / os.environ.get('CHECKPOINT_ISOLATION_NAME', 'isolated')
COPY = WORK / 'repo'
HELD_REL = Path('results-wt/heldout-acquisition')


def prepare():
    COPY.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(['git', '-C', str(REPO), 'ls-files', '-z']).decode().split('\0')
    public = json.loads((HERE / 'PUBLIC-FILES.json').read_text())['files']
    selected = sorted(set(public) | {p for p in tracked if p.endswith('.py') or p.startswith(str(HELD_REL) + '/')})
    copied = []
    for name in selected:
        src = REPO / name
        dst = COPY / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied.append(name)
    changes = []
    for name in ['INPUT-HASHES.json', 'INPUT-HASHES-v2.json']:
        path = COPY / HELD_REL / 'preparation/provider-syntax' / name
        old = path.read_bytes()
        obj = json.loads(old)
        new = {str(COPY / Path(k).relative_to(REPO)) if Path(k).is_relative_to(REPO) else str(WORK / 'main-not-present' / Path(k).relative_to(MAIN)): v for k, v in obj.items()}
        assert list(obj.values()) == list(new.values())
        path.write_text(json.dumps(new, indent=2) + '\n')
        changes.append({'file': name, 'transformation': 'absolute path keys relocated only; every expected content hash unchanged', 'original_sha256': hashlib.sha256(old).hexdigest()})
    (WORK / 'COPY-RECEIPT.json').write_text(json.dumps({'files': copied, 'relocations': changes, 'private_sources_copied': False, 'raw_cache_copied': False}, indent=2) + '\n')


def run():
    os.chdir(COPY)
    held = COPY / HELD_REL
    paths = [held, held / 'preparation/provider-syntax', held / 'preparation/checkpoint', COPY]
    sys.path[:] = [str(p) for p in paths] + [p for p in sys.path if not p.startswith(str(REPO)) and not p.startswith(str(MAIN)) or '/.venv' in p]
    os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', BIGBRANE_LIVE='0', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(WORK))
    sys.dont_write_bytecode = True
    def prohibit_original_data(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(WORK) or path.is_relative_to(MAIN / '.venv-hybrid') or path.is_relative_to(MAIN / '.venv'):
            return
        if path.is_relative_to(REPO) or path.is_relative_to(MAIN):
            raise RuntimeError('Original checkout access prohibited: ' + str(path))
    sys.addaudithook(prohibit_original_data)
    from offline import install
    install()
    import pytest
    files = [held / 'test_harness.py', held / 'preparation/provider-syntax/test_helper_fix.py', held / 'preparation/checkpoint/test_checkpoint.py']
    args = ['-q', '-c', str(held / 'pytest.ini'), '--confcutdir=' + str(held), '--basetemp=' + str(WORK / 'test-work'), '-p', 'no:cacheprovider', '--junitxml=' + str(WORK / 'tests.xml'), *map(str, files)]
    code = pytest.main(args)
    groups = {}
    for case in ET.parse(WORK / 'tests.xml').getroot().iter('testcase'):
        counts = groups.setdefault(case.get('classname'), {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0})
        counts['total'] += 1
        key = 'failed' if any(case.find(k) is not None for k in ['error', 'failure']) else 'skipped' if case.find('skipped') is not None else 'passed'
        counts[key] += 1
    (WORK / 'TESTS.json').write_text(json.dumps({'exit_code': int(code), 'groups': groups, 'original_checkout_access_denied': True, 'network_prohibited': True, 'raw_cache_or_private_bodies_required': False, 'input_pin_path_relocation_only': True, 'full_suite_run': False}, indent=2) + '\n')
    return int(code)


if __name__ == '__main__':
    prepare()
    raise SystemExit(run())
