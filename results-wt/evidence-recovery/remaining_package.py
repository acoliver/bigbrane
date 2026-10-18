"""Copy a cache-free package and run only the new owned evidence tests offline."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'results-wt/evidence-recovery'
PACKAGE = REPORT / 'remaining-isolated-final/package'
assert not PACKAGE.exists()
tracked = subprocess.check_output(['git','ls-files'], cwd=ROOT, text=True).splitlines()
new = [str(p.relative_to(ROOT)) for p in (ROOT/'hybrid/tests/research/fixtures/reference_replays').glob('remaining*.json')]
new += [str(p.relative_to(ROOT)) for p in (ROOT/'hybrid/tests/research/fixtures/reference_replays/programs').glob('remaining-*.py')]
new += ['hybrid/tests/research/remaining_replays.py', 'hybrid/tests/research/test_remaining_historical_evidence.py',
        'results-wt/evidence-recovery/remaining_capture.py', 'results-wt/evidence-recovery/remaining_measure.py',
        'results-wt/evidence-recovery/remaining_verify.py', 'results-wt/evidence-recovery/remaining-network-guard/sitecustomize.py']
inventory = {}
for name in sorted(set(tracked+new)):
    src = ROOT/name; dst = PACKAGE/name
    if src.is_file():
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if name in new:
            inventory[name] = hashlib.sha256(src.read_bytes()).hexdigest()
assert not (PACKAGE/'results-wt/evidence-recovery/remaining-full-capture.json').exists()
assert not (PACKAGE/'results-wt/evidence-recovery/remaining-sources').exists()
log = REPORT/'remaining-isolated.log'
env = dict(os.environ, PYTHONPATH=str(PACKAGE/'results-wt/evidence-recovery/remaining-network-guard')+':'+str(PACKAGE), PYTHONHASHSEED='0')
with log.open('w') as out:
    code = subprocess.run([sys.executable,'-m','pytest','hybrid/tests/research/test_remaining_historical_evidence.py','-q',
                           '--basetemp=results-wt/evidence-recovery/remaining-pytest-isolated',
                           '--junitxml=results-wt/evidence-recovery/remaining-isolated-junit.xml'],cwd=PACKAGE,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
(REPORT/'remaining-isolated.exit').write_text(str(code)+'\n')
shutil.copy2(PACKAGE/'results-wt/evidence-recovery/remaining-isolated-junit.xml', REPORT/'remaining-isolated-junit.xml')
(REPORT/'remaining-isolated-inventory.json').write_text(json.dumps({'owned_files':inventory,'exit_code':code,
    'raw_capture_in_package':False,'publication_snapshots_in_package':False,'network_prohibited':True},indent=2)+'\n')
raise SystemExit(code)
