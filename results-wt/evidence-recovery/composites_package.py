"""Export selected tracked files and new essentials; no captures, source bodies or caches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'results-wt/evidence-recovery'
LABEL=sys.argv[1] if len(sys.argv)>1 else 'isolated'
PACKAGE=REPORT/f'composites-{LABEL}/package'
assert not PACKAGE.exists()
tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
new=['hybrid/tests/research/composites_replays.py','hybrid/tests/research/test_composites_historical_evidence.py',
     'results-wt/evidence-recovery/composites_capture.py','results-wt/evidence-recovery/composites_measure.py',
     'results-wt/evidence-recovery/composites_verify.py','results-wt/evidence-recovery/composites-network-guard/sitecustomize.py']
new += [str(p.relative_to(ROOT)) for p in (ROOT/'hybrid/tests/research/fixtures/reference_replays').glob('composites_*.json')]
new += [str(p.relative_to(ROOT)) for p in (ROOT/'hybrid/tests/research/fixtures/reference_replays/programs').glob('composites-*.py')]
inventory={}
for name in sorted(set(tracked+new)):
    src=ROOT/name;dst=PACKAGE/name
    if src.is_file():
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if name in new:inventory[name]=hashlib.sha256(src.read_bytes()).hexdigest()
assert not (PACKAGE/'results-wt/evidence-recovery/composites-full-capture.json').exists()
assert not (PACKAGE/'results-wt/evidence-recovery/composites-sources').exists()
assert not (PACKAGE/'hybrid/literature/cache').exists()
env=dict(os.environ,PYTHONPATH=str(PACKAGE/'results-wt/evidence-recovery/composites-network-guard')+':'+str(PACKAGE),
         COMPOSITES_ISOLATED='1',PYTHONHASHSEED='0',BIGBRANE_LIVE='0')
with (REPORT/f'composites-{LABEL}.log').open('w') as out:
    child=subprocess.Popen([sys.executable,'results-wt/evidence-recovery/composites_verify.py',LABEL,'focused'],
                           cwd=PACKAGE,env=env,stdout=out,stderr=subprocess.STDOUT)
    while child.poll() is None:
        print('isolated composite verification running',flush=True);time.sleep(15)
    code=child.returncode
for suffix in ('exit','phase7-rows.json','junit.xml'):
    shutil.copy2(PACKAGE/f'results-wt/evidence-recovery/composites-{LABEL}.{suffix}' if suffix=='exit' else
                 PACKAGE/f'results-wt/evidence-recovery/composites-{LABEL}-{suffix}',
                 REPORT/f'composites-{LABEL}.{suffix}' if suffix=='exit' else REPORT/f'composites-{LABEL}-{suffix}')
(REPORT/f'composites-{LABEL}-inventory.json').write_text(json.dumps({'owned_files':inventory,'exit_code':code,
    'raw_capture_in_package':False,'publication_bodies_in_package':False,'network_prohibited':True,
    'external_original_runs_masked_absent_and_open_prohibited':True},indent=2)+'\n')
raise SystemExit(code)
