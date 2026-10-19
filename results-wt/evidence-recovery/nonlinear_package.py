"""Cache-free research verification without local raw sources, capture files or network."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'results-wt/evidence-recovery'
LABEL=sys.argv[1];PACKAGE=REPORT/f'nonlinear-{LABEL}/package'
assert not PACKAGE.exists()
tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
new=['hybrid/tests/research/nonlinear_replays.py','hybrid/tests/research/test_nonlinear_historical_evidence.py',
     'results-wt/evidence-recovery/nonlinear_capture.py','results-wt/evidence-recovery/nonlinear_measure.py',
     'results-wt/evidence-recovery/nonlinear_verify.py','results-wt/evidence-recovery/composites-network-guard/sitecustomize.py']
new += [str(p.relative_to(ROOT)) for p in (ROOT/'hybrid/tests/research/fixtures/reference_replays').glob('nonlinear_*.json')]
new += [str(p.relative_to(ROOT)) for p in (ROOT/'hybrid/tests/research/fixtures/reference_replays/programs').glob('nonlinear-*.py')]
inventory={}
for name in sorted(set(tracked+new)):
    src=ROOT/name;dst=PACKAGE/name
    if src.is_file():
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if name in new:inventory[name]=hashlib.sha256(src.read_bytes()).hexdigest()
assert not (PACKAGE/'results-wt/evidence-recovery/nonlinear-full-capture.json').exists()
assert not (PACKAGE/'results-wt/evidence-recovery/nonlinear-sources').exists()
assert not (PACKAGE/'hybrid/literature/cache').exists()
env=dict(os.environ,PYTHONPATH=str(PACKAGE/'results-wt/evidence-recovery/composites-network-guard')+':'+str(PACKAGE),
         COMPOSITES_ISOLATED='1',PYTHONHASHSEED='0',BIGBRANE_LIVE='0')
with (REPORT/f'nonlinear-{LABEL}.log').open('w') as out:
    child=subprocess.Popen([sys.executable,'results-wt/evidence-recovery/nonlinear_verify.py',LABEL],
                           cwd=PACKAGE,env=env,stdout=out,stderr=subprocess.STDOUT)
    while child.poll() is None:
        print('isolated nonlinear verification running',flush=True);time.sleep(15)
    code=child.returncode
for suffix in ('exit','phase7-rows.json','junit.xml'):
    filename=f'nonlinear-{LABEL}.exit' if suffix=='exit' else f'nonlinear-{LABEL}-{suffix}'
    shutil.copy2(PACKAGE/'results-wt/evidence-recovery'/filename,REPORT/filename)
(REPORT/f'nonlinear-{LABEL}-inventory.json').write_text(json.dumps({'owned_files':inventory,'exit_code':code,
    'raw_capture_in_package':False,'publication_bodies_in_package':False,'literature_cache_in_package':False,
    'network_prohibited':True,'external_original_runs_masked_absent_and_open_prohibited':True},indent=2)+'\n')
raise SystemExit(code)
