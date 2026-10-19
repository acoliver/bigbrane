"""Inventory only the selected delivery and separately preserve local capture receipts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'results-wt/evidence-recovery'
F=ROOT/'hybrid/tests/research/fixtures/reference_replays'
files=[ROOT/'hybrid/tests/research/composites_replays.py',ROOT/'hybrid/tests/research/test_composites_historical_evidence.py']
files+=list(F.glob('composites_*.json'))+list((F/'programs').glob('composites-*.py'))
files+=[REPORT/name for name in ('composites_capture.py','composites_measure.py','composites_verify.py',
    'composites_package.py','composites_summarize.py','composites_inventory.py','composites-network-guard/sitecustomize.py',
    'COMPOSITES.md','composites-STATUS.md','composites-tests.json','composites-rows.json',
    'composites-isolated-final-inventory.json')]
assert all(p.is_file() for p in files)
files=sorted(set(files))
checksums=''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n' for p in files)
(REPORT/'composites-SHA256SUMS').write_text(checksums)
files.append(REPORT/'composites-SHA256SUMS')
inventory={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
(REPORT/'composites-INVENTORY.json').write_text(json.dumps({'files':inventory,'inventory_excludes_itself':True,
    'required_new_files':len(inventory)+1,'main_head':'09dd7568f6a7ee2931859bdc0cf86adf1dcc1ffe',
    'sibling_head':'a7bb1e0a95d49688007672de47c87ac0a7d6bfe5','uncommitted':True,
    'prohibited_changes':'No existing source, tests, host contracts, schemas, expected codes, math, README, EXPERIMENT or policy changed.'},indent=2)+'\n')
files.append(REPORT/'composites-INVENTORY.json')
subprocess.run(['git','add','-N','-f','--']+[str(p.relative_to(ROOT)) for p in files],cwd=ROOT,check=True)
tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
assert all(str(p.relative_to(ROOT)) in tracked for p in files)
protected=json.loads((F/'composites_protected.json').read_text())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in protected.items())
local=[]
for p in REPORT.glob('composites*'):
    if p.is_file() and p not in files and p.name!='composites-LOCAL-SHA256SUMS':
        local.append(p)
local+=list((REPORT/'composites-sources').glob('*'))
local=[p for p in local if p.is_file()]
(REPORT/'composites-LOCAL-SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n' for p in sorted(set(local))))
print('Required files inventoried and intent-to-add:',len(files))
print('Protected files unchanged:',len(protected))
print('Local-only capture/source/log files inventoried:',len(local))
