"""Persist tested-file receipts, actual row accounting and a compact delivery inventory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'results-wt/evidence-recovery'
FIXTURES=ROOT/'hybrid/tests/research/fixtures/reference_replays'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def git(path,*args):
    return subprocess.check_output(['git',*args],cwd=path,text=True).strip()


results={}
for label in ('unit','unit-final','current','isolated','final','delivery','isolated-delivery'):
    path=REPORT/f'nonlinear-{label}-junit.xml';root=ET.parse(path).getroot()
    cases=list(root.iter('testcase'));failed=[c for c in cases if c.find('failure') is not None]
    results[label]={'tests':len(cases),'passed':len(cases)-len(failed),'failed':len(failed),
        'failures':[c.attrib['classname']+'.'+c.attrib['name'] for c in failed],
        'exit':int((REPORT/f'nonlinear-{label}.exit').read_text()),
        'log':str((REPORT/f'nonlinear-{label}.log').relative_to(ROOT)),
        'junit':str(path.relative_to(ROOT)),'junit_sha256':sha(path)}
rows=json.loads((REPORT/'nonlinear-delivery-phase7-rows.json').read_text())
old=json.loads((REPORT/'composites-current-phase7-rows.json').read_text())
isolated=json.loads((REPORT/'nonlinear-isolated-delivery-phase7-rows.json').read_text())
assert len(rows)==len(old)==len(isolated)==21
assert [(r['card_id'],r['replay_codes']) for r in rows]==[(r['card_id'],r['replay_codes']) for r in old]==[(r['card_id'],r['replay_codes']) for r in isolated]
matched=sum(bool(r['expected_codes']) and r['expected_codes']==r['replay_codes'] for r in rows)
assert matched==7
accounting={'rows':21,'required':19,'matched':matched,'remaining':19-matched,'all_codes_unchanged':True,
            'gap_cards':[r['card_id'] for r in rows if r['expected_codes'] and r['expected_codes']!=r['replay_codes']],
            'actual_rows':str((REPORT/'nonlinear-delivery-phase7-rows.json').relative_to(ROOT))}
dump(REPORT/'nonlinear-tests.json',{'executions':results,'phase7':accounting,
    'network_prohibited':True,'full_suite_run':False,'independent_review_run':False})
protected=json.loads((FIXTURES/'nonlinear_protected.json').read_text())
assert all(sha(ROOT/name)==wanted for name,wanted in protected.items())
main=ROOT.parent/'bigbrane'
assert git(main,'status','--porcelain')=='' and git(main,'rev-parse','--short','HEAD')=='8347c7f'
assert git(ROOT,'diff','--name-only')=='' and git(ROOT,'diff','--cached','--name-only')==''
assert git(ROOT,'rev-parse','--short','HEAD')=='4dd4864'
assert git(ROOT,'ls-files','results-wt/evidence-recovery/nonlinear-full-capture.json')==''
safety={'main_branch':git(main,'branch','--show-current'),'main_head':git(main,'rev-parse','HEAD'),'main_clean':True,
    'sibling_branch':git(ROOT,'branch','--show-current'),'sibling_head':git(ROOT,'rev-parse','HEAD'),
    'preexisting_tracked_files_unchanged':len(protected),'tracked_diff':[],'staged_diff':[],
    'raw_capture_tracked':False,'no_new_operation_or_schema_or_acceptance_edits':True,
    'source_receipts_checked':True,'checked_utc':datetime.now(timezone.utc).isoformat()}
dump(REPORT/'nonlinear-safety.json',safety)
local=[REPORT/'nonlinear-full-capture.json',*sorted((REPORT/'nonlinear-sources').glob('*')),
       *sorted(REPORT.glob('nonlinear-*.log')),*sorted(REPORT.glob('nonlinear-*-junit.xml')),
       REPORT/'nonlinear-isolated-inventory.json']
local=list(dict.fromkeys(p for p in local if p.is_file() and p.name!='nonlinear-inventory.log'))
(REPORT/'nonlinear-LOCAL-SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in local))
required=[ROOT/'hybrid/tests/research/nonlinear_replays.py',ROOT/'hybrid/tests/research/test_nonlinear_historical_evidence.py',
          *sorted(FIXTURES.glob('nonlinear_*.json')),*sorted((FIXTURES/'programs').glob('nonlinear-*.py')),
          *[REPORT/name for name in ('nonlinear_capture.py','nonlinear_measure.py','nonlinear_verify.py','nonlinear_package.py',
              'nonlinear_fetch.py','nonlinear_inventory.py','nonlinear-STATUS.md','NONLINEAR.md','nonlinear-tests.json',
              'nonlinear-safety.json','nonlinear-delivery-phase7-rows.json','nonlinear-isolated-delivery-phase7-rows.json',
              'nonlinear-LOCAL-SHA256SUMS')],
          REPORT/'composites-network-guard/sitecustomize.py']
inventory={'required_files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(required)},
    'local_only_capture_and_publications':{'capture_bytes':(REPORT/'nonlinear-full-capture.json').stat().st_size,
        'capture_sha256':sha(REPORT/'nonlinear-full-capture.json'),'publication_bodies':3,
        'required_for_tests':False,'hash_receipts':'results-wt/evidence-recovery/nonlinear-LOCAL-SHA256SUMS'},
    'phase7':accounting,'tests':results,'safety':safety}
dump(REPORT/'nonlinear-INVENTORY.json',inventory)
required.append(REPORT/'nonlinear-INVENTORY.json')
(REPORT/'nonlinear-SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in sorted(required)))
print(json.dumps({'required_files':len(required),'bytes':sum(p.stat().st_size for p in required),
                  'protected_files':len(protected),'matched':matched,'remaining':19-matched,'executions':results},indent=2))
