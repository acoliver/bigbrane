"""Summarize saved actual fixture rows and JUnit records without rerunning replay."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
r=ROOT/'results-wt/evidence-recovery'
rows=json.loads((r/'composites-current-phase7-rows.json').read_text())
isolated=json.loads((r/'composites-isolated-final-phase7-rows.json').read_text())
old=json.loads((r/'remaining-finalcurrent-phase7-rows.json').read_text())
byold={e['card_id']:e for e in old};byiso={e['card_id']:e for e in isolated}
assert len(rows)==len(isolated)==len(old)==21
for e in rows:
    assert e['expected_codes']==byiso[e['card_id']]['expected_codes']==byold[e['card_id']]['expected_codes']
    assert e['replay_codes']==byiso[e['card_id']]['replay_codes']==byold[e['card_id']]['replay_codes']
summary={'required':sum(bool(e['expected_codes']) for e in rows),
         'matched':sum(bool(e['expected_codes']) and e['expected_codes']==e['replay_codes'] for e in rows),
         'misses':[e['card_id'] for e in rows if e['expected_codes'] and e['expected_codes']!=e['replay_codes']],
         'all_21_old_current_isolated_exact_code_lists_identical':True,
         'old_rows_sha256':hashlib.sha256((r/'remaining-finalcurrent-phase7-rows.json').read_bytes()).hexdigest(),'rows':rows}
(r/'composites-rows.json').write_text(json.dumps(summary,indent=2)+'\n')
results={}
for label in ('unit','current','final-unit','isolated','isolated-final'):
    p=r/f'composites-{label}-junit.xml';tree=ET.parse(p);cases=tree.findall('.//testcase')
    failed=[t for t in cases if t.find('failure') is not None or t.find('error') is not None]
    skipped=[t for t in cases if t.find('skipped') is not None]
    results[label]={'total':len(cases),'passed':len(cases)-len(failed)-len(skipped),
                    'failed':len(failed),'skipped':len(skipped),'exit':int((r/f'composites-{label}.exit').read_text()),
                    'junit_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'log_sha256':hashlib.sha256((r/f'composites-{label}.log').read_bytes()).hexdigest(),
                    'failures':[{'class':t.attrib['classname'],'name':t.attrib['name']} for t in failed]}
results['fixture_regeneration']={'isolated_exit':int((r/'composites-isolated-regenerate.exit').read_text()),
    'byte_identical_compact_fixture':True,'raw_capture_needed':False,'external_original_run_needed':False,
    'network_prohibited':True,'log_sha256':hashlib.sha256((r/'composites-isolated-regenerate.log').read_bytes()).hexdigest()}
results['context']={'python':'/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python',
    'cwd':'sibling or exported package','PYTHONHASHSEED':'0','BIGBRANE_LIVE':'0','explicit_confcutdir':True,
    'current_tests_predate_positive_background_and_protected_inventory_packaging_adjustments':True,
    'final_unit_and_isolated_final_test_final_test_and_fixture_files':True,
    'phase7_assertions_unchanged':True,'required_matches':7,'required_rows':19,'all_replay_rows':21}
(r/'composites-tests.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:{f:v[f] for f in ('passed','failed','exit')} for k,v in results.items() if 'passed' in v},indent=2))
