import json,os,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
python='/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python'
tools=['results-wt/kd8/tools/tests/test_evidence_tools.py','results-wt/retrieval-diagnosis/test_diagnose.py','results-wt/transport-measurement/test_measure.py','results-wt/acquisition-windows/test_extract_windows.py']
isolated=['hybrid/tests/literature']+['hybrid/tests/research/'+n for n in ['test_nonlinear_historical_evidence.py','test_composites_historical_evidence.py','test_remaining_historical_evidence.py','test_next_historical_references.py','test_historical_binding.py','test_historical_references.py','test_reference_replays.py','test_review_gate_evidence.py','test_phase7_replay.py']]+tools+['results-wt/heldout-acquisition/test_harness.py']
started=time.monotonic()
results={}
for label,root,targets in [('combined',r,['hybrid/tests/research','hybrid/tests/literature']+tools+['results-wt/heldout-acquisition/test_harness.py']),('isolated',out/'package',isolated)]:
 env=dict(os.environ,BIGBRANE_LIVE='0',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'),NEXT_ROWS_OUTPUT=str(out/(label+'-phase7-rows.json')),PYTHONPATH=':'.join(str(root/p) for p in ['results-wt/evidence-recovery/nonlinear-checkpoint-network-guard','','results-wt/evidence-recovery','results-wt/transport-measurement','results-wt/acquisition-windows','results-wt/heldout-acquisition']))
 if label=='isolated':env['COMPOSITES_ISOLATED']='1'
 command=[python,'-m','pytest','-c',str(root/'pyproject.toml'),'--confcutdir='+str(root),'-q','-p','next_results','--basetemp='+str(out/(label+'-pytest')),'--junitxml='+str(out/(label+'-junit.xml'))]+targets
 (out/(label+'-command.json')).write_text(json.dumps({'cwd':str(root),'command':command,'environment':{k:env[k] for k in ['BIGBRANE_LIVE','PYTHONHASHSEED','PYTHONPATH','TMPDIR','NEXT_ROWS_OUTPUT']}},indent=2)+'\n')
 with (out/(label+'.log')).open('w') as log:
  child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (out/(label+'.pid')).write_text(str(child.pid)+'\n')
  while child.poll() is None:
   elapsed=time.monotonic()-started
   print(label,'pid',child.pid,'elapsed',round(elapsed),flush=True)
   (out/'heartbeat.json').write_text(json.dumps({'phase':label,'pid':child.pid,'elapsed_seconds':elapsed})+'\n')
   if elapsed>1800:
    (out/'status').write_text('LIMIT: process still running; inspect persisted PID/log before continuing\n');sys.exit(124)
   time.sleep(30)
 results[label]=child.returncode
 (out/(label+'.exit')).write_text(str(child.returncode)+'\n')
 (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 if child.returncode not in [0,1]:break
(out/'status').write_text('finished '+json.dumps(results)+'\n')
