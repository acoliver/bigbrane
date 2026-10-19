"""Replay only the enumerated local saved programs; append trace capture, never change card computations."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / 'hybrid/tests/research/fixtures'
OUT = FIX / 'reference_replays'


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


def execute(path, code, changes=()):
    source = path.read_text()
    text = code
    for before, after in changes:
        assert text.count(before) == 1, (path, before)
        text = text.replace(before, after)
    ns = {'captured': [], '__name__': '__saved_replay__'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(text, str(path), 'exec'), ns)
    return ns, {'source': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                'sha256': hashlib.sha256(source.encode()).hexdigest(),
                'capture_insertions': [{'before': a, 'after': b} for a, b in changes],
                'executed_source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'method': 'original saved card computations re-executed; trace snapshots appended at stated boundaries'}


def evidence(operation, samples, provenance):
    assert samples
    return {'reference_spec': {'operation': operation, 'parameters': samples[0]['parameters']},
            'reference_samples': samples, 'provenance': provenance}


saved = OUT/'programs'
all_evidence = {}
for card, operation, marker, capture in [
 ('card-ea50345f', 'jeffrey_ipf', '    yc = card_run(M, mu, lab, xs)',
  "{'parameters': {'p0': [1/d]*d, 'masks': M, 'targets': mu, 'labels': lab, 'label_count': V}, 'inputs': xs, 'card_outputs': yc}"),
 ('card-17100004', 'inside_pcfg', '    yc = np.array(card_run(x, P, E, S))',
  "{'parameters': {'binary': P, 'lexical': E, 'start': S}, 'inputs': x, 'card_outputs': yc}"),
 ('card-17100005', 'earley', '    yc = readout(Cc)',
  "{'parameters': {'grammar': G, 'start': 'S'}, 'inputs': list(s), 'card_outputs': yc}"),
 ('card-17100006', 'viterbi_max', '    c = card_run(pi, A, e, xs)',
  "{'parameters': {'initial': pi, 'transition': A, 'emission': e}, 'inputs': xs, 'card_outputs': c}"),
 ('card-17100009', 'selective_diagonal', '            yc = card(X, Wf, bf, Wu)',
  "{'parameters': {'Wf': Wf, 'bf': bf, 'Wu': Wu}, 'inputs': X, 'card_outputs': yc}"),
 ('card-1710000a', 'delta_rule', '        yc = card_run(K, V, Q, B)',
  "{'parameters': {'S0': np.zeros((dv,dk))}, 'inputs': [{'key':k, 'value':v, 'query':q, 'beta':b} for k,v,q,b in zip(K,V,Q,B)], 'card_outputs': yc}"),
]:
    path = saved/(card+'.py');code=path.read_text();indent=marker[:len(marker)-len(marker.lstrip())]
    ns, prov = execute(path,code,[(marker,marker+'\n'+indent+'captured.append('+capture+')')])
    all_evidence[card] = evidence(operation, ns['captured'], prov)

for card, operation, fn, inputs in [
 ('card-17100001','dyck2','card_run','seqs'),
 ('card-17100002','queue_copy','card_outputs','inputs'),
 ('card-17100003','two_counter_abc','card_run','inputs'),
]:
    path=saved/(card+'.py');ns,prov=execute(path,path.read_text())
    samples=[]
    for seq in ns[inputs]:
        if not seq: continue
        inp=[['(', '[', ')', ']'].index(x) for x in seq] if operation=='dyck2' else list(seq)
        outputs=ns[fn](seq)
        if operation=='dyck2': outputs=[[y] for y in outputs]
        samples.append({'parameters':{},'inputs':inp,'card_outputs':outputs})
    all_evidence[card]=evidence(operation,samples,prov)

from hybrid.research.stages.sequitur_reference import grammar_vector
path=saved/'card-17100007.py'
ns,prov=execute(path,path.read_text())
samples=[]
for seq in ns['seqs']:
    card=ns['CardGrammar']()
    outputs=[]
    for symbol in seq:
        card.append(symbol)
        outputs.append(grammar_vector(card.grammar(),4*len(seq)+4))
    samples.append({'parameters':{},'inputs':list(seq),'card_outputs':outputs})
all_evidence['card-17100007']=evidence('sequitur',samples,prov)

fix12={}
base=FIX/'phase7_live_20261014T2051Z'
path=base/'job-e284e0a16e43.json';code=json.loads(path.read_text())['files']['main.py']
marker='    # ---- claimed known form: causal linear attention (Katharopoulos et al. 2020),'
insert="    captured.append({'parameters': {'theta': Theta, 'bias':u, 'beta':beta, 'tau':tau}, 'inputs':x, 'card_outputs':card_p})\n"
ns,prov=execute(path,code,[(marker,insert+marker)])
for id,op in [('la_published','diagonal_attention_count'),('la_phase_normalized','diagonal_attention_phase')]:
 fix12[id]=evidence(op,ns['captured'],prov)
path=base/'job-720f18f6a55a.json';code=json.loads(path.read_text())['files']['main.py']
marker='        p_card = softmax(W_o @ y)'
insert="\n        if t == 1:\n            captured.append({'parameters': {'phi':Phi, 'output':W_o, 'gate':w_beta, 'mixture':A, 'ridge':lam}, 'inputs':[], 'card_outputs':[]})\n        captured[-1]['inputs'].append([int(x[t-1]),int(x[t])])\n        captured[-1]['card_outputs'].append(p_card.copy())"
ns,prov=execute(path,code,[(marker,marker+insert)])
for id,op in [('mesanet_single','mesa_single'),('mesanet_hop_mixture','mesa_mixture')]:
 fix12[id]=evidence(op,ns['captured'],prov)
path=base/'job-bf095b15bbd6.json';code=json.loads(path.read_text())['files']['main.py'];ns,prov=execute(path,code)
p={'rho':[[ns['rho_of'](i,j) for j in range(ns['V'])] for i in range(ns['V'])], 'prior':ns['u'], 'eta':ns['eta'], 'threshold':ns['kappa'], 'capacity':ns['Pmax']}
samples=[{'parameters':p,'inputs':seq,'card_outputs':ns['run_card'](seq)} for seq in ns['seqs']]
for id,op in [('hard_bimap','bimap_hard'),('weighted_bimap','bimap_weighted')]:fix12[id]=evidence(op,samples,prov)
path=FIX/'phase7_live_20261014T1752Z/job-137fb29ee378.json'
code=json.loads(path.read_text())['files']['main.py']
marker='    o1, _ = card_run(seq, Vb, rho_kv, ub, etab, 64)'
insert="\n    captured.append({'parameters': {'rho':[[rho_kv(i,j) for j in range(Vb)] for i in range(Vb)], 'prior':ub, 'eta':etab, 'threshold':kappa, 'capacity':64}, 'inputs':seq, 'card_outputs':o1})"
ns,prov=execute(path,code,[(marker,marker+insert)])
fix10={'hard_bimap':evidence('bimap_hard',ns['captured'],prov)}

phase7={}
for card,op,marker,capture in [
 ('card-1184f7a7','diagonal_attention_epsilon_logits_probs','    lc, pc = card_model(x, Theta, u, beta)',
  "{'parameters':{'theta':Theta,'bias':u,'beta':beta,'epsilon':1e-9},'inputs':x,'card_outputs':np.concatenate([lc,pc],axis=1)}"),
 ('card-537e161c','mesa_mixture_logits','    a = card_model(seq, Phi, W_o, w_beta, A, lam, M)',
  "{'parameters':{'phi':Phi,'output':W_o,'gate':w_beta,'mixture':A,'ridge':lam},'inputs':list(zip(seq[:-1],seq[1:])),'card_outputs':a}"),
]:
    path=saved/('phase7-'+card+'.py')
    ns,prov=execute(path,path.read_text(),[(marker,marker+'\n    captured.append('+capture+')')])
    phase7[card]=evidence(op,ns['captured'],prov)

for name,data in [('kd8',all_evidence),('fix12',fix12),('fix10',fix10),('phase7',phase7)]:
 (OUT/(name+'.json')).write_text(json.dumps(data,default=plain,separators=(',',':'))+'\n')
 print(name,len(data),'records', (OUT/(name+'.json')).stat().st_size,'bytes')

manifest={}
for record in json.loads((FIX/'kd8/replay_records.json').read_text()):
    if record['card'] not in all_evidence: continue
    source=Path('/Volumes/XS1000/acoliver/projects')/record['source']
    artifact=json.loads(source.read_text())
    original=source.parents[2]/'jobs'/artifact['reduction_check']['job_id']/'in/main.py'
    copied=saved/(record['card']+'.py')
    assert copied.read_bytes()==original.read_bytes()
    manifest[record['card']]={'original':str(original),'fixture':str(copied.relative_to(ROOT)),
                             'sha256':hashlib.sha256(copied.read_bytes()).hexdigest()}
(OUT/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
(OUT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n' for p in files))
