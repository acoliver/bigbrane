"""Non-admitting diagnostics for the masked pair solver and complex fragment learner."""
import ast
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT/'hybrid/tests/research/fixtures/reference_replays'
CARDS = ('card-a36f1a3a', 'card-ddb071bd')
OBSERVER = ROOT/'results-wt/evidence-recovery/nonlinear_capture.py'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(name):
    return json.loads((FIXTURES/f'nonlinear_{name}.json').read_text())


def evidence_for(entry):
    evidence = load('evidence')['cards'].get(entry['card_id'])
    if evidence is None or any(digest(entry[k]) != evidence['specimen'][f'{k}_sha256'] for k in ('card','critic')):
        return None
    return copy.deepcopy(evidence)


@lru_cache(maxsize=1)
def full_capture():
    manifest = load('sources')
    for record in manifest['cards'].values():
        p = record['provenance']
        assert hashlib.sha256((ROOT/p['source']).read_bytes()).hexdigest() == p['source_sha256']
        assert hashlib.sha256(OBSERVER.read_bytes()).hexdigest() == p['observer_sha256']
    program = '''import json,runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);ns=runpy.run_path(str(root/'results-wt/evidence-recovery/nonlinear_capture.py'))
records=json.loads((root/'hybrid/tests/research/fixtures/reference_replays/nonlinear_sources.json').read_text())['cards']
for card,record in records.items():
    source=root/record['provenance']['source']
    observed=ns['capture'](card,source.read_text(),str(source))
    assert observed['reexecuted_stdout']==record['reexecuted_stdout']
    record.update(observed)
json.dump(records,sys.stdout,separators=(',',':'))
'''
    completed = subprocess.run([sys.executable,'-c',program,str(ROOT)],capture_output=True,text=True,check=True,
                              env=dict(os.environ,PYTHONHASHSEED='0'),timeout=120)
    records = json.loads(completed.stdout)
    for record in records.values():
        p = record['provenance']; episodes = record['episodes']
        assert digest(episodes) == p['trace_sha256']
        assert digest([{'parameters':e['parameters'],'inputs':e['inputs']} for e in episodes]) == p['draws_sha256']
    return records


def complex_array(encoded):
    return np.asarray(encoded['real']) + 1j*np.asarray(encoded['imag'])


def definitions(card, mutation=None):
    p = load('sources')['cards'][card]['provenance'];path = ROOT/p['source']
    source = path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == p['source_sha256']
    if mutation:
        old,new = mutation
        assert source.count(old) == 1
        source = source.replace(old,new)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
    ns = {'np':np,'math':math}
    exec(compile(tree,str(path),'exec'),ns)
    return ns


def run_original(card, episode, mutation=None):
    ns = definitions(card,mutation);p = episode['parameters'];inputs = episode['inputs']
    if card == CARDS[0]:
        return ns['card_run'](inputs,p['d'],p['alpha'],p['beta'])
    return ns['card_run'](np.array(p['Theta']),inputs,p['K'],p['tau'],p['beta'],np.array(p['u']))


def observe_handcase(card, episode):
    p = load('sources')['cards'][card]['provenance'];raw = (ROOT/p['source']).read_text()
    source = 'import numpy as np, math\n' + ast.get_source_segment(raw,next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='card_run'))
    observer = runpy.run_path(str(OBSERVER))
    p = episode['parameters']
    def invoke(ns):
        if card == CARDS[0]:
            ns['card_run'](episode['inputs'],p['d'],p['alpha'],p['beta'])
        else:
            ns['card_run'](np.array(p['Theta']),episode['inputs'],p['K'],p['tau'],p['beta'],np.array(p['u']))
    return observer['capture'](card,source,'<isolated-original-card>',invocation=invoke)['episodes'][0]


def scalar_closure(evidence, alpha, beta, depth, mask=True, reinject=True):
    """Entrywise scalar diagnostic, not the historical einsum comparator or a published full learner."""
    E = np.asarray(evidence);d = len(E);H = np.zeros_like(E);trace = []
    for _ in range(depth):
        G = np.array([[sum(float(H[i,k])*float(H[k,j]) for k in range(d)) for j in range(d)] for i in range(d)])
        H = np.array([[math.tanh(alpha*(float(E[i,j]) if reinject else float(H[i,j]))+beta*float(G[i,j])) if not mask or i!=j else 0.
                       for j in range(d)] for i in range(d)])
        trace.append(H.copy())
    return H,trace


def published_ppgn_block(X, m1, m2, m3):
    """Maron et al. section 6: concatenate m3(X) and matching-channel matrix products."""
    A,B,C = m1(X),m2(X),m3(X)
    product = np.stack([A[:,:,k]@B[:,:,k] for k in range(A.shape[2])],axis=-1)
    return np.concatenate([C,product],axis=-1)


def dpmeans(points, penalty):
    """Kulis and Jordan Algorithm 1, global-mean initialization and full reassignment passes."""
    X = np.asarray(points,dtype=float)
    means = [X.mean(0)];labels = np.zeros(len(X),dtype=int);history = []
    while True:
        previous = labels.copy();old_means = np.array(means)
        for i,x in enumerate(X):
            distances = [float(np.sum((x-mu)**2)) for mu in means]
            j = int(np.argmin(distances))
            if distances[j] > penalty:
                j = len(means);means.append(x.copy())
            labels[i] = j
        # Empty categories contribute no mean; deleting them preserves every assigned cluster.
        occupied = sorted(set(labels.tolist()))
        remap = {old:new for new,old in enumerate(occupied)}
        means = [X[labels==j].mean(0) for j in occupied]
        labels = np.array([remap[j] for j in labels])
        history.append({'labels':labels.copy(),'means':np.array(means)})
        if np.array_equal(labels,previous) and np.array_equal(np.array(means),old_means):
            return labels,np.array(means),history


def fuzzy_art(points, choice, vigilance, rate=1.):
    """Survey Algorithm 1 with Fuzzy ART equations 15,16,19,22 and fast category commitment."""
    weights = [];events = []
    for raw in np.asarray(points,dtype=float):
        assert np.all((0<=raw)&(raw<=1))
        x = np.concatenate([raw,1-raw]);rejected = []
        candidates = weights+[np.ones_like(x)]
        scores = [float(np.minimum(x,w).sum()/(choice+w.sum())) for w in candidates]
        order = sorted(range(len(candidates)),key=lambda j:(-scores[j],j))
        selected = None
        for j in order:
            match = float(np.minimum(x,candidates[j]).sum()/x.sum())
            if match >= vigilance:
                selected = j
                updated = (1-rate)*candidates[j]+rate*np.minimum(x,candidates[j])
                if j == len(weights):
                    weights.append(updated)
                else:
                    weights[j] = updated
                break
            rejected.append(j)
        assert selected is not None  # The uncommitted category always passes vigilance.
        events.append({'winner':selected,'rejected':rejected,'scores':scores,'weights':np.array(weights).copy()})
    return np.array(weights),events


def closure_loss_gradient(E, a0, b0, pair, target, depth):
    """Forward sensitivities of the CARD's two learned scalars through all finite masked steps."""
    alpha,beta = np.logaddexp(0.,[a0,b0]);d = len(E);mask = 1-np.eye(d)
    H = np.zeros((d,d));DA = H.copy();DB = H.copy()
    for _ in range(depth):
        G = H@H
        nextH = mask*np.tanh(alpha*E+beta*mask*G)
        slope = mask*(1-nextH**2)
        DA,DB = slope*(E+beta*mask*(DA@H+H@DA)),slope*(mask*G+beta*mask*(DB@H+H@DB))
        H = nextH
    h = H[pair];sign = 1-2*target;prob = (1+sign*h)/2
    factor = -sign/(2*prob)
    return -math.log(prob),np.array([factor*DA[pair]/(1+math.exp(-a0)),factor*DB[pair]/(1+math.exp(-b0))])


def coherence_loss_gradient(theta, tokens, drop_negative=False, drop_distinct=False):
    """CARD's separate smooth phase-training objective, not gradients through hard assignment."""
    theta = np.asarray(theta);V,d = theta.shape;loss = 0.;gradient = np.zeros_like(theta)
    rel = [theta[b]-theta[a] for a,b in zip(tokens[:-1],tokens[1:])]
    positive = {0,3,6,9}
    for i in range(len(rel)):
        for j in range(i+1,len(rel)):
            delta = rel[j]-rel[i];c = np.cos(delta).mean();is_positive = i in positive and j in positive
            coefficient = -1. if is_positive else 1. if c>.2 and not drop_negative else 0.
            loss += 1-c if is_positive else max(0.,c-.2) if not drop_negative else 0.
            g = -coefficient*np.sin(delta)/d
            for v,mult in ((tokens[j+1],1),(tokens[j],-1),(tokens[i+1],-1),(tokens[i],1)):
                gradient[v] += mult*g
    if not drop_distinct:
        for a in range(V):
            for b in range(a+1,V):
                delta = theta[b]-theta[a];c = np.cos(delta).mean()
                loss += max(0.,c-.2)
                if c>.2:
                    g = -np.sin(delta)/d;gradient[b] += g;gradient[a] -= g
    return float(loss),gradient
