"""Diagnostic composite evidence; nothing here supplies an admitted host operation."""
import ast
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT/'hybrid/tests/research/fixtures/reference_replays'
CARDS = ('card-14429dbc', 'card-68612a2a', 'card-8dc4db49')
OBSERVER = ROOT/'results-wt/evidence-recovery/composites_capture.py'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(name):
    return json.loads((FIXTURES/f'composites_{name}.json').read_text())


def evidence_for(entry):
    evidence = load('evidence')['cards'].get(entry['card_id'])
    if evidence is None or any(digest(entry[k]) != evidence['specimen'][f'{k}_sha256'] for k in ('card', 'critic')):
        return None
    return copy.deepcopy(evidence)


def reproduce_capture():
    manifest = load('sources')
    for record in manifest['cards'].values():
        p = record['provenance']
        assert hashlib.sha256((ROOT/p['source']).read_bytes()).hexdigest() == p['source_sha256']
        assert hashlib.sha256(OBSERVER.read_bytes()).hexdigest() == p['observer_sha256']
    program = '''import json,runpy,sys
from pathlib import Path
root=Path(sys.argv[1]); ns=runpy.run_path(str(root/'results-wt/evidence-recovery/composites_capture.py'))
manifest=json.loads((root/'hybrid/tests/research/fixtures/reference_replays/composites_sources.json').read_text())
records=manifest['cards']
for card,record in records.items():
 source=root/record['provenance']['source']
 observed=ns['capture'](card,source.read_text(),str(source))
 assert observed['reexecuted_stdout']==record['reexecuted_stdout']
 record.update(observed)
json.dump(records,sys.stdout,separators=(',',':'))
'''
    completed = subprocess.run([sys.executable, '-c', program, str(ROOT)], capture_output=True, text=True,
                               check=True, env=dict(os.environ, PYTHONHASHSEED='0'), timeout=120)
    result = json.loads(completed.stdout)
    for card, record in result.items():
        p = manifest['cards'][card]['provenance']
        assert digest(record['episodes']) == p['trace_sha256']
        assert digest([{'parameters': e['parameters'], 'inputs': e['inputs']} for e in record['episodes']]) == p['draws_sha256']
    return result


@lru_cache(maxsize=1)
def full_capture():
    return reproduce_capture()


def original_definitions(card, parameters, mutation=None):
    p = load('sources')['cards'][card]['provenance']
    source = (ROOT/p['source']).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == p['source_sha256']
    if mutation:
        old, new = mutation
        assert source.count(old) == 1
        source = source.replace(old, new)
    names = {'unif', 'onehot', 'Card'} if card == CARDS[0] else {'softmax_logp', 'Card'} if card == CARDS[1] else {'card_run'}
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    ns = {'np': np, 'math': math, 'LABELS': list(range(5))}
    ns.update({k: np.asarray(v) if isinstance(v, list) and k in ('C','E','F','u','l','E_raw','F_raw','vals') else v
               for k,v in parameters.items()})
    exec(compile(tree, str(ROOT/p['source']), 'exec'), ns)
    return ns


def run_original(card, episode, mutation=None):
    p = episode['parameters']; ns = original_definitions(card, p, mutation)
    if card == CARDS[2]:
        return ns['card_run'](episode['inputs'])
    obj = ns['Card'](p['d'], p['B']) if card == CARDS[0] else ns['Card']()
    outputs, states = [], []
    for token in episode['inputs']:
        if card == CARDS[0]:
            obj.step(token)
            probes = set(p['extra_probes']) | {x for x,_ in obj.E}
            if token[0] in ('DEMO','QUERY'):
                probes.add(token[1])
            outputs.append([{'x': x, 'p': obj.read(x)} for x in sorted(probes)])
        else:
            outputs.append(obj.step(*token))
        states.append(copy.deepcopy(vars(obj)))
    return outputs, states


def lattice_closure(records):
    """HSW's integer span, in dimension one, applied to all within-label translations."""
    translations = [abs(x-z) for x,y in records for z,w in records if y == w]
    generator = math.gcd(*translations)
    residues = {}
    valid = True
    for x,y in records:
        r = x % generator if generator else x
        if r in residues and residues[r] != y:
            valid = False
        residues[r] = y
    return generator, residues if valid else {}, valid


def dense_binding_sweeps(parameters, states, inputs, cold=False):
    """Independent dense mass-action arithmetic, not a claimed published stream learner."""
    p = parameters; base = np.asarray(p['E']) @ np.asarray(p['F']).T
    x = np.zeros(p['V']); y = x.copy(); a = x.copy()
    result, sweeps = [], []
    for w, state in zip(inputs, states):
        if cold:
            x = np.array(state['a']); y = x.copy()
        elif a[w] == 0:
            x[w] = y[w] = 1.
        a = np.array(state['a'])
        counts = np.zeros_like(base)
        for i,j,c,_,_ in state['table']:
            counts[i,j] = c
        kernel = base * np.exp(p['beta']*counts)
        trace = []
        for _ in range(p['J']):
            y = a/(1+kernel.T@x)
            x = a/(1+kernel@y)
            trace.append({'x':x.copy(), 'y':y.copy()})
        result.append(x[w]*(kernel[w]*y+np.asarray(p['u'])))
        sweeps.append(trace)
    return np.asarray(result), sweeps


def superglue_one_pair(kernel, dustbin=1.):
    """SuperGlue equations 8-9: both 2x2 augmented marginals are (1,1)."""
    matched = math.sqrt(kernel)/(math.sqrt(kernel)+math.sqrt(dustbin))
    return np.array([[matched, 1-matched], [1-matched, matched]])


def pinned_one_pair(kernel):
    """Pinned dustbin potentials: x + kernel*x*y = y + kernel*x*y = 1."""
    free = 2/(1+math.sqrt(1+4*kernel))
    return np.array([[kernel*free*free, free], [free, 1.]])
