"""Source-bound negative evidence, deliberately separate from accepted Phase7 declarations."""
import ast
import copy
import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'hybrid/tests/research/fixtures/reference_replays'
OBSERVER = ROOT / 'results-wt/evidence-recovery/remaining_capture.py'
CARDS = ('card-2c5dd003', 'card-6fc7efec', 'card-dc69816b')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(name):
    return json.loads((FIXTURES / f'{name}.json').read_text())


def evidence_for(entry):
    """Negative evidence cannot provide a comparator declaration or a death code."""
    record = load('remaining_evidence')['cards'].get(entry['card_id'])
    if record is None:
        return None
    if any(digest(entry[k]) != record['specimen'][f'{k}_sha256'] for k in ('card', 'critic')):
        return None
    return copy.deepcopy(record)


@lru_cache(maxsize=1)
def full_capture():
    manifest = load('remaining_sources')
    for record in manifest['cards'].values():
        p = record['provenance']
        assert hashlib.sha256((ROOT / p['source']).read_bytes()).hexdigest() == p['sha256']
        assert hashlib.sha256(OBSERVER.read_bytes()).hexdigest() == p['observer_sha256']
    raw = ROOT / 'results-wt/evidence-recovery/remaining-full-capture.json'
    if raw.exists():
        data = raw.read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['capture_sha256']
        result = json.loads(data)
    else:
        result = reproduce_capture()
    for card, record in result.items():
        assert record['provenance'] == manifest['cards'][card]['provenance']
        p = record['provenance']
        assert digest(record['episodes']) == p['trace_sha256']
        assert digest([{'parameters': e['parameters'], 'inputs': e['inputs']} for e in record['episodes']]) == p['draws_sha256']
        assert digest(record['soft_contracts']) == p['soft_contracts_sha256']
        assert digest(record['isotope_draws']) == p['isotope_draws_sha256']
    return result


def reproduce_capture():
    """The subprocess fixes only the unsaved process hash seed, never program bytes or NumPy seeds."""
    program = '''import json,runpy,sys
from pathlib import Path
root=Path(sys.argv[1]); ns=runpy.run_path(str(root/'results-wt/evidence-recovery/remaining_capture.py'))
f=root/'hybrid/tests/research/fixtures/reference_replays'
manifest=json.loads((f/'remaining_sources.json').read_text()); records=manifest['cards']
for card,record in records.items():
    source=root/record['provenance']['source']
    observed=ns['capture'](card,source.read_text(),str(source))
    assert observed['reexecuted_stdout']==record['reexecuted_stdout']
    record.update(observed)
json.dump(records,sys.stdout,separators=(',',':'))
'''
    completed = subprocess.run([sys.executable, '-c', program, str(ROOT)], check=True, capture_output=True,
                               text=True, env=dict(os.environ, PYTHONHASHSEED='0'), timeout=120)
    return json.loads(completed.stdout)


def original_definitions(card, mutation=None, p=7):
    """Extract only unchanged CARD definitions, excluding the original model comparator and driver."""
    path = ROOT / load('remaining_sources')['cards'][card]['provenance']['source']
    text = path.read_text()
    wanted = {'Card'} if card == 'card-6fc7efec' else {'card_run', 'gelu', 'ident', 'softmax'}
    definitions = '\n\n'.join(ast.get_source_segment(text, node) for node in ast.parse(text).body
                              if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted)
    if mutation is not None:
        old, new = mutation
        assert definitions.count(old) == 1
        definitions = definitions.replace(old, new)
    ns = {'np': np, 'math': math, 'p': p, 'V': p+5, 'U': np.full(p+5, 1/(p+5)),
          'ROW0': p, 'ROW1': p+1, 'COL0': p+2, 'MUL': p+3, 'ASK': p+4}
    exec(compile(definitions, '<original-card-definitions>', 'exec'), ns)
    return ns


def run_original(card, episode, mutation=None):
    p = episode['parameters']
    ns = original_definitions(card, mutation, p=p.get('p', 7))
    if card == 'card-2c5dd003':
        return ns['card_run'](*(np.asarray(p[k]) for k in ('T', 'pi0', 'B0')), episode['inputs'])
    if card == 'card-dc69816b':
        pr = {k: np.asarray(v) if k in ('E', 'U', 'Wo', 'P') else v for k, v in p.items()}
        pr['phi'] = ns[pr['phi']]
        return ns['card_run'](pr, episode['inputs'])
    obj = ns['Card']()
    return np.asarray([obj.step(x) for x in episode['inputs']]), obj


def smoothing_posteriors(T, initial, emission_columns):
    """Forward/backward identity underlying Cappé 2011, section 2.2, equations 4-7.

    Emission columns are observed CARD bets. This is an E-step diagnostic conditional
    on that whole parameter path, not an independently published learner for the CARD.
    """
    T, initial, likelihoods = np.asarray(T), np.asarray(initial), np.asarray(emission_columns)
    forward = []
    for i, col in enumerate(likelihoods):
        weights = (initial if i == 0 else forward[-1] @ T) * col
        forward.append(weights / weights.sum())
    beta = np.ones(len(initial)); result = [None] * len(forward)
    for i in reversed(range(len(forward))):
        weights = forward[i] * beta
        result[i] = weights / weights.sum()
        beta = T @ (likelihoods[i] * beta)
        beta /= beta.sum()
    return np.asarray(result)


def revised_paid_counts(episode, stop):
    """Revisit every paid position, deliberately unlike permanent lag-one deposits."""
    p = episode['parameters']; stream = episode['inputs'][:stop]
    columns = [np.asarray(state['Phi'])[:, x] for state, x in zip(episode['states'][:stop], stream)]
    posterior = smoothing_posteriors(p['T'], p['pi0'], columns)
    counts = np.zeros_like(p['B0'], dtype=float)
    for i, x in enumerate(stream[:-1]):
        counts[:, x] += posterior[i]
    return counts, posterior


def boolean_find_s(positive_examples):
    """Mitchell 1997, chapter 2, slide 8: retain each equal Boolean attribute constraint."""
    examples = np.asarray(positive_examples, dtype=int)
    return [int(column[0]) if np.all(column == column[0]) else None for column in examples.T]


def hypothesis_matches(hypothesis, example):
    return all(constraint is None or constraint == value for constraint, value in zip(hypothesis, example))


def nge_single_exemplar_prediction(residual):
    """Salzberg 1991, section 3.2: with one exemplar it is necessarily the nearest one.

    The result is independent of feature/reliability weights. No learned weights
    are frozen and no claim about a complete NGE learner is made by this witness.
    """
    return np.asarray(residual, dtype=float)


def residual_handcase(values=(1., 1.4, 1.8)):
    E = np.eye(6); Wo = np.zeros((6, 6)); P = np.zeros((6, 6))
    for cue, target, rho in zip(range(3), range(3, 6), values):
        Wo[:, cue] = E[target] - rho * E[5]
        P[0, cue] = 2; P[cue+1, cue] = 1
    Wo[:, 3] = E[1]; Wo[:, 4] = E[2]
    P[4, 3:] = 2; P[5, 3:] = 1
    return {'parameters': {'E': E.tolist(), 'U': E[None].tolist(), 'Wo': Wo.tolist(), 'P': P.tolist(),
                            'phi': 'ident', 'k': 2, 'B': 4, 'm': 1, 'tau': .5, 'beta': 1.},
            'inputs': [0, 3, 1, 4, 2, 5]}


def observe_original(card, episode):
    """Observe isolated handcases through the same original definitions, with no comparator."""
    states = []

    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        raise TypeError(type(value).__name__)

    def snapshot(value):
        return json.loads(json.dumps(value, default=plain))

    text = (ROOT / load('remaining_sources')['cards'][card]['provenance']['source']).read_text()
    definition = next(ast.get_source_segment(text, node) for node in ast.parse(text).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'card_run')
    # The extracted function retains its relative line numbers in the concatenated definitions.
    ns = original_definitions(card)
    code = ns['card_run'].__code__
    own_lines = definition.splitlines()
    marker = next(i for i, line in enumerate(own_lines) if
                  (line.strip().startswith('ps.append(p.copy())') if card == 'card-2c5dd003'
                   else line.strip() == 'for t in range(n):')) + code.co_firstlineno

    def observer(frame, event, arg):
        if frame.f_code is not code:
            return None
        v = frame.f_locals
        if card == 'card-2c5dd003' and event == 'line' and frame.f_lineno == marker:
            states.append(snapshot({k: v[k] for k in ('t', 'x', 'C', 'R', 'M', 'Phi', 'q', 'p')}))
        elif card == 'card-dc69816b' and ((event == 'line' and frame.f_lineno == marker) or event == 'return'):
            if 't' in v and len(states) == v['t']:
                states.append(snapshot({k: v[k] for k in ('t', 'H', 'u', 'z', 'kappa', 'h', 'ev')}))
        return observer

    p = episode['parameters']; previous = sys.gettrace()
    try:
        sys.settrace(observer)
        if card == 'card-2c5dd003':
            result = ns['card_run'](*(np.asarray(p[k]) for k in ('T', 'pi0', 'B0')), episode['inputs'])
            ps, hs, ss, pn = snapshot(result)
            return dict(copy.deepcopy(episode), states=states, card_outputs=ps, filtered=hs,
                        lag_one_credits=ss, next_output=pn)
        pr = {k: np.asarray(v) if k in ('E', 'U', 'Wo', 'P') else v for k, v in p.items()}
        pr['phi'] = ns[pr['phi']]
        hs, ps, ev = snapshot(ns['card_run'](pr, episode['inputs']))
        return dict(copy.deepcopy(episode), states=states, hidden=hs, card_outputs=ps, events=ev)
    finally:
        sys.settrace(previous)
