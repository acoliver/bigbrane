"""Separate executed trace evidence for immutable historical reports. No saved verdict is rewritten."""
import copy
import hashlib
import json
import runpy
from functools import lru_cache
from pathlib import Path

FIXTURES = Path(__file__).parent / 'fixtures' / 'reference_replays'


@lru_cache(maxsize=None)
def load(name):
    if name == 'historical':
        return historical_capture()
    return json.loads((FIXTURES / f'{name}.json').read_text())


def historical_capture():
    """Load the retained full trace, or reproduce it from immutable programs and the observer."""
    manifest = load('historical_sources')
    path = FIXTURES / 'historical.json'
    if path.exists():
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['capture_sha256']
        return json.loads(data)
    root = FIXTURES.parents[4]
    observer = root / 'results-wt/evidence-recovery/capture_historical.py'
    capture = runpy.run_path(str(observer))['capture']
    recovered = copy.deepcopy(manifest['cards'])
    for card, record in recovered.items():
        provenance = record['provenance']
        source = root / provenance['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance['sha256']
        assert hashlib.sha256(observer.read_bytes()).hexdigest() == provenance['observer_sha256']
        episodes, stdout = capture(card, source.read_text(), str(source))
        assert stdout == record['reexecuted_stdout']
        record['episodes'] = episodes
    return recovered


def fix12_spec(comparator):
    evidence = load('fix12').get(comparator['id'])
    return dict(comparator, reference_spec=evidence['reference_spec']) if evidence else comparator


def kd8_record(record):
    replay = copy.deepcopy(record)
    card = record['card']
    evidence = load('kd8').get(card)
    if evidence is None:
        return replay
    ids = {
        'card-ea50345f': {'jeffrey_ipf_step': 'jeffrey_ipf'},
        'card-17100001': {'dpda_dyck2_textbook': 'dyck2', 'hard_stack_rnn': 'dyck2'},
        'card-17100002': {'queue_automaton': 'queue_copy'},
        'card-17100003': {'minsky_two_counter': 'two_counter_abc'},
        'card-17100004': {'inside_algorithm_pcfg': 'inside_pcfg'},
        'card-17100005': {'earley_1970': 'earley'},
        'card-17100006': {'viterbi_published': 'viterbi_max'},
        'card-17100007': {'sequitur_published': 'sequitur'},
        'card-17100009': {'selective_diag_ssm': 'selective_diagonal', 'mamba_zoh_n1': 'mamba_zoh'},
        'card-1710000a': {'deltanet_delta_rule': 'delta_rule', 'additive_linear_attention': 'gated_additive_attention'},
    }[card]
    for declaration in replay['comparators']:
        if declaration['id'] in ids:
            if card == 'card-17100003' and declaration['id'] == 'minsky_two_counter' and not declaration.get('components'):
                declaration['components'] = [{'role': 'transition', 'name': 'INC/DEC with zero-test branches',
                                               'origin': 'published', 'cite': 'Minsky 1967, chapter 14'}]
            declaration['reference_spec'] = dict(evidence['reference_spec'], operation=ids[declaration['id']])
    for comparison in replay['comparisons']:
        if comparison['declared_id'] in ids:
            comparison['reference_samples'] = evidence['reference_samples']
    return replay


def historical_evidence(card):
    """Recovered declarations for the two independently validated historical protocols."""
    return load('historical_declarations').get(card)


def historical_phase7_evidence(entry):
    """Bind recovered evidence to the original card and critic, never to an expected kill code."""
    evidence = historical_evidence(entry['card_id'])
    if evidence is None:
        return None
    for field in ('card', 'critic'):
        encoded = json.dumps(entry[field], sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(encoded).hexdigest() != evidence['specimen'][f'{field}_sha256']:
            return None
    return copy.deepcopy(evidence)


def historical_samples(card):
    """Only input/output evidence is passed to the host; state snapshots remain provenance data."""
    return [{key: episode[key] for key in ('parameters', 'inputs', 'card_outputs')}
            for episode in load('historical')[card]['episodes']]


def concept_queries(episode):
    """Map raw CARD-side examples to fixed geometric scores, without consulting the model comparator."""
    import math
    import numpy as np

    embeddings = {key: np.asarray(value) for key, value in episode['parameters']['embeddings'].items()}

    def similarity(x, y):
        u, v = embeddings[x], embeddings[y]
        distance = math.acosh(1 + 2 * np.sum((u - v) ** 2) / ((1 - u @ u) * (1 - v @ v)))
        return (2 * math.atanh(np.linalg.norm(u)) + 2 * math.atanh(np.linalg.norm(v)) - distance) / 2

    examples = {}
    for i, ((word, op, point), output) in enumerate(zip(episode['inputs'], episode['card_outputs'])):
        if output[0] == 'none':
            continue
        if op == '+':
            examples.setdefault(word, []).append((op, point))
        elif op == '-' and word in examples:
            examples[word].append((op, point))
        elif op == '?' and word in examples:
            entries = examples[word]
            anchor = entries[0][1]
            yield i, {'positive_scores': [similarity(x, anchor) for sign, x in entries if sign == '+'],
                      'negative_scores': [similarity(x, anchor) for sign, x in entries if sign == '-'],
                      'query_score': similarity(point, anchor)}, output[1]


def next_historical_phase7_evidence(entry):
    """Second evidence batch, bound to the complete immutable card and critic."""
    evidence = load('next_declarations').get(entry['card_id'])
    if evidence is None:
        return None
    for field in ('card', 'critic'):
        encoded = json.dumps(entry[field], sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(encoded).hexdigest() != evidence['specimen'][f'{field}_sha256']:
            return None
    return copy.deepcopy(evidence)


@lru_cache(maxsize=1)
def next_historical_capture():
    """Retain all observed draws locally; an isolated package reexecutes immutable programs in memory."""
    import os
    import subprocess
    import sys

    root = FIXTURES.parents[4]
    manifest = load('next_sources')
    observer = root / 'results-wt/evidence-recovery/next_capture.py'
    for record in manifest['cards'].values():
        p = record['provenance']
        assert hashlib.sha256((root / p['source']).read_bytes()).hexdigest() == p['sha256']
        assert hashlib.sha256(observer.read_bytes()).hexdigest() == p['observer_sha256']
    raw = root / 'results-wt/evidence-recovery/next-full-capture.json'
    if raw.exists():
        data = raw.read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['capture_sha256']
        return json.loads(data)
    program = '''import json, runpy, sys
from pathlib import Path
root = Path(sys.argv[1])
ns = runpy.run_path(str(root / 'results-wt/evidence-recovery/next_capture.py'))
manifest = json.loads((root / 'hybrid/tests/research/fixtures/reference_replays/next_sources.json').read_text())
records = manifest['cards']
for card, record in records.items():
    path = root / record['provenance']['source']
    episodes, stdout = ns['capture'](card, path.read_text(), str(path))
    assert stdout == record['reexecuted_stdout']
    record['episodes'] = episodes
json.dump(records, sys.stdout, separators=(',', ':'))
'''
    result = subprocess.run([sys.executable, '-c', program, str(root)], check=True, capture_output=True,
                            text=True, env=dict(os.environ, PYTHONHASHSEED='0'))
    records = json.loads(result.stdout)
    for record in records.values():
        draws = [{'parameters': e['parameters'], 'inputs': e['inputs']} for e in record['episodes']]
        encoded = json.dumps(draws, sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(encoded).hexdigest() == record['provenance']['draws_sha256']
    return records
