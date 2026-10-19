"""Package compact negative evidence and measure independent diagnostics, without admission changes."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from hybrid.tests.research.remaining_replays import (
    ROOT, FIXTURES, CARDS, digest, full_capture, observe_original, residual_handcase,
    revised_paid_counts, run_original,
)

REPORT = ROOT / 'results-wt/evidence-recovery'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    captures = full_capture(); entries = {e['card_id']: e for e in json.loads((ROOT / 'hybrid/tests/research/fixtures/phase7_replay/run-20261010T181300Z.json').read_text())['cards']}
    run = ROOT.parent / 'bigbrane/runs/run-20261010T181300Z'
    ledger = [json.loads(l) for l in (run / 'ledger.jsonl').read_text().splitlines()]
    selected = {'card-2c5dd003': [0, 1, 200], 'card-6fc7efec': [0], 'card-dc69816b': [0, 201, 202]}
    classes = {'card-2c5dd003': 'full-estimator source gap; all-history smoothing is not permanent lag-one deposit',
               'card-6fc7efec': 'inference-component match only; published calibration learner not established',
               'card-dc69816b': 'direct NGE read mapping refuted; specified composite remains unsupported'}
    cards = {}
    for card in CARDS:
        cap = captures[card]; receipts = {}
        for kind in ('card_version', 'reduction_critic'):
            events = [e for e in ledger if e['data'].get('card_id') == card and
                      (e['type'] == kind if kind == 'card_version' else
                       e['type'] == 'artifact' and e['data'].get('kind') == kind)]
            event = events[-1]; raw = (run / event['data']['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == event['data']['sha256']
            obj = json.loads(raw)
            if kind == 'card_version':
                assert {k:v for k,v in obj.items() if k != 'status'} == entries[card]['card']
            else:
                assert all(obj[k] == v for k,v in entries[card]['critic'].items())
            receipts[kind] = {'ledger_event': event, 'path': str(run / event['data']['path']),
                              'sha256': hashlib.sha256(raw).hexdigest()}
        cards[card] = {'classification': classes[card], 'specimen': cap['provenance']['specimen'],
                       'original_receipts': receipts, 'selected_episode_indices': selected[card],
                       'episodes': [cap['episodes'][i] for i in selected[card]],
                       'accepted_reference': False, 'missing_full_published_operation': True}
    hmm = observe_original('card-2c5dd003', {'parameters': {'T': [[.9,.1],[.1,.9]], 'pi0': [.5,.5],
                           'B0': [[9.,1.],[1.,9.]]}, 'inputs': [0,1,1]})
    revised, posterior = revised_paid_counts(hmm, 3)
    old = np.asarray(hmm['states'][-1]['C'])
    phirevised = (np.asarray(hmm['parameters']['B0']) + revised)
    phirevised /= phirevised.sum(axis=1, keepdims=True)
    revised_next = (np.asarray(hmm['parameters']['T']).T @ hmm['filtered'][-1]) @ phirevised
    handcases = {'hmm': {'episode': hmm, 'revised_paid_counts': revised.tolist(), 'revised_posteriors': posterior.tolist(),
                         'count_difference': float(np.max(np.abs(old-revised))),
                         'revised_counts_only_next_output': revised_next.tolist(),
                         'scope': 'E-step identity on observed adaptive bet path; not a full published lag-one estimator'},
                 'nge': {'episode': observe_original('card-dc69816b', residual_handcase()),
                         'single_exemplar_inputs': [0,3,1], 'nearest_residual': [0,0,0,0,0,1],
                         'scope': 'Salzberg section 3.2 single-exemplar classification versus hard-subset abstention'},
                 'tensor': {'p': 2, 'inputs': [2,0,1,3,1,0,4,0,1,0,0,5,6,
                                              2,1,0,3,0,1,4,1,0,0,0,5,6],
                            'query_indices': [12,25], 'answers': [0,1],
                            'scope': 'Two complete calibrations, addition then complemented addition; a supplied inference tensor does not specify its own updates'}}
    tensor = {'parameters': {'p': 2}, 'inputs': handcases['tensor']['inputs']}
    outputs, obj = run_original('card-6fc7efec', tensor)
    assert obj.compiles == 2
    handcases['tensor']['card_outputs'] = outputs.tolist()
    handcases['tensor']['compiles'] = obj.compiles
    measurements = {'capture_counts': {c: {'episodes': len(r['episodes']),
                    'tokens': sum(len(e['inputs']) for e in r['episodes']), 'soft_contracts': len(r['soft_contracts']),
                    'isotope_draws': len(r['isotope_draws'])} for c,r in captures.items()}}
    first = None; differing_episodes = 0; maxgap = 0.; prefixes = 0; differing_prefixes = 0
    lag_max = 0.
    for ei,e in enumerate(captures['card-2c5dd003']['episodes']):
        changed = False
        for stop in range(2, len(e['inputs'])+1):
            revised, posterior = revised_paid_counts(e, stop)
            lag_max = max(lag_max, float(np.max(np.abs(posterior[-2] - e['lag_one_credits'][stop-2]))))
            gap = float(np.max(np.abs(revised - e['states'][stop-1]['C'])))
            prefixes += 1; maxgap = max(maxgap, gap)
            if gap > 1e-12:
                changed = True; differing_prefixes += 1
                if first is None:
                    first = {'episode': ei, 'prefix_length': stop, 'max_count_difference': gap,
                             'card_counts': e['states'][stop-1]['C'], 'all_history_revised_counts': revised.tolist()}
        differing_episodes += changed
    measurements['hmm'] = {'prefixes': prefixes, 'different_paid_count_prefixes': differing_prefixes,
                            'episodes_with_differences': differing_episodes, 'max_count_difference': maxgap,
                            'lag_one_identity_max_difference': lag_max, 'first_difference': first,
                            'scope': 'Observed adaptive parameter path held identical only to isolate smoothing scope; no full-estimator match claimed'}
    measurements['nge'] = {'original_consolidations': sum(e['events'].count('b') for e in captures['card-dc69816b']['episodes']),
                            'original_writes': sum(sum(x != '-' for x in e['events']) for e in captures['card-dc69816b']['episodes'])}
    (FIXTURES / 'remaining_evidence.json').write_text(json.dumps({'cards': cards, 'handcases': handcases,
                                               'measurements': measurements}, separators=(',', ':')) + '\n')
    dump(REPORT / 'remaining-measurements.json', measurements)
    tracked = subprocess.check_output(['git','ls-files'], cwd=ROOT, text=True).splitlines()
    protected = [p for p in tracked if p.startswith(('bigbrane/','tests/','hybrid/math/','hybrid/research/schemas/')) or
                  p in ('README.md','EXPERIMENT.md','hybrid/tests/research/test_phase7_replay.py',
                        'hybrid/tests/research/phase7_replay.py', 'hybrid/tests/research/fixtures/phase7_replay/run-20261010T181300Z.json',
                        'hybrid/research/stages/published_references.py','hybrid/research/stages/contracts.py')]
    dump(FIXTURES / 'remaining_protected.json', {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected})
    print(json.dumps(measurements, indent=2))


if __name__ == '__main__':
    main()
