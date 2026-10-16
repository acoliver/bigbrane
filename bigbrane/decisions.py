"""Prespecified generator-level decisions; never called by fitting or selection."""
import numpy as np
from .core import rng


def bootstrap_interval(deltas, root=20260930):
    values = np.asarray(deltas, dtype=np.float64).reshape(2, 4)
    r = rng((root, 9, 0))
    indexes = r.integers(0, 4, size=(5000, 2, 4))
    means = np.stack([values[regime, indexes[:, regime]] for regime in range(2)], axis=1).mean((1, 2))
    return np.percentile(means, [2.5, 97.5]).tolist()


def advantage(deltas, root=20260930):
    values = np.asarray(deltas, dtype=np.float64).reshape(2, 4)
    interval = bootstrap_interval(values, root)
    mean = float(values.mean()); regimes = values.mean(1).tolist(); positive = int((values > 0).sum())
    passed = mean >= .02 and all(x > 0 for x in regimes) and positive >= 6 and interval[0] > 0
    return {'generator_deltas': values.tolist(), 'mean_delta': mean, 'regime_means': regimes,
            'positive_generators': positive, 'bootstrap_95_interval': interval, 'passed': passed,
            'directional_claim_contradicted': mean <= 0,
            'bootstrap_seed': (root, 9, 0), 'bootstrap_draws': 5000,
            'bootstrap_description': 'four generators resampled within each regime; all coordinates travel together; descriptive interval'}


def error_table(evaluations):
    result = {}
    for record in evaluations:
        s = record['selection']['spec']
        coordinate = (s['stage'], s['regime'], s['generator'], s['size_index'], s['family'], s['replica'])
        if coordinate in result:
            raise ValueError('duplicate selected-model outcome')
        result[coordinate] = record['metrics']['all']['kl']
    return result


def deltas_at(table, stage, control, sizes):
    return [[float(np.mean([table[(stage, regime, g, size, control, replica)]-
                            table[(stage, regime, g, size, 'hmm', replica)]
                            for size in sizes for replica in (0, 1)])) for g in range(4)] for regime in range(2)]


def target_crossing(errors):
    tokens = (512, 2048, 8192, 16384)
    for index in range(4):
        if all(value <= .05 for value in errors[index:]):
            return {'tokens': tokens[index], 'censoring': 'left' if index == 0 else 'grid',
                    'display': '<=512' if index == 0 else str(tokens[index])}
    return {'tokens': None, 'censoring': 'right', 'display': '>16384'}


def matched_report(evaluations, root=20260930):
    table = error_table(evaluations)
    controls = {control: advantage(deltas_at(table, 0, control, (0, 1)), root) for control in ('rnn', 'attention')}
    passed = all(record['passed'] for record in controls.values())
    curves = []
    for regime in range(2):
        for g in range(4):
            families = {}
            for family in ('hmm', 'rnn', 'attention'):
                errors = [float(np.mean([table[(0, regime, g, size, family, replica)] for replica in (0, 1)])) for size in range(4)]
                families[family] = {'errors': errors, 'target': target_crossing(errors)}
            curves.append({'regime': regime, 'generator': g, 'families': families})
    comparisons = {}
    for control in ('rnn', 'attention'):
        outcomes = {'hmm_reaches_at_smaller_sampled_size': 0, 'same_crossing': 0,
                    'control_reaches_at_smaller_sampled_size': 0, 'both_right_censored': 0}
        for row in curves:
            h = row['families']['hmm']['target']['tokens']; c = row['families'][control]['target']['tokens']
            if h is None and c is None:
                outcomes['both_right_censored'] += 1
            elif h == c:
                outcomes['same_crossing'] += 1
            elif c is None or (h is not None and h < c):
                outcomes['hmm_reaches_at_smaller_sampled_size'] += 1
            else:
                outcomes['control_reaches_at_smaller_sampled_size'] += 1
        comparisons[control] = outcomes
    fewer_both = 0
    for row in curves:
        h = row['families']['hmm']['target']['tokens']
        if h is not None and all(row['families'][c]['target']['tokens'] is None or h < row['families'][c]['target']['tokens'] for c in ('rnn', 'attention')):
            fewer_both += 1
    regime_curves = {str(regime): {family: np.mean([row['families'][family]['errors'] for row in curves if row['regime'] == regime], axis=0).tolist()
                                  for family in ('hmm', 'rnn', 'attention')} for regime in range(2)}
    headline = 'Matched low-data advantage passed against both controls.' if passed else 'The prespecified matched advantage was not established.'
    headline += f' HMM reached the target on a smaller sampled budget than both controls on {fewer_both}/8 generators.'
    return {'verdict': headline, 'fewer_token_target_generators_against_both': fewer_both,
            'matched_advantage_passed': passed, 'controls': controls, 'curves': curves,
            'regime_mean_curves': regime_curves,
            'tokens_to_target_comparisons': comparisons, 'training_tokens': [512, 2048, 8192, 16384],
            'accessible_tokens_including_validation': [2560, 4096, 10240, 18432],
            'target': .05, 'stress_stage': 'triggered' if passed else 'not triggered',
            'scope': 'Eight sampled HMM generators and fixed version-one optimization budgets; no general sample-efficiency claim.'}


def stress_report(matched_evaluations, stress_evaluations, root=20260930):
    table = error_table(matched_evaluations+stress_evaluations)
    results = []
    for size in (1, 3):
        controls = {}
        for control in ('rnn', 'attention'):
            outcome = advantage(deltas_at(table, 1, control, (size,)), root)
            outcome['matched_same_size'] = advantage(deltas_at(table, 0, control, (size,)), root)
            outcome['label'] = 'retained' if outcome['passed'] else ('reversed' if outcome['mean_delta'] < 0 and outcome['bootstrap_95_interval'][1] < 0 else 'reduced, absent, or inconclusive')
            controls[control] = outcome
        results.append({'training_tokens': (2048 if size == 1 else 16384), 'controls': controls,
                        'retained_against_both': all(v['passed'] for v in controls.values())})
    return results
