"""Independent historical checks, original CARD-side provenance, and non-equivalent controls."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from hybrid.research.stages.published_references import _projective_group, reference_outputs
from hybrid.research.stages.s2_prior_art import independent_comparison
from hybrid.tests.research.reference_replays import concept_queries, historical_samples, load, historical_evidence

ROOT = Path(__file__).resolve().parents[3]


def test_observer_capture_has_immutable_source_and_ledger_attribution():
    for card, record in load('historical').items():
        p = record['provenance']
        source = (ROOT / p['source']).read_bytes()
        assert hashlib.sha256(source).hexdigest() == p['sha256'] == p['executed_source_sha256']
        assert p['capture_insertions'] == [] and p['purpose'] == 'reduction_check'
        observer = ROOT / 'results-wt/evidence-recovery/capture_historical.py'
        assert hashlib.sha256(observer.read_bytes()).hexdigest() == p['observer_sha256']
        assert p['ledger_event']['data']['card_id'] == card
        artifact_path = Path(p['artifact'])
        if artifact_path.exists():
            artifact_bytes = artifact_path.read_bytes()
            assert hashlib.sha256(artifact_bytes).hexdigest() == p['artifact_sha256']
            assert json.loads(artifact_bytes)['files']['main.py'].encode() == source
            assert Path(p['original_source']).read_bytes() == source
        for episode in record['episodes']:
            assert len(episode['states']) == len(episode['inputs']) - (card == 'card-36725a7c')
        assert record['reexecuted_stdout']


@pytest.mark.parametrize('card,operation,episodes,inputs', [
    ('card-582f8abf', 'parity_union_find', 300, 13580),
    ('card-d86dca4b', 'pgl_version_space', 720, 10108),
])
def test_every_original_draw_matches_independent_host(card, operation, episodes, inputs):
    samples = historical_samples(card)
    assert len(samples) == episodes and sum(len(s['inputs']) for s in samples) == inputs
    for sample in samples:
        expected = reference_outputs(operation, sample['parameters'], sample['inputs'])
        np.testing.assert_allclose(sample['card_outputs'], expected, rtol=0, atol=1e-12)
    evidence = historical_evidence(card)
    declaration = {'base_family': 'other_known', 'reference_spec': evidence['reference_spec'], 'transformations': []}
    measured = independent_comparison({'reference_samples': evidence['reference_samples'], 'max_abs_diff': 123}, declaration)
    assert measured['max_abs_diff'] == 0 and measured['reported_max_abs_diff'] == 123
    for i, selected in zip(evidence['selected_episode_indices'], evidence['reference_samples']):
        assert selected == samples[i]


def test_parity_against_exhaustive_boolean_assignments_and_unknowns():
    n = 3
    tokens = [2 * n * n + 2, 3, 2 * n * n + 1, 10, 2 * n * n + 2,
              4, 2 * n * n + 2, 3 * n * n]
    assignments = list(itertools.product((0, 1), repeat=n))
    actual = reference_outputs('parity_union_find', {'vertices': n}, tokens)
    for i, token in enumerate(tokens):
        if token < 2 * n * n:
            pair, parity = divmod(token, 2)
            a, b = divmod(pair, n)
            assignments = [bits for bits in assignments if bits[a] ^ bits[b] == parity]
            q = .5
        elif token < 3 * n * n and assignments:
            a, b = divmod(token - 2 * n * n, n)
            q = sum(bits[a] ^ bits[b] for bits in assignments) / len(assignments)
        else:
            q = .5
        assert actual[i, -1] == q


def _original_card(card, mutation=None, prime=None):
    source = (ROOT / load('historical')[card]['provenance']['source']).read_text()
    tree = ast.parse(source)
    if prime is None:
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ('decode', 'Card')]
        namespace = {'np': np}
    else:
        body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run').body
        names = {'point', 'bracket', 'decode', 'frame', 'inv', 'mul', 'act', 'Card'}
        nodes = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        namespace = {'np': np, 'p': prime, 'V': prime + 1}
    definitions = '\n\n'.join(ast.get_source_segment(source, n) for n in nodes)
    # Nested source blocks retain their original indentation in all lines after the first.
    if prime is not None:
        import textwrap
        definitions = '\n\n'.join(textwrap.dedent('    ' + ast.get_source_segment(source, n)) for n in nodes)
    if mutation:
        old, new = mutation
        assert definitions.count(old) == 1
        definitions = definitions.replace(old, new)
    exec(compile(definitions, '<isolated-original-card>', 'exec'), namespace)
    return namespace['Card']


def test_dropping_constraint_parity_from_original_card_is_detected():
    card = 'card-582f8abf'
    mutant = _original_card(card, ('self.H[v] = ha ^ hb ^ s', 'self.H[v] = ha ^ hb'))
    differences = []
    for sample in historical_samples(card)[:12]:
        model = mutant(sample['parameters']['vertices'])
        bad = [model.step(tok) for tok in sample['inputs']]
        expected = reference_outputs('parity_union_find', sample['parameters'], sample['inputs'])
        differences.append(float(np.max(np.abs(bad - expected))))
    assert max(differences) == 1


def test_projective_group_counts_and_three_anchor_identification():
    group = _projective_group(3)
    assert group.shape == (24, 4)
    assert len({tuple(row) for row in group}) == 24
    # Three identity labels determine the identity, including the point at infinity.
    out = reference_outputs('pgl_version_space', {'prime': 3, 'mode': 'passive'}, [0, 0, 1, 1, 3, 3, 2])
    np.testing.assert_array_equal(out[-1], [0, 0, 1, 0])
    for prime in (5, 7, 11):
        assert len(_projective_group(prime)) == (prime + 1) * prime * (prime - 1)


def test_suppressing_original_version_space_update_is_detected():
    differences = []
    for sample in historical_samples('card-d86dca4b')[:12]:
        p = sample['parameters']['prime']
        model_class = _original_card('card-d86dca4b', ('self.r = r + 1', 'self.r = r'), p)
        model = model_class(sample['parameters']['mode'])
        outputs = []
        for token in sample['inputs']:
            outputs.append(model.dist()); model.consume(token)
        outputs.append(model.dist())
        differences.append(float(np.max(np.abs(np.asarray(outputs) - sample['card_outputs']))))
    assert max(differences) > 0.5


def test_nested_posterior_has_known_answers_and_rejects_empty_support():
    q = {'positive_scores': [2.], 'negative_scores': [1.], 'query_score': 1.5}
    assert reference_outputs('nested_size_posterior', {'delta': 1., 'prior_rate': 1.}, [q])[0] == pytest.approx(.5)
    assert reference_outputs('nested_size_posterior', {'delta': 2., 'prior_rate': 1.}, [q])[0] == pytest.approx(
        np.expm1(.5) / np.expm1(1))
    q['negative_scores'] = [2.]
    with pytest.raises(ValueError, match='empty'):
        reference_outputs('nested_size_posterior', {'delta': 1., 'prior_rate': 1.}, [q])
    with pytest.raises(ValueError, match='positive'):
        reference_outputs('nested_size_posterior', {'delta': 1., 'prior_rate': 0.}, [q])


def test_all_original_size_queries_retain_conditional_and_unsupported_results():
    valid = empty = domain = 0
    for episode in load('historical')['card-ffdbe331']['episodes']:
        params = {key: episode['parameters'][key] for key in ('delta', 'prior_rate')}
        for _, query, actual in concept_queries(episode):
            try:
                expected = reference_outputs('nested_size_posterior', params, [query])[0]
            except ValueError as exc:
                if 'empty' in str(exc):
                    empty += 1
                else:
                    domain += 1
            else:
                assert expected == pytest.approx(actual, abs=1e-12)
                valid += 1
    assert (valid, empty, domain) == (680, 84, 2)
    assert historical_evidence('card-ffdbe331') is None


def test_sugs_normalized_predictive_differs_from_unnormalized_greedy_score():
    params = {'ridge': 1., 'variance': 1., 'concentration': 1.1}
    observation = {'key': [1.], 'value': [0.], 'weight': 1.}
    # After observation one, old/new variances are 1.5 and 2. The determinant favors the old cluster.
    np.testing.assert_array_equal(reference_outputs('sugs_regression_assignments', params, [observation] * 2), [0, 0])
    assert np.log(1.1) > np.log(1)  # Dropping the density normalizer selects a new cluster instead.


def test_original_sugs_allocations_are_refuted_without_a_repair():
    compared = mismatches = 0
    for episode in load('historical')['card-36725a7c']['episodes']:
        inputs = [{key: row[key] for key in ('key', 'value', 'weight')} for row in episode['accepted']]
        params = {key: episode['parameters'][key] for key in ('ridge', 'variance', 'concentration')}
        if inputs:
            expected = reference_outputs('sugs_regression_assignments', params, inputs)
            mismatches += np.count_nonzero(expected != [row['card_assignment'] for row in episode['accepted']])
            compared += len(inputs)
    assert (compared, mismatches) == (2418, 1174)
    assert historical_evidence('card-36725a7c') is None


@pytest.mark.parametrize('operation,params,inputs', [
    ('parity_union_find', {'vertices': True}, [0]),
    ('parity_union_find', {'vertices': 2}, [-1]),
    ('pgl_version_space', {'prime': 9, 'mode': 'passive'}, [0]),
    ('pgl_version_space', {'prime': 5, 'mode': 'invented'}, [0]),
    ('pgl_version_space', {'prime': 5, 'mode': 'passive'}, [6]),
    ('sugs_regression_assignments', {'ridge': 0, 'variance': 1, 'concentration': 1}, [{'key': [1], 'value': [0], 'weight': 1}]),
])
def test_mapping_constraints_fail_fast(operation, params, inputs):
    with pytest.raises(ValueError):
        reference_outputs(operation, params, inputs)


def test_only_validated_historical_references_enter_the_live_contract():
    from hybrid.research.stages.published_references import HISTORICAL_REFERENCE_OPERATIONS, reference_policy_text
    schema = json.loads((ROOT / 'hybrid/research/schemas/reduction_critic.schema.json').read_text())
    admitted = schema['properties']['comparators']['items']['properties']['reference_spec']['properties']['operation']['enum']
    for operation in HISTORICAL_REFERENCE_OPERATIONS:
        assert operation in admitted
        assert operation in reference_policy_text()
    for operation in ('parity_union_find', 'pgl_version_space', 'nested_size_posterior', 'sugs_regression_assignments'):
        assert operation not in admitted
        assert operation not in reference_policy_text()


def test_retrieved_equation_source_snapshots_are_hashed():
    sources = ROOT / 'results-wt/evidence-recovery/sources'
    provenance = load('historical_publications')
    assert {source['file'] for source in provenance} == {
        'parity.html', 'mitchell-ch6.pdf', 'projective.pdf',
        'size-principle.pdf', 'sugs.html', 'cappe.pdf',
    }
    for source in provenance:
        assert len(bytes.fromhex(source['sha256'])) == 32
        assert source['retrieved_utc'] and source['url'].startswith('https://')
        assert source['final_url'].startswith('https://')
        snapshot = sources / source['file']
        if snapshot.exists():
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == source['sha256']
