"""Contract synchronization and source-bound replay evidence for two historical specimens."""
import ast
import copy
import hashlib
import textwrap
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from hybrid.research.schemas import load_schema, validate
from hybrid.research.stages.published_references import (
    HISTORICAL_REFERENCE_OPERATIONS, REFERENCE_OPERATIONS, reference_outputs, reference_policy_text,
)
from hybrid.research.stages.s2_prior_art import (
    declaration_problem, independent_comparison, judge_comparisons, report_problem,
)
from hybrid.tests.research.critic_answers import novelty_fields
from hybrid.tests.research.phase7_replay import load_fixture
from hybrid.tests.research.reference_replays import FIXTURES, historical_capture, historical_phase7_evidence, load

SUPPORTED = {
    'card-582f8abf': 'parityunionfind582f8abf',
    'card-d86dca4b': 'finitePGLversionspaced86dca4b',
}
UNSUPPORTED = ('unknown', 'parity_union_find', 'pgl_version_space',
               'nested_size_posterior', 'sugs_regression_assignments')


def _entry(card):
    return next(e for e in load_fixture()['cards'] if e['card_id'] == card)


def _comparison(evidence):
    return dict(evidence['comparison'], reference_samples=evidence['reference_samples'])


def test_schema_registry_and_advertisement_are_synchronized():
    schema = load_schema('reduction_critic')
    enum = schema['properties']['comparators']['items']['properties']['reference_spec']['properties']['operation']['enum']
    assert len(enum) == len(set(enum))
    assert set(enum) == set(REFERENCE_OPERATIONS)
    assert set(HISTORICAL_REFERENCE_OPERATIONS) == set(SUPPORTED.values())
    assert set(load('historical_declarations')) == set(SUPPORTED)
    policy = reference_policy_text()
    assert all(f'- {operation}: family ' in policy for operation in enum)


@pytest.mark.parametrize('card,operation', SUPPORTED.items())
def test_recorded_declaration_and_envelope_pass_the_existing_contract(card, operation):
    entry = _entry(card)
    before = copy.deepcopy(entry)
    evidence = historical_phase7_evidence(entry)
    declaration = evidence['declaration']
    answer = dict({k: v for k, v in entry['critic'].items() if k != 'radical_enough'},
                  **novelty_fields(entry['critic']), comparators=[declaration])
    validate('reduction_critic', answer)
    assert answer['fixed_learned_parts'] == []
    assert declaration_problem(answer) is None
    assert declaration['reference_spec'] == evidence['reference_spec']
    assert declaration['reference_spec']['operation'] == operation
    assert declaration['parameter_mapping'] == entry['critic']['mapping']
    assert declaration['transformations'] == [] and declaration['limit_only'] is False
    comparison = _comparison(evidence)
    assert report_problem({'comparisons': [comparison], 'notes': 'recovered'}, [declaration]) is None
    outcome, _, measured = judge_comparisons([comparison], [declaration], entry['critic']['reduces_to'])
    assert outcome == 'verified'
    assert measured[0]['counted'] is True and measured[0]['max_abs_diff'] == 0
    assert measured[0]['reference_operation'] == operation
    assert entry == before
    parameters = {tuple(sorted(s['parameters'].items())) for s in evidence['reference_samples']}
    if card == 'card-582f8abf':
        assert {dict(p)['vertices'] for p in parameters} == {2, 4, 6, 8}
    else:
        assert {(dict(p)['prime'], dict(p)['mode']) for p in parameters} == {
            (p, m) for p in (5, 7, 11) for m in ('active', 'passive')}


@pytest.mark.parametrize('operation', UNSUPPORTED)
def test_unsupported_names_are_rejected_by_schema_and_not_registered(operation):
    schema = load_schema('reduction_critic')['properties']['comparators']['items']['properties']['reference_spec']
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'operation': operation, 'parameters': {}}, schema)
    assert operation not in REFERENCE_OPERATIONS
    evidence = historical_phase7_evidence(_entry('card-582f8abf'))
    evidence['declaration']['reference_spec']['operation'] = operation
    measured = independent_comparison(_comparison(evidence), evidence['declaration'])
    assert 'reference_problem' in measured and 'max_abs_diff' not in measured


@pytest.mark.parametrize('card', SUPPORTED)
def test_changed_specimens_cannot_borrow_the_original_evidence(card):
    entry = _entry(card)
    entry['card']['summary'] += ' changed specimen'
    assert historical_phase7_evidence(entry) is None
    entry = _entry(card)
    entry['critic']['mapping'] += ' changed mapping'
    assert historical_phase7_evidence(entry) is None
    entry = _entry(card)
    entry['card_id'] = 'card-00000000'
    assert historical_phase7_evidence(entry) is None


def test_evidence_is_not_selected_by_an_expected_code_or_old_verdict():
    for entry in load_fixture()['cards']:
        assert (historical_phase7_evidence(entry) is not None) == (entry['card_id'] in SUPPORTED)
    entry = _entry('card-582f8abf')
    evidence = historical_phase7_evidence(entry)
    entry['original']['kill_codes'] = ['invented:desiredcode']
    entry['reduction_outcome'] = 'refuted'
    entry['max_abs_diff'] = 99
    assert historical_phase7_evidence(entry) == evidence


@pytest.mark.parametrize('card', SUPPORTED)
def test_perturbed_trace_is_refuted_even_when_reported_equality_is_zero(card):
    evidence = historical_phase7_evidence(_entry(card))
    comparison = _comparison(evidence)
    sample = comparison['reference_samples'][0]
    sample['card_outputs'][0][0] += 1
    comparison['max_abs_diff'] = 0
    outcome, _, measured = judge_comparisons([comparison], [evidence['declaration']], 'other_known')
    assert outcome == 'refuted' and measured[0]['max_abs_diff'] == 1
    assert measured[0]['reported_max_abs_diff'] == 0


@pytest.mark.parametrize('card', SUPPORTED)
def test_parameter_and_metadata_perturbations_cannot_verify(card):
    evidence = historical_phase7_evidence(_entry(card))
    declaration = evidence['declaration']
    comparison = _comparison(evidence)
    comparison['reference_samples'][0]['parameters']['unsupported'] = 1
    assert 'reference_problem' in independent_comparison(comparison, declaration)
    comparison = _comparison(historical_phase7_evidence(_entry(card)))
    comparison['declared_id'] = 'undeclared'
    assert 'undeclared' in report_problem({'comparisons': [comparison], 'notes': ''}, [declaration])
    comparison['declared_id'] = declaration['id']
    comparison['invented_equality'] = True
    assert report_problem({'comparisons': [comparison], 'notes': ''}, [declaration]) is not None
    declaration['reference_spec']['invented_parameter_policy'] = True
    schema = load_schema('reduction_critic')['properties']['comparators']['items']
    with pytest.raises(jsonschema.ValidationError, match='Additional properties'):
        jsonschema.validate(declaration, schema)


def test_compact_samples_are_checked_against_the_unchanged_source_functions():
    for card in SUPPORTED:
        evidence = historical_phase7_evidence(_entry(card))
        p = evidence['provenance']
        root = Path(__file__).resolve().parents[3]
        assert hashlib.sha256((root / p['source']).read_bytes()).hexdigest() == p['sha256']
        source = (root / p['source']).read_text()
        tree = ast.parse(source)
        for sample in evidence['reference_samples']:
            params = sample['parameters']
            if card == 'card-582f8abf':
                nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                         and n.name in ('decode', 'Card')]
                namespace = {'np': np}
                definitions = '\n\n'.join(ast.get_source_segment(source, n) for n in nodes)
                exec(compile(definitions, '<source-card>', 'exec'), namespace)
                model = namespace['Card'](params['vertices'])
                actual = [model.step(x) for x in sample['inputs']]
            else:
                body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run').body
                names = {'point', 'bracket', 'decode', 'frame', 'inv', 'mul', 'act', 'Card'}
                nodes = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
                namespace = {'np': np, 'p': params['prime'], 'V': params['prime'] + 1}
                definitions = '\n\n'.join(textwrap.dedent('    ' + ast.get_source_segment(source, n)) for n in nodes)
                exec(compile(definitions, '<source-card>', 'exec'), namespace)
                model = namespace['Card'](params['mode'])
                actual = []
                for token in sample['inputs']:
                    actual.append(model.dist())
                    model.consume(token)
                actual.append(model.dist())
            np.testing.assert_array_equal(actual, sample['card_outputs'])
            np.testing.assert_allclose(reference_outputs(SUPPORTED[card], params, sample['inputs']), actual,
                                       rtol=0, atol=1e-12)


def test_compact_negative_controls_preserve_unsupported_results():
    controls = load('historical_negative_controls')
    for case in controls['concept']['cases']:
        expected_error = 'empty' if case['category'] == 'empty' else 'nonnegative'
        with pytest.raises(ValueError, match=expected_error):
            reference_outputs('nested_size_posterior', case['parameters'], [case['query']])
    row = controls['sugs']['counterexample']
    outputs = reference_outputs('sugs_regression_assignments', row['parameters'], row['prefix'])
    assert outputs[-1] == row['reference_assignment'] != row['card_assignment']
    for card in ('card-36725a7c', 'card-ffdbe331'):
        assert historical_phase7_evidence(_entry(card)) is None
    root = Path(__file__).resolve().parents[3]
    for name in ('concept', 'sugs'):
        p = controls[name]['provenance']
        assert hashlib.sha256((root / p['source']).read_bytes()).hexdigest() == p['sha256']


def test_full_capture_can_be_reproduced_without_packaging_its_matrices(monkeypatch):
    path = FIXTURES / 'historical.json'
    retained = path.exists()
    original = load('historical')
    exists = Path.exists
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'exists', lambda self: False if self == path else exists(self))
        reproduced = historical_capture()
    assert reproduced == original
    assert path.exists() == retained
