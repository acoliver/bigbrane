"""Negative full-mechanism evidence must not turn partial similarities into accepted reductions."""
import copy
from fractions import Fraction
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from hybrid.tests.research.phase7_replay import load_fixture
from hybrid.tests.research.remaining_replays import (
    ROOT, CARDS, digest, load, full_capture, reproduce_capture, evidence_for,
    run_original, observe_original, revised_paid_counts,
    boolean_find_s, hypothesis_matches, nge_single_exemplar_prediction, residual_handcase,
)
from hybrid.tests.research.reference_replays import (
    historical_phase7_evidence, next_historical_phase7_evidence,
)


@pytest.fixture(scope='module')
def reproduced():
    return reproduce_capture()


def entry(card):
    return next(e for e in load_fixture()['cards'] if e['card_id'] == card)


@pytest.mark.parametrize('card', CARDS)
def test_all_original_seeded_draws_states_and_outputs_reproduce_without_saved_raw_capture(card, reproduced):
    manifest = load('remaining_sources')['cards'][card]
    observed = reproduced[card]
    assert digest(observed['episodes']) == manifest['provenance']['trace_sha256']
    assert digest(observed['soft_contracts']) == manifest['provenance']['soft_contracts_sha256']
    assert digest(observed['isotope_draws']) == manifest['provenance']['isotope_draws_sha256']
    assert observed == full_capture()[card]
    expected = load('remaining_evidence')['measurements']['capture_counts'][card]
    assert len(observed['episodes']) == expected['episodes']
    assert sum(len(e['inputs']) for e in observed['episodes']) == expected['tokens']
    assert len(observed['soft_contracts']) == expected['soft_contracts']
    assert len(observed['isotope_draws']) == expected['isotope_draws']
    assert observed['provenance']['capture_insertions'] == []
    assert 'not recovered historical bytes' in observed['provenance']['draw_limit']


@pytest.mark.parametrize('card', CARDS)
def test_compact_samples_are_actual_card_outputs_not_original_comparator_outputs(card):
    evidence = evidence_for(entry(card))
    for i, episode in zip(evidence['selected_episode_indices'], evidence['episodes']):
        assert episode == full_capture()[card]['episodes'][i]
        result = run_original(card, episode)
        if card == 'card-2c5dd003':
            for got, field in zip(result, ('card_outputs', 'filtered', 'lag_one_credits', 'next_output')):
                np.testing.assert_array_equal(got, episode[field])
        elif card == 'card-dc69816b':
            for got, field in zip(result, ('hidden', 'card_outputs', 'events')):
                np.testing.assert_array_equal(got, episode[field])
        else:
            np.testing.assert_array_equal(result[0], episode['card_outputs'])


@pytest.mark.parametrize('card', CARDS)
def test_negative_evidence_is_specimen_bound_and_never_an_admission_declaration(card):
    original = entry(card); evidence = evidence_for(original)
    assert evidence['accepted_reference'] is False and evidence['missing_full_published_operation'] is True
    assert 'declaration' not in evidence and 'reference_spec' not in evidence
    assert historical_phase7_evidence(original) is None
    assert next_historical_phase7_evidence(original) is None
    for field, key in (('card', 'summary'), ('critic', 'mapping')):
        changed = copy.deepcopy(original); changed[field][key] += ' change'
        assert evidence_for(changed) is None
    changed = copy.deepcopy(original)
    changed['original']['kill_codes'] = ['made_up']; changed['max_abs_diff'] = -1.
    assert evidence_for(changed) == evidence


@pytest.mark.parametrize('card', CARDS)
def test_original_card_critic_and_program_receipts_are_locally_verifiable_when_available(card):
    e = evidence_for(entry(card)); p = full_capture()[card]['provenance']
    for record in e['original_receipts'].values():
        path = Path(record['path'])
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
            assert record['sha256'] == record['ledger_event']['data']['sha256']
    for key, hash_key in (('original_source', 'sha256'), ('artifact', 'artifact_sha256')):
        path = Path(p[key])
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == p[hash_key]


def test_every_hmm_draw_preserves_lag_one_identity_but_not_all_paid_sufficient_statistics():
    changed_episodes = changed_prefixes = prefixes = 0; largest = lag_error = 0.
    for episode in full_capture()['card-2c5dd003']['episodes']:
        changed = False
        for stop in range(2, len(episode['inputs'])+1):
            counts, posterior = revised_paid_counts(episode, stop)
            lag_error = max(lag_error, float(np.max(np.abs(posterior[-2]-episode['lag_one_credits'][stop-2]))))
            gap = float(np.max(np.abs(counts-episode['states'][stop-1]['C'])))
            prefixes += 1; largest = max(largest, gap)
            if gap > 1e-12:
                changed = True; changed_prefixes += 1
        changed_episodes += changed
    saved = load('remaining_evidence')['measurements']['hmm']
    assert (prefixes, changed_prefixes, changed_episodes) == (saved['prefixes'], saved['different_paid_count_prefixes'], saved['episodes_with_differences'])
    assert lag_error == saved['lag_one_identity_max_difference'] and largest == saved['max_count_difference']
    assert changed_episodes == 201 and largest > 1.


def test_hmm_three_observation_counterexample_by_independent_rational_path_enumeration():
    case = load('remaining_evidence')['handcases']['hmm']; e = case['episode']
    assert observe_original('card-2c5dd003', {k:e[k] for k in ('parameters','inputs')}) == e
    T = [[Fraction(9,10),Fraction(1,10)], [Fraction(1,10),Fraction(9,10)]]
    # The first deposit is exactly 81/122, 41/122; derive the third bet from those counts.
    emission = [[[Fraction(9,10),Fraction(1,10)], [Fraction(1,10),Fraction(9,10)]]]*2
    emission.append([[(9+Fraction(81,122))/(10+Fraction(81,122)), 1/(10+Fraction(81,122))],
                     [(1+Fraction(41,122))/(10+Fraction(41,122)), 9/(10+Fraction(41,122))]])
    path_mass = {}
    for path in itertools.product(range(2), repeat=3):
        mass = Fraction(1,2)
        for i,z in enumerate(path):
            mass *= emission[i][z][e['inputs'][i]]
            if i:
                mass *= T[path[i-1]][z]
        path_mass[path] = mass
    total = sum(path_mass.values())
    exact = np.array([[float(sum(v for p,v in path_mass.items() if p[i] == z)/total)
                       for z in range(2)] for i in range(3)])
    np.testing.assert_allclose(exact, case['revised_posteriors'], rtol=1e-14, atol=0.)
    assert exact[0,0] == float(Fraction(19455,35494))
    assert e['states'][1]['C'][0][0] == pytest.approx(81/122)
    assert e['states'][2]['C'][0][0] == e['states'][1]['C'][0][0]
    assert case['count_difference'] > .11
    assert np.max(np.abs(np.array(case['revised_counts_only_next_output'])-e['next_output'])) > .005


def test_hmm_dropped_deposit_mutant_is_detected_without_changing_original_source():
    e = load('remaining_evidence')['handcases']['hmm']['episode']
    mutated = run_original('card-2c5dd003', e, ('C[:, xs[t - 2]] += M[2]', 'pass'))
    assert np.max(np.abs(np.array(mutated[0])-e['card_outputs'])) > .01


def test_tensor_all_calibrations_complete_the_hidden_generator_tables_and_soft_messages():
    record = full_capture()['card-6fc7efec']; episode = record['episodes'][0]
    compiles = {}
    for state in episode['states']:
        if state['valid']:
            compiles.setdefault(state['compiles'], state['Z'])
    assert sorted(compiles) == [1,2,3]
    for block, hidden in enumerate(record['isotope_draws']):
        table = np.array([[hidden['ginv'][(hidden['ah'][a]+hidden['bh'][b]) % 7] for b in range(7)] for a in range(7)])
        _,_,ginv,alpha,beta = compiles[block+1]
        compiled = [[ginv[(alpha[a]+beta[b]) % 7] for b in range(7)] for a in range(7)]
        np.testing.assert_array_equal(compiled, table)
        for message in record['soft_contracts'][block*20:(block+1)*20]:
            independent = np.zeros(7)
            for a,b in itertools.product(range(7),repeat=2):
                independent[table[a,b]] += np.float64(message['u'][a])*np.float64(message['v'][b])
            np.testing.assert_array_equal(independent, message['card_output'])
    assert episode['states'][-1]['compiles'] == 3


def test_tensor_two_calibrations_require_an_actual_learning_rule_not_supplied_card_state():
    case = load('remaining_evidence')['handcases']['tensor']
    got, obj = run_original('card-6fc7efec', {'parameters': {'p':2}, 'inputs': case['inputs']})
    np.testing.assert_array_equal(got, case['card_outputs'])
    assert obj.compiles == 2
    assert [int(np.argmax(got[i])) for i in case['query_indices']] == [0,1]
    # Pure bilinear evaluation of the same two leaves with unchanged tensor is identical each time.
    initial_tensor = np.array([[[1,0],[0,1]], [[0,1],[1,0]]])
    first = np.einsum('cab,a,b->c', initial_tensor, [1,0], [1,0])
    np.testing.assert_array_equal(first, [1,0])
    assert np.max(np.abs(first-got[25,:2])) == 1.


def test_tensor_argument_order_mutant_is_detected_on_the_original_full_stream():
    episode = full_capture()['card-6fc7efec']['episodes'][0]
    mutant, _ = run_original('card-6fc7efec', episode,
                             ('m[ginv[(alpha[a] + beta[b]) % p]] += u[a] * v[b]',
                              'm[ginv[(alpha[b] + beta[a]) % p]] += u[a] * v[b]'))
    assert np.max(np.abs(mutant-episode['card_outputs'])) == 1.


def test_nge_single_nearest_exemplar_does_not_abstain_on_a_noncontaining_key():
    e = residual_handcase(); e['inputs'] = [0,3,1]
    observed = observe_original('card-dc69816b', e)
    assert observed['events'] == ['c','-']
    assert observed['states'][-1]['kappa'] == [0,2]
    assert observed['states'][-1]['H'][0]['K'] == [0,1]
    card_hidden = np.array(observed['hidden'][-1]); base = np.array(observed['states'][-1]['z'])
    np.testing.assert_array_equal(card_hidden, base)
    nge_hidden = base + nge_single_exemplar_prediction([0,0,0,0,0,1])
    assert np.linalg.norm(card_hidden-nge_hidden) == 1.
    E = np.array(e['parameters']['E'])
    logits = -np.sum((E-nge_hidden)**2,axis=1); probability = np.exp(logits-logits.max()); probability /= probability.sum()
    assert probability[-1] == pytest.approx(.03795525840089027)
    assert observed['card_outputs'][-1][-1] == pytest.approx(.005310984579836764)
    assert probability[-1] > observed['card_outputs'][-1][-1]


def test_find_s_boolean_constraints_and_monotone_intersection_have_different_domains():
    positive = [[1,1,0,0],[1,0,1,0]]
    hypothesis = boolean_find_s(positive)
    assert hypothesis == [1,None,None,0]
    assert hypothesis_matches(hypothesis, [1,0,1,0])
    assert not hypothesis_matches(hypothesis, [1,0,0,1])
    assert {0,1} & {0,2} == {0}
    assert {0} <= {0,3}
    # Dropping negative literals is a specified hypothesis-space restriction, not a full learner match.


def test_original_consolidation_retains_the_first_residual_and_all_saved_write_events():
    e = load('remaining_evidence')['handcases']['nge']['episode']
    assert observe_original('card-dc69816b', residual_handcase()) == e
    assert e['events'] == ['c','-','b','-','c']
    assert e['states'][2]['H'][0]['K'] == [0]
    assert e['states'][2]['H'][0]['rho'][-1] == 1.
    assert e['states'][4]['H'][1]['rho'][-1] == 1.8
    captures = full_capture()['card-dc69816b']['episodes']
    assert sum(e['events'].count('b') for e in captures) == 525
    assert {e['parameters']['phi'] for e in captures} == {'gelu','ident'}
    assert len({digest(e['parameters']['P']) for e in captures}) == 401


def test_intersection_mutant_and_learned_projection_change_are_detected():
    e = residual_handcase(values=(1.,1.,1.)); original = run_original('card-dc69816b', e)
    mutant = run_original('card-dc69816b', e, ("c['K'] = c['K'] & kappa", 'pass'))
    assert np.max(np.abs(original[0]-mutant[0])) == 1.
    changed = copy.deepcopy(e); P = changed['parameters']['P']
    P[0][2] = -3; P[4][2] = 3; P[5][2] = 2
    perturbed = run_original('card-dc69816b', changed)
    assert np.max(np.abs(original[0]-perturbed[0])) == 1.


def test_local_publication_snapshots_and_their_extracted_text_have_url_hash_provenance():
    publications = load('remaining_publications')
    assert len(publications) == 5
    for source in publications:
        assert source['url'].startswith('https://') and source['retrieved_utc'] and source['section']
        assert 'error' not in source and len(source['sha256']) == 64
        for field, hash_field in (('file','sha256'), ('text_file','text_sha256')):
            path = ROOT / source[field]
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == source[hash_field]


def test_all_protected_files_including_phase7_expecteds_schemas_and_host_operations_are_unchanged():
    protected = load('remaining_protected')
    assert len(protected) > 60
    for name, expected_hash in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected_hash, name
