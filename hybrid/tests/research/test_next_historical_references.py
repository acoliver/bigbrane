"""Independent full-protocol checks and retained limitations for the next historical subset."""
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from hybrid.research.schemas import validate, load_schema
from hybrid.research.stages.published_references import (
    NEXT_HISTORICAL_REFERENCE_OPERATIONS, REFERENCE_OPERATIONS, reference_outputs,
)
from hybrid.research.stages.s2_prior_art import (
    declaration_problem, independent_comparison, judge_comparisons, report_problem,
)
from hybrid.tests.research.critic_answers import novelty_fields
from hybrid.tests.research.phase7_replay import load_fixture
from hybrid.tests.research.reference_replays import (
    load, next_historical_capture, next_historical_phase7_evidence,
)

ROOT = Path(__file__).resolve().parents[3]
SUPPORTED = {'card-7ec5e5c2': 'automatonMorphism7ec5e5c2', 'card-bc3935ac': 'deweyForestbc3935ac'}


def entry(card):
    return next(e for e in load_fixture()['cards'] if e['card_id'] == card)


def original_function(card, mutation=None):
    p = load('next_sources')['cards'][card]['provenance']
    source = (ROOT / p['source']).read_text()
    tree = ast.parse(source)
    names = {'card_run'} if card != 'card-bc3935ac' else {'run_card', 'mul', 'card_C', 'card_adj'}
    definitions = '\n\n'.join(ast.get_source_segment(source, node) for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name in names)
    if mutation:
        old, new = mutation
        assert definitions.count(old) == 1
        definitions = definitions.replace(old, new)
    namespace = {'np': np}
    if card == 'card-7ec5e5c2':
        from collections import deque
        namespace['deque'] = deque
    exec(compile(definitions, '<untouched-card-definitions>', 'exec'), namespace)
    return namespace


def run_original(card, sample, mutation=None):
    ns = original_function(card, mutation)
    p, stream = sample['parameters'], sample['inputs']
    if card == 'card-7ec5e5c2':
        return ns['card_run'](p['source'], p['target'], stream, len(p['source'][0]))
    ns.update(RES=['<', '?', 'yes', 'no'], V=len(p['vocabulary']), EPS=p['epsilon'],
              IDX={x: i for i, x in enumerate(p['vocabulary'])})
    return ns['run_card'](stream)[0]


@pytest.mark.parametrize('card', SUPPORTED)
def test_every_observed_original_draw_matches_full_independent_operation(card):
    record = next_historical_capture()[card]
    assert len(record['episodes']) == 200
    total = sum(len(e['inputs']) for e in record['episodes'])
    assert total == (30693 if card == 'card-7ec5e5c2' else 18234)
    for e in record['episodes']:
        assert len(e['states']) == len(e['inputs']) == len(e['card_outputs'])
        actual = reference_outputs(SUPPORTED[card], e['parameters'], e['inputs'])
        np.testing.assert_array_equal(actual, e['card_outputs'])
    if card == 'card-bc3935ac':
        assert 'not recovery of byte-exact historical draws' in record['provenance']['draw_limit']


@pytest.mark.parametrize('card', SUPPORTED)
def test_compact_samples_match_the_original_card_functions_and_capture(card):
    evidence = next_historical_phase7_evidence(entry(card))
    for i, sample in zip(evidence['selected_episode_indices'], evidence['reference_samples']):
        original = next_historical_capture()[card]['episodes'][i]
        assert sample == {k: original[k] for k in ('parameters', 'inputs', 'card_outputs')}
        np.testing.assert_array_equal(run_original(card, sample), sample['card_outputs'])


@pytest.mark.parametrize('card', SUPPORTED)
def test_source_bound_envelope_validates_without_changing_the_critic_or_policy(card):
    e = entry(card); before = copy.deepcopy(e)
    evidence = next_historical_phase7_evidence(e); d = evidence['declaration']
    answer = dict({k: v for k, v in e['critic'].items() if k != 'radical_enough'},
                  **novelty_fields(e['critic']), comparators=[d])
    validate('reduction_critic', answer)
    assert declaration_problem(answer) is None and answer['fixed_learned_parts'] == []
    assert d['parameter_mapping'] == e['critic']['mapping']
    assert not d['transformations'] and not d['limit_only']
    comparison = dict(evidence['comparison'], reference_samples=evidence['reference_samples'])
    assert report_problem({'comparisons': [comparison], 'notes': ''}, [d]) is None
    outcome, _, measured = judge_comparisons([comparison], [d], 'other_known')
    assert outcome == 'verified' and measured[0]['max_abs_diff'] == 0 and measured[0]['counted']
    assert e == before


@pytest.mark.parametrize('card', SUPPORTED)
def test_changed_card_or_critic_cannot_borrow_the_envelope(card):
    for field, key in (('card', 'summary'), ('critic', 'mapping')):
        e = entry(card); e[field][key] += ' perturbation'
        assert next_historical_phase7_evidence(e) is None
    e = entry(card); evidence = next_historical_phase7_evidence(e)
    e['original']['kill_codes'] = ['invented']
    e['max_abs_diff'] = 123.; e['reduction_outcome'] = 'refuted'
    assert next_historical_phase7_evidence(e) == evidence


@pytest.mark.parametrize('card', SUPPORTED)
def test_deliberately_nonequivalent_trace_is_refuted_despite_reported_zero(card):
    evidence = next_historical_phase7_evidence(entry(card))
    comparison = dict(evidence['comparison'], reference_samples=evidence['reference_samples'])
    comparison['reference_samples'][0]['card_outputs'][0][0] += 1
    comparison['max_abs_diff'] = 0
    outcome, _, measured = judge_comparisons([comparison], [evidence['declaration']], 'other_known')
    assert outcome == 'refuted' and measured[0]['max_abs_diff'] == 1
    assert measured[0]['reported_max_abs_diff'] == 0


def test_homomorphism_hand_answers_cover_parser_order_unassigned_and_conflict():
    p = {'source': [[1, 2, 0], [0, 2, 1]], 'target': [[1, 2, 0], [0, 2, 1]]}
    stream = ['Q', 0, '?', 0, 'D', 'Q', 0, 'Q', 0, '?', 0,
              'D', 0, 0, 'Q', 0, 'L', 'R', '?', 0,
              'Q', 0, 'R', 'L', '?', 0, 'D', 0, 1, 'Q', 2, '?', 0]
    result = reference_outputs(SUPPORTED['card-7ec5e5c2'], p, stream)
    np.testing.assert_array_equal(result[2], [1/3]*3+[0]*5)
    np.testing.assert_array_equal(result[6], [1/8]*8)  # Invalid demo token resets, not reprocessed.
    assert result[18, 2] == 1 and result[24, 1] == 1  # LR and RL do not commute.
    np.testing.assert_array_equal(result[31], [1/3]*3+[0]*5)
    sample = {'parameters': p, 'inputs': stream}
    np.testing.assert_array_equal(run_original('card-7ec5e5c2', sample), result)


def test_homomorphism_unanchored_states_are_abstention_not_global_existence():
    p = {'source': [[0, 1], [0, 1]], 'target': [[0, 1], [0, 1]]}
    stream = ['D', 0, 1, 'Q', 0, '?', 0, 'Q', 1, '?']
    out = reference_outputs(SUPPORTED['card-7ec5e5c2'], p, stream)
    assert out[5, 1] == 1
    np.testing.assert_array_equal(out[-1], [.5, .5, 0, 0, 0, 0, 0])
    # Independent enumeration of all maps: the demonstrated orbit has image 1, the other stays free.
    candidates = [f for f in itertools.product(range(2), repeat=2) if f[0] == 1]
    assert {f[1] for f in candidates} == {0, 1}


def test_dropping_one_homomorphism_generator_from_original_is_detected():
    p = {'source': [[0, 1], [1, 0]], 'target': [[0, 1], [1, 0]]}
    sample = {'parameters': p, 'inputs': ['D', 0, 0, 'Q', 1, '?']}
    mutant = run_original('card-7ec5e5c2', sample, ('W.append((T[1][x], U[1][y]))', 'pass'))
    expected = reference_outputs(SUPPORTED['card-7ec5e5c2'], p, sample['inputs'])
    assert np.max(np.abs(mutant - expected)) == .5


def test_forest_hand_answers_cover_unknowns_rejection_roots_reflexivity_and_smoothing():
    vocab = ['<', '?', 'yes', 'no', 'a', 'b', 'c', 'd', 'u']
    p = {'vocabulary': vocab, 'epsilon': .18}
    stream = 'u ? u a ? b b < a c < b c ? a a < d d ? a c ? b c ? d < ? yes c ? c'.split()
    sample = {'parameters': p, 'inputs': stream}
    out = reference_outputs(SUPPORTED['card-bc3935ac'], p, stream)
    np.testing.assert_array_equal(out, run_original('card-bc3935ac', sample))
    assert out[2, 2] == pytest.approx(.84)  # Reflexive even before insertion.
    assert out[5, 2] == pytest.approx(.43) and out[5, 3] == pytest.approx(.43)
    assert out[14, 2] == pytest.approx(.84)  # Transitive positive.
    assert out[20, 2] == pytest.approx(.43)  # Rejected a<d must not create d.
    assert out[23, 2] == pytest.approx(.84)
    np.testing.assert_array_equal(out[29], np.full(len(vocab), 1 / len(vocab)))
    assert np.allclose(out.sum(axis=1), 1)


def test_changing_original_forest_smoothing_is_detected():
    sample = load('next_declarations')['card-bc3935ac']['reference_samples'][0]
    # A smoothing perturbation remains executable and differs at every query; original file is unchanged.
    mutant = run_original('card-bc3935ac', sample, ('p = np.full(V, EPS/V)', 'p = np.full(V, 0.0)'))
    expected = reference_outputs(SUPPORTED['card-bc3935ac'], sample['parameters'], sample['inputs'])
    assert np.max(np.abs(mutant - expected)) > 0


def test_stern_brocot_matrix_paths_and_containment_match_independent_tree_walks():
    for episode in next_historical_capture()['card-bc3935ac']['episodes']:
        parent, ordinal, count = {'RHO': None}, {'RHO': None}, {'RHO': 0}
        stream = episode['inputs']; reserved = {'<', '?', 'yes', 'no'}
        for x, op, y in zip(stream, stream[1:], stream[2:]):
            if op == '<' and x not in reserved and y not in reserved and x not in parent and x != y:
                if y not in parent:
                    parent[y] = 'RHO'; ordinal[y] = count['RHO']; count['RHO'] += 1; count[y] = 0
                parent[x] = y; ordinal[x] = count[y]; count[y] += 1; count[x] = 0
        labels = episode['final_labels']
        assert labels.keys() == parent.keys()
        for node in parent:
            path, current = [], node
            while parent[current] is not None:
                path.append(ordinal[current]); current = parent[current]
            word = ''.join('U' * j + 'L' for j in reversed(path))
            m = np.eye(2, dtype=object)
            for letter in word:
                m = m @ (np.array([[1, 1], [0, 1]], dtype=object) if letter == 'U' else
                         np.array([[1, 0], [1, 1]], dtype=object))
            np.testing.assert_array_equal(m, labels[node])
        # Exact matrix left-divisibility is checked against a parent walk, including the virtual root.
        for x, y in itertools.product(parent, repeat=2):
            current = x
            while current is not None and current != y:
                current = parent[current]
            a,b,c,d = np.asarray(labels[y], dtype=object).ravel()
            relative = np.array([[d,-b],[-c,a]], dtype=object) @ labels[x]
            assert bool((relative >= 0).all()) == (current == y)


def cubic_energy(W, s, field, gamma):
    plus, minus = (1+s)/2, (1-s)/2
    return (-sum(s[c] * (s @ W[:, :, c] @ s) for c in range(len(s))) / 3 - field @ s
            + np.sum(plus*np.log(plus) + minus*np.log(minus))/gamma)


def cubic_diagnostic_trace(p, stream, *, warm=True, accumulate=True):
    """Check the specified card recurrence through energy derivatives, not a published full-operation claim."""
    E, W = np.asarray(p['E']), np.asarray(p['W'])
    h, state, previous = [np.zeros(E.shape[1]) for _ in range(3)]
    outputs = []
    for token in stream:
        h = (h if accumulate else 0) + previous * E[token]
        u = state if warm else np.zeros_like(state)
        for _ in range(p['K']):
            field = np.array([u @ W[:, :, c] @ u for c in range(len(u))])
            u = np.tanh(p['gamma'] * (field + p['beta'] * h))
        state = u
        logits = np.array([np.dot(state, E[token] * row) for row in E])
        probability = np.exp(logits-logits.max()); probability /= probability.sum()
        outputs.append(np.r_[h, state, probability]); previous = E[token]
    return np.asarray(outputs)


def test_all_cubic_draws_check_full_recurrence_but_do_not_establish_a_published_mapping():
    episodes = next_historical_capture()['card-d1d383d5']['episodes']
    assert len(episodes) == 6 and sum(len(e['inputs']) for e in episodes) == 72
    differences = []
    for e in episodes:
        p = e['parameters']; W = np.array(p['W'])
        assert all(np.allclose(W, W.transpose(perm), rtol=0, atol=1e-15)
                   for perm in itertools.permutations(range(3)))
        difference = np.max(np.abs(cubic_diagnostic_trace(p, e['inputs']) - e['card_outputs']))
        differences.append(float(difference))
    assert max(differences) < 1e-6  # Existing host threshold; this diagnostic is never admitted to it.
    assert next_historical_phase7_evidence(entry('card-d1d383d5')) is None
    assert set(NEXT_HISTORICAL_REFERENCE_OPERATIONS) == set(SUPPORTED.values())
    assert 'cubic_diagnostic_trace' not in REFERENCE_OPERATIONS


def test_cubic_energy_gradient_for_general_learned_tensor_and_repeated_indices():
    p = load('next_dam_diagnostic')['episodes'][0]['parameters']
    W = np.asarray(p['W']); s = np.linspace(-.4, .4, len(W)); field = np.linspace(.1, .3, len(W))
    gradient = -np.array([s @ W[:, :, c] @ s for c in range(len(s))]) - field + np.arctanh(s)/p['gamma']
    numeric = []
    for c in range(len(s)):
        change = np.zeros_like(s); change[c] = 1e-6
        numeric.append((cubic_energy(W,s+change,field,p['gamma'])-cubic_energy(W,s-change,field,p['gamma']))/2e-6)
    np.testing.assert_allclose(gradient, numeric, rtol=0, atol=1e-9)
    # Binary-spin diagonal cubic is linear: E[spin^3]=m, not m^3. Do not silently claim exact Ising mean field.
    w, m = .9, .2
    assert w/3 != w*m*m
    assert np.tanh(w/3) != np.tanh(w*m*m)


def test_cubic_warm_start_and_evidence_accumulation_are_not_optional():
    e = load('next_dam_diagnostic')['episodes'][0]
    actual = np.array(e['card_outputs'])
    for option in ({'warm':False}, {'accumulate':False}):
        assert np.max(np.abs(cubic_diagnostic_trace(e['parameters'], e['inputs'], **option)-actual)) > 1e-3


def test_softmax_memory_is_not_cubic_memory_even_on_a_scalar_handcase():
    # Two positive real patterns: the cubic field is quadratic in the cue, the softmax weighted read is not.
    cue = .2
    cubic = np.tanh((1.**3 + 2.**3) * cue**2)
    softmax = reference_outputs('softmax_memory', {'keys':[[1.],[2.]], 'values':[[1.],[2.]], 'beta':1.}, [[cue]])[0,0]
    assert abs(cubic-softmax) > 1


def test_synchronous_cubic_step_can_increase_energy_unlike_source_descent_dynamics():
    W = np.array([[[-.1603012650944002,.1295389728165167], [.1295389728165167,1.2396935386591073]],
                  [[.1295389728165167,1.2396935386591073], [1.2396935386591073,.8513389134881956]]])
    field = np.array([-.47249961359819065,.06368280426220527])
    s = np.array([-.3454735991555013,-.19515254085083583]); gamma = 2.
    u = np.tanh(gamma*(np.array([s @ W[:,:,c] @ s for c in range(2)])+field))
    assert cubic_energy(W,u,field,gamma)-cubic_energy(W,s,field,gamma) > .06


@pytest.mark.parametrize('operation,p,stream', [
    ('automatonMorphism7ec5e5c2', {'source':[[0],[1]],'target':[[0],[0]]}, ['Q']),
    ('automatonMorphism7ec5e5c2', {'source':[[0.],[0.]],'target':[[0],[0]]}, ['Q']),
    ('automatonMorphism7ec5e5c2', {'source':[[0],[0]],'target':[[0],[0]]}, [True]),
    ('deweyForestbc3935ac', {'vocabulary':['<','?','yes','no'],'epsilon':0}, ['yes']),
    ('deweyForestbc3935ac', {'vocabulary':['<','?','yes','no'],'epsilon':.1}, ['outside']),
])
def test_external_domain_errors_fail_without_repairs(operation, p, stream):
    with pytest.raises(ValueError):
        reference_outputs(operation,p,stream)


def test_ledger_program_observer_draw_and_publication_provenance():
    for card, record in next_historical_capture().items():
        p = record['provenance']; raw = (ROOT/p['source']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == p['sha256'] == p['executed_source_sha256']
        assert p['capture_insertions'] == [] and p['purpose'] == 'reduction_check'
        assert p['ledger_event']['data']['card_id'] == card
        assert p['ledger_event']['data']['sha256'] == p['artifact_sha256']
        artifact = Path(p['artifact'])
        if artifact.exists():
            data = artifact.read_bytes(); assert hashlib.sha256(data).hexdigest() == p['artifact_sha256']
            assert json.loads(data)['files']['main.py'].encode() == raw == Path(p['original_source']).read_bytes()
        for field in ('card','critic'):
            raw_specimen = json.dumps(entry(card)[field],sort_keys=True,separators=(',',':')).encode()
            assert hashlib.sha256(raw_specimen).hexdigest() == record['specimen'][field+'_sha256']
    for source in load('next_publications'):
        assert len(bytes.fromhex(source['sha256'])) == 32
        assert source['url'].startswith('https://') and source['retrieved_utc']
        snapshot = ROOT/'results-wt/evidence-recovery/next-sources'/source['file']
        if snapshot.exists():
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == source['sha256']


def test_second_batch_registry_schema_and_negative_reference_boundary():
    schema = load_schema('reduction_critic')
    enum = schema['properties']['comparators']['items']['properties']['reference_spec']['properties']['operation']['enum']
    assert set(enum) == set(REFERENCE_OPERATIONS)
    assert set(load('next_declarations')) == set(SUPPORTED)
    for card in ('card-d1d383d5','card-36725a7c','card-ffdbe331','card-6de11b01'):
        assert next_historical_phase7_evidence(entry(card)) is None


def test_allowing_reparenting_in_original_forest_is_detected():
    sample = {'parameters': {'vocabulary':['<','?','yes','no','a','b','d'], 'epsilon':.1},
              'inputs':'b < a a < d d ? b'.split()}
    mutant = run_original('card-bc3935ac', sample,
                          ('if not (X in M or X == Y):', 'if X != Y:'))
    expected = reference_outputs(SUPPORTED['card-bc3935ac'], sample['parameters'], sample['inputs'])
    assert np.max(np.abs(mutant - expected)) == pytest.approx(.45)


@pytest.mark.parametrize('card', SUPPORTED)
def test_extra_parameters_or_undeclared_comparisons_do_not_verify(card):
    evidence = next_historical_phase7_evidence(entry(card)); d = evidence['declaration']
    comparison = dict(evidence['comparison'], reference_samples=evidence['reference_samples'])
    comparison['reference_samples'][0]['parameters']['unsupported'] = 1
    assert 'reference_problem' in independent_comparison(comparison, d)
    comparison = dict(evidence['comparison'], declared_id='undeclared')
    assert 'undeclared' in report_problem({'comparisons':[comparison],'notes':''},[d])


@pytest.mark.parametrize('mutation', [
    ('u = s.copy()', 'u = np.zeros_like(s)'),
    ('h = h + prev * E[x]', 'h = prev * E[x]'),
])
def test_changing_original_cubic_warm_start_or_accumulator_is_detected(mutation):
    e = load('next_dam_diagnostic')['episodes'][0]; p = e['parameters']
    model = original_function('card-d1d383d5', mutation)['card_run']
    outputs = model(np.asarray(p['E']), e['inputs'], np.asarray(p['W']), p['beta'], p['gamma'], p['K'])
    flattened = [np.r_[h, s, probability] for h, s, probability in outputs]
    assert np.max(np.abs(np.asarray(flattened) - e['card_outputs'])) > 1e-3


def test_cubic_zero_coupling_handcase_has_cumulative_evidence_and_no_completion():
    p = {'E':[[0.],[1.],[-1.]],'W':[[[0.]]],'beta':.5,'gamma':2.,'K':3}
    out = cubic_diagnostic_trace(p,[1,1,2,0])
    np.testing.assert_array_equal(out[:,0],[0,1,0,0])
    np.testing.assert_allclose(out[:,1],[0,np.tanh(1),0,0],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(out[-1,2:],[1/3]*3)


def test_homomorphism_exhaustive_two_state_orbit_assignments():
    tables = [np.asarray(bits).reshape(2,2).tolist() for bits in itertools.product(range(2),repeat=4)]
    for source, target in itertools.product(tables, repeat=2):
        for i,j in itertools.product(range(2),repeat=2):
            orbit = {i}
            while True:
                grown = orbit | {source[a][x] for a in range(2) for x in orbit}
                if grown == orbit:
                    break
                orbit = grown
            maps = [f for f in itertools.product(range(2),repeat=2) if f[i] == j and
                    all(f[source[a][x]] == target[a][f[x]] for a in range(2) for x in orbit)]
            stream = ['D',i,j,'Q',0,'?',0,'Q',1,'?']
            actual = reference_outputs(SUPPORTED['card-7ec5e5c2'], {'source':source,'target':target}, stream)
            for x, position in ((0,5),(1,9)):
                expected = np.zeros(7)
                if maps and x in orbit:
                    assert len({f[x] for f in maps}) == 1
                    expected[maps[0][x]] = 1
                else:
                    expected[:2] = .5
                np.testing.assert_array_equal(actual[position], expected)


def test_homomorphism_partial_closure_does_not_claim_global_extendability():
    source = [[1,0,2],[1,0,2]]; target = [[1,0,0],[1,0,1]]
    # The anchored two-cycle is consistent. The isolated fixed source has no common fixed target.
    maps = [f for f in itertools.product(range(3),repeat=3) if
            all(f[source[a][x]] == target[a][f[x]] for a in range(2) for x in range(3))]
    assert maps == []
    result = reference_outputs(SUPPORTED['card-7ec5e5c2'], {'source':source,'target':target},
                               ['D',0,0,'Q',1,'?',0,'Q',2,'?'])
    assert result[5,1] == 1
    np.testing.assert_array_equal(result[-1],[1/3]*3+[0]*5)


def test_original_card_and_critic_raw_receipts_bind_the_replay_specimens():
    for card, records in load('next_specimens').items():
        for field, record in records.items():
            encoded = json.dumps(entry(card)[field],sort_keys=True,separators=(',',':')).encode()
            assert hashlib.sha256(encoded).hexdigest() == record['replay_specimen_sha256']
            assert record['sha256'] == record['ledger_event']['data']['sha256']
            assert record['ledger_event']['data']['card_id'] == card
            original = Path(record['path'])
            if original.exists():
                raw = original.read_bytes()
                assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
                data = json.loads(raw)
                selected = ({k:v for k,v in data.items() if k != 'status'} if field == 'card' else
                            {k:data[k] for k in entry(card)['critic']})
                assert selected == entry(card)[field]
