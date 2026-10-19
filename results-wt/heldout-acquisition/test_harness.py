import copy
import json
import socket
import sys
from pathlib import Path

import pytest
import harness as h
from synthetic import metadata, record, envelope, make_pool, abort_at, study


def mutate(env, edit):
    data = copy.deepcopy(env['payload'])
    edit(data)
    return envelope(data)


def test_pins_pending_and_no_acceptance_claim():
    h.verify_freeze()
    h.check_pin(h.ROOT / 'exclusions.json', h.ROOT / 'exclusions.sha256')
    corpus = json.loads((h.ROOT / 'corpus.json').read_text())
    registry = json.loads((h.ROOT / 'exclusions.json').read_text())
    result = h.factorial([], corpus, registry)
    assert result['status'] == 'PENDING' and result['query_effect'] == 'unmeasured'
    assert not result['acceptance_improvement'] and not corpus['cards']


@pytest.mark.parametrize('arm', h.POLICY['arms'])
def test_equal_budgets_determinism_and_bounded_exposure(arm):
    env = make_pool(arm)
    before = h.digest(env)
    result = h.replay(env)
    assert result == h.replay(env) and h.digest(env) == before
    assert result['consumed']['search'] == 36 and result['consumed']['expansion'] == 10
    assert result['unspent'] == {'search': 0, 'expansion': 0, 'seed': 0}
    assert len(result['seeds']) == 5 and len(result['exposed']) == 60
    assert result['unique_counts']['acquired'] == 216 and result['availability']['ratio'] == 1
    assert result['stage_ids']['fetched'] is None and result['stage_ids']['reference_used'] is None


def test_query_syntax_keeps_all_tokens_and_other_sources_unchanged():
    raw = 'synthetic group-structured lattice'
    q0, q1 = [h.search_request(raw, 'arxiv', q)['params']['search_query'] for q in ('Q0', 'Q1')]
    assert q0 == 'all:synthetic AND all:group AND all:structured AND all:lattice'
    assert q1 == ' AND '.join(f'(ti:{t} OR abs:{t})' for t in ['synthetic', 'group', 'structured', 'lattice'])
    for src in ('openalex', 's2'):
        assert h.search_request(raw, src, 'Q0') == h.search_request(raw, src, 'Q1')


@pytest.mark.parametrize('bad', ['target_title', 'target_id', 'independent'])
def test_metadata_rejects_target_and_boolean_shortcuts(bad):
    meta = metadata()
    meta[bad] = True
    with pytest.raises(ValueError, match='unregistered fields'):
        h.queries(meta)


def test_duplicates_alias_bridges_and_support_not_overcounted():
    pool = h.Pool()
    first, second = record(1, 's2'), record(2, 'openalex')
    pool.add(first, ('lens0', 0, 's2'))
    for _ in range(10):
        alias = dict(first, source='semantic_scholar')
        pool.add(alias, ('lens0', 0, 's2'))
    pool.add(second, ('lens1', 1, 'openalex'))
    bridge = dict(first, aliases=[{'doi': second['doi']}])
    pool.add(bridge, ('lens1', 1, 's2'))
    assert len(pool.items) == 1 and len(pool.seeds(True)) == 1
    assert len({s[1] for s in pool.items[0]['strata']}) == 2
    assert h.public_id('semantic_scholar', first['source_id']) == h.public_id('s2', first['source_id'])


def test_baseline_first_ties_and_balanced_lens_query_source():
    pool = h.Pool()
    for qi in range(12):
        for j, src in enumerate(h.POLICY['sources']):
            pool.add(record(qi * 10 + j + 1, src), ('lens' + str(qi // 3), qi, src))
    assert pool.seeds(False) == [0, 1, 2, 3, 4]
    balanced = pool.seeds(True)
    assert balanced == pool.seeds(True)
    strata = [next(iter(pool.items[k]['strata'])) for k in balanced]
    assert len({s[0] for s in strata}) == 4 and len({s[1] for s in strata}) == 5
    assert len({s[2] for s in strata}) == 3 and len(set(balanced)) == 5
    assert len({s[0] for k in pool.exposed() for s in pool.items[k]['strata']}) == 4


def test_cached_call_consumes_slot_and_cannot_add_another():
    env = mutate(make_pool(), lambda d: d['calls'][0].update(outcome='cache', attempts=0, exchanges=[], request_count=0, response_count=0))
    result = h.replay(env)
    assert result['consumed']['search'] == 36 and result['consumed']['requests'] == 45
    assert result['outcomes']['cache'] == 1
    with pytest.raises(ValueError, match='extra calls'):
        h.replay(mutate(env, lambda d: d['calls'].append(d['calls'][0])))


@pytest.mark.parametrize('outcome', ['cooldown', 'error', 'unavailable'])
def test_terminal_abort_accounting_and_no_padding(outcome):
    result = h.replay(abort_at(make_pool(), 1, outcome))
    assert not result['complete'] and result['consumed']['search'] == 2
    assert result['unspent'] == {'search': 34, 'expansion': 10, 'seed': 5}
    assert result['availability'] == {'answered': 1, 'consumed': 2, 'ratio': 0.5}
    assert result['consumed']['attempts'] == (5 if outcome == 'error' else 1)
    empty = h.replay(make_pool(populated=False))
    assert empty['complete'] and empty['unique_counts']['acquired'] == 0
    assert empty['unspent']['seed'] == 5 and empty['unspent']['expansion'] == 10


def test_retries_redirects_and_logical_operations_are_distinct():
    def change(d):
        d['calls'][0].update(attempts=2, request_count=3, response_count=3,
            exchanges=[{'attempt': 1, 'status': 429}, {'attempt': 2, 'status': 302}, {'attempt': 2, 'status': 200}])
    result = h.replay(mutate(make_pool(), change))
    assert result['consumed']['search'] == 36 and result['consumed']['requests'] == 48
    assert result['consumed']['retries'] == 1 and result['consumed']['redirects'] == 1


@pytest.mark.parametrize('field,value', [('card_id', 'foreign'), ('arm', 'Q1S1'), ('call_id', 'wrong'),
    ('source', 'openalex'), ('query_sha', 'a' * 64), ('query_index', 1), ('lens', 'wrong'),
    ('request_count', 2), ('response_count', 0), ('attempts', 5), ('returned_count', 0), ('limit', 7)])
def test_strict_call_identity_and_counts(field, value):
    with pytest.raises(ValueError):
        h.replay(mutate(make_pool(), lambda d: d['calls'][0].update({field: value})))


def test_hidden_window_missing_call_and_blob_tamper_fail():
    with pytest.raises(ValueError, match='missing required call'):
        h.replay(mutate(make_pool(), lambda d: d['calls'].pop()))
    def window(d):
        d['calls'][0]['request'] = copy.deepcopy(d['calls'][0]['request'])
        d['calls'][0]['request']['params']['start'] = '6'
    with pytest.raises(ValueError, match='request'):
        h.replay(mutate(make_pool(), window))
    env = make_pool()
    env['payload']['calls'][0]['error'] = 'tampered'
    with pytest.raises(ValueError, match='pool SHA'):
        h.replay(env)
    def blob(d):
        d['blobs'][d['calls'][0]['body_sha']]['records'][0]['title'] = 'tampered'
    with pytest.raises(ValueError, match='tampered blob'):
        h.replay(mutate(make_pool(), blob))


@pytest.mark.parametrize('identity', [{'title': '  SYNTHETIC, excluded specimen!'},
    {'arxiv_id': 'https://arxiv.org/abs/9912.99999v7'}, {'doi': 'https://doi.org/10.99999/EXCLUSION'},
    {'source': 'semantic_scholar', 'source_id': 'e' * 40}])
def test_manifest_rejects_normalized_title_arxiv_doi_and_source_alias(identity):
    corpus, registry, _ = study()
    registry['entries'].append({'scope': 'original_pa2_11', 'origin_sha256': '1' * 64,
        'identity': {'title': 'Synthetic excluded specimen', 'arxiv_id': '9912.99999',
                     'doi': '10.99999/exclusion', 'source': 's2', 'source_id': 'e' * 40}})
    corpus['cards'][0]['reference_identities'] = [identity]
    with pytest.raises(ValueError, match='intersection'):
        h.corpus_gate(corpus, registry)


def test_exclusion_tamper_incomplete_inventory_and_free_bool_rejected(tmp_path):
    corpus, registry, _ = study()
    assert h.corpus_gate(corpus, registry)['status'] == 'READY_FOR_EXTERNAL_ACQUISITION'
    registry['entries'][0]['identity']['doi'] += '-tamper'
    with pytest.raises(ValueError, match='inventory SHA'):
        h.corpus_gate(corpus, registry)
    corpus, registry, _ = study()
    registry['entries'] = []
    for inventory in registry['scopes'].values():
        inventory['inventory_sha256'] = h.digest([])
    with pytest.raises(ValueError, match='too small'):
        h.corpus_gate(corpus, registry)
    path, pin = tmp_path / 'registry.json', tmp_path / 'registry.sha256'
    path.write_text('{}')
    pin.write_text(h.file_sha(path))
    path.write_text('{"independent": true}')
    with pytest.raises(ValueError, match='SHA mismatch'):
        h.check_pin(path, pin)
    with pytest.raises(ValueError, match='unexpected identity'):
        h.identities({'title': 'Synthetic', 'independent': True})


def test_unknown_usage_metrics_and_verified_artifact():
    env = make_pool()
    result = h.replay(env)
    metrics = h.coverage(result, [{'doi': record(1)['doi']}, {'doi': record(1)['doi']}])
    assert metrics['acquired'] == {'hits': 1, 'denominator': 1, 'coverage': 1.0, 'status': 'observed'}
    assert metrics['reference_used']['coverage'] is None
    identity = {'doi': record(1)['doi']}
    def evidence(d):
        blob = {'stage': 'reference_used', 'identity': identity, 'text': 'Synthetic citation evidence fixture.'}
        sha = h.digest(blob)
        d['blobs'][sha] = blob
        d['evidence']['reference_used'] = [{'identity': identity, 'artifact_sha': sha}]
    metrics = h.coverage(h.replay(mutate(env, evidence)), [identity])
    assert metrics['reference_used']['coverage'] == 1
    with pytest.raises(ValueError, match='zero reference'):
        h.coverage(result, [])
    def missing_content(d):
        blob = {'stage': 'fetched', 'identity': identity, 'text': ''}
        sha = h.digest(blob)
        d['blobs'][sha] = blob
        d['evidence']['fetched'] = [{'identity': identity, 'artifact_sha': sha}]
    with pytest.raises(ValueError, match='content'):
        h.replay(mutate(env, missing_content))


def test_missing_arm_and_query_pool_reuse_and_seed_confound_fail():
    corpus, registry, _ = study()
    with pytest.raises(ValueError, match='missing arm'):
        h.factorial([], corpus, registry)
    envs = [make_pool(a, c['card_id'], False, c['metadata']) for c in corpus['cards'] for a in h.POLICY['arms']]
    assert h.factorial(envs, corpus, registry)['status'] == 'PENDING labels'
    reused = copy.deepcopy(envs)
    for i in (2, 3):
        reused[i] = mutate(reused[i], lambda d: d.update(acquisition_id=d['card_id'] + ':Q0'))
    with pytest.raises(ValueError, match='reuse acquisition'):
        h.factorial(reused, corpus, registry)
    confounded = copy.deepcopy(envs)
    confounded[1] = mutate(confounded[1], lambda d: d.update(acquisition_id='different search acquisition'))
    with pytest.raises(ValueError, match='share fixed'):
        h.factorial(confounded, corpus, registry)


def test_factorial_contrasts_falsification_and_pilot_uncertainty():
    contrasts = h.contrast(dict(zip(h.POLICY['arms'], [0.1, 0.3, 0.2, 0.7])))
    assert contrasts['query_at_S0'] == pytest.approx(0.1)
    assert contrasts['seed_at_Q0'] == pytest.approx(0.2)
    assert contrasts['interaction'] == pytest.approx(0.3)
    assert h.uncertainty([0.2] * 12)['bootstrap_95'] == pytest.approx([0.2, 0.2])
    corpus, registry, labels = study()
    envs = [make_pool(a, c['card_id'], False, c['metadata']) for c in corpus['cards'] for a in h.POLICY['arms']]
    result = h.factorial(envs, corpus, registry, labels)
    assert result['complete_cards'] == 12 and result['incomplete_cards'] == 0
    assert all('falsified' in value for value in result['decision'].values())
    envs[0], envs[1] = abort_at(envs[0], 0), abort_at(envs[1], 0)
    result = h.factorial(envs, corpus, registry, labels)
    assert result['complete_cards'] == 11 and result['incomplete_cards'] == 1
    assert len(result['paired_cards']) == 11 and result['intention_to_acquire']['query_average'] == 0


@pytest.mark.parametrize('event,args', [('socket.connect', (None, ('127.0.0.1', 9))),
    ('socket.sendto', (None, ('127.0.0.1', 9))), ('socket.getaddrinfo', ('localhost', 80, 0, 0, 0))])
def test_network_audit_rejects_connections_dns_datagrams(event, args):
    with pytest.raises(RuntimeError, match='Network prohibited'):
        sys.audit(event, *args)


def test_real_socket_and_dns_attempts_blocked():
    for action in [lambda: socket.socket(), lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
                   lambda: socket.getaddrinfo('localhost', 80), lambda: socket.gethostbyname('localhost')]:
        with pytest.raises(RuntimeError, match='Network prohibited'):
            action()


def test_expansion_identity_and_zero_network_cached_counts():
    with pytest.raises(ValueError, match='seed'):
        h.replay(mutate(make_pool(), lambda d: d['calls'][36].update(seed=['doi:10.99999/foreign-seed'])))
    env = mutate(make_pool(), lambda d: d['calls'][0].update(source='arxiv'))
    env = mutate(env, lambda d: d['calls'][2].update(source='semantic_scholar'))
    assert h.replay(env)['complete']
    def false_cache(d):
        d['calls'][0]['outcome'] = 'cache'
    with pytest.raises(ValueError, match='zero-network'):
        h.replay(mutate(make_pool(), false_cache))


def test_synthetic_ids_disjoint_from_real_known_exclusions():
    registry = json.loads((h.ROOT / 'exclusions.json').read_text())
    excluded = set().union(*(h.identities(e['identity']) for e in registry['entries']))
    env = make_pool()
    for call in env['payload']['calls']:
        for item in call['records']:
            assert not h.identities(item).intersection(excluded)
    assert all('synthetic' in item['title'].lower() for c in env['payload']['calls'] for item in c['records'])


def test_duplicate_mechanisms_and_empty_usage_evidence_not_truth():
    corpus, registry, _ = study()
    corpus['cards'][1]['mechanism_identity'] = dict(corpus['cards'][0]['mechanism_identity'])
    with pytest.raises(ValueError, match='duplicate mechanism'):
        h.corpus_gate(corpus, registry)
    env = mutate(make_pool(), lambda d: d['evidence'].update(reference_used=[], fetched=[]))
    result = h.replay(env)
    assert result['unique_counts']['reference_used'] is None and result['unique_counts']['fetched'] is None
