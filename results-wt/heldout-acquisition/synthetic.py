"""Invented public-format IDs and receipts, never real corpus data or relevance labels."""
from copy import deepcopy
import hashlib
from harness import POLICY, Pool, digest, identities, queries, search_request, expansion_request


def metadata():
    return {'concept': 'synthetic lattice', 'lenses': [
        {'name': 'lens' + str(i), 'phrases': [f'axis {i} facet {j}' for j in range(3)]} for i in range(4)]}


def record(n, src='s2'):
    sid = {'arxiv': f'9912.{n:05d}', 'openalex': 'W' + str(900000000 + n), 's2': f'{n:040x}'}[src]
    return {'source': src, 'source_id': sid, 'title': f'Synthetic lattice specimen {n}',
            'doi': f'10.99999/synthetic-heldout-{n}'}


def envelope(payload):
    return {'payload': payload, 'sha256': digest(payload)}


def make_pool(arm='Q0S0', card='synthetic-card', populated=True, meta=None):
    meta = deepcopy(meta or metadata())
    data = {'card_id': card, 'arm': arm, 'metadata': meta, 'acquisition_id': card + ':' + arm[:2],
            'calls': [], 'blobs': {}, 'evidence': {}}
    pool = Pool()

    def append(kind, request, records, qi=None, lens=None, raw=None, src='s2', seed=None):
        body = {'request': request, 'records': records}
        sha = digest(body)
        data['blobs'][sha] = body
        data['calls'].append({'card_id': card, 'arm': arm, 'call_id': f'{card}:{arm}:{len(data["calls"]):03d}',
            'operation': kind, 'query_index': qi, 'lens': lens,
            'query_sha': hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None,
            'source': src, 'request': request, 'seed': seed, 'limit': 6, 'records': records,
            'returned_count': len(records), 'outcome': 'network', 'error': None, 'attempts': 1,
            'exchanges': [{'attempt': 1, 'status': 200}], 'request_count': 1, 'response_count': 1, 'body_sha': sha})
    for qi, (lens, raw) in enumerate(queries(meta)):
        for si, src in enumerate(POLICY['sources']):
            records = [record(qi * 100 + si * 10 + j + 1, src) for j in range(6)] if populated else []
            append('search', search_request(raw, src, arm[:2]), records, qi, lens, raw, src)
            for item in records:
                pool.add(item, (lens, qi, src))
    seeds = [set(pool.items[k]['ids']) for k in pool.seeds(arm.endswith('S1'))]
    for ids in seeds:
        for ki, kind in enumerate(POLICY['expansion_kinds']):
            n = int(digest(sorted(ids))[:6], 16) % 40000 + 50000
            append(kind, expansion_request(ids, kind), [record(n + ki)], seed=sorted(ids))
    return envelope(data)


def abort_at(envelope_value, index, outcome='cooldown'):
    data = deepcopy(envelope_value['payload'])
    data['calls'] = data['calls'][:index + 1]
    call = data['calls'][-1]
    call.update(records=[], returned_count=0, outcome=outcome, error='synthetic terminal failure',
                attempts=0, exchanges=[], request_count=0, response_count=0, body_sha=None)
    if outcome == 'error':
        call.update(attempts=4, exchanges=[{'attempt': i, 'status': 429} for i in range(1, 5)],
                    request_count=4, response_count=4)
    return envelope(data)


def study():
    entries, scopes = [], {}
    for si, (scope, count) in enumerate(zip(POLICY['required_exclusion_scopes'], [11, 19, 109, 1])):
        items = [{'scope': scope, 'identity': {'doi': f'10.99999/excluded-{si}-{i}'},
                  'origin_sha256': digest(['synthetic inventory', si])} for i in range(count)]
        entries.extend(items)
        scopes[scope] = {'status': 'complete', 'inventory_sha256': digest(items), 'reason': 'synthetic fixture only'}
    registry = {'scopes': scopes, 'entries': entries, 'unknown_aliases': ['Synthetic fixture, no global identity claim.']}
    cards, labels = [], {}
    for i in range(12):
        card = 'synthetic-card-' + str(i)
        meta = metadata()
        meta['concept'] += ' ' + str(i)
        refs = [{'doi': record(1)['doi']}]
        labels[card] = refs
        cards.append({'card_id': card, 'family': 'family' + str(i // 3), 'metadata': meta,
            'mechanism_identity': {'title': 'Synthetic external mechanism ' + str(i)},
            'reference_identities': refs, 'provenance_sha256': digest(['synthetic provenance', i])})
    corpus = {'status': 'frozen', 'reason': 'Synthetic control fixture only, not a held-out study.',
              'cards': cards, 'vault_sha256': digest(labels)}
    return corpus, registry, labels
