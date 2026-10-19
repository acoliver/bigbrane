"""Offline factorial replay. Inputs are external boundaries; malformed receipts fail fast."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata
from urllib.parse import quote

from offline import install

install()
ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / 'policy.json').read_text())
SELECT = ('id,doi,title,display_name,publication_year,authorships,primary_location,'
          'cited_by_count,abstract_inverted_index,referenced_works')
FIELDS = 'paperId,externalIds,url,title,abstract,year,authors,venue,citationCount'
SOURCES = {'arxiv': 'arxiv', 'openalex': 'openalex', 's2': 's2', 'semantic_scholar': 's2'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pin(path, pin):
    require(file_sha(path) == Path(pin).read_text().split()[0], 'file SHA mismatch: ' + str(path))


def verify_freeze():
    for line in (ROOT / 'PREREGISTRATION.sha256').read_text().splitlines():
        sha, name = line.split()
        require(file_sha(ROOT / name) == sha, 'preregistration changed')


def source(value):
    require(value in SOURCES, 'unknown source')
    return SOURCES[value]


def norm_title(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value).casefold() if c.isalnum())


def arxiv(value):
    value = re.sub(r'^https?://(?:www\.)?arxiv.org/(?:abs|pdf)/', '', value.strip(), flags=re.I)
    value = re.sub(r'^arxiv:', '', value, flags=re.I).removesuffix('.pdf')
    value = re.sub(r'v\d+$', '', value, flags=re.I).lower()
    require(bool(re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})', value)), 'bad arxiv ID')
    return value


def doi(value):
    value = re.sub(r'^(?:https?://(?:dx\.)?doi.org/|doi:)', '', value.strip(), flags=re.I).casefold()
    require(bool(re.fullmatch(r'10\.\d{4,9}/\S+', value)), 'bad DOI')
    return value


def public_id(src, value):
    src = source(src)
    if src == 'arxiv':
        value = arxiv(value)
    elif src == 'openalex':
        value = value.removeprefix('https://openalex.org/').upper()
        require(bool(re.fullmatch(r'W\d+', value)), 'bad OpenAlex ID')
    else:
        value = value.lower()
        require(bool(re.fullmatch(r'[0-9a-f]{40}', value)), 'bad S2 ID')
    return src + ':' + value


def identities(item):
    require(isinstance(item, dict), 'identity must be an object')
    require(set(item) <= {'title', 'arxiv_id', 'doi', 'source', 'source_id', 'aliases'}, 'unexpected identity field')
    ids = set()
    if item.get('title'):
        title = norm_title(item['title'])
        require(bool(title), 'empty normalized title')
        ids.add('title:' + title)
    if item.get('arxiv_id'):
        ids.add('arxiv:' + arxiv(item['arxiv_id']))
    if item.get('doi'):
        d = doi(item['doi'])
        ids.add('doi:' + d)
        if d.startswith('10.48550/arxiv.'):
            ids.add('arxiv:' + arxiv(d.removeprefix('10.48550/arxiv.')))
    if 'source' in item or 'source_id' in item:
        ids.add(public_id(item['source'], item['source_id']))
    for alias in item.get('aliases', []):
        require('aliases' not in alias, 'nested alias')
        ids.update(identities(alias))
    require(bool(ids), 'identity has no identifiers')
    return ids


def queries(metadata):
    require(set(metadata) == {'concept', 'lenses'}, 'metadata contains unregistered fields')
    require(isinstance(metadata['concept'], str) and metadata['concept'].strip(), 'missing concept')
    require(len(metadata['lenses']) == POLICY['lenses'], 'need four lenses')
    result, lens_names = [], set()
    for lens in metadata['lenses']:
        require(set(lens) == {'name', 'phrases'}, 'bad lens fields')
        require(isinstance(lens['name'], str) and lens['name'] and lens['name'] not in lens_names, 'duplicate lens')
        lens_names.add(lens['name'])
        require(len(lens['phrases']) == POLICY['phrases_per_lens'], 'need three phrases')
        for phrase in lens['phrases']:
            require(isinstance(phrase, str) and phrase.strip(), 'empty phrase')
            raw = ' '.join((metadata['concept'] + ' ' + phrase).split())
            require(bool(re.findall(r'[A-Za-z0-9]+', raw)), 'no ASCII search terms')
            result.append((lens['name'], raw))
    require(len({q.casefold() for _, q in result}) == 12, 'duplicate query')
    return result


def search_request(raw, src, qpolicy):
    require(qpolicy in ('Q0', 'Q1'), 'unknown query policy')
    src = source(src)
    if src == 'arxiv':
        tokens = re.findall(r'[A-Za-z0-9]+', raw)
        require(bool(tokens), 'no ASCII search terms')
        query = ' AND '.join('all:' + t if qpolicy == 'Q0' else f'(ti:{t} OR abs:{t})' for t in tokens)
        return {'endpoint': 'https://export.arxiv.org/api/query', 'params': {
            'search_query': query, 'start': '0', 'max_results': '6', 'sortBy': 'relevance'}}
    if src == 'openalex':
        return {'endpoint': 'https://api.openalex.org/works',
                'params': {'search': raw, 'per-page': '6', 'select': SELECT}}
    return {'endpoint': 'https://api.semanticscholar.org/graph/v1/paper/search',
            'params': {'query': raw, 'limit': '6', 'fields': FIELDS}}


class Pool:
    def __init__(self):
        self.items = {}
        self.owners = {}
        self.next_id = 0

    def add(self, record, stratum=None):
        ids = identities(record)
        matches = {self.owners[i] for i in ids if i in self.owners}
        key = min(matches) if matches else self.next_id
        if not matches:
            self.next_id += 1
            self.items[key] = {'key': key, 'ids': set(), 'strata': set()}
        kept = self.items[key]
        for other in sorted(matches - {key}):
            merged = self.items.pop(other)
            kept['ids'].update(merged['ids'])
            kept['strata'].update(merged['strata'])
        kept['ids'].update(ids)
        if stratum is not None:
            kept['strata'].add(stratum)
        for identity in kept['ids']:
            self.owners[identity] = key
        return key

    def ranked(self):
        return sorted(self.items, key=lambda k: (-max(1, len({s[1] for s in self.items[k]['strata']})), k))

    def seeds(self, balanced):
        ranked = self.ranked()
        if not balanced:
            return ranked[:POLICY['seed_limit']]
        chosen, lc, qc, sc = [], {}, {}, {}
        while len(chosen) < min(POLICY['seed_limit'], len(ranked)):
            options = []
            for rank, key in enumerate(ranked):
                if key not in chosen:
                    for lens, qi, src in sorted(self.items[key]['strata']):
                        options.append(((lc.get(lens, 0), qc.get(qi, 0), sc.get(src, 0), rank,
                                         qi, POLICY['sources'].index(src)), key, lens, qi, src))
            _, key, lens, qi, src = min(options)
            chosen.append(key)
            lc[lens], qc[qi], sc[src] = lc.get(lens, 0) + 1, qc.get(qi, 0) + 1, sc.get(src, 0) + 1
        return chosen

    def exposed(self):
        ranked = self.ranked()
        fields = dict.fromkeys(s[0] for k in ranked for s in sorted(self.items[k]['strata']))
        chosen = set()
        for field in fields:
            if len(chosen) < POLICY['exposure_limit']:
                chosen.add(next(k for k in ranked if any(s[0] == field for s in self.items[k]['strata'])))
        for key in ranked:
            if len(chosen) >= POLICY['exposure_limit']:
                break
            chosen.add(key)
        return [k for k in ranked if k in chosen]


def expansion_request(ids, kind):
    lookup = next((i[3:] for i in sorted(ids) if i.startswith('s2:')), None)
    if lookup is None:
        lookup = next(('ARXIV:' + i[6:] for i in sorted(ids) if i.startswith('arxiv:')), None)
    if lookup is None:
        lookup = next(('DOI:' + i[4:] for i in sorted(ids) if i.startswith('doi:')), None)
    if lookup is None:
        return None
    return {'endpoint': 'https://api.semanticscholar.org/graph/v1/paper/' + quote(lookup, safe=':') + '/' + kind,
            'params': {'fields': FIELDS, 'limit': '6'}}


CALL_FIELDS = {'card_id', 'arm', 'call_id', 'operation', 'query_index', 'lens', 'query_sha', 'source',
               'request', 'seed', 'limit', 'records', 'returned_count', 'outcome', 'error', 'attempts',
               'exchanges', 'request_count', 'response_count', 'body_sha'}


def validate_call(call, expected, blobs):
    require(set(call) == CALL_FIELDS, 'bad call schema')
    for key, value in expected.items():
        actual = source(call[key]) if key == 'source' else call[key]
        require(actual == value, 'call identity mismatch: ' + key)
    require(type(call['limit']) is int and call['limit'] == 6, 'window limit mismatch')
    records = call['records']
    require(isinstance(records, list) and len(records) <= 6 and type(call['returned_count']) is int and call['returned_count'] == len(records), 'bad returns')
    for record in records:
        require(source(record['source']) == source(call['source']), 'return source mismatch')
        identities(record)
    outcome = call['outcome']
    require(outcome in {'network', 'cache', 'cooldown', 'unavailable', 'error'}, 'unknown outcome')
    require(type(call['attempts']) is int and 0 <= call['attempts'] <= POLICY['max_attempts'], 'attempt count')
    exchanges = call['exchanges']
    require(isinstance(exchanges, list), 'missing exchanges')
    for i, exchange in enumerate(exchanges):
        require(set(exchange) == {'attempt', 'status'}, 'bad exchange schema')
        require(type(exchange['attempt']) is int and 1 <= exchange['attempt'] <= call['attempts'], 'exchange attempt')
        status = exchange['status']
        require(status is None or type(status) is int and 100 <= status <= 599, 'bad status')
        if i and exchange['attempt'] == exchanges[i - 1]['attempt']:
            require(exchanges[i - 1]['status'] in {301, 302, 303, 307, 308}, 'extra exchange without redirect')
        elif i:
            require(exchange['attempt'] == exchanges[i - 1]['attempt'] + 1, 'attempt ordering')
        else:
            require(exchange['attempt'] == 1, 'first attempt mismatch')
    require(type(call['request_count']) is int and call['request_count'] == len(exchanges), 'request count')
    require(type(call['response_count']) is int and call['response_count'] == sum(e['status'] is not None for e in exchanges), 'response count')
    require(set(e['attempt'] for e in exchanges) == set(range(1, call['attempts'] + 1)), 'attempt/exchange mismatch')
    require(all(sum(e['attempt'] == a for e in exchanges) <= 21 for a in range(1, call['attempts'] + 1)), 'redirect cap')
    success = outcome in {'network', 'cache'}
    require(call['error'] is None if success else isinstance(call['error'], str) and bool(call['error']), 'terminal error mismatch')
    if outcome in {'cache', 'cooldown', 'unavailable'}:
        require(call['attempts'] == 0 and not exchanges, 'zero-network outcome dispatched HTTP')
    if outcome == 'network':
        require(bool(exchanges) and exchanges[-1]['status'] is not None and 200 <= exchanges[-1]['status'] < 300, 'network success status')
    if outcome == 'error':
        require(call['attempts'] > 0, 'network error without attempts')
    if success:
        sha = call['body_sha']
        require(sha in blobs and digest(blobs[sha]) == sha, 'missing/tampered response blob')
        require(blobs[sha] == {'request': call['request'], 'records': records}, 'body request/return mismatch')
    else:
        require(not records and call['body_sha'] is None, 'failed call has records/body')
    require(call['request'] is not None or outcome == 'unavailable', 'missing request without unavailable')
    return success


def unwrap(envelope):
    require(set(envelope) == {'payload', 'sha256'}, 'bad immutable envelope')
    require(digest(envelope['payload']) == envelope['sha256'], 'pool SHA mismatch')
    return envelope['payload']


def replay(envelope):
    data = unwrap(envelope)
    require(set(data) == {'card_id', 'arm', 'metadata', 'acquisition_id', 'calls', 'blobs', 'evidence'}, 'bad pool schema')
    arm = data['arm']
    require(arm in POLICY['arms'] and isinstance(data['acquisition_id'], str) and data['acquisition_id'], 'bad arm/acquisition identity')
    qs = queries(data['metadata'])
    calls, blobs = data['calls'], data['blobs']
    require(isinstance(calls, list), 'missing calls')
    for sha, blob in blobs.items():
        require(digest(blob) == sha, 'tampered blob')
    pool, index, answered, aborted = Pool(), 0, 0, False
    counts = {'search': 0, 'expansion': 0, 'attempts': 0, 'requests': 0, 'responses': 0, 'redirects': 0, 'retries': 0}
    outcomes = {}

    def consume(operation, qi=None, lens=None, raw=None, src='s2', seed=None, request=None):
        nonlocal index, answered, aborted
        require(index < len(calls), 'missing required call')
        call_id = f"{data['card_id']}:{arm}:{index:03d}"
        expected = {'card_id': data['card_id'], 'arm': arm, 'call_id': call_id, 'operation': operation,
                    'query_index': qi, 'lens': lens, 'query_sha': hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None,
                    'source': src, 'request': request, 'seed': seed}
        call = calls[index]
        ok = validate_call(call, expected, blobs)
        require(request is not None or call['outcome'] == 'unavailable', 'unresolved seed must be unavailable')
        index += 1
        counts['search' if operation == 'search' else 'expansion'] += 1
        counts['attempts'] += call['attempts']
        counts['requests'] += call['request_count']
        counts['responses'] += call['response_count']
        counts['retries'] += max(0, call['attempts'] - 1)
        counts['redirects'] += sum(e['status'] in {301, 302, 303, 307, 308} for e in call['exchanges'])
        outcomes[call['outcome']] = outcomes.get(call['outcome'], 0) + 1
        answered += ok
        aborted = not ok
        return call['records']

    acquired_ids, expanded_ids = set(), set()
    for qi, (lens, raw) in enumerate(qs):
        for src in POLICY['sources']:
            records = consume('search', qi, lens, raw, src, request=search_request(raw, src, arm[:2]))
            for record in records:
                acquired_ids.update(identities(record))
                pool.add(record, (lens, qi, src))
            if aborted:
                break
        if aborted:
            break
    seeds = [] if aborted else pool.seeds(arm.endswith('S1'))
    seed_ids = [set(pool.items[k]['ids']) for k in seeds]
    for ids in seed_ids:
        for kind in POLICY['expansion_kinds']:
            records = consume(kind, seed=sorted(ids), request=expansion_request(ids, kind))
            for record in records:
                expanded_ids.update(identities(record))
                pool.add(record)
            if aborted:
                break
        if aborted:
            break
    require(index == len(calls), 'extra calls after budget/abort')
    exposed = pool.exposed()
    exposed_ids = set().union(*(pool.items[k]['ids'] for k in exposed)) if exposed else set()
    evidence_sets = {'fetched': None, 'reference_used': None}
    require(set(data['evidence']) <= set(evidence_sets), 'unknown evidence stage')
    for stage, entries in data['evidence'].items():
        require(isinstance(entries, list), 'evidence entries must be a list')
        evidence_sets[stage] = set() if entries else None
        for entry in entries:
            require(set(entry) == {'identity', 'artifact_sha'}, 'bad evidence schema')
            ids = identities(entry['identity'])
            owners = {pool.owners[i] for i in ids if i in pool.owners}
            require(len(owners) == 1, 'evidence identity not uniquely acquired')
            key = next(iter(owners))
            require(stage != 'reference_used' or key in exposed, 'reference-used item not exposed')
            artifact = blobs[entry['artifact_sha']]
            require(set(artifact) == {'stage', 'identity', 'text'} and artifact['stage'] == stage,
                    'wrong artifact stage')
            require(identities(artifact['identity']) == ids and isinstance(artifact['text'], str) and artifact['text'].strip(), 'evidence lacks content/identity')
            evidence_sets[stage].update(pool.items[key]['ids'])
    acquired_ids = set().union(*(pool.items[pool.owners[i]]['ids'] for i in acquired_ids)) if acquired_ids else set()
    expanded_ids = set().union(*(pool.items[pool.owners[i]]['ids'] for i in expanded_ids)) if expanded_ids else set()
    stages = {'acquired': acquired_ids, 'expanded': expanded_ids, 'exposed': exposed_ids, **evidence_sets}
    unique_counts = {name: None if ids is None else len({pool.owners[i] for i in ids}) for name, ids in stages.items()}
    return {'card_id': data['card_id'], 'arm': arm, 'acquisition_id': data['acquisition_id'],
            'pool_sha256': envelope['sha256'], 'metadata_sha256': digest(data['metadata']),
            'search_pool_sha256': digest([{k: v for k, v in c.items() if k not in {'card_id', 'arm', 'call_id'}} for c in calls if c['operation'] == 'search']),
            'complete': not aborted, 'consumed': counts, 'outcomes': outcomes,
            'availability': {'answered': answered, 'consumed': index, 'ratio': answered / index},
            'unspent': {'search': 36 - counts['search'], 'expansion': 10 - counts['expansion'], 'seed': 5 - len(seeds)},
            'seeds': [sorted(ids) for ids in seed_ids], 'exposed': [sorted(pool.items[k]['ids']) for k in exposed],
            'stage_ids': {name: None if ids is None else sorted(ids) for name, ids in stages.items()},
            'unique_counts': unique_counts,
            'identity_limit': 'Known aliases and equal normalized titles only; unresolved aliases may remain.'}


def corpus_gate(corpus, registry):
    require(set(registry) == {'scopes', 'entries', 'unknown_aliases'}, 'bad registry schema')
    excluded = set()
    for entry in registry['entries']:
        require(set(entry) == {'scope', 'identity', 'origin_sha256'}, 'bad exclusion entry')
        require(entry['scope'] in POLICY['required_exclusion_scopes'], 'unknown exclusion scope')
        require(bool(re.fullmatch('[a-f0-9]{64}', entry['origin_sha256'])), 'missing exclusion provenance')
        excluded.update(identities(entry['identity']))
    require(set(corpus) == {'status', 'reason', 'cards', 'vault_sha256'}, 'bad corpus schema')
    mechanisms = Pool()
    for card in corpus['cards']:
        require(set(card) == {'card_id', 'family', 'metadata', 'mechanism_identity', 'reference_identities', 'provenance_sha256'}, 'bad corpus card')
        queries(card['metadata'])
        before = len(mechanisms.items)
        mechanisms.add(card['mechanism_identity'])
        require(len(mechanisms.items) == before + 1, 'duplicate mechanism identity')
        for identity in [card['mechanism_identity'], *card['reference_identities']]:
            require(not identities(identity).intersection(excluded), 'held-out/exclusion intersection')
    require(set(registry['scopes']) == set(POLICY['required_exclusion_scopes']), 'missing exclusion scope')
    pending = []
    minimums = dict(zip(POLICY['required_exclusion_scopes'], [11, 19, 109, 1]))
    for scope, inventory in registry['scopes'].items():
        require(set(inventory) == {'status', 'inventory_sha256', 'reason'}, 'bad inventory schema')
        if inventory['status'] != 'complete':
            pending.append(scope)
            continue
        entries = [e for e in registry['entries'] if e['scope'] == scope]
        require(digest(entries) == inventory['inventory_sha256'], 'exclusion inventory SHA mismatch')
        unique = Pool()
        for entry in entries:
            unique.add(entry['identity'])
        require(len(unique.items) >= minimums[scope], 'exclusion inventory too small')
    if not corpus['cards'] or pending or corpus['status'] != 'frozen':
        return {'status': 'PENDING', 'reason': corpus['reason'], 'incomplete_exclusions': pending,
                'disjointness': 'No global disjointness claim; aliases remain uncertain.'}
    require(len(corpus['cards']) == POLICY['cards'], 'planned sample must contain 12 cards')
    require(len({c['card_id'] for c in corpus['cards']}) == 12, 'duplicate card')
    families = [c['family'] for c in corpus['cards']]
    require(len(set(families)) >= 4 and max(families.count(f) for f in families) <= 3, 'family allocation')
    require(bool(re.fullmatch('[a-f0-9]{64}', corpus['vault_sha256'])), 'missing frozen label vault hash')
    for card in corpus['cards']:
        require(bool(card['reference_identities']) and bool(re.fullmatch('[a-f0-9]{64}', card['provenance_sha256'])), 'missing card provenance/references')
    return {'status': 'READY_FOR_EXTERNAL_ACQUISITION', 'disjointness': 'Conditional on inventoried aliases; human independence still required.'}


def coverage(result, references):
    refpool = Pool()
    for reference in references:
        refpool.add(reference)
    denominator = len(refpool.items)
    require(denominator > 0, 'zero reference denominator')
    metrics = {}
    for stage, ids in result['stage_ids'].items():
        hits = None if ids is None else sum(bool(item['ids'].intersection(ids)) for item in refpool.items.values())
        metrics[stage] = {'hits': hits, 'denominator': denominator,
                          'coverage': None if hits is None else hits / denominator,
                          'status': 'PENDING evidence' if hits is None else 'observed'}
    return metrics


def contrast(values):
    a, b, c, d = (values[arm] for arm in POLICY['arms'])
    return {'query_at_S0': c - a, 'query_at_S1': d - b, 'seed_at_Q0': b - a,
            'seed_at_Q1': d - c, 'query_average': ((c - a) + (d - b)) / 2,
            'seed_average': ((b - a) + (d - c)) / 2, 'interaction': (d - c) - (b - a)}


def uncertainty(values):
    n = len(values)
    rng = random.Random(POLICY['bootstrap_seed'])
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(POLICY['bootstrap_resamples']))
    pos, neg = sum(v > 0 for v in values), sum(v < 0 for v in values)
    m = pos + neg
    p = min(1.0, 2 * sum(math.comb(m, k) for k in range(min(pos, neg) + 1)) / 2 ** m) if m else 1.0
    return {'mean': sum(values) / n, 'bootstrap_95': [means[249], means[9749]],
            'positive': pos, 'negative': neg, 'ties': n - m, 'sign_test_two_sided': p}


def factorial(envelopes, corpus, registry, labels=None):
    gate = corpus_gate(corpus, registry)
    if gate['status'] == 'PENDING':
        return {'status': 'PENDING', 'corpus': gate, 'query_effect': 'unmeasured', 'acceptance_improvement': False}
    expected_cards = {c['card_id']: c for c in corpus['cards']}
    results = {}
    for envelope in envelopes:
        result = replay(envelope)
        key = (result['card_id'], result['arm'])
        require(key[0] in expected_cards and key not in results, 'foreign/duplicate card arm')
        require(result['metadata_sha256'] == digest(expected_cards[key[0]]['metadata']), 'cross-arm metadata mismatch')
        results[key] = result
    require(set(results) == {(card, arm) for card in expected_cards for arm in POLICY['arms']}, 'missing arm')
    for card in expected_cards:
        for q in ('Q0', 'Q1'):
            left, right = results[card, q + 'S0'], results[card, q + 'S1']
            require(left['acquisition_id'] == right['acquisition_id'] and left['search_pool_sha256'] == right['search_pool_sha256'], 'seed arms must share fixed query acquisition')
        require({results[card, a]['acquisition_id'] for a in ('Q0S0', 'Q0S1')}.isdisjoint(
                {results[card, a]['acquisition_id'] for a in ('Q1S0', 'Q1S1')}), 'query arms reuse acquisition pool')
    output = {'status': 'PENDING labels', 'corpus': gate, 'arms': list(results.values()), 'acceptance_improvement': False,
              'query_effect': 'Requires independently acquired query-specific receipts; replay cannot authenticate provider history.'}
    if labels is None:
        return output
    require(digest(labels) == corpus['vault_sha256'], 'label vault SHA mismatch')
    require(set(labels) == set(expected_cards), 'label card mismatch')
    rows, sensitivity = [], []
    for card, manifest in expected_cards.items():
        references = labels[card]
        require(digest(references) == digest(manifest['reference_identities']), 'reference identity manifest mismatch')
        metrics = {arm: coverage(results[card, arm], references) for arm in POLICY['arms']}
        complete = all(results[card, arm]['complete'] for arm in POLICY['arms'])
        sens = {a: metrics[a]['exposed']['coverage'] if results[card, a]['complete'] else 0 for a in POLICY['arms']}
        sensitivity.append(contrast(sens))
        if complete:
            rows.append({'card_id': card, 'metrics': metrics,
                         'contrasts': contrast({a: metrics[a]['exposed']['coverage'] for a in POLICY['arms']})})
    output.update(status='pilot_accounting_only', complete_cards=len(rows), incomplete_cards=12 - len(rows),
                  paired_cards=rows, intention_to_acquire={k: sum(r[k] for r in sensitivity) / 12 for k in sensitivity[0]})
    output['contrasts'] = {k: uncertainty([r['contrasts'][k] for r in rows]) for k in rows[0]['contrasts']} if rows else {}
    decisions = {}
    for policy, fixed in [('query', ['query_at_S0', 'query_at_S1']), ('seed', ['seed_at_Q0', 'seed_at_Q1'])]:
        if not rows:
            decision = 'inconclusive: no complete pairs'
        elif any(output['contrasts'][k]['mean'] <= 0 for k in fixed):
            decision = 'broad improvement falsified in pilot'
        elif len(rows) < 12 or any(r['metrics'][a]['reference_used']['coverage'] is None for r in rows for a in POLICY['arms']):
            decision = 'inconclusive: incomplete transport or usage evidence'
        else:
            use_effects = [contrast({a: r['metrics'][a]['reference_used']['coverage'] for a in POLICY['arms']})[policy + '_average'] for r in rows]
            decision = ('eligible for further independent testing, never automatic rollout'
                        if output['contrasts'][policy + '_average']['bootstrap_95'][0] > 0 and sum(use_effects) >= 0
                        else 'inconclusive: uncertainty or usage safety condition')
        decisions[policy] = decision
    output['decision'] = decisions
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pools', type=Path)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'corpus.json')
    parser.add_argument('--registry', type=Path, default=ROOT / 'exclusions.json')
    parser.add_argument('--registry-pin', type=Path, default=ROOT / 'exclusions.sha256')
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    verify_freeze()
    check_pin(args.registry, args.registry_pin)
    corpus, registry = json.loads(args.corpus.read_text()), json.loads(args.registry.read_text())
    gate = corpus_gate(corpus, registry)
    if gate['status'] == 'PENDING':
        result = factorial([], corpus, registry)
    else:
        require(args.pools is not None, 'missing query-specific receipt pools')
        result = factorial(json.loads(args.pools.read_text()), corpus, registry,
                           json.loads(args.labels.read_text()) if args.labels else None)
    paths = [args.corpus, args.registry, ROOT / 'PLAN.md', ROOT / 'policy.json', ROOT / 'harness.py']
    if args.pools is not None:
        paths.append(args.pools)
    if args.labels is not None:
        paths.append(args.labels)
    result['input_sha256'] = {str(p): file_sha(p) for p in paths}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
