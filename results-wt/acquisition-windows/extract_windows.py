"""Deterministic offline projection of pinned acquisition evidence, not acceptance."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import socket
import xml.etree.ElementTree as ET

from hybrid.literature.paper import Paper, dedupe, normalize_title, paper_key_of
from hybrid.literature.sources import arxiv, openalex, semantic_scholar, arxiv_local
from pin_inputs import sha, OUT, WT, MAIN

MODULES = {'arxiv': arxiv, 'openalex': openalex, 's2': semantic_scholar}
ATOM = '{http://www.w3.org/2005/Atom}'
OPEN_SEARCH = ('http://a9.com/-/spec/opensearch/1.1/', 'http://a9.com/-/spec/opensearch/1.0/')
PARAMS = frozenset(('search_query', 'start', 'max_results', 'sortBy', 'search', 'per-page', 'select', 'query', 'limit', 'fields'))


def block_network(*args, **kwargs):
    raise AssertionError('Network prohibited in acquisition-window investigation')


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def query_sha(query):
    return hashlib.sha256(query.encode()).hexdigest()


def tokens(text):
    return re.findall(r'[a-z0-9]+', (text or '').casefold())


def integer(value):
    if value is None:
        return None
    if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
        raise ValueError('Invalid nonnegative pagination/count metadata')
    return int(value)


def pool_class(total, start, count, next_page):
    if total == 0:
        if count:
            raise ValueError('Zero matching pool contains records')
        return 'explicit_zero_pool'
    if next_page is not None or (total is not None and start is not None and total > start + count):
        return 'more_unseen_records'
    if total is not None and start is not None:
        return 'exhausted_nonempty_pool' if count else 'empty_page_nonzero_pool'
    return 'unknown_total_or_start'


def response_metadata(source, body, status=200):
    """Parse only declared provider fields; absent totals must stay unknown."""
    if status != 200:
        return {'pool_class': 'non_pool_http_status', 'status': status, 'total': None, 'raw_count': None}
    if source == 'arxiv':
        root = ET.fromstring(body)
        if root.tag != ATOM + 'feed':
            raise ValueError('Not an Atom feed')
        entries = root.findall(ATOM + 'entry')
        if any('/api/errors' in e.findtext(ATOM + 'id', '') for e in entries):
            return {'pool_class': 'api_error_not_pool', 'total': None, 'raw_count': None}
        values = {}
        fields = {'total': 'totalResults', 'start': 'startIndex', 'items_per_page': 'itemsPerPage'}
        for label, field in fields.items():
            found = [root.findtext('{' + ns + '}' + field) for ns in OPEN_SEARCH]
            present = [value for value in found if value is not None]
            assert len(present) <= 1, 'Ambiguous OpenSearch namespaces'
            values[label] = integer(present[0]) if present else None
        total, start, size = values['total'], values['start'], values['items_per_page']
        next_page = None
        total_field = 'opensearch:totalResults'
        count = len(entries)
        metadata = values
        # Links expose server-parsed conjunction nesting; do not emit raw hrefs.
        from urllib.parse import urlsplit, parse_qs
        server_queries = []
        for link in root.findall(ATOM + 'link'):
            query = parse_qs(urlsplit(link.get('href', '')).query).get('search_query')
            if query:
                server_queries.append(query[0])
        metadata['server_search_queries'] = server_queries
    else:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError('Not a JSON object')
        if 'error' in payload or 'message' in payload and not any(k in payload for k in ('results', 'data', 'total')):
            return {'pool_class': 'api_error_not_pool', 'total': None, 'raw_count': None}
        if source == 'openalex':
            items = payload.get('results')
            if not isinstance(items, list):
                raise ValueError('OpenAlex list response lacks results array')
            meta = payload.get('meta') or {}
            total = integer(meta.get('count'))
            page, size = integer(meta.get('page')), integer(meta.get('per_page'))
            start = (page - 1) * size if page is not None and size is not None else None
            next_page = meta.get('next_cursor')
            total_field = 'meta.count'
            metadata = {'page': page, 'items_per_page': size, 'next_cursor_present': next_page is not None}
        elif source == 's2':
            total = integer(payload.get('total'))
            items = payload.get('data')
            if items is None and total == 0:
                items = []
            if not isinstance(items, list):
                raise ValueError('S2 list response lacks data array')
            start = integer(payload.get('offset'))
            next_page = integer(payload.get('next'))
            size = None
            total_field = 'total'
            metadata = {'offset': start, 'next': next_page}
        else:
            raise ValueError('Unknown source')
        count = len(items)
    return {**metadata, 'total': total, 'total_field': total_field, 'start': start,
            'raw_count': count, 'pool_class': pool_class(total, start, count, next_page),
            'unseen_count_after_page': max(0, total - start - count) if total is not None and start is not None else None}


def public_request(source, query, limit):
    req = MODULES[source].search_request(query, limit)
    return {'url': req.url, 'params': dict(req.params)}


def normalized_query(query):
    ts = re.findall(r'[A-Za-z0-9]+', query)
    local = arxiv_local.match_expressions(query)
    local_terms = re.findall(r'"([^"]*)"', local[0])
    return {'ascii_tokens': ts, 'ascii_token_count': len(ts), 'unique_ascii_tokens': list(dict.fromkeys(t.casefold() for t in ts)),
            'arxiv_ast': {'operator': 'AND', 'children': [{'field': 'all', 'term': t} for t in ts]},
            'local_expressions': local, 'local_unique_terms': local_terms,
            'local_removed_tokens': sorted(set(t.casefold() for t in ts) - set(local_terms)),
            'remote_arxiv_fields': 'all; not limited to saved title/abstract; full field scope not resolved offline',
            'openalex_s2_boolean_and_fields': 'Provider interpretation unknown; adapter passes original string unchanged'}


def literal_coverage(query, paper):
    q = set(tokens(query))
    title = set(tokens(paper.get('title')))
    abstract = set(tokens(paper.get('abstract')))
    union = title | abstract
    matched = q & union
    return {'query_unique_tokens': len(q), 'matched_tokens': len(matched), 'all_literal_tokens_in_title_abstract': q <= union,
            'any_literal_token_in_title_abstract': bool(matched), 'abstract_present': bool(paper.get('abstract')),
            'missing_literal_tokens': sorted(q - union)}


def project_error(error):
    if not error:
        return None
    if 'cooling down' in error:
        return {'kind': 'cooldown_skip'}
    match = re.fullmatch(r'HTTP (\d{3}) after (\d+) attempts', error)
    if match:
        return {'kind': 'exhausted_http', 'status': int(match[1]), 'declared_attempts': int(match[2])}
    return {'kind': 'other_source_error_details_omitted'}


def assert_attribution(row, card, source, query):
    assert row['card_id'] == card, 'card attribution mismatch'
    assert row['source'] == source, 'source attribution mismatch'
    assert row['query'] == query, 'query attribution mismatch'


def signature(source, url, params):
    safe = {k: v for k, v in params.items() if k in PARAMS}
    return source, url, json.dumps(safe, sort_keys=True)


def index_envelopes(paths):
    indexed = defaultdict(list)
    for path in paths:
        envelope = json.loads(path.read_text())
        indexed[signature(envelope['source'], envelope['url'], envelope['params'])].append((path, envelope))
    return indexed


def find_envelope(indexed, source, query, limit, historical):
    request = public_request(source, query, limit)
    url = 'http://export.arxiv.org/api/query' if historical and source == 'arxiv' else request['url']
    choices = indexed.get(signature(source, url, request['params']), [])
    if not choices:
        return None
    assert len(choices) == 1, 'Ambiguous raw cache envelope'
    return choices[0]


def window_row(cid, search, source, index, historical, expected_papers=None):
    query, limit = search['query'], search['limit']
    expected_ids = search['per_source_ids'][source]
    row = {'card_id': cid, 'source': source, 'query': query, 'query_sha256': query_sha(query), 'limit': limit,
           'seq': search['seq'], 'round': search['round'], 'field': search['field'],
           'source_order_position': search['sources'].index(source) + 1,
           'expected_count': len(expected_ids), 'result_ids': expected_ids, 'historical': historical}
    error = project_error(search['errors'].get(source))
    if error:
        assert not expected_ids
        return {**row, 'availability': error['kind'], 'error': error, 'pool_class': 'unobserved_error', 'total': None}
    if source == 'arxiv_local':
        return {**row, 'availability': 'historical_local_receipt', 'pool_class': 'local_pool_not_recomputed', 'total': None}
    row['request'] = public_request(source, query, limit)
    if historical and source == 'arxiv':
        row['request']['url'] = 'http://export.arxiv.org/api/query'
    found = find_envelope(index, source, query, limit, historical)
    if found is None:
        return {**row, 'availability': 'missing_raw_source_not_refetched', 'pool_class': 'unknown_missing_raw', 'total': None}
    path, envelope = found
    expected_params = row['request']['params']
    assert {k: v for k, v in envelope['params'].items() if k in PARAMS} == expected_params
    parsed = MODULES[source].parse_search(envelope['body'], envelope['fetched_at'])
    ids = [p.source_id for p in parsed]
    assert ids == expected_ids, 'raw parsed IDs disagree with saved receipt'
    if expected_papers is not None:
        assert ids == [p['source_id'] for p in expected_papers]
    if historical:
        assert datetime.fromisoformat(envelope['fetched_at']) <= datetime.fromisoformat(search['event_ts'])
    meta = response_metadata(source, envelope['body'], envelope['status'])
    raw_count = meta.get('raw_count')
    return {**row, **meta, 'raw_path': str(path), 'raw_sha256': sha(path), 'fetched_at': envelope['fetched_at'],
            'availability': 'historical_raw_cache_evidence_network_path_unknown' if historical else 'saved_fresh_network_evidence',
            'parsed_count': len(parsed), 'parser_drop_count': raw_count - len(parsed) if raw_count is not None else None,
            'returned_metadata_metrics': [dict(candidate_id=paper_key_of(p), **literal_coverage(query, asdict(p))) for p in parsed]}


def generic_ordering(cid, card, raw_index):
    papers, support, origins, order = {}, defaultdict(set), defaultdict(list), {}
    queries = card['searches']
    for qi, search in enumerate(queries, 1):
        if search['round'] != 1:
            continue
        batch = [Paper(**p) for s in search['sources'] for p in search['source_records'][s]]
        for paper in dedupe(batch):
            key = paper_key_of(paper)
            if key not in papers:
                papers[key] = paper
                order[key] = len(order) + 1
            support[key].add(' '.join(search['query'].casefold().split()))
            origins[key].append({'query_position': qi, 'query_sha256': query_sha(search['query']), 'field': search['field']})
    ranked = sorted(papers, key=lambda k: -len(support[k]))
    seeds = card['refinement']['seeds']
    assert ranked[:5] == seeds, 'Historical seed order not reproduced'
    records = [{'candidate_id': key, 'rank': i, 'discovery_position': order[key], 'query_support': len(support[key]),
                'source': papers[key].source, 'query_origins': origins[key], 'selected_seed': key in seeds}
               for i, key in enumerate(ranked, 1)]
    support_hist = dict(sorted(Counter(len(v) for v in support.values()).items()))
    source_first_seen = dict(Counter(p.source for p in papers.values()))
    first_ids = {paper_key_of(p) for p in dedupe(Paper(**p) for s in queries[0]['sources'] for p in queries[0]['source_records'][s])}
    # Exact OpenAlex identities only. This is a lower bound on cross-provider bibliography overlap.
    acquired_oa = {p['source_id'] for q in queries if q['round'] == 1 for p in q['source_records']['openalex']}
    seed_oa = set()
    seed_titles = {normalize_title(papers[k].title) for k in seeds}
    seed_arxiv = {papers[k].arxiv_id for k in seeds if papers[k].arxiv_id}
    for q in queries:
        if q['round'] != 1:
            continue
        for p in q['source_records']['openalex']:
            if normalize_title(p['title']) in seed_titles or p.get('arxiv_id') in seed_arxiv:
                seed_oa.add(p['source_id'])
    bibliography = []
    for qi, q in enumerate(queries, 1):
        found = find_envelope(raw_index, 'openalex', q['query'], q['limit'], True)
        if not found:
            continue
        payload = json.loads(found[1]['body'])
        refs = [str(ref).rsplit('/', 1)[-1] for item in payload['results'] for ref in item.get('referenced_works') or []]
        bibliography.append({'query_position': qi, 'round': q['round'], 'field': q['field'], 'query_sha256': query_sha(q['query']),
                             'raw_reference_occurrences': len(refs), 'unique_raw_references': len(set(refs)),
                             'references_to_initial_acquired_openalex_ids': len(set(refs) & acquired_oa),
                             'references_to_selected_seed_openalex_ids': len(set(refs) & seed_oa)})
    final = card['current_records']
    final_support = Counter(c['query_hits'] for c in final)
    return {'card_id': cid, 'initial_candidates': len(ranked), 'initial_support_histogram': support_hist,
            'initial_single_query_tie_count': support_hist.get(1, 0), 'initial_first_seen_sources': source_first_seen,
            'selected_seeds': [r for r in records if r['selected_seed']],
            'selected_seeds_present_in_first_query': sum(k in first_ids for k in seeds),
            'selected_seed_source_counts': dict(Counter(papers[k].source for k in seeds)),
            'selected_seed_lenses': sorted({o['field'] for k in seeds for o in origins[k]}),
            'all_initial_ranked': records, 'bibliography_query_metrics': bibliography,
            'overlap_identity_policy': 'Exact OpenAlex IDs; seed aliases resolved only by exact normalized title or arXiv ID; not a complete identity census',
            'final_query_support_histogram': dict(sorted(final_support.items())),
            'related_receipts': len(card['related']),
            'empty_related_both_directions': sum(not r['reference_ids'] and not r['citation_ids'] for r in card['related'])}


def card_texts(inputs, saved):
    proposed = {cid: card['proposal']['queries'] for cid, card in saved['cards'].items()}
    result = {}
    for filename in inputs:
        if '/llm/calls/' not in filename:
            continue
        call = json.loads(Path(filename).read_text())
        if call.get('role') != 'prior_art' or not isinstance(call.get('parsed'), dict):
            continue
        qs = call['parsed'].get('queries', [])
        if not qs or not isinstance(qs[0], dict):
            continue
        strings = [q['query'] for q in qs]
        for cid, expected in proposed.items():
            if strings == expected:
                assert cid not in result
                result[cid] = {'text': call['user'], 'path': filename, 'sha256': sha(Path(filename))}
    assert set(result) == set(proposed)
    return result


def registered_evidence(saved, audit, diagnostics, matcher):
    rows = []
    for cid, card in saved['cards'].items():
        for label, match in card['matches'].items():
            work = (match['registered_arxiv_id'], match['registered_title'])
            metadata = []
            for filename, operations in diagnostics:
                for op in operations:
                    for p in op.get('papers', []):
                        if matcher.exact_match(p, work):
                            fingerprint = (p['source'], p['source_id'], p.get('abstract'))
                            if not any(m['fingerprint'] == fingerprint for m in metadata):
                                metadata.append({'fingerprint': fingerprint, 'path': filename, 'label': op['label'], 'paper': p})
            local = card['local_index'][label]
            if local.get('metadata'):
                p = local['metadata']
                metadata.append({'fingerprint': ('arxiv_local', p['arxiv_id'], p['abstract']),
                                 'path': 'retrieval-diagnosis/offline.json', 'label': 'preexisting local metadata',
                                 'paper': {'source': 'arxiv_local', 'source_id': p['arxiv_id'], **p}})
            comparisons = []
            for m in metadata:
                comparisons.append({'source': m['paper']['source'], 'source_id': m['paper']['source_id'],
                                    'evidence_file': m['path'], 'operation_label': m['label'],
                                    'metadata_sha256': query_sha(json.dumps(m['paper'], sort_keys=True)),
                                    'queries': [dict(query_sha256=query_sha(q['query']), **literal_coverage(q['query'], m['paper']))
                                                for q in card['searches']]})
            direct = [{'query_sha256': query_sha(q['query']), 'query_position': i, 'round': q['round'], 'field': q['field'],
                       'source': source, 'candidate_ids': hits[label]}
                      for i, q in enumerate(card['searches'], 1) for source, hits in q['exact_raw_matches'].items() if hits[label]]
            bib = []
            for entry in audit['pre_reference_cap_matches']:
                if label not in entry['known_work_matches']:
                    continue
                own = [c for c in entry['contexts'] if c['card_id'] == cid]
                if own:
                    bib.append({'source_work': entry['source_work'], 'reference_position': entry['reference_position'],
                                'initial_seed_ranks': [r for ctx in own for r in ctx['initial_rank_at_seed_selection']],
                                'selected_seed_keys': [k for ctx in own for k in ctx['selected_seed_keys']],
                                'cache_sha256': sha(Path(entry['cache']))})
            if match['old']:
                earliest = 'acquired_then_exposed' if match['all_prompt_mentions'] else 'acquired_then_rank_exposure_loss'
            elif bib:
                earliest = 'raw_bibliography_identity_without_fetched_work_record'
            else:
                earliest = 'absent_from_all_saved_direct_and_selected_seed_edge_windows'
            local_queries = local.get('queries', [])
            rows.append({'card_id': cid, 'label': label, 'registered_arxiv_id': work[0], 'registered_title': work[1],
                         'earliest_explicit_trace': earliest, 'direct_hits': direct, 'own_card_raw_bibliography_hits': bib,
                         'old': match['old'], 'current': match['new'], 'prompt_mention_count': len(match['all_prompt_mentions']),
                         'preexisting_local_fts': {'indexed': local['indexed'], 'query_count': len(local_queries),
                                                   'all_term_matches': sum(q['target_membership'][0] for q in local_queries),
                                                   'any_term_matches': sum(q['target_membership'][-1] for q in local_queries),
                                                   'first_query': local_queries[0] if local_queries else None},
                         'literal_metadata_comparisons': comparisons,
                         'metadata_available': bool(metadata),
                         'remote_eligibility_conclusion': 'Unknown. Literal title/abstract coverage is not provider boolean eligibility or proof of full-text exclusion.'})
    return rows


def main():
    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    socket.create_connection = block_network
    manifest = json.loads((OUT / 'INPUTS-before.json').read_text())
    for path, digest in {**manifest['original_sha256'], **manifest['supplemental_sha256']}.items():
        assert sha(Path(path)) == digest, 'Pinned input changed before extraction: ' + path
    saved = json.loads((WT / 'results-wt/retrieval-diagnosis/offline.json').read_text())
    audit = json.loads((WT / 'results-wt/retrieval-diagnosis/raw-audit.json').read_text())
    replay = json.loads((WT / 'results-wt/retrieval-diagnosis/replay-evidence.json').read_text())
    assert replay['provenance']['sha256'] == sha(WT / 'results-wt/retrieval-diagnosis/offline.json')
    plan = json.loads((WT / 'results-wt/transport-measurement/PLAN.json').read_text())
    results = json.loads((WT / 'results-wt/transport-measurement/RESULTS.json').read_text())
    summary = json.loads((WT / 'results-wt/transport-measurement/SUMMARY.json').read_text())
    historical_paths = [Path(p) for p in manifest['original_sha256'] if '/hybrid/literature/cache/' in p]
    fresh_paths = [Path(p) for p in manifest['supplemental_sha256'] if '/transport-measurement/raw-cache/' in p]
    historical_index, fresh_index = index_envelopes(historical_paths), index_envelopes(fresh_paths)
    texts = card_texts(manifest['original_sha256'], saved)
    query_contexts, windows, first_windows, ordering = [], [], [], []
    for cid, card in saved['cards'].items():
        for qi, search in enumerate(card['searches'], 1):
            assert search['card_id'] == cid
            assert search['limit'] == 6
            text_set, query_set = set(tokens(texts[cid]['text'])), set(tokens(search['query']))
            query_contexts.append({'card_id': cid, 'query': search['query'], 'query_sha256': query_sha(search['query']),
                                   'query_position': qi, 'round': search['round'], 'field': search['field'],
                                   'normalization': normalized_query(search['query']),
                                   'card_text_sha256': texts[cid]['sha256'], 'card_text_path': texts[cid]['path'],
                                   'literal_card_token_overlap': {'shared': len(query_set & text_set), 'query_unique_tokens': len(query_set),
                                                                  'absent_from_card': sorted(query_set - text_set)},
                                   'source_counts': {s: len(search['per_source_ids'][s]) for s in search['sources']}})
            windows.extend(window_row(cid, search, s, historical_index, True) for s in search['sources'])
        ordering.append(generic_ordering(cid, card, historical_index))
    fixtures = json.loads((WT / 'results-wt/transport-measurement/FIXTURES.json').read_text())
    for entry in plan['cards']:
        cid = entry['card_id']
        search = saved['cards'][cid]['searches'][0]
        assert search['query'] == entry['query'] and query_sha(search['query']) == entry['query_sha256']
        fixture = next(f for f in fixtures if f['expected_callback_card_id'] == cid)['original_receipt']
        assert fixture['seq'] == search['seq']
        for source in plan['sources']:
            result = next(r for r in results if r['card_id'] == cid and r['source'] == source)
            assert_attribution(result, cid, source, entry['query'])
            transported = next(r for r in summary['rows'] if r['card_id'] == cid and r['source'] == source)
            assert transported['query_sha256'] == entry['query_sha256']
            assert transported['request_id'] == entry['requests'][source]['request_id']
            assert transported['terminal'] == 'success' and transported['outcome'] == 'network'
            assert result['errors'] == {}
            fresh_search = {**search, 'errors': {}, 'per_source_ids': {source: [p['source_id'] for p in result['papers']]}}
            row = window_row(cid, fresh_search, source, fresh_index, False, result['papers'])
            assert row['request']['params'] == entry['requests'][source]['public_params']
            row['saved_transport'] = {k: transported[k] for k in ['call_id', 'request_id', 'outcome', 'terminal', 'attempts', 'http_requests', 'http_responses', 'statuses']}
            first_windows.append(row)
    spec = importlib.util.spec_from_file_location('existing_matcher', WT / 'results-wt/kd8/tools/extract_live.py')
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)
    diagnostics = [(name, json.loads((WT / 'results-wt/retrieval-diagnosis' / name).read_text()))
                   for name in ['source-plan-results.json', 'primary-plan-results.json']]
    registered = registered_evidence(saved, audit, diagnostics, matcher)
    write('QUERY-CONTEXTS.json', query_contexts)
    write('HISTORICAL-WINDOWS.json', windows)
    write('FIRST-WINDOWS.json', first_windows)
    write('ORDERING.json', ordering)
    write('REGISTERED-TRACE.json', registered)
    groups = defaultdict(list)
    for row in windows:
        groups[(row['source'], row['round'], row['field'] or 'untagged')].append(row)
    group_rows = []
    for (source, round_, field), rows in sorted(groups.items()):
        raw_rows = [r for r in rows if 'raw_path' in r]
        group_rows.append({'source': source, 'round': round_, 'field': field, 'contexts': len(rows),
                           'availability': dict(Counter(r['availability'] for r in rows)),
                           'pool_classes': dict(Counter(r['pool_class'] for r in rows)),
                           'returned_positions': sum(r['expected_count'] for r in rows),
                           'raw_positions': sum(r['raw_count'] for r in raw_rows),
                           'parser_drops': sum(r['parser_drop_count'] for r in raw_rows),
                           'query_ascii_token_count_min': min(len(tokens(r['query'])) for r in rows),
                           'query_ascii_token_count_max': max(len(tokens(r['query'])) for r in rows)})
    metric_rows = [m for r in windows for m in r.get('returned_metadata_metrics', [])]
    counts = {'historical_query_contexts': len(query_contexts), 'historical_source_contexts': len(windows),
              'historical_unique_raw_files_used': len({r['raw_path'] for r in windows if 'raw_path' in r}),
              'historical_raw_contexts': sum('raw_path' in r for r in windows),
              'historical_availability': dict(Counter(r['availability'] for r in windows)),
              'historical_pool_classes': dict(Counter(r['pool_class'] for r in windows)),
              'historical_returned_positions': sum(r['expected_count'] for r in windows),
              'historical_raw_positions': sum(r['raw_count'] for r in windows if 'raw_path' in r),
              'historical_parser_drops': sum(r['parser_drop_count'] for r in windows if 'raw_path' in r),
              'historical_returned_metadata': {'records': len(metric_rows), 'abstract_present': sum(m['abstract_present'] for m in metric_rows),
                                               'all_literal_tokens_in_title_abstract': sum(m['all_literal_tokens_in_title_abstract'] for m in metric_rows),
                                               'any_literal_token_in_title_abstract': sum(m['any_literal_token_in_title_abstract'] for m in metric_rows)},
              'fresh_pool_classes': dict(Counter(r['pool_class'] for r in first_windows)),
              'fresh_positions': sum(r['raw_count'] for r in first_windows),
              'fresh_parser_drops': sum(r['parser_drop_count'] for r in first_windows),
              'fresh_raw_files': len({r['raw_path'] for r in first_windows if 'raw_path' in r}),
              'registered_trace': dict(Counter(r['earliest_explicit_trace'] for r in registered)),
              'registered_metadata_available': sum(r['metadata_available'] for r in registered)}
    write('SUMMARY.json', {'diagnostic': True, 'acceptance_input': False, 'counts': counts, 'groups': group_rows,
                          'acceptance_unchanged': saved['acceptance'], 'pm2': 'Original PM2 failed; not rerun',
                          'historical_initial_429_cause': 'unknown', 'plan_sha256': sha(OUT / 'PLAN.md'),
                          'source_semantics_evidence': ['hybrid/literature/sources/arxiv.py', 'hybrid/literature/sources/arxiv_local.py',
                                                        'hybrid/literature/sources/openalex.py', 'hybrid/literature/sources/semantic_scholar.py'],
                          'external_query_docs': 'Not fetched. Provider-internal boolean, field scope and default ranking remain unknown.'})
    original_after = {p: sha(Path(p)) for p in manifest['original_sha256']}
    supplemental_after = {p: sha(Path(p)) for p in manifest['supplemental_sha256']}
    assert original_after == manifest['original_sha256']
    assert supplemental_after == manifest['supplemental_sha256']
    write('INPUTS-after.json', {'original_sha256': original_after, 'supplemental_sha256': supplemental_after,
                               'original_count': len(original_after), 'all_original_unchanged': True,
                               'all_supplemental_unchanged': True, 'raw_copies_made': 0, 'network_allowed': False})
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
