"""B4 diagnostics. Saved replay is offline; live mode makes bounded source-only calls.

Run with the repaired worktree as cwd/PYTHONPATH and main's .venv-hybrid Python.
Targeted requests are diagnostic evidence, never PA2 acceptance inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import sqlite3
import time

import httpx

from hybrid.literature.cache import ResponseCache, request_key
from hybrid.literature.client import Literature, PROCESS_PACER
from hybrid.literature.keys import KEY_SPECS, load_api_keys
from hybrid.literature.paper import Paper, dedupe, paper_key_of, normalize_title
from hybrid.literature.prior_art import _Pool, _candidate_record, select_candidates
from hybrid.literature import sources
from hybrid.literature.sources.arxiv_local import match_expressions
from hybrid.literature.transport import Fetched, SourceError

OUT = Path(__file__).resolve().parent


def load_extractor(wt):
    path = wt / 'results-wt/kd8/tools/extract_live.py'
    spec = importlib.util.spec_from_file_location('saved_extractor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(paper):
    return asdict(paper)


def candidate_paper(item):
    return Paper(source=item['source'], source_id=item['source_id'], title=item['title'],
                 authors=(), year=item.get('year'), abstract=item.get('excerpt'), url=item['url'],
                 venue=None, citation_count=None, fetched_at='', arxiv_id=item.get('arxiv_id'))


class SavedLiterature(Literature):
    """Read existing cache envelopes only. A miss fails; no HTTP or cache writes."""
    def __init__(self, wt):
        self.cache = ResponseCache(wt / 'hybrid/literature/cache')
        self._extra_params = {'openalex': {}}
        self.inputs = {}

    def _fetch(self, source, request):
        # Historical receipts used HTTP cache keys; replay must read those exact inputs.
        if source == 'arxiv':
            request = replace(request, url='http://export.arxiv.org/api/query')
        path = self.cache.path_for(source, request_key(source, request.url, request.params))
        data = self.cache.get(source, path.stem)
        if data is None:
            raise FileNotFoundError(path)
        self.inputs[str(path)] = digest(path)
        return Fetched(source, request.url, request.params, data['body'], data['fetched_at'], True)


def reconstruct_search(client, search, db, receipt_finished_at):
    per_source = {}
    for source in search['sources']:
        ids = search['per_source_ids'][source]
        if source in search['errors']:
            assert ids == []
            per_source[source] = []
        elif source == 'arxiv_local':
            papers = []
            for sid in ids:
                row = db.execute('SELECT * FROM papers WHERE arxiv_id=?', (sid,)).fetchone()
                if row is None:
                    raise ValueError(f'saved local id missing: {sid}')
                papers.append(Paper('arxiv', sid, row['title'], tuple(row['authors'].splitlines()),
                                    int(row['created'][:4]) if row['created'] else None,
                                    row['abstract'], f'https://arxiv.org/abs/{sid}', row['journal_ref'],
                                    None, '', sid, row['doi']))
            per_source[source] = papers
        else:
            request = sources.MODULES[source].search_request(search['query'], search['limit'])
            fetched = client._fetch(source, request)
            if datetime.fromisoformat(fetched.fetched_at) > datetime.fromisoformat(receipt_finished_at):
                raise ValueError(f'cache postdates saved search: {source} {search["query"]}')
            papers = sources.MODULES[source].parse_search(fetched.body, fetched.fetched_at)
            if [p.source_id for p in papers] != ids:
                raise ValueError(f'cache differs from saved receipt: {source} {search["query"]}')
            per_source[source] = papers
    papers = dedupe(p for source in search['sources'] for p in per_source[source])
    assert len(papers) == search['result_count']
    return papers, per_source


def matching(papers, works, extractor):
    return {label: [paper_key_of(p) for p in papers if extractor.exact_match(record(p), work)]
            for label, work in works.items()}


def fts_target_evidence(db, works, queries):
    """Test target membership in exact saved FTS expressions, without changing query terms."""
    result = {}
    for label, (sid, title) in works.items():
        if sid is None:
            rows = db.execute('SELECT arxiv_id,title FROM papers WHERE title = ? COLLATE NOCASE',
                              (title,)).fetchall()
            result[label] = {'indexed': bool(rows), 'exact_title_records': [dict(row) for row in rows],
                             'reason': 'No registered arXiv id; checked exact title in the local snapshot.'}
            continue
        row = db.execute('SELECT id,arxiv_id,title,abstract,categories,doi FROM papers WHERE arxiv_id=?', (sid,)).fetchone()
        if row is None:
            result[label] = {'indexed': False}
            continue
        details = []
        for query in queries:
            expressions = match_expressions(query)
            matches = [bool(db.execute('SELECT rowid FROM papers_fts WHERE papers_fts MATCH ? AND rowid=?',
                                       (expr, row['id'])).fetchone()) for expr in expressions]
            all_count_capped = len(db.execute(
                'SELECT rowid FROM papers_fts WHERE papers_fts MATCH ? LIMIT 6', (expressions[0],)).fetchall())
            missing_terms = [term for term in re.findall(r'"[^"]*"', expressions[0])
                             if not db.execute('SELECT rowid FROM papers_fts WHERE papers_fts MATCH ? AND rowid=?',
                                               ('{title abstract} : (' + term + ')', row['id'])).fetchone()]
            details.append({'query': query, 'expressions': expressions, 'target_membership': matches,
                            'missing_all_term_components': missing_terms,
                            'all_term_count_capped_at_6': all_count_capped,
                            'any_term_fallback_reached': len(expressions) > 1 and all_count_capped < 6})
        result[label] = {'indexed': True, 'metadata': dict(row), 'queries': details}
    return result


def offline(wt):
    extractor = load_extractor(wt)
    client = SavedLiterature(wt)
    dbpath = wt / 'data/arxiv-index/arxiv.sqlite'
    db = sqlite3.connect(f'file:{dbpath}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    output = {'scope': 'unchanged saved acquisition; current ranking/field selection only',
              'acceptance': {'retrieved_cards': 2, 'denominator': 4, 'required': 3, 'status': 'FAIL'},
              'cards': {}, 'runs': {}, 'inputs_sha256': {}, 'http_attempts': None}
    for run in ['flipA', 'flipB']:
        runpath = wt / 'results-wt/kd8/runs' / run
        ledger = runpath / 'ledger.jsonl'
        output['inputs_sha256'][str(ledger)] = digest(ledger)
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        searches = [e for e in events if e['type'] == 'literature_search']
        output['runs'][run] = {'searches': len(searches),
            'arxiv_errors': dict(Counter(e['data']['errors'].get('arxiv', 'success') for e in searches)),
            'timeline': [{'seq': e['seq'], 'event_ts': e['ts'], **e['data']} for e in searches]}
        calls = []
        for path in sorted((runpath / 'llm/calls').glob('*.json')):
            call = json.loads(path.read_text())
            calls.append((path, call))
        for cid in extractor.INCREMENTAL:
            own = [e for e in events if e['data'].get('card_id') == cid]
            if not own:
                continue
            works = extractor.REFEREE[cid]
            proposal = next(e for e in own if e['type'] == 'prior_art_queries')['data']
            refinement = next(e for e in own if e['type'] == 'prior_art_round2_queries')['data']
            original = next(e for e in own if e['type'] == 'prior_art_result')['data']
            fields = dict(zip(proposal['queries'], proposal['fields']))
            pool = _Pool()
            historical_papers = {}
            hits = Counter()
            query_rows, related_rows = [], []
            for event in own:
                data = event['data']
                if event['type'] == 'literature_search':
                    papers, per_source = reconstruct_search(client, data, db, event['ts'])
                    for p in papers:
                        historical_papers.setdefault(paper_key_of(p), p)
                        hits[paper_key_of(p)] += 1
                    pool.add_all(papers, query=data['query'], field=fields.get(data['query']))
                    query_rows.append({'seq': event['seq'], 'event_ts': event['ts'], **data,
                        'round': 1 if data['query'] in fields else 2, 'field': fields.get(data['query']),
                        'source_records': {s: [record(p) for p in ps] for s, ps in per_source.items()},
                        'exact_raw_matches': {s: matching(ps, works, extractor) for s, ps in per_source.items()},
                        'exact_dedup_matches': matching(papers, works, extractor)})
                elif event['type'] == 'literature_related':
                    seed = historical_papers[data['paper']]
                    refs, cites = getattr(client, '_related_' + data['via'])(seed)
                    assert [p.source_id for p in refs] == data['reference_ids']
                    assert [p.source_id for p in cites] == data['citation_ids']
                    used = (*refs[:6], *cites[:6])
                    for p in used:
                        historical_papers.setdefault(paper_key_of(p), p)
                        hits[paper_key_of(p)] += 1
                    pool.add_all(used)
                    related_rows.append({'seq': event['seq'], **data,
                        'all_exact_references': matching(refs, works, extractor),
                        'all_exact_citations': matching(cites, works, extractor),
                        'used_exact_matches': matching(used, works, extractor),
                        'used_keys': [paper_key_of(p) for p in used],
                        'reference_count': len(refs), 'citation_count': len(cites)})
            old_keys = list(historical_papers)
            old_keys.sort(key=lambda key: -hits[key])
            assert old_keys == [c['key'] for c in original['candidates']], cid
            # Retain original stored metadata. Cache reconstruction supplies only acquisition provenance.
            original_by_key = {c['key']: c for c in original['candidates']}
            for key in list(pool.papers):
                stored = original_by_key[key]
                pool.papers[key] = candidate_paper(stored)
            records = [dict(_candidate_record(key, p), **pool.provenance(key), rank=n)
                       for n, (key, p) in enumerate(pool.ranked().items(), 1)]
            own_calls = []
            for path, call in calls:
                if extractor.prompt_card(call['user']) == cid:
                    output['inputs_sha256'][str(path)] = digest(path)
                    own_calls.append(call)
            listings = [value for call in own_calls if (value := extractor.exposure(call)) is not None]
            prior_calls = sorted((c for c in own_calls if c['role'] == 'prior_art'), key=lambda c: c['started_at'])
            proposed = prior_calls[0]['parsed']
            refined = prior_calls[1]['parsed']
            assert [q['query'] for q in proposed['queries']] == proposal['queries']
            assert refined['queries'] == refinement['queries']
            refine_titles = prior_calls[1]['user'].split('TITLES FOUND SO FAR\n', 1)[1].splitlines()
            matches = {}
            for label, work in works.items():
                old = [dict(c, rank=n) for n, c in enumerate(original['candidates'], 1) if extractor.exact_match(c, work)]
                new = [c for c in records if extractor.exact_match(c, work)]
                matches[label] = {'registered_arxiv_id': work[0], 'registered_title': work[1],
                    'old': [{'key': c['key'], 'rank': c['rank'], 'exposures': [
                        {'kind': l['kind'], 'call_id': l['call_id'], 'position': l['keys'].index(c['key']) + 1}
                        for l in listings if c['key'] in l['keys']]} for c in old],
                    'all_prompt_mentions': [
                        {'call_id': call['call_id'], 'role': call['role'],
                         'arxiv_id_mentioned': bool(work[0] and work[0] in call['user']),
                         'full_title_mentioned': normalize_title(work[1]) in normalize_title(call['user'])}
                        for call in own_calls if (work[0] and work[0] in call['user'])
                        or normalize_title(work[1]) in normalize_title(call['user'])],
                    'refinement_title_positions': [n for n, title in enumerate(refine_titles, 1)
                                                  if normalize_title(work[1]) in normalize_title(title)],
                    'new': [{'key': c['key'], 'rank': c['rank'], 'query_support': c['query_support'],
                        'query_fields': c['query_fields'], 'aliases': c['aliases'],
                        'positions': {str(limit): next((n for n, item in enumerate(select_candidates(records, limit), 1)
                                                      if item['key'] == c['key']), None) for limit in [30, 35, 60, 80]}}
                        for c in new]}
            output['cards'][cid] = {'run': run, 'proposal': proposal, 'refinement': refinement,
                'refinement_call_id': prior_calls[1]['call_id'], 'actual_refinement_titles': refine_titles,
                'classifier_citations': original['evidence'], 'rejected_evidence': original['rejected_evidence'],
                'searches': query_rows, 'related': related_rows, 'matches': matches,
                'original_candidates': len(original['candidates']), 'new_candidates': len(records),
                'historical_rank_reproduced': True, 'query_count': len(query_rows),
                'fields': sorted(set(proposal['fields'])),
                'source_errors': dict(Counter(s for q in query_rows for s in q['errors'])),
                'local_index': fts_target_evidence(db, works, [q['query'] for q in query_rows]),
                'current_records': records}
    db.close()
    output['inputs_sha256'].update(client.inputs)
    write_json(OUT / 'offline.json', output)
    print(json.dumps({cid: {'candidates': (v['original_candidates'], v['new_candidates']),
                           'matches': v['matches']} for cid, v in output['cards'].items()}, indent=2))


class Recorder:
    """HTTP event hooks retain request/status facts, never auth headers or response bodies."""
    def __init__(self, path, keys):
        self.path = path
        self.keys = keys
        self.label = None

    def response(self, response):
        request = response.request
        entry = {'diagnostic': True, 'acceptance_input': False, 'label': self.label,
                 'ts': datetime.now(timezone.utc).isoformat(), 'method': request.method,
                 'url': str(request.url.copy_with(query=None)), 'params': dict(request.url.params),
                 'status': response.status_code,
                 'retry_after': response.headers.get('retry-after'),
                 'location': response.headers.get('location'),
                 'elapsed_s': time.monotonic() - request.extensions['diagnostic_start']}
        text = json.dumps(entry)
        for key in self.keys.values():
            text = key.redact(text)
        with self.path.open('a') as handle:
            handle.write(text + '\n')

    def request(self, request):
        request.extensions['diagnostic_start'] = time.monotonic()


def live(wt, planpath):
    plan = json.loads(planpath.read_text())
    keys = load_api_keys()
    recorder = Recorder(OUT / 'http.jsonl', keys)
    http = httpx.Client(timeout=30, follow_redirects=True,
                        event_hooks={'request': [recorder.request], 'response': [recorder.response]})
    # max_retries=0 bounds diagnosis to one attempt, not a production retry policy change.
    client = Literature(cache_root=OUT / 'diagnostic-cache', http=http, api_keys=keys,
                        pacer=PROCESS_PACER, max_retries=0)
    results = []
    for item in plan:
        recorder.label = item['label']
        started = time.monotonic()
        entry = {**item, 'diagnostic': True, 'acceptance_input': False,
                 'key_configured': item['source'] in keys,
                 'key_env_name': KEY_SPECS[item['source']].env_var if item['source'] in KEY_SPECS else None}
        try:
            if item['kind'] == 'get':
                entry['papers'] = [record(client.get_paper(item['source'], item['id']))]
            else:
                receipt = client.search_with_receipt(item['query'], (item['source'],), 6)
                entry['receipt'] = receipt.event_data()
                entry['papers'] = [record(p) for p in receipt.papers]
        except SourceError as exc:
            entry['error'] = exc.reason
        entry['elapsed_s'] = time.monotonic() - started
        results.append(entry)
        write_json(OUT / (planpath.stem + '-results.json'), results)
        print(item['label'], entry.get('error', entry.get('receipt', {}).get('errors', 'OK')), flush=True)
    http.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.plan:
        live(args.worktree, args.plan)
    else:
        offline(args.worktree)


if __name__ == '__main__':
    main()
