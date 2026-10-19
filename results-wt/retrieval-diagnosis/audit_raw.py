"""Check cached raw records and pre-100-reference IDs for parser/expansion losses.

Live diagnostic identities are used only to audit saved raw records, never to rank or add candidates.
"""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from hybrid.literature.paper import normalize_title
from hybrid.literature.sources.openalex import _arxiv_id_from

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('diag_raw', OUT / 'diagnose.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def raw_records(source, payload):
    if source == 'openalex':
        return payload['results'] if 'results' in payload else [payload]
    if 'data' not in payload:
        return [payload] if 'paperId' in payload or 'title' in payload else []
    return [entry.get('citedPaper') or entry.get('citingPaper') or entry for entry in payload['data']]


def main():
    saved = json.loads((OUT / 'offline.json').read_text())
    works = {name: (work['registered_arxiv_id'], work['registered_title'])
             for card in saved['cards'].values() for name, work in card['matches'].items()}
    extractor = diag.load_extractor(Path('/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2'))
    known_ids = {name: set() for name in works}
    diagnostics = [*json.loads((OUT / 'source-plan-results.json').read_text()),
                   *json.loads((OUT / 'primary-plan-results.json').read_text())]
    for operation in diagnostics:
        for paper in operation.get('papers', []):
            for name, work in works.items():
                if extractor.exact_match(paper, work):
                    known_ids[name].add(paper['source'] + ':' + paper['source_id'])
    original = {}
    initial = {}
    for cid, card in saved['cards'].items():
        papers, hits = {}, Counter()
        for query in card['searches']:
            if query['round'] != 1:
                continue
            records = [diag.Paper(**item) for source in query['sources']
                       for item in query['source_records'][source]]
            for paper in diag.dedupe(records):
                key = diag.paper_key_of(paper)
                papers.setdefault(key, paper)
                hits[key] += 1
        initial[cid] = [dict(diag.record(papers[key]), key=key)
                        for key in sorted(papers, key=lambda key: -hits[key])]
        assert [item['key'] for item in initial[cid][:5]] == card['refinement']['seeds']

        ledger = Path('/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/kd8/runs') / card['run'] / 'ledger.jsonl'
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        original[cid] = next(e['data']['candidates'] for e in events
                             if e['type'] == 'prior_art_result' and e['data']['card_id'] == cid)
    output = {'diagnostic_identity_usage': 'raw saved-record audit only; excluded from ranking and acceptance',
              'known_diagnostic_ids': {k: sorted(v) for k, v in known_ids.items()}, 'records': [],
              'pre_reference_cap_matches': [], 'missing_title_or_id_counts': {}}
    for filename in saved['inputs_sha256']:
        path = Path(filename)
        if '/hybrid/literature/cache/' not in filename:
            continue
        envelope = json.loads(path.read_text())
        source, body = envelope['source'], envelope['body']
        if source == 'arxiv':
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            items = []
            for entry in ET.fromstring(body).findall('a:entry', ns):
                sid = entry.findtext('a:id', '', ns).split('/abs/')[-1]
                items.append({'source': source, 'source_id': sid,
                              'title': entry.findtext('a:title', '', ns), 'url': '', 'arxiv_id': sid})
        else:
            payload = json.loads(body)
            items = []
            for entry in raw_records(source, payload):
                if source == 's2':
                    if not any(k in entry for k in ['paperId', 'externalIds', 'title']):
                        continue
                    item = {'source': source, 'source_id': entry.get('paperId', ''),
                            'title': entry.get('title') or '', 'url': entry.get('url') or '',
                            'arxiv_id': (entry.get('externalIds') or {}).get('ArXiv', '')}
                else:
                    sid = str(entry.get('id') or '').rsplit('/', 1)[-1]
                    if not sid.startswith('W') and not any(k in entry for k in ['title', 'referenced_works']):
                        continue
                    location = entry.get('primary_location') or {}
                    doi = (entry.get('doi') or '').removeprefix('https://doi.org/')
                    item = {'source': source, 'source_id': sid, 'title': entry.get('title') or entry.get('display_name') or '',
                            'url': location.get('landing_page_url') or '',
                            'arxiv_id': _arxiv_id_from(doi, location.get('landing_page_url')) or ''}
                    for n, rid in enumerate(entry.get('referenced_works') or [], 1):
                        key = 'openalex:' + str(rid).rsplit('/', 1)[-1]
                        names = [name for name, ids in known_ids.items() if key in ids]
                        if names:
                            title = entry.get('title') or entry.get('display_name') or ''
                            contexts = []
                            for cid, card in saved['cards'].items():
                                old = [(n, c['key']) for n, c in enumerate(original[cid], 1)
                                       if c['key'] == 'openalex:' + sid or normalize_title(c['title']) == normalize_title(title)]
                                new = [(c['rank'], c['key']) for c in card['current_records']
                                       if 'openalex:' + sid in c['aliases'] or normalize_title(c['title']) == normalize_title(title)]
                                queries = [q['query'] for q in card['searches'] if sid in q['per_source_ids']['openalex']]
                                initial_ranks = [(n, c['key']) for n, c in enumerate(initial[cid], 1)
                                                 if c['key'] == 'openalex:' + sid or normalize_title(c['title']) == normalize_title(title)]
                                seeds = [key for key in card['refinement']['seeds'] if any(k == key for _, k in old)]
                                if old or queries:
                                    contexts.append({'card_id': cid, 'queries': queries, 'old_candidates': old,
                                                     'current_candidates': new, 'initial_rank_at_seed_selection': initial_ranks,
                                                     'selected_seed_keys': seeds})
                            output['pre_reference_cap_matches'].append({'cache': filename, 'source_work': sid,
                                'source_title': title, 'request_params': envelope['params'], 'contexts': contexts,
                                'reference_position': n, 'within_100_cap': n <= 100, 'known_work_matches': names})
                items.append(item)
        missing = Counter()
        for item in items:
            if not item['title']:
                missing['title'] += 1
            if not item['source_id']:
                missing['id'] += 1
            names = [name for name, work in works.items() if extractor.exact_match(item, work)
                     or item['source'] + ':' + item['source_id'] in known_ids[name]]
            if names:
                output['records'].append({'cache': filename, 'request_params': envelope['params'],
                                          'record': item, 'matches': names})
        if missing:
            output['missing_title_or_id_counts'][filename] = dict(missing)
    (OUT / 'raw-audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print('exact raw matching records', len(output['records']))
    print('known pre-reference-cap matches', len(output['pre_reference_cap_matches']))
    print('envelopes with missing title/id', len(output['missing_title_or_id_counts']))
    for entry in output['pre_reference_cap_matches']:
        print(entry)


if __name__ == '__main__':
    main()
