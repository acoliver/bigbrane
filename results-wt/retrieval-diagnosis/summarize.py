"""Render query-level evidence and request counts from diagnostic artifacts only."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    data = json.loads((OUT / 'offline.json').read_text())
    lines = ['# Original query and exposure trace', '',
             'All query text below is saved proposal/refinement output. No targeted diagnostic query is included.',
             'Counts are returned records, not recall estimates. A zero with no error is an answered empty search.',
             'Raw per-source identities, parsed metadata, source errors and cache hashes are in offline.json.', '']
    metrics = {'cards': {}, 'source_diagnostics': {}, 'historical_http_attempts': None}
    for cid, card in data['cards'].items():
        lines += [f'## {cid}', '',
                  f"Saved pool {card['original_candidates']}; current replay {card['new_candidates']}. "
                  f"{card['query_count']} queries, {len(card['fields'])} fields, {len(card['related'])} seeds.", '',
                  '| Query | Round / field | local / arXiv / OpenAlex / S2 | Exact registered raw hits | Errors |',
                  '|---|---|---|---|---|']
        for search in card['searches']:
            counts = ' / '.join(str(len(search['per_source_ids'][s]))
                                for s in ['arxiv_local', 'arxiv', 'openalex', 's2'])
            hits = '; '.join(source + ': ' + label for source, labels in search['exact_raw_matches'].items()
                             for label, values in labels.items() if values) or 'none'
            errors = '; '.join(f'{s}: {v}' for s, v in search['errors'].items()) or 'none'
            lines.append(f"| {search['query']} | {search['round']} / {search['field'] or 'untagged refinement'} | {counts} | {hits} | {errors} |")
        lines += ['', '### Saved citation/reference expansion', '',
                  '| Seed | Provider | References returned / used | Citations returned / used | Exact registered works anywhere in returned edges |',
                  '|---|---|---|---|---|']
        for related in card['related']:
            matches = '; '.join(label for label in card['matches']
                                if related['all_exact_references'][label] or related['all_exact_citations'][label]) or 'none'
            lines.append(f"| {related['paper']} | {related['via']} | {related['reference_count']} / {min(6, related['reference_count'])} | {related['citation_count']} / {min(6, related['citation_count'])} | {matches} |")
        lines += ['', '### Titles actually shown to refinement', '']
        lines.extend(f'{n}. {title.removeprefix("- ")}' for n, title in enumerate(card['actual_refinement_titles'], 1))
        lines += ['', '### Registered-work trace', '',
                  '| Work | Local exact record | Any-term eligibility across saved queries | Old rank / new rank | Actual deciding listing count | Refinement title positions |',
                  '|---|---|---|---|---|---|']
        local = {}
        for label, work in card['matches'].items():
            item = card['local_index'][label]
            eligibility = sum(any(q['target_membership']) for q in item.get('queries', []))
            local[label] = {'indexed': item['indexed'], 'any_term_queries': eligibility,
                            'all_term_queries': sum(q['target_membership'][0] for q in item.get('queries', [])),
                            'fallback_reached_queries': sum(q['any_term_fallback_reached'] for q in item.get('queries', []))}
            old = ','.join(str(v['rank']) for v in work['old']) or 'missing'
            new = ','.join(str(v['rank']) for v in work['new']) or 'missing'
            exposures = sum(len(v['exposures']) for v in work['old'])
            lines.append(f"| {label} | {item['indexed']} | {eligibility if item['indexed'] else 'not applicable'} / {card['query_count']} | {old} / {new} | {exposures} | {work['refinement_title_positions'] or 'none'} |")
        lines.append('')
        metrics['cards'][cid] = {'queries': card['query_count'], 'fields': len(card['fields']),
            'source_receipts': len(card['searches']) * 4, 'errors': card['source_errors'], 'local': local,
            'returned_reference_edges': sum(r['reference_count'] for r in card['related']),
            'returned_citation_edges': sum(r['citation_count'] for r in card['related'])}
    (OUT / 'QUERIES.md').write_text('\n'.join(lines))
    plans = [*json.loads((OUT / 'source-plan-results.json').read_text()),
             *json.loads((OUT / 'primary-plan-results.json').read_text())]
    exchanges = [json.loads(line) for line in (OUT / 'http.jsonl').read_text().splitlines()]
    metrics['source_diagnostics']['operations'] = len(plans)
    metrics['source_diagnostics']['exchanges'] = len(exchanges)
    metrics['source_diagnostics']['status_counts'] = dict(Counter(e['status'] for e in exchanges))
    metrics['source_diagnostics']['cooldown_skips'] = sum('cooling down' in str(e.get('error') or e.get('receipt', {}).get('errors', {})) for e in plans)
    metrics['source_diagnostics']['query_results'] = [
        {'label': e['label'], 'query': e['query'], 'source': e['source'],
         'returned': len(e.get('papers', [])), 'errors': e.get('error') or e.get('receipt', {}).get('errors', {})}
        for e in plans if e['kind'] == 'search']
    arxiv_starts = []
    for e in exchanges:
        if e['url'] == 'http://export.arxiv.org/api/query':
            arxiv_starts.append(datetime.fromisoformat(e['ts']).timestamp() - e['elapsed_s'])
    metrics['source_diagnostics']['minimum_arxiv_paced_attempt_gap_s'] = min(
        b - a for a, b in zip(arxiv_starts, arxiv_starts[1:]))
    postfix = [json.loads(line) for line in (OUT / 'postfix-http.jsonl').read_text().splitlines()]
    metrics['postfix_validation'] = {'exchanges': len(postfix), 'status_counts': dict(Counter(e['status'] for e in postfix)),
                                    'arxiv_exchanges': sum('export.arxiv.org' in e['url'] for e in postfix),
                                    'arxiv_redirects': sum('export.arxiv.org' in e['url'] and e['status'] in [301, 302, 307, 308] for e in postfix)}
    (OUT / 'MEASUREMENTS.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
