"""Offline analysis of the completed diagnostic only; no network path."""
from collections import Counter
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess

from measure import OUT, WT, MAIN, digest, implementation_hashes, preservation, write


def main():
    plan = json.loads((OUT / 'PLAN.json').read_text())
    results = json.loads((OUT / 'RESULTS.json').read_text())
    summary = json.loads((OUT / 'SUMMARY.json').read_text())
    fixtures = {x['expected_callback_card_id']: x for x in json.loads((OUT / 'FIXTURES.json').read_text())}
    assert digest(OUT / 'FIXTURES.json') == plan['fixture_sha256']
    current_hashes = implementation_hashes()
    changed = {path: {'before': sha, 'after': current_hashes[path],
                      'mtime': datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone().isoformat()}
               for path, sha in plan['implementation_sha256'].items() if current_hashes[path] != sha}
    finished = datetime.fromisoformat(json.loads((OUT / 'RUN-END.json').read_text())['finished_at'])
    assert all(Path(path).name in {'seed.py', 'build_seed.py'} and datetime.fromisoformat(item['mtime']) > finished
               for path, item in changed.items()), 'Unexpected runtime implementation change'
    write(OUT / 'RUNTIME-HASH-CHANGES.json', {'changes': changed,
          'scope': 'Other-worker seed changes with file mtimes after the completed measurement.',
          'all_transport_client_source_sink_ledger_hashes_unchanged': True})
    assert all(digest(OUT / name) == sha for name, sha in plan['tool_sha256'].items())
    assert digest(OUT / 'PLAN.json') == (OUT / 'PLAN.sha256').read_text().strip()
    matcher_path = WT / 'results-wt/kd8/tools/extract_live.py'
    spec = importlib.util.spec_from_file_location('existing_exact_matcher', matcher_path)
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)
    cards = []
    for entry in plan['cards']:
        cid = entry['card_id']
        old = fixtures[cid]['original_receipt']['data']
        own = [r for r in results if r['card_id'] == cid]
        rows = [r for r in summary['rows'] if r['card_id'] == cid]
        registered = matcher.REFEREE[cid]
        source_rows = {}
        for r in own:
            source = r['source']
            hits = {name: [{'source_id': p['source_id'], 'title': p['title'],
                            'match_basis': matcher.exact_match(p, work)}
                           for p in r['papers'] if matcher.exact_match(p, work)]
                    for name, work in registered.items()}
            fresh_ids = [p['source_id'] for p in r['papers']]
            source_rows[source] = {'original_count': len(old['per_source_ids'][source]),
                                   'original_error': old['errors'].get(source),
                                   'fresh_count': len(fresh_ids), 'fresh_errors': r['errors'],
                                   'fresh_registered_matches': hits,
                                   'same_returned_id_order': fresh_ids == old['per_source_ids'][source]}
        cards.append({'card_id': cid, 'query': entry['query'], 'query_sha256': entry['query_sha256'],
                      'registered_work_count': len(registered),
                      'fresh_registered_match_count': sum(any(hits for s in source_rows.values()
                                                            for name2, hits in s['fresh_registered_matches'].items() if name2 == name)
                                                         for name in registered),
                      'historical_local_only': {'count': len(old['per_source_ids']['arxiv_local']),
                                                'ids': old['per_source_ids']['arxiv_local'], 'fresh_local_measured': False},
                      'source_outcomes': source_rows,
                      'transport': {name: sum(r[name] for r in rows) for name in ('source_receipts', 'attempts', 'http_requests', 'http_responses')},
                      'statuses': dict(Counter(str(status) for r in rows for status in r['statuses']))})
    events = [json.loads(line) for line in (OUT / 'runtime/ledger.jsonl').read_text().splitlines()]
    attempts_by_source = {s: sorted([a for e in events if e['type'] == 'literature_transport' and e['data']['source'] == s
                                  for a in e['data']['attempts']], key=lambda a: a['started_s']) for s in plan['sources']}
    pacing = {}
    for source, attempts in attempts_by_source.items():
        gaps = [b['started_s'] - a['started_s'] for a, b in zip(attempts, attempts[1:])]
        pacing[source] = {'attempts': len(attempts), 'observed_start_gaps_s': gaps,
                          'minimum_observed_gap_s': min(gaps) if gaps else None,
                          'configured_interval_s': plan['min_interval_s'][source],
                          'all_own_gaps_meet_interval': all(g >= plan['min_interval_s'][source] for g in gaps)}
        assert pacing[source]['all_own_gaps_meet_interval']
    before = json.loads((OUT / 'PRESERVATION-before.json').read_text())['saved_inputs_sha256']
    after = preservation()
    assert before == after
    write(OUT / 'PRESERVATION-after.json', {'at': datetime.now().astimezone().isoformat(), 'count': len(after),
                                          'saved_inputs_sha256': after,                                           'same_as_before': True,
                                          'pinned_transport_implementation_unchanged': True,
                                          'other_worker_seed_changes': changed, 'pinned_tools_unchanged': True})
    write(OUT / 'COVERAGE.json', {'diagnostic': True, 'acceptance_input': False,
                                'matcher_path': str(matcher_path), 'matcher_sha256': digest(matcher_path),
                                'matcher_policy': 'Existing exact arXiv ID or normalized full-title predicate, diagnostic only.',
                                'cards': cards, 'pacing': pacing,
                                'registered_matches': sum(c['fresh_registered_match_count'] for c in cards),
                                'registered_works': sum(c['registered_work_count'] for c in cards),
                                'original_pa2': {'status': 'FAIL', 'retrieved_cards': 2, 'cards': 4, 'required': 3}})
    start = json.loads((OUT / 'RUN-ONCE.json').read_text())['started_at']
    end = json.loads((OUT / 'RUN-END.json').read_text())['finished_at']
    write(OUT / 'VALIDATION.json', {'plan_sha256': digest(OUT / 'PLAN.json'), 'fixture_sha256': digest(OUT / 'FIXTURES.json'),
                                  'all_274_original_hashes_unchanged': True, 'pinned_transport_runtime_hashes_unchanged': True,
                                  'other_worker_seed_changes': changed, 'pinned_tool_hashes_unchanged': True,
                                  'runtime_started_at': start, 'runtime_finished_at': end,
                                  'runtime_elapsed_s': (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(),
                                  'main_git_status': subprocess.check_output(['git', '-C', str(MAIN), 'status', '--porcelain'], text=True),
                                  'sibling_git_status_other_workers': subprocess.check_output(['git', '-C', str(WT), 'status', '--porcelain'], text=True)})
    print(json.dumps({'registered_matches': sum(c['fresh_registered_match_count'] for c in cards),
                      'cards': [{k: c[k] for k in ('card_id', 'fresh_registered_match_count', 'transport', 'statuses')} for c in cards],
                      'pacing': pacing}, indent=2))


if __name__ == '__main__':
    main()
