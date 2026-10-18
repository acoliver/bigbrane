"""Package measured audit metadata without duplicating raw bodies or input histories."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def check(data):
    rows = data['historical_windows']
    pools = Counter(r['pool_class'] for r in rows)
    availability = Counter(r['availability'] for r in rows)
    assert len(data['queries']) == 72 and len(rows) == 288
    assert pools == {'explicit_zero_pool': 37, 'exhausted_nonempty_pool': 11,
                     'more_unseen_records': 122, 'unobserved_error': 46,
                     'local_pool_not_recomputed': 72}
    assert availability['cooldown_skip'] == 44 and availability['exhausted_http'] == 2
    answered = [r for r in rows if 'raw_sha256' in r]
    assert len(answered) == 170
    assert sum(r['raw_count'] for r in answered) == 761
    assert all(r['parsed_count'] == r['raw_count'] == len(r['result_ids'])
               and r['parser_drop_count'] == 0 for r in answered)
    queries = {(r['card_id'], r['query_sha256']): r for r in data['queries']}
    for row in rows + data['first_windows']:
        assert (row['card_id'], row['query_sha256']) in queries
    for row in data['queries']:
        assert hashlib.sha256(row['query'].encode()).hexdigest() == row['query_sha256']
    order = data['ordering']
    assert sum(r['initial_candidates'] for r in order) == 848
    assert sum(r['initial_single_query_tie_count'] for r in order) == 845
    assert sum(r['selected_seeds_present_in_first_query'] for r in order) == 18
    first = data['first_windows']
    assert len(first) == 12 and sum(r['raw_count'] for r in first) == 50
    expected = {'arxiv': [(0, 0), (0, 0), (2, 2), (0, 0)],
                'openalex': [(6, 839), (6, 1973), (6, 4703), (6, 160)],
                's2': [(6, 8), (6, 16505), (6, 21761), (6, 49)]}
    for source, values in expected.items():
        assert [(r['raw_count'], r['total']) for r in first if r['source'] == source] == values


def main():
    target = ROOT / 'DELIVERY-RECEIPTS.json'
    if sys.argv[1:] == ['--check']:
        check(read(target.name))
        print('Compact receipts: 288 contexts, 170 answered remote, 761 records, zero parser drops')
        return
    assert not sys.argv[1:]
    names = ['QUERY-CONTEXTS.json', 'HISTORICAL-WINDOWS.json', 'FIRST-WINDOWS.json',
             'ORDERING.json', 'REGISTERED-TRACE.json', 'INPUTS-before.json', 'INPUTS-after.json',
             'VALIDATION.json', 'ARTIFACTS.json', 'PLAN.md', 'PLAN.sha256']
    def window(row):
        return {k: v for k, v in row.items() if k not in
                {'query', 'request', 'returned_metadata_metrics', 'server_search_queries'}}
    data = {'scope': 'Measured metadata projection. Raw bodies and complete derivations remain local. '
                     'Counts can be checked here; provider bodies cannot be regenerated from hashes.',
            'local_source_files': {name: {'sha256': sha(ROOT / name), 'bytes': (ROOT / name).stat().st_size}
                                   for name in names},
            'queries': [{k: v for k, v in row.items() if k != 'normalization'}
                        for row in read('QUERY-CONTEXTS.json')],
            'historical_windows': [window(row) for row in read('HISTORICAL-WINDOWS.json')],
            'first_windows': [window(row) for row in read('FIRST-WINDOWS.json')],
            'ordering': [{k: v for k, v in row.items() if k != 'all_initial_ranked'}
                         for row in read('ORDERING.json')],
            'registered_trace': [{k: v for k, v in row.items() if k != 'literal_metadata_comparisons'}
                                 for row in read('REGISTERED-TRACE.json')]}
    check(data)
    target.write_text(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n')
    print(target.name, target.stat().st_size, sha(target))


if __name__ == '__main__':
    main()
