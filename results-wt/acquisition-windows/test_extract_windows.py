"""Scoped synthetic extraction tests. No credentials, model calls or network."""
import copy
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest

from extract_windows import (OUT, assert_attribution, block_network, generic_ordering, index_envelopes,
                             literal_coverage, normalized_query, project_error, public_request,
                             response_metadata, signature, window_row)

socket.socket.connect = block_network
socket.socket.connect_ex = block_network
socket.create_connection = block_network


def atom(total=None, start=None, size=None, entries='', namespace='http://a9.com/-/spec/opensearch/1.1/'):
    values = ''.join(f'<os:{key}>{value}</os:{key}>' for key, value in
                     [('totalResults', total), ('startIndex', start), ('itemsPerPage', size)] if value is not None)
    return f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:os="{namespace}">{values}{entries}</feed>'


ENTRY = '<entry><id>http://arxiv.org/abs/2107.04683v1</id><title>Permutation automata</title><summary>A group</summary></entry>'


def paper(sid, title=None):
    return {'source': 'arxiv', 'source_id': sid, 'title': title or sid, 'authors': [], 'year': 2020,
            'abstract': 'alpha beta', 'url': 'https://arxiv.org/abs/' + sid, 'venue': None,
            'citation_count': None, 'fetched_at': '', 'arxiv_id': sid, 'doi': None}


class WindowTests(unittest.TestCase):
    def test_atom_namespaces_and_zero(self):
        for namespace in ['http://a9.com/-/spec/opensearch/1.1/', 'http://a9.com/-/spec/opensearch/1.0/']:
            with self.subTest(namespace=namespace):
                d = response_metadata('arxiv', atom(0, 0, 6, namespace=namespace))
                self.assertEqual((d['total'], d['start'], d['items_per_page'], d['raw_count']), (0, 0, 6, 0))
                self.assertEqual(d['pool_class'], 'explicit_zero_pool')

    def test_atom_exhausted_and_more(self):
        exhausted = response_metadata('arxiv', atom(1, 0, 6, ENTRY))
        self.assertEqual(exhausted['pool_class'], 'exhausted_nonempty_pool')
        more = response_metadata('arxiv', atom(9, 0, 6, ENTRY))
        self.assertEqual(more['pool_class'], 'more_unseen_records')
        self.assertEqual(more['unseen_count_after_page'], 8)

    def test_missing_total_is_unknown_not_zero(self):
        examples = [('arxiv', atom(start=0, size=6)),
                    ('openalex', json.dumps({'meta': {'page': 1, 'per_page': 6}, 'results': []})),
                    ('s2', json.dumps({'offset': 0, 'data': []}))]
        for source, body in examples:
            with self.subTest(source=source):
                row = response_metadata(source, body)
                self.assertIsNone(row['total'])
                self.assertEqual(row['pool_class'], 'unknown_total_or_start')

    def test_wrong_total_field_not_silently_accepted(self):
        row = response_metadata('openalex', json.dumps({'total': 42, 'meta': {'page': 1, 'per_page': 6}, 'results': []}))
        self.assertIsNone(row['total'])
        row = response_metadata('s2', json.dumps({'count': 42, 'offset': 0, 'data': []}))
        self.assertIsNone(row['total'])
        row = response_metadata('arxiv', '<feed xmlns="http://www.w3.org/2005/Atom"><totalResults>0</totalResults></feed>')
        self.assertIsNone(row['total'])

    def test_openalex_count_page_and_exhausted_short_window(self):
        row = response_metadata('openalex', json.dumps({'meta': {'count': 8, 'page': 2, 'per_page': 6}, 'results': [{}, {}]}))
        self.assertEqual((row['total'], row['start'], row['raw_count']), (8, 6, 2))
        self.assertEqual(row['pool_class'], 'exhausted_nonempty_pool')
        row = response_metadata('openalex', json.dumps({'meta': {'count': 100, 'page': 1, 'per_page': 6}, 'results': [{}] * 6}))
        self.assertEqual(row['unseen_count_after_page'], 94)

    def test_s2_zero_without_data(self):
        row = response_metadata('s2', '{"total":0,"offset":0}')
        self.assertEqual((row['raw_count'], row['pool_class']), (0, 'explicit_zero_pool'))

    def test_s2_next_and_offset(self):
        row = response_metadata('s2', json.dumps({'total': 100, 'offset': 6, 'next': 12, 'data': [{}] * 6}))
        self.assertEqual(row['next'], 12)
        self.assertEqual(row['unseen_count_after_page'], 88)
        self.assertEqual(row['pool_class'], 'more_unseen_records')
        row = response_metadata('s2', json.dumps({'offset': 0, 'next': 6, 'data': [{}] * 6}))
        self.assertIsNone(row['total'])
        self.assertEqual(row['pool_class'], 'more_unseen_records')

    def test_exhausted_error_cooldown_redirect_are_not_zero_pools(self):
        for code in (301, 302, 429, 500):
            row = response_metadata('s2', 'not JSON and not a pool', code)
            self.assertEqual(row['pool_class'], 'non_pool_http_status')
            self.assertIsNone(row['total'])
        search = {'query': 'alpha', 'limit': 6, 'seq': 1, 'round': 1, 'field': 'optimization',
                  'sources': ['arxiv'], 'per_source_ids': {'arxiv': []}, 'errors': {}}
        for error in ['HTTP 429 after 4 attempts', 'skipped: source is cooling down after rate limiting']:
            search['errors'] = {'arxiv': error}
            row = window_row('card-A', search, 'arxiv', {}, True)
            self.assertEqual(row['pool_class'], 'unobserved_error')
            self.assertIsNone(row['total'])

    def test_api_error_is_not_pool(self):
        entry = '<entry><id>http://arxiv.org/api/errors#bad</id><summary>bad query</summary></entry>'
        self.assertEqual(response_metadata('arxiv', atom(0, 0, 6, entry))['pool_class'], 'api_error_not_pool')
        self.assertEqual(response_metadata('s2', '{"error":"private detail"}')['pool_class'], 'api_error_not_pool')

    def test_invalid_or_contradictory_counts_fail_fast(self):
        for total in [-1, True, 1.5]:
            with self.subTest(total=total), self.assertRaises(ValueError):
                response_metadata('s2', json.dumps({'total': total, 'offset': 0, 'data': []}))
        with self.assertRaises(ValueError):
            response_metadata('s2', '{"total":0,"offset":0,"data":[{}]}')
        with self.assertRaises(ValueError):
            response_metadata('openalex', '{"meta":{"count":0}}')

    def test_exact_request_normalization(self):
        query = 'in-context "alpha beta" from top-K'
        self.assertEqual(public_request('arxiv', query, 6)['params'],
                         {'search_query': 'all:in AND all:context AND all:alpha AND all:beta AND all:from AND all:top AND all:K',
                          'start': '0', 'max_results': '6', 'sortBy': 'relevance'})
        self.assertEqual(public_request('openalex', query, 6)['params']['search'], query)
        self.assertEqual(public_request('s2', query, 6)['params']['query'], query)
        normalized = normalized_query(query)
        self.assertIn('from', normalized['local_removed_tokens'])
        self.assertEqual(normalized['arxiv_ast']['operator'], 'AND')
        self.assertIn('"alpha beta"', normalized['local_expressions'][0])

    def test_literal_coverage_not_stemmed_or_fulltext(self):
        row = literal_coverage('learning memory from examples', {'title': 'Learns memories', 'abstract': 'Some examples'})
        self.assertEqual(row['missing_literal_tokens'], ['from', 'learning', 'memory'])
        self.assertFalse(row['all_literal_tokens_in_title_abstract'])
        self.assertTrue(row['any_literal_token_in_title_abstract'])

    def test_attribution_rejects_card_source_and_query_crossing(self):
        row = {'card_id': 'card-A', 'source': 's2', 'query': 'alpha'}
        assert_attribution(row, 'card-A', 's2', 'alpha')
        for args in [('card-B', 's2', 'alpha'), ('card-A', 'openalex', 'alpha'), ('card-A', 's2', 'beta')]:
            with self.subTest(args=args), self.assertRaises(AssertionError):
                assert_attribution(row, *args)

    def test_error_and_parameter_projection_drops_private_details(self):
        private = 'synthetic-private-marker'
        self.assertNotIn(private, json.dumps(project_error('Failure Authorization: ' + private)))
        projected = signature('s2', 'https://example.org/search', {'query': 'alpha', 'limit': '6', 'api_key': private, 'headers': private})
        self.assertNotIn(private, str(projected))
        self.assertEqual(project_error('HTTP 429 after 4 attempts'), {'kind': 'exhausted_http', 'status': 429, 'declared_attempts': 4})

    def test_raw_to_parsed_count_and_id_preservation(self):
        root = OUT / 'test-work'
        root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / 'raw.json'
            request = public_request('arxiv', 'permutation', 6)
            path.write_text(json.dumps({'source': 'arxiv', 'url': request['url'], 'params': request['params'], 'status': 200,
                                        'fetched_at': '2026-10-18T00:00:00+00:00', 'body': atom(1, 0, 6, ENTRY)}))
            search = {'query': 'permutation', 'limit': 6, 'seq': 1, 'round': 1, 'field': 'optimization',
                      'sources': ['arxiv'], 'per_source_ids': {'arxiv': ['2107.04683']}, 'errors': {}}
            row = window_row('card-A', search, 'arxiv', index_envelopes([path]), False)
            self.assertEqual((row['raw_count'], row['parsed_count'], row['parser_drop_count']), (1, 1, 0))
            search['per_source_ids']['arxiv'] = ['1110.5667']
            with self.assertRaises(AssertionError):
                window_row('card-A', search, 'arxiv', index_envelopes([path]), False)

    def test_missing_raw_source_has_no_fetch_fallback(self):
        search = {'query': 'alpha', 'limit': 6, 'seq': 1, 'round': 1, 'field': 'optimization',
                  'sources': ['s2'], 'per_source_ids': {'s2': []}, 'errors': {}}
        row = window_row('card-A', search, 's2', {}, True)
        self.assertEqual(row['pool_class'], 'unknown_missing_raw')
        self.assertIsNone(row['total'])

    def test_ordering_uses_unique_query_support_and_discovery_only(self):
        batch = [paper('2001.00001'), paper('2001.00002'), paper('2001.00003'), paper('2001.00004'), paper('2001.00005')]
        query = {'query': 'alpha', 'limit': 6, 'round': 1, 'field': 'optimization',
                 'sources': ['arxiv_local', 'openalex'], 'source_records': {'arxiv_local': batch + [batch[0]], 'openalex': []}}
        query2 = copy.deepcopy(query)
        query2.update(query='beta', source_records={'arxiv_local': [batch[4]], 'openalex': []})
        card = {'searches': [query, query2], 'refinement': {'seeds': ['arxiv:2001.00005', 'arxiv:2001.00001', 'arxiv:2001.00002', 'arxiv:2001.00003', 'arxiv:2001.00004']},
                'current_records': [], 'related': []}
        result = generic_ordering('card-A', card, {})
        self.assertEqual(result['initial_support_histogram'], {1: 4, 2: 1})
        self.assertEqual(result['selected_seeds'][0]['candidate_id'], 'arxiv:2001.00005')
        self.assertEqual(result['selected_seeds'][1]['query_support'], 1)

    def test_network_connections_prohibited(self):
        with self.assertRaises(AssertionError):
            socket.create_connection(('example.org', 443))


if __name__ == '__main__':
    unittest.main(verbosity=2)
