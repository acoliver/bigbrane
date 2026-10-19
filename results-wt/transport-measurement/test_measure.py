"""Offline tests for the scoped diagnostic and its event joins."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import httpx

from measure import (assert_secret_free, check_transport_redaction, configured_requests,
                     extract, identity_hash)
from hybrid.literature.client import Literature
from hybrid.research.ledger import Ledger
from hybrid.research.stages.s2_prior_art import CardLiteratureSink

ATOM = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class Clock:
    def __init__(self):
        self.value = 1000.0

    def clock(self):
        return self.value

    def sleep(self, delay):
        self.value += delay


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        self.calls = []

    def capture(self):
        def handler(req):
            self.calls.append(str(req.url))
            n = len(self.calls)
            if n == 1:
                return httpx.Response(429, headers={'Retry-After': '7'})
            if n == 2:
                return httpx.Response(301, headers={'Location': 'https://export.arxiv.org/redirected?token=sentinel'})
            return httpx.Response(200, text=ATOM)
        http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        self.addCleanup(http.close)
        client = Literature(cache_root=self.root / 'cache', http=http,
                            clock=self.clock.clock, sleep=self.clock.sleep)
        ledger = Ledger(self.root / 'ledger')
        ctx = SimpleNamespace(ledger=ledger)
        client.search_with_receipt('alpha', ('arxiv',), 6, CardLiteratureSink(ctx, 'card-A'))
        client.search_with_receipt('alpha', ('arxiv',), 6, CardLiteratureSink(ctx, 'card-B'))
        assert ledger.verify() == (True, None)
        events = list(ledger.iter_events())
        plan = {'sources': ['arxiv'], 'cards': [
            {'card_id': cid, 'query': 'alpha', 'query_sha256': identity_hash('alpha'), 'limit': 6,
             'requests': configured_requests('alpha', 6)} for cid in ('card-A', 'card-B')]}
        return events, plan

    def test_interleaved_retry_redirect_and_zero_network_cache(self):
        events, plan = self.capture()
        # Place card B's cached completion before card A's transport delivery.
        events = [events[i] for i in (3, 0, 4, 1, 5, 2)]
        result = extract(events, plan)
        counts = result['counts']
        self.assertEqual(counts['operations'], 2)
        self.assertEqual(counts['source_receipts'], 2)
        self.assertEqual(counts['attempts'], 2)
        self.assertEqual(counts['http_requests'], 3)
        self.assertEqual(counts['http_responses'], 3)
        self.assertEqual(counts['statuses'], {'429': 1, '301': 1, '200': 1})
        self.assertEqual(counts['redirects'], 1)
        self.assertEqual(counts['retry_wait_s'], 7)
        rows = {r['card_id']: r for r in result['rows']}
        self.assertEqual(rows['card-B']['outcome'], 'cache_hit')
        self.assertEqual(rows['card-B']['http_requests'], 0)
        self.assertEqual(rows['card-A']['attempts'], 2)
        self.assertNotEqual(rows['card-A']['call_id'], rows['card-B']['call_id'])
        self.assertTrue(result['completed_plan'])
        check_transport_redaction(events)
        transports = [e for e in events if e['type'] == 'literature_transport']
        self.assertNotIn('sentinel', json.dumps(transports))

    def test_cross_card_transport_rejected(self):
        events, plan = self.capture()
        events[2]['data']['card_id'] = 'card-B'
        with self.assertRaises(AssertionError):
            extract(events, plan)

    def test_missing_transport_rejected(self):
        events, plan = self.capture()
        with self.assertRaises(AssertionError):
            extract(events[:-1], plan)

    def test_duplicate_transport_rejected(self):
        events, plan = self.capture()
        with self.assertRaises(AssertionError):
            extract(events + [copy.deepcopy(events[-1])], plan)

    def test_exchange_count_corruption_rejected(self):
        events, plan = self.capture()
        events[2]['data']['http_request_count'] = 2
        with self.assertRaises(AssertionError):
            extract(events, plan)

    def test_credential_and_url_query_redaction_checks(self):
        assert_secret_free({'safe': 'public'}, ['dummy-credential'])
        with self.assertRaises(AssertionError):
            assert_secret_free({'unsafe': 'dummy-credential'}, ['dummy-credential'])
        for value in ({'endpoint': 'https://user:dummy@example.org/path?api_key=secret'},
                      {'headers': {'Authorization': 'Bearer dummy'}},
                      {'url': 'https://example.org/?token=secret'}):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                check_transport_redaction([{'type': 'literature_transport', 'data': value}])

    def test_cooldown_has_no_invented_http(self):
        http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(429)))
        self.addCleanup(http.close)
        client = Literature(cache_root=self.root / 'cooldown-cache', http=http,
                            clock=self.clock.clock, sleep=self.clock.sleep)
        ledger = Ledger(self.root / 'cooldown-ledger')
        ctx = SimpleNamespace(ledger=ledger)
        entries = []
        for cid, query in (('card-A', 'alpha'), ('card-B', 'beta')):
            client.search_with_receipt(query, ('arxiv',), 6, CardLiteratureSink(ctx, cid))
            entries.append({'card_id': cid, 'query': query, 'query_sha256': identity_hash(query),
                            'limit': 6, 'requests': configured_requests(query, 6)})
        result = extract(list(ledger.iter_events()), {'sources': ['arxiv'], 'cards': entries})
        self.assertEqual(result['counts']['attempts'], 4)
        self.assertEqual(result['counts']['http_requests'], 4)
        self.assertEqual(result['counts']['outcomes'], {'network': 1, 'cooldown': 1})
        self.assertEqual(result['rows'][1]['statuses'], [])
        self.assertEqual(result['rows'][1]['source_receipts'], 1)


if __name__ == '__main__':
    unittest.main()
