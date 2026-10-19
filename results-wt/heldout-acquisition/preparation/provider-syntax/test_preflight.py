"""Focused network-prohibited checks against actual frozen and production contracts."""
from copy import deepcopy
import hashlib
import json
import re
import socket
from urllib.parse import parse_qs, urlsplit
import httpx
import pytest
import harness as h
from synthetic import make_pool, metadata, envelope
import preflight as p
from hybrid.literature import provenance
from hybrid.literature.cache import ResponseCache, request_key
from hybrid.literature.client import Literature
from hybrid.literature.keys import ApiKey
from hybrid.literature.sources import arxiv, openalex, semantic_scholar as s2
from hybrid.literature.transport import Pacer, Request, SourceError, Transport
from hybrid.tests.literature.support import FakeClock, RecordingSink

CASES = ['electron proton', 'alpha-beta, gamma!', 'alpha alpha 22 22', 'Alpha alpha ALPHA',
         'a:"b" OR (c) & q=other # x', 'naïve 中文 12', 'AND OR']
EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"/>'
ERROR = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#incorrect_id_format_for</id><summary>synthetic error</summary></entry></feed>'


def test_preparation_roots_and_plan():
    assert p.SIBLING == p.MAIN.parent / 'bigbrane-wt-r2'
    p.check_freeze()


@pytest.mark.parametrize('raw', CASES)
def test_q0_matches_actual_production(raw):
    assert h.search_request(raw, 'arxiv', 'Q0') == p.public_request(arxiv.search_request(raw, 6))


@pytest.mark.parametrize('raw', CASES)
def test_q1_retains_every_ascii_token_and_exact_groups(raw):
    got = h.search_request(raw, 'arxiv', 'Q1')
    assert got == p.public_request(p.q1_request(raw))
    query = got['params']['search_query']
    tokens = re.findall(r'[A-Za-z0-9]+', raw)
    assert re.findall(r'\(ti:([A-Za-z0-9]+) OR abs:\1\)', query) == tokens
    assert query == ' AND '.join(f'(ti:{t} OR abs:{t})' for t in tokens)
    encoded = str(httpx.Request('GET', got['endpoint'], params=got['params']).url)
    assert parse_qs(urlsplit(encoded).query)['search_query'] == [query]
    assert '%28' in encoded and '%29' in encoded and '%3A' in encoded
    assert set(parse_qs(urlsplit(encoded).query)) == set(got['params'])


@pytest.mark.parametrize('src,module', [('openalex', openalex), ('s2', s2)])
@pytest.mark.parametrize('raw', CASES)
def test_other_providers_identical_bytes_and_metadata(src, module, raw):
    before = metadata(); copy = deepcopy(before)
    q0, q1 = h.search_request(raw, src, 'Q0'), h.search_request(raw, src, 'Q1')
    assert json.dumps(q0).encode() == json.dumps(q1).encode()
    assert q0 == p.public_request(module.search_request(raw, 6))
    assert q0['params']['search' if src == 'openalex' else 'query'] == raw
    h.queries(before)
    assert before == copy


@pytest.mark.parametrize('raw', ['', '---', '中文'])
def test_production_empty_tokens_rejected(raw):
    with pytest.raises(ValueError): arxiv.search_request(raw, 6)
    with pytest.raises(ValueError): p.q1_request(raw)


@pytest.mark.parametrize('policy', ['Q0', 'Q1'])
@pytest.mark.parametrize('raw', ['', '---', '中文'])
def test_frozen_empty_tokens_rejected(raw, policy):
    with pytest.raises(ValueError): h.search_request(raw, 'arxiv', policy)


@pytest.mark.parametrize('src', ['arxiv', 'openalex', 's2'])
def test_frozen_unknown_policy_rejected(src):
    with pytest.raises(ValueError): h.search_request('alpha', src, 'FOREIGN')


@pytest.mark.parametrize('field,value', [('card_id','foreign'), ('arm','Q1S1'), ('source','openalex'),
    ('call_id','foreign-call'), ('query_sha','0'*64)])
def test_foreign_ownership_rejected(field, value):
    data = make_pool(populated=False)['payload']; data['calls'][0][field] = value
    with pytest.raises(ValueError): h.replay(envelope(data))


@pytest.mark.parametrize('source_index', [0, 1, 2])
def test_foreign_extra_request_parameter_rejected(source_index):
    data = make_pool(populated=False)['payload']
    call = data['calls'][source_index]
    call['request']['params']['foreign'] = 'unexpected'
    body = {'request':call['request'], 'records':call['records']}
    call['body_sha'] = h.digest(body)
    data['blobs'] = {h.digest(blob):blob for blob in data['blobs'].values()}
    with pytest.raises(ValueError, match='call identity mismatch: request'):
        h.replay(envelope(data))


def test_foreign_metadata_rejected():
    meta = metadata(); meta['target'] = 'forbidden'
    with pytest.raises(ValueError): h.queries(meta)


def test_query_empty_at_public_metadata_boundary():
    meta = metadata(); meta['concept'] = '---'
    for lens in meta['lenses']: lens['phrases'] = ['---'] * 3
    with pytest.raises(ValueError): h.queries(meta)


def test_request_hash_identity_and_blob_tampering():
    req = arxiv.search_request('alpha beta', 6)
    key = request_key('arxiv', req.url, req.params)
    assert key == request_key('arxiv', req.url, dict(reversed(list(req.params.items()))))
    assert key != request_key('arxiv', req.url, {**req.params, 'start':'6'})
    assert key != request_key('s2', req.url, req.params)
    data = make_pool(populated=False)['payload']
    data['blobs'][data['calls'][0]['body_sha']]['request']['params']['start'] = '6'
    with pytest.raises(ValueError): h.replay(envelope(data))


@pytest.mark.parametrize('module,body', [(arxiv, ERROR), (openalex,'{"error":"synthetic"}'),
                                        (s2,'{"error":"synthetic"}')])
def test_http200_api_error_is_not_parse_pool(tmp_path, module, body):
    clock = FakeClock(); sink = RecordingSink()
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200,text=body))) as http:
        lit = Literature(cache_root=tmp_path, http=http, pacer=Pacer(clock.clock,clock.sleep),
                         clock=clock.clock, sleep=clock.sleep, api_keys={})
        receipt = lit.search_with_receipt('synthetic neutral', (module.NAME,), 6, sink)
    assert receipt.errors and not receipt.papers
    record = sink.of_type('literature_transport')[0]
    assert record['terminal'] == 'parse_error'
    assert record['attempt_count'] == record['http_request_count'] == record['http_response_count'] == 1


def test_missing_total_unknown_not_zero_and_valid_empty():
    assert p.atom_facts(EMPTY) == {'total_results': None,'atom_entries':0}
    assert arxiv.parse_search(EMPTY, 'synthetic') == []
    body = EMPTY.replace('/>', '><totalResults xmlns="http://a9.com/-/spec/opensearch/1.1/">0</totalResults></feed>')
    assert p.atom_facts(body)['total_results'] == 0
    with pytest.raises(ValueError): p.atom_facts('<error/>')


def test_error_atom_rejected_despite_success_http():
    assert p.atom_facts(ERROR)['atom_entries'] == 1
    with pytest.raises(ValueError, match='arXiv API error'): arxiv.parse_search(ERROR,'synthetic')


def test_request_and_secret_projection_redacted(tmp_path):
    clock = FakeClock(); sink = RecordingSink(); secret = 'synthetic-key-value'
    query = 'synthetic-private-query'; seen = []
    def handler(request):
        seen.append(request)
        assert request.headers['x-api-key'] == secret
        return httpx.Response(200,text='{"data":[],"total":0}')
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        lit = Literature(cache_root=tmp_path, http=http, clock=clock.clock, sleep=clock.sleep,
            pacer=Pacer(clock.clock,clock.sleep), api_keys={'s2':ApiKey('s2',secret,'synthetic')})
        receipt = lit.search_with_receipt(query, ('s2',), 6, sink)
    record = sink.of_type('literature_transport')[0]
    text = json.dumps(record)
    assert secret not in text and query not in text
    assert record['association_id'] == hashlib.sha256(query.encode()).hexdigest()
    req = s2.search_request(query,6)
    assert record['request_id'] == request_key('s2',req.url,req.params)
    assert receipt.event_data()['transport_call_id'] == record['call_id']
    assert record['endpoint'] == 'https://api.semanticscholar.org'
    assert secret not in ''.join(file.read_text() for file in tmp_path.rglob('*.json'))
    assert provenance.endpoint('https://user:secret@host.invalid/path/secret?q=secret') == 'https://host.invalid'


def test_existing_transport_retry_redirect_cooldown_and_pacing(tmp_path):
    clock = FakeClock(); seen = []
    def handler(request):
        seen.append(clock.now)
        if len(seen) == 1: return httpx.Response(302,headers={'Location':'https://export.arxiv.org/api/query?redirected=1'})
        return httpx.Response(429)
    with httpx.Client(transport=httpx.MockTransport(handler),follow_redirects=True,timeout=30,max_redirects=20) as http:
        tr = Transport(ResponseCache(tmp_path),http,clock=clock.clock,sleep=clock.sleep,
                       pacer=Pacer(clock.clock,clock.sleep),api_keys={})
        with pytest.raises(SourceError) as exc: tr.get('arxiv',arxiv.search_request('alpha',6))
        record = exc.value.provenance
        assert record['attempt_count'] == 4 and record['http_request_count'] == record['http_response_count'] == 5
        assert [a['retry_wait_s'] for a in record['attempts']] == [5,10,20,0]
        assert record['cooldown_remaining_s'] == 600
        assert seen == [0,0,5,15,35]
        with pytest.raises(SourceError) as skipped: tr.get('arxiv',arxiv.search_request('beta',6))
        assert skipped.value.provenance['outcome'] == 'cooldown'
        assert skipped.value.provenance['http_request_count'] == 0
        assert len(seen) == 5


@pytest.mark.parametrize('action', [lambda:socket.socket(), lambda:socket.getaddrinfo('example.invalid',443),
                                  lambda:socket.socket(socket.AF_INET,socket.SOCK_DGRAM)])
def test_offline_network_is_prohibited(action):
    with pytest.raises(RuntimeError,match='Network prohibited'): action()


@pytest.mark.parametrize('src,module', [('openalex', openalex), ('s2', s2)])
@pytest.mark.parametrize('raw', CASES)
def test_unchanged_providers_actual_mock_transport_differential(tmp_path, src, module, raw):
    sent, records = [], []
    body = '{"results":[]}' if src == 'openalex' else '{"data":[]}'
    for policy in ['Q0','Q1']:
        clock = FakeClock(); sink = RecordingSink()
        def handler(request):
            sent.append((str(request.url), dict(request.url.params), dict(request.headers)))
            return httpx.Response(200,text=body)
        spec = h.search_request(raw,src,policy)
        request = Request(spec['endpoint'],spec['params'])
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            tr = Transport(ResponseCache(tmp_path/policy),http,clock=clock.clock,sleep=clock.sleep,
                           pacer=Pacer(clock.clock,clock.sleep),api_keys={})
            with provenance.operation('provider_syntax_preflight',raw,sink):
                fetched = tr.get(src,request)
                assert module.parse_search(fetched.body,fetched.fetched_at) == []
        records.extend(sink.of_type('literature_transport'))
    assert sent[0] == sent[1]
    assert records[0]['source'] == records[1]['source'] == src
    assert records[0]['association_id'] == records[1]['association_id']
    assert records[0]['request_id'] == records[1]['request_id']
    assert records[0]['call_id'] != records[1]['call_id']
    assert all(r['attempt_count'] == r['http_request_count'] == r['http_response_count'] == 1 for r in records)


def test_read_only_shared_leases_preserve_stamps_and_enforce_spacing(tmp_path, monkeypatch):
    import fcntl
    from types import SimpleNamespace
    clock = FakeClock(); clock.now = 100
    roots = [tmp_path/'main',tmp_path/'sibling']
    for root in roots:
        slots = root/'var/slots'; slots.mkdir(parents=True)
        (slots/'literature-arxiv.lock').write_bytes(b'')
        (slots/'literature-arxiv.last').write_text('99')
    monkeypatch.setattr(p,'MAIN',roots[0]); monkeypatch.setattr(p,'SIBLING',roots[1])
    monkeypatch.setattr(p,'time',SimpleNamespace(time=clock.clock,monotonic=clock.clock,sleep=clock.sleep))
    before = {file:file.read_bytes() for root in roots for file in (root/'var/slots').iterdir()}
    state = {}
    with p.shared_leases(state):
        assert clock.now == 102
        for root in roots:
            with (root/'var/slots/literature-arxiv.lock').open('rb') as handle:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert clock.now == 105 and state['ingress_wait_s'] == 2 and state['egress_wait_s'] == 3
    assert all(file.read_bytes() == body for file,body in before.items())
