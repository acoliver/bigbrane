"""Offline regression checks for the two direct-helper contract corrections."""
import hashlib
import json
from pathlib import Path
import socket

import httpx
import pytest
import harness as h
import preflight as p
from hybrid.literature import provenance
from hybrid.literature.cache import ResponseCache
from hybrid.literature.sources import arxiv, openalex, semantic_scholar as s2
from hybrid.literature.transport import Pacer, Request, Transport
from hybrid.tests.literature.support import FakeClock, RecordingSink

FIX = Path(__file__).resolve().parent / 'helper-fix-v2'
OLD_SHA = '1546667435c4206f384ded06862da66cfcc2b8105bf0828423f1a2bcaf73b867'
OLD_BYTES = (FIX / 'previous/harness.py').read_bytes()
assert hashlib.sha256(OLD_BYTES).hexdigest() == OLD_SHA
OLD = {'__file__': str(h.ROOT / 'harness.py'), '__name__': 'helper_v1'}
exec(compile(OLD_BYTES, str(FIX / 'previous/harness.py'), 'exec'), OLD)
CASES = ['electron proton', 'alpha-beta, gamma!', 'alpha alpha 22 22', 'Alpha alpha ALPHA',
         'a:"b" OR (c) & q=other # x', 'naïve 中文 12', 'AND OR',
         ' \tAlpha\nbeta  ', 'a_0/9 é']
MODULES = {'arxiv': arxiv, 'openalex': openalex, 's2': s2, 'semantic_scholar': s2}
PARITY = []


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture(scope='module', autouse=True)
def save_parity():
    yield
    rows = sorted(PARITY, key=lambda row: (row['source'], row['policy'], row['raw_sha256']))
    (FIX / 'REQUEST-PARITY-v2.json').write_text(json.dumps({
        'old_harness_sha256': OLD_SHA, 'new_harness_sha256': h.file_sha(h.ROOT / 'harness.py'),
        'valid_cases': len(rows), 'mock_dispatches': 2 * len(rows), 'live_dispatches': 0,
        'all_equal': all(row['before'] == row['after'] for row in rows), 'cases': rows,
    }, indent=2, sort_keys=True) + '\n')


@pytest.mark.parametrize('raw', CASES)
@pytest.mark.parametrize('src', MODULES)
@pytest.mark.parametrize('policy', ['Q0', 'Q1'])
def test_valid_requests_equal_original_through_mock_transport(tmp_path, raw, src, policy):
    before = OLD['search_request'](raw, src, policy)
    after = h.search_request(raw, src, policy)
    assert before == after
    module = MODULES[src]
    expected = p.q1_request(raw) if src == 'arxiv' and policy == 'Q1' else module.search_request(raw, 6)
    assert after == p.public_request(expected)
    body = {'arxiv': '<feed xmlns="http://www.w3.org/2005/Atom"/>',
            'openalex': '{"results":[]}', 's2': '{"data":[]}'}[h.source(src)]
    projections = []
    for version, spec in [('before', before), ('after', after)]:
        sent = []
        clock = FakeClock()
        sink = RecordingSink()
        def handler(request):
            sent.append({'url': str(request.url), 'headers': dict(request.headers),
                         'body_sha256': hash_bytes(request.content)})
            return httpx.Response(200, text=body)
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            tr = Transport(ResponseCache(tmp_path / version), http, clock=clock.clock,
                           sleep=clock.sleep, pacer=Pacer(clock.clock, clock.sleep), api_keys={})
            with provenance.operation('helper_fix_valid_parity', raw, sink):
                fetched = tr.get(h.source(src), Request(spec['endpoint'], spec['params']))
                assert module.parse_search(fetched.body, fetched.fetched_at) == []
        record = sink.of_type('literature_transport')[0]
        assert len(sent) == 1
        assert record['attempt_count'] == record['http_request_count'] == record['http_response_count'] == 1
        projections.append({'spec_sha256': h.digest(spec), 'wire_sha256': h.digest(sent),
                            'request_id': record['request_id'], 'association_id': record['association_id'],
                            'source': record['source']})
    assert projections[0] == projections[1]
    PARITY.append({'source': src, 'policy': policy, 'raw_sha256': hash_bytes(raw.encode()),
                   'before': projections[0], 'after': projections[1]})


@pytest.mark.parametrize('raw', ['', ' \t\n', '---', '中文'])
@pytest.mark.parametrize('src', ['openalex', 's2'])
@pytest.mark.parametrize('policy', ['Q0', 'Q1'])
def test_other_providers_keep_raw_tokenless_text(raw, src, policy):
    spec = h.search_request(raw, src, policy)
    assert spec == OLD['search_request'](raw, src, policy)
    assert spec == p.public_request(MODULES[src].search_request(raw, 6))
    assert spec['params']['search' if src == 'openalex' else 'query'] == raw


@pytest.mark.parametrize('raw', [' ', '\t\r\n', 'éΩ'])
@pytest.mark.parametrize('policy', ['Q0', 'Q1'])
def test_arxiv_whitespace_and_non_ascii_tokens_rejected(raw, policy):
    with pytest.raises(ValueError, match='no ASCII search terms'):
        h.search_request(raw, 'arxiv', policy)


@pytest.mark.parametrize('policy', ['FOREIGN', '', None, 'q0', 'Q2'])
@pytest.mark.parametrize('src', MODULES)
def test_unknown_policy_rejected_for_every_source(src, policy):
    with pytest.raises(ValueError, match='unknown query policy'):
        h.search_request('alpha', src, policy)


@pytest.mark.parametrize('src', ['FOREIGN', '', None])
@pytest.mark.parametrize('policy', ['Q0', 'Q1'])
def test_unknown_source_rejection_unchanged(src, policy):
    for builder in [OLD['search_request'], h.search_request]:
        with pytest.raises(ValueError, match='unknown source'):
            builder('alpha', src, policy)


def test_policy_rejected_before_source_dispatch():
    with pytest.raises(ValueError, match='unknown query policy'):
        h.search_request('alpha', 'FOREIGN', 'FOREIGN')


def test_versioned_pins_preserve_historical_manifest():
    old = json.loads((p.ROOT / 'INPUT-HASHES.json').read_text())
    new = json.loads((p.ROOT / 'INPUT-HASHES-v2.json').read_text())
    assert set(old) == set(new)
    changed = [name for name in old if old[name] != new[name]]
    assert changed == [str(h.ROOT / 'harness.py')]
    assert old[changed[0]] == OLD_SHA
    assert new[changed[0]] == h.file_sha(h.ROOT / 'harness.py')
    p.check_freeze()
    h.verify_freeze()


def test_offline_guard_allows_unix_socketpair():
    first, second = socket.socketpair()
    first.close()
    second.close()
