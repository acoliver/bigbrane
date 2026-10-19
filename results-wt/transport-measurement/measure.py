"""One pinned, source-only diagnostic. No acceptance or model execution."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

from hybrid.literature import sources
from hybrid.literature.cache import request_key
from hybrid.literature.client import Literature, PROCESS_PACER, CONTACT_ENV
from hybrid.literature.keys import load_api_keys
from hybrid.literature.provenance import endpoint, identity_hash
from hybrid.literature.transport import MIN_INTERVAL_S
from hybrid.research.ledger import Ledger
from hybrid.research.stages.s2_prior_art import CardLiteratureSink

OUT = Path(__file__).resolve().parent
MAIN = OUT.parents[1]
WT = Path('/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2')
CARDS = ('card-3485a13f', 'card-f5ee7edb', 'card-a4f684fb', 'card-89248ee5')
SOURCES = ('arxiv', 'openalex', 's2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def revision(root):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def preservation():
    expected = json.loads((MAIN / 'results-wt/retrieval-diagnosis/PRESERVATION.json').read_text())['saved_inputs_sha256']
    assert len(expected) == 274
    actual = {path: digest(path) for path in expected}
    assert actual == expected, 'Original saved input changed'
    return actual


def implementation_hashes():
    paths = sorted((WT / 'hybrid/literature').glob('*.py'))
    paths += sorted((WT / 'hybrid/literature/sources').glob('*.py'))
    paths += [WT / 'hybrid/research/ledger.py', WT / 'hybrid/research/stages/s2_prior_art.py', WT / 'hybrid/slots.py']
    return {str(p): digest(p) for p in paths}


def configured_requests(query, limit):
    requests = {}
    for source in SOURCES:
        req = sources.MODULES[source].search_request(query, limit)
        params = dict(req.params)
        if source == 'openalex' and os.environ.get(CONTACT_ENV):
            params['mailto'] = os.environ[CONTACT_ENV]
        requests[source] = {'endpoint': endpoint(req.url), 'public_url': req.url,
                            'public_params': req.params,
                            'request_id': request_key(source, req.url, params)}
    return requests


def prepare():
    assert not (OUT / 'PLAN.json').exists(), 'Plan already pinned'
    before = preservation()
    write(OUT / 'PRESERVATION-before.json', {'at': now(), 'count': len(before), 'saved_inputs_sha256': before})
    keys = load_api_keys()
    entries = []
    fixtures = []
    for cid in CARDS:
        found = []
        for run in ('flipA', 'flipB'):
            path = WT / 'results-wt/kd8/runs' / run / 'ledger.jsonl'
            lines = path.read_bytes().splitlines()
            events = [json.loads(line) for line in lines]
            own = [e for e in events if e['data'].get('card_id') == cid]
            if not own:
                continue
            proposal = next(e for e in own if e['type'] == 'prior_art_queries')
            initial = set(proposal['data']['queries'])
            event = next(e for e in own if e['type'] == 'literature_search' and e['data']['query'] in initial)
            raw = lines[event['seq'] - 1]
            assert json.loads(raw) == event
            found.append((path, event, raw, proposal))
        assert len(found) == 1
        path, event, raw, proposal = found[0]
        data = event['data']
        query, limit = data['query'], data['limit']
        assert limit == 6
        entry = {'card_id': cid, 'query': query, 'query_sha256': identity_hash(query), 'limit': limit,
                 'original_ledger': str(path), 'original_ledger_sha256': digest(path),
                 'original_receipt_seq': event['seq'], 'original_receipt_sha256': hashlib.sha256(raw).hexdigest(),
                 'original_event_hash': event['hash'], 'proposal_seq': proposal['seq'],
                 'original_errors': data['errors'],
                 'original_counts': {s: len(ids) for s, ids in data['per_source_ids'].items()},
                 'requests': configured_requests(query, limit)}
        entries.append(entry)
        fixtures.append({'expected_callback_card_id': cid, 'original_receipt': event,
                         'original_receipt_sha256': entry['original_receipt_sha256']})
    assert len({e['query_sha256'] for e in entries}) == 4
    plan = {'created_at': now(), 'diagnostic': True, 'acceptance_input': False,
            'selection_rule': 'First saved initial literature_search per card in original ledger order. No semantic selection.',
            'main_revision': revision(MAIN), 'runtime_revision': revision(WT), 'runtime_root': str(WT),
            'client': 'hybrid.literature.client.Literature.search_with_receipt',
            'operation_order': 'card order below, then arxiv/openalex/s2; one single-source search per operation',
            'source_operations': 12, 'sources': SOURCES,
            'local_policy': 'No fresh local calls. Historical local counts retained separately from network measurements.',
            'cache_root': str(OUT / 'raw-cache'), 'pacer_slots': str(PROCESS_PACER._slots_dir),
            'min_interval_s': MIN_INTERVAL_S, 'max_retries': 3, 'max_attempts': 4,
            'http_timeout_s': 30.0, 'follow_redirects': True, 'base_backoff_s': 5.0,
            'max_backoff_s': 120.0, 'cooldown_s': 600.0,
            'retry_after_policy': 'Existing transport unchanged, including numeric cap and HTTP-date exponential fallback.',
            'configured_key_present': {s: s in keys for s in SOURCES},
            'contact_configured': bool(os.environ.get(CONTACT_ENV)),
            'no_extra_retries': True, 'no_fallback_keys': True, 'max_run_wall_s': 1600,
            'implementation_sha256': implementation_hashes(),
            'tool_sha256': {p.name: digest(p) for p in OUT.glob('*.py')}, 'cards': entries}
    write(OUT / 'FIXTURES.json', fixtures)
    plan['fixture_sha256'] = digest(OUT / 'FIXTURES.json')
    write(OUT / 'PLAN.json', plan)
    (OUT / 'PLAN.sha256').write_text(digest(OUT / 'PLAN.json') + '\n')
    print(json.dumps({'stage': 'prepared', 'plan_sha256': digest(OUT / 'PLAN.json'), 'queries': [e['query'] for e in entries]}))


def extract(events, plan):
    """Join by invocation identity, never by adjacent events or global event position."""
    expected = {(e['card_id'], s): e for e in plan['cards'] for s in plan['sources']}
    summaries = {}
    transports = {}
    receipts = []
    for event in events:
        d = event['data']
        if event['type'] == 'literature_search':
            call = d['transport_call_id']
            assert call not in summaries, 'duplicate search call'
            summaries[call] = d
        elif event['type'] == 'literature_transport':
            call = d['call_id']
            assert call not in transports, 'duplicate transport call'
            transports[call] = d
        elif event['type'] == 'literature_receipt':
            receipts.append(d)
    assert summaries.keys() == transports.keys(), 'missing invocation summary or transport'
    rows, seen = [], set()
    for call, d in transports.items():
        summary = summaries[call]
        key = (d['card_id'], d['source'])
        assert key in expected and key not in seen, 'unexpected or duplicate planned operation'
        seen.add(key)
        e = expected[key]
        assert summary['card_id'] == d['card_id'], 'cross-card attribution'
        assert summary['query'] == e['query'] and summary['sources'] == [d['source']]
        assert summary['limit'] == e['limit']
        assert d['association_id'] == e['query_sha256']
        assert d['request_id'] == e['requests'][d['source']]['request_id']
        assert d['endpoint'] == e['requests'][d['source']]['endpoint']
        assert d['operation'] == 'search_with_receipt'
        attempts = d['attempts']
        assert d['attempt_count'] == len(attempts) <= 4
        exchanges = [x for a in attempts for x in a['exchanges']]
        assert d['http_request_count'] == len(exchanges)
        assert d['http_response_count'] == sum(x['http_status'] is not None for x in exchanges)
        for ordinal, a in enumerate(attempts, 1):
            assert a['ordinal'] == ordinal
            assert a['http_request_count'] == len(a['exchanges'])
            assert a['http_response_count'] == sum(x['http_status'] is not None for x in a['exchanges'])
        if d['outcome'] != 'network':
            assert not attempts and not exchanges
        receipt_source = 'semantic_scholar' if d['source'] == 's2' else d['source']
        matches = [r for r in receipts if r['card_id'] == d['card_id'] and r['query'] == e['query'] and r['source'] == receipt_source]
        assert len(matches) == 1
        assert matches[0]['result_ids'] == summary['per_source_ids'][d['source']]
        rows.append({'card_id': d['card_id'], 'source': d['source'], 'call_id': call,
                     'query_sha256': d['association_id'], 'request_id': d['request_id'],
                     'outcome': d['outcome'], 'terminal': d['terminal'], 'error_type': d['error_type'],
                     'attempts': d['attempt_count'], 'http_requests': d['http_request_count'],
                     'http_responses': d['http_response_count'], 'source_receipts': 1,
                     'result_count': len(matches[0]['result_ids']), 'result_ids': matches[0]['result_ids'],
                     'statuses': [x['http_status'] for x in exchanges],
                     'redirects': sum(len(a['redirect_chain']) for a in attempts),
                     'pacer_wait_s': sum(a['pacer_wait_s'] for a in attempts),
                     'retry_wait_s': sum(a['retry_wait_s'] for a in attempts),
                     'retry_delay_s': sum(a['retry_delay_s'] for a in attempts),
                     'retry_after': [{'kind': a['retry_after_kind'], 'seconds': a['retry_after_s']} for a in attempts],
                     'duration_s': d['duration_s'], 'cooldown_remaining_s': d['cooldown_remaining_s'],
                     'errors': summary['errors']})
    assert len(receipts) == len(rows)
    counts = {name: sum(r[name] for r in rows) for name in ('attempts', 'http_requests', 'http_responses', 'source_receipts', 'redirects', 'pacer_wait_s', 'retry_wait_s', 'retry_delay_s')}
    counts.update(operations=len(rows), outcomes=dict(Counter(r['outcome'] for r in rows)),
                  terminals=dict(Counter(r['terminal'] for r in rows)),
                  statuses=dict(Counter(str(s) for r in rows for s in r['statuses'])))
    return {'diagnostic': True, 'acceptance_input': False, 'counts': counts, 'rows': rows,
            'completed_plan': seen == set(expected), 'attribution_verified': True}


def assert_secret_free(payload, secrets):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    assert all(not secret or secret not in text for secret in secrets), 'credential found'


def check_transport_redaction(events):
    forbidden = {'headers', 'authorization', 'x-api-key', 'url', 'params', 'query', 'body', 'location'}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(k.lower() for k in value)
            for key, item in value.items():
                if key == 'endpoint':
                    assert endpoint(item) == item
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    for event in events:
        if event['type'] == 'literature_transport':
            check(event['data'])


def run():
    plan = json.loads((OUT / 'PLAN.json').read_text())
    assert digest(OUT / 'PLAN.json') == (OUT / 'PLAN.sha256').read_text().strip()
    assert implementation_hashes() == plan['implementation_sha256']
    assert {p.name: digest(p) for p in OUT.glob('*.py')} == plan['tool_sha256']
    assert preservation() == json.loads((OUT / 'PRESERVATION-before.json').read_text())['saved_inputs_sha256']
    assert not (OUT / 'raw-cache').exists()
    assert not (OUT / 'runtime').exists()
    keys = load_api_keys()
    assert {s: s in keys for s in SOURCES} == plan['configured_key_present']
    for entry in plan['cards']:
        assert configured_requests(entry['query'], entry['limit']) == entry['requests']
    # The sentinel prevents repeating even a partially completed measurement.
    with (OUT / 'RUN-ONCE.json').open('x') as f:
        json.dump({'started_at': now(), 'plan_sha256': digest(OUT / 'PLAN.json')}, f)
    logging.disable(logging.CRITICAL)
    ledger = Ledger(OUT / 'runtime')
    ctx = SimpleNamespace(ledger=ledger)
    results = []
    def timeout(signum, frame):
        raise TimeoutError('diagnostic wall bound')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(plan['max_run_wall_s'])
    failure = None
    try:
        client = Literature(cache_root=OUT / 'raw-cache', local_index_path=None,
                            contact_email=os.environ.get(CONTACT_ENV) or None,
                            api_keys=keys, pacer=PROCESS_PACER)
        for entry in plan['cards']:
            for source in plan['sources']:
                receipt = client.search_with_receipt(entry['query'], (source,), entry['limit'],
                                                     CardLiteratureSink(ctx, entry['card_id']))
                item = {'card_id': entry['card_id'], 'source': source, 'query': entry['query'],
                        'transport': list(receipt.transport), 'errors': receipt.errors,
                        'papers': [{k: getattr(p, k) for k in ('source', 'source_id', 'title', 'year', 'abstract', 'arxiv_id', 'doi')} for p in receipt.papers]}
                assert_secret_free(item, [k.value for k in keys.values()])
                results.append(item)
                write(OUT / 'RESULTS.json', results)
                print(json.dumps({'stage': 'completed_operation', 'card_id': entry['card_id'], 'source': source,
                                  'result_count': len(receipt.papers), 'outcome': receipt.transport[0]['outcome'],
                                  'terminal': receipt.transport[0]['terminal']}), flush=True)
    except BaseException as exc:
        failure = type(exc).__name__
    finally:
        signal.alarm(0)
        write(OUT / 'RUN-END.json', {'finished_at': now(), 'completed_operations': len(results), 'failure_type': failure})
        write(OUT / 'PRESERVATION-after.json', {'at': now(), 'count': 274, 'saved_inputs_sha256': preservation()})
    if failure:
        raise RuntimeError(f'Diagnostic stopped: {failure}') from None
    finalize(keys)


def finalize(keys=None):
    keys = load_api_keys() if keys is None else keys
    plan = json.loads((OUT / 'PLAN.json').read_text())
    ledger = Ledger(OUT / 'runtime')
    assert ledger.verify() == (True, None)
    events = list(ledger.iter_events())
    check_transport_redaction(events)
    summary = extract(events, plan)
    summary['ledger_chain_verified'] = True
    summary['transport_redaction_verified'] = True
    write(OUT / 'SUMMARY.json', summary)
    scanned = []
    for path in OUT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            assert_secret_free(path.read_bytes().decode('utf-8', errors='replace'), [k.value for k in keys.values()])
            scanned.append(str(path.relative_to(OUT)))
    write(OUT / 'SECURITY.json', {'actual_key_values_absent': True, 'transport_origin_only': True,
                                'no_transport_headers_urls_queries_bodies': True, 'files_scanned': scanned,
                                'configured_key_present': plan['configured_key_present']})
    print(json.dumps(summary['counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run', 'finalize'))
    mode = parser.parse_args().mode
    {'prepare': prepare, 'run': run, 'finalize': finalize}[mode]()
