"""Diagnostic preparation only; this module never patches the frozen harness."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
import httpx
from hybrid.literature import provenance
from hybrid.literature.cache import ResponseCache, request_key
from hybrid.literature.events import utc_now
from hybrid.literature.sources import arxiv
from hybrid.literature.transport import Pacer, Request, SourceError, Transport

ROOT = Path(__file__).resolve().parent
SIBLING = ROOT.parents[3]
MAIN = Path('/Volumes/XS1000/acoliver/projects/bigbrane')
DOC_URL = 'https://info.arxiv.org/help/api/user-manual.html'
PHRASE = 'electron proton'
NS = {'a': 'http://www.w3.org/2005/Atom', 'o': 'http://a9.com/-/spec/opensearch/1.1/'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def check_freeze():
    assert sha((ROOT / 'PLAN.md').read_bytes()) == (ROOT / 'PLAN.sha256').read_text().split()[0]
    for name, expected in json.loads((ROOT / 'INPUT-HASHES-v2.json').read_text()).items():
        if '/var/slots/' not in name:
            assert sha(Path(name).read_bytes()) == expected, name


def q1_request(raw):
    tokens = re.findall(r'[A-Za-z0-9]+', raw)
    if not tokens:
        raise ValueError('no ASCII search terms')
    return Request(arxiv.BASE_URL, {'search_query': ' AND '.join(f'(ti:{t} OR abs:{t})' for t in tokens),
                                  'start': '0', 'max_results': '6', 'sortBy': 'relevance'})


def public_request(request):
    return {'endpoint': request.url, 'params': request.params}


def atom_facts(body):
    """Preparation boundary: preserve an absent provider total as unknown."""
    root = ET.fromstring(body)
    if root.tag != '{http://www.w3.org/2005/Atom}feed':
        raise ValueError('not an Atom feed')
    raw_total = root.findtext('o:totalResults', namespaces=NS)
    return {'total_results': None if raw_total is None else int(raw_total),
            'atom_entries': len(root.findall('a:entry', NS))}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)


def fetch_docs():
    check_freeze()
    marker = ROOT / 'docs/STARTED.json'
    if marker.exists():
        raise RuntimeError('documentation fetch already started; do not repeat')
    save('docs/STARTED.json', {'url': DOC_URL, 'started_at': utc_now(), 'max_exchanges': 21})
    exchanges = []
    def request_hook(request):
        if len(exchanges) >= 21:
            raise RuntimeError('documentation exchange budget exhausted')
        exchanges.append({'endpoint': provenance.endpoint(request.url), 'status': None})
    def response_hook(response):
        exchanges[-1]['status'] = response.status_code
        body = response.read()
        (ROOT / f'docs/exchange-{len(exchanges):02d}.bin').write_bytes(body)
        exchanges[-1]['body_sha256'] = sha(body)
    receipt = {'url': DOC_URL, 'exchanges': exchanges, 'status': 'pending'}
    try:
        with httpx.Client(timeout=30, follow_redirects=True, max_redirects=20,
                          event_hooks={'request': [request_hook], 'response': [response_hook]}) as http:
            response = http.get(DOC_URL)
            response.raise_for_status()
            body = response.content
            (ROOT / 'docs/user-manual.html').write_bytes(body)
            text = Text(); text.feed(response.text)
            plain = '\n'.join(text.parts)
            (ROOT / 'docs/user-manual.txt').write_text(plain)
            terms = ['ti', 'abs', 'all', 'AND', 'OR', '(', ')', 'search_query', 'start',
                     'max_results', 'sortBy', 'relevance', 'electron', 'proton']
            evidence = {t: t in plain for t in terms}
            receipt.update(status='downloaded' if all(evidence.values()) else 'pending_evidence',
                           final_url=str(response.url), body_sha256=sha(body), terms_present=evidence)
    except httpx.HTTPError as exc:
        receipt.update(status='terminal_error', error_type=type(exc).__name__)
    finally:
        receipt.update(http_requests=len(exchanges), http_responses=sum(e['status'] is not None for e in exchanges))
        save('docs/RECEIPT.json', receipt)
    print(json.dumps(receipt))


class Sink:
    def append(self, kind, actor, data):
        with (ROOT / 'runtime/events.jsonl').open('a') as output:
            output.write(json.dumps({'type': kind, 'actor': actor, 'data': data}) + '\n')


@contextmanager
def shared_leases(state):
    paths = sorted(root / 'var/slots/literature-arxiv.lock' for root in [MAIN, SIBLING])
    handles = []
    try:
        for path in paths:
            handle = path.open('rb')
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX)
        state['leases_acquired_at'] = utc_now()
        latest = max(float(path.with_suffix('.last').read_text()) for path in paths)
        wait = max(0.0, min(latest, time.time()) + 3 - time.time())
        start = time.monotonic(); time.sleep(wait)
        state['ingress_wait_s'] = time.monotonic() - start
        yield
    finally:
        if len(handles) == len(paths):
            start = time.monotonic(); time.sleep(3)
            state['egress_wait_s'] = time.monotonic() - start
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN); handle.close()
        state['leases_released_at'] = utc_now()


def exclusions_from_bodies():
    papers = {}
    for file in sorted((ROOT / 'runtime/raw').glob('*.bin')):
        try:
            root = ET.fromstring(file.read_bytes())
        except ET.ParseError:
            continue
        for entry in root.findall('a:entry', NS):
            sid = entry.findtext('a:id', '', NS)
            if '/api/errors' in sid or not sid:
                continue
            title = ' '.join(entry.findtext('a:title', '', NS).split())
            aid = sid.rsplit('/abs/', 1)[-1]
            key = re.sub(r'v\d+$', '', aid)
            item = papers.setdefault(key, {'source': 'arxiv', 'source_id': key, 'arxiv_id': key,
                                          'title': title, 'source_urls': [], 'body_sha256': []})
            if sid not in item['source_urls']: item['source_urls'].append(sid)
            bsha = sha(file.read_bytes())
            if bsha not in item['body_sha256']: item['body_sha256'].append(bsha)
            doi = entry.findtext('{http://arxiv.org/schemas/atom}doi')
            if doi: item['doi'] = doi
    handoff = {'kind': 'diagnostic_provider_probe_exclusions', 'not_a_study': True,
               'plan_sha256': (ROOT / 'PLAN.sha256').read_text().split()[0],
               'future_scope': 'Exclude all listed identities and resolved aliases from mechanisms and reference works.',
               'unknown_aliases': 'Only returned arXiv identifiers, titles, URLs and any supplied DOI; unresolved links remain.',
               'papers': list(papers.values())}
    save('PROBE-EXCLUSIONS.json', handoff)
    (ROOT / 'PROBE-EXCLUSIONS.sha256').write_text(sha((ROOT / 'PROBE-EXCLUSIONS.json').read_bytes()) + '  PROBE-EXCLUSIONS.json\n')
    return len(papers)


def probe():
    check_freeze()
    assert json.loads((ROOT / 'docs/RECEIPT.json').read_text())['status'] == 'downloaded'
    assert json.loads((ROOT / 'OFFLINE-SUMMARY.json').read_text())['required_probe_builder_checks_passed']
    assert not (ROOT / 'runtime/STARTED.json').exists(), 'source batch already started; no repeats'
    save('runtime/STARTED.json', {'phrase': PHRASE, 'started_at': utc_now(), 'operations_max': 2})
    (ROOT / 'runtime/raw').mkdir()
    exchanges, operations, state = [], [], {}
    def started(request):
        if len(exchanges) >= 168: raise RuntimeError('provider exchange budget exhausted')
        exchanges.append({'ordinal': len(exchanges) + 1, 'endpoint': provenance.endpoint(request.url),
                          'status': None, 'started_at': utc_now()})
    def received(response):
        exchanges[-1]['status'] = response.status_code
        raw = response.read()
        path = ROOT / f'runtime/raw/exchange-{len(exchanges):03d}.bin'
        path.write_bytes(raw)
        exchanges[-1]['body_sha256'] = sha(raw)
    summary = {'status': 'pending', 'source_operations': operations, 'exchanges': exchanges, 'pacing': state}
    try:
        with shared_leases(state), httpx.Client(timeout=30, follow_redirects=True, max_redirects=20,
                    event_hooks={'request': [started], 'response': [received]}) as http:
            tr = Transport(ResponseCache(ROOT / 'runtime/cache'), http,
                           pacer=Pacer(time.time, slots_dir=ROOT / 'runtime/slots'), api_keys={})
            for policy, request in [('Q0', arxiv.search_request(PHRASE, 6)), ('Q1', q1_request(PHRASE))]:
                op = {'policy': policy, 'source': 'arxiv', 'phrase_sha256': provenance.identity_hash(PHRASE),
                      'request': public_request(request), 'request_id': request_key('arxiv', request.url, request.params),
                      'status': 'pending', 'provenance': None}
                operations.append(op)
                with provenance.operation('provider_syntax_preflight', PHRASE, Sink()) as context:
                    try:
                        fetched = tr.get('arxiv', request)
                        op['provenance'] = fetched.provenance
                        op['body_sha256'] = sha(fetched.body.encode())
                        facts = atom_facts(fetched.body)
                        papers = arxiv.parse_search(fetched.body, fetched.fetched_at)
                        if len(papers) > 6: raise ValueError('response exceeds frozen first window')
                        op.update(status='syntax_valid', returned_count=len(papers), **facts)
                    except (SourceError, ValueError, ET.ParseError) as exc:
                        op.update(status='terminal_error', error_type=type(exc).__name__)
                        if context.records: op['provenance'] = context.records[-1]
                save('runtime/RECEIPT.json', summary)
                exclusions_from_bodies()
                if op['status'] != 'syntax_valid': break
            summary['status'] = 'syntax_valid' if len(operations) == 2 and all(o['status'] == 'syntax_valid' for o in operations) else 'pending_terminal_failure'
    except (OSError, httpx.HTTPError, RuntimeError) as exc:
        summary.update(status='pending_terminal_failure', error_type=type(exc).__name__)
    finally:
        count = exclusions_from_bodies()
        records = [o['provenance'] for o in operations if o['provenance']]
        summary.update(logical_operations=len(operations), attempts=sum(r['attempt_count'] for r in records),
            http_requests=len(exchanges), http_responses=sum(e['status'] is not None for e in exchanges),
            redirects=sum(e['status'] in [301,302,303,307,308] for e in exchanges),
            retries=sum(max(0,r['attempt_count']-1) for r in records), diagnostic_unique_papers=count,
            cooldown_remaining_s=max([r['cooldown_remaining_s'] for r in records] or [0]),
            source_operations_unspent=2-len(operations))
        save('runtime/RECEIPT.json', summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ['source_operations','exchanges','pacing']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['docs', 'probe'])
    args = parser.parse_args()
    (fetch_docs if args.stage == 'docs' else probe)()
