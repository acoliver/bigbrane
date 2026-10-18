"""Two source-only checks of the HTTPS endpoint plus exact public source URLs."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx

from hybrid.literature.client import Literature, PROCESS_PACER

OUT = Path(__file__).resolve().parent


def request_hook(request):
    request.extensions['started_at'] = time.monotonic()


def response_hook(response):
    request = response.request
    value = {'diagnostic': True, 'acceptance_input': False, 'label': response_hook.label,
             'ts': datetime.now(timezone.utc).isoformat(), 'url': str(request.url.copy_with(query=None)),
             'params': dict(request.url.params), 'status': response.status_code,
             'location': response.headers.get('location'),
             'elapsed_s': time.monotonic() - request.extensions['started_at']}
    with (OUT / 'postfix-http.jsonl').open('a') as handle:
        handle.write(json.dumps(value) + '\n')


def main():
    results = []
    with httpx.Client(timeout=30, follow_redirects=True,
                      event_hooks={'request': [request_hook], 'response': [response_hook]}) as http:
        client = Literature(cache_root=OUT / 'postfix-cache', http=http, pacer=PROCESS_PACER, max_retries=0)
        response_hook.label = 'HTTPS exact Hwang'
        paper = client.get_paper('arxiv', '1110.5667')
        results.append({'label': response_hook.label, 'paper': asdict(paper)})
        response_hook.label = 'HTTPS original Hwang query'
        query = 'Bayesian program learning library abstraction DreamCoder'
        receipt = client.search_with_receipt(query, ('arxiv',), 6)
        results.append({'label': response_hook.label, 'receipt': receipt.event_data()})
        for name, url in [
            ('scipy-qap-2opt', 'https://docs.scipy.org/doc/scipy/reference/optimize.qap-2opt.html'),
            ('rehn-publications', 'https://www.math.uni-rostock.de/~rehn/publications.html'),
            ('rehn-thesis', 'https://www.math.uni-rostock.de/~rehn/docs/diploma-thesis-cs-rehn.pdf'),
        ]:
            response_hook.label = name
            response = http.get(url)
            response.raise_for_status()
            suffix = '.pdf' if name == 'rehn-thesis' else '.html'
            path = OUT / (name + suffix)
            path.write_bytes(response.content)
            results.append({'label': name, 'url': url, 'status': response.status_code,
                            'sha256': hashlib.sha256(response.content).hexdigest(), 'file': path.name})
    (OUT / 'postfix-results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
