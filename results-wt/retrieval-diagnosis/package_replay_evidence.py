"""Project saved offline measurements into a small, source-hashed test fixture."""
import hashlib
import json
from pathlib import Path


def project(path):
    raw = path.read_bytes()
    saved = json.loads(raw)
    result = {key: saved[key] for key in ('acceptance', 'http_attempts', 'inputs_sha256')}
    result['provenance'] = {'source': path.name, 'bytes': len(raw),
                            'sha256': hashlib.sha256(raw).hexdigest(),
                            'method': 'field projection only; no replay or rescoring'}
    result['cards'] = {}
    for card, data in saved['cards'].items():
        item = {key: data[key] for key in ('historical_rank_reproduced', 'matches')}
        item['searches'] = []
        for search in data['searches']:
            receipt = {key: search[key] for key in
                       ('seq', 'event_ts', 'query', 'limit', 'per_source_ids')}
            receipt['source_records'] = {
                source: [{key: record[key] for key in
                          ('source', 'source_id', 'title', 'arxiv_id', 'doi')}
                         for record in records]
                for source, records in search['source_records'].items()}
            item['searches'].append(receipt)
        result['cards'][card] = item
    return result


if __name__ == '__main__':
    root = Path(__file__).parent
    (root / 'replay-evidence.json').write_text(
        json.dumps(project(root / 'offline.json'), separators=(',', ':')) + '\n')
