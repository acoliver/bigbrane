"""Validate the opt-in compact delivery without accessing original source bodies."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
HELD = HERE.parents[1]
sys.path.insert(0, str(HELD))
from harness import Pool, corpus_gate, identities, require


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def public_metadata(value):
    if isinstance(value, dict):
        require(not set(value) & {'api_key', 'access_token', 'authorization', 'client_secret', 'private_key', 'password'}, 'secret field')
        for child in value.values():
            public_metadata(child)
    elif isinstance(value, list):
        for child in value:
            public_metadata(child)
    elif isinstance(value, str):
        require(not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|\bsk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}', value), 'credential pattern')


def load(directory=HERE):
    directory = Path(directory)
    manifest_raw = (directory / 'DELIVERY.json').read_bytes()
    require(sha(manifest_raw) == (directory / 'DELIVERY.sha256').read_text().split()[0], 'delivery pin mismatch')
    manifest = json.loads(manifest_raw)
    for name, expected in manifest['artifacts'].items():
        require(sha((directory / name).read_bytes()) == expected, 'delivery artifact mismatch: ' + name)
    raw = gzip.decompress((directory / 'known-aliases.v1.json.gz').read_bytes())
    require(sha(raw) == manifest['v1_uncompressed_sha256'], 'v1 registry mismatch')
    base = json.loads(raw)
    checkpoint = json.loads(gzip.decompress((directory / '../inventory/v2/shipping-checkpoint.v2.json.gz').read_bytes()))
    require(checkpoint['base_registry_sha256'] == sha(raw), 'v2 base mismatch')
    pool = Pool()
    delta = checkpoint['registry_delta']
    for row in base['entries'] + delta['entries'] + delta['probe_entries']:
        pool.add(row['identity'])
    require(sorted(sorted(r['ids']) for r in pool.items.values()) == checkpoint['alias_groups'], 'v2 alias groups mismatch')
    require(len(pool.owners) == delta['union']['v2_aliases'] == 23538, 'alias count mismatch')
    require(len(pool.items) == delta['union']['v2_conditional_unions'] == 13554, 'union count mismatch')
    require(len(delta['probe_entries']) == 12, 'probe identity count mismatch')
    require(len(checkpoint['source_paths']) == len(checkpoint['source_bindings']) == 57604, 'source binding count mismatch')
    require(len(checkpoint['source_hashes']) == 29515, 'source digest count mismatch')
    require([r[0] for r in checkpoint['source_bindings']] == list(range(57604)), 'source path binding mismatch')
    require(all(0 <= r[1] < 29515 for r in checkpoint['source_bindings']), 'source digest binding mismatch')
    require(checkpoint['status'] == delta['status'] == manifest['status'] == 'PENDING', 'pending scope lost')
    require(not checkpoint['future_acquisition_ready'] and not delta['full_known_repo_scope_complete'] and not manifest['future_acquisition_ready'], 'invalid readiness claim')
    gate = corpus_gate(json.loads((HELD / 'corpus.json').read_bytes()), base)
    require(gate['status'] == 'PENDING', 'invalid corpus readiness')
    public_metadata(base)
    public_metadata(checkpoint)
    return pool, checkpoint, manifest


if __name__ == '__main__':
    pool, checkpoint, manifest = load()
    print(json.dumps({'status': manifest['status'], 'aliases': len(pool.owners), 'conditional_unions': len(pool.items), 'source_paths': len(checkpoint['source_paths']), 'raw_sources_opened': 0}))
