"""Opt-in delta/checkpoint readers. These do not change the frozen harness registry."""
from pathlib import Path
import re
from build_delta import Pool, identities, load, require, sha, verify_map


def compose(base_entries, delta_entries):
    pool = Pool()
    for row in [*base_entries, *delta_entries]:
        pool.add(row['identity'])
    return pool


def excluded(pool, identity):
    return bool(identities(identity) & set(pool.owners))


def ready(registry):
    return (registry['full_known_repo_scope_complete']
            and registry['future_acquisition_ready'] and registry['status'] != 'PENDING')


def public_only(value):
    text = str(value)
    require(not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|\bsk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}', text),
            'possible private credential in projected metadata')
    if isinstance(value, dict):
        require(not (set(value) & {'api_key', 'access_token', 'authorization', 'client_secret', 'private_key', 'password'}),
                'secret field in projected metadata')
        for child in value.values():
            public_only(child)
    elif isinstance(value, list):
        for child in value:
            public_only(child)
    return True


def check_stage(root):
    root = Path(root)
    require(sha(root / 'STAGE.v2.json') == (root / 'STAGE.v2.sha256').read_text().split()[0], 'v2 stage pin mismatch')
    stage = load(root / 'STAGE.v2.json')
    verify_map({str(root / name): h for name, h in stage['artifacts'].items()})
    require(sha(stage['implementation_revision']['path']) == stage['implementation_revision']['sha256'],
            'v2 implementation revision changed')
    require(sha(root.parent / 'exclusions.v1.json') == stage['v1_registry_sha256'], 'v1 base changed')
    require(sha(root.parent / 'STAGE.v1.json') == stage['historical_v1_stage_sha256'], 'historical v1 stage changed')
    return stage


def read_registry(root):
    root = Path(root)
    check_stage(root)
    delta = load(root / 'registry.v2.json')
    base = root / delta['base']['path']
    require(sha(base) == delta['base']['sha256'], 'base registry pin mismatch')
    pool = compose(load(base)['entries'], delta['entries'] + delta['probe_entries'])
    checkpoint = load(root / 'alias-checkpoint.json')
    require(sorted(sorted(r['ids']) for r in pool.items.values()) == checkpoint['groups'], 'alias checkpoint mismatch')
    public_only(checkpoint)
    public_only(delta)
    return pool, delta


def prose_excluded(root, descriptor):
    from build_delta import norm_title
    rows = load(Path(root) / 'descriptor-prose-aliases.json')
    return norm_title(descriptor) in {r['normalized_descriptor'] for r in rows}
