"""Package existing exclusion metadata only; never acquire or rebuild inventories."""
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELD = HERE.parents[1]
INV = HELD / 'preparation/inventory'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    original = (INV / 'exclusions.v1.json').read_bytes()
    assert sha(original) == (INV / 'exclusions.v1.sha256').read_text().split()[0]
    target = HERE / 'known-aliases.v1.json.gz'
    target.write_bytes(gzip.compress(original, mtime=0))
    names = ['known-aliases.v1.json.gz', '../inventory/v2/shipping-checkpoint.v2.json.gz',
             '../inventory/STAGE.v1.json', '../inventory/STAGE.v1.sha256',
             '../inventory/v2/STAGE.v2.json', '../inventory/v2/STAGE.v2.sha256']
    manifest = {'status': 'PENDING', 'future_acquisition_ready': False,
                'format': 'selective-exclusion-delivery-v1',
                'v1_uncompressed_sha256': sha(original),
                'named_logical_counts': {'original_pa2_11': 11, 'historical_recovery_19': 19, 'seen_seed_109': 109},
                'current_harness_sha256': sha((HELD / 'harness.py').read_bytes()),
                'artifacts': {n: sha((HERE / n).read_bytes()) for n in names},
                'source_authentication': 'Scoped local declared-source hashes, not publisher authentication. Full reauthentication requires preserved local originals.',
                'historical_stage': 'Original STAGE.v1/v2 are unchanged. Full stage artifact bodies are intentionally not all distributed. This delivery validates only its explicit artifacts.'}
    path = HERE / 'DELIVERY.json'
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (HERE / 'DELIVERY.sha256').write_text(sha(path.read_bytes()) + '  DELIVERY.json\n')
    print(json.dumps({'delivery_sha256': sha(path.read_bytes()), 'v1_compressed_bytes': target.stat().st_size}))


if __name__ == '__main__':
    build()
