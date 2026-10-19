"""Cache-free metadata assertions, including false readiness and tamper rejection."""
import gzip
import json
from pathlib import Path
import shutil
import socket

import pytest
from load_checkpoint import HERE, HELD, load, public_metadata
from harness import queries


@pytest.fixture(scope='module')
def snapshot():
    return load()


def test_conditional_inventory(snapshot):
    pool, checkpoint, manifest = snapshot
    assert manifest['named_logical_counts'] == {'original_pa2_11': 11, 'historical_recovery_19': 19, 'seen_seed_109': 109}
    assert len(pool.owners) == 23538 and len(pool.items) == 13554
    assert checkpoint['status'] == 'PENDING'
    assert not checkpoint['future_acquisition_ready']


def test_all_probe_aliases_excluded(snapshot):
    from harness import identities
    pool, checkpoint, _ = snapshot
    assert all(identities(r['identity']) <= set(pool.owners) for r in checkpoint['registry_delta']['probe_entries'])


def test_declared_source_bindings(snapshot):
    _, checkpoint, _ = snapshot
    assert len(checkpoint['source_bindings']) == 57604
    assert len(checkpoint['source_hashes']) == 29515
    assert all(len(digest) == 64 for digest in checkpoint['source_hashes'])


def test_blinded_provisional_shape():
    rows = json.loads((HELD / 'preparation/corpus-author/v2/BLINDED-METADATA.json').read_text())
    assert len(rows) == 11 and 'h03' not in {r['id'] for r in rows}
    all_queries = []
    for row in rows:
        assert set(row) == {'id', 'concept', 'lenses'}
        all_queries.extend(q for _, q in queries({k: row[k] for k in ['concept', 'lenses']}))
    assert len(all_queries) == len(set(all_queries)) == 132


def test_invalid_cohort_is_not_ready():
    receipt = json.loads((HERE / 'CORPUS-DISPOSITION.json').read_text())
    assert receipt['units'] == 11 and receipt['expected_units'] == 12
    assert not receipt['protocol_compliant'] and not receipt['future_acquisition_ready']
    assert receipt['body_request_limit_violations'] == 5
    assert not receipt['complete_historical_word_accounting_verified']
    assert receipt['initial_false_literal_records_preserved_locally'] == 4
    assert not receipt['independent_label_vault_present'] and not receipt['study_run']


def test_private_snapshot_receipt():
    receipt = json.loads((HERE / 'PRIVATE-PRESERVATION-PIN.json').read_text())
    assert receipt['snapshot_artifact_count'] == 160
    assert receipt['checks'] == {'checked': 160, 'mismatches': []}
    assert not receipt['labels_or_vault_created'] and not receipt['private_bodies_dossiers_identities_published']


def test_pin_tamper_rejected(tmp_path):
    shutil.copy(HERE / 'DELIVERY.json', tmp_path)
    shutil.copy(HERE / 'DELIVERY.sha256', tmp_path)
    with (tmp_path / 'DELIVERY.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='delivery pin mismatch'):
        load(tmp_path)


def test_secret_field_rejected():
    with pytest.raises(ValueError, match='secret field'):
        public_metadata({'api_key': 'synthetic'})


@pytest.mark.parametrize('action', [lambda: socket.socket(), lambda: socket.getaddrinfo('example.invalid', 443), lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)])
def test_network_prohibited(action):
    with pytest.raises(RuntimeError, match='Network prohibited'):
        action()


def test_unix_socket_construction_only():
    first, second = socket.socketpair()
    first.close()
    second.close()
