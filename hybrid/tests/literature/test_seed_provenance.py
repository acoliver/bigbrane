"""Durable seed receipts, input authentication, and bounded historical projection."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hybrid.literature import seed
from hybrid.literature.build_seed import write_seed_bundle
from hybrid.literature.cache import request_key
from hybrid.literature.paper import paper_to_dict
from hybrid.literature.sources import openalex


def checksum(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def test_shipped_history_is_complete_and_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = seed.read_build_log()
    assert sum(len(found) for row in rows for found in row['per_source'].values()) == 109
    document = json.loads(seed.SEED_PROVENANCE_PATH.read_bytes())
    assert document['payload']['log_sha256'] == '5a256f4a78b16415e85dd6ed553529f1d5361d4804d8f2bba206215077036e26'
    assert len(rows) == 59


@pytest.mark.parametrize('field', ['source_id', 'title', 'url', 'lens', 'source', 'log_sha256',
                                   'cache_sha256', 'row_sha256', 'query', 'fetched_at'])
def test_tampered_history_fails_checksum(tmp_path, field):
    doc = json.loads(seed.SEED_PROVENANCE_PATH.read_bytes())
    row = doc['payload']['rows'][0]
    source = next(iter(row['per_source']))
    if field == 'log_sha256':
        doc['payload'][field] = '0' * 64
    elif field == 'lens' or field == 'query':
        row[field] += '-changed'
    elif field == 'row_sha256':
        row['origin'][field] = '0' * 64
    elif field == 'cache_sha256':
        row['origin']['receipts'][source][field] = '0' * 64
    else:
        row['per_source'][source][0][field] += '-changed'
    path = tmp_path / 'provenance.json'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='checksum'):
        seed.read_seed_provenance(path)


@pytest.mark.parametrize('field', ['source_id', 'title', 'url', 'lens', 'source', 'fetched_at'])
def test_rechecksummed_metadata_still_must_match_seed(tmp_path, field):
    doc = json.loads(seed.SEED_PROVENANCE_PATH.read_bytes())
    row = doc['payload']['rows'][0]
    source = next(iter(row['per_source']))
    if field == 'lens':
        row[field] = 'morphogenesis'
    else:
        row['per_source'][source][0][field] += '-changed'
    doc['payload_sha256'] = checksum(doc['payload'])
    path = tmp_path / 'provenance.json'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        seed.read_seed_provenance(path)


def test_missing_explicit_log_and_provenance_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        seed.read_build_log(tmp_path / 'missing.jsonl')
    with pytest.raises(FileNotFoundError):
        seed.read_seed_provenance(tmp_path / 'missing.json')


@pytest.fixture
def original_inputs(tmp_path):
    query = 'echo state network'
    request = openalex.search_request(query, 10)
    body = json.dumps({'results': [{'id': 'https://openalex.org/W123', 'title': 'Original title',
                                    'publication_year': 2024}]})
    envelope = dict(source='openalex', url=request.url, params=request.params,
                    status=200, fetched_at='2026-10-09T23:00:00+00:00', body=body)
    paper = openalex.parse_search(body, envelope['fetched_at'])[0]
    entry = seed.SeedEntry('reservoir_computing', paper, 'A source-derived test entry.')
    cache = tmp_path / 'cache'
    path = cache / 'openalex' / (request_key('openalex', request.url, request.params) + '.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(envelope))
    row = dict(lens=entry.lens, query=query, ts=envelope['fetched_at'], sources=['openalex'],
               limit=10, errors={}, per_source={'openalex': [paper_to_dict(paper)]})
    log = tmp_path / 'original.jsonl'
    seed.write_build_log([row], log)
    return entry, log, cache, path, row


def test_builder_emits_validated_bundle(original_inputs, tmp_path):
    entry, log, cache, _, _ = original_inputs
    target, provenance = tmp_path / 'seed.toml', tmp_path / 'seed.json'
    write_seed_bundle([entry], '2026-10-09', seed_path=target, provenance_path=provenance,
                      log_path=log, cache_root=cache)
    assert seed.read_seed_provenance(provenance, target)[0]['per_source']['openalex'] == [paper_to_dict(entry.paper)]
    assert seed.read_build_log(log)[0]['query'] == 'echo state network'


@pytest.mark.parametrize('field', ['source_id', 'title', 'lens', 'source'])
def test_compiler_rejects_unsupported_seed_metadata(original_inputs, field):
    entry, log, cache, _, _ = original_inputs
    changed = replace(entry, lens='morphogenesis') if field == 'lens' else replace(
        entry, paper=replace(entry.paper, **{field: 'arxiv' if field == 'source' else 'changed'}))
    with pytest.raises(ValueError, match='unsupported'):
        seed.compile_seed_provenance([changed], log, cache)


def test_compiler_rejects_fabricated_log_hit(original_inputs):
    entry, log, cache, _, row = original_inputs
    row['per_source']['openalex'][0]['title'] = 'Fabricated title'
    seed.write_build_log([row], log)
    with pytest.raises(ValueError, match='original source receipt'):
        seed.compile_seed_provenance([entry], log, cache)


def test_compiler_rejects_wrong_source_receipt(original_inputs):
    entry, log, cache, path, _ = original_inputs
    envelope = json.loads(path.read_bytes())
    envelope['params']['search'] = 'different search'
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='request mismatch'):
        seed.compile_seed_provenance([entry], log, cache)


def test_missing_receipt_cannot_fall_back_to_seed(original_inputs):
    entry, log, cache, path, _ = original_inputs
    path.unlink()
    with pytest.raises(FileNotFoundError):
        seed.compile_seed_provenance([entry], log, cache)


def test_compiler_streams_log_once_and_discards_irrelevant_history(original_inputs, monkeypatch):
    entry, log, cache, _, row = original_inputs
    unrelated = dict(row, lens='morphogenesis')
    seed.write_build_log((unrelated if n < 2000 else row for n in range(2001)), log)
    original_open = Path.open
    opens = []

    def guarded_open(path, *args, **kwargs):
        if path == log:
            opens.append(path)
            assert len(opens) == 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    document = seed.compile_seed_provenance([entry], log, cache)
    assert len(opens) == 1
    assert document['payload']['log_rows'] == 2001
    assert len(document['payload']['rows']) == 1
    assert document['payload']['rows'][0]['origin']['line'] == 2001


def test_default_history_never_reads_ignored_cache(tmp_path, monkeypatch):
    poison = tmp_path / 'seed_build_log.jsonl'
    poison.write_text('not a receipt\n')
    monkeypatch.setattr(seed, 'BUILD_LOG_PATH', poison)
    assert seed.read_build_log()


def test_builder_leaves_existing_bundle_unchanged_on_unsupported_input(original_inputs, tmp_path):
    entry, log, cache, path, _ = original_inputs
    target, provenance = tmp_path / 'seed.toml', tmp_path / 'seed.json'
    target.write_text('existing seed')
    provenance.write_text('existing provenance')
    path.unlink()
    with pytest.raises(FileNotFoundError):
        write_seed_bundle([entry], '2026-10-09', seed_path=target, provenance_path=provenance,
                          log_path=log, cache_root=cache)
    assert target.read_text() == 'existing seed'
    assert provenance.read_text() == 'existing provenance'
