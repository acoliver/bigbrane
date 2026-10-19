"""Small offline source fixtures and explicit persisted delta assertions."""
import json
from pathlib import Path
import socket
import pytest
from build_delta import (HERE, V1, Reader, digest, extract, identities, load,
                         pending_count, resolve_descriptors, scan_root, sha, verify_map)
from registry_tools import compose, excluded, public_only, ready, read_registry, check_stage


def test_known_scope_counts():
    assert [load(V1 / (n + '.v1.json'))['logical_count'] for n in
            ['original_pa2_11', 'historical_recovery_19', 'seen_seed_109']] == [11, 19, 109]


def test_probe_excluded_and_registry_pending():
    pool, delta = read_registry(HERE)
    assert len(delta['probe_entries']) == 12
    assert all(excluded(pool, row['identity']) for row in delta['probe_entries'])
    assert not ready(delta)
    assert delta['unknown']['historical_recovery_19']['remaining_ambiguity_records'] == 91
    assert delta['unknown']['other_used_study_data']['remaining_ambiguity_records'] == 2634


def test_duplicate_generated_does_not_add(tmp_path):
    obj = {'id': 'card-invented', 'title': 'Invented fixture mechanism',
           'core_idea': 'fixture', 'mechanism': {},
           'references': [{'title': 'Invented fixture publication', 'arxiv_id': '9999.12345'}]}
    p = tmp_path / 'card.json'
    p.write_text(json.dumps(obj))
    copy = tmp_path / 'copy.json'
    copy.write_bytes(p.read_bytes())
    base = [{'identity': {'title': obj['title']}}, {'identity': obj['references'][0]}]
    pool = compose(base, [])
    count = len(pool.items)
    calls = []
    def add(item, *args):
        calls.append(item)
        pool.add(item)
    receipt = scan_root(tmp_path, Reader(), {sha(p): ['fixture/original-card.json']}, add)
    assert len(pool.items) == count == 2
    assert len(calls) == 4
    assert len(receipt['copy_groups']) == 1
    assert len(receipt['copy_groups'][sha(p)]['selected']) == 2


def test_unresolved_stays_pending():
    assert pending_count(['a', 'b'], ['a']) == 1
    assert not ready({'full_known_repo_scope_complete': False, 'exhaustive_aliases': False,
                      'future_acquisition_ready': False, 'status': 'PENDING'})


def test_source_tamper_rejected(tmp_path):
    p = tmp_path / 'source.json'
    p.write_text('{}')
    expected = {str(p): sha(p)}
    verify_map(expected)
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='source SHA mismatch'):
        verify_map(expected)


def test_stage_tamper_rejected(tmp_path):
    p = tmp_path / 'STAGE.v2.json'
    p.write_text('{}')
    (tmp_path / 'STAGE.v2.sha256').write_text(sha(p))
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='v2 stage pin mismatch'):
        check_stage(tmp_path)


def test_relative_explicit_identifier_not_author_year(tmp_path):
    context = {'name': 'Named family (9999.12345)', 'relation': 'x = A x', 'claimed_difference': 'fixture'}
    p = tmp_path / 'card.json'
    p.write_text(json.dumps({'known_relatives': [context, {'name': 'Someone 1997', 'relation': 'different'}]}))
    paper = {'identity': {'title': 'A Completely Specified Fixture Publication', 'arxiv_id': '9999.12345'},
             'normalized_aliases': ['arxiv:9999.12345'], 'record_id': 'fixture-record'}
    scope = {'unresolved': [{'unresolved_id': str(j), 'descriptor': text, 'field': 'known_relatives', 'sources': [str(p)]}
                           for j, text in enumerate([context['name'], 'Someone 1997'])]}
    links, prose = resolve_descriptors({'fixture': scope}, Reader(), [paper])
    assert len(links) == 1 and links[0]['unresolved_id'] == '0'
    assert links[0]['links'][0]['context_sha256'] == digest(context)
    assert all(not r['paper_merge'] for r in prose)


def test_ledger_stream_once(tmp_path):
    p = tmp_path / 'ledger.jsonl'
    p.write_text(json.dumps({'card_id': 'card-fixture', 'artifact_path': 'cards/card-fixture/v1.json'}) + '\n')
    receipt = scan_root(tmp_path, Reader(), {}, lambda *args: None)
    assert receipt['ledgers'][0]['passes'] == 1
    assert receipt['ledgers'][0]['rows'] == 1
    assert receipt['ledgers'][0]['card_ids'] == ['card-fixture']


def test_non_card_object_not_mechanism():
    assert extract({'id': {}, 'title': 'Third-party schema', 'core_idea': '', 'mechanism': {}})[1] == []


def test_public_projection_no_secret_fields():
    papers, _ = extract({'papers': [{'title': 'Invented public title', 'arxiv_id': '9999.12345',
                                     'abstract': 'not copied', 'api_key': 'fixture-secret'}]})
    assert 'api_key' not in papers[0][0] and 'abstract' not in papers[0][0]
    assert public_only(papers[0][0])
    with pytest.raises(ValueError, match='secret field'):
        public_only({'api_key': 'fixture'})


def test_mitchell_title_binding():
    named = load(HERE / 'named-gaps.json')
    record = named['mitchell']
    binding = record['binding']
    lines = Path(binding['path']).read_text().splitlines()
    span = '\n'.join(lines[binding['line_start'] - 1:binding['line_end']])
    assert 'lecture slides for textbook' in span and 'Machine Learning' in span
    assert sha(binding['path']) == binding['sha256']
    assert set(record['identity']) == {'title'}
    assert not named['pass46dbb349']['closest_paper_invented']


def test_preservation_receipts():
    before, after = (load(HERE / ('preservation-' + s + '.json')) for s in ['before', 'after'])
    assert before['replay_inputs'] == after['replay_inputs'] == 274
    assert before['seed_inputs'] == after['seed_inputs'] == 75
    assert before['unique_paths'] == after['unique_paths'] == 349
    assert after['v1_unchanged'] and after['all_match']


@pytest.mark.parametrize('operation', [lambda: socket.socket(), lambda: socket.getaddrinfo('example.invalid', 443),
                                      lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)])
def test_network_denied(operation):
    with pytest.raises(RuntimeError, match='Network prohibited'):
        operation()


def test_special_files_not_opened(tmp_path):
    import os
    fifo = tmp_path / 'special.json'
    os.mkfifo(fifo)
    reader = Reader()
    receipt = scan_root(tmp_path, reader, {}, lambda *args: None)
    assert str(fifo) in receipt['special_files_excluded']
    assert not reader.records


def test_prose_alias_is_not_paper_union():
    from registry_tools import prose_excluded
    rows = load(HERE / 'descriptor-prose-aliases.json')
    assert rows and all(r['classification'] == 'prose_alias_not_paper_title' and not r['paper_merge'] for r in rows)
    assert prose_excluded(HERE, rows[0]['normalized_descriptor'])
    assert not prose_excluded(HERE, 'Invented never-seen fixture descriptor')
