"""Regression checks for saved extraction and source-only request recording."""
import importlib.util
import json
from pathlib import Path
import sqlite3

import httpx
import pytest

PATH = Path(__file__).with_name('diagnose.py')
spec = importlib.util.spec_from_file_location('diagnose', PATH)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_saved_cache_miss_cannot_fall_through_to_network(tmp_path):
    client = diag.SavedLiterature(tmp_path)
    with pytest.raises(FileNotFoundError):
        client._fetch('arxiv', diag.sources.arxiv.get_request('1110.5667'))


def test_receipt_and_cache_identity_must_agree(tmp_path):
    client = diag.SavedLiterature(tmp_path)
    req = diag.sources.semantic_scholar.search_request('original query', 6)
    cache = client.cache
    cache.put('s2', diag.request_key('s2', req.url, req.params), {
        'body': json.dumps({'data': [{'paperId': 'unexpected', 'title': 'Other paper'}]}),
        'fetched_at': '2026-10-18T01:00:00+00:00'})
    receipt = {'sources': ['s2'], 'errors': {}, 'per_source_ids': {'s2': ['expected']},
               'query': 'original query', 'limit': 6, 'result_count': 1}
    with pytest.raises(ValueError, match='cache differs from saved receipt'):
        diag.reconstruct_search(client, receipt, None, '2026-10-18T02:00:00+00:00')


def test_later_cache_cannot_supply_historical_evidence(tmp_path):
    client = diag.SavedLiterature(tmp_path)
    req = diag.sources.semantic_scholar.search_request('original query', 6)
    client.cache.put('s2', diag.request_key('s2', req.url, req.params), {
        'body': json.dumps({'total': 0}), 'fetched_at': '2026-10-19T01:00:00+00:00'})
    receipt = {'sources': ['s2'], 'errors': {}, 'per_source_ids': {'s2': []},
               'query': 'original query', 'limit': 6, 'result_count': 0}
    with pytest.raises(ValueError, match='postdates saved search'):
        diag.reconstruct_search(client, receipt, None, '2026-10-18T02:00:00+00:00')


def test_recording_excludes_headers_and_redacts_values(tmp_path):
    from hybrid.literature.keys import ApiKey
    key = ApiKey('s2', 'diagnostic-secret-value', 'test')
    path = tmp_path / 'http.jsonl'
    recorder = diag.Recorder(path, {'s2': key})
    recorder.label = 'targeted-only'
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(
        429, headers={'Retry-After': '5', 'Location': 'https://example.org/' + key.value})),
        event_hooks={'request': [recorder.request], 'response': [recorder.response]}) as http:
        http.get('https://example.org/paper', headers=key.headers)
    text = path.read_text()
    assert key.value not in text
    assert 'x-api-key' not in text
    value = json.loads(text)
    assert value['status'] == 429
    assert value['retry_after'] == '5'
    assert value['diagnostic'] is True and value['acceptance_input'] is False


def test_local_membership_distinguishes_term_miss_from_selection_limit():
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.executescript('''
        CREATE TABLE papers(id INTEGER PRIMARY KEY, arxiv_id TEXT, title TEXT, abstract TEXT, categories TEXT, doi TEXT);
        CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract);
        INSERT INTO papers VALUES(1, '1110.5667', 'Bayesian programs', 'induction', 'cs.AI', NULL);
        INSERT INTO papers_fts(rowid,title,abstract) VALUES(1,'Bayesian programs','induction');
    ''')
    result = diag.fts_target_evidence(db, {'Example': ('1110.5667', 'Bayesian programs')},
                                      ['Bayesian DreamCoder'])['Example']
    assert result['indexed'] is True
    assert result['queries'][0]['target_membership'] == [False, True]
    assert result['queries'][0]['any_term_fallback_reached'] is True
    db.close()


def test_saved_replay_retains_failed_acceptance_and_actual_exposure():
    value = json.loads(PATH.with_name('replay-evidence.json').read_text())
    assert value['acceptance'] == {'retrieved_cards': 2, 'denominator': 4, 'required': 3, 'status': 'FAIL'}
    assert value['http_attempts'] is None
    assert all(c['historical_rank_reproduced'] for c in value['cards'].values())
    matches = {label: item for card in value['cards'].values() for label, item in card['matches'].items()}
    assert len(matches) == 11
    assert sum(bool(v['old']) for v in matches.values()) == 3
    assert sum(bool(v['new']) for v in matches.values()) == 3
    assert matches['Rossi']['old'][0]['exposures'] == []
    assert matches['Zhou']['old'][0]['exposures'] == []
    assert matches['Sacramento']['refinement_title_positions'] == [21]
    assert sum(any(hit['positions']['60'] is not None for hit in v['new']) for v in matches.values()) == 1
    assert sum(bool(v['all_prompt_mentions']) for v in matches.values()) == 1


def test_targeted_diagnostics_never_enter_replay_inputs():
    value = json.loads(PATH.with_name('replay-evidence.json').read_text())
    assert not any('retrieval-diagnosis' in path for path in value['inputs_sha256'])
    for card in value['cards'].values():
        for search in card['searches']:
            for records in search['source_records'].values():
                assert len(records) <= search['limit'] == 6


def test_raw_api_record_boundaries_exclude_abstract_word_dictionaries():
    spec = importlib.util.spec_from_file_location('raw_audit_test', PATH.with_name('audit_raw.py'))
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    item = {'id': 'https://openalex.org/W1', 'title': 'A paper',
            'abstract_inverted_index': {'title': [0], 'paper': [1]}}
    assert raw.raw_records('openalex', {'results': [item]}) == [item]
    edge = {'paperId': 'abc', 'title': 'Another paper'}
    assert raw.raw_records('s2', {'data': [{'citedPaper': edge}]}) == [edge]


def test_raw_audit_does_not_count_bibliography_ids_as_retrieved_papers():
    value = json.loads(PATH.with_name('raw-audit.json').read_text())
    assert value['missing_title_or_id_counts'] == {}
    assert [r['matches'] for r in value['records']] == [['Rossi']]
    assert len(value['pre_reference_cap_matches']) == 14


def test_hwang_bibliography_leads_were_outside_the_original_seed_bound():
    value = json.loads(PATH.with_name('raw-audit.json').read_text())
    leads = [r for r in value['pre_reference_cap_matches'] if 'Hwang' in r['known_work_matches']]
    assert {r['source_work'] for r in leads} == {'W3174357804', 'W4310557242'}
    assert {rank for r in leads for c in r['contexts']
            for rank, _ in c['initial_rank_at_seed_selection']} == {65, 66}
    assert all(not c['selected_seed_keys'] for r in leads for c in r['contexts'])


def test_saved_arxiv_replay_pins_historical_http_cache_without_fallback(tmp_path):
    client = diag.SavedLiterature(tmp_path)
    request = diag.sources.arxiv.search_request('original query', 6)
    assert request.url == 'https://export.arxiv.org/api/query'
    historic_url = 'http://export.arxiv.org/api/query'
    historic_key = diag.request_key('arxiv', historic_url, request.params)
    historic_path = client.cache.path_for('arxiv', historic_key)
    client.cache.put('arxiv', historic_key, {
        'body': 'original saved response', 'fetched_at': '2026-10-18T01:00:00+00:00'})
    client.cache.put('arxiv', diag.request_key('arxiv', request.url, request.params), {
        'body': 'new diagnostic response', 'fetched_at': '2026-10-19T01:00:00+00:00'})
    original_bytes = historic_path.read_bytes()
    fetched = client._fetch('arxiv', request)
    assert fetched.url == historic_url
    assert fetched.body == 'original saved response'
    assert client.inputs == {str(historic_path): diag.digest(historic_path)}
    assert historic_path.read_bytes() == original_bytes
    historic_path.unlink()
    with pytest.raises(FileNotFoundError):
        client._fetch('arxiv', request)


def test_packaged_measurements_retain_source_hashes_and_exact_projection():
    import hashlib
    spec = importlib.util.spec_from_file_location('package_replay_evidence',
                                                PATH.with_name('package_replay_evidence.py'))
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    packaged = PATH.with_name('replay-evidence.json')
    checksums = json.loads(PATH.with_name('PACKAGED-SHA256.json').read_text())
    for name, expected in checksums.items():
        assert hashlib.sha256(PATH.with_name(name).read_bytes()).hexdigest() == expected
    value = json.loads(packaged.read_text())
    assert value['provenance']['method'] == 'field projection only; no replay or rescoring'
    assert len(bytes.fromhex(value['provenance']['sha256'])) == 32
    assert len(value['inputs_sha256']) == 274
    for card in value['cards'].values():
        for search in card['searches']:
            for source, records in search['source_records'].items():
                assert [record['source_id'] for record in records] == search['per_source_ids'][source]
    original = PATH.with_name('offline.json')
    if original.exists():
        assert package.project(original) == value
