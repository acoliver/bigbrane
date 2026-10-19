"""Compact inventory checks use metadata only; synthetic negatives never edit originals."""
import copy
import json
from pathlib import Path
import socket
import pytest
import build_inventory as b


def load(name):
    return b.read_json(b.HERE / name)


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def test_required_logical_counts_and_historical_ids():
    pa = load('original_pa2_11.v1.json')
    h = load('historical_recovery_19.v1.json')
    s = load('seen_seed_109.v1.json')
    assert len(pa['mechanisms']) == len({m['registration_id'] for m in pa['mechanisms']}) == 11
    assert {m['cid'] for m in h['mechanisms']} == b.HISTORICAL
    assert len(h['mechanisms']) == 19
    assert len(s['mechanisms']) == len({m['mechanism_id'] for m in s['mechanisms']}) == 109
    assert all(m['identity'].get('title') and m['semantic_label'] and m['receipts'] for m in s['mechanisms'])
    assert all(m['bound_card_sha256'] and m['bound_critic_sha256'] and m['card_versions'] and m['exact_bound_version_matches'] > 0 for m in h['mechanisms'])
    used = load('other_used_study_data.v1.json')
    assert b.PASSES <= {m['mechanism_id'] for m in used['mechanisms']}
    assert b.PASSES.isdisjoint({m['cid'] for m in h['mechanisms']})


def test_positive_exclusion_for_every_original_paper_identity():
    registry = load('exclusions.v1.json')
    excluded = set().union(*(b.identities(e['identity']) for e in registry['entries']))
    for scope in b.SCOPES:
        inv = load(scope + '.v1.json')
        for paper in inv['papers']:
            assert set(paper['normalized_aliases']) == b.identities(paper['identity'])
            assert b.identities(paper['identity']) <= excluded
            assert paper['evidence']
        for e in registry['entries']:
            for alias in b.identities(e['identity']):
                if alias.startswith('title:'):
                    # NFKC/case/punctuation normalization preserves the title token.
                    title = alias[6:]
                    assert b.identities({'title': title.upper() + ' . !'}) <= excluded
                elif alias.startswith('doi:'):
                    assert b.identities({'doi': 'HTTPS://DOI.ORG/' + alias[4:].upper()}) <= excluded
                elif alias.startswith('arxiv:'):
                    assert b.identities({'arxiv_id': 'HTTPS://ARXIV.ORG/abs/' + alias[6:] + 'v9'}) <= excluded


def test_synthetic_alias_versions_urls_and_title_unicode():
    item = {'title': 'Ａ Synthetic: Paper!', 'arxiv_id': '2601.12345v2', 'doi': '10.9999/Test',
            'source': 'openalex', 'source_id': 'W123456789'}
    ids = b.identities(item)
    assert 'title:asyntheticpaper' in ids
    assert b.identities({'arxiv_id':'https://arxiv.org/pdf/2601.12345v99.pdf'}) <= ids
    assert b.identities({'doi':'https://dx.doi.org/10.9999/TEST'}) <= ids
    assert b.identities({'source':'openalex','source_id':'https://openalex.org/w123456789'}) <= ids
    p = b.Pool(); p.add(item)
    p.add({'doi':'DOI:10.9999/test'})
    p.add({'arxiv_id':'arxiv:2601.12345v3'})
    assert len(p.items) == 1


def test_doi_arxiv_bridge_and_cross_scope_union():
    p = b.Pool()
    p.add({'arxiv_id':'2601.12345'})
    p.add({'doi':'10.48550/arXiv.2601.12345v2'})
    assert len(p.items) == 1
    registry = load('exclusions.v1.json'); p = b.Pool()
    for e in registry['entries']:
        p.add(e['identity'])
    unions = load('alias-unions.v1.json')
    assert len(p.items) == unions['union_count']
    assert len(p.owners) == unions['unique_alias_count']
    assert len(p.items) <= sum(load(s+'.v1.json')['identity_union_count'] for s in b.SCOPES)


def test_unknown_source_is_not_guessed_or_author_merged():
    assert b.bare_identity('unknown:123') is None
    assert b.bare_identity('Wunknown') is None
    assert b.bare_identity('Smith 1992') is None
    assert b.bare_identity('a'*40) is None
    assert b.paper_identity({'source':'unknown','source_id':'123'}) is None
    with pytest.raises(ValueError, match='unknown source'):
        b.identities({'source':'unknown','source_id':'123'})
    p = b.Pool()
    p.add({'title':'Synthetic One'})
    p.add({'title':'Synthetic Two'})
    assert len(p.items) == 2


def test_named_scopes_validate_but_broader_scope_stays_pending():
    result = b.validate_stage()
    assert result['status'] == 'PENDING'
    assert 'other_used_study_data' in result['incomplete_exclusions']
    roots = load('root-manifest.v1.json')
    assert roots['unknown_roots']
    assert all(row.get('reason') for row in roots['roots'])
    assert load('preservation-after.json')['all_unchanged']
    assert load('preservation-after.json')['replay_inputs'] == 274
    assert load('preservation-after.json')['seed_inputs'] == 75


def test_pending_inventories_cannot_pass_with_empty_corpus():
    registry = load('exclusions.v1.json')
    for scope in b.SCOPES:
        r = copy.deepcopy(registry)
        r['scopes'][scope]['status'] = 'pending'
        gate = b.corpus_gate(b.read_json(b.HELD/'corpus.json'), r)
        assert gate['status'] == 'PENDING'
        assert scope in gate['incomplete_exclusions']


def synthetic_stage(tmp_path):
    source = tmp_path/'original.txt'; source.write_text('synthetic original\n')
    registry = {'entries':[], 'scopes':{s:{'status':'pending','inventory_sha256':b.digest([]),'reason':'synthetic missing evidence'} for s in b.SCOPES}, 'unknown_aliases':[]}
    dump(tmp_path/'exclusions.v1.json',registry)
    (tmp_path/'exclusions.v1.sha256').write_text(b.file_sha(tmp_path/'exclusions.v1.json')+'  exclusions.v1.json\n')
    names = ['exclusions.v1.json','exclusions.v1.sha256','source-hashes.v1.json']
    dump(tmp_path/'source-hashes.v1.json', {'files':{str(source):{'sha256':b.file_sha(source)}}})
    for scope in b.SCOPES:
        name = scope+'.v1.json';names.append(name)
        dump(tmp_path/name, {'scope':scope,'status':'pending','mechanisms':[],'papers':[]})
    dump(tmp_path/'STAGE.v1.json',{'artifacts':{n:b.file_sha(tmp_path/n) for n in names},'frozen_base':{}})
    (tmp_path/'STAGE.v1.sha256').write_text(b.file_sha(tmp_path/'STAGE.v1.json')+'  STAGE.v1.json\n')
    return source


def test_synthetic_valid_pending_stage(tmp_path):
    synthetic_stage(tmp_path)
    assert b.validate_stage(tmp_path,sources=True)['status']=='PENDING'


@pytest.mark.parametrize('target',['exclusions.v1.json','seen_seed_109.v1.json','STAGE.v1.json','source-hashes.v1.json'])
def test_artifact_tampering_rejected(tmp_path,target):
    synthetic_stage(tmp_path)
    with (tmp_path/target).open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError,match='SHA mismatch'):
        b.validate_stage(tmp_path)


def test_source_tamper_rejected_without_rebuilding_body(tmp_path):
    source = synthetic_stage(tmp_path)
    source.write_text('tampered')
    with pytest.raises(ValueError,match='source SHA mismatch'):
        b.validate_stage(tmp_path,sources=True)


def test_missing_inventory_rejected(tmp_path):
    synthetic_stage(tmp_path)
    (tmp_path/'historical_recovery_19.v1.json').unlink()
    with pytest.raises(FileNotFoundError):
        b.validate_stage(tmp_path)


def test_registry_scope_missing_rejected():
    r = load('exclusions.v1.json');del r['scopes'][b.SCOPES[0]]
    with pytest.raises(ValueError,match='missing exclusion scope'):
        b.corpus_gate(b.read_json(b.HELD/'corpus.json'),r)


def test_hash_changed_complete_scope_rejected():
    r = load('exclusions.v1.json')
    r['scopes']['seen_seed_109']['inventory_sha256']='0'*64
    with pytest.raises(ValueError,match='inventory SHA mismatch'):
        b.corpus_gate(b.read_json(b.HELD/'corpus.json'),r)


def test_synthetic_bibliography_preserves_unknowns():
    found=[];missing=[]
    b.inspect_reference_fields({'references':[{'source_id':'unknown:77'}],
       'known_relatives':[{'name':'Doe and Smith (1992)'}]},lambda *x:found.append(x),missing,'synthetic')
    assert not found and len(missing)==2


@pytest.mark.parametrize('operation',[lambda:socket.getaddrinfo('example.invalid',443),
 lambda:socket.socket(socket.AF_INET,socket.SOCK_DGRAM),lambda:socket.socket(socket.AF_INET,socket.SOCK_STREAM)])
def test_network_dns_and_datagrams_are_prohibited(operation):
    with pytest.raises(RuntimeError,match='Network prohibited'):
        operation()


def test_source_qualified_publication_title_coverage():
    bindings=load('source-title-bindings.v1.json')['bindings']
    registry=load('exclusions.v1.json')
    excluded=set().union(*(b.identities(e['identity']) for e in registry['entries']))
    assert len(bindings)==22
    for row in bindings:
        assert b.identities({'title':row['title']}) <= excluded


def test_all_supported_public_source_ids_positive_exclusions():
    registry=load('exclusions.v1.json')
    excluded=set().union(*(b.identities(e['identity']) for e in registry['entries']))
    for alias in excluded:
        if alias.startswith('openalex:'):
            sid=alias.split(':',1)[1]
            assert b.identities({'source':'openalex','source_id':'https://openalex.org/'+sid.lower()}) <= excluded
        elif alias.startswith('s2:'):
            assert b.identities({'source':'semantic_scholar','source_id':alias[3:].upper()}) <= excluded


def test_frozen_harness_url_spelling_limits_are_visible():
    with pytest.raises(ValueError, match='bad arxiv ID'):
        b.identities({'arxiv_id':'https://arxiv.org/pdf/2601.12345v2.PDF'})
    with pytest.raises(ValueError, match='bad OpenAlex ID'):
        b.identities({'source':'openalex','source_id':'HTTPS://OPENALEX.ORG/W123'})


def test_partial_scope_blocks_even_a_populated_future_corpus():
    registry=load('exclusions.v1.json')
    metadata={'concept':'invented synthetic concept','lenses':[{'name':'lens'+str(i),
               'phrases':['phrase'+str(i)+'a','phrase'+str(i)+'b','phrase'+str(i)+'c']} for i in range(4)]}
    corpus={'status':'frozen','reason':'synthetic only','vault_sha256':'a'*64,
        'cards':[{'card_id':'synthetic-'+str(i),'family':'family'+str(i//3),'metadata':metadata,
            'mechanism_identity':{'title':'Invented synthetic unit '+str(i)},
            'reference_identities':[{'title':'Invented synthetic reference '+str(i)}],
            'provenance_sha256':'b'*64} for i in range(12)]}
    assert b.corpus_gate(corpus,registry)['status']=='PENDING'


def test_test_runner_is_cache_free_and_never_reconstructs_missing_response_bodies():
    # Compact replay checks provenance links against the shipped source-hash manifest only.
    hashes=load('source-hashes.v1.json')['files']
    for scope in b.SCOPES:
        for paper in load(scope+'.v1.json')['papers']:
            for evidence in paper['evidence']:
                assert hashes[evidence['path']]['sha256']==evidence['sha256']


def test_semantic_labels_normalize_the_exact_exported_source_label():
    for scope in b.SCOPES:
        for mech in load(scope+'.v1.json')['mechanisms']:
            if 'semantic_label' in mech:
                assert mech['normalized_semantic_label']==b.norm_title(mech['semantic_label'])


@pytest.mark.parametrize('scope',b.SCOPES[:3])
def test_partial_logical_inventory_cannot_claim_complete(scope):
    inv=load(scope+'.v1.json');inv['mechanisms']=inv['mechanisms'][:-1]
    with pytest.raises(ValueError):
        b.validate_inventory(inv,scope)


def test_missing_seed_source_paper_is_rejected():
    inv=load('seen_seed_109.v1.json');inv['papers']=[]
    with pytest.raises(ValueError,match='missing source paper'):
        b.validate_inventory(inv,'seen_seed_109')


def test_frozen_base_tampering_rejected_without_editing_real_plan(tmp_path):
    synthetic_stage(tmp_path)
    frozen=tmp_path/'synthetic-plan.txt';frozen.write_text('synthetic frozen plan')
    stage=b.read_json(tmp_path/'STAGE.v1.json')
    stage['frozen_base']={str(frozen):b.file_sha(frozen)}
    dump(tmp_path/'STAGE.v1.json',stage)
    (tmp_path/'STAGE.v1.sha256').write_text(b.file_sha(tmp_path/'STAGE.v1.json')+'  STAGE.v1.json\n')
    frozen.write_text('changed')
    with pytest.raises(ValueError,match='frozen base SHA mismatch'):
        b.validate_stage(tmp_path)


def test_logical_registrations_survive_duplicate_underlying_paper():
    inv=load('original_pa2_11.v1.json')
    first=inv['papers'][0]
    inv['papers']=[first]
    for mechanism in inv['mechanisms']:
        mechanism['paper_record_id']=first['record_id']
    b.validate_inventory(inv,'original_pa2_11')
    assert len({m['registration_id'] for m in inv['mechanisms']})==11
