"""Full source observations and negative witnesses never imply a full composite admission."""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hybrid.tests.research.phase7_replay import load_fixture
from hybrid.tests.research.composites_replays import (
    ROOT, FIXTURES, CARDS, load, digest, full_capture, evidence_for, run_original,
    original_definitions, lattice_closure, dense_binding_sweeps, pinned_one_pair, superglue_one_pair,
)
from hybrid.tests.research.reference_replays import historical_phase7_evidence, next_historical_phase7_evidence


def normalize(value):
    return json.loads(json.dumps(value, default=lambda x:x.tolist()))


def entry(card):
    return next(e for e in load_fixture()['cards'] if e['card_id']==card)


@pytest.mark.parametrize('card',CARDS)
def test_every_original_source_draw_state_iteration_and_output_reexecutes_without_raw_cache(card):
    observed=full_capture()[card]; expected=load('sources')['cards'][card]
    assert observed['provenance']==expected['provenance']
    assert digest(observed['episodes'])==expected['provenance']['trace_sha256']
    assert digest([{'parameters':e['parameters'],'inputs':e['inputs']} for e in observed['episodes']])==expected['provenance']['draws_sha256']
    counts=load('evidence')['measurements']['capture_counts'][card]
    assert len(observed['episodes'])==counts['episodes']
    assert sum(len(e['inputs']) for e in observed['episodes'])==counts['tokens']
    assert observed['provenance']['capture_insertions']==[]
    assert 'not recovered historical bytes' in observed['provenance']['draw_limit']


@pytest.mark.parametrize('card',CARDS)
def test_complete_original_card_critic_and_executor_receipts_have_hash_bound_content(card):
    specimen=load('specimens')[card]; p=load('sources')['cards'][card]['provenance']
    assert all(digest(specimen[k])==p['specimen'][f'{k}_sha256'] for k in ('card','critic'))
    assert specimen['card']==entry(card)['card'] and specimen['critic']==entry(card)['critic']
    assert set(specimen['original_receipts'])=={'card_version','reduction_critic'}
    for receipt in specimen['original_receipts'].values():
        assert receipt['sha256']==receipt['ledger_event']['data']['sha256']
        path=Path(receipt['path'])
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    assert p['purpose']=='reduction_check'
    assert p['artifact_sha256']==p['ledger_event']['data']['sha256']
    for key,hashkey in (('source','source_sha256'),('original_source','source_sha256'),('artifact','artifact_sha256')):
        path=ROOT/p[key] if key=='source' else Path(p[key])
        if key=='source' or path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==p[hashkey]


@pytest.mark.parametrize('card',CARDS)
def test_compact_episodes_are_source_bound_and_not_comparator_results(card):
    e=load('evidence')['cards'][card]
    for i,episode in zip(e['selected_episode_indices'],e['episodes']):
        assert episode==full_capture()[card]['episodes'][i]
        outputs,states=run_original(card,episode) if card!=CARDS[2] else (run_original(card,episode)[0],None)
        if card==CARDS[0]:
            assert normalize(outputs)==episode['card_outputs']
            assert normalize(states)==episode['states']
        elif card==CARDS[1]:
            for got,want in zip(outputs,episode['card_outputs']):
                assert got[0]==want[0] and got[2:]==tuple(want[2:])
                np.testing.assert_array_equal(got[1],want[1])
            assert normalize(states)==episode['states']
        else:
            np.testing.assert_array_equal(outputs,episode['card_outputs'])


@pytest.mark.parametrize('card',CARDS)
def test_diagnostics_cannot_admit_changed_specimens_or_depend_on_expected_kill_codes(card):
    original=entry(card); e=evidence_for(original)
    assert e['accepted_reference'] is False and e['missing_full_published_operation'] is True
    assert 'declaration' not in e and 'reference_spec' not in e
    assert historical_phase7_evidence(original) is None and next_historical_phase7_evidence(original) is None
    for field,key in (('card','summary'),('critic','mapping')):
        changed=copy.deepcopy(original);changed[field][key]+=' mutation'
        assert evidence_for(changed) is None
    changed=copy.deepcopy(original);changed['original']['kill_codes']=['invented'];changed['max_abs_diff']=-3.
    assert evidence_for(changed)==e


def test_every_gcd_stream_keeps_full_fifo_schedule_multiclass_conflicts_and_readout():
    checks=readouts=0
    for e in full_capture()[CARDS[0]]['episodes']:
        records=[]; clock=0; compiled=(0,{},True)
        for token,state,outputs in zip(e['inputs'],e['states'],e['card_outputs']):
            if token[0]=='DEMO':
                records=(records+[token[1:]])[-e['parameters']['d']:]
                clock=(clock+1)%e['parameters']['B']
            elif token[0]=='COMMIT':
                clock=0
            assert state['E']==records and state['c']==clock
            if token[0]=='COMMIT' or token[0]=='DEMO' and clock==0:
                compiled=lattice_closure(records);checks+=1
            g,table,valid=compiled
            assert state['g']==g and state['v']==valid
            assert state['P']=={str(k):v for k,v in table.items()}
            for output in outputs:
                x=output['x']; matches=[y for z,y in records if z==x]
                chosen=matches[-1] if matches else table.get(x%g if g else x) if valid else None
                wanted=[float(i==chosen) for i in range(5)] if chosen is not None else [1/5]*5
                assert output['p']==wanted;readouts+=1
    assert checks==load('evidence')['measurements']['gcd_compilation_checks']==2037
    assert readouts==63981


def test_gcd_minimal_schedule_and_forgetting_witnesses_disagree_with_all_seen_closure():
    cases=load('evidence')['handcases']['gcd']
    schedule=cases['schedule'];out,states=run_original(CARDS[0],schedule)
    assert next(o['p'] for o in out[2] if o['x']==12)==[.2]*5
    assert next(o['p'] for o in out[-1] if o['x']==12)==[1.,0.,0.,0.,0.]
    assert lattice_closure([[0,0],[6,0]])[0]==6
    eviction=cases['eviction'];out,states=run_original(CARDS[0],eviction)
    assert states[-1]['g']==4 and lattice_closure([[0,0],[6,0],[2,0]])[0]==2
    assert next(o['p'] for o in out[-1] if o['x']==8)==[.2]*5
    # Integer span of translations 6 and 2 contains 8; the retained {6,2} coset does not.
    assert 8%2==0 and (8-6)%4!=0
    conflict=cases['newest_conflict'];out,states=run_original(CARDS[0],conflict)
    assert out[3][0]['p']==[0.,1.,0.,0.,0.]
    assert states[-1]['v'] is False and states[-1]['P']=={}


@pytest.mark.parametrize('mutation',[
    ('s.E.pop(0)','s.E.pop()'),
    ('if s.c == 0:','if True:'),
    ('for a, y in reversed(s.E):','for a, y in s.E:'),
])
def test_gcd_eviction_clock_and_recency_mutants_change_witness_outputs(mutation):
    cases=load('evidence')['handcases']['gcd']
    assert any(normalize(run_original(CARDS[0],e,mutation)[0])!=e['card_outputs'] for e in cases.values())


def test_all_original_omp_calls_obey_published_support_refit_and_coupled_value_equations():
    e=full_capture()[CARDS[1]]['episodes'][0];p=e['parameters']; prior={'K':[],'Y':[],'N':[]};iterations=0
    for token,pu,out,state in zip(e['inputs'],e['pursuits'],e['card_outputs'],e['states']):
        q=np.asarray(p['C'])[token[1]].sum(0);r=q.copy();selected=[]
        for step in pu['iterations']:
            scores=[abs(float(np.array(k)@r))/n if i not in selected else -1. for i,(k,n) in enumerate(zip(prior['K'],prior['N']))]
            idx=max(range(len(scores)),key=lambda i:scores[i])
            assert scores[idx]>p['tau'] and step['S']==selected+[idx]
            selected=step['S'];A=np.asarray(prior['K'])[selected].T;alpha=np.array(step['alpha']);r=q-A@alpha
            np.testing.assert_allclose(step['r'],r,atol=1e-14,rtol=1e-14)
            np.testing.assert_allclose(A.T@r,0.,atol=1e-12,rtol=0.)
            iterations+=1
        S,alpha,finalr=pu['result'];assert S==selected
        np.testing.assert_allclose(finalr,r,atol=1e-14,rtol=1e-14)
        prediction=sum(float(a)*prior['Y'][i] for i,a in zip(S,alpha))
        assert out[0]==prediction
        logits=-p['beta']*(np.array(p['E'])-prediction)**2
        probability=np.exp(logits-logits.max());probability/=probability.sum()
        np.testing.assert_allclose(np.exp(out[1]),probability,atol=1e-15,rtol=1e-13)
        wrote=token[0]=='DEMO' and (np.linalg.norm(r)>p['eps'] or abs(p['E'][token[2]]-prediction)>p['eps'])
        assert out[2]==wrote
        expected=copy.deepcopy(prior)
        if wrote:
            expected['K']=(expected['K']+[q.tolist()])[-p['d']:]
            expected['Y']=(expected['Y']+[float(p['E'][token[2]])])[-p['d']:]
            expected['N']=(expected['N']+[float(np.linalg.norm(q))])[-p['d']:]
        assert state==expected and out[3]==len(state['K']);prior=state
    assert iterations==187 and sum(o[2] for o in e['card_outputs'])==75


def test_signed_transfer_fifo_and_prediction_before_write_have_independent_hand_answers():
    e=load('evidence')['handcases']['omp_signed_fifo'];outs,states=run_original(CARDS[1],e)
    np.testing.assert_allclose([o[0] for o in outs],[0.,3.5,4.,0.,0.],atol=1e-14,rtol=0.)
    assert [o[2] for o in outs]==[True,True,False,True,False]
    assert states[-1]['Y']==[3.,5.]
    for mutation in [('self.K.pop(0); self.Y.pop(0); self.N.pop(0)','self.K.pop(); self.Y.pop(); self.N.pop()'),
                     ('alpha[j] * self.Y[i]','abs(alpha[j]) * self.Y[i]')]:
        mutant,_=run_original(CARDS[1],e,mutation)
        assert max(abs(a[0]-b[0]) for a,b in zip(outs,mutant))>1.


def test_repeat_input_output_error_allocates_where_published_ran_and_ald_do_not():
    e=load('evidence')['handcases']['omp_repeat_allocation'];out,states=run_original(CARDS[1],e)
    assert [o[2] for o in out]==[True,True,False]
    assert states[1]['Y']==[1.,2.]
    q=np.array(e['parameters']['C'][0]);center=q.copy()
    assert np.linalg.norm(q-center)==0.
    ran_allocates=np.linalg.norm(q-center)>1e-6 and abs(2.-out[1][0])>1e-6
    ald_squared_residual=np.linalg.norm(q-center)**2
    assert not ran_allocates and ald_squared_residual==0.
    # This is the allocation decision only, not a complete RAN/KRLS simulation.


@pytest.mark.parametrize('seed',[13,41])
def test_free_scalar_output_codes_feature_codes_and_temperature_are_not_frozen_for_handchecks(seed):
    rng=np.random.default_rng(seed);p={'m':4,'d':3,'s':2,'U':3,'V':4,'eps':1e-6,'tau':1e-9,
       'beta':float(rng.uniform(.2,3.)),'C':rng.normal(size=(3,4)).tolist(),'E':rng.normal(size=4).tolist()}
    e={'parameters':p,'inputs':[['DEMO',[0],1],['QUERY',[0],None]]};outs,_=run_original(CARDS[1],e)
    assert outs[1][0]==pytest.approx(p['E'][1],abs=1e-12)
    logits=-p['beta']*(np.array(p['E'])-p['E'][1])**2
    expected=np.exp(logits-logits.max());expected/=expected.sum()
    np.testing.assert_allclose(np.exp(outs[1][1]),expected,rtol=1e-12,atol=1e-14)


def test_dense_binding_matches_all_iterations_but_cold_restart_changes_stream_outputs():
    e=full_capture()[CARDS[2]]['episodes'][0]
    outputs,steps=dense_binding_sweeps(e['parameters'],e['states'],e['inputs'])
    np.testing.assert_allclose(outputs,e['card_outputs'],atol=1e-14,rtol=1e-13)
    assert len(steps)==60 and sum(map(len,steps))==180
    for observed,expected in zip(e['sweeps'],steps):
        for a,b in zip(observed,expected):
            for name in ('x','y'):
                np.testing.assert_allclose(a[name],b[name],atol=1e-14,rtol=1e-13)
    cold,_=dense_binding_sweeps(e['parameters'],e['states'],e['inputs'],cold=True)
    assert np.max(.5*np.sum(np.abs(cold-e['card_outputs']),axis=1))>.29


def test_full_binding_count_table_preserves_gate_lfu_increment_recency_and_injection():
    e=full_capture()[CARDS[2]]['episodes'][0];p=e['parameters'];base=np.array(p['E'])@np.array(p['F']).T
    table={};seen=set();evictions=gated=0
    for t,(w,state) in enumerate(zip(e['inputs'],e['states']),1):
        if t>1:
            key=(e['inputs'][t-2],w)
            if base[key]>p['kappa']:
                old=table.get(key,[0,0,base[key]])
                table[key]=[old[0]+1,t,old[2]]
                if len(table)>p['Pmax']:
                    victim=min(table,key=lambda k:(table[k][0],table[k][1]));del table[victim];evictions+=1
            else:
                gated+=1
        seen.add(w)
        assert state['a']==[float(i in seen) for i in range(p['V'])]
        assert [(r[:4]) for r in state['table']]==[[*key,*v[:2]] for key,v in sorted(table.items())]
        assert state['evictions']==evictions and state['gated_out']==gated
    assert (evictions,gated)==(29,7)


def test_superglue_prescribed_dustbin_marginals_differ_from_pinned_potentials_and_finite_sweeps():
    e=load('evidence')['handcases']['dustbin'];out,_,_=run_original(CARDS[2],e)
    # Starting x=1: three y-then-x sweeps give x=13/21 and x*y=8/21.
    np.testing.assert_allclose(out[0],[float(Fraction(15,28)),float(Fraction(13,28))],atol=1e-15,rtol=0.)
    balanced=superglue_one_pair(1.);pinned=pinned_one_pair(1.)
    np.testing.assert_array_equal(balanced,np.full((2,2),.5))
    np.testing.assert_allclose(balanced.sum(axis=0),[1,1],atol=0.,rtol=0.)
    assert pinned[1].sum()>1.6 and pinned[:,1].sum()>1.6
    matched=1-out[0][1]/e['parameters']['u'][1]
    assert abs(pinned[0,0]-.5)>.11 and abs(matched-.5)>.11
    assert abs(pinned[0,0]-matched)>1e-3
    assert abs(out[0][0]-(.5+.5*.25))>.08


@pytest.mark.parametrize('mutation',[
    ('(table[k][0], table[k][1])','(-table[k][0], table[k][1])'),
    ('for _ in range(J):','for _ in range(1):'),
    ('if k0 > kappa:','if True:'),
])
def test_lfu_iteration_and_gate_mutants_change_full_source_stream(mutation):
    e=full_capture()[CARDS[2]]['episodes'][0];outputs,_,_=run_original(CARDS[2],e,mutation)
    assert np.max(np.abs(outputs-e['card_outputs']))>1e-4


def test_original_omp_comparator_discrepancy_is_preserved_without_becoming_a_reference():
    diagnostic=load('evidence')['measurements']['original_omp_comparator']
    assert diagnostic['different_value_outputs']==1 and diagnostic['first_step_zero_based']==32
    first=diagnostic['first']
    assert first['card']['S']==[9,1] and first['known']['S']==[0,9]
    assert abs(first['card']['output']-first['known']['output'])>1.4
    assert abs(first['first_pass_scores'][0]-first['first_pass_scores'][9])<1e-15
    # The original code's floating-point tie behavior is retained, not repaired by a tolerance.


def test_publication_equation_snapshots_are_small_source_bound_and_locally_checkable():
    sources=load('publications');assert len(sources)==6
    for source in sources:
        assert source['equations'] and source['section'] and source['excerpt'] and source['retrieved_utc']
        assert source['url'].startswith('https://') and len(source['sha256'])==64
        for name,hashkey in (('file','sha256'),('text_file','text_sha256')):
            path=ROOT/source[name]
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest()==source[hashkey]


def test_protected_acceptance_code_math_tests_and_existing_controls_remain_unchanged():
    protected=load('protected');assert len(protected)>60
    for name,expected in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
