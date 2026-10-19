"""Whole-source diagnostics and narrow refutations, with no full-operation admission."""
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hybrid.tests.research.phase7_replay import load_fixture
from hybrid.tests.research.nonlinear_replays import (
    ROOT,FIXTURES,CARDS,load,digest,full_capture,evidence_for,complex_array,
    run_original,observe_handcase,scalar_closure,published_ppgn_block,dpmeans,fuzzy_art,
    closure_loss_gradient,coherence_loss_gradient,
)
from hybrid.tests.research.reference_replays import historical_phase7_evidence,next_historical_phase7_evidence


def hand(name):
    return load('evidence')['handcases'][name]


def entry(card):
    return next(e for e in load_fixture()['cards'] if e['card_id']==card)


@pytest.mark.parametrize('card',CARDS)
def test_all_original_draws_reexecute_without_reading_the_full_capture_cache(card):
    observed=full_capture()[card];p=observed['provenance'];expected=load('sources')['cards'][card]
    assert p==expected['provenance'] and p['purpose']=='reduction_check'
    assert digest(observed['episodes'])==p['trace_sha256']
    assert digest([{'parameters':e['parameters'],'inputs':e['inputs']} for e in observed['episodes']])==p['draws_sha256']
    counts=load('evidence')['measurements']['counts'][card]
    assert len(observed['episodes'])==counts['episodes']
    assert sum(len(e['inputs']) for e in observed['episodes'])==counts['tokens']
    assert p['capture_insertions']==[] and 'not recovered historical bytes' in p['draw_limit']


@pytest.mark.parametrize('card',CARDS)
def test_full_card_critic_job_ledger_artifact_and_program_bytes_are_bound(card):
    p=load('sources')['cards'][card]['provenance'];specimen=load('specimens')[card]
    for field in ('card','critic'):
        assert specimen[field]==entry(card)[field]
        assert digest(specimen[field])==p['specimen'][f'{field}_sha256']
    assert p['artifact_sha256']==p['ledger_event']['data']['sha256']
    assert p['ledger_event']['data']['card_id']==card
    for receipt in specimen['original_receipts'].values():
        assert receipt['sha256']==receipt['ledger_event']['data']['sha256']
        path=Path(receipt['path'])
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    for name,key in (('source','source_sha256'),('original_source','source_sha256'),('artifact','artifact_sha256'),('ledger_file','ledger_sha256')):
        path=ROOT/p[name] if name=='source' else Path(p[name])
        if name=='source' or path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==p[key]


@pytest.mark.parametrize('card',CARDS)
def test_compact_episodes_and_free_parameters_are_complete_source_specimens(card):
    evidence=load('evidence')['cards'][card]
    for i,episode in zip(evidence['selected_episode_indices'],evidence['episodes']):
        assert episode==full_capture()[card]['episodes'][i]
        result=run_original(card,episode)
        np.testing.assert_array_equal(result if card==CARDS[0] else result[0],episode['card_outputs'])
    parameters=[e['parameters'] for e in full_capture()[card]['episodes']]
    if card==CARDS[0]:
        assert len({p['alpha'] for p in parameters})==36 and len({p['beta'] for p in parameters})==36
    else:
        assert len({digest(p['Theta']) for p in parameters})==115
        assert len({p['tau'] for p in parameters})>2 and len({p['beta'] for p in parameters})>2
        assert any(any(u!=0 for u in p['u']) for p in parameters)


@pytest.mark.parametrize('card',CARDS)
def test_negative_evidence_never_admits_a_reference_or_borrows_changed_specimens(card):
    original=entry(card);e=evidence_for(original)
    assert e['accepted_reference'] is False and e['missing_full_published_operation'] is True
    assert 'declaration' not in e and 'reference_spec' not in e
    assert historical_phase7_evidence(original) is None and next_historical_phase7_evidence(original) is None
    for field,key in (('card','summary'),('critic','mapping')):
        changed=copy.deepcopy(original);changed[field][key]+=' mutation'
        assert evidence_for(changed) is None
    changed=copy.deepcopy(original);changed['original']['kill_codes']=['invented'];changed['max_abs_diff']=-4
    assert evidence_for(changed)==e


def test_every_pair_count_solver_iteration_and_pair_readout_matches_scalar_diagnostic():
    iterations=queries=0
    for episode in full_capture()[CARDS[0]]['episodes']:
        p=episode['parameters'];d=p['d'];E=np.zeros((d,d));depth=math.ceil(math.log2(d))+2
        for token,state,out in zip(episode['inputs'],episode['states'],episode['card_outputs']):
            if token<2*d*d:
                a,b,s=token//(2*d),(token//2)%d,token%2
                if a!=b:
                    E[a,b]+=1-2*s;E[b,a]+=1-2*s
            np.testing.assert_array_equal(state['E'],E)
            expected=[.5,.5]
            if 2*d*d<=token<3*d*d:
                queries+=1;a,b=divmod(token-2*d*d,d)
                if a==b:
                    expected=[1.,0.];assert not state['iterations']
                else:
                    H,trace=scalar_closure(E,p['alpha'],p['beta'],depth)
                    assert len(state['iterations'])==depth
                    prior=np.zeros((d,d))
                    for observed,scalar in zip(state['iterations'],trace):
                        np.testing.assert_allclose(observed['G'],(1-np.eye(d))*(prior@prior),atol=2e-14,rtol=1e-13)
                        np.testing.assert_allclose(observed['H'],scalar,atol=2e-14,rtol=1e-13)
                        assert np.all(np.diag(observed['H'])==0)
                        prior=scalar;iterations+=1
                    expected=[(1+H[a,b])/2,(1-H[a,b])/2]
            else:
                assert not state['iterations']
            np.testing.assert_allclose(out,expected,atol=1e-14,rtol=1e-12)
    assert (iterations,queries)==(1980,489)


def test_published_ppgn_block_uses_current_features_and_does_not_zero_its_diagonal():
    X=np.array([[0.,2.],[2.,0.]])[...,None]
    block=published_ppgn_block(X,lambda x:x,lambda x:x,lambda x:x)
    np.testing.assert_array_equal(block[:,:,0],X[:,:,0])
    np.testing.assert_array_equal(block[:,:,1],[[4.,0.],[0.,4.]])
    assert block.shape==(2,2,2)
    # The published invariant graph readout has no unmarked pair-query argument.
    e=hand('ppgn_pair_head');out=run_original(CARDS[0],e)
    assert out[1,0]==pytest.approx((1+math.tanh(.7))/2) and out[2,0]==.5
    assert e['states'][1]['E']==e['states'][2]['E']


def test_two_site_handcase_refutes_the_stated_single_channel_current_skip_substitution():
    e=hand('ppgn_isolated');out=run_original(CARDS[0],e);p=e['parameters'];E=np.array([[0.,1.],[1.,0.]])
    expected=(1+math.tanh(p['alpha']))/2
    assert out[-1,0]==pytest.approx(expected)
    # With m1=m2=id, m3=id and the stated m4, starting current X at zero never injects E.
    no_input,_=scalar_closure(E,p['alpha'],p['beta'],3,reinject=False)
    assert no_input[0,1]==0.
    # Grant a first embedding step, then use current X rather than the missing original-input skip.
    X=np.tanh(p['alpha']*E)[...,None]
    for _ in range(2):
        B=published_ppgn_block(X,lambda x:x,lambda x:x,lambda x:x)
        X=np.tanh(p['alpha']*B[:,:,:1]+p['beta']*B[:,:,1:])
    assert abs(float(X[0,1,0])-math.tanh(.7))>.01
    assert abs(float(X[0,0,0]))>.1


def test_three_site_path_is_a_minimal_undemonstrated_pair_counterexample_to_count_only_read():
    e=hand('ppgn_chain');out=run_original(CARDS[0],e)
    assert e['states'][-1]['E'][0][2]==0.
    assert out[-1,0]>.6
    # First two steps can be calculated without a matrix library.
    q=math.tanh(.7);second=math.tanh(1.1*q*q)
    assert e['states'][-1]['iterations'][1]['H'][0][2]==pytest.approx(second)


@pytest.mark.parametrize('name,mutation',[
 ('ppgn_chain',('beta * G','0.0 * G')),
 ('ppgn_chain',('M = 1.0 - np.eye(d)','M = np.ones((d, d))')),
 ('ppgn_chain',('E[b, a] += sig','E[b, a] += 0')),
])
def test_pair_solver_dropped_term_mask_and_write_mutants_change_outputs(name,mutation):
    e=hand(name);base=run_original(CARDS[0],e);mutant=run_original(CARDS[0],e,mutation)
    assert np.max(abs(base-mutant))>1e-3


def test_every_complex_update_selection_mass_eviction_pointer_and_readout_is_checked():
    merges=evictions=tokens=0
    for episode in full_capture()[CARDS[1]]['episodes']:
        p=episode['parameters'];E=np.exp(1j*np.array(p['Theta']));V,d=E.shape
        np.testing.assert_array_equal(complex_array(episode['embedding']),E)
        priorP=[];priorM=[];active=None
        for t,(token,state,out) in enumerate(zip(episode['inputs'],episode['states'],episode['card_outputs'])):
            expectedP=[v.copy() for v in priorP];expectedM=priorM.copy()
            if t:
                w=np.conj(E[episode['inputs'][t-1]])*E[token]
                np.testing.assert_array_equal(complex_array(state['w']),w)
                scores=[sum(float(a.real*b.real+a.imag*b.imag) for a,b in zip(proto,w))/d for proto in priorP]
                np.testing.assert_allclose(state['c'],scores,atol=1e-14,rtol=1e-13)
                j=int(np.argmax(state['c'])) if scores else -1
                assert state['j']==j
                if j>=0 and state['c'][j]>=p['tau']:
                    assert state['action']=='merge';merges+=1
                    z=priorM[j]*priorP[j]+w
                    np.testing.assert_array_equal(complex_array(state['z']),z)
                    # Check each complex coordinate with scalar real arithmetic, independently of np.where.
                    expectedP[j]=np.array([complex(zv.real,zv.imag)/math.hypot(zv.real,zv.imag) if abs(zv)>0 else old for zv,old in zip(z,priorP[j])])
                    expectedM[j]+=1;active=j
                else:
                    assert state['action']=='spawn'
                    if len(priorP)==p['K']:
                        victim=min((i for i in range(len(priorP)) if i!=active),key=lambda i:(priorM[i],i))
                        assert state['evicted']['index']==victim and state['evicted']['m']==priorM[victim]
                        np.testing.assert_array_equal(complex_array(state['evicted']['P']),priorP[victim])
                        del expectedP[victim];del expectedM[victim];evictions+=1
                        if active is not None and active>victim:active-=1
                    expectedP.append(w);expectedM.append(1)
            else:
                assert state['action']=='initial'
            assert state['m']==expectedM and state['alpha']==active
            for got,want in zip(state['P'],expectedP):
                np.testing.assert_allclose(complex_array(got),want,atol=1e-14,rtol=1e-13)
            s=np.zeros(V) if active is None else np.array([sum(float((expectedP[active][k]*E[token,k]*np.conj(E[y,k])).real) for k in range(d))/d for y in range(V)])
            np.testing.assert_allclose(state['s'],s,atol=1e-14,rtol=1e-13)
            lg=np.array(p['u'])+p['beta']*s;prob=np.exp(lg-lg.max());prob/=prob.sum()
            np.testing.assert_allclose(out,prob,atol=2e-14,rtol=1e-12)
            priorP=[complex_array(v) for v in state['P']];priorM=state['m'];tokens+=1
    assert (tokens,merges,evictions)==(2119,467,950)


def test_three_relations_minimally_refute_running_mean_then_phase_projection():
    e=hand('phase_mean');out,_=run_original(CARDS[1],e);states=e['states']
    assert [s['action'] for s in states]==['initial','spawn','merge','merge']
    first_phase=math.pi/6
    phase_card=math.atan(1/(math.sqrt(3)+1))
    phase_mean=math.atan(math.sqrt(3)/5)
    assert np.angle(complex_array(states[2]['P'][0])[0])==pytest.approx(first_phase)
    assert np.angle(complex_array(states[3]['P'][0])[0])==pytest.approx(phase_card)
    points=[[1.,0.],[.5,math.sqrt(3)/2],[1.,0.]]
    labels,means,history=dpmeans(points,1.2)
    assert labels.tolist()==[0,0,0]
    np.testing.assert_allclose(means,[[float(Fraction(5,6)),math.sqrt(3)/6]],atol=1e-15)
    assert math.atan2(means[0,1],means[0,0])==pytest.approx(phase_mean)
    assert phase_card-phase_mean>.01
    # Even granting phase projection and the CARD head to the published mean changes probability.
    theta=np.array(e['parameters']['Theta'])[:,0]
    s=np.cos(phase_mean+theta[e['inputs'][-1]]-theta)
    logits=np.array(e['parameters']['u'])+e['parameters']['beta']*s
    prob=np.exp(logits-logits.max());prob/=prob.sum()
    assert np.max(abs(prob-out[-1]))>1e-3


def test_full_dpmeans_recruits_and_reassigns_with_arithmetic_mean_hand_answers():
    labels,means,history=dpmeans([[0.],[1.],[10.]],4.)
    assert labels.tolist()==[0,0,1]
    np.testing.assert_array_equal(means,[[.5],[10.]])
    assert len(history)>=2
    assert not np.any(np.isclose(np.linalg.norm(means,axis=1),1.))


def test_fuzzy_art_published_choice_vigilance_minimum_update_and_recruitment_are_distinct():
    weights,events=fuzzy_art([[.25,.75],[.5,.5]],choice=.1,vigilance=.5)
    np.testing.assert_array_equal(weights,[[.25,.5,.5,.25]])
    assert [e['winner'] for e in events]==[0,0]
    assert events[1]['scores'][0]==pytest.approx(float(Fraction(5,7)))
    weights,events=fuzzy_art([[0.],[1.],[0.]],choice=.1,vigilance=.9)
    assert [e['winner'] for e in events]==[0,1,0] and events[1]['rejected']==[]
    _,search=fuzzy_art([[.3],[.7],[.3]],choice=.1,vigilance=.9)
    assert search[1]['rejected']==[0]
    np.testing.assert_array_equal(weights,[[0.,1.],[1.,0.]])
    # Category recruitment selects an output immediately; the CARD's last-merged pointer stays unset.
    e=hand('recruitment');out,_=run_original(CARDS[1],e)
    assert e['states'][-1]['alpha'] is None
    expected=np.exp([.2,-.1]);expected/=expected.sum()
    np.testing.assert_allclose(out[-1],expected,atol=1e-15)
    assert e['states'][-1]['m']==[1]


def test_protected_active_slot_differs_from_ordinary_lfu_fifo_with_two_equal_masses():
    e=hand('protected_slot');out,stats=run_original(CARDS[1],e)
    before,last=e['states'][-2:]
    assert before['m']==[3,3] and before['alpha']==0
    assert last['evicted']['index']==1 and last['m']==[3,1] and last['alpha']==0
    ordinary=min(range(2),key=lambda j:(before['m'][j],j))
    assert ordinary==0
    assert stats['evict']==1


@pytest.mark.parametrize('name,mutation',[
 ('phase_mean',('z = m[j] * P[j] + w','z = P[j] + w')),
 ('phase_mean',('P[j] = np.where(az > 0, z / np.where(az > 0, az, 1.0), P[j])','P[j] = z / (m[j] + 1)')),
 ('phase_mean',('alpha = j','alpha = None')),
 ('phase_mean',('if j >= 0 and c[j] >= tau:','if j >= 0 and c[j] >= 1.0:')),
 ('protected_slot',('if i != alpha','if True')),
])
def test_complex_update_normalization_selection_vigilance_and_eviction_mutants_are_detected(name,mutation):
    e=hand(name);original,stats=run_original(CARDS[1],e);mutated,ms=run_original(CARDS[1],e,mutation)
    assert np.max(abs(original-mutated))>1e-5 or stats!=ms


@pytest.mark.parametrize('a0,b0',[(-.4,.3),(.2,-.7)])
def test_two_free_parameter_training_gradients_match_finite_differences_without_frozen_solver(a0,b0):
    E=np.array([[0.,1.,0.],[1.,0.,-1.],[0.,-1.,0.]])
    loss,g=closure_loss_gradient(E,a0,b0,(0,2),1,4);epsilon=1e-6;finite=[]
    for i in range(2):
        plus=np.array([a0,b0]);minus=plus.copy();plus[i]+=epsilon;minus[i]-=epsilon
        finite.append((closure_loss_gradient(E,*plus,(0,2),1,4)[0]-closure_loss_gradient(E,*minus,(0,2),1,4)[0])/(2*epsilon))
    np.testing.assert_allclose(g,finite,atol=1e-9,rtol=1e-7)
    assert np.all(abs(g)>1e-3)
    # First Adam step after bias correction, with zero moments and the specified scalar learning rate.
    updated=np.array([a0,b0])-.01*g/(abs(g)+1e-8)
    assert closure_loss_gradient(E,*updated,(0,2),1,4)[0]<loss
    assert np.max(abs(updated-np.array([a0,b0])))>.009
    _,truncated=closure_loss_gradient(E,a0,b0,(0,2),1,2)
    assert np.max(abs(truncated-g))>1e-3


def test_phase_training_positive_negative_and_distinctness_terms_have_independent_numeric_gradients():
    theta=np.random.default_rng(29).uniform(-.7,.7,(5,3));tokens=[0,1,4,2,3,4,1,2,4,3,0]
    loss,g=coherence_loss_gradient(theta,tokens);epsilon=1e-6
    for a,k in ((0,0),(1,2),(4,1)):
        plus=theta.copy();minus=theta.copy();plus[a,k]+=epsilon;minus[a,k]-=epsilon
        numeric=(coherence_loss_gradient(plus,tokens)[0]-coherence_loss_gradient(minus,tokens)[0])/(2*epsilon)
        assert g[a,k]==pytest.approx(numeric,abs=1e-8,rel=1e-6)
    assert coherence_loss_gradient(theta-.001*g,tokens)[0]<loss
    for kwargs in ({'drop_negative':True},{'drop_distinct':True}):
        mutant,mg=coherence_loss_gradient(theta,tokens,**kwargs)
        assert abs(mutant-loss)>.1 and np.max(abs(mg-g))>.01
    e=copy.deepcopy(hand('phase_mean'));base,_=run_original(CARDS[1],e)
    e['parameters']['Theta'][1][0]+=.15
    changed,_=run_original(CARDS[1],e)
    assert np.max(abs(base-changed))>.01


def test_historical_complex_refutation_is_retained_separately_from_local_reexecution():
    d=load('evidence')['measurements']['complex_original_comparator']
    old=json.loads(d['historical_stdout']);new=json.loads(d['reexecuted_stdout'])
    assert old['equivalent'] is False and old['max_abs_diff']==.44461806885749666
    assert new['max_abs_diff']<1e-12
    assert 'not localized or retracted' in d['scope']


def test_independent_publications_have_equation_excerpts_and_optional_local_body_checks():
    publications=load('publications');assert {p['name'] for p in publications}=={'ppgn','dpmeans','art-survey'}
    for p in publications:
        assert p['equations'] and p['excerpt'] and p['section'] and p['retrieved_utc']
        required={'ppgn':['(m3 (X), W)','MLP m4'],
                  'dpmeans':['Algorithm 1 DP-means','Repeat until convergence','compute µj'],
                  'art-survey':['Set output','(22)','wJ (new)']}[p['name']]
        assert all(text in p['excerpt'] for text in required)
        assert p['url'].startswith('https://') and p['final_url'].startswith('https://')
        assert len(p['sha256'])==64 and len(p['text_sha256'])==64
        for name,hashkey in (('file','sha256'),('text_file','text_sha256')):
            path=ROOT/p[name]
            if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==p[hashkey]


def test_all_preexisting_tracked_code_policy_assertions_and_prior_negative_evidence_are_unchanged():
    protected=load('protected');assert len(protected)>100
    for name,wanted in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name


@pytest.mark.parametrize('name,card',[
 ('ppgn_pair_head',CARDS[0]),('ppgn_chain',CARDS[0]),('ppgn_isolated',CARDS[0]),
 ('phase_mean',CARDS[1]),('recruitment',CARDS[1]),('protected_slot',CARDS[1]),
])
def test_compact_handcase_states_are_reobserved_from_the_original_card_function(name,card):
    expected=hand(name)
    assert observe_handcase(card,expected)==expected


def test_fuzzy_art_on_affinely_encoded_relation_phasors_is_not_phase_averaging():
    points=np.array([[1.,0.],[.5,math.sqrt(3)/2],[1.,0.]])
    weights,events=fuzzy_art((points+1)/2,choice=.1,vigilance=.5)
    assert len(weights)==1 and [e['winner'] for e in events]==[0,0,0]
    np.testing.assert_allclose(weights,[[.75,.5,0.,(2-math.sqrt(3))/4]],atol=1e-15)
    # Fuzzy ART stores a containing rectangle. Even granting a center-to-phase decoder,
    # that rectangle's center has phase pi/6, unlike the CARD's third merged phase.
    center=(weights[0,:2]+1-weights[0,2:])/2
    decoded=2*center-1
    assert math.atan2(decoded[1],decoded[0])==pytest.approx(math.pi/6)
    card=complex_array(hand('phase_mean')['states'][-1]['P'][0])[0]
    assert abs(math.atan2(decoded[1],decoded[0])-np.angle(card))>.1
