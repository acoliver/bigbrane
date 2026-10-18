"""Build compact source-bound diagnostics and witnesses, without acceptance declarations."""
import copy
import json
from pathlib import Path
import sys

import numpy as np

from hybrid.tests.research.composites_replays import (
    ROOT, FIXTURES, CARDS, load, digest, full_capture, run_original, lattice_closure,
    dense_binding_sweeps, superglue_one_pair, pinned_one_pair,
)

REPORT = ROOT/'results-wt/evidence-recovery'


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump(path, value):
    path.write_text(json.dumps(value, default=plain, separators=(',', ':'))+'\n')


def comparator_diagnostic():
    """Observe the saved comparator as a diagnostic only; never an independent reference."""
    p = load('sources')['cards'][CARDS[1]]['provenance']
    source = (ROOT/p['source']).read_text()
    filename = str(ROOT/p['source'])
    rows = []; current = {}
    def observer(frame,event,arg):
        if frame.f_code.co_filename != filename:
            return None
        v = frame.f_locals; name = frame.f_code.co_name
        if name == 'omp' and event == 'line' and source.splitlines()[frame.f_lineno-1].strip() == 'i = int(np.argmax(c))' and not v['S']:
            current['known_first_pass_scores'] = v['c'].tolist()
        if name == 'step' and event == 'return':
            kind = type(v['self']).__name__
            if kind == 'Card':
                current['card'] = {'S':v['S'].copy(),'coefficients':v['alpha'].tolist(),'output':v['yhat'],
                                   'keys':[k.tolist() for k in v['self'].K],'values':v['self'].Y.copy(), 'query':v['k'].tolist()}
            elif kind == 'Coupled':
                current['known'] = {'S':v['S'].copy(),'coefficients':v['a'].tolist(),'output':v['yhat']}
                rows.append(copy.deepcopy(current))
        return observer if name in ('step','omp','<module>') else None
    import contextlib,io
    previous=sys.gettrace()
    try:
        sys.settrace(observer)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source,filename,'exec'),{'__name__':'__diagnostic__'})
    finally:
        sys.settrace(previous)
    different = [(i,r) for i,r in enumerate(rows) if abs(r['card']['output']-r['known']['output'])>1e-8]
    first_i,first = different[0]
    # At this point dictionaries are identical because the first divergence has not altered writes.
    previous_state = full_capture()[CARDS[1]]['episodes'][0]['states'][first_i-1]
    query=np.array(first['card']['query']); keys=np.array(previous_state['K']); norms=np.array(previous_state['N'])
    scores=[abs(float(k@query))/n for k,n in zip(keys,norms)]
    first['card_prior_state']=previous_state
    first['first_pass_scores']=scores
    first['vectorized_first_pass_scores']=np.abs((keys.T/norms).T@query).tolist()
    return {'different_value_outputs':len(different),'first_step_zero_based':first_i,'first':first,
            'scope':'Original comparator only, not independent publication evidence. Scalar and vectorized dot products can choose different near-tied keys.'}


def main():
    records=full_capture();specimens=load('specimens')
    selections={CARDS[0]:[0,1,400],CARDS[1]:[0],CARDS[2]:[0]}
    cards={c:{'specimen':records[c]['provenance']['specimen'], 'original_receipts':specimens[c]['original_receipts'],
              'selected_episode_indices':selections[c], 'episodes':[records[c]['episodes'][i] for i in selections[c]],
              'accepted_reference':False, 'missing_full_published_operation':True}
           for c in CARDS}
    counts={c:{'episodes':len(r['episodes']),'tokens':sum(len(e['inputs']) for e in r['episodes']),
               'readouts':sum(sum(len(o) for o in e['card_outputs']) if c==CARDS[0] else len(e['card_outputs']) for e in r['episodes'])}
            for c,r in records.items()}
    gcases={
      'schedule':{'parameters':{'d':2,'B':10,'extra_probes':[12]},'inputs':[['DEMO',0,0],['DEMO',6,0],['QUERY',12],['COMMIT'],['QUERY',12]]},
      'eviction':{'parameters':{'d':2,'B':10,'extra_probes':[8]},'inputs':[['DEMO',0,0],['DEMO',6,0],['COMMIT'],['DEMO',2,0],['COMMIT'],['QUERY',8]]},
      'newest_conflict':{'parameters':{'d':3,'B':10,'extra_probes':[0]},'inputs':[['DEMO',0,0],['COMMIT'],['DEMO',0,1],['QUERY',0],['COMMIT']]}}
    for case in gcases.values():
        case['card_outputs'],case['states']=run_original(CARDS[0],case)
    p={'m':3,'d':2,'s':2,'U':3,'V':10,'eps':1e-6,'tau':1e-9,'beta':1.3,'C':np.eye(3).tolist(),'E':list(range(10))}
    omp={'parameters':p,'inputs':[['DEMO',[0,1],7],['DEMO',[0],3],['QUERY',[1],None],['DEMO',[2],5],['QUERY',[1],None]]}
    omp['card_outputs'],omp['states']=run_original(CARDS[1],omp)
    repeated={'parameters':p,'inputs':[['DEMO',[0],1],['DEMO',[0],2],['QUERY',[0],None]]}
    repeated['card_outputs'],repeated['states']=run_original(CARDS[1],repeated)
    sp={'V':2,'r':1,'Pmax':2,'J':3,'kappa':1e-3,'E':[[1.],[1.]],'F':[[1.],[1.]],'beta':.7,'u':[.25,.75]}
    sink={'parameters':sp,'inputs':[0]}
    sink['card_outputs'],sink['evictions'],sink['gated_out']=run_original(CARDS[2],sink)
    sink['pinned_equilibrium_plan']=pinned_one_pair(1.)
    sink['superglue_plan']=superglue_one_pair(1.)
    sink['scope']='One active real row/column, kernel=1, dustbin kernel=1. Strictly positive background splits free mass (1/4,3/4). SuperGlue balanced dustbin differs from pinned potentials and from three sweeps.'
    closure_checks=sum(1 for e in records[CARDS[0]]['episodes'] for token,state in zip(e['inputs'],e['states'])
                       if token[0]=='COMMIT' or token[0]=='DEMO' and state['c']==0)
    se=records[CARDS[2]]['episodes'][0]
    dense,sweeps=dense_binding_sweeps(se['parameters'],se['states'],se['inputs'])
    cold,_=dense_binding_sweeps(se['parameters'],se['states'],se['inputs'],cold=True)
    measurements={'capture_counts':counts,'gcd_compilation_checks':closure_checks,
                  'omp_iterations':sum(len(pu['iterations']) for pu in records[CARDS[1]]['episodes'][0]['pursuits']),
                  'omp_writes':sum(o[2] for o in records[CARDS[1]]['episodes'][0]['card_outputs']),
                  'binding_sweeps':sum(map(len,se['sweeps'])),'binding_evictions':se['evictions'],'binding_gated_out':se['gated_out'],
                  'binding_dense_max_difference':float(np.max(np.abs(dense-se['card_outputs']))),
                  'binding_cold_max_tv':float(np.max(.5*np.sum(np.abs(cold-se['card_outputs']),axis=1))),
                  'original_omp_comparator':comparator_diagnostic()}
    evidence={'cards':cards,'handcases':{'gcd':gcases,'omp_signed_fifo':omp,'omp_repeat_allocation':repeated,'dustbin':sink},'measurements':measurements}
    dump(FIXTURES/'composites_evidence.json',evidence)
    dump(REPORT/'composites-measurements.json',measurements)
    print(json.dumps({k:v for k,v in measurements.items() if k!='original_omp_comparator'},indent=2))
    print('OMP first discrepancy',measurements['original_omp_comparator']['first_step_zero_based'])


if __name__=='__main__':
    main()
