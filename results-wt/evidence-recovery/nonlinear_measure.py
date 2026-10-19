"""Package compact negative fixtures and bounded publication equation snapshots."""
import json
from pathlib import Path

import numpy as np

from hybrid.tests.research.nonlinear_replays import ROOT,FIXTURES,CARDS,load,full_capture,observe_handcase,digest

REPORT=ROOT/'results-wt/evidence-recovery'


def main():
    records=full_capture();specimens=load('specimens')
    selections={CARDS[0]:[0,18],CARDS[1]:[25,85]}
    evidence={'cards':{c:{'specimen':records[c]['provenance']['specimen'],
                'selected_episode_indices':selections[c], 'episodes':[records[c]['episodes'][i] for i in selections[c]],
                'accepted_reference':False,'missing_full_published_operation':True}
              for c in CARDS}}
    ppgn={'parameters':{'d':3,'alpha':.7,'beta':1.1},'inputs':[2,19,20]}
    chain={'parameters':{'d':3,'alpha':.7,'beta':1.1},'inputs':[2,10,20]}
    isolated={'parameters':{'d':2,'alpha':.7,'beta':1.1},'inputs':[2,9]}
    mean={'parameters':{'Theta':[[0.],[np.pi/3]],'K':2,'tau':.4,'beta':2.,'u':[.2,-.1]},'inputs':[0,0,1,1]}
    recruit={'parameters':{'Theta':[[0.],[np.pi/3]],'K':2,'tau':.4,'beta':2.,'u':[.2,-.1]},'inputs':[0,1]}
    theta=(2*np.pi*np.outer(np.arange(4),np.arange(4))/4).tolist()
    pinned={'parameters':{'Theta':theta,'K':2,'tau':.5,'beta':2.,'u':[.2,-.1,.3,-.2]},'inputs':[0,1,2,0,2,0,1,0]}
    evidence['handcases']={name:observe_handcase(c,e) for name,c,e in (
        ('ppgn_pair_head',CARDS[0],ppgn),('ppgn_chain',CARDS[0],chain),('ppgn_isolated',CARDS[0],isolated),
        ('phase_mean',CARDS[1],mean),('recruitment',CARDS[1],recruit),('protected_slot',CARDS[1],pinned))}
    counts={c:{'episodes':len(r['episodes']),'tokens':sum(len(e['inputs']) for e in r['episodes'])} for c,r in records.items()}
    counts[CARDS[0]]['solver_steps']=sum(len(s['iterations']) for e in records[CARDS[0]]['episodes'] for s in e['states'])
    counts[CARDS[0]]['queries']=sum(2*e['parameters']['d']**2<=t<3*e['parameters']['d']**2 for e in records[CARDS[0]]['episodes'] for t in e['inputs'])
    counts[CARDS[1]]['merges']=sum(e['stats']['merge'] for e in records[CARDS[1]]['episodes'])
    counts[CARDS[1]]['evictions']=sum(e['stats']['evict'] for e in records[CARDS[1]]['episodes'])
    diagnostic={'historical_stdout':records[CARDS[1]]['original_stdout'],'reexecuted_stdout':records[CARDS[1]]['reexecuted_stdout'],
        'scope':'Saved comparator diagnostic only. Algebraically matching formulas have different floating evaluation paths. Historical raw states/runtime are missing, so the old 0.444618 mismatch is retained, not localized or retracted.'}
    evidence['measurements']={'counts':counts,'complex_original_comparator':diagnostic}
    (FIXTURES/'nonlinear_evidence.json').write_text(json.dumps(evidence,separators=(',',':'))+'\n')
    publications=json.loads((REPORT/'nonlinear-publications-local.json').read_text())
    specifications={
      'ppgn':('Maron et al., Provably Powerful Graph Networks (2019)','Section 6, equation 6 and block definition; section 7 implementation and readouts',[(325,341),(425,437),(475,492)],
              ['W_ijc = sum_k m1(X)_ikc m2(X)_kjc; B(X)=concat(m3(X),W)', 'F=m o h o Bd o ... o B1 with invariant h', 'Section 7 adds m4(concat(X,W)); its skip is current X, not original X0']),
      'dpmeans':('Kulis and Jordan, Revisiting k-means: New Algorithms via Bayesian Nonparametrics (2011)','Section 3.1, Algorithm 1; section 3.2, equation 1',[(270,308)],
                 ['Initialize one global-mean cluster; repeatedly assign nearest Euclidean centroid or recruit when squared distance > lambda', 'Recompute each centroid as the arithmetic mean of its assigned points until convergence', 'Objective=sum_c sum_x_in_c ||x-mu_c||^2 + lambda*k']),
      'art-survey':('Brito da Silva, Elnabarawy and Wunsch, A Survey of Adaptive Resonance Theory Neural Network Models for Engineering Applications (2019)', 'Algorithm 1; section 2.1.3 equations 15,16,19,22',[(284,326),(431,501)],
                    ['Complement code x=[raw,1-raw]', 'T_j=||min(x,w_j)||_1/(choice+||w_j||_1); M_j=||min(x,w_j)||_1/||x||_1', 'Reject failed category and search others; recruit if none pass; output selected category even on recruitment', 'w_new=(1-rate)*w_old+rate*min(x,w_old)'])}
    for p in publications:
        title,section,ranges,equations=specifications[p['name']];lines=(ROOT/p['text_file']).read_text().split('\n')
        p.update(title=title,section=section,equations=equations,excerpt='\n\n'.join('\n'.join(lines[a-1:b]) for a,b in ranges),
                 excerpt_text_lines=ranges,scope='Independent publication snapshot; no claim that it defines the CARD-specific full composition.')
    (FIXTURES/'nonlinear_publications.json').write_text(json.dumps(publications,indent=2)+'\n')
    # Keep all preceding tracked evidence and acceptance surfaces byte-for-byte unchanged.
    import subprocess,hashlib
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    protected={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in tracked if (ROOT/name).is_file()}
    (FIXTURES/'nonlinear_protected.json').write_text(json.dumps(protected,indent=2)+'\n')
    (REPORT/'nonlinear-measurements.json').write_text(json.dumps(evidence['measurements'],indent=2)+'\n')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    main()
