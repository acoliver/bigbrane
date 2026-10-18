# Original Phase7 misses remain visible

Two original aggregate assertions fail. Of 19 registered S2 deaths, 16 do not reproduce the exact required code. Two supported original repairs now have executed evidence; the other historical evidence remains unverified rather than inferred from its recorded verdict.

| Card | Expected | Current | Recorded comparison |
| --- | --- | --- | --- |
| card-14429dbc | reduces_to_known:other_known | none | The card has two parts. Both map onto known constructions with no residue left over. |
| card-2c5dd003 | reduces_to_known:other_known | none | Online (recursive, quasi-Bayes) EM for HMM emission parameters, with fixed transitions and fixed-lag-1 smoothing for the sufficient statistics. Sources: Titterington 1984 recursive |
| card-36725a7c | reduces_to_known:other_known | none | Known form: sequential greedy (local-MAP) Dirichlet-process / CRP mixture of conjugate Bayesian linear regressions. This is SUGS (Wang & Dunson 2011, 'Sequential Updating and Greed |
| card-582f8abf | reduces_to_known:other_known | none | The card is Z2-labeled (parity / weighted) union-find, the standard online solver for XOR-pair equations x_a ⊕ x_b = s. This is the same as online 2-coloring and bipartiteness chec |
| card-68612a2a | reduces_to_known:other_known | none | Exemplar-dictionary coupled sparse coding (Yang, Wright, Huang & Ma, 'Image super-resolution as sparse representation of raw image patches', CVPR 2008; also sparse-representation r |
| card-6de11b01 | excluded_family:delta_rule | none | Let a_t = h1(kappa_t) and phi(k_t) = phi(q_t) = e_{a_t}, a one-hot vector over the B slots, computed from the cortical state u_t through the fixed hash of TopK_k(P u_t). The fast s |
| card-6fc7efec | not_radical:R2 | none | Inference reduces to a shift-reduce evaluation of a tree-structured tensor network (a recursive neural tensor network or Socher-style RNTN without nonlinearity, equivalently a weig |
| card-7ec5e5c2 | reduces_to_known:other_known | not_radical:R2 | The card is the standard worklist (Hopcroft–Karp-style) closure for extending a homomorphism between two deterministic automata over the alphabet {L,R}. Equivalently, it extends a  |
| card-8dc4db49 | reduces_to_known:other_known | not_radical:R2 | Update: N_t = N_{t-1} + gate * e_{w_{t-1}} e_{w_t}^T is the linear-attention state S_t = S_{t-1} + phi(k_t) v_t^T. Here phi(k_t) = e_{w_{t-1}} and v_t = e_{w_t} are one-hot, and a  |
| card-a36f1a3a | reduces_to_known:other_known | none | The state and write rule are a count table over one-hot pair keys, which the card concedes is linear attention or an n-gram table. Each DEMO(a,b,s) adds sigma = 1-2s at key (a,b) a |
| card-bc3935ac | reduces_to_known:other_known | not_radical:R2 | The known form is a path-labeling scheme for ancestor queries from the XML/database literature: Dewey path labels with a prefix test, realized as Tropashko's nested-interval / cont |
| card-d1d383d5 | reduces_to_known:hopfield_or_kernel_memory | not_radical:R2 | The card's per-token step splits into two known pieces. |
| card-d86dca4b | reduces_to_known:other_known | none | The card is the candidate-elimination (version-space) algorithm, with the Bayes-optimal posterior predictive, for the hypothesis class G = PGL(2,p) acting on P^1(F_p). It is packag |
| card-dc69816b | reduces_to_known:other_known | none | Target: an incremental specific-to-general exemplar/rule learner with specificity-ordered conflict resolution. This is Salzberg's Nested Generalized Exemplars (NGE, 1991) and Domin |
| card-ddb071bd | not_radical:R2 | none | Known form: Adaptive Resonance Theory, i.e. ART-style vigilance categorization (Carpenter & Grossberg). Equivalent forms are Hartigan's sequential leader clustering and bounded DP- |
| card-ffdbe331 | reduces_to_known:other_known | none | Known form: Bayesian concept learning with the size principle over a continuous nested family of consequential regions. Sources are Shepard (1987), Tenenbaum (1999, the 'healthy le |

## Why the tagged-memory control cannot be silently preserved

The original card-6de11b01 job-073666f82626 computes a hash-selected residual table, a second tag memory, an equality gate read from that memory and an overwrite gate from the prediction error. The saved job reports that removing the tag changes outputs by 1.23. A plain delta-rule reference cannot establish this mapping. The existing repair policy permits fixed, learned or input-dependent gates, while these two decisions inspect the current memory or prediction. No independent implementation and policy-conformance argument for that mapping was added. Re-labelling its old zero self-comparison as a plain delta rule would fabricate the missing family evidence. The untouched expected assertion still fails.

## Recovery

Implement only separately specified published operations and policy-conforming repairs for any additional comparison whose preservation is claimed. Capture the actual original card-side traces, not the comparator outputs or saved scalar verdict. Leave unsupported cases unverified. The historical labels and test predicates must stay unchanged. The original source run is /Volumes/XS1000/acoliver/projects/bigbrane/runs/run-20261010T181300Z; executor artifacts are attributable by card through ledger artifact events and carry purpose=reduction_check.

Rerun:

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
PYTHONPATH="$PWD" /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest hybrid/tests/research/test_phase7_replay.py -q --basetemp=results-wt/review-fixes/recovery-phase7-pytest
```
