# r2-bar report

## Verdict

The saved replay FAILS both pre-registered named-gate guards. Rule 1 requires all eight published mechanisms to die
at `reduces_to_known:*` and both controls at `excluded_family:*`; 9/10 do so. SEQUITUR dies later at `prior_art_tested`
after its numerical reduction is refuted. Rule 2 requires all six previously counted reductions to die at a reduction
gate; only 3/6 do so. Card-440f0204 dies at no_difference, card-62785c10 at R3, and card-d6dfeac2 at R2.
The secondary all-dead diagnostic is 10/10 and 6/6, respectively. It does not satisfy the registered predicates.
No acceptance threshold or preregistration was changed to produce this correction.

Five cards went from kill to S2 pass: card-3485a13f, card-f5ee7edb, card-8154d3b4, card-a4f684fb, card-89248ee5. Whether
each one is truly new is not decided here. The critic's known-work answer, the sources it weighed and the accepted
difference statement for each are in the flips section below, and an independent adversarial literature check of the
five runs separately. Until that check is in, treat the five as unconfirmed, not as findings.

Historical pre-KD8/PA2 suite, not verification of the current tree: research suite 1391 passed, 2 skipped, 0 failed (the earlier run had 17 failed and 3 errors); the other hybrid suites 1831 passed (1827 plus 4 environment failures that pass once the worktree has its local data), 18 skipped, 0 code failures. No failure was a regression in the new code.

## What changed

- R2A: the reduction critic judges the general learned mechanism with learned parts free. A resemblance that holds only
  with a learned part fixed is recorded in `fixed_learned_parts` (only when the card states a capability the frozen form
  lacks) and cannot carry a cited-source kill.
- R2C: `radical_enough` is replaced by `known_work` (gated), `known_work_sources` and `surprise` (recorded). The host
  acts on `known_work: true` only with a host-counted comparator or a cited source it accepts (paper id present in the
  search results, named textbook section or standard algorithm). `textbook_transplant: true` now gates. Exclusion stays
  with families.toml and the counted-comparator gate. Code: hybrid/research/stages/known_work.py, s2_prior_art.py,
  schemas/reduction_critic.schema.json, prompts/reduction_critic.md.
- R2D: the difference checker reports claims its excerpts cannot support as `excerpt_gaps` (scoped, not a kill); the two
  required closest_prior entries count only relevant sources and family entries fill the rest (f74f2f2 removed the
  two-entry floor); a paper the critic cited must be a literature entry. Code: stages/difference.py,
  prompts/difference.md, difference_check.md.
- DECISIONS.md rows R2A, R2C, R2D.

## Test suites

Commands: pytest hybrid/tests/research (run as 4 file shards, 408 + 394 + 248 + 341 passed, 2 skipped, logs/tests-11-shard*.log)
and pytest over hybrid/tests/literature, llm, gateway, bench, math and the top-level test files except test_probes.py,
which needs a live P0 run (BIGBRANE_P0_RUN) and cannot be collected here (logs/tests-7-other.log: 1827 passed, 4 failed;
logs/tests-9-envfix.log: the 4 pass, 31 tests).

The earlier research run (logs/tests-2.log) had 17 failed and 3 errors. Classification of all 20:

- Stale fixtures and tests that asserted the old contract on purpose, updated to the R2C contract (commit 267db47):
  test_orchestrator (4: radical fakes defaulted to textbook_transplant true, and one test asserted a textbook transplant
  is recorded and not gated, which is now a gate), test_escape (1: NOT_REDUCED inherited textbook_transplant true),
  test_fix10_phase7_live (1: the requirements call passed a critic without known_work_sources, which the schema
  requires), test_orchestrator_units (1: asserted the prompt says the textbook answer "does not decide the gate").
  test_kd7 and test_difference tests were already moved to the new difference-gate rule in f74f2f2 and pass.
- Replay harness (commit 7a916a7): test_phase7_replay (3 errors). The fixture predates R2C and holds radical_enough,
  which the schema no longer allows. The harness maps it with novelty_fields. The expected Phase 7 codes were not touched.
  The two cards that passed S2 in the source run (card-46dbb349, card-c9615069) were recorded with radical_enough true,
  textbook_transplant false, no reduction and a `novel` prior-art label. The recorded critic named no known work for
  either, so the new rule still passes them. The 19 deaths keep their codes, including the three excluded-family kills.
  No FIX-10, FIX-12, KD4, KD5, KD7 or Phase 7 replay assertion was weakened.
- Real regressions or inconsistencies in the new code: none found.
- Environment, not code: the worktree lacked gitignored local data. test_budgets and test_fix7_match need
  runs/run-20261010T000322Z and results/phase4-.../fix7-staging (copied from the main workspace, read only). test_seed
  needs hybrid/literature/cache/seed_build_log.jsonl, and three bench/retention tests need .venv-hybrid inside the
  worktree (a temporary symlink, removed). With that data present they pass. test_planted failures from the earlier run
  were gone in the rerun without a change to them.

## Acceptance rules (pre-registered in STATUS.md, scored in SCORE.md)

| rule | registered requirement | result | observed gate |
|---|---|---|---|
| 1. novelty probe guard | all 8 published at reduces_to_known, both controls at excluded_family | FAIL | 9/10; card-17100007 died at prior_art_tested |
| 2. counted reductions stay dead | all 6 at the reduction gate, no later-gate substitution | FAIL | 3/6; card-440f0204 no_difference, card-62785c10 R3, card-d6dfeac2 R2 |
| 3. flips are inspected | every flip listed and the general-mechanism question answered | PASS for historical inspection | five listed below; independent audit classifies four incremental, tensor circulation a narrow unvalidated lead |

Card-17100007 (SEQUITUR) died on a different gate than rule 1 named. The rule expected `reduces_to_known:*`. The kill
code is `prior_art_tested and no untested element`. The kill came from the prior-art classifier finding cs/9709102, and
the numeric check did not verify the reduction. Had the prior-art gate not fired, the textbook_transplant and
known_work gates (R2C) would have been next, since the critic answered true to both. This replay does not show that
chain, so it does not show that R2C alone kills SEQUITUR.

Of the 29 replayed cards, 24 died and 5 passed S2. The 29 are the 11 recheck cards, 8 batch cards and 10 probe cards.
Replay limits: one sample of model judgment per card; the batch group replayed 8 of its R2 kills, so the other four are
not covered. The replay started before commit f74f2f2 (family entries fill slots without a two-entry floor) and was
not rerun. No replay difference artifact contains the removed `at least 2 entries are required` rejection, so that
commit changed no recorded outcome.

## The five flips

Taken from the replay run artifacts by tools/flips.py (runs/batch8 and runs/recheck, reduction_critic and difference
artifacts). "Known work" is the critic's `known_work` field with its `why`. The critic answered false for all five, so
none was killed by the known-work gate. The listed sources are the ones the critic weighed in `known_work_sources`.
The accepted difference statement is the one the difference checker accepted.

### card-3485a13f

- Prior-art label: prior_art_undertested. Critic: reduces_to other_known, known_work False, surprise False, textbook_transplant False.
- Critic's known_work answer: The charge and readout are, symbol for symbol, prior-weighted Weighted Majority / Bayesian model averaging with an eps-symmetric noise likelihood over the finite program class Pi_D. beta = 1/rho, m is the mistake count, and the readout is the aggregating-algorithm mixture. The prune order is just descending posterior score under A3, as the card itself proves. With the class fully tracked, this is also a diagonal input-dependent multiplicative recurrence (SSM form, B = 0) with a normalized query-dependent read, and the card names this relative. But the card runs with K = 16 < 72. Its context is a state-selected top-K beam plus a rebirth write that charges absent consistent programs k-1 ...
- Sources it weighed:
  - standard_algorithm Weighted Majority / aggregating algorithm with prior weights (Littlestone & Warmuth 1994; Vovk 1990), i.e. Bayesian model averaging over a finite hypothesis ...: The charge step (mistake counts), the posterior weights lambda * rho^{-m}, and the eps-smoothed mixture readout (readout[0], readout[1], update[1]) when all programs are tracked. It does not cover the K-slot truncation or the k-1 rebirth (update[2], update[3]).
  - textbook T. Mitchell, Machine Learning, ch. 2 (version spaces, candidate elimination): The noiseless elimination of hypotheses inconsistent with demonstrations and prediction by surviving hypotheses (few-shot steps ii-iii). It does not cover the noise counts, the beam, or the rebirth.
  - paper arxiv:1904.06317: An approximate Bayesian posterior over a finite set of enumerated programs from few demonstrations, with prediction by a posterior-weighted program mixture. It works offline over stored demonstrations and does not cover the position-free beam with pessimistic rebirth.
  - paper arxiv:2211.16605: The offline MDL library abstraction of D10 (promoting frequent compositions to library entries by compression gain). It does not cover the in-context update.
- Resemblances that held only with a learned part fixed: beam width K set to |Pi_D| = 72 (no pruning) for the SSM / full Weighted-Majority equality (capability: The carried state stays at K(D+1)+2 = 50 words of 64 bits)
- Difference statement accepted (attempt 1, changed equation update[2]): In update[2], the state is no longer a weight for every program of Pi_D (or a store of pairs): it is a K = 16 slot beam into which every absent program that maps the pending key c to the current value is inserted with miss count k-1, an upper bound on its true miss count and so a lower bound on its posterior, because no history is kept to check. update[3] then prunes the beam to the top K by exact integer miss count and prior, so a pruned or noise-killed rule is reborn at the next pair it explains.
- Closest prior entries in the statement: family ssm; family ssm; literature arxiv:1904.06317; literature arxiv:2211.16605; family exact_match_cache; literature openalex:W178818641
- Claims the checker scoped as unsupported by the excerpts: The characterization of arxiv:1904.06317 as operating offline over stored demonstrations is not established by its supplied abstract. | The claim that arxiv:1904.06317 uses the specific prior-weighted Weighted Majority or symmetric-noise Bayesian mixture described in the statement is not established by its abstract, which specifies ... | The characterization of arxiv:2211.16605 as the source of the card's particular MDL composition-promotion procedure is not established by its supplied abstract, which describes corpus-guided ...
- Artifacts: runs/batch8/artifacts/reduction_critic/card-3485a13f-v7-dc6577.json; runs/batch8/artifacts/difference/card-3485a13f-v7-abf035.json

### card-f5ee7edb

- Prior-art label: prior_art_undertested. Critic: reduces_to linear_attention, known_work False, surprise False, textbook_transplant False.
- Critic's known_work answer: The state is a one-hot bigram outer-product memory with saturating (idempotent) writes, which is a normalized linear-attention store keyed by the previous token. The eta=0 readout is a degree-normalized hop-1 plus hop-3 multi_read of that memory, alternating forward and reverse reads, so it reduces exactly to repaired linear attention; the recall and copy guarantees all hold in this limit. The only part outside that form is eta*Q. Q squares an intermediate read output and subtracts a read of the elementwise-squared memory, so it counts products between distinct supporting witnesses. That transformation is typed 'other', which leaves the full comparator unresolved and makes the mapping a ...
- Sources it weighed:
  - paper arxiv:1906.05059: Scoring an unseen link (a, j) from closed two-path and higher-order motifs through shared neighbours. This is the analogue of S (odd-path a->b<-i->j transport) and of counting multiple supporting paths. It does not use the distinct-pair product form Q, sequence streaming or saturating writes.
  - paper arxiv:1902.06679: Motif-count features for link prediction, including multiple-witness motifs. This is the closest published account of using co-occurring witnesses to score a candidate edge. It covers the S and Q motif idea only in a static-graph, feature-learning setting.
  - standard_algorithm Symmetric degree-normalized biadjacency with odd-power propagation (HITS / bipartite spectral co-clustering, Dhillon 2001; Kosko bidirectional associative ...: Readout steps N = D_r^{-1/2} B D_c^{-1/2}, overlap = N N^T e_a and S = N^T N N^T e_a, plus the forward and reverse reads of one matrix memory.
  - paper arxiv:2401.17377: The direct term N_{a,j}: exact successor lookup from a transition (n-gram) table built from the context.
- Resemblances that held only with a learned part fixed: eta (set to 0) (capability: For every candidate j, the eta=1 score minus the eta=0 score equals Q_j = sum over ordered pairs of distinct witnesses ...)
- Difference statement accepted (attempt 1, changed equation readout[2]): Readout[2] adds Q_j = S_j^2 - sum_i W_ij^2 = sum over i != i' of W_ij W_i'j, which is zero for one witness and 2uv for two. Readout[3] adds it as eta*Q_j to the unchanged additive odd-path score N_aj + S_j/6. The card separates Q from squared mass at rank level (D14) but not from minimum or geometric-mean corroboration, which differ only in probabilities.
- Closest prior entries in the statement: literature arxiv:1906.05059; literature arxiv:1902.06679; literature arxiv:2401.17377; family linear_attention; literature openalex:W4379958458
- Claims the checker scoped as unsupported by the excerpts: The claim that arxiv:1906.05059 scores unseen links via shared-neighbour paths and counts multiple supporting paths is not stated in the abstract excerpt, which only describes motif closure and the ... | The claim that arxiv:1902.06679 uses motif-count features including multiple-witness motifs is not shown by the truncated excerpt, which says only 'higher-order analysis of the graph topology'. | The claim that none of the cited papers forms a product over distinct witness pairs or excludes self-products is not supported or contradicted by the excerpts, which do not give scoring formulas. | The claim that infini-gram assigns score only to observed continuations apart from backoff is not stated in the excerpt. | The claim about the form of the nonlinearity in Long Sequence Hopfield Memory is not shown by the truncated excerpt.
- Artifacts: runs/batch8/artifacts/reduction_critic/card-f5ee7edb-v7-8b87e3.json; runs/batch8/artifacts/difference/card-f5ee7edb-v7-a51b33.json

### card-8154d3b4

- Prior-art label: novel. Critic: reduces_to none, known_work False, surprise False, textbook_transplant False.
- Critic's known_work answer: I tried each target family in turn. Linear attention and delta rules: there is no additive outer-product write. The state is multiplied by input masks and transformed cubically. SSM: in P = A² coordinates, assertions are diag(C_τ) gates with a normalizer and resets are fixed linear kernels. At J=0 this is an input-gated linear recurrence with a normalizer, i.e. Bayesian/HMM filtering over a 2^N table. With J free, the update adds D⊙D, where D = ρ_L J B − B J ρ_R. That term is state-dependent and nonlinear in P, so no gate, normalizer, change of basis, lifted state or sparsity reproduces it. A comparator would need type `other`. RNN and tensor RNN: the hidden-to-hidden map is cubic in the ...
- Sources it weighed:
  - textbook Koller & Friedman, Probabilistic Graphical Models, ch. 15 (inference in temporal models: belief-state filtering) and ch. 9 (conditioning/variable elimination): Support masking and renormalization as conditioning on deterministic evidence, reset as marginalize-and-re-randomize a variable, and marginal readout. Together these are the whole core at J=0. It does not cover the circulation term D or the quadrature reinjection.
  - standard_algorithm HMM forward algorithm (normalized filtering with a transition kernel): Assertion as a diagonal observation gate with normalization, and reset as a linear transition kernel on P. A learned transition between steps also yields non-idempotent repeated observations, but it is linear and state-independent, unlike D.
  - paper arxiv:1605.05775: Amplitude-tensor representation over binary assignments with Born (squared-amplitude) readout and gradient training of tensor parameters. It does not cover sequential masking, reset or circulation.
  - paper arxiv:1405.7375: Encoding Boolean constraints (including AND/XOR clauses) as 0/1 tensors over assignments and contracting them. This corresponds to the support mask M and C_τ. It does not cover the amplitude dynamics.
- Resemblances that held only with a learned part fixed: J (learned antisymmetric parameter) set to 0, under which D=0 and the core is exact conditioning / normalized diagonal-gated filtering on P (capability: redistribute surviving probability through a learned, state-dependent Schmidt circulation that preserves logical ...)
- Difference statement accepted (attempt 1, changed equation update[1]): After the standard support mask, update[1] adds a state-dependent circulation D=ρ_L J B − B J ρ_R. It uses the current reduced densities ρ_L=BB^T and ρ_R=B^TB and a learned antisymmetric J, so D is cubic in B and not a fixed linear kernel; update[2] reinjects it as M⊙sqrt(B⊙B+D⊙D). D is exactly zero at J=0 and on rank-one B, where the core is plain conditioning, and the mask, reset and deterministic guarantees are unchanged.
- Closest prior entries in the statement: literature arxiv:1405.7375; literature arxiv:1605.05775; literature arxiv:2203.12579
- Claims the checker scoped as unsupported by the excerpts: The statement says arxiv:1405.7375 encodes Boolean constraints as tensors over assignments, which the excerpt supports only as 'an axiomatic tensor contraction language for n-variable #SAT'. The ... | The statement says arxiv:1605.05775 has no sequential masking, reset or in-context state update, and that its models are fixed classifiers. The abstract does not describe update procedures, so this ... | The statement says arxiv:2203.12579 defines no dynamics, learning rule or relational task. The abstract mentions none, but it does not exclude them. | The statement says a learned linear HMM transition is state-independent and non-idempotent. This is a claim about the HMM and the textbook, which have no excerpt in the search, so no supplied excerpt ...
- Artifacts: runs/batch8/artifacts/reduction_critic/card-8154d3b4-v7-d1708d.json; runs/batch8/artifacts/difference/card-8154d3b4-v7-37833f.json

### card-a4f684fb

- Prior-art label: novel. Critic: reduces_to none, known_work False, surprise True, textbook_transplant False.
- Critic's known_work answer: I tried every listed form. - RNN/SSM: written as a permutation matrix, the update is P_t = G_t P_{t-1} G_t^T. G_t is computed from P_{t-1} and A, not from the token alone. That is 'keys from the memory state', which is type other. For the bare d x d state, the card's restriction R holds: (gPg^{-1})[a]=b forces P[g^{-1}a]=g^{-1}b, and P[k] varies across the class. - One-hot lift: a lift over the reachable (P,A) states, of size up to |C(P0)| * 2^d, gives an input-only linear form. That is a generic finite-automaton embedding, not a lifted state 'of the same kind', so I do not count it as a repair. - Exact-match cache (with transformed_input adding reverse pairs): this reproduces the ...
- Sources it weighed:
  - textbook Dummit & Foote, Abstract Algebra, Sec. 4.3 (conjugacy classes in S_n: conjugates have the same cycle type, and g sigma g^{-1} relabels sigma's cycles by g): Lemma L1 (cycle type is invariant under conjugation) and the relabeling formula P'[u]=g(P[g(u)]) used in update[2]. It does not cover the state-read choice of g (update[0], update[1]), the pin-safety test, or its use as an in-context binding learner.
  - standard_algorithm Constraint propagation / exact enumeration of permutations of a fixed cycle type consistent with given assignments (CSP over S_d): The answers on forced keys: any correct solver over C(P0) ∩ Ext(pins) returns the same forced reverse, closing, triad and C4 completions. It does not cover the O(1) deterministic transition or the order-dependent completion on unforced keys.
  - paper arxiv:2402.10116: Only the conjugation-invariant structure of permutation laws, which is related to L1. It has no update or learning use.
- Difference statement accepted (attempt 2, changed equation update[2]): The one-sided installation P<-P∘(a h), which rewrites P[a] and P[h], is replaced by update[2]'s conjugation P<-gPg^{-1}, with g=(b P[a]) or (a P^{-1}[b]) chosen in update[1] from pre-event P, Pinv and pins. It runs only on conjugating demos, rewrites at most four images, keeps cycle type and earlier pins, then pins a (update[3]). Already-present demos (c=b) only pin in update[0]; ignored demos change nothing and install no binding.
- Closest prior entries in the statement: literature arxiv:2402.10116; family exact_match_cache; family hopfield_associative_memory
- Artifacts: runs/recheck/artifacts/reduction_critic/card-a4f684fb-v7-222e79.json; runs/recheck/artifacts/difference/card-a4f684fb-v7-9d305c.json

### card-89248ee5

- Prior-art label: prior_art_undertested. Critic: reduces_to hopfield_or_kernel_memory, known_work False, surprise False, textbook_transplant False.
- Critic's known_work answer: Both state updates are textbook Willshaw clipped outer-product writes on a shifted stream: key = the previous (or two-back) code, value = the current code. The immediate read is the standard Willshaw thresholded retrieval, and the 4s/D factor is a normalizer from the memory's own column sums. Together these cover everything except the triangle read. That read, under the card's premise A2, equals the AND of two full-support Willshaw reads from two different lag memories. In effect it is a factorized trigram conjunction with an IDF-like apex tax. A multiplicative cross-memory conjunction is not a REPAIR POLICY type, so a comparator containing it is unresolved, and the bare Willshaw form does ...
- Sources it weighed:
  - paper arxiv:1207.7196: The clipped binary Hebbian (Willshaw) writes G <- max(G, b1 b0^T) and H <- max(H, b2 b0^T), and the threshold-at-cue-activity read r1.
  - paper openalex:W1525999779: Sparse hard-location activation by fixed address blocks C, the activation buffer of sequence-pointer-style writes from the previous activation to the current one, and decoding the read back over the token's locations (z).
  - paper arxiv:2411.14480: Sequence storage as sparse edges between token nodes and retrieval by context support over stored transitions, including frequency/degree-aware scoring of candidates. It does not cover the lag-two typed graph H, the restricted triangle test c_k = P, or the 4s/D tax applied only to full triangle support.
  - paper arxiv:2302.11443: Per-vertex triangle counting as a graph primitive, i.e. the summation form of c_k. It has no sequence semantics, lag typing or degree tax.
- Difference statement accepted (attempt 1, changed equation readout[3]): In readout[3], the token score z_y = (1/s) sum_k C_yk (r1_k + 4s*r2_k/D_k) adds to the unchanged Willshaw read r1 a bonus that is nonzero only when P>0 and c_k=P (every active lag-one base edge closes through k via H and G). The bonus is divided by the apex's clamped G column degree D_k. On the rectangle this gives u=5, v=3 where the single-cue read and the constant bonus tie, but it picks the wrong token when the target is the more popular apex (7/3 vs 3).
- Closest prior entries in the statement: family hopfield_associative_memory; literature arxiv:1207.7196; literature openalex:W1525999779; literature arxiv:2411.14480; literature arxiv:2302.11443
- Claims the checker scoped as unsupported by the excerpts: The statement says the Willshaw paper (1207.7196) studies the update rule for familiarity discrimination; the excerpt supports this, but its claim that the card's r1 read matches the paper's readout ... | The statement says SDM (W1525999779) has no lag typing and no conjunctive or degree-normalized read; the excerpt describes only the two-matrix architecture and does not state read or update ... | The statement says arxiv:2411.14480 has no lag-two graph, triangle test or popularity tax; the excerpt does not state edge types, update procedure or scoring, so this is neither supported nor ... | The statement's description of the family's failure (identical-key softmax read weights u and v equally) rests on the card's own computation, not on any excerpt.
- Artifacts: runs/recheck/artifacts/reduction_critic/card-89248ee5-v6-48cc06.json; runs/recheck/artifacts/difference/card-89248ee5-v6-e85ea7.json


## Limits

- One sample of model judgment per card, and the replay code predates f74f2f2 (checked above: no recorded outcome depends on it).
- The 8 batch cards are the six that froze a learned part plus card-084a2510 and card-909ad773; the other four R2 kills of
  the batch were not replayed. Rule 2 covers the six recheck cards that had a counted reduction in the earlier run.
- The flips are listed, not judged. Whether the learned general mechanism of each is published or textbook is the question
  rule 3 asks; the independent adversarial literature check answers it. Several accepted statements rest on excerpts that
  the checker marked as not supporting the claims about the prior papers (`excerpt_gaps`, shown per flip).

## Evidence correction after final findings

These are historical observations from the saved run snapshots, not new outcomes from the repaired tree. The original
Rule 1 and Rule 2 misses remain failures. Their exact gate reasons are retained in SCORE.md. A later kill cannot
validate a failed reduction check. The independent audit at main `results/flip-audit-20261018/AUDIT.md` classifies
3485a13f, f5ee7edb, a4f684fb and 89248ee5 as incremental, and 8154d3b4 as a narrow unvalidated lead. It establishes
no sample-efficiency or language-model advantage. The newer KD8/PA2 results, source failures, and suite3 failure
are reported separately in `../kd8/REPORT.md`. No full-suite result on the final repaired tree is claimed here.
