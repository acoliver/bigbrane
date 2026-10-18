# Three mechanism gaps now have executable evidence; exact codes remain 7/19

**No additional exact historical code is restored.** The three prioritized mechanisms now have untouched-program CARD-side captures, source-bound compact evidence, independent handcases and mutation controls. They do not have independently published full-operation references. The original Phase7 assertions still fail. This result does not establish novelty or show that every alternative reduction is impossible.

The completed subset is `2c5dd003` (permanent lag-one HMM deposits), `6fc7efec` (tensor contraction with calibration), and `dc69816b` (residual-key consolidation). The GCD/exemplar composite `14429dbc` was assessed from its original card, critic and executor program, but was not promoted to an additional captured mechanism. Its FIFO cache, periodic commits, multiclass conflict checks and quotient compiler still need a full composition argument. A gcd identity alone does not supply that argument.

## What was executed and preserved

The writable checkout is `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, branch `r2-bar`, HEAD `e0e2c5a5a5b8d2541428ae978c226a7f8f8f9b77`. Main remains on `p4-diagnosis`, HEAD `82580a6630c4b5c9155f00d603f96682d5818e7c`, with clean git status. No commit, push, integration or review was performed. New work has not been independently reviewed.

`remaining_capture.py` executes the complete original programs with `sys.settrace`. It does not replace their random generators, inputs, CARD functions, state updates or comparators. The original comparator runs, but its scalar result is retained only as a diagnostic. Independent comparisons use only captured CARD inputs, parameters, state and outputs. Embedded executor source, original `jobs/<job>/in/main.py`, and packaged program bytes are identical.

| Card | Complete source reexecution | Captured state and free parameters |
| --- | --- | --- |
| `2c5dd003` | 200 random episodes plus the original toy, 2,404 tokens | Arbitrary drawn T, pi0 and strictly positive B0; every Phi, C, R, M, prediction, filtered posterior, lag-one credit and final forecast |
| `6fc7efec` | One continuous 330-token stream: 72 calibration tokens, 258 expression tokens, three compilations; 60 additional soft contractions | All three hidden permutation draws; every calibration D, compiled Z, parser mode/cursor, validity flag, stack, output, and soft-message inputs/results |
| `dc69816b` | All 601 scenarios, 12,008 tokens | All drawn E, U, Wo, P, identity/GELU choices and hyperparameters; every sparse key, complete notebook, residual, stamp, prediction, distribution and write-event sequence |

`remaining_sources.json` binds each capture to its original executor ledger event, source hash, observer hash and complete trace/draw hashes. `remaining_evidence.json` also binds the original full card and critic contents and their original ledger artifact receipts. Changing a card summary or critic mapping prevents use of its evidence. Changing an old expected code or reported scalar does not choose evidence. These negative records contain no admitted declaration or reference-spec identifier.

The full capture is retained locally as `remaining-full-capture.json`. Seven compact episodes, including the complete tensor stream, occupy about 264 KiB in `remaining_evidence.json`. The full capture is reconstructed from the byte-identical programs and hash-bound observer when absent. Tests separately force that reconstruction even when the local raw capture exists.

The historical process hash seed and historical raw draws were not saved. Reexecution uses `PYTHONHASHSEED=0`; these programs draw with seeded NumPy generators and do not sample from set-to-list conversions. Their recovered source-reexecution draws must not be described as recovered historical input bytes. The older forest hash-seed limitation remains unchanged.

## HMM: the one-step identity holds, but it does not identify the full estimator

The original [CARD program, lines 12-26](../../hybrid/tests/research/fixtures/reference_replays/programs/remaining-card-2c5dd003.py) computes the current bets, filters the current token and deposits one credit vector into the **previous** token's column. That deposit is never revisited. Every original free T, pi0 and B0 is retained; no learned emission table is frozen to produce a match.

[Cappé, Online EM Algorithm for Hidden Markov Models, section 2.2, equations 4-7, and Algorithm 1, equations 8-10](https://hal.science/hal-00410106v2/document) distinguishes filtering from recursively carrying the complete auxiliary sufficient statistic. Its backward kernel is

\[
r_t(i\mid j)=\frac{h_{t-1}(i)T_{ij}}{\sum_a h_{t-1}(a)T_{aj}}.
\]

Marginalizing the current filtered state through this kernel gives the CARD's one-step smoothed share. Across all 2,203 observed prefixes of length at least two, the maximum difference is `5.551115123125783e-16`. This establishes the local smoothing identity.

The paper's auxiliary recursion carries the previous conditional statistic through that kernel, weighted by the step size. It does not define online EM by an instruction that every deposited responsibility must remain unchanged forever. To isolate this distinction, the diagnostic keeps the CARD's **observed, changing bet path** identical and recomputes the posterior at every paid position using all observations currently available. It still excludes the latest unpaid token. The resulting paid counts differ on 2,001 of 2,203 prefixes and in all 201 episodes; the largest entry difference is `1.9393429759373326`.

The isolated two-state witness uses T = `[[.9,.1],[.1,.9]]`, pi0 = `[.5,.5]`, B0 = `[[9,1],[1,9]]`, and tokens `[0,1,1]`. At the second token, the first position receives credit `(81/122,41/122)`. The CARD retains those entries at the third token. Independent rational enumeration of all eight hidden paths gives the revised first-state posterior `(19455/35494,16039/35494)`, approximately `(0.5481208092,0.4518791908)`. The count difference is `0.11581361707866589`. Substituting just those revised paid counts, while keeping the same final filter, changes the forecast from `(0.2336989379,0.7663010621)` to `(0.2411988292,0.7588011708)`.

The counterexample isolates smoothing scope and **does not implement Cappé's entire adaptive estimator**. The observed parameter path is supplied only for this diagnostic. No full-estimator admission is based on it. It also does not refute every fixed-lag quasi-Bayes estimator cited by the original critic. A complete published derivation must still establish the CARD's exact permanent lag-one deposit, positive pseudo-count initialization, omission of the latest token, stale-parameter ordering, forecast timing and outer training interpretation. The accessible Cappé source does not close those requirements. The original numerical comparator reproduces the CARD's deposit schedule; that agreement alone does not provide the missing independent source.

## Tensor: the old comparison imports the CARD's learner result

The [original program, lines 35-67](../../hybrid/tests/research/fixtures/reference_replays/programs/remaining-card-6fc7efec.py) identifies a row-comparison cycle and compiles permutations. Its evaluator then contracts the shared operation at each postfix MUL. The original comparator receives `card.Z` directly at **line 189**. Thus its reported zero difference does not test an independently implemented calibration learner.

[Socher et al., EMNLP 2013, sections 4.3-4.4](https://aclanthology.org/D13-1170.pdf) specifies shared tensor composition, a node classifier, cross-entropy training and tensor backpropagation with AdaGrad. Its composition has the form

\[
h=f\big([u;v]^\top V^{[1:d]}[u;v]+W[u;v]\big).
\]

A suitably restricted bilinear cross-term explains the arithmetic of one contraction. It does not specify the CARD's cycle identification from ROW0, ROW1 and COL0, calibration rejection/reset behavior, parameter overwrite, or ASK/stack protocol. Removing the nonlinearity and using a root message as a distribution does not supply a learner for the missing operation.

The captured hidden permutations provide a check independent of `card.Z`: all 49 entries of each of the three compiled operations agree with the corresponding generated hidden table. All 60 soft-message contractions agree exactly when accumulated in the same floating-point order. These are positive component diagnostics, not an independently published full calibration theorem.

A two-label handcase calibrates addition mod 2, evaluates `0 0 MUL ASK`, then calibrates complemented addition and repeats the same expression. The untouched CARD answers 0 and then 1. An inference tensor supplied once cannot account for both answers without a specified parameter-update rule. This witness rejects treating the inference-only comparison as a full learning comparison; it does not assert that every possible tensor learner fails. No frozen tensor is admitted as a repair. The identity/gauge reconstruction, complete parser behavior, output/reset schedule and learning procedure remain outside the cited RNTN definition.

## Consolidation: published NGE's nearest read differs from hard subset abstention

The [original CARD program, lines 23-55](../../hybrid/tests/research/fixtures/reference_replays/programs/remaining-card-dc69816b.py) builds a learned cortical state and top-k code, chooses the most specific **contained** key, refreshes its time before writing, and learns only on prediction error. Consolidation changes a key by intersection while retaining its existing residual. Eviction uses the last-use stamp. All 400 random parameter settings, in addition to the shared toy settings, remain variable in the captures. The notebook records 8,405 writes and 525 consolidation events.

[Salzberg, A Nearest Hyperrectangle Learning Method, section 3.2, pages 257-260](https://link.springer.com/content/pdf/10.1023/A:1022661727670.pdf) classifies by the nearest stored exemplar, including a nonzero-distance exemplar outside the query. Its distance uses feature and exemplar-reliability weights; nesting resolves containment. With exactly one stored exemplar, the nearest choice is independent of those weights. The CARD instead falls back to its base predictor whenever no stored key is a subset of the query code.

The three-token witness `[0,3,1]` is executed through the untouched CARD function with a fully specified six-dimensional encoder. It stores one key `{0,1}` with residual `(0,0,0,0,0,1)`. The final query code is `{0,2}`, so the CARD adds no residual. Published single-exemplar NGE returns the sole exemplar's residual under the critic's residual-as-target mapping. The hidden-vector difference is exactly 1. Under the same nearest-embedding softmax, the last token receives `0.005310984579836764` from the CARD and `0.03795525840089027` from the NGE-read witness.

**Salzberg already permits continuous targets with an error tolerance**, in section 3.1, page 257. Continuous residuals or nontransitivity of a tolerance relation are therefore not used here as a refutation of NGE. Likewise, [Mitchell's Find-S algorithm, slide 2-8](https://www.cs.cmu.edu/afs/cs/project/theo-20/www/mlbook/ch2.pdf), can specialize to positive-only conjunctions, where intersection is the right primitive. In the unrestricted Boolean attribute space it retains shared negative constraints too. These domain distinctions are tested without claiming they exclude the positive-only specialization.

The direct nearest-read mapping is refuted. A different composite with hard-subset abstention, specificity/recency ordering, overlap-threshold candidate selection, retained residuals, LRU capacity and the CARD's learned projection/code loss still needs an independently specified full derivation. The original critic explicitly says that no single published system combines its proposed pieces. Its saved `nge_run` implements those pieces in another data layout; it cannot supply independent publication evidence for them. No claim is made that no such derivation could exist.

## Exact-code accounting

Old replay means the immediately preceding `NEXT.md` checkpoint, as stored in `next-final-phase7-rows.json`. New replay comes from the actual `rows` fixture of the unchanged Phase7 tests in this run. `none` means an empty code list. `remaining-rows.json` retains every exact code list, including the original source-run codes, for all 21 rows. All 21 old/new replay code lists are identical. The two cards without an S2 death expectation still pass S2.

| Card | Required exact code | Old replay | New replay | Evidence classification |
| --- | --- | --- | --- | --- |
| `1184f7a7` | `excluded_family:linear_attention` | `excluded_family:linear_attention` | `excluded_family:linear_attention` | Prior supported comparison unchanged |
| `14429dbc` | `reduces_to_known:other_known` | none | none | Assessed; complete exemplar/GCD composition unverified |
| `2c5dd003` | `reduces_to_known:other_known` | none | none | New full capture; local lag-one identity verified; full permanent-deposit estimator source gap |
| `36725a7c` | `reduces_to_known:other_known` | none | none | Prior SUGS allocation refutation retained |
| `537e161c` | `excluded_family:delta_rule` | `excluded_family:delta_rule` | `excluded_family:delta_rule` | Prior supported comparison unchanged |
| `582f8abf` | `reduces_to_known:other_known` | `reduces_to_known:other_known` | `reduces_to_known:other_known` | Prior parity reference unchanged |
| `68612a2a` | `reduces_to_known:other_known` | none | none | OMP-related allocation/value/FIFO composition remains unverified |
| `6de11b01` | `excluded_family:delta_rule` | none | none | Tagged-memory mapping still exceeds written input-gate repair policy |
| `6fc7efec` | `not_radical:R2` | none | none | New full capture; sampled tensor arithmetic verified; full calibration reduction unsupported |
| `7ec5e5c2` | `reduces_to_known:other_known` | `reduces_to_known:other_known` | `reduces_to_known:other_known` | Prior homomorphism reference unchanged |
| `8dc4db49` | `reduces_to_known:other_known` | `not_radical:R2` | `not_radical:R2` | LFU counts and warm-started dustbin Sinkhorn composition remains unverified |
| `a36f1a3a` | `reduces_to_known:other_known` | none | none | Nonlinear masked PPGN composition remains unverified |
| `bc3935ac` | `reduces_to_known:other_known` | `reduces_to_known:other_known` | `reduces_to_known:other_known` | Prior forest reference unchanged |
| `d1d383d5` | `reduces_to_known:hopfield_or_kernel_memory` | `not_radical:R2` | `not_radical:R2` | Existing cubic-DAM diagnostic retained; no repeated energy-source search |
| `d86dca4b` | `reduces_to_known:other_known` | `reduces_to_known:other_known` | `reduces_to_known:other_known` | Prior projective reference unchanged |
| `dc69816b` | `reduces_to_known:other_known` | none | none | New full capture; direct NGE read mapping refuted; complete composite unsupported |
| `ddb071bd` | `not_radical:R2` | none | none | Existing ART/leader counterexample retained |
| `ff8f801a` | `not_radical:R2` | `not_radical:R2` | `not_radical:R2` | Prior supported comparison unchanged |
| `ffdbe331` | `reduces_to_known:other_known` | none | none | Concept empty-support and roundoff-domain counterexamples retained |

The required count remains **7/19, with 12 gaps**. Full exact preservation has not been established under the existing criteria. The tagged-memory policy obstruction remains a reason that restoring all old codes may be impossible by the original mappings. This is not permission to change an expected code or expand a family repair.

## Verification, source preservation and limits

The new file has **24 passing tests**. The network-prohibited current-tree run of the new file plus existing reference replay, historical binding and original Phase7 tests produced **50 passed, two failed; exit 1**. The failures remain `test_phase7_replay.py:29` and `:36`. The actual replay rows and ledger artifacts are retained; they were not regenerated by a second reporting replay. The two failures are not xfailed, skipped or relabeled.

A separate cache-free package excluded `remaining-full-capture.json`, all publication snapshots and raw source-run artifacts. Its new-evidence tests produced **24 passed; exit 0**. It reconstructed every original draw in memory from the packaged unchanged programs. Original artifact/snapshot byte checks are optional when those local files are absent, while program, observer, complete trace and draw hashes remain mandatory. Both verification runs used the main checkout's `.venv-hybrid/bin/python` with sibling/package cwd and PYTHONPATH. A Python audit hook prohibited network calls, including inherited Python subprocesses. No full suite or literature-worker suite was run.

Mutation checks drop the HMM deposit, reverse tensor operand roles, suppress consolidation's key intersection, and perturb the learned projection. Each changes an observed output. The rational HMM witness and published single-exemplar nearest decision do not depend on the old comparators.

The first exploratory new-test execution had 22 passes and two test-authoring failures. One incorrectly demanded bit-exact agreement between NumPy's sequential scalar additions and Python's compensated `sum`; the independent table calculation now preserves the original arithmetic order. The other guessed a probability gap above 0.1; the derived hand answer is the exact positive gap of approximately 0.03264427. The underlying hidden-vector difference is exactly 1. Neither correction changed a production tolerance, original assertion or acceptance rule. The failed exploratory log is retained as `remaining-unit.log`.

Five full independent publication captures and extracted texts are preserved locally. `remaining_publications.json` records their requested/final URLs, retrieval timestamps, PDF/text SHA256 values and relevant sections. The sources are Cappé, Socher et al., Salzberg, Mitchell chapter 2, and the NNGE comparison paper by Daniela Zaharie, Lavinia Perian and Viorel Negru. The NNGE paper is contextual evidence distinguishing non-nested conflict-excluding learners from EACH; no NNGE result is used to claim an NGE refutation. The source list is a bounded investigation, not an exhaustive search of every paper cited by the historical critic.

`remaining_protected.json` checks 62 protected tracked files, including root code/tests, README, EXPERIMENT, math, critic schemas, the host registry/contracts and original Phase7 fixture/helper/assertions. None changed. No host operation, schema entry, accepted declaration, policy, preregistration or tolerance was added or modified. Concurrent changes to `hybrid/literature/seed.py` and `build_seed.py` belong to the other worker and are outside this evidence delivery.

## Next evidence work

For `2c5dd003`, obtain the full equations of an independently published finite-lag or quasi-Bayes estimator before adding any reference. Check permanent deposition, emission pseudo-counts and forecast timing together. The local smoothing identity is already covered and should not be treated as the missing proof.

For `6fc7efec`, look for a published cyclic-isotope identification derivation that covers the learner, rather than another tensor evaluator. Preserve the calibration/reset behavior and avoid importing Z from the CARD into a purported independent learner.

For `dc69816b`, either specify and source a complete residual-rule composition or keep the negative result. Do not rename the current `nge_run` as a published operation. Treat positive-only Find-S and nearest-exemplar NGE as distinct ingredients with different read semantics.

For `14429dbc`, the remaining question is a full lattice-closure/cache/schedule composition, not whether gcd computes a subgroup generator. No new conclusion about this card is inferred from its old scalar comparison. SUGS, concept support failures, tagged-memory policy and cubic-DAM limitations remain unchanged.
