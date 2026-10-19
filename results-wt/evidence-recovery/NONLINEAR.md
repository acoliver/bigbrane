# Neither nonlinear comparison supports a new exact code; the replay remains 7/19

Both remaining nonlinear cases now have complete source reexecutions, publication-bound equations and small counterexamples to the stated mappings. Neither has a proved full published reduction. The unchanged Phase 7 replay still matches **7/19 required exact codes**, leaving 12 gaps. Its two aggregate assertions still fail. No operation, schema member, declaration, acceptance criterion or family-policy repair was added.

The conclusions concern the specified comparisons below. They do not establish novelty, exclude every wider PPGN or ART variant, or justify forcing all 19 historical codes.

## Source execution and provenance

All writes are in `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, branch `r2-bar`, HEAD `4dd4864`. Main remains clean on `p4-diagnosis`, HEAD `8347c7f`. Existing tracked files in the sibling have no changes. No commit, push, integration, merge, independent review or live model call was performed. The other worker's tools were used only as held-out offline controls and were not changed.

The complete original cards and critiques were read, including both learning specifications. The source programs are selected from executor artifacts attributed by the original run's ledger to each card and checked for `purpose=reduction_check`. PPGN uses job `job-70e748c284d9`, ledger sequence 1213. The complex case uses `job-7292aee625ba`, sequence 1143. The embedded `files/main.py`, original job input and packaged program have identical bytes. Complete card/critic contents, original artifact receipts, executor ledger events and the source ledger file hash are retained in `nonlinear_specimens.json` and `nonlinear_sources.json`.

`nonlinear_capture.py:42-124` uses `sys.settrace` without inserting or rewriting source. The original comparators also execute, but their outputs are diagnostic only. Observer, program, normalized full card/critic, original artifacts, ledger, draw set and complete CARD trace all have SHA256 receipts. Changed card summaries or critic mappings cannot borrow the evidence; changed historical codes or scalar verdicts do not choose evidence.

| CARD | Untouched source execution | Captured inputs, parameters and states |
| --- | --- | --- |
| `a36f1a3a` | 36 seeded streams, 1,440 tokens, 489 queries, 1,980 solver steps | Every token, all 36 alpha/beta draws, signed pair table after every token, every G/H update and every answer, including diagonal/non-query conventions |
| `ddb071bd` | 115 seeded streams, 2,119 tokens, 467 merges, 950 evictions | Every Theta, capacity, vigilance, beta and prior draw; embedding, relation spectrum, all coherence scores, winner, merge z and modulus, every prototype/mass/pointer, evicted record, scores and output |

These are **source reexecutions with `PYTHONHASHSEED=0` and NumPy 2.5.3**, not recovered historical random-input bytes. Historical raw draws and intermediate states were not saved. Seeds, source code and original sampling bounds are unchanged. Neither program samples a set-to-list conversion. The previously recorded forest hash-seed limitation is unchanged.

The local complete capture is 76,116,517 bytes and remains under the ignored results tree. Tests always reconstruct complete traces in memory from packaged programs and the hash-bound observer; there is no raw-capture fast path. The compact evidence is 147,696 bytes: four complete source episodes plus six small handcases. It does not package the complete capture.

## Masked pair closure versus published PPGN

### The published block differs from the scalar comparator's stated block

[Maron et al., Provably Powerful Graph Networks, section 6, equation 6 and block definition](https://arxiv.org/pdf/1905.11136) applies three featurewise MLPs to the **current** pair tensor X and forms

\[
W_{ijc}=\sum_k m_1(X)_{ikc}m_2(X)_{kjc},\qquad B(X)=[m_3(X),W].
\]

The network ends with an invariant graph layer and an MLP. Section 7 adds an MLP m4 after each block and takes m3 to be the identity of that block's current X. Its graph classification/regression readouts use invariant diagonal/off-diagonal pooling and learned heads. The relevant sections and literal equation excerpts are retained with PDF/text hashes and requested/final URLs in `nonlinear_publications.json`.

The CARD's only learned parameters are two scalars. It has no learned token embedding or learned output head: token decoding, the signed pair table and the selected-pair probability head are fixed. Its recurrence is

\[
H^0=0,\quad H^{k+1}=M\odot\tanh(\alpha E+\beta M\odot(H^kH^k)),
\quad K=\lceil\log_2d\rceil+2.
\]

The original program's `ppgn_readout` supplies the unchanged evidence E to **every** layer, multiplies by an external diagonal mask and reads the queried entry. Those additions appear directly at `programs/nonlinear-card-a36f1a3a.py:53-62`; the CARD solve is at `:8-34`. Paths in this section are relative to `hybrid/tests/research/fixtures/reference_replays`. Numerical agreement of that comparator verifies its supplied recurrence, not its attribution to the published single-channel block.

A minimal two-site witness has one positive demonstration on pair (0,1), alpha 0.7 and beta 1.1. The masked CARD answers

\[
p(y_0)=(1+\tanh(0.7))/2=0.8021838885585817.
\]

With m1=m2=m3=id, the claimed one-unit m4 and current X initialized at zero, the published block substitution never receives E and stays zero. Granting the first embedding `X=tanh(alpha E)` still does not make current X equal the original E at later layers. Its off-diagonal changes, and its unmasked square creates diagonal terms. The direct block handcheck on `X=[[0,2],[2,0]]` produces product diagonal `(4,4)`, rather than zero. These tests refute the **stated one-channel substitution**. They do not refute an explicitly derived wider network that preserves E and identity information in additional channels.

A separate three-site witness separates a count-table read from the nonlinear solver. Demonstrations (0,1,0) and (1,2,0) leave `E[0,2]=0`. Nevertheless the second solver step has

\[
H^2_{02}=\tanh(1.1\tanh(0.7)^2)>0,
\]

and the fourth-step query probability is about 0.770179. A direct count-only read returns 0.5. Three sites and two demonstrations are the smallest undemonstrated path of this kind. This is a component witness, not a full comparison with a PPGN learner.

A third witness issues ASK(0,1) and ASK(0,2) on the same evidence table. It returns about 0.802184 and exactly 0.5. A published graph-invariant output that receives only that table has no query selector. A pair-conditioned readout needs an explicit mapping. The observation does not prohibit adding properly specified query features.

### Inference and training scope

Scalar summation and scalar tanh arithmetic independently check all 1,980 captured solver steps, every pair-table update and all 1,440 outputs. The query solver resets to zero; no equilibrium or infinite-parameter limit is substituted. Alpha/beta remain free across all original draws. The old comparator's historical maximum difference is `8.881784197001252e-16`; local reexecution gives `2.220446049250313e-16`. Neither scalar establishes the missing full mapping.

The card trains `alpha=softplus(a0)` and `beta=softplus(b0)` by query NLL through the finite unroll. It specifies Adam, a noisy relational-context generator, d=32, 96 demonstrations and 32 subsequent queries. The original reduction-check program executes no training. New diagnostic forward sensitivities retain both product derivative terms, the mask, softplus and selected-pair NLL. They match central differences for two free parameter settings. A first bias-corrected Adam step decreases the handcase loss; truncating the derivative path changes the gradient. These are learning arithmetic controls, not execution of the original 20,000-context experiment or proof of equivalence to a published PPGN optimizer trajectory.

The remaining full-operation gap is specific: derive evidence retention across layers, diagonal treatment, tied finite depth, selected-pair inference, token write/query/answer timing and the constrained two-parameter training map together. Published matrix multiplication plus an arbitrary learned MLP is insufficient evidence for this exact composition. No learned embedding, frozen learned head, input-dependent repair gate or label-dependent state correction was invented to close it.

## Complex fragments versus specified clustering algorithms

### Phase-normalized merging loses the arithmetic resultant's magnitude

The CARD's merge is `P <- phase(m*P+w)`, followed by incrementing m. P stores a unit phasor, not the unnormalized sum. After normalization, m*P generally differs from the sum of its m members. That distinction affects the next update.

[Kulis and Jordan, Revisiting k-means, section 3.1, Algorithm 1](https://arxiv.org/pdf/1111.0352) specifies global-mean initialization, nearest squared-Euclidean assignments, recruitment when distance exceeds lambda, and recomputation of each arithmetic mean over its assigned points until convergence. Section 3.2 gives the penalized objective. It is a batch reassigning algorithm, without a fixed slot cap, protected item or last-merged output. The independent implementation at `hybrid/tests/research/nonlinear_replays.py:122-141` executes those passes; it does not rename the saved `ARTPool`. Empty clusters are removed before mean recomputation, an explicit convention for the paper's otherwise undefined empty mean. The distinguishing phase witness has one nonempty cluster throughout and does not depend on that convention.

The small phase witness uses d=1, two token phases `(0,pi/3)`, token sequence `[0,0,1,1]`, vigilance 0.4 and capacity 2. Its three relation phasors are

\[
1,\quad e^{i\pi/3},\quad 1.
\]

All belong to the same cluster. After the second relation, the CARD stores `exp(i*pi/6)` with mass 2. Its third merged phase is

\[
\arg(2e^{i\pi/6}+1)
=\arctan(1/(\sqrt3+1))=0.35087941087225777.
\]

The arithmetic mean is `(5/6,sqrt(3)/6)`, with phase

\[
\arctan(\sqrt3/5)=0.3334731722518321.
\]

The phases differ by about 0.0174062. Even granting a final phase projection and the CARD's readout to the published mean changes the next-token probability by more than 0.001. Three relations are minimal for this directional distinction: at two equal-weight unit phasors, projecting their arithmetic mean gives the same direction as the CARD's first merge. At two relations, the unprojected arithmetic prototype already differs in magnitude.

The original merge/update is at `programs/nonlinear-card-ddb071bd.py:14-33`; its head is at `:34-42`. The witness does not claim that every possible spherical or manifold clustering update is excluded. It rejects the critic's identification with an ordinary count-weighted running mean, even when a final projection is granted.

### Fuzzy ART supplies a complete, different learning/search procedure

[Brito da Silva, Elnabarawy and Wunsch, ART survey, Algorithm 1 and section 2.1.3, equations 15,16,19,22](https://arxiv.org/pdf/1905.11437) specifies complement coding, fuzzy-min intersection, distinct choice and match scores, inhibition/search on failed vigilance, category commitment and the update

\[
T_j=\frac{\|x\wedge w_j\|_1}{a+\|w_j\|_1},\quad
M_j=\frac{\|x\wedge w_j\|_1}{\|x\|_1},\quad
w'_j=(1-\eta)w_j+\eta(x\wedge w_j).
\]

The independent implementation includes the always-available uncommitted category in the choice competition. It selects an output category upon commitment. It does not replace these equations with complex coherence and phase averaging. The source is a survey stating the complete algorithm; a guessed primary-paper URL did not return a PDF and is not treated as evidence.

A rational fast-learning handcase with raw inputs `(1/4,3/4)` then `(1/2,1/2)`, choice 0.1 and vigilance 0.5 yields one weight vector `(1/4,1/2,1/2,1/4)` and choice score `5/7`. A separate vigilance handcase exercises actual failed-category inhibition before commitment. Affinely encoding the same three relation phasors from the phase witness into `[0,1]^2` yields a Fuzzy ART rectangle. Even granting its center a phase decoder gives pi/6, which differs from the CARD's third phase by about 0.172719. This extra decoder is a diagnostic concession, not a published HRR readout.

With just one relation, Fuzzy ART already selects the recruited category. The CARD leaves its active pointer undefined and returns only `softmax(u)`. Its pointer moves on **merges only**. That distinction requires an additional output/selection protocol even before capacity is reached.

### Pinning changes the eviction decision

With K=2, group-Z4 lattice codes and relation shifts `[1,1,2,2,2,1,3]`, the first two fragments have masses `(3,3)` immediately before the last novel relation. The older fragment is active. Ordinary LFU/FIFO chooses that older index 0. The CARD excludes it and evicts index 1. This witness is short and avoids approximate vigilance ties. Removing active-slot exclusion changes an observed output. The source's capacity/pointer behavior is not supplied by Fuzzy ART or DP-means.

The complete source trace is checked at every token: relation construction, coherence arithmetic, captured argmax and threshold decision, normalization, masses, victim, protected pointer/reindexing, bind scores, priors and softmax. Scalar coordinate arithmetic checks the complex merge and readout. The diagnostic starts each step from the captured prior state and validates the recorded decision; it is not an independently published full learner. No routing tolerance is added to production. Roundoff bounds in the new diagnostic arithmetic assertions do not repair a selection or authorize admission.

### The historical comparator mismatch is preserved, not explained away

The saved original comparator reports `equivalent=false`, maximum difference **0.44461806885749666**. Untouched local reexecution reports `7.216449660063518e-15`, with the same total 467 merges and 950 evictions. The historical parent comparison remains about 0.982, also in local stdout.

The saved `ARTPool` already contains the CARD-specific coherence, phase normalization, slot replacement and last-merged head. In exact unit-phasor algebra it implements the same operations. Its numerical evaluation differs: exp versus cos/sin embeddings, conjugate multiplication versus division, vdot versus real sums, and abs versus hypot. A plausible cause of the historical/local difference is a threshold or winner decision sensitive to those arithmetic paths. Different NumPy/BLAS/runtime or seeded draw realizations are also possible. **None is localized by the available historical record.** There is no saved divergent step or raw parameter/input trace to compare with the current execution. No retries or arithmetic/tolerance changes were made to manufacture a mismatch or agreement. The independently derived publication witnesses above do not depend on choosing an explanation.

### Learning remains a separate unsourced composition

The complex card's claim 5 learns Theta with relation-coherence positives, negative hinges and a distinctness hinge, then uses hard inference. Its learning specification explicitly says that merge, eviction, pointer and readout do not run during training. It is not end-to-end differentiation through ART decisions. The optional soft-margin extension is outside the core comparison and is not substituted for it.

New analytic gradients include all six positive relation pairs, all 39 negative pairs and distinctness over every code, including the delimiter. Central differences and a loss-decreasing update check selected coordinates. Dropping negative or distinctness terms changes both loss and gradient; perturbing Theta changes inference. No 2,000-step/five-seed training run or held-out capability benchmark is claimed. A full independent reduction would have to source this phase-training procedure, sequential projected merging, bounded protected pool, merge-only active selection and tied HRR head together. The name ART/leader does not supply that operation.

## Verification and reproducibility

The final current-tree targeted run produced **108 passed, two failed**, exit 1. It includes all **38 new evidence cases**, existing reference replay, historical binding and review-gate controls, and the original Phase 7 test file. The failures are unchanged assertions at `hybrid/tests/research/test_phase7_replay.py:29` and `:36`. They remain ordinary failures, not xfails or skips.

The cache-free focused package produced **40 passed, two failed**, exit 1: all 38 new cases pass, with the same two Phase 7 failures and its two passing controls. It has no raw capture, publication body, literature cache or accessible original-run tree. Original-run existence is masked absent, and an audit hook prohibits opening that external run. Network calls are prohibited in both local and isolated verification, including inherited Python subprocesses. The main virtualenv Python runs with sibling/package cwd, PYTHONPATH and explicit confcutdir. No full suite ran.

All six handcase states are reobserved from the unchanged CARD function. Mutants change signed writes, the quadratic term, both diagonal masks, mass weighting, phase normalization, vigilance, active selection and protected-slot exclusion. Learning controls detect truncation and dropped loss terms. No existing assertion, expected code, preregistration, family policy or acceptance tolerance was edited. All 1,030 preexisting tracked files are hash-protected, including prior negative evidence and root `bigbrane`, tests, README, EXPERIMENT and math files.

The initial new-file run had 27 passes and four test-authoring failures. The pair-head witness encoded a diagonal query instead of the intended pair. Removing only the outer mask had no effect because E and the inner-masked G already had zero diagonals. An eviction-order mutation had no effect with one eligible victim. The Fuzzy ART handcheck initially expected inhibition even when the uncommitted category won the choice competition first. The input encoding, mutation targets and inhibition handcase were corrected from these mechanisms. No production or preexisting test changed. The initial logs remain preserved. A later snapshot check found that Python splitlines also split PDF form-feed characters, so some excerpt ranges stopped before the cited equations. Excerpt extraction now counts newline characters only, and tests require the PPGN block, ART output step and equation 22 to be present. The final local and isolated delivery runs include that correction. Counts from different executions are not added together.

Reproduce from the sibling:

```sh
PYTHONHASHSEED=0 BIGBRANE_LIVE=0 \
PYTHONPATH="$PWD/results-wt/evidence-recovery/composites-network-guard:$PWD" \
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python \
results-wt/evidence-recovery/nonlinear_verify.py reproduce controls
```

Use `nonlinear_package.py <new-label>` for a fresh isolated package; existing package directories are rejected. The expected recorded command result is exit 1 because the original Phase 7 assertions still fail. `nonlinear-tests.json` lists separate logs, JUnit results and exact row files.

## Actual unchanged Phase 7 rows

The rows below come from the real `rows` fixture captured during the unchanged tests, not a second reporting replay. `nonlinear-delivery-phase7-rows.json` and `nonlinear-isolated-delivery-phase7-rows.json` contain all 21 records. All code lists match the preceding COMPOSITES checkpoint. `none` denotes an empty code list.

| Card | Required exact code | Actual replay |
| --- | --- | --- |
| card-1184f7a7 | excluded_family:linear_attention | excluded_family:linear_attention |
| card-14429dbc | reduces_to_known:other_known | none |
| card-2c5dd003 | reduces_to_known:other_known | none |
| card-36725a7c | reduces_to_known:other_known | none |
| card-46dbb349 | none | none |
| card-537e161c | excluded_family:delta_rule | excluded_family:delta_rule |
| card-582f8abf | reduces_to_known:other_known | reduces_to_known:other_known |
| card-68612a2a | reduces_to_known:other_known | none |
| card-6de11b01 | excluded_family:delta_rule | none |
| card-6fc7efec | not_radical:R2 | none |
| card-7ec5e5c2 | reduces_to_known:other_known | reduces_to_known:other_known |
| card-8dc4db49 | reduces_to_known:other_known | not_radical:R2 |
| card-a36f1a3a | reduces_to_known:other_known | none |
| card-bc3935ac | reduces_to_known:other_known | reduces_to_known:other_known |
| card-c9615069 | none | none |
| card-d1d383d5 | reduces_to_known:hopfield_or_kernel_memory | not_radical:R2 |
| card-d86dca4b | reduces_to_known:other_known | reduces_to_known:other_known |
| card-dc69816b | reduces_to_known:other_known | none |
| card-ddb071bd | not_radical:R2 | none |
| card-ff8f801a | not_radical:R2 | not_radical:R2 |
| card-ffdbe331 | reduces_to_known:other_known | none |

Required exact matches remain 7/19; 12 gaps remain. The two original S2-pass cards still pass. Earlier sourced negatives are unchanged: SUGS 1174/2418, concept empty/nonfinite support, tagged-memory policy, permanent HMM deposits, tensor calibration, NGE subset abstention, GCD commit/FIFO, OMP allocation/FIFO and floating selection, LFU wrapping, DAM diagnostic-only scope and forest hash-seed limits. None was reinvestigated or relabeled.

## Compact inventory, checksums and next root gap

The required set is listed with exact byte sizes and hashes in `nonlinear-INVENTORY.json` and `nonlinear-SHA256SUMS`. It includes the two untouched programs, observer, diagnostic helper, new test file, compact evidence/specimen/source/publication/protection JSON, verification/package/fetch/measurement scripts and this report. Files under the ignored results path are explicitly listed; that list is needed for later packaging. Nothing is staged or committed here.

The full local capture and complete PDFs/texts are separately listed in `nonlinear-LOCAL-SHA256SUMS`. They remain unchanged locally and are not delivery prerequisites. Program/observer/trace hashes are mandatory even when those local bodies are absent. Publication hashes establish integrity of retrieved bytes, not a publisher signature.

| Receipt | SHA256 |
| --- | --- |
| PPGN original program | `d193e05c1cb85ca59c798ba55a07bd1417eba68c1350e3ba66726a00dbe7b3b8` |
| Complex original program | `ab7af5e57788d469b0af43bdcaba7c2c19f5273c5278d688d93eb6e90b4a945b` |
| Observer | `06d44296b601a1a339b09fb976416d0da57270550ff992ce79e13a6946e6e8b3` |
| PPGN complete CARD trace | `6bc7dcadd795a9bee3ebf96d0b7f1b6910e71a5393d80bfb1ed67bf48651bee7` |
| Complex complete CARD trace | `72fe7f58f52ea8195bb168f95c0b258871577282871d262da6e4352e5705d531` |
| Original ledger file | `9dac183f8fa511a89e6e7c228ef97130acd466fbe26c11bb6c1daffeb053d114` |
| Full local capture | `670b2050378c8018c90ca744d52d7841e66d797eb2452328ee0f036f0deb74eb` |
| PPGN PDF | `30f0923967ef7cd852f87bd3eaf9ae4e6c31fae13dd5d225b2b184ffa22a5f29` |
| DP-means PDF | `e725700a6fb4e32cc27b194f2714c0868fd6a14c5a055df331fe62a5da677ba3` |
| ART survey PDF | `ca361cf69752575b84f8243bfbda0bc77c75802b7bf8dbf5ce1116db52b1ca37` |

The next root gap for PPGN is a full exact architecture-and-training derivation with retained evidence channels, diagonal handling and a query-conditioned head. The current witnesses prevent treating the critic's scalar single-channel formula as that derivation. For the complex case it is an independently specified projected sequential estimator with protected replacement and merge-only activation, plus its separate phase-learning objective. The arithmetic-mean and Fuzzy ART identifications tested here do not provide it. Local comparator agreement does not change either gap.
