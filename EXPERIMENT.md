# Experiment 1: measure the benefit and cost of a four-state prior

Protocol version 1, dated 2026-09-30. Status: specified, not implemented or run. [RESEARCH.md](RESEARCH.md) contains the literature findings and source-access record. This protocol must be frozen in a copied run manifest, with document hashes, before confirmatory test evaluation. Amendments require a new version and a stated reason.

## 1. Question, hypothesis, and scope

The question is whether a learned four-state HMM requires fewer observed training tokens than two tiny neural controls to predict sequences from several independently sampled HMM generators, and whether any advantage survives a fixed violation of the four-state Markov assumption.

The hypothesis is that the correctly specified HMM prior improves low-data estimation, with a possible bias cost when the true predictive state is larger. This is a test of specified learners and budgets. There is no proposed new architecture.

The learned competitors are:

- A categorical four-state HMM fit by Baum-Welch with fixed weak smoothing.
- A single-layer, ungated tanh RNN with hidden width eight.
- A single-layer, single-head causal attention block with width eight and a width-sixteen feed-forward sublayer.

The true-generator oracle is an evaluator reference only. It cannot receive credit as a learned model. A smoothed unigram predictor is a context-free diagnostic, excluded from the primary comparison. Retrieval, composition, learned state counts, spectral fitting, and hierarchical summaries are excluded.

## 2. Generator family and independent replication

### 2.1 Matched HMMs

Set K=4 hidden states, V=8 observed symbols, and L=64 tokens per independent sequence. Symbol IDs are 0,...,7. BOS is input-only ID 8. There is no EOS, padding, tokenizer learning, or variable-length preprocessing.

Use two prespecified regimes:

| Regime ID | Name | Persistence rho | Emission strength eta |
| --- | --- | --- | --- |
| 0 | Mixing, more informative emissions | 0.60 | 0.80 |
| 1 | Persistent, ambiguous emissions | 0.95 | 0.40 |

For each regime and generator index g in {0,1,2,3}, independently draw a dense row-stochastic R by drawing its four rows from Dirichlet(1,1,1,1). Set

\[
A=\rho I+(1-\rho)R.
\]

Independently permute the eight symbol IDs. Assign the first two to hidden state 0, the next two to state 1, and so on. Let S_{iv}=1/2 for the two symbols assigned to state i and zero otherwise. Define

\[
B_{iv}=\eta S_{iv}+(1-\eta)/8.
\]

Compute pi as the unique stationary distribution satisfying pi A=pi and sum_i pi_i=1. Solve this linear system in float64, replacing one redundant stationarity equation with the normalization equation. Assert nonnegative normalized probabilities and stationarity residual below 1e-10. All entries of A and B are positive. The overlapping emissions leave latent-state uncertainty even though pairs identify a preferred state.

Sample z_1 from pi, emit x_1 from B[z_1], then sample z_t from A[z_{t-1}] and x_t from B[z_t] for t=2,...,64. Draw each sequence with a fresh initial state; never concatenate sequences before fitting or scoring.

These give eight independently sampled generator matrices, four per regime. The two regimes jointly change persistence and emission strength. They cannot separately identify the effect of either factor. Report both regime results and the aggregate. This small family is intentionally favorable to an HMM; no general sequence-distribution claim is warranted.

### 2.2 Random streams and artifacts

Use an explicitly recorded PRNG and independent streams. Preferred data-generation definition: NumPy PCG64, seeded by SeedSequence from integer tuples below. A later implementer must verify NumPy availability rather than assume it. An alternative PRNG requires a protocol amendment before any confirmatory data are drawn. Once generated, saved token arrays and matrices, rather than a package-dependent seed alone, define the run data.

For seed tuples, use root 20260930 and integer roles:

| Role | Seed tuple |
| --- | --- |
| Generator matrices and symbol permutation | (root, 0, regime, g) |
| Matched training master dataset | (root, 1, regime, g, 0) |
| Matched validation | (root, 1, regime, g, 1) |
| Matched test | (root, 1, regime, g, 2) |
| Stress datasets | (root, 2, regime, g, split), with split 0/1/2 |
| Training randomness | (root, 3, stage, regime, g, size_index, family, search_replica, candidate, role) |

Stage is 0 for matched and 1 for stress; families are 0 HMM, 1 RNN, 2 attention. Use role=0 for initialization. For neural initialization, convert its SeedSequence to one uint64 seed word and pass that integer to the selected neural library, recording the conversion and library version. For HMM initialization use PCG64 with its SeedSequence. For neural shuffling, use role=1 and append the one-based epoch number to the tuple, then draw that epoch's sequence permutation with PCG64. Stress sample sizes retain matched size_index values 1 and 3. Generator randomness is never a model-initialization seed. No global PRNG reseeding inside a data loop is permitted.

For the neural learning-rate candidates, initialize from identical weights within a search replica and use identical epoch shuffle seeds. In their tuples set candidate=0 for both initialization and shuffling; the learning rate is their only intended difference. HMM restart candidates use distinct candidate IDs. Families and search replicas have separate random streams.

Save A, B, pi, S, the symbol permutation, token arrays, seed tuples, library versions, and file hashes. Latent paths may be saved for generator diagnostics in an evaluator-only artifact. Training code must not read them. The learner data interface exposes only observed tokens and sequence boundaries; the HMM additionally receives the declared K=4. Generator matrices and oracle distributions are evaluator-only.

### 2.3 Training sample sizes and split definition

Generate 256 independent training sequences once per generator. Use nested subsets comprising the first n sequences in a fixed stored order:

| Size index | Independent sequences n | Observed training tokens N=nL | Scored targets N | Noninitial conditional targets n(L-1) |
| --- | --- | --- | --- | --- |
| 0 | 8 | 512 | 512 | 504 |
| 1 | 32 | 2,048 | 2,048 | 2,016 |
| 2 | 128 | 8,192 | 8,192 | 8,064 |
| 3 | 256 | 16,384 | 16,384 | 16,128 |

A training token means one observed symbol in the saved training subset, counted once regardless of training epochs. BOS contributes no observed token or target. Repeated optimization passes are token exposures and must be counted separately.

Use 32 independent validation sequences, 2,048 observed tokens, and 128 independent test sequences, 8,192 observed tokens, for every size. Validation and test have independent random streams from training and from each other, with the same generator matrices. Exact duplicate symbol sequences occurring by chance are not removed. Splits are independent draws rather than disjoint strings.

All learned families receive identical arrays at a given generator, stage, and size. Validation contains more tokens than the smallest training set. Report both training-only N and total accessible observed tokens N+2,048. The oracle-only test set supplies no training or selection information. Reusing validation for repeated checkpoint selection means this is a validation-assisted learning curve, not a training-data-only sample-complexity theorem.

Train each generator's models from scratch. Aggregate replication across generators is not zero-shot transfer to new matrices. There is no pooled pretraining, warm start across sample sizes, or adaptation on test data.

## 3. Prefix alignment and objective

The target at position t is x_t conditioned on x_{1:t-1}, for t=1,...,64. At t=1 the prefix is empty.

For neural models, the input is [BOS,x_1,...,x_63], length 64, and the target is [x_1,...,x_64]. Output position 0 predicts x_1 from BOS; position 1 predicts x_2 from BOS and x_1; position 63 predicts x_64 from BOS and x_1,...,x_63. Never input x_64. Attention may attend to its own input position because that input is already shifted.

Each sequence starts with a fresh RNN state and a fresh HMM prior. Attention positions restart from zero. No state crosses sequence boundaries. Train the neural models by teacher forcing on all 64 targets, with mean negative log likelihood in natural logarithms:

\[
\mathcal L(\theta)=-\frac{1}{nL}\sum_{a=1}^{n}\sum_{t=1}^{L}
\log p_\theta(x_t^{(a)}\mid x_{1:t-1}^{(a)}).
\]

The HMM's observed-data likelihood includes these same 64 symbols, including the initial marginal. It uses smoothing internally during fitting, but only filtering for scoring. No competitor receives hidden-state labels, future oracle probabilities, or generator parameters.

## 4. True HMM oracle

Use row-vector convention throughout. Define q_t(i)=P(z_t=i | x_{1:t-1}). Initialize q_1=pi. At each target position:

\[
p_t^*(v)=\sum_i q_t(i)B_{iv},\qquad
c_t=p_t^*(x_t),\qquad
b_t(i)=q_t(i)B_{i,x_t}/c_t,\qquad
q_{t+1}=b_t A.
\]

Store p_t^* before incorporating x_t. The next-state transition happens after the current emission update. The same implementation, with fitted matrices, scores the learned HMM, subject to independent correctness tests below.

Oracle filtering knows A, B, and pi; the learned HMM knows only V, K, and the token data. Never compare their training sample requirements. The oracle defines conditional distributions and sampled-reference NLL. A learned model can have slightly lower sampled test NLL by chance, but its expected conditional KL from the oracle cannot be negative.

## 5. Learned HMM and Baum-Welch definition

Fit an unrestricted dense categorical HMM with K=4, learning pi, A, and B. Do not constrain the fitted A to the generator's persistence mixture or B to paired symbols. Do not initialize from the generator or the oracle. Each candidate initializes every probability row independently from symmetric Dirichlet(1), including pi.

There are 52 stored probability entries: 4+16+32. Simplex constraints give 43 free scalar parameters: 3+4(3)+4(7). Report both counts. State labels are interchangeable, so raw hidden-label accuracy and matrix-entry error are not outcome measures.

For each training sequence a, obtain by scaled float64 forward-backward:

\[
\gamma_t^{(a)}(i)=P_\theta(z_t=i\mid x_{1:L}^{(a)}),
\quad
\xi_t^{(a)}(i,j)=P_\theta(z_t=i,z_{t+1}=j\mid x_{1:L}^{(a)}).
\]

With fixed epsilon=0.001 added to each expected-count cell, update

\[
\pi_i'=\frac{\epsilon+\sum_a\gamma_1^{(a)}(i)}{K\epsilon+n},
\]

\[
A_{ij}'=\frac{\epsilon+\sum_a\sum_{t=1}^{L-1}\xi_t^{(a)}(i,j)}
{K\epsilon+\sum_a\sum_{t=1}^{L-1}\gamma_t^{(a)}(i)},
\]

\[
B_{iv}'=\frac{\epsilon+\sum_a\sum_{t=1}^{L}\gamma_t^{(a)}(i)\mathbf 1[x_t^{(a)}=v]}
{V\epsilon+\sum_a\sum_{t=1}^{L}\gamma_t^{(a)}(i)}.
\]

This is MAP Baum-Welch with independent symmetric Dirichlet(1+epsilon) priors, not unsmoothed maximum-likelihood EM. The small fixed pseudocount is a disclosed statistical assumption. It prevents zero-probability categories without ad hoc score clipping. Do not tune epsilon on these test sets.

Use full-data E and M steps, capped at 50 iterations per candidate. Score validation with causal filtering at iteration 0 and after every iteration. Keep the lowest validation NLL checkpoint, with earlier iteration winning exact ties. Stop a candidate after three successive updates whose improvement in the penalized training objective, divided by N, is below 1e-6 nats, or at the iteration cap. The maximized objective is

\[
J(\theta)=\log P_\theta(D)+\epsilon\left(\sum_i\log\pi_i+
\sum_{ij}\log A_{ij}+\sum_{iv}\log B_{iv}\right).
\]

An objective decrease beyond numerical tolerance is a correctness failure, not a convergence condition. Record each J, training NLL, validation NLL, row normalization residual, and chosen iteration. Expected occupancy per state is an evaluator diagnostic for collapsed fits.

## 6. Tiny ordinary RNN

Use embedding E in R^(9x8), hidden width h=8, one bias per recurrent update, and an untied eight-symbol output head. For shifted input u_t:

\[
e_t=E[u_t],\quad h_0=0,\quad
h_t=\tanh(W_e e_t+W_h h_{t-1}+b_h),
\quad p_t=\operatorname{softmax}(W_y h_t+b_y).
\]

Use W_e and W_h of shape 8x8, b_h of size 8, W_y of shape 8x8, and b_y of size 8. The model has 280 trainable scalars: 72+64+64+8+64+8. h_t is an unconstrained learned state, unlike the HMM's four-entry probability vector.

Use full backpropagation through each length-64 sequence, no truncation or detach inside a sequence, and no gating, layer normalization, attention, auxiliary losses, or recurrent dropout. Initialize embeddings with independent Normal(0,0.1^2), W_h orthogonally, other weight matrices with Xavier uniform, and biases to zero. Specify the library's orthogonal and Xavier routines in the run manifest.

## 7. One-layer, one-head causal attention control

Use d=8 and feed-forward width 16. Embed the same shifted input and add a fixed positional vector:

\[
v_t=E[u_t]+PE(t),\quad
PE(t,2i)=\sin(t/10000^{2i/8}),\quad
PE(t,2i+1)=\cos(t/10000^{2i/8}),
\]

with zero-based position t=0,...,63 and i=0,...,3. There is no embedding scale factor. One pre-normalized attention block is

\[
a_t=LN_1(v_t),\quad Q=aW_Q,\ K'=aW_K,\ V'=aW_V,
\]

\[
C=\operatorname{softmax}_{\text{row}}(QK'^T/\sqrt{8}+M),
\quad M_{ts}=0\ (s\leq t),\quad M_{ts}=-\infty\ (s>t),
\]

\[
r=v+CV'W_O,\quad
h=r+\operatorname{ReLU}(LN_2(r)W_1+b_1)W_2+b_2,
\quad p_t=\operatorname{softmax}(h_t W_y+b_y).
\]

E has shape 9x8. Q/K/V/O projections have shape 8x8 with no biases. W_1 is 8x16, W_2 is 16x8, with biases 16 and 8. Each layer norm has learned scale and shift of size eight and epsilon 1e-5. Output W_y is 8x8 with bias eight. There is no final normalization. Total parameters are 712: embedding 72, projections 256, feed-forward 280, norms 32, and output 72.

Use explicit dense causal attention, a single head, no cross-attention, no recurrence, no dropout, and no weight tying. Initialize embeddings with Normal(0,0.1^2), all linear matrices with Xavier uniform, biases and normalization shifts with zero, and normalization scales with one. This is a small attention control, not a reproduction of the full Transformer in [RESEARCH.md, S9](RESEARCH.md#sources-accessed-and-access-limits).

## 8. Training and selection budget

Each generator, stage, size, and model family has two independent search replicas r=0,1. A search replica yields one validation-selected model, so uncertainty includes repeated selection procedures rather than a cherry-picked successful initialization.

| Family | Candidates per search replica | Fit cap | Checkpoint selection |
| --- | --- | --- | --- |
| HMM | Two random restarts, same fixed prior | 50 full-data EM iterations each | Lowest validation NLL across all saved candidate checkpoints |
| RNN | Adam learning rates 0.001 and 0.01, identical starting weights within replica | 80 epochs each | Lowest validation NLL across both candidates and all saved epochs |
| Attention | Same two Adam learning rates and initialization pairing rule | 80 epochs each | Same as RNN |

For both neural controls use Adam with betas (0.9,0.999), epsilon 1e-8, constant learning rate, zero weight decay, gradient global-norm clipping at 1, and no label smoothing. Batch size is min(32,n). Shuffle independent sequences each epoch with the stored epoch seed; include the final incomplete batch. Average the loss over all tokens in each batch. An epoch visits every training sequence once.

Evaluate validation at initialization and after every epoch. After at least 20 epochs, stop if ten successive epochs have failed to improve the best validation NLL by at least 1e-4 nats/token; otherwise stop at 80. For stopping, maintain a separate tolerance-based progress record. For checkpoint selection, retain the absolute lowest raw validation NLL among all evaluated epochs, even improvements below that stopping tolerance. Exact ties choose the earlier epoch, then lower learning rate or lower HMM restart ID.

No oracle KL, test NLL, hidden state, or test generator diagnostic enters selection. Validation selects checkpoints using sampled-token NLL only. Failed numerical fits must be reported and investigated; do not silently redraw seeds, omit generators, or change the search grid. Unit-test failures invalidate the affected run. A later fit-budget extension must be separately labeled.

There are equal numbers of candidate fits, but different optimization algorithms, stopping rules, and validation opportunities. This does not equalize compute, optimizer quality, statistical prior strength, or parameter count. Report candidate fit counts, completed iterations or epochs, optimizer steps, validation evaluations, and measured cost, including losing candidates. Neural capacities are fixed controls, not parameter-matched or optimally tuned alternatives.

Matched stage: 8 generators x 4 sizes x 3 families x 2 search replicas x 2 candidates = 384 candidate fits. Stress stage, if triggered: 8 x 2 sizes x 3 x 2 x 2 = 192 additional fits. A CPU correctness pilot uses only one generator, one size, and one candidate per family; it is excluded from outcomes. Use an independent pilot seed root 20260931. Pilot loss results cannot be used to redesign the generator or selection grid under version 1.

## 9. Evaluation and preregistered decisions

### 9.1 Metrics

Primary error is conditional KL in nats/target, computed on actual held-out prefixes using the complete oracle next-symbol distribution:

\[
E_m(N,g,r)=\frac{1}{128L}\sum_{a=1}^{128}\sum_{t=1}^{L}
\sum_{v=0}^{7}p_t^*(v\mid x_{1:t-1}^{(a)})
\log\frac{p_t^*(v\mid x_{1:t-1}^{(a)})}{p_m(v\mid x_{1:t-1}^{(a)})}.
\]

This is expected excess log loss conditional on sampled histories. It removes random variation from the next-symbol draw at each observed prefix; it still depends on a finite sample of histories. Oracle probabilities are used by evaluation only.

Also report sampled test NLL, oracle sampled NLL, their paired difference, and NLL in bits/token by division by ln(2). The sampled difference may be negative. Report prediction error and NLL separately for t=1, t=2,...,64 collectively, and four consecutive sixteen-target blocks, without choosing a favorable block as the primary endpoint. Store per-sequence summaries for error analysis.

Compute neural probabilities with a stable log-softmax: cast evaluation logits to float64 before log normalization and KL accumulation. HMM and oracle distributions use float64. Do not add arbitrary evaluation floors. All specified HMM probabilities are strictly positive; normalized neural log probabilities remain finite for finite logits. A material negative KL or nonfinite value fails correctness checks. Roundoff-sized discrepancies up to 1e-10 are numerical diagnostics, not model victories.

The context-free diagnostic estimates P(v)=(count_train(v)+1)/(N+8). Score it using the same targets and oracle KL. Report oracle conditional entropy and the unigram error to indicate how much prediction benefit is available from context. Do not remove an inconvenient generator after seeing these diagnostics.

### 9.2 Primary low-data advantage test

For each generator g and neural family c, average over both search replicas and N in {512,2,048}:

\[
\Delta_{g,c}=\frac{1}{4}\sum_{r=0}^{1}\sum_{N\in\{512,2048\}}
[E_c(N,g,r)-E_{HMM}(N,g,r)].
\]

Positive delta favors the learned HMM. Weight the eight generators equally. The generator, rather than a token or overlapping prefix, is the independent replication unit.

Declare that version 1 supports a matched low-data advantage only if all of these hold for each neural control separately:

1. Mean delta across eight generators is at least 0.02 nats/target.
2. Mean delta is positive within each of the two regimes.
3. Delta is positive on at least six of the eight generator instances.
4. The lower endpoint of the descriptive 95% stratified generator-bootstrap interval is above zero.

Use 5,000 bootstrap draws, sampling four generators with replacement independently within each regime. Each selected generator carries all sizes, model families, and search replicas together. Use PCG64 SeedSequence(20260930,9,0) for resampling. Take empirical 2.5th and 97.5th percentiles. Record generator-level deltas as well as this interval. Four generators per stratum give limited uncertainty resolution; this interval is descriptive evidence, not a high-powered population claim or a guarantee of coverage. There is no post hoc choice of which control is stronger. Requiring the conjunction against both controls avoids reporting a result against only the weaker one.

Failure to meet this rule means the prespecified advantage was not established. An aggregate delta at or below zero against either control contradicts the directional claim at this budget. Small positive deltas or intervals spanning zero are inconclusive. Optimization traces may motivate a later experiment but cannot change this verdict.

### 9.3 Tokens-to-target curve

Set the prespecified error target to 0.05 nats/target. For each generator and family, average E over its two search replicas. Define N* as the smallest sampled N at which this mean is at most 0.05 and remains at most 0.05 at every larger sampled N. If none qualifies, report N*>16,384; if the first qualifies, report N*<=512. This grid does not identify the exact crossing or justify interpolation.

Report the full errors as well as N*, including generators where all models reach the target at the smallest size or none reaches it. A failure to find smaller N* for the HMM means a fewer-training-tokens claim is not established even if its low-data KL is lower. The 0.05 target and 0.02 advantage margin are operational thresholds chosen for this study, without a prior empirical calibration claim. Never treat them as universal learning standards.

Any fewer-parameters claim must state that the learned HMM has 43 free scalar parameters, the RNN 280 trainable scalars, and attention 712, plus their different representation assumptions. Numerical parameter economy is available by design. Equivalent-quality parameter economy is established only on measured cases with comparable error, and does not prove a general representational advantage.

## 10. Conditional stress test: second-order latent dynamics

Trigger this stage only if section 9.2's matched advantage rule passes. The stage definition is fixed now, before seeing results. Run both N=2,048 and N=16,384, all eight base generators, both search replicas, and every learned family. Generate fresh train/validation/test draws using stress streams; do not reuse matched tokens or warm-start matched fits. The train and test distributions match within this stress stage. This is structural model mismatch, not an unannounced train-to-test shift.

Keep V=8, L=64, and each base generator's A, B, and pi. Initialize z_0 from pi and z_1 from A[z_0]. For t>=2 define

\[
P(z_t=k\mid z_{t-2}=i,z_{t-1}=j)
=(1-\lambda)A_{jk}+\lambda(0.9\mathbf 1[k=i]+0.1/4),
\quad \lambda=0.5.
\]

Emit x_t from B[z_t] as before. The pair-state process (z_{t-1},z_t) is an exact first-order HMM with sixteen latent pair states. Its initial pair distribution is pi_i A_{ij}, and its transition from (i,j) to (j,k) is the probability above. The stress generator's pair distribution need not start in its stationary distribution; the oracle uses the specified initial pair distribution exactly.

Use a true sixteen-pair-state oracle. If q_t(i,j) is the prior distribution of (z_{t-1},z_t) before token t, then

\[
p_t^*(v)=\sum_{i,j}q_t(i,j)B_{jv},\qquad
b_t(i,j)=q_t(i,j)B_{j,x_t}/p_t^*(x_t),
\]

\[
q_{t+1}(j,k)=\sum_i b_t(i,j)
[(1-\lambda)A_{jk}+\lambda(0.9\mathbf 1[k=i]+0.1/4)],
\quad q_1(i,j)=\pi_i A_{ij}.
\]

The learned candidate still fits only four states. In matched data it is told the true state count. In stress data four states is a deliberately restricted budget, not the true pair-state count. Its hidden states need not correspond to the generator's four base labels. Identical base emissions can sometimes allow a smaller predictive representation than sixteen pairs; no impossibility theorem for every sampled generator is assumed.

At each stress size, compute paired E_control-E_HMM by generator, averaging search replicas, and the same descriptive stratified bootstrap interval. Report matched values at those same sizes beside stress values. Label the advantage retained at a size only if the mean is at least 0.02, both regime means are positive, at least six generators favor HMM, and the interval lower endpoint is positive, against both controls. Label it reversed against a control if its mean is negative and interval upper endpoint is negative; otherwise report reduced, absent, or inconclusive evidence with actual values. Passing at only one size is a size-specific result.

If matched advantage fails, mark stress stage not triggered rather than missing. An optional later stress run requires separate labeling. If stress passes, it supports only this particular mismatch. If stress fails, it does not rule out fitting a larger HMM or learning a predictive-state model. Those are outside this comparison.

## 11. Resource accounting

Run on CPU only, without MPS, CUDA, external services, compilation-dependent acceleration changes, or concurrent candidate training. Record OS, CPU, installed library versions, CPU thread settings, and numeric dtypes. Use one compute thread for each selected BLAS/neural backend when its configuration supports that setting, and record any inability to enforce it. Implementer must inspect the environment first; this document does not assert any library is installed.

Measure per-candidate training elapsed time, validation elapsed time, optimizer steps or EM iterations, token exposures, and peak process RSS. Use fresh processes for candidate memory measurements because process high-water marks cannot be reset reliably in place. Include process/library overhead and distinguish it from tensor storage. Count all search candidates in family-level selection cost; report the retained model's own fit cost separately.

After freezing each selected checkpoint, benchmark length-64 next-token scoring on the same stored test arrays with batch size one and batch size 32. Exclude file I/O and model construction, record two warmup passes, then ten timed passes and report the median total elapsed time and tokens/second. Include prediction-head and softmax work for every target. For neural models disable gradients. These are measurements, not projected rates.

For a sequence-length scaling diagnostic, use only the selected N=16,384 matched models from search replica 0 of each generator. Generate evaluator-only sequences of lengths 16,32,64,128,256 with seed tuple (root,4,regime,g,length). Use 32 sequences per length. No training or selection on these arrays. Benchmark the same passes and report oracle KL to disclose any longer-context quality change. Use whole-sequence dense neural scoring for all targets and causal filtering for HMMs. Separately, an optional incremental attention-cache benchmark must be labeled and must not replace dense measurements. Sinusoidal positions extend to all specified lengths.

Analytical resource statements, labeled estimates rather than measurements:

| Model | Scoring work for a length-L sequence | Changing state or inference workspace |
| --- | --- | --- |
| Dense four-state HMM | O(L(K^2+KV)); linear in L for fixed K,V | O(K) filter state, plus output probabilities; matrices O(K^2+KV) |
| Ordinary RNN | O(L(h^2+dh+hV)); linear in L for fixed widths | O(h) persistent state for streaming; training activations grow with L |
| Dense attention block | O(Ld^2+L^2d), plus output head | Explicit dense attention O(L^2+Ld) workspace; incremental K/V cache O(Ld) |

EM fitting uses forward-backward work O(NK^2), plus emission/count work, per iteration. Smoothing memory generally grows with L even though causal filtering state does not. Neural backpropagation and optimizer state differ from these inference descriptions. Report allocated parameter tensors and optimizer buffers separately, specifying byte widths. Four float64 HMM filter entries use 32 bytes; eight float32 RNN hidden entries also use 32 bytes. Coordinate count alone would misstate this particular memory comparison.

Small-L timing overhead can dominate measured slopes, and CPU kernels can favor particular models. Finite timings do not prove asymptotic complexity. The O(L) filtering result is an existing property conditional on fixed state count; a task-dependent increase in K can change its practical cost.

## 12. Required correctness checks before results count

1. **Oracle enumeration:** On independent test fixtures with K=2, V=2, L<=5 and nonstationary pi, enumerate all latent paths and compare joint probabilities and each prefix-conditioned next-token distribution to filtering, within 1e-10 in float64. Distinct A and B must make emission/transition order errors detectable.
2. **Initial target:** Verify p_1^*=pi B and that all learned neural x_1 predictions see only BOS. Check target and input lengths and token counts exactly.
3. **Causality:** Change tokens after a fixed prefix in evaluation mode and assert earlier predicted distributions are unchanged, within 1e-6 for neural float32 outputs and 1e-10 for HMMs. Explicitly test mask orientation, inclusion of the current shifted input, and sequence reset behavior.
4. **Forward-backward:** Compare scaled likelihood to exhaustive tiny fixtures; check gamma rows sum to one and xi marginals agree with adjacent gamma within 1e-10. Compare the sum of causal log probabilities with forward observed-data likelihood.
5. **EM objective:** Verify MAP J is nondecreasing within relative tolerance 1e-8 times max(1,abs(J)) on deterministic fixtures and fits. Confirm expected-count update denominators and sequence boundaries. Do not demand monotonic raw likelihood under MAP smoothing.
6. **Probability integrity:** Validate generator and fitted rows, finite neural logits, normalized evaluation probabilities, oracle-self KL near zero, and identical per-model prefix/target indexing. Fail on internal invariant violations rather than silently renormalizing malformed matrices.
7. **Stress oracle:** Compare pair-state recursion with exhaustive second-order latent paths on tiny fixtures. At lambda=0, verify stress-oracle observation probabilities equal the base HMM's for the same observed sequence and specified initialization. This ablation is a correctness test, not another data outcome condition.
8. **No leakage:** Audit learner/evaluator artifact access. Confirm validation selection reads only validation tokens, all checkpoints are frozen before test scoring, and test outcomes cannot branch into fit or selection logic.
9. **Reproducibility:** Repeated same-seed pilot runs reproduce tokens, initial tensors, checkpoint choices, and scores within stated numerical tolerances. Verify across model families that saved datasets have identical hashes.
10. **Accounting:** Assert measured parameter counts 52 stored HMM entries with 43 free, 280 RNN scalars, and 712 attention scalars. Check actual epochs/iterations, validation-call counts, exposure counts, and every attempted candidate appear in the manifest.

A shared HMM scorer alone is insufficient evidence of oracle correctness; exhaustive fixtures supply an independent calculation. Do not inspect confirmatory test error to debug optimization. Use the separate pilot and correctness fixtures.

## 13. Deliverables for the later implementer

Create a run manifest containing the frozen protocol, hashes, seeds, datasets, matrices in evaluator-only storage, model definitions, environment, search candidates, fit traces, and chosen checkpoints. Store raw records under the workspace, for example results/experiment-001/<unique-run-id>/, never shared temporary log names. No implementation or dependencies are supplied by this documentation task.

The result report must lead with whether the matched advantage and fewer-token target criteria passed, with generator-level evidence and all missing/invalid runs identified. Include regime-specific curves, target crossings or censoring, sample counts including validation, measured resources, analytical estimates labeled separately, and the stress-stage trigger and outcome. A negative result is a complete result.

Unresolved implementation matters are installed dependency availability and reproducible CPU performance. Unresolved scientific matters include optimization adequacy, sensitivity to neural width and tuning budget, unknown-state-count learning, broader generator families, and retrieval/composition transfer. These require later experiments. They are not prerequisites that authorize widening this first harness.
