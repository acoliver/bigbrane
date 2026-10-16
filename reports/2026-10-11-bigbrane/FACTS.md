# Bigbrane fact sheet for the October 11, 2026 report

## Reporting rules and verdict

Workspace: `/Volumes/XS1000/acoliver/projects/bigbrane`. Paths are relative to this workspace. Times are UTC−03:00. Sources are named beside claims and equations. Recorded outcomes are distinguished from later corrections. No candidate discovered by the research loop has beaten the baselines. Two cards reached S6 through the gates, both in run 4. Experiment 1 established a narrower result: a specified four-state HMM prior lowered low-data prediction error against two tiny neural controls. Sources: `MORNING-REPORT.md` §§1,5; `RESULT.md`.

## 1. Timeline and how it developed

### Date qualification

The task brief places the opening question about mathematical models of LLMs on October 9. The historical documents contain earlier internal dates: `RESEARCH.md` and `EXPERIMENT.md` say September 30; `MAINTENANCE-SCREEN.md` reports a machine date of October 7; `PLATFORM-TRIAL.md` names a run begun October 8 UTC, October 7 locally. These are source-document dates, not independently verified conversation dates. Do not silently rewrite them into an October 9 chronology. Precise time of the opening question is unavailable in the inspected sources.

### Development path

- The first literature investigation separated known Bayesian filtering from learning a generator, and considered predictive-state representations, spectral HMM identification, information bottleneck, MDL and multiscale coarse graining. It chose a controlled HMM versus tiny RNN versus attention comparison with exact predictive distributions. Sources: `RESEARCH.md` §§1–5; `EXPERIMENT.md`.
- Experiment 1 completed 384 matched and 192 stress candidate fits. It supported the predefined low-data KL advantage against both controls, but earlier sustained target crossing against both occurred on only 1 of 8 generators. Second-order latent dynamics reversed the advantage against RNN at both sizes and attention at the larger size. Source: `RESULT.md`.
- The platform comparison separated scientific search, general orchestration and literature synthesis. The maintenance screen admitted DeerFlow, GPT Researcher, Local Deep Research, RD-Agent and Open Deep Research under its six-calendar-month rule, excluded AI Scientist v2, and separately discouraged archived Open Deep Research. Source: `APPROACH-COMPARISON.md`; `MAINTENANCE-SCREEN.md`.
- The initial selection was RD-Agent's research interfaces, with Bigbrane retaining scientific authority and protected evaluation. The scripted Mac trial exercised both RD-Agent and DeerFlow, including invalid attempts and new-process recovery. It made zero live model calls and did not measure discovery quality. Sources: `PLATFORM-SELECTION.md`; `PLATFORM-TRIAL.md`.
- `HYBRID-PLAN.md` (October 9) shifted to a swappable controller comparison and made the thin loop the default unless RD-Agent won a matched test. This is a plan, not evidence that that comparison happened. It retained Bigbrane-owned schemas, execution, scoring, ledger, budgets and confirmation boundaries. The implemented prototype is the custom `hybrid/` loop. Sources: `HYBRID-PLAN.md` §§1–2; `FINAL-REPORT.md` §2.
- The P0 executor run ID is `p0-20261009T204723Z`, corresponding to October 9 17:47:23 at −03 in its name; the name is not an independent wall-clock start record. It tested constrained Linux/arm64 execution before the research-loop build. Source: `P0-RESULT.md`.
- October 9 supporting literature reviews addressed harness simplification, agent skills and AI mathematics failures. The harness review supported comparing a thinner controller while preserving independent scoring. The skills review found no demonstrated result on this mathematical-discovery workload. The mathematics review separated prior-art failures, theorem-statement mismatch, proof validity and attempt accounting. These informed the prototype's explicit obligations and evidence records; they do not establish Bigbrane's effectiveness. Sources: `LIT-HARNESS-2026.md`, implications; `LIT-SKILLS-2026.md`, verdict; `LIT-MATH-CLAIM-FAILURES.md`; `RESEARCH-SYSTEM-SPEC.md`, goal.
- October 10 build integrated the existing Podman executor with a host model gateway/private credential files, asyncio research scheduler, stage prompts/schemas, exact-oracle bench, literature search and inspectable records. The early llxprt-bridge attempt `run-20261010T000322Z` produced 4 cards and 12 calls, stopped after S1 without a recorded stop, and exposed card size and keychain-prompt problems. Sources: `FINAL-REPORT.md` §§2–4; `hybrid/research/ORCH-STATUS.md`, first-live-run section.
- The Claude gateway port compared LLxprt Code's TypeScript OAuth/provider behavior against the Python gateway. It ported browser-loopback/paste racing, request headers, prompt/tool conversion, model-specific reasoning and retry/cache behavior while retaining private Bigbrane token files and cross-process locking. Live subscription-endpoint probes are dated October 10, with logs under `results/gateway-20261011/`; an exact port-completion clock time is not supplied. Source: `hybrid/gateway/CLAUDE-PORT.md`, introduction and difference table. The project uses its own provider transport rather than delegating research calls to a full CLI session.
- Run 1, October 10 15:11:19 to 17:28:07: 23 cards, 233 calls. S2 reduced most cards to known methods. The graveyard fed those failures back to explorers; the escape operator required a child to identify and break a reduction invariant; signature equations warned against recurring rediscoveries. Both S3 entrants were escape children. S1 changed from blocking any residual issue to accepting reproduced toys with no fatal finding after two revisions, while carrying fixable issues to S3. Source: run 1 inspect, `ledger.jsonl`; `hybrid/research/ORCH-STATUS.md`; `FINAL-REPORT.md` §§3.9,4.
- Run 2, October 10 17:43:43 to 18:37:34: 19 cards, 178 calls, no S3 entrant. It used run 1's graveyard. Empty literature results previously counted as novel; the changed gate broadened the query and returned `undetermined` if still empty. Escape slots were reserved and parent/escape ancestors excluded from duplicate comparison. Source: run 2 inspect; `FINAL-REPORT.md` §§3.6,3.9; `hybrid/research/ORCH-STATUS.md`.
- Overnight: own ChatGPT sign-in for the plan account entered at October 10 23:45; GPT rejoined explorers/judges. The gateway used its own OAuth rather than the Codex CLI. Proactive ChatGPT refreshes returned 200 eight times from 00:40 to 07:05. Local arXiv harvesting, falsifier validation and executed/replayed S3 obligation checks were added. Source: `MORNING-REPORT.md` §4.
- Run 3, October 10 22:09:32 to October 11 01:14:09 by current inspect: 47 cards, 438 calls, six S3 entrants, five refuted and one undecided. The morning narrative instead gives an end near 01:22. Use ledger/inspect bounds for charts and flag this discrepancy. Sources: `results/morning-20261011/inspect-run-20261011T011300Z.txt`; `MORNING-REPORT.md` §2.
- Run 4, October 11 01:23:14 to 06:28:51: 103 cards, 1046 calls, two S6 entrants. It stopped on an independent 600-call model budget at 04:26 and was resumed after that cap was raised. A 04:29 preflight stalled on provider model-list access; provider lists later gained a timeout and cache. Source: run 4 inspect; `MORNING-REPORT.md` §4.
- The arXiv harvest stalled at 01:15 after repeated cs-page timeouts. Set deferral and end-of-pass retries preserved progress. It finished at 08:00 with 1,419,455 records across all 15 sets, including cs 1,029,985. Source: `MORNING-REPORT.md` §4.
- Gateway restart at 06:47; run 5 October 11 06:47:46 to 08:32:46 by inspect (morning report rounds to 06:48–08:33): 38 cards, 205 calls. Its sole S2 survivor was refuted at S3. It used graveyards from runs 1–4 and the fixed S6 rule, which had no new S6 entrant to exercise it. Source: run 5 inspect; `MORNING-REPORT.md` §§3–4.

## 2. Mathematical models studied

### Experiment 1 specification

Source for all equations and dimensions in this subsection: `EXPERIMENT.md` §§2–10. Use natural logarithms for reported KL and NLL.

Generator: four latent states, eight symbols, 64 tokens per independent sequence. Two regimes use `(rho,eta)=(0.60,0.80)` and `(0.95,0.40)`. Four independent generators per regime. Each row of R is Dirichlet(1,1,1,1); S assigns two preferred symbols per state with mass 1/2 each.

\[
A=\rho I+(1-\rho)R,\quad B_{iv}=\eta S_{iv}+(1-\eta)/8,\quad \pi A=\pi,\quad\sum_i\pi_i=1.
\]
\[
P(z_{1:L},x_{1:L})=\pi_{z_1}B_{z_1x_1}\prod_{t=2}^{L}A_{z_{t-1}z_t}B_{z_tx_t}.
\]

Causal filtering, with row-vector convention, emits the prediction before observing the current target:
\[
q_1=\pi,\quad p_t^*(v)=\sum_iq_t(i)B_{iv},\quad b_t(i)=q_t(i)B_{i,x_t}/p_t^*(x_t),\quad q_{t+1}=b_tA.
\]

Learned HMM: unrestricted pi, A, B fit by MAP Baum-Welch; 52 stored entries, 43 free parameters. Forward-backward computes gamma (state occupancy posterior) and xi (adjacent transition posterior). Independent symmetric Dirichlet(1+epsilon) priors have epsilon=0.001:
\[
\pi'_i={\epsilon+\sum_a\gamma^{(a)}_1(i)\over K\epsilon+n},\quad
A'_{ij}={\epsilon+\sum_a\sum_{t=1}^{L-1}\xi_t^{(a)}(i,j)\over K\epsilon+\sum_a\sum_{t=1}^{L-1}\gamma_t^{(a)}(i)},
\]
\[
B'_{iv}={\epsilon+\sum_a\sum_{t=1}^L\gamma_t^{(a)}(i)\mathbf1[x_t^{(a)}=v]\over V\epsilon+\sum_a\sum_{t=1}^L\gamma_t^{(a)}(i)}.
\]

RNN, embedding 9 by 8 including BOS, hidden width 8, 280 scalars:
\[
e_t=E[u_t],\quad h_0=0,\quad h_t=\tanh(W_ee_t+W_hh_{t-1}+b_h),\quad p_t=\mathrm{softmax}(W_yh_t+b_y).
\]
Shifted inputs are `[BOS,x_1,...,x_63]`, targets `[x_1,...,x_64]`.

Attention control, width 8, one head, feed-forward width 16, 712 scalars:
\[
v_t=E[u_t]+PE(t),\quad PE(t,2i)=\sin(t/10000^{2i/8}),\quad PE(t,2i+1)=\cos(t/10000^{2i/8}),
\]
\[
a=LN_1(v),\ Q=aW_Q,\ K'=aW_K,\ V'=aW_V,\quad C=\mathrm{softmax}_{row}(QK'^T/\sqrt8+M),
\]
\[
M_{ts}=\begin{cases}0&s\le t\\-\infty&s>t\end{cases},\quad r=v+CV'W_O,
\]
\[
h=r+\mathrm{ReLU}(LN_2(r)W_1+b_1)W_2+b_2,\quad p_t=\mathrm{softmax}(h_tW_y+b_y).
\]

Primary error against the exact next-symbol oracle:
\[
E_m(N,g,r)={1\over128L}\sum_{a=1}^{128}\sum_{t=1}^L\sum_{v=0}^7p_t^*(v\mid x_{<t}^{(a)})\log{p_t^*(v\mid x_{<t}^{(a)})\over p_m(v\mid x_{<t}^{(a)})}.
\]
Low-data contrast, each generator independent, each control separately:
\[
\Delta_{g,c}={1\over4}\sum_{r=0}^1\sum_{N\in\{512,2048\}}(E_c-E_{HMM}).
\]
The acceptance rule requires mean delta at least 0.02 nats/target, positive in both regimes, positive on at least six of eight generators and a positive lower endpoint of a descriptive 95% stratified generator bootstrap (5,000 draws). The separate target crossing is the smallest sampled N with replica-mean KL at most 0.05 at that and every larger sampled N. No crossing interpolation. Two candidates per replica, two replicas, validation-only checkpoint selection. HMM capped at 50 EM iterations; neural controls at 80 epochs. Validation has 2,048 additional accessible tokens. Source: `EXPERIMENT.md` §§8–9.

### Experiment 1 results

Source: `RESULT.md`, learning curves and primary contrasts. Means over generators and replicas, nats/target:

| Regime | Family | 512 | 2048 | 8192 | 16384 |
|---|---|---:|---:|---:|---:|
| mixing/informative | HMM | .09817 | .04387 | .02698 | .04838 |
| mixing/informative | RNN | .14456 | .03642 | .00938 | .00627 |
| mixing/informative | attention | .18983 | .08266 | .03917 | .02351 |
| persistent/ambiguous | HMM | .08872 | .04227 | .03188 | .04157 |
| persistent/ambiguous | RNN | .11221 | .07152 | .01617 | .01032 |
| persistent/ambiguous | attention | .15019 | .09976 | .04041 | .02602 |
| aggregate | HMM | .09345 | .04307 | .02943 | .04497 |
| aggregate | RNN | .12839 | .05397 | .01277 | .00830 |
| aggregate | attention | .17001 | .09121 | .03979 | .02477 |

RNN contrast: .02292, interval [.01200,.03208], favorable 7/8 generators. Attention contrast: .06235, interval [.05350,.07242], favorable 8/8. Both gates passed. HMM crossed earlier/tied/later than RNN on 1/4/3 generators and than attention on 5/1/2. It crossed earlier than both only once. Its worsening high-data curve leaves fitting adequacy unresolved.

Stress changes latent transition to depend on two previous states:
\[
P(z_t=k\mid z_{t-2}=i,z_{t-1}=j)=(1-\lambda)A_{jk}+\lambda(0.9\mathbf1[k=i]+0.1/4),\quad\lambda=0.5.
\]
The exact oracle has sixteen pair states, the learned HMM still four. Source: `EXPERIMENT.md` §10.

| N | Control | Stress control-minus-HMM KL | 95% interval | Outcome |
|---:|---|---:|---|---|
|2048|RNN|−.01779|[−.02798,−.00711]|reversed|
|2048|attention|+.05869|[.05153,.06594]|retained|
|16384|RNN|−.05108|[−.06035,−.03955]|reversed|
|16384|attention|−.03215|[−.04208,−.02027]|reversed|

Source: `RESULT.md`, stress table. Neither size retains advantage against both. Both controls already beat HMM at matched N=16,384; the full stress gap cannot be attributed to mismatch.

### Bench baselines and what each computes

These are a separate experiment family from Experiment 1. Mathematical restatements below follow `hybrid/bench/baselines/bb_ngram.py` and `bb_neural.py`; dimensions and measured counts also appear in `hybrid/bench/README.md` and `BASELINES.md`. Analytical-cost entries and the expanded GRU gates are explicitly inference from implementation and the library contract, not new measurements. Let n be the length of a single prefix processed by a neural model. The scaling probe times prediction of the last target of each supplied prefix. A flat n-gram probe therefore does not mean generating an entire length-n sequence takes constant time.

|Baseline|Computation and shape|Analytical cost of processing one length-n sequence, fixed dimensions|Recorded parameter count in default/quick baseline scaling fixture|
|---|---|---|---:|
|n-gram|Witten-Bell 4-gram; last 3 symbols index count tables|O(n) for all targets; O(1) in prefix length for one query, fixed vocabulary/order|7146 / 5003 nonzero count entries plus unigram counts|
|GRU|Embedding 64, recurrent hidden width 128, output vocabulary head|O(n(h²+dh+hV)); fixed streaming hidden state|76812 / 76812|
|transformer|Width 64, 2 layers, 4 heads, feed-forward width 256, sinusoidal positions|O(n d²+n²d+n dV), fixed layer count|101644 / 101644|
|diagonal SSM|2 complex LRU-style layers, state width 64 each, embedding width 64, nonlinear residual mixing|Streaming recurrence O(n); implemented whole-prefix FFT convolution O(n log n), fixed widths, plus projections|51724 / 51724|
|linear attention|2 blocks, width 64, elu+1 feature map, feed-forward width 256, chunk size 64|O(n d²+n C d), C=64 fixed; streaming accumulator fixed in n|101644 / 101644|

Counts refer to the recorded scaling-task vocabulary, not every task's vocabulary. The n-gram count is an occupied-entry convention, not a dense tensor parameter count. Source: `hybrid/bench/BASELINES.md`, default and quick scaling tables; implementations above.

N-gram interpolated backoff, h' drops the oldest context symbol and T(h) counts distinct successors:
\[
P(w\mid h)=\frac{c(h,w)+T(h)P(w\mid h')}{c(h)+T(h)},\qquad P_0(w)=\frac{c(w)+1}{\sum_v c(v)+V}.
\]
An unseen table backs off. Source: `hybrid/bench/baselines/bb_ngram.py:1–8,31–41`.

GRU mathematical restatement of `nn.GRU` in `bb_neural.py:30–42`, retaining PyTorch's candidate-state convention:
\[
r_t=\sigma(W_{ir}e_t+b_{ir}+W_{hr}h_{t-1}+b_{hr}),\quad z_t=\sigma(W_{iz}e_t+b_{iz}+W_{hz}h_{t-1}+b_{hz}),
\]
\[
\widetilde h_t=	anh(W_{in}e_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})),\quad h_t=(1-z_t)\odot\widetilde h_t+z_t\odot h_{t-1}.
\]
The implementation delegates these gate equations to the installed PyTorch GRU; they are explanatory expansion rather than a printed source equation. An untied linear head produces vocabulary logits.

Transformer uses pre-normalized causal multihead attention and GELU feed-forward residuals. For each head:
\[
\operatorname{Att}(Q,K,V)=\operatorname{softmax}_{row}(QK^	op/\sqrt{d_h}+M)V,\quad d_h=64/4=16.
\]
Source: `bb_neural.py:54–91`. Unlike Experiment 1's control it has multiple layers/heads and embedding scaling by square root of width.

SSM core:
\[
h_t=\lambda\odot h_{t-1}+Bu_t,\quad y_t=\operatorname{Re}(Ch_t)+D\odot u_t,\quad
\lambda=\exp(-\exp(
u)+i\exp(\vartheta)),\quad\gamma=\sqrt{1-|\lambda|^2}.
\]
B is input-normalized by gamma. The recurrence is evaluated with the causal kernel lambda^k and a zero-padded FFT of length 2n. Each outer block adds GLU(linear(GELU(SSM(LN(h))))). Source: `bb_neural.py:94–147`. Its near-linear measured timing is finite-range evidence; its implemented FFT cost should not be described as exactly O(n).

Linear attention, phi(x)=elu(x)+1:
\[
S_t=S_{t-1}+\phi(k_t)v_t^	op,\quad z_t=z_{t-1}+\phi(k_t),\quad y_t=\frac{\phi(q_t)^	op S_t}{\phi(q_t)^	op z_t+10^{-6}}.
\]
One block retains 64²+64 state floats; two retain 8320. Chunked within-block masking plus the prefix accumulator computes the same causal sum. Source: `bb_neural.py:150–215`. The implementation has no positional encoding. Its source comment about causal sums carrying order does not prove arbitrary order sensitivity of the additive accumulator.

### Five exact-oracle tasks

Source: `hybrid/bench/README.md`, tasks table and oracle conventions.

- HMM: predict a stochastic next symbol using the exact latent-state filtering distribution.
- Variable-order Markov (vom): predict with the context-tree generator's exact suffix-conditioned distribution.
- Associative recall: a prefix supplies key-value bindings; an answer oracle names the value for the queried key. Capacity is probed as the number of bindings grows.
- Hierarchical composition (compose): typed bracket sequences require closing the correct open bracket; the generator supplies an exact next-symbol distribution. This measures the supplied grammar, without establishing general reasoning.
- Few-shot rule induction: demonstrations identify an affine rule modulo 13; held-out rules differ between train and test. Queries use 2, 4 or 8 shots. Chance is 1/13. Answer-oracle KL is negative log probability of the correct answer.

The research-loop quick config differs from `hybrid/bench/configs/quick.toml`: `hybrid/config/bench_quick.toml` uses training sizes 32/128, 128 tests, 200 steps, at most 20 epochs, all five baselines, and a short scaling range 64 through 1024. Recall uses K=2/8 with 8 possible values; these chances differ from the larger baseline demonstration's 16-value recall task. Do not mix tables from these configurations.

### Mechanism fingerprint

Source for definitions and thresholds: `hybrid/bench/README.md`, mechanism fingerprint and calibration sections. It combines six kinds of evidence: recall capacity, measured time scaling, memory growth, longer-prefix generalization, few-shot response to added examples, and prediction similarity to known baselines.

Recall capacity interpolates in log K to a half-accuracy knee, then divides stored pairs by reported state floats. If recall is below twice chance at the smallest K, it is `not_learned`. If the knee lies beyond the tested range, capacity is a lower bound. Length generalization evaluates 2× and 4× training lengths. The few-shot slope is accuracy change per doubling of shots. Time slope is a fitted exponent of time against length on log-log coordinates. RSS growth has a floor and is weak evidence.

For distributions p,q, with m=(p+q)/2:
\[
\operatorname{JS}(p,q)=	frac12\operatorname{KL}(p\Vert m)+	frac12\operatorname{KL}(q\Vert m).
\]
The similarity ratio normalizes the square-root mean JS distance between candidate a and baseline b by their distances from oracle o:
\[
R(a,b)=\frac{\sqrt{\overline{\operatorname{JS}}(a,b)}}{\sqrt{\overline{\operatorname{JS}}(a,o)}+\sqrt{\overline{\operatorname{JS}}(b,o)}}.
\]
A task is close when mean JS ≤0.02 and ratio ≤0.5; `behaves_like` requires more than half the compared tasks to be close. Other flags are `no_capacity_advantage` at no improvement over the best fixed-state baseline's pairs per state float, `quadratic_or_worse` for tail slope >1.5, and `no_fewshot_gain` for few-shot slope <0.03 per doubling. These are operational thresholds. A slope of 1.59 does not prove an asymptotically quadratic program.

Calibration includes identical GRU code giving JS=0 on all five tasks, a different-seed GRU remaining close on 4/5, and different-seed linear attention close to itself on only 2/5. On the default baseline run transformer and linear attention flag each other on 3/5 tasks at mean ratio 0.30. The fingerprint does not reliably distinguish those families on this bench. Source: `hybrid/bench/BASELINES.md`, calibration and similarity tables.

### Known-family signature equations

Source for every signature below: `hybrid/research/prompts/explorer.md:22–34`. Orientations differ between signatures because the stored matrices differ; do not silently transpose only one term.

|Family|LaTeX-ready update/readout|Meaning of the warning|
|---|---|---|
|Delta rule / fast weights|$S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^	op$, $y_t=S_tq_t$|Error-correcting associative writes, including one-hot or hashed keys|
|Linear attention|$S_t=S_{t-1}+\phi(k_t)v_t^	op$, $y_t=S_t^	op\phi(q_t)$|Additive bound-pair accumulators, with possible decay/normalization|
|Recursive least squares|$P_t=P_{t-1}-\frac{P_{t-1}k_tk_t^	op P_{t-1}}{1+k_t^	op P_{t-1}k_t}$, $W_t=W_{t-1}+(v_t-W_{t-1}k_t)k_t^	op P_t$|Least-squares operator estimation from context|
|MesaNet / in-context least squares|$G_t=\sum_i k_ik_i^	op$, $C_t=\sum_i k_iv_i^	op$, $y_t=C_t^	op(G_t+\lambda I)^{-1}q_t$|A fitted operator or repeated reads of it|
|Modern Hopfield / kernel memory|$y=\sum_i v_i\operatorname{softmax}_i(\beta x_i^	op q)$|Similarity-weighted retrieval of stored patterns|
|SSM / S4 / Mamba / LRU / RetNet|$h_t=A_th_{t-1}+B_tx_t$, $y_t=C_th_t$|Fixed, diagonal or input-conditioned linear dynamics|

Additional families are exact-match caches/hard attention, labeled union-find/constraint closure, n-gram/last-seen lookup, and classical data structures such as tries, hash maps, stacks, lattice learners and shift-reduce parsers. The prompt's family warnings are search heuristics; a mathematical reduction needs a mapping, assumptions and evidence.

### Lenses

`hybrid/config/lenses.toml` lists 23 active source fields and one non-exploring known-alternatives reference lens. Active fields: reservoir computing; vector-symbolic/hyperdimensional computing; holographic reduced representations; modern Hopfield networks; predictive coding; active inference; dendritic/spiking computation; neural ODEs/Koopman theory; tensor networks for language; renormalization/multiscale methods; kernels/Gaussian processes; Bayesian program induction; cellular automata; optimal transport; sheaves/category models; compression/MDL; algorithmic information; sparse distributed memory; complementary learning systems; hyperbolic geometry; quantum-inspired methods; chemical reaction networks; morphogenesis. A lens is a source of possible constructions, not evidence of a biological or physical implementation.

Seven cross-lens pairings are vector-symbolic+renormalization; kernels+program induction; sparse distributed memory+complementary learning systems; tensor networks+hyperbolic geometry; chemical reactions+predictive coding; Koopman+MDL; cellular automata+sheaves. Source: the same TOML, `[[pairings]]`.

### Concrete S2 reductions with executed numeric checks

All five examples are run 1, `runs/run-20261010T181300Z/ledger.jsonl`; cited line numbers name exact deciding events. Numeric checks are finite fixtures, not universal proofs.

|Card|Mapping summary|Recorded maximum absolute difference|Source|
|---|---|---:|---|
|card-537e161c|Koopman/EDMD dictionary fit becomes weighted RLS, a preconditioned delta rule; degree-one read is MesaNet least squares. Operator powers are multi-hop fast-weight reads.|1.84456894203322e-11|ledger line 450; `artifacts/reduction_critic/card-537e161c-v7-3c5dea.json`|
|card-6de11b01|Hashed residual slot overwrite becomes gated delta rule with one-hot key e_a: $S_t=(I-g_te_ae_a^	op)S_{t-1}+g_te_aho_t^	op$. Read adds the selected residual to a slow predictor.|6.106226635438361e-16|line 441; `artifacts/reduction_critic/card-6de11b01-v6-090bba.json`|
|card-582f8abf|Gauge/parity reservoir becomes Z₂-labeled union-find; parents, XOR edge labels, union-by-size and cycle consistency coincide with the known algorithm.|0.0|line 294; `artifacts/reduction_critic/card-582f8abf-v5-58f4e3.json`|
|card-14429dbc|Exact FIFO cache plus gcd of within-label integer differences becomes a one-dimensional most-specific consistent lattice learner with a residue-to-label table.|0.0|line 434; `artifacts/reduction_critic/card-14429dbc-v7-bbeb3d.json`|
|card-1184f7a7|Phasor/holographic write becomes diagonal no-decay linear attention/H3: $R_t=R_{t-1}+\overline E_{x_{t-1}}\odot E_{x_t}$, followed by phase normalization and tied decoding.|2.6645352591003757e-15|line 569; `artifacts/reduction_critic/card-1184f7a7-v7-f2ff96.json`|

The `prior_art_undertested` label applies to examples 1, 2, 3 and 5 even though their mechanisms were reduced to known methods. The lattice card's retrieved-paper label was `novel` while its algorithm reduced to known work. Search labels and reduction labels answer different questions. Source: the same verdict events.

### Card-03fa1d38: Two-Sheet Residual Rewriting

Source for all core equations and theoretical constants: `runs/run-20261011T042400Z/cards/card-03fa1d38/v11.json`, `mechanism` and `claims`. This is a cross between residual rules on binary token codes and remapping on their images. E is a distinct N-of-M code for each of V tokens. R holds at most Kmax rules (sheet flag ell_j, address a_j, addition b_j); tau_x records the first demonstration's recruitment index. Claimed hand-factorized setup: V=125, M=15, N=3, Kmax=8.

Packing orders matching rules by descending address size, then newest first. Starting with U=0, each visited rule fires only when its address is contained in c and does not overlap already claimed units:
\[
s_j=[a_j\le c][a_j^	op U=0],\quad U\leftarrow U+s_ja_j,\quad P(S,c)=\left(c+\sum_{j\in S}s_j(b_j-a_j),Uight).
\]
The prediction first applies the code sheet, then eligible image-sheet rows:
\[
(o_1,U_1)=P(\{j\le K:\ell_j=0\},E_x),\quad
(o,U_2)=P(\{j\le K:\ell_j=1,\ j\ge	au_x\},o_1).
\]
On DEMO(x,y), w records whether x had been demonstrated; an unstamped x gets tau_x=K+1 after prediction. Residual r=E_y−o gives D=[r<0], A=[r>0]. Set (beta,U,ell)=(E_x,U1,0) for first demonstrations and ([o1>0],U2,1) otherwise. With nonzero residual and available capacity, recruit (ell,D,A) if |D|>0, max|r_i|≤1, D≤beta and DᵀU=0; otherwise recruit the full pair (ell,beta,E_y). Queries do not write.
\[
j^*=\min\arg\max_v E_v^	op o(x),\qquad p_v=[v=j^*].
\]
Offline code learning hill-climbs $J(E)=\sum_{ep\in B}(K_{fin}+10\sum_{(q,t)\in ep}[j^*(q)
e t])$. The claims instead use hand-factorized codes and isolate the rewrite law. Claimed total work is O(n(Kmax+V)M), excluding offline learning.

Claims: a single witnessed remap transfers to other stamped class members; three independent atom demonstrations compose on never-shown tokens; specificity preserves an exception; stamps prevent later newcomers inheriting an earlier class remap. The card also explicitly predicts an undesirable co-motion bias in a separate defection probe.

Stage evidence, run 4 ledger:
- S0 pass line 4123; S1 two revisions then pass line 4378.
- S2 pass line 4616: `prior_art_undertested`, closest tested ripple-down-rule work arXiv 1412.4021 and 1709.06307. Reduction critic found no fixed kernel reproducing order-dependent non-overlap packing. This remains a model judgment.
- S3 pass line 5019: Opus feasible confidence .85, GPT feasible .97. Eight judge obligations were marked discharged, but executed checks did not discharge all checkable obligations. Capacity fixtures each reproduced 1000 random protocols; operation counts varied M and V. Offline learnability and parent ablation were unexecuted; closure was sampled over 9000 events and marked unknown. Artifact `artifacts/s3_gate/card-03fa1d38-v7-c3e5ab.json` explicitly has `all_checkable_discharged: false`.
- S4 pass line 5029: described 42 parameters and 176 state floats on the seven-symbol construction fixture. Those dimensions differ from both the theoretical setup and recall fingerprint. On the recall task the reported state is 266 floats, estimated capacity at least 8 pairs, 33.25 floats per pair. Source: `artifacts/bench_raw/card-03fa1d38-9.json`.
- S5 pass line 5594: trusted scoring; tail slope .996592309873441, claimed 1.0, match. Flags `no_capacity_advantage`, `no_fewshot_gain`; few-shot slope .01171875. Nearest baseline transformer by normalized ratio, linear attention by mean JS. Neither means mathematical equivalence.
- Two validated falsifier task programs ran, producing three rule decisions. A third proposed task, class_transfer_locality, was dropped because `rule 0 is met by the recency_4 control, so the answer is predictable without the mechanism`. Designers were GPT-6.1 Sol, separate from the Claude builder. Source: `artifacts/falsifier_tasks/card-03fa1d38-v9-b11e0a.json`. This corrects any shorthand saying three tasks ran.
- All three rules were inconclusive: `the candidate did not train on this task: it does not beat the better of the uniform and unigram references by 0.05 nats of KL or 0.05 accuracy`. Source: line 5594 and raw artifact.
- S6 line 5601 originally recorded 1 supported/1 refuted/3 inconclusive/11 untested. The analyst used composed-token accuracy .0078125 to refute falsifier 4 despite the task's failed training gate. Later read-only reconstruction changes this to 1/0/4/11, without rewriting the run. The supported item is finite-range sequence-time scaling; the proposed class-remap learning mechanism remains untested by S5. Sources: original ledger, `MORNING-REPORT.md` §4 and current S6 code.

### Card-779795a5: Conjugate Sewing

Source for equations: `runs/run-20261011T042400Z/cards/card-779795a5/v12.json`. This escape child replaces an additive output projection with an executable involution. State is frozen scalar theta, a list P of protected polynomial shears, and endpoints A. F0(x)=−x. At most K=4 demonstrations are accepted; theta is calibrated to 1/4 using the stated toy continuation, which is not held-out evidence.

For roots D and target a not in D:
\[
b(s;a,D)=\prod_{u\in D}\frac{s-u}{a-u}\,[1+	heta(s-a)^2].
\]
On TEACH(p,q), set r=F_m(p), B=A concatenated with p; accept only when m<K, q0 is absent from B0 and r1 absent from B1. Define
\[
h(s)=(r_1-q_1)b(s;q_0,B_0),\quad H(x_0,x_1)=(x_0,x_1+h(x_0)),
\]
\[
j(s)=(r_0-q_0)b(s;r_1,B_1),\quad J(x_0,x_1)=(x_0+j(x_1),x_1),\quad T=J\circ H,
\]
\[
F_{m+1}=T^{-1}\circ F_m\circ T,\quad P\leftarrow P\Vert[(H,J)],\quad A\leftarrow A\Vert[p,q].
\]
At query x, execute z=Fell(x), then $p(v)=\exp(-\|c_v-z\|^2)/\sum_w\exp(-\|c_w-z\|^2)$. Query and rejected TEACH events do not change state. Claimed arithmetic work O(n(K²+V)), memory O(K²+V); this excludes guarantees about bit growth and conditioning. The claimed one-demonstration bidirectional binding follows from conjugation preserving involution. Unseen continuation is an imposed bias.

Stages: S0 line 5636; S1 pass after two revisions line 7077; S2 pass line 7318 (`novel` among retrieved papers, no accepted reduction); S3 pass line 7603 (Opus .88/GPT .98). Exact rational one-pair checks gave F(p)=(2,3), F(q)=(1,1), F(1,0)=(11/4,511/64), round trip (1,0). The card's float64 toy had maximum coordinate error 0. Multi-binding capacity timed out; general protection/involution and offline learnability were not fully executed. `artifacts/s3_gate/card-779795a5-v8-faffac.json` has `all_checkable_discharged: false`.

S4 line 7645 passed on repair attempt 1, described 15 parameters/64 state floats at V=7. Executor's array-only count was 14; the description includes the scalar theta. Do not call all 15 learned weights: token coordinates are fixed and theta alone is calibrated. S5 line 7700 measured tail slope 1.5904128264473698 against claimed 1.0, mismatch at tolerance .5; full-range slope .9952346778782979. Flags `quadratic_or_worse`, `no_fewshot_gain`. Source: `artifacts/bench_raw/card-779795a5-10.json`.

One GPT-designed falsifier task, accepted_one_shot_endpoints, ran. Deciding evidence: `the candidate produced no valid predictions on this task`, with problem `non-finite values`. S6 line 7714 recorded 0 supported/2 refuted/1 inconclusive/8 untested, unchanged by the later S6 correction. The two refutations concern measured timing, not a proof that the exact-arithmetic involution construction is false. The implemented token interface also failed longer compose/vom probes. Sources: the same raw artifact and ledger.

### Card-ffcfe0e5: Sticky-Bound Latent Causes

Source for equations: `runs/run-20261011T095000Z/cards/card-ffcfe0e5/v9.json`. State comprises a trained embedding/executor/abductor parameter block Phi, P rule-code rows R_p, per-slot evidence z_p in [−theta,theta], and one-step predictive q. Theta=3 nats, surprise threshold kappa=1.5. Slots start vacant at z=−theta. A birth sets R from the latest transition and z=0; it receives no evidence credit from its own birth transition.
\[
h(\cdot\mid a)=\operatorname{softmax}(EW_he_a),\quad u_p=	anh(W_f[R_p;e_a]),\quad g_p(a)=\sigma(w_g^	op u_p),
\]
\[
p_p(\cdot\mid a)=g_p(a)\operatorname{softmax}(Eu_p)+(1-g_p(a))h(\cdot\mid a),\quad s_t=-\log q_{t-1}(x_t),
\]
\[
\delta_{p,t}=\log p_p(x_t\mid x_{t-1})-\log h(x_t\mid x_{t-1}),\quad z_p\leftarrow\min(z_p+\delta_{p,t},	heta).
\]
Values at or below −theta are absorbed to vacancy. If surprise exceeds kappa and every previously occupied slot has nonpositive drift, a vacant slot gets $R_{p^*}=	anh(W_a[e_{x_{t-1}};e_{x_t}])$, z=0. The highest-gated committed slot predicts the next symbol; if none is committed, the habit predicts. Offline training includes ordinary habit loss and abduce-then-explain future-window loss. Claimed inference cost O(nPd(d+V)), independent of n in persistent state.

The claimed step at two demonstrations assumes an executor already assigning .9 to the correct token on unseen inputs: log(.9×26)>3. Such abduction must be learned; the evidence accumulator does not create it. S0/S1 passed, S2 pass line 377 had `prior_art_undertested`, S3 line 667 recorded `infeasible_proved` under a confirmed-disproof rule. Both judges rejected the unrestricted permutation/table one-shot generalization claim. Deciding quote: `For a uniformly random permutation of 26 tokens, an unseen mapping has optimal prediction accuracy 1/25 after one observed pair, contradicting the required 0.9 one-shot generalization.` Brute-force checks for vocabularies 4,5,6 found unseen-answer probabilities 1/3,1/4,1/5, respectively. A shift family was identifiable after one pair, so the rejection depends on family assumptions. Source: `runs/run-20261011T095000Z/ledger.jsonl:667`.

The idealized sticky-ceiling switch experiment did reproduce constant latency 2 across earlier confirmations m=2,5,20,50; no-ceiling ablation latencies were 2,4,13,30. That conditional mechanism result did not rescue the generalization claim. The ledger also labels a complexity check `violated` while its explanation says the O bounds hold with fixed operation ratios. This is a reporting inconsistency in model-written checking evidence, not a valid complexity disproof. The run's recorded final verdict must be attributed to the identifiable-family counterexample and confirmed disproofs. Source: the same line. No S4–S6 build or learned benchmark occurred.

### Three S3 refutations illustrating what the gate can detect

All are run 4. Equations follow the latest card JSON at the cited version. Claims, outcomes and quotes follow the ledger lines below.

**Householder Rebinding, card-c8ef549e**, `cards/card-c8ef549e/v9.json`: state S is d×d orthogonal binder, E a token codebook. With k=E[x_{t−2}], v=E[x_t], gate g and epsilon=1e−9:
\[
w=\frac{k-S^	op v}{\|k-S^	op v\|}\quad	ext{when norm}\ge\epsilon,\quad w=0	ext{ otherwise},\quad S_t=S_{t-1}(I-2gw w^	op),
\]
\[
y_t=S_tE[x_{t-1}],\quad p(j)=\operatorname{softmax}_j(\beta E_j^	op y_t).
\]
S2 rated it under-tested prior art, distinct from a fixed orthogonal binder, and passed. At S3, 5000 episodes at d=64 gave random-code recall .76505 at K=8 against ≥.9, and .62446875 at K=64 against ≤.5. Unseen bijection completion gave .5626 against ≥.9, although exhaustive orthonormal-letter cases passed. Deciding quote: `random K=8 mean recall 0.7651 < 0.9`. Both judges infeasible; gate `check_violation`, reproduced then replayed. Source: `ledger.jsonl:790`. No construction followed.

**Committed Sign Agreements, card-b2761f4a**, `cards/card-b2761f4a/v9.json`: state is pairwise sign-agreement matrix S. Evidence coefficient c=rho y for SAME/DIFF, otherwise 0. On relation (i,j), rows update through a rational normalization and symmetry; other entries remain:
\[
\delta=1+cS_{ij},\quad\widehat S_{i\cdot}=\frac{S_{i\cdot}+cS_{j\cdot}}\delta,\quad\widehat S_{j\cdot}=\frac{S_{j\cdot}+cS_{i\cdot}}\delta,
\]
\[
\widehat S_t\leftarrow\operatorname{clip}(\widehat S_t,-1,1),\quad S_t=\operatorname{sign}(\widehat S_t)g(|\widehat S_t|),\quad g(z)=z-z(z-\theta)(z-1),\quad p_{ASK(a,b)}=\left(\frac{1+S_{ab}}2,\frac{1-S_{ab}}2\right).
\]
S2 found no accepted known-form reduction and passed. S3 refuted claimed exact float64 settlement: `All 2001 tested committed starts failed to reach exactly 1.0 within 300 events, including start 0.6 and the protocol-relevant start 0.6745.` Noise replay also produced mean spurious commitments 358.4 and maximum 1165 against bound 45; noise placement was checker-selected because the card underspecified it. The precision counterexample is cleaner than generalizing that selected noise protocol. Both judges infeasible, gate `check_violation`. Source: `ledger.jsonl:1856`.

**Residual Reaction Nucleation, card-c60cacaa**, `cards/card-c60cacaa/v9.json`: E is a binary codebook and R a bounded list of consumption/production vectors alpha_r/pi_r with weights w_r. Propensity, allocation, reaction and decoding are
\[
q_r=w_r\min_{i\in\operatorname{supp}\alpha_r}c_i,\quad Q_i=\sum_{s:\alpha_{si}=1}q_s,\quad \xi_r=\min_{i\in\operatorname{supp}\alpha_r}\frac{c_iq_r}{Q_i}\quad(q_r>0),
\]
\[
c\leftarrow c+\sum_r\xi_r(\pi_r-\alpha_r),\quad p(y)=\operatorname{softmax}_y(-\beta\|E[y]-c^*\|^2).
\]
Each demonstration rewards or penalizes fired reactions, reruns the network, nucleates a residual reaction if its error exceeds tau, and evicts the smallest-weight reaction when over capacity. S2 rated it under-tested prior art and passed. Counterexample: confirm four rules to weight 2, append a new rule at weight 1; eviction immediately deletes that new rule. Deciding quote: `after four rules are confirmed to weight 2, a new weight-1 rule is immediately evicted, giving p(correct)=4.54e-05.` Opus was undecided, GPT infeasible; a narrower executed capacity check discharged a different protocol but did not negate the broader history counterexample. Recorded gate `disproof`, final `infeasible_proved`. Source: `ledger.jsonl:1928`. This example shows why the exact claim and protocol scope matter when checks disagree.

## 3. Validation and limits

Each stage records its output and deciding evidence. S0 validates draft/schema, grounds source IDs and rejects near-duplicate cards. S1 combines a model critic with an executed toy and comparison to expected JSON. S2 combines retrieved literature with a reduction critic and executed equivalence programs. S3 formalizes obligations, attacks them, runs available attack and obligation checks, replays decisive obligation programs, and consults two blind judges from different model families. S4 builds candidate code, static-checks it, executes fitting/output/causality checks and permits repairs. S5 runs candidate and baseline models in Podman and scores their prediction arrays through a separate evaluator. S6 joins claims with benchmark evidence. An S5/S6 `pass` means the stage completed, not that the candidate won. Sources: `hybrid/research/ORCH-STATUS.md`; `FINAL-REPORT.md` §§2,3.7; `MORNING-REPORT.md` §6.

Current inspect verified all five hash-chained ledgers and every recorded card/artifact sha256. Event totals and abbreviated ledger heads: run 1 1502/f549ff5a8b89; run 2 1143/b44f87279afb; run 3 3143/1c6883bad297; run 4 7724/e44ca7728ec6; run 5 1482/de194d2819b9. Source: live `python -m hybrid.research inspect` on the named runs. Hashes establish byte identity, not scientific truth.

S2 in the first two runs had 38 verdicts, 36 claimed reductions with numeric checks, 32 confirmed equivalences (maximum recorded difference 3.5e-11, eight exactly zero), and four rejected reductions (differences .060,.445,.716,.889). This demonstrated rejection of a critic's own false mapping. Source: `FINAL-REPORT.md` §1.

S3 obligation check decisions require agreement across first execution and replay on status and counterexample. A non-optional violation overrides judge votes. Unsupported kinds remain judge-only; a crash or disagreement leaves the check unknown. Run 4's nine S3 refutations were seven judge disproofs confirmed by the other judge and two executed/replayed violations. Source: `hybrid/research/ORCH-STATUS.md`, executed-obligation section; `MORNING-REPORT.md` §2.

S6 correction: task rules and task validity/training gates now override an analyst claim using that task's evidence. For card-03fa1d38 the original 1 supported, 1 refuted, 3 inconclusive, 11 untested becomes 1 supported, 0 refuted, 4 inconclusive, 11 untested in a read-only reconstruction. Original run records remain unchanged. This correction depends on the analyst declaring which tasks it used. Source: `hybrid/research/ORCH-STATUS.md`, S6 section; `MORNING-REPORT.md` §4.

Recorded full hybrid suite: 1182 passed, 9 skipped, excluding P0 container probes. These are previous test results, not a rerun for this report. Sources: `results/morning-20261011/pytest-all.log`; `MORNING-REPORT.md` §4.

### Experiment 1's independent instrument checks

The frozen protocol requires independent exhaustive latent-path enumeration on tiny HMM fixtures; agreement of filtering and forward likelihood; normalized forward/backward marginals; initial BOS alignment and sequence resets; changing future tokens without changing earlier predictions; monotonic MAP objective within numeric tolerance; pair-state stress recursion checks including the lambda=0 reduction; learner/evaluator data-access audit; seeded reproducibility; and parameter/fit accounting. Source: `EXPERIMENT.md` §12. Enumeration supplies a different computation from the filter used in scoring, helping detect shared-code oracle errors. These are protocol requirements; this fact sheet did not rerun them.

Validation alone selects models, and confirmatory test/oracle scores do not feed fitting. Every generator is kept in the aggregate, independent generators define uncertainty, and the low-data acceptance rule was fixed before scoring. This constrains favorable post-hoc selection but does not equalize model compute, parameterization or optimization adequacy. Sources: `EXPERIMENT.md` §§8–9,12; `RESULT.md`.

### Which models check which stages

Role routing below describes the gateway tables in `hybrid/config/models.toml`; actual fallbacks are recorded per call. Run 1 started on gateway and resumed with Claude/local routing, run 2 used Claude/local, and overnight gateway runs restored GPT. Opus, Sonnet and Haiku are members of the same Claude model family. Multiple Claude roles are not independent model families.

|Stage|Model role or deterministic checker|Execution and remaining judgment|
|---|---|---|
|S0|Explorers alternate Claude Opus 5.5 and GPT-6.1 Sol; Claude/local table uses Opus for both|Schema, citation-grounding and TF-IDF duplicate check are programmatic; proposal content is generated|
|S1|Claude Haiku critic, Sonnet fallback; explorer revises|Podman reproduces the toy, numeric comparison rtol 1e−6/atol 1e−9; definitions and fatal/fixable classification are judged|
|S2|Claude Haiku prior-art classifier; Opus reduction critic, Sonnet fallback|Retrieved IDs grounded; reduction/check scripts execute in Podman; coverage, mapping and radicality remain judged|
|S3|Sonnet formalizer/disprover; Haiku obligation-check author, Sonnet fallback; Opus judge A; GPT judge B or local Qwen 3.6 35B-A3B|Attack scripts and available SymPy/NumPy checks execute in Podman and decisive obligation checks replay; judges see identical evidence without each other's votes|
|S4|Claude Sonnet builder, Haiku fallback|AST import/operation restrictions and executor fitting/output/causality checks, with repair; faithful mechanism translation still needs assessment|
|S5|No scoring model; GPT falsifier designer, local Qwen fallback; Claude/local table uses Qwen|Designer sees card and budget, not candidate code. Podman validates tasks, runs all entrants and separately scores arrays. Generic tasks and card-specific tasks differ|
|S6|Claude Haiku analyst, Sonnet fallback; deterministic rule merger|Rules/task gates override linked analyst outcomes. Unlinked claims remain analyst judgments; completion pass is distinct from scientific support|

Source: `hybrid/config/models.toml`; `RESEARCH-SYSTEM.md`, pipeline table, supplemented by current `stages/obligation_checks.py`, `stages/falsify.py`, `stages/s6_verdict.py` and run-specific artifacts. Stage counts are in §4.

### Container and independent-scoring boundary

Jobs have no network, read-only root/input mount, writable staged output, unprivileged UID 10001, dropped capabilities, no-new-privileges and wall/memory/PID/thread caps. Host oracle data is absent from candidate jobs. A separate evaluator receives oracle data and hashed candidate prediction arrays, validates output shape/probability normalization, and computes scores. These constraints apply to toy, equivalence, disproof, obligation, construction, candidate and task-validation jobs in trusted mode. Local mode explicitly reports `trusted: false`. Sources: `P0-RESULT.md`, probe table; `FINAL-REPORT.md` §2; `hybrid/bench/README.md`.

P0 had 8 broad probes, 7 passing; its full contract suite had 44/45 tests pass. The bit-identical native/container fixture failed: HMM KL .12230803167784307 on macOS versus .12230803167784293 in Linux, a 10-ulp difference of about 1.39e−16. Accelerate versus OpenBLAS matrix-vector rounding caused it; RNN KL .005312276268561993 matched exactly. The failing assertion was retained. Subsequent trusted scoring uses container numbers consistently rather than mixing platforms. Source: `P0-RESULT.md`, probe 8.

Remaining container weaknesses are candidate access to test prefixes during fit because it shares the runner process, ability to manipulate runner timing, self-reported state size in capacity denominators, and no disk quota for `/out`. Output-copy bounds do not stop a running job filling VM storage. Source: `FINAL-REPORT.md` §2; `PODMAN-LIFECYCLE.md`; `RESEARCH-SYSTEM.md`, trust boundaries.

### Record integrity

Every ledger event carries sequence, timestamp, actor, data, previous hash and its own hash. Writing it as an explanatory equation, with serialize_json denoting the implementation's stable JSON serialization:
\[
H_e=\operatorname{SHA256}(H_{e-1}\Vert\operatorname{serialize\_json}(e\setminus\{	ext{hash}\})).
\]
Source: `RESEARCH-SYSTEM-SPEC.md`, store/ledger section; `hybrid/research/ledger.py:22–28`; `hybrid/research/store.py`. Cards and artifacts also carry SHA256 digests in ledger events. Inspect recomputes the chain and checks those file bytes. It cannot certify the truth of the model's interpretation, and without an externally preserved head a completely rewritten chain would not be authenticated history. The latter is an inference from the hash design, not a reported incident.

### Recall-falsifier investigation and implemented changes

The investigation used an operator-override plumbing run, `runs/plumb-20261011T012002Z`, card-537e161c v4, task `393eafr-assoc_recall_capacity`. It is separate from gate-reaching results. Source for all investigation numbers below: `results/falsifier-20261011/recall-investigation.md`.

The candidate and all five baselines scored accuracy 0 at K=16,32,48,256. The candidate was later treated as supporting low-performance limits despite never showing the capability. Sparse training supplied one relevant answer per 52–148 tokens, only 256 episodes/160 steps with vocabulary 1027. Candidate training truncated at 128 tokens and discarded every one of the 82 K=48 training queries. K=256 lacked matched-length training; candidate used d=32 against card d=64. Other discrepancies were 50 rather than 500 trials, nondistinct values, wrong claim quote/polarity, softened thresholds and changed validation code between attempts. These made the reported rule decisions scientifically unearned.

Mechanism controls separated learning failure from algebraic capacity. At d=64, ideal least-squares recall accuracies over K=16,32,48,256 were 1/1/1/.70 (.72 on a fresh lift, .84 with unit-normalized lift); linear attention gave 1/1/1/.54. At d=32 least squares gave 1/1/.96/.09 and linear attention .90/.78/.63/.06. The d=64 least-squares operator exceeded the card's claimed ≤.6 high-K limit. Hence neither the low learned scores nor the claimed decoding-capacity ceiling followed from the operator alone.

Current code implements these gates, with evidence in `hybrid/research/stages/falsify.py:119–256`, `hybrid/bench/falsifier_harness.py:113–396`, and `hybrid/bench/falsifier_eval.py:105–210`:

- Rule quotes must come from the exact named claim. A met falsifier is `refuted`, and cannot share a rule with a supported capability. Numeric thresholds/groups must be grounded in the quote. Absolute thresholds cannot be changed by subtracting uniform/unigram.
- Protocol parameters and deviations are explicit. Candidate `describe()` reports `train_max_len` and model hyperparameters; tasks exceeding its guaranteed training length or requiring a different model-side size cannot decide the claim.
- Generated data validates types, vocabulary, bounds, oracle probabilities, group sizes and training-length coverage; generation is replayed for reproducibility. Oracle self-scoring and analytic uniform/unigram reference levels must agree.
- Trained means beating the better of uniform and unigram by at least .05 nats KL or .05 accuracy. A worse-than-uniform unigram alone no longer lets an ignorant predictor qualify.
- Any rule capable of supporting a low-score limit must name a same-task positive capability control. Without passing that control the outcome is inconclusive.
- Recency controls for the last 1–8 tokens and 1/2-gram controls reject leaked or cheaply answered tests. Tasks declared lookup-solvable must score at least .99 with exact lookup in every group. A declared reference RLS mechanism must use the card's d and must agree with the rule/control limits.
- Each validation attempt hashes the staged harness and evaluator code. The current gate version is 2; older task artifacts are not reused as current-gate designs. Retries can repair named failures but cannot arbitrarily alter unmentioned rules.

These are implemented checks, not a claim that every possible task receives a custom analytic mechanism control. Reference controls are conditional on a declaration. Protocol extraction and omitted declarations remain model-authoring weaknesses. Exact agreement on generated data also cannot prove a designer's oracle encodes the intended scientific problem. No fresh learned recall confirmation was run for this fact sheet.

### Why S6 needed a rule-over-analyst correction

For card-03fa1d38, the analyst correctly read a low composition score but ignored that the candidate had failed the task's training gate. A low score from a mismatched/unlearned implementation did not settle the card's rule mechanism. `hybrid/research/stages/s6_verdict.py:52–82,146–185` now gathers task gates and overrides supported/refuted analyst outcomes that cite gated tasks to inconclusive, retaining `analyst_override` as commentary. Claims named in rules are decided by those rules. This still depends on the analyst listing all falsifier task IDs it used. Source: current code and `MORNING-REPORT.md` §4.

Limits: model judgments still decide novelty, radicality, reduction mappings without runnable checks, many proof/learnability obligations and analysis of unlinked claims. Numeric equivalence on fixtures does not prove universal equivalence. Two agreeing judges do not constitute a formal proof. Search coverage bounds prior-art claims. Both S3-passing cards had executed obligations still unknown or unexecuted. The run-5 complexity check's contradictory `violated` label illustrates that model-authored executable checks also need statement/result review. Sources: selected S3 gate artifacts and run-5 ledger line 667; `FINAL-REPORT.md` §§1–2; `hybrid/research/ORCH-STATUS.md`.

## 4. Results across all runs

Sources: current inspect for each named run; stored morning inspect files for runs 3–5. Revision counts count verdict events, not separate cards. Terminal-stage distribution is distinct from stage reach.

| Measure | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---:|---:|---:|---:|---:|
|cards|23|19|47|103|38|
|calls, including all-backend failures|233|178|438|1046|205|
|all-backend failed calls|7|0|2|1|0|
|S0 pass / fail|23/0|17/2|42/5|100/3|33/5|
|S1 pass / fail / revise events|21/2/45|17/0/34|41/1/82|100/0/199|17/0/42|
|S2 pass|2|0|6|12|1|
|S2 reduces_to_known|16|13|23|58|12|
|S2 fail|3|2|12|30|4|
|S2 undetermined|0|2|0|0|0|
|S3 pass / infeasible_proved / undetermined|0/1/1|0/0/0|0/5/1|2/9/1|0/1/0|
|S4 pass|0|0|0|2|0|
|S5 pass|0|0|0|2|0|
|S6 pass|0|0|0|2|0|
|escape requested / skipped / created|21/12/9|5/8/5|25/10/19|40/20/40|12/0/10|
|textbook / modified / unassessed|7/14/2|5/12/2|16/25/6|7/93/3|2/15/21|

Escape event denominators are not a single mutually exclusive partition. Use inspect's exact counters rather than subtracting requested minus skipped to infer created.

The five runs contain 230 cards and 2100 calls in total; these are arithmetic sums of the per-run rows. S2 assessed 196 cards: 21 passed, 122 reduced to known work, 51 failed, 2 undecided. S3 assessed 21: 2 passed, 16 recorded infeasible, 3 undecided. Those totals exclude the earlier llxprt attempt and operator-override plumbing runs.

Latest recorded stage, including failed or unfinished cards, from inspect and each latest `cards/<id>/v<N>.json`:

|Latest stage|Run 1|Run 2|Run 3|Run 4|Run 5|
|---|---:|---:|---:|---:|---:|
|S0|0|2|5|3|17|
|S1|2|0|1|0|4|
|S2|19|17|35|88|16|
|S3|2|0|6|10|1|
|S6|0|0|0|2|0|

Run 5's S0 row includes 5 duplicate failures and 12 passes waiting for S1; another 4 cards remained in S1 revision. The wall budget ended with work unfinished. Thus created cards are not the denominator for assessed S2 creativity. Source: `results/morning-20261011/inspect-run-20261011T095000Z.txt`; `MORNING-REPORT.md` §3.

|Reduction family|Run 1|Run 2|Run 3|Run 4|Run 5|
|---|---:|---:|---:|---:|---:|
|other_known|12|11|23|38|10|
|linear_attention|3|0|0|12|0|
|hopfield_or_kernel_memory|1|1|0|0|1|
|transformer_variant|0|1|0|1|0|
|rnn|0|0|0|6|0|
|ssm|0|0|0|1|0|
|lstm_gru|0|0|0|0|1|

Creativity counts by originating explorer model, textbook/modified: Opus run 1 6/10, run 2 5/12, run 3 16/20, run 4 7/44, run 5 2/7; GPT run 1 1/4, run 2 no assessed cards, run 3 0/5, run 4 0/49, run 5 0/8. The GPT zero-transplant observation applies to runs 3–5, not run 1. These labels are S2 critic judgments and do not establish novelty. Source: inspect creativity blocks.

## 5. What looks promising

Nothing in the discovery loop has established a baseline-beating learner. The evidence supports prioritizing narrower mechanisms or improvements to the checking system.

`prior_art_undertested` is a candidate-selection cue. Examples include the Koopman operator-power readout, phase-only rule memory, sparse residual-slot collision behavior and parity-constraint learner in the S2 reduction table. Each was reduced to known work, so exploring an untested property would be a study of a known method or a specified modification. Card-03fa1d38's class-remap law escaped those reductions, but its implemented token model did not exercise the intended demonstration/query mechanism. Card-ffcfe0e5's sticky ceiling reproduced an idealized switching result, while its broad one-shot generalization claim failed. Sources: §2's cited S2 and S3 ledger events. These are proposals for bounded follow-up, not discovery claims.

- Mathematical evidence: the low-data structured-prior advantage in Experiment 1 passed a prespecified rule. It is limited to known-state synthetic generators and fixed tiny controls; broader fewer-token advantage was not established. Source: `RESULT.md`.
- Mathematical candidate: card-03fa1d38 measured a time slope .997 for a claimed 1.0. Its symbolic rewrite/rule mechanism remains untested because it did not train beyond trivial references on its own tasks. Source: `MORNING-REPORT.md` §1.
- Process evidence: the escape operator produced both first-run S3 entrants and one run-4 S6 entrant. This shows reach through the pipeline, not prediction improvement. Sources: `FINAL-REPORT.md` §1; `MORNING-REPORT.md` §2.
- Process evidence: GPT had zero textbook transplants among its assessed cards in runs 3–5. No measured candidate improvement follows from that label. Source: inspect creativity blocks.
- Process evidence: numeric reductions, executed S3 violations and the recall investigation exposed false claims and scoring failures. Source: `FINAL-REPORT.md`; `results/falsifier-20261011/recall-investigation.md`; `MORNING-REPORT.md`.

## 6. Open questions and decisions

### OpenAlex contact and Semantic Scholar key

OpenAlex is a free open index of scholarly works, authors and venues, used here to search paper metadata and follow references/citations. `mailto` sends a contact email with API requests so the service can identify and contact the client and place it in the documented polite pool. It is not a login or proof of literature completeness. The implementation enables it only when `BIGBRANE_CONTACT_EMAIL` is configured. Sources: `hybrid/literature/README.md`, environment/rate-limit sections; `hybrid/literature/client.py:25,79–100`; `FINAL-REPORT.md` §5.

Semantic Scholar is AI2's research-paper search service and citation graph. A service-issued API key identifies an application and changes which rate-limit allocation its requests use; it does not add scientific authority. The project currently calls the shared anonymous pool. Important implementation status: the inspected `Literature` constructor, `Request` type and transport contain no Semantic Scholar API-key setting or `x-api-key` header. Obtaining a key alone will not change Bigbrane's requests; integration is outstanding. Sources: `hybrid/literature/client.py:79–100`; `transport.py:35–39,102–115`; `sources/semantic_scholar.py:1–20`. The exact service-side key allocation was not verified here, so do not promise a numeric upstream limit or that a key/contact will end all skips.

Local pacing is arXiv ≥3 seconds between requests, OpenAlex ≥0.1 seconds, Semantic Scholar ≥1 second. These are client settings, not confirmed server entitlements. There are four attempts (initial plus three retries), default waits 5/10/20 seconds, numeric Retry-After honored up to 120 seconds. Exhausted HTTP 429 enters a 600-second cooldown; skipped-source receipts are then emitted without a network call. Cached successes still work, and failures are not cached. Source: `hybrid/literature/transport.py:19–21,58–63,79–147`; `README.md`, sources/rate limits.

Observed source cooldown skips, Semantic Scholar/OpenAlex: run 1 92/7; run 2 91/93; run 3 196/98; run 4 461/457; run 5 76/77. Additional HTTP 429 failures are separate: 10/1, 4/6, 9/9, 26/35, 7/8 respectively. Sources: `FINAL-REPORT.md` §4; `results/morning-20261011/lit-*.txt`. Run 4 had zero empty searches because local arXiv supplied 2958 hits. A nonempty result is not exhaustive coverage.

S6 fail-on-falsifier remains a decision: should a validated rule that refutes a card change its overall S6 verdict to fail? Currently completion still records pass while claim-level evidence records the refutation. Source: `MORNING-REPORT.md` §6.

Bench interface decision: hand-built rule mechanisms need explicit demonstration and query events. Consecutive token-pair training did not test the intended rules. Increasing budget alone does not establish that the protocol matches the model. Source: `MORNING-REPORT.md` §§1,5–6.

### SIWC refresh risk and earlier decisions resolved overnight

SIWC is Sign in with ChatGPT, the OAuth route that lets the gateway use the account's ChatGPT plan. Its rotating refresh token replaces expiring access tokens; loss of that grant can interrupt unattended work. Eight scheduled refreshes succeeded overnight, but grants can still be revoked or return `invalid_grant`. The qualified issue is [openai/sign-in-with-chatgpt-devkit#5](https://github.com/openai/sign-in-with-chatgpt-devkit/issues/5), linked in `hybrid/gateway/STATUS.md`, SIWC durability section. That source reports account-security-state explanations in related issues and no demonstrated browserless recovery. It also contains an older refresh-result placeholder; prefer the later successful observations in `MORNING-REPORT.md` §4.

The gateway refreshes at max(earliest_refresh_at, expires_at−300 seconds), serializes refreshes, saves replacement tokens before use, never retries a consumed/rejected refresh token, and records outcomes without credentials. After a refresh refusal it can use the current access token until expiry; after loss of usable access, judge B and the falsifier designer fall back to local Qwen. A new browser login is required. Pacing is an application precaution, not a vendor-guaranteed remedy for grant revocation. Source: `hybrid/gateway/STATUS.md`, changes and still-uncertain sections; `MORNING-REPORT.md` §6.

`FINAL-REPORT.md` ended with decisions about restoring a second model family and building a local index. Both were implemented overnight through a fresh ChatGPT login and local arXiv metadata, so they should not be presented as wholly unresolved. The unfinished decisions are key integration/contact configuration, coverage beyond arXiv, interface/protocol fidelity and S6 disposition. Source: `FINAL-REPORT.md`, closing decisions; `MORNING-REPORT.md` §§4–6.

### Other measurement and operational decisions

- Larger bench: default all-baseline runs provide more training data and a broader probe than the research-loop quick configuration. All five baselines were added to the current quick config, but the switch to the larger default is still a recommendation. Source: `FINAL-REPORT.md` §5; `hybrid/config/bench_quick.toml`.
- Fingerprint distinction: transformer and linear attention remain difficult to separate. A long-gap copy or beyond-state-capacity recall task is a proposed discriminating probe, not an implemented success. Source: `FINAL-REPORT.md` §5.
- Formal checking: executed numerical/SymPy obligations were added; there is no Lean proof-assistant certification. Agreement can pass cards with checks unknown. Whether all required checkable obligations should block progression until discharged is a proposal from the selected S3 evidence, not an existing decision rule.
- Host/container numeric identity: P0's bitwise host match remains failed. The selected policy keeps scoring in one container build. Requiring host-bit-identical outcomes would reopen that decision. Source: `P0-RESULT.md`.
- Services at handoff: morning records no research process, finished harvest, gateway PID 9118 on port 8765, Podman VM running without Bigbrane containers, and local Qwen loaded. These are recorded handoff states, not freshly checked process status. Source: `MORNING-REPORT.md` §7. Earlier FINAL-REPORT teardown states were superseded.
- HMM optimization: high-budget HMM worsening could reflect local fitting/selection behavior. The causal mechanism is unresolved; a new development investigation is needed before attributing the entire stress gap to the structured prior. Source: `RESULT.md`, limits and follow-up.

## 7. Next steps

Stated priorities and current status, from `FINAL-REPORT.md` §5, `MORNING-REPORT.md` §§4–6 and current configuration:

|Priority|Status at October 11 handoff|Next action from source|
|---|---|---|
|Literature reliability|Local arXiv index completed; anonymous remote throttling persists|Configure OpenAlex contact and integrate Semantic Scholar key; consider OpenAlex/local coverage beyond arXiv|
|Second model family|GPT restored; local Qwen remains alternate judge/designer|Maintain usable refresh/login path and separate-family fallback|
|Escapes and duplicate filtering|Reserved slots, graveyard context and ancestor exception implemented|Assess their effect on valid measured candidates rather than raw idea labels|
|Card-specific falsifiers|Different-family designer and validation gates implemented|Make built models able to consume their own demonstration/query protocols|
|Executed S3 obligations|Numerical/SymPy scripts and replay implemented|Complete missing checks; formal proofs remain outside current pipeline|
|Fingerprint discrimination|All five baselines available; attention-family overlap remains|Add principled recall/copy probes separating mechanisms|
|Larger benchmark|Research loop still defaults to its short configuration|Use full default comparison for models that can follow the target protocol|

Proposals that follow directly from evidence, not already established requirements:

1. Prioritize an explicit demonstration/query candidate contract and protocol-matched positive controls before further discovery runs. Independently run card-03fa1d38's hand-factorized class-remap and ablation protocols, because the scored token model did not test its central mechanism.
2. Review statement-to-check correspondence for executed S3 obligations, including the run-5 complexity-label inconsistency. Consider blocking mathematical-support labels when required executable checks remain unknown, without retrospectively changing recorded stage outcomes.
3. Preserve original run ledgers and publish corrected claim-ledger reconstructions with the source task links and correction reason. Keep S6 completion, supported claims and overall acceptance separate.
4. Diagnose HMM optimization on new development generators. Then freeze a separate study of unknown state count or predictive-state learners, broader generator classes and larger neural controls. Do not revise the completed Experiment 1 result in place.
5. Measure search-policy changes in matched replicated runs if claiming that graveyard/escapes/GPT explorers improve discovery. Current trajectories show card reach and critic labels, without a causal comparison or a baseline-beating learner.

Sources motivating proposals: `RESULT.md`; selected card/bench artifacts in §2; `LIT-HARNESS-2026.md`, implications for Bigbrane; `MORNING-REPORT.md` §§1,5–6.

## 8. Glossary

Definitions below paraphrase the project's usage in `EXPERIMENT.md`, `RESEARCH-SYSTEM.md`, `hybrid/bench/README.md`, `hybrid/literature/README.md`, `hybrid/gateway/STATUS.md` and `PODMAN-LIFECYCLE.md`. Equations are explanatory restatements of those sources, with natural logarithms.

|Term|Explanation for the report reader|
|---|---|
|KL divergence|Expected extra log loss from predicting q when the target distribution is p: $\operatorname{KL}(p\Vert q)=\sum_vp_v\log(p_v/q_v)$. Zero means equal distributions on the target support; direction matters. Here measured in nats per target|
|JS divergence|Symmetric distribution distance built from each distribution's KL to their average; formula in §2. Used to compare predictions, without proving identical mechanisms|
|HMM|Hidden Markov model: unseen finite state evolves by a transition matrix and emits observable symbols. Filtering tracks a probability distribution over that state|
|RNN|Recurrent neural network: learned hidden state is updated as each token arrives. Experiment 1 uses an ungated tanh RNN; bench GRU adds learned update/reset gates|
|SSM|State-space model: persistent state follows a specified dynamics equation driven by inputs. The bench uses diagonal complex linear recurrence inside nonlinear blocks|
|Linear attention|Associative retrieval computed from an accumulated feature-key/value matrix rather than all pairwise token attention scores; accumulator has fixed size at fixed width|
|Delta rule|Error-correcting write to a fast associative matrix: update toward the desired value while subtracting its current prediction for that key|
|Oracle|An evaluator's exactly known target distribution or answer for a synthetic task. It is withheld from learned-model fitting. It does not mean an all-knowing model|
|Falsifier|A stated observable outcome that would contradict a claim. The model-designed task, implementation and decision rule must be checked before treating its outcome as evidence|
|Prior art|Previously known constructions or results. `prior_art_undertested` means retrieved related work exists but the classifier thinks the specific addition lacks tests; it is not proof of novelty|
|S0|Explore: generate a paper-grounded theory card and validate its schema/diversity|
|S1|Well-formedness: critique definitions/claims and execute the card's toy example|
|S2|Prior-art/reduction gate: search literature and test mappings to known mechanisms|
|S3|Feasibility: obligations, attempted disproofs, executable/replayed checks and separate-family judges|
|S4|Construct: build and check a candidate implementation after S3 pass|
|S5|Evaluate: run candidate/baselines and separate trusted scoring; completion alone implies no scientific success|
|S6|Claim verdict: record support, refutation, inconclusive or untested status from rules and analysis|
|Lens|A source field assigned to an explorer, such as Koopman theory or chemical reactions, supplying papers and possible mathematical objects|
|Escape|A new child card asked to break the explicit invariant that made its parent reducible to a known method|
|Graveyard|Feedback assembled from earlier failed/reduced cards, retaining reasons so explorers can avoid rediscovering them|
|Podman|The container engine used to run generated programs with constrained files, network and resources. On this Mac it uses a Linux VM|
|Gateway|Bigbrane's localhost service that routes model calls, translates request formats, stores OAuth records privately, meters usage and refreshes access|
|SIWC|Sign in with ChatGPT: OAuth access to the ChatGPT plan through Bigbrane's own client, including expiring access tokens and rotating refresh tokens|
|arXiv|An open repository of scholarly preprints. Metadata/abstracts are searchable literature evidence; listing there does not mean peer review|
|OAI-PMH|Open Archives Initiative Protocol for Metadata Harvesting: a paginated metadata exchange protocol, used here to copy selected arXiv sets into a local search index|
|OpenAlex|An open scholarly metadata index including works and citation relationships; the loop uses its search and graph endpoints|
|Semantic Scholar|AI2's paper-search/citation-graph service; `s2` in literature receipts denotes this service and is distinct from research stage S2|
|Nats|Logarithmic loss units using the natural logarithm; conversion to bits divides by ln(2)|
|Inference / proposal|Inference is an interpretation derived from observed evidence; proposal is a recommended future action. Neither is an executed result|


## 9. Chart data

Experiment 1 aggregate KL coordinates, x=training tokens, y=nats/target. Source: `RESULT.md` learning curves.

- HMM: `(512,.09345) (2048,.04307) (8192,.02943) (16384,.04497)`.
- RNN: `(512,.12839) (2048,.05397) (8192,.01277) (16384,.00830)`.
- Attention: `(512,.17001) (2048,.09121) (8192,.03979) (16384,.02477)`.

Cards reaching stages, x=run number. Source: inspect verdict counts; S0 includes every created card, later reach is previous-stage pass count.

- Created: `(1,23) (2,19) (3,47) (4,103) (5,38)`.
- S2 assessed: `(1,21) (2,17) (3,41) (4,100) (5,17)`.
- S3 entered: `(1,2) (2,0) (3,6) (4,12) (5,1)`.
- S4/S5/S6 entered: each `(1,0) (2,0) (3,0) (4,2) (5,0)`.

### Reduction-family coordinates

x=run number, y=cards labeled `reduces_to_known`. Source: the per-run inspect reduction-family blocks and §4.

- other_known: `(1,12) (2,11) (3,23) (4,38) (5,10)`.
- linear_attention: `(1,3) (2,0) (3,0) (4,12) (5,0)`.
- hopfield_or_kernel_memory: `(1,1) (2,1) (3,0) (4,0) (5,1)`.
- transformer_variant: `(1,0) (2,1) (3,0) (4,1) (5,0)`.
- rnn: `(1,0) (2,0) (3,0) (4,6) (5,0)`.
- ssm: `(1,0) (2,0) (3,0) (4,1) (5,0)`.
- lstm_gru: `(1,0) (2,0) (3,0) (4,0) (5,1)`.

### S5 timing slopes and coordinate series

Source for every number in this block: `runs/run-20261011T042400Z/artifacts/bench_raw/card-03fa1d38-9.json`, `entrants.<name>.scaling`. Tail fits use the last three points; measurements are finite-range prediction runtimes with batch 2, excluding model fitting. Use labels rather than implying an asymptotic proof. Below, runtimes are rounded to 9 significant digits for plotting; exact floats remain in the source.

|Plot category x|Entrant|Tail slope y|
|---:|---|---:|
|1|card-03fa1d38|0.996592309873441|
|2|n-gram|−0.007938236556203536|
|3|GRU|0.9640099087410472|
|4|SSM|0.9568197111902896|
|5|linear attention|0.9610331053110368|
|6|transformer|1.4157532162731972|

x=prefix length, y=median seconds:

- candidate: `(64,0.00187655800) (128,0.00358882300) (256,0.00775352500) (512,0.0162104310) (1024,0.0308679330)`.
- GRU: `(64,0.00272018900) (128,0.00547646100) (256,0.0106959190) (512,0.0209338770) (1024,0.0407014520)`.
- linear attention: `(64,0.00125321900) (128,0.00185201400) (256,0.00326031700) (512,0.00664151199) (1024,0.0123554740)`.
- n-gram: `(64,0.0000159170013) (128,0.0000152090070) (256,0.0000151669956) (512,0.0000148749969) (1024,0.0000150010019)`.
- SSM: `(64,0.000957049997) (128,0.00154559499) (256,0.00295789901) (512,0.00582167201) (1024,0.0111441310)`.
- transformer: `(64,0.00119492900) (128,0.00233119000) (256,0.00422625100) (512,0.0119137440) (1024,0.0300831740)`.

### Additional compact plots

Experiment 1 stress contrasts, x=training tokens, y=control-minus-HMM KL, source `RESULT.md` stress table: RNN `(2048,-0.01779) (16384,-0.05108)`; attention `(2048,0.05869) (16384,-0.03215)`. The zero line marks equality, not the full advantage rule.

Literature cooldown skips, x=run number, y=skipped source requests, sources `FINAL-REPORT.md` §4 and `results/morning-20261011/lit-*.txt`: Semantic Scholar `(1,92) (2,91) (3,196) (4,461) (5,76)`; OpenAlex `(1,7) (2,93) (3,98) (4,457) (5,77)`. Plot terminal HTTP 429 failures separately from these skips.

## Coverage and unresolved evidence

All nine requested sections are populated. The fact sheet covers dated development and platform choices, Experiment 1 equations/results/stress, five baseline mechanisms and fingerprint limits, lens inventory, five concrete numeric reductions, the two S6 entrants and run-5 survivor, three additional S3-refuted cards, stages and model families, container/ledger checks, falsifier investigation and implemented gates, corrected S6 status, exact per-run counts, decisions, proposals, glossary and plot coordinates.

Sources sampled or skimmed for context include `PLATFORM-CORE-ASSESSMENT.md`, `HYBRID-PLAN.md`, `LIT-HARNESS-2026.md`, `LIT-SKILLS-2026.md` and `LIT-MATH-CLAIM-FAILURES.md`; their external results are not presented as reproduced Bigbrane results. The prototype specification describes intended behavior, while current code and recorded run artifacts establish implemented or executed behavior.

Unresolved evidence and report qualifications:

- The exact timestamp of Andrew's opening question and reconciliation with older internal document dates are unavailable. Preserve the date qualification in §1.
- The current Semantic Scholar service-side authenticated quota and effectiveness of OpenAlex contact under present provider policy were not independently verified. The code's anonymous pacing is verified; key integration is absent.
- No model or experiment was rerun for this fact sheet. The suite result is log-confirmed prior evidence. Run inspect verified hashes and regenerated only the permitted HTML report views; original cards, ledgers and scientific artifacts were left unchanged.
- The two S3 passes came from judge agreement with some checks unknown/unexecuted. Neither establishes a formal theorem, learned few-shot advantage or baseline superiority. Generic and card-specific benchmark protocol mismatch remains unresolved.
- Run 4's original S6 counts and later corrected reconstruction are explicitly separated. The reconstruction source is the morning report/current merger behavior, not a new scored model run.
- Run 5 contains a contradictory complexity-check status and explanation; §2 records it without presenting it as a valid mathematical refutation. Its information-identifiability disproof stands on separate evidence.
- Stronger baseline comparisons, learned-code verification, parent ablations, general proof coverage, future OAuth durability and controlled search-policy benefits remain future work.

