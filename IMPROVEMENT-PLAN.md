# Bigbrane improvement plan: aim the loop at the recall-versus-cost frontier

Date: 2026-10-11. Status: plan only. No code, config, run or result file was changed to write it. Sources are named beside each claim; paper claims carry URLs that were checked with live searches on 2026-10-11, and arXiv IDs were also looked up in the local index (`data/arxiv-index/arxiv.sqlite`, 1,419,455 records, read-only query).

## Summary

Stop judging candidates until the bench can rank known models. Today it cannot: under `hybrid/config/bench_quick.toml` (200 steps) no entrant learned the card-designed recall task (`results/falsifier-20261011/recall-investigation.md`, every entrant at 0.000 accuracy), on `quick.toml` the transformer baseline sits at chance on recall (0.062 at K=2), and on `default.toml` linear attention and the transformer are flagged `behaves_like` each other (`hybrid/bench/BASELINES.md`). Candidate numbers from this bench therefore cannot be read against the target.

The plan, in dependency order:

1. **Fix the question** (section A). Replace "sub-quadratic plus few-shot", which known models already meet, with a measurable target: move the recall-versus-state and recall-versus-compute Pareto frontier past the best known fixed-state and hybrid baselines. This target is an assumption until Andrew confirms it.
2. **Calibrate the bench first** (B, C). Add MQAR from Zoology and the copy and n-gram lookup tasks from "Repeat After Me", add Gated DeltaNet, Mamba-2, sliding-window attention and hybrid baselines in pure PyTorch, and require the bench to reproduce the published orderings before any candidate result counts.
3. **Give hand-built mechanisms a demonstration and query interface** with same-task positive controls (D).
4. **Tighten the gates** (E): S6 fails a falsified card, S3 cannot pass with required obligations unchecked, every card states how it handles recall beyond a fixed state, and the reduction critic knows the 2024 to 2026 families.
5. **Point the generator at the frontier** (F) and allow labelled improvements to known mechanisms.
6. **Check on real text** (G) with a few-million-parameter language model, judged by loss against compute.
7. **Seed the literature** (H) for the recall frontier. The local arXiv index already holds every frontier paper checked for this plan; the gap is in the seed list, the lenses and the graveyard.

Phases with exit criteria are in section I. Decisions for Andrew are in section J.

---

## A. Target and success criteria

### A.1 The question (assumption pending Andrew's confirmation)

> Is there a sequence-mixing mechanism that, for a given inference-state budget and a given training compute per token, recalls more in-context associations (and copies longer spans) than the best known fixed-state and hybrid architectures, without losing language-model quality at equal compute? Secondary: does it reach a given recall accuracy with fewer training examples?

The obstacle is information-theoretic, and the plan treats it as a constraint every card must answer. "Repeat After Me" proves that a generalized state space model with state space S has error greater than 1 − |S|/D^L on uniformly random strings of length L over D tokens (Theorem 2.7), so a fixed state of b bits cannot copy strings with L·log2(D) well above b ([PMLR v235](https://proceedings.mlr.press/v235/jelassi24a.html), [arXiv 2402.01032](https://arxiv.org/abs/2402.01032v1)). Zoology shows that gated-convolution models need model dimension to grow with sequence length to solve MQAR, while attention solved it at every tested length with d = 64 ([arXiv 2312.04927](https://arxiv.org/abs/2312.04927v1)). Based measures a recall-versus-state tradeoff that holds within and across architecture classes and frames progress as moving that Pareto frontier ([arXiv 2402.18668](https://arxiv.org/abs/2402.18668), [PMLR v235](https://proceedings.mlr.press/v235/arora24a.html)).

Given that bound, a win under this target is a better exchange rate: more recall per state float, per FLOP, or per training example than the known frontier, or a state that grows more slowly than the KV cache while keeping attention-level recall. Production labs already ship hybrids (the brief cites Qwen3.8-Flash-Next with 36 Gated DeltaNet layers and 12 attention layers; that layer count is from the brief and was not independently verified), so the hybrid frontier is the bar, and pure attention and pure recurrence are its two ends.

### A.2 Metrics

All metrics are computed by the trusted evaluator from prediction arrays, as now.

| Metric | Definition | Where |
| --- | --- | --- |
| R(s, N, K) | MQAR accuracy on query positions, as a function of inference-state size s, sequence length N and number of key-value pairs K | new `mqar` task |
| C(L) | copy accuracy, per token and per string, for strings of length L, trained on L ≤ L_train and tested up to 8 × L_train | new `copy` task |
| Lookup | suffix-key and prefix-key n-gram lookup accuracy (prefix-key is a control that fixed-state models should pass) | new `ngram_lookup` task |
| State floats | floats a model must keep per sequence to continue from position N: KV cache 2 · layers · d · N for attention, 2 · layers · d · w for a window w, the recurrent state for fixed-state models, the sum for hybrids. Computed by the host from the architecture for baselines, and measured for candidates (D.4) | registry in `hybrid/bench/baselines/__init__.py` |
| FLOPs per token | analytic forward FLOPs per token at length N (training cost is a fixed multiple), from a per-family formula the bench owns | new `hybrid/bench/cost.py` |
| Wall time slope | the existing scaling probe; reported, never used alone, because single slopes move between runs on the shared VM (BASELINES.md: SSM 1.37 on one run, 1.19 to 1.26 elsewhere) | `scaling.py` |
| Sample efficiency | training examples (and tokens) to reach R ≥ 0.9 at a fixed (N, K); demonstrations needed on the few-shot rule task | `metrics.tokens_to_target`, extended |
| LM quality | validation loss per token (or bits per byte) against training FLOPs on a public text set | section G |

### A.3 Pareto framing

For each (N, K) setting the bench draws two plots:

1. **Recall against state.** x = state floats (log scale), y = R. Every baseline family is swept over its state-setting hyperparameter, so each family contributes a curve. The **known frontier** is the upper envelope of all baseline points.
2. **Recall against compute.** x = FLOPs per token at length N, y = R, same envelope.

A candidate point gets two distances to the envelope: the vertical gain (accuracy above the envelope at the same state or FLOPs) and the horizontal ratio (state or FLOPs the envelope needs to reach the candidate's accuracy, divided by the candidate's). `hybrid/bench/frontier.py` computes both, with seed intervals.

### A.4 What counts as a win, a null result and a refutation

The thresholds below are proposals for Andrew to set (J, decision 1). Every comparison uses at least 3 seeds and the learning-rate sweep in B.4.

- **Win.** At 2 or more of the 3 (N, K) evaluation settings, the candidate's horizontal ratio is at least 2 (half the state or half the FLOPs for the same accuracy) or its vertical gain is at least 0.10, with the 95% seed interval of the gain above 0; its per-token cost exponent matches its claim; and on the real-text check (G) its loss at equal training FLOPs is no worse than the best baseline in its cost class by more than that baseline's seed spread. A labelled improvement to a known family (F.3) that meets this counts as a **known-family win** and is reported apart from a new-mechanism win.
- **Null result.** The candidate trains (beats the trivial references and meets its positive control) but stays within the seed interval of the envelope, or below it. Recorded with its curve in the frontier registry and the graveyard; not a refutation of the idea, and not grounds to scale it.
- **Refutation.** On a calibrated task with its positive control met, a pre-registered rule shows the card's own prediction failing: recall below the card's predicted curve while a same-state baseline meets that level, a measured cost exponent above the claim, or measured state growth faster than claimed. A refuted card fails S6 (E.1).
- **No result.** The candidate did not train, or no positive control was met. This is neither null nor refutation, and it blocks nothing except that card's claims (`inconclusive`, as `falsifier_eval.decide` already does).

---

## B. Bench calibration first

### B.1 What the published tasks are (verified)

**MQAR, Zoology** (Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré; [arXiv 2312.04927](https://arxiv.org/abs/2312.04927v1), [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/448fc91f669c15d10364ee01d512cc10-Paper-Conference.pdf), code [HazyResearch/zoology](https://github.com/HazyResearch/zoology)):

- Motivation: earlier associative-recall synthetics used one query per input, at a fixed position, with vocabularies smaller than the model dimension (|V| < 50). MQAR asks for multiple recalls at varying positions with a vocabulary larger than the model dimension (paper section 3.2).
- Format, from the generator docstring in `zoology/data/associative_recall.py` ([commit d90de38](https://github.com/HazyResearch/zoology/blob/d90de385ad9febeae8b53d98f105af089ac2a477/zoology/data/associative_recall.py)): key-value pairs first, then queries that repeat earlier keys; labels are −100 (ignored) everywhere except the position after each query key, whose label is that key's value. Defaults: `vocab_size` 8,192 ("large vocabulary sizes (>1k) can be important for highlighting differences between model architectures"), `num_kv_pairs` 8, `power_a` 0.01 (the gap between a key and its query follows a power law, as repeated bigrams do in real text), `random_non_queries` True (non-query positions are filled with random tokens). For Figure 2 the paper varied sequence length from 64 to 512 and pairs from 4 to 64.
- Keys are drawn from the first half of the vocabulary and values from the second half, without repeats in the prompt. This detail comes from a third-party report on the generator ([W&B report](https://wandb.ai/stud76/zoology/reports/MQAR-and-data-driven-expansion-of-state--Vmlldzo3NDUwMzI2), citing Appendix E.1 of the paper) and should be checked against the code when the task is written.
- Training recipe: two-layer models that interleave sequence mixing and MLPs, four learning rates swept between 1e-4 and 1e-2, maximum test accuracy reported (paper section 4.3). The repository's example config uses 10,000 training and 1,000 test examples and `max_epochs=20`.
- Published result used for calibration: attention solves MQAR at all tested sequence lengths with d = 64, while the gated-convolution models do not exceed 0.9 accuracy unless d ≥ N (paper, Claim 1).

**Recall-versus-state frontier, Based** ([arXiv 2402.18668](https://arxiv.org/abs/2402.18668)): on MQAR, larger recurrent state almost always improves accuracy within a family; at a fixed state, families differ, and Mamba made the best use of a limited state among prior architectures; sliding-window attention's recall range is limited by its window; linear attention plus a 64 to 128-token sliding window recovered 90.8% of full attention's recall accuracy.

**Copying, "Repeat After Me"** (Jelassi, Brandfonbrener, Kakade, Malach, ICML 2024; [PMLR](https://proceedings.mlr.press/v235/jelassi24a.html), [arXiv 2402.01032](https://arxiv.org/abs/2402.01032v1), [PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/jelassi24a/jelassi24a.pdf)):

- Copy task: the model reads a string and must reproduce it. Trained on strings of at most 50 tokens and tested up to 1,000, with string-level accuracy; transformers generalized to longer strings while the LSTM and Mamba dropped to zero almost immediately past the training length (section 3.3). The experiments used models of about 160M parameters; this bench will be far smaller, so it targets the orderings, never their numbers.
- Theory: a depth-2 transformer of dimension O(n log D) copies strings with error below the repeated n-gram probability (Theorem 2.3); any GSSM has error above 1 − |S|/D^L (Theorem 2.7).
- n-gram lookup, suffix-key version: a sequence, a separator, then an n-gram from the sequence; output the k tokens that followed it. Trained on at most 30 tokens; transformers held up to length 100 and GSSMs did not. Prefix-key version: the n-gram comes first, so a fixed-state model can store the key and wait; GSSMs reached perfect length generalization there. This pair gives the bench a positive and a negative control for fixed-state models.

### B.2 New tasks

| Task | File | Generator | Oracle | Groups |
| --- | --- | --- | --- | --- |
| `mqar` | `hybrid/bench/tasks/mqar.py` | Zoology format: K pairs (keys and values from disjoint vocabulary halves, distinct), then queries at power-law gaps (`power_a` 0.01), other positions random tokens; per-sequence `loss_positions` marking answer positions | the queried key's value at each query position | (N, K); also gap bucket |
| `copy` | `hybrid/bench/tasks/copy.py` | BOS, a uniform string of length L over D content tokens, a COPY token, then the string again | the next string token | L in training range and at 2×, 4×, 8× |
| `ngram_lookup` | `hybrid/bench/tasks/ngram_lookup.py` | suffix-key and prefix-key variants as in the paper | the k tokens after the n-gram | variant × length |
| `recall` (existing) | unchanged | kept for continuity, demoted to a smoke test | | |

Two changes the existing task contract needs:

1. **Answer-only loss.** Every bench baseline now trains on next-token loss at every position (`bb_neural._TorchCandidate._loss`). On MQAR with random fillers most targets are unpredictable noise; the recall investigation found the same dilution (256 answer tokens among 25,168 training tokens). `TaskData` gains an optional per-sequence `loss_positions`; the runner passes it to `fit(train, loss_positions=None)`; the baselines use it when present. This is an API change to `hybrid/bench/api.py`, `candidate_template.py` and the S4 builder prompt, made once, with no fallback path.
2. **Multiple scored positions per sequence.** MQAR and copy score many positions per sequence. Today every test prefix is one independent sequence. Keep that rule (it stops a candidate reading later answers), and draw one scored query position per test sequence, with enough test sequences per group (n_test 1,000 per (N, K), as in the Zoology example config).

### B.3 State-size sweep

Each family is swept over the hyperparameter that sets its state, at fixed depth (2 mixing layers plus MLPs, as Zoology), so that each contributes a curve on the recall-versus-state plot:

| Family | Swept | State floats per sequence at length N (2 layers) |
| --- | --- | --- |
| Transformer | d ∈ {32, 64, 128} | 2 · 2 · d · N |
| Sliding-window attention | w ∈ {16, 32, 64, 128} at d = 64 | 2 · 2 · d · min(w, N) |
| Linear attention (existing, plus feature dim) | d ∈ {32, 64, 128} | 2 · (d² + d) |
| Gated DeltaNet | d ∈ {32, 64, 128}, heads ∈ {1, 2} | 2 · heads · d_k · d_v |
| Mamba-2 (SSD) | d_state ∈ {16, 64, 128} at d = 64 | 2 · d_inner · d_state |
| Hybrid A: Gated DeltaNet + one global attention layer | d ∈ {32, 64, 128} | fixed part + 2 · d · N |
| Hybrid B: linear attention + sliding window (Based recipe) | w ∈ {16, 64} × d | fixed part + 2 · d · w |
| GRU, diagonal LRU SSM (existing) | unchanged | as now; kept as weak references |

Evaluation grid: N ∈ {64, 256, 512} and K ∈ {4, 8, 16, 32, 64} with K ≤ N/4, trained on mixed K per N and reported per K. Copy: L_train = 32, test L ∈ {32, 64, 128, 256}, D = 16 content tokens (a design choice for this bench, not from the paper).

### B.4 Training protocol

- Learning rate: 3 values on a log grid from 1e-4 to 1e-2 per (family, width), report the best on a held-out split, as Zoology does with 4.
- Steps: fixed per N, set so the full transformer at d = 64 reaches its plateau on MQAR at N = 256 (measured in Phase 3, not assumed). The current 200-step and 600-step budgets are below what recall needs here: under the 200-step quick budget none of the five baselines learned the card-designed recall task at K = 16 with V = 1,024 (recall investigation, open questions).
- Seeds: 1 seed for the learning-rate sweep, then 3 seeds at the chosen rate for every point on the plots.
- Vocabulary: V = 512 for the grid (smaller output layer on CPU) and one check at V = 8,192 at N = 256, d = 64, because Zoology reports that a large vocabulary separates architectures.
- Short convolutions: the Zoology README notes that its comparisons give every architecture the Based recipe of a short convolution. Add the same short causal convolution to every fixed-state baseline, so differences are not a missing local mixer.

### B.5 Calibration acceptance

The bench is calibrated when all of the following hold on seeds 1, 2 and 3 in the pinned one-thread Podman CPU container (`trusted: true`). Until then, no S5 result, falsifier decision or frontier claim is reported as evidence.

**Execution policy (decision B10, 2026-10-16).** Acceptance evidence comes only from the pinned one-thread Podman CPU container. Seed 0 sweeps the learning rate (B.4) and is descriptive; the criteria are judged on the three post-selection seeds 1, 2 and 3. Host MPS runs are diagnostics: a report that holds any of them is labeled `scope: diagnostic`, and `python -m hybrid.bench.ladder gate` refuses it, as it refuses a report that does not record `trusted: true`, `policy.seeds = [1, 2, 3]` and the current code hash. The thresholds below are fixed in `hybrid/bench/ladder_criteria.py` before the acceptance sweep and stored in every report as `policy`. Criteria 2, 4, 6 and 9 were revised under decision B10, on the evidence in `results/phase10-diag3-20261013T2009Z/summary.md` ("Proposed criterion revisions").

The brief asks that attention beat fixed-state models on recall at a matched state size. Attention's state is its KV cache, which grows with N, so "matched" is tested two ways: at matched model width d and depth, which is the Zoology comparison (criteria 1 and 2), and on the recall-versus-state plot, where attention held to a fixed state budget is sliding-window attention (criterion 4). Based reports that Mamba expands the frontier beyond sliding-window attention, so at equal state a good fixed-state model should beat the window on long-gap queries while full attention stays above both.

1. **Attention learns MQAR.** Full attention at d = 64 reaches R ≥ 0.95 at N = 256 for every K ≤ 64 (Zoology Claim 1). If it does not, the budget is too small; raise it before anything else.
2. **Recall falls with K for fixed-state models at small state** (revised, decision B10, 2026-10-16). At d = 32 and N = 512, a model counts on a seed only if it learns the low-K control: K = 8 accuracy of at least 0.25, twice the 0.125 reached by guessing one of the 8 stored values. On learned seeds, linear attention, Mamba-2 and Gated DeltaNet each lose at least 0.2 accuracy between K = 8 and K = 64, and full attention (which must learn the same control) loses at most 0.05. A learned seed that misses its drop fails the criterion. A model that never learns the control on some seed, with no learned seed failing, is `unresolved`: an optimization result that neither passes nor measures a capacity limit. A drop between two near-zero accuracies therefore cannot pass or fail on noise.
3. **Recall rises with state within each fixed-state family.** Over the swept sizes, R at (N = 512, K = 64) is non-decreasing in state for every family, allowing one inversion within seed noise (Based, section 3.1).
4. **Window limits recall** (revised, decision B10, 2026-10-16). Two stacked window layers pass information up to layers × w tokens, so the limit is the end-to-end receptive field, 2 × 32 = 64 tokens for the w = 32 model, not w. Per gap bucket, accuracy on queries whose bucket lies wholly beyond the receptive field is at most 0.1 for the window model on every seed; buckets that straddle the field are left out. The better of Mamba-2 and Gated DeltaNet scores above the window on those gaps on every seed, and the comparison is at counted state including convolution buffers (8,192 floats for the window, 9,344 for the d = 64 rivals): a rival must hold at most 1.25 times the window's state.
5. **Ordering among fixed-state families.** At matched state, Gated DeltaNet and Mamba-2 sit at or above plain linear attention (Based ranks Mamba first among prior architectures; the Gated DeltaNet paper reports gains over Mamba-2 and DeltaNet on retrieval). A reversal larger than the seed spread triggers an implementation review of the reversed family before calibration can pass. It does not by itself fail calibration, because small-scale orderings may differ from the papers.
6. **Copy separates the families** (revised, decision B10, 2026-10-16), in two parts judged separately. *Empirical copy separation:* the transformer's per-string copy accuracy at 2 × L_train exceeds every fixed-state model's on every seed, and every fixed-state model passes prefix-key lookup at ≥ 0.9 (the learned positive control). *Finite-state information bound:* the report counts the payload bits L · log2(D) and the counted FP32 state bits for each fixed-state model (Theorem 2.7). A deterministic model whose state holds S bits cannot copy a string of P > S payload bits with probability above 2^(S − P), so the part is tested only at copy lengths where that cap is at most 0.05, where measured string accuracy must stay under the cap plus binomial slack. At D = 16 and L = 256 the payload is 1,024 bits against 303,104 / 299,008 / 299,008 state bits for LA / GDN / Mamba-2 at d = 64, and payload first exceeds state at L = 75,776 / 74,752 / 74,752, far outside the grid. The acceptance slice adds a `copy_long` task (training identical to `copy`, test length L = 76,000), where the cap is at most 0.05 for every fixed-state model (896-bit margin), so the bound part is tested there; missing long-copy runs give `incomplete`, and a config without a long-copy task still reports the part as `info` (decision B10t, 2026-10-16). A short-copy failure is never read as a test of the theorem, and the empirical part must pass.
7. **Cost ordering.** State floats are flat in N for fixed-state models and linear in N for attention; analytic FLOPs per token at N = 4,096 order attention above the fixed-state families; the measured time slope agrees in direction.
8. **Fingerprint separation.** On the calibrated bench, linear attention and the transformer are no longer flagged `behaves_like` each other, because MQAR at large K and copy beyond L_train separate them.
9. **Rank stability** (revised, decision B10, 2026-10-16), in two parts judged separately. *Meaningful order:* for any two models whose mean R differs by more than 0.1 at a grid point, every seed keeps that order unless the two models are tied on that seed. A tie is predeclared: a difference of at most 0.01 (evaluation resolution), or both models at or above 0.95 (the level criterion 1 calls learned, where saturation hides the order). A tied seed neither supports nor reverses the order; a reversal beyond the tie margin fails the criterion whatever its size. *Repeatability:* a cell whose seeds spread by more than 0.5 on some eval set has unstable onset, so three seed labels do not show that its outcome reproduces. Such a cell needs a same-seed repeat attempt (a model that differs from it only by an `attempt` label, run at the same learning rate), and the repeat must stay within 0.1 of the original on every eval set. A missing repeat leaves the criterion `incomplete`, a disagreeing repeat fails it. Repeat attempts are not ranked as separate models, and seed and attempt spread stay visible in the report.

Each criterion becomes a test in `hybrid/tests/bench/test_calibration.py` that reads a stored calibration result, so a later change to tasks, runner or baselines (which changes the task config hash and empties the cache) must pass calibration again.

### B.6 Feasibility on this machine

The host is an M4 Max with 16 cores (12 performance, 4 efficiency) and 128 GB; the Podman VM has 8 CPUs and 24 GB (`sysctl`, `podman machine inspect`). The image has CPU torch 2.14.1, NumPy and sympy, and no network (`hybrid/executor/requirements.txt`).

Measured rate to extrapolate from: the `default.toml` transformer baseline job (d 64, 2 layers, 101,644 parameters) took 344 s in a four-job run (`hybrid/bench/STATUS.md`) for about 5,500 optimizer steps at batch 32 on sequences of 48 to 128 tokens, plus prediction and the scaling probe. That is at most 0.063 s per step, on the order of 40,000 training tokens per second on one thread.

Extrapolation, to be replaced by the Phase 1 microbenchmark:

- One MQAR run at N = 256, batch 32 (8,192 tokens per step), d = 64, 6,000 steps: about 0.2 s per step, so about 20 minutes on one thread. d = 128 roughly triples the per-step cost.
- Phase 3 grid: 8 families × 3 widths × 3 learning rates at one seed, then 3 seeds at the chosen rate, at N ∈ {64, 256, 512}: on the order of 300 runs. With 8 single-thread jobs in parallel, that is on the order of a day of container time, and less if the VM is given more of the host's 16 cores (decision 2).
- Recurrences written as Python loops over tokens run slower than this rate. The chunked forms in C.3 keep most work in matrix products; Phase 1 measures each family's step rate before the grid is sized.

---

## C. Baselines

### C.1 What exists

From `hybrid/bench/baselines/bb_neural.py` and `bb_ngram.py`:

| Name | What it is | Gap for this target |
| --- | --- | --- |
| `ngram` | Witten-Bell 4-gram | reference only |
| `gru` | GRU, hidden 128 | reference only |
| `transformer` | d 64, 2 layers, 4 heads, FFN 256, sinusoidal positions, LayerNorm, GELU, causal SDPA | older recipe; no RoPE, RMSNorm or SwiGLU |
| `ssm` | 2 LRU-style diagonal complex layers, time-invariant, FFT convolution | not selective; Mamba-class models make A, B, C input-dependent |
| `linear_attention` | elu+1 feature map, no decay, no delta rule, chunked | no gating, no short convolution |

Missing: Gated DeltaNet or DeltaNet, Mamba-2 or another selective SSM, sliding-window attention, and any hybrid.

### C.2 Verified references for the new baselines

- **Gated DeltaNet**: S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ (Eq. 10), with a chunkwise parallel algorithm extending DeltaNet's WY form; the paper also builds hybrids with sliding-window attention or Mamba-2 layers ([arXiv 2412.06464](https://arxiv.org/abs/2412.06464)).
- **DeltaNet, parallel form**: the delta rule reparameterized with generalized Householder products and the compact WY representation; DeltaNet beat GLA at equal state at 340M on recall-heavy tasks and fell behind at 1.3B because of state-size scalability; sliding-window and global-attention hybrids beat Transformer++ ([arXiv 2406.06484](https://arxiv.org/html/2406.06484v5)).
- **Mamba-2 / SSD**: the state space dual layer, scalar-times-identity A, with a block-decomposition algorithm whose "complete implementation" is given "in a few lines of code" in section 6.2 ([arXiv 2405.21060](https://arxiv.org/html/2405.21060v1), [PMLR](https://proceedings.mlr.press/v235/dao24a.html)).
- **Sliding window plus linear attention (Based)**: [arXiv 2402.18668](https://arxiv.org/abs/2402.18668).
- **Modern transformer recipe**: RoPE ([arXiv 2104.09864](https://arxiv.org/abs/2104.09864)) and SwiGLU ([arXiv 2002.05202](https://arxiv.org/abs/2002.05202)); both are in the local index.

### C.3 CPU feasibility in the arm64 container

The reference implementations do not run in the container: `flash-linear-attention` and the Gated DeltaNet code use Triton GPU kernels, `mamba_ssm` needs CUDA, and the container has no network to install anything. So each new baseline is written in pure PyTorch inside `hybrid/bench/baselines/` (new file `bb_recall.py`, sharing `_TorchCandidate` from `bb_neural.py`), with two forms per layer:

| Baseline | Training form | Reference form for tests | State floats | CPU notes |
| --- | --- | --- | --- | --- |
| Transformer++ | SDPA, RoPE, RMSNorm, SwiGLU | existing `TransformerNet` math | 2 · L · d · N | as fast as the current transformer |
| Sliding-window attention | SDPA with a banded causal mask | full attention when w ≥ N | 2 · L · d · w | the dense masked form costs O(N²) on CPU at these lengths; acceptable to N = 512, and the FLOP count uses the windowed cost |
| DeltaNet / Gated DeltaNet | chunked WY form, chunk 64 | token-by-token recurrence of Eq. 10 | heads · d_k · d_v per layer | chunk-internal triangular solve via `torch.linalg.solve_triangular` |
| Mamba-2 (SSD) | chunked SSD as in section 6.2 | naive recurrence h_t = a_t h_{t−1} + B_t x_t | d_inner · d_state per layer | pure tensor ops; no custom kernel |
| Hybrid A | Gated DeltaNet layers with one global attention layer (3:1 at larger depth, matching the ratio in the brief's Qwen example) | composition of the above | sum | |
| Hybrid B | linear attention + sliding window per layer | composition | sum | |

All run on one CPU thread like the existing baselines. The host registry entry gains a `state_formula` (floats as a function of N) so the frontier plot does not trust any entrant's `describe()` for baselines, and a new `state` class `hybrid` for models whose state has a fixed and a growing part.

### C.4 Tests for the baselines (`hybrid/tests/bench/test_baselines_recall.py`)

- Chunked output equals the reference recurrence to 1e-5 (float64) for Gated DeltaNet, DeltaNet and SSD, at lengths that are and are not multiples of the chunk.
- Setting α_t = 1 in Gated DeltaNet gives DeltaNet; setting β_t = 0 leaves the state unchanged.
- Sliding-window attention with w ≥ N equals full attention.
- `api.check_causality` passes for every baseline.
- The registry's state formula equals the size of the state tensor the reference form carries.
- Parameter counts are within 10% across families at each width (reported, so curves are compared at known sizes).
- One fit step on random data gives finite gradients.

---

## D. Demonstration and query protocol

### D.1 Problem

Hand-built mechanisms (the two run-4 cards that reached S6) define their behavior on demonstrations: "write this pair", "apply this rule". The bench gives them a flat token stream and asks them to learn segmentation, the task format and the mechanism at once in a few hundred steps. Both failed to train on their own falsifier tasks, and 11 of 16 and 8 of 11 claims stayed untested (`MORNING-REPORT.md`, section 1).

### D.2 Contract

One episode generator feeds two views, so token models and event models are scored on the same episodes and answers.

```python
# hybrid/bench/episodes.py
@dataclass(frozen=True)
class Episode:
    demos: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]   # (key tokens, value tokens), in order
    distractors: tuple[tuple[int, ...], ...]                     # filler spans between demos, may be empty
    queries: tuple[tuple[int, ...], ...]                          # query key tokens
    answers: tuple[tuple[int, ...], ...]                          # oracle value tokens, host and evaluator only

def to_tokens(ep: Episode, fmt: TokenFormat) -> tuple[list[int], list[int]]   # sequence, loss_positions
def to_events(ep: Episode) -> list[Event]                                     # without answers

# hybrid/bench/api.py
class EventCandidate(Protocol):
    def __init__(self, vocab_size: int, seed: int, budget: dict) -> None: ...
    def fit(self, episodes: list[list[Event]], answers: list[list[tuple[int, ...]]]) -> None: ...
    def reset(self) -> None: ...                                   # empty memory for a new episode
    def write(self, key: tuple[int, ...], value: tuple[int, ...]) -> None: ...
    def skip(self, tokens: tuple[int, ...]) -> None: ...           # distractor span
    def read(self, key: tuple[int, ...]) -> np.ndarray: ...        # (vocab,) distribution for the first answer token
    def describe(self) -> dict: ...                                 # params, state_floats(K), hyper
```

Rules: reads come after the writes of the same episode; `read` must not change the memory (checked by repeating a read); the episode order of events is the only order the candidate sees.

The protocol covers MQAR (demos are pairs), copy (demos are positions and tokens; query is a position), n-gram lookup, and the existing few-shot rule task (demos are (x, y) examples, query is a new x). Falsifier designers must express card protocols as episodes when the card's mechanism is defined on demonstrations; `prompts/falsifier_designer.md` and `falsifier.py` gain that option.

### D.3 Same-task positive controls

Every event-mode task runs these controls, built by the bench, at the same state budgets as the candidate:

| Control | Role |
| --- | --- |
| exact dictionary (state grows with K) | ceiling: 1.0 when answers are in context |
| capacity-M exact cache (LRU and random eviction) | fixed-state exact memory |
| outer-product memory with d × d state (linear attention written directly) | known fixed-state curve |
| delta-rule and least-squares memory with d × d state | the closed-form references from the recall investigation (`tmp/falsify-check/pair_memory_control.py`: least squares at d = 64 scored 1.00 at K ≤ 48 and 0.70 at K = 256; linear attention at d = 64 scored 0.54 at K = 256) |
| chance and unigram | floors |

A task where no positive control reaches its predicted level is invalid and decides nothing. A candidate's event-mode curve is compared with the control curves at matched state, which is the frontier for event-mode mechanisms.

Event-mode results answer "does the mechanism do what the card says once segmentation is given". They do not count toward a win under A.4, because segmentation is handed over. A card that does well in event mode must then be built in token mode (it learns or implements the segmentation from tokens) and pass the token bench.

### D.4 Measured state

The fingerprint's one self-reported number is `state_floats_per_sequence` (`hybrid/bench/README.md`, trust model). In event mode the runner measures it: after K writes for K ∈ {8, 64, 512} it counts the floats held in the candidate's NumPy and torch attributes (S4 already does this for parameters), and the evaluator compares growth with the card's `recall_strategy.state_growth` (E.3). A mechanism that claims a fixed state but grows is refuted on that claim.

### D.5 Tests (`hybrid/tests/bench/test_episodes.py`)

- The token view and the event view of an episode give the same oracle answers.
- `to_tokens` loss positions are exactly the answer positions.
- Exact dictionary scores 1.0; the d × d least-squares control reproduces the recall-investigation numbers within binomial error (regression fixture).
- A candidate that reads during `write` or mutates on `read` is caught.
- Measured state growth flags a candidate that appends to a list while describing a fixed state.

---

## E. Gate changes

### E.1 S6 fails on a validated falsifier

Today `s6_verdict.run` records `falsified_by` and puts "card falsified by ..." first in the reasons, then calls `ctx.verdict(card_id, "S6", "pass", ...)` unconditionally. Change: the verdict is `fail` when `ledger["falsified_by"]` is non-empty, and no mutation is queued for a falsified card. A falsifier counts as validated only when its task passed harness validation, the bench was calibrated for the config used, the candidate trained, and the rule's `control_rule` was met. Those gates already exist in `falsifier_eval.decide` (`trained` against the better trivial reference, `control_rule`, `task_blocked` on `hyper` and `train_max_len`), so S6 only needs to act on the outcome.

Tests (`hybrid/tests/research/test_s6_verdict.py`): falsified card gets `fail` and no `mutation_requested`; an inconclusive falsifier keeps `pass`; a falsifier decided by the analyst alone never fails the card.

### E.2 S3 cannot pass with required obligations unchecked

Today S3 passes when both judges vote feasible, and only records how many judge entries were not discharged and whether every checkable obligation was discharged (`all_checkable_discharged` goes to the `s3_gate` artifact and gates nothing). Change:

- The formalizer schema (`contracts.FORMALIZER`) marks each obligation `required: bool` and `checkable: bool`. Required: the complexity obligation, one capacity obligation per recall or copy claim, and the state-growth obligation from E.3.
- New rule `required_checks` before the judge rules: if any required checkable obligation is not `discharged` by an executed and replayed check, the verdict is `undetermined` with the obligation names, whatever the votes. Config: `s3.require_checked_obligations = true`.
- New check kind `capacity_reference`: the obligation checker runs the card's own mechanism, written directly in NumPy in event mode, on MQAR episodes at the card's stated state size, and compares with the card's predicted curve. This is the recall investigation's "reference-mechanism capacity control" moved into S3, and it gives S3 a measured result where it now has a judge's opinion.
- New check kind `information_bound`: from the card's state size in bits and its recall or copy claim, compute whether the claim exceeds the Theorem 2.7 bound; a claim above the bound is a violation.

Tests (`hybrid/tests/research/test_s3_feasibility.py`): two feasible votes with an open required obligation give `undetermined`; a violated `information_bound` gives `infeasible_proved`; a card with every required obligation discharged and two feasible votes passes.

### E.3 Every card states its recall strategy

Theory card schema gains a required block:

```json
"recall_strategy": {
  "beyond_fixed_state": "how the mechanism answers when more items were shown than a fixed state can hold",
  "state_growth": "fixed | O(log N) | O(N/c) | O(N) | other: <formula>",
  "trade": "what it pays: compute per token, retrieval latency, lossy recall, precision, training cost",
  "target_point": {"task": "mqar", "N": 512, "K": 64, "state_floats": 16384, "accuracy": 0.9}
}
```

- S1 (`stages/s1_wellformed.py`, `prompts/salad_critic.md`): missing block, or a fixed state with a claim of exact recall for unbounded K, is a fatal finding. The `information_bound` arithmetic runs on the host at S1 so the critic does not do it by hand.
- S2 (`stages/s2_prior_art.py`, `prompts/reduction_critic.md`): the critic checks the strategy against the family list in E.4 and records which known strategy it matches (KV cache, window, compression, retrieval, gated forgetting, delta replacement, test-time optimization).

### E.4 Reduction critic families

`prompts/reduction_critic.md` and the `reduction_critic` schema enum gain these families, each with a signature equation in `prompts/explorer.md` "Commonly re-invented families". All IDs below were found in the local arXiv index:

| Family | Representative papers (arXiv) |
| --- | --- |
| gated_delta_rule | DeltaNet parallel form 2406.06484; Gated DeltaNet 2412.06464; RWKV-7 2503.14456 |
| selective_ssm_or_gated_linear_attention | Mamba 2312.00752; Mamba-2 / SSD 2405.21060; GLA 2312.06635; Griffin 2402.19427; GateLoop 2311.01927 |
| hybrid_interleaved | Jamba 2403.19887; Samba 2406.07522; Hymba 2411.13676; Griffin 2402.19427; "A Systematic Analysis of Hybrid Linear Attention" 2507.06457 |
| sliding_window_or_local_plus_global | Based 2402.18668 |
| sparse_attention | Native Sparse Attention 2502.11089; MoBA 2502.13189 |
| retrieval_or_knn_memory | Memorizing Transformers 2203.08913; RETRO 2112.04426 |
| landmark_or_compressive_memory | Landmark Attention 2305.16300; Compressive Transformer 1911.05507; Infini-attention 2404.07143 |
| test_time_memory_optimization | TTT 2407.04620; Titans 2501.00663; ATLAS 2505.23735; MesaNet 2506.05233 |
| hierarchical_state_growth | Log-Linear Attention 2506.04761 |
| prompt_repetition | Just Read Twice 2407.05483 |

The existing `partitioned_or_windowed_attention` and `transformer_variant` stay. A reduction to one of these families with a confirmed numeric check gives `reduces_to_known` for a `new_mechanism` card, and is the expected outcome for a `known_family_improvement` card (F.3).

### E.5 Less reliance on model judges

Problem 3 in the brief. The changes above move three decisions from judges to executed checks: capacity (E.2 `capacity_reference`), information bound (E.2, E.3), and frontier position (F.2, checked at S5 against the frontier registry). Novelty keeps the literature path but gains a measured component: a card whose `target_point` lies inside the measured envelope of a known family is not novel for this target, whatever the literature says. Judges remain for the parts no check covers (learnability arguments), and their votes cannot pass a card with an open required check.

---

## F. Generator changes

Across runs 1 to 5, S2 assessed 196 cards: 21 passed, 122 reduced to known work, 51 failed and 2 were undecided (`reports/2026-10-11-bigbrane/FACTS.md`). The lenses come from physics, biology and mathematics, and the seed list holds one group of modern sequence architectures, so explorers rediscover classical algorithms and 2020 to 2023 recurrences while the 2024 to 2026 recall literature stays out of view. The changes below put that literature and its measured curves in front of the explorers.

### F.1 Seed the lenses and the graveyard with the recall frontier

- `hybrid/config/lenses.toml`: add an explorable lens `recall_memory_frontier` (description: the information bound, the measured curves, and the open question of state that grows sublinearly while keeping exact recall), and pairings of it with existing lenses that have produced S3 cards (`neural_odes_koopman`, `sparse_distributed_memory`, `holographic_reduced_representations`). Add the frontier families to `known_llm_alternatives_to_avoid_reinventing`.
- `hybrid/literature/seed_reading_list.toml`: add a `recall_frontier` group (section H.2). Today its only modern-architecture entries are Linear Transformers, Mamba, RWKV, RetNet, Hyena, xLSTM, BitNet, LCM and TTT.
- `hybrid/research/graveyard.py`: a seed graveyard built from the frontier registry, one line per known family with its signature equation and its **measured** point from Phase 3 ("Gated DeltaNet d 64: R = x at N 512, K 64, state y floats"), so explorers see where each family sits as numbers.

### F.2 Explorers target a point on the tradeoff

Replace "The target" in `prompts/explorer.md` with the A.1 question and require `recall_strategy.target_point`. The orchestrator assigns each explore task a regime (small state at long N; attention-level recall at a fraction of KV-cache state; fewer training examples to reach R = 0.9) and passes the measured envelope for that regime, so explorers aim at a gap in a known curve. The escape operator (`stages/escape.py`, `prompts/explorer_escape.md`) adds the parent's measured or predicted point and asks the child to move it beyond the envelope.

### F.3 Known-mechanism improvements as a labelled card type

The explorer prompt now bans every attention variant, windowing and partitioning, and any renamed SSM or linear attention. That ban excludes hybrids, which hold the production frontier. Add `card_type`:

- `new_mechanism`: current rules; a confirmed reduction ends it at S2.
- `known_family_improvement`: names its base family (from E.4) and the exact change; S2 checks that the change is real (the reduction check with the change removed must match the base family and with it must not), sets `radical_enough` aside, and records the card as an improvement. It is judged on the bench, and a success is reported as a known-family win.

The inspect report counts the two types separately, and the creativity block reports transplant shares per type.

### F.4 Tests

`hybrid/tests/research/test_prompts.py`: the explorer prompt contains the target text and the `card_type` and `recall_strategy` fields; schema tests for both fields; S2 path tests for `known_family_improvement` with a real change and with a zero change; graveyard test that seed entries carry measured numbers and a source run.

---

## G. Small real-text check

### G.1 Data

- **TinyStories** (Eldan and Li; [arXiv 2305.07759](https://arxiv.org/abs/2305.07759)): models below 10 million parameters produce fluent stories; the released models used a GPT-Neo tokenizer cut to the 10,000 most common tokens and a context of 512. First choice, because a few-million-parameter model learns something measurable on it.
- **FineWeb-Edu** ([arXiv 2406.17557](https://arxiv.org/abs/2406.17557), "The FineWeb Datasets"): second check on web text, on a small sample.

The container has no network and no tokenizer library. Data is downloaded and tokenized on the host into fixed shards of uint16 token IDs under `data/lm/` (gitignored path, read-only volume), with a manifest of sha256 hashes. Tokenizer choice is decision 3: a byte-level vocabulary (V = 256, no dependency, about 4 times more tokens per story) or a BPE trained on the host and frozen.

### G.2 Models and protocol

- Families: Transformer++, Gated DeltaNet, Mamba-2, Hybrid A, and the best candidate if one exists. Two sizes: about 2M and about 5M non-embedding parameters (with V = 256, d = 256 and 4 to 6 layers lands in that range).
- Context 512 for training; evaluation at 512 and 2,048 to see length behavior.
- Judge: validation loss against analytic training FLOPs, at matched parameters and at matched FLOPs; plus two recall probes on the trained LMs, adapted from the papers: Zoology's natural-text probe (key-value pairs inserted into tokenized text, with a perplexity gap measured on the second occurrence of each key; [blog](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)) and the "Repeat After Me" text copy probe (two copies of a string plus the first word of a third).
- Trust: the LM evaluator loads the saved weights and the candidate code in a separate container job and computes loss itself. Prediction arrays of shape (tokens, V) are too large to stage at this size, and loss computed by the candidate's own process is not trusted.

### G.3 Feasibility on this machine

Extrapolated from the measured rate in B.6 (about 40,000 tokens per second on one thread for a 101,644-parameter model, which implies roughly 2 × 10¹⁰ training FLOPs per second per thread including overhead):

| Model | Training FLOPs per token (6 × params) | One thread | 8 threads (assumes 5× speedup, to be measured) | 40M tokens | 100M tokens |
| --- | --- | --- | --- | --- | --- |
| 2M params | 1.2 × 10⁷ | about 2,000 tokens/s | about 10,000 tokens/s | about 1.1 h | about 2.8 h |
| 5M params | 3 × 10⁷ | about 800 tokens/s | about 4,000 tokens/s | about 2.8 h | about 7 h |

These are process runtimes per model run, from one measured job and an assumed thread scaling; larger matrix products run more efficiently than the tiny baseline did, so the figures are more likely high than low. Attention at context 512 adds its quadratic term. With 4 families × 2 sizes × 2 seeds, the 2M-parameter, 40M-token setting fits the container; the 5M-parameter, 100M-token setting needs either sequential runs over several days of container time or more CPUs (decision 2). The host GPU (MPS) is outside the container and outside the trust model; using it for training with scoring in the container is decision 2b.

### G.4 Tests (`hybrid/tests/bench/test_lm.py`)

Tokenizer round trip; train and validation shards share no document; the evaluator's loss equals a hand computation on a two-layer model with fixed weights; the FLOP counter matches the analytic formula per family; a model that reads the next token (an acausal shift) is caught by the causality check.

---

## H. Literature and prior-art coverage

### H.1 State

- Local arXiv index: complete for the 15 harvested sets (`MORNING-REPORT.md`, section 4). Every frontier paper named in this plan is present, including MQAR, Based, Repeat After Me, Gated DeltaNet, DeltaNet, Mamba, Mamba-2, GLA, RWKV-7, Griffin, Samba, Jamba, Hymba, Native Sparse Attention, MoBA, Landmark Attention, Compressive Transformer, Infini-attention, Memorizing Transformers, RETRO, TTT, Titans, ATLAS, MesaNet, Log-Linear Attention, Just Read Twice and RULER (lookup by ID on 2026-10-11).
- OpenAlex now has a key (`BIGBRANE_OPENALEX_API_KEY` in `hybrid/literature/keys.py`). The API takes `api_key` as a query parameter, and the `cites:` filter returns works that cite a given work ([OpenAlex docs](https://developers.openalex.org/api-reference/introduction), [filter docs](https://github.com/ourresearch/openalex-docs/blob/main/api-entities/works/filter-works.md)).
- Semantic Scholar: no key, and in run 4 it skipped 461 searches in cooldown. Plan without it: remove it from the S2 required path and keep it as an opportunistic source whose failures are not counted as empty searches.

### H.2 Additions

1. **Seed group `recall_frontier`** in `seed_reading_list.toml`, from the papers in E.4 plus Zoology, Based, Repeat After Me, "Can Mamba Learn How to Learn?" (2402.04248) and RULER (2404.06654). Each entry carries its signature equation for the graveyard.
2. **Citation snapshot.** A host job (`hybrid/literature/frontier_snapshot.py`) uses OpenAlex `cites:` on the seed group to pull works citing them, filters to the last 24 months and to the cs.LG, cs.CL and stat.ML arXiv sets through the local index, and stores the IDs and abstracts as a local table that S2 searches before live sources. Refreshed at run start when older than a set age, with a receipt.
3. **Frontier registry as prior art.** `results/frontier/registry.json` holds the measured curves from Phase 3 for every family, with the bench config hash. S2 and S5 compare a card's `target_point` with it, so a card that predicts a point a known family already reaches is caught by a measurement as well as by a paper.
4. **Grounding stays.** Evidence must match retrieved IDs, as now.

### H.3 Tests

`hybrid/tests/literature/test_frontier_snapshot.py`: the snapshot is built from recorded OpenAlex responses (fixtures, no network); S2 finds Gated DeltaNet for a card that renames the gated delta rule; a Semantic Scholar failure does not make a search count as empty.

---

## I. Order of work and checkpoints

Research runs pause until Phase 3 exits (decision 7). Phase 6 (literature) does not depend on the bench and runs alongside Phases 1 to 3. Gate and generator code (Phase 5) may be written during Phases 3 and 4, but is switched on only after Phase 3 exits.

| Phase | Work | Files likely touched | Tests | Exit criterion |
| --- | --- | --- | --- | --- |
| 0. Target | Andrew confirms or edits A (question, metrics, thresholds) | `IMPROVEMENT-PLAN.md`, `RESEARCH-SYSTEM-SPEC.md` goal section | none | written confirmation of A.1 and A.4 thresholds |
| 1. Tasks and harness | `mqar`, `copy`, `ngram_lookup` tasks; `loss_positions` in `TaskData`, runner and baselines; `cost.py`; `frontier.py` (envelope, distances, Theorem 2.7 bound); `configs/frontier.toml`; throughput microbenchmark per family and thread count | `hybrid/bench/tasks/{mqar,copy,ngram_lookup,common,__init__}.py`, `runner.py`, `api.py`, `candidate_template.py`, `baselines/bb_neural.py`, `cost.py`, `frontier.py`, `configs/frontier.toml` | generator determinism and oracle self-score; MQAR invariants (distinct keys, queries from stored keys, disjoint key and value halves, power-law gap sampler); loss positions equal answer positions; copy structure; bound calculator; envelope and distance on hand-made curves | tests pass in the container; measured step rates recorded in `hybrid/bench/STATUS.md` and the Phase 3 grid sized from them |
| 2. Baselines | Transformer++, sliding window, DeltaNet, Gated DeltaNet, Mamba-2 SSD, Hybrids A and B, short convolutions; registry state formulas and `hybrid` state class | `hybrid/bench/baselines/{bb_recall,__init__}.py`, `fingerprint.py` | C.4 list | every baseline passes C.4 in the container |
| 3. Calibration | the B.3 grid with the B.4 protocol; frontier registry; BASELINES.md rewritten from the new results | `results/calibration-<date>/`, `results/frontier/registry.json`, `hybrid/bench/BASELINES.md`, `hybrid/tests/bench/test_calibration.py` | the nine B.5 criteria as tests over the stored result | all nine B.5 criteria hold on 3 seeds; if one fails, fix budget or implementation and rerun; Phases 4 to 8 do not start their runs until this exits |
| 4. Demonstration protocol | `episodes.py`, `EventCandidate`, event runner, controls, measured state | `hybrid/bench/{episodes,api,runner,evaluator,falsifier,falsifier_eval}.py`, `prompts/{builder,falsifier_designer}.md`, `stages/s4_construct.py`, `stages/s5_evaluate.py` | D.5 list | controls reproduce their predicted curves; the two run-4 cards (card-03fa1d38, card-779795a5) rebuilt in event mode get decided statuses (supported, refuted or inconclusive with a reason) on their capacity claims as a plumbing check, under the operator override |
| 5. Gates and generator | E.1 to E.5 and F.1 to F.3 | `stages/{s1_wellformed,s2_prior_art,s3_feasibility,obligation_checks,s6_verdict,contracts,escape,s0_explore}.py`, `graveyard.py`, schemas, `prompts/{explorer,explorer_escape,salad_critic,reduction_critic,formalizer,obligation_checker}.md`, `config/{lenses,run}.toml` | tests in E.1, E.2, F.4; an offline replay of the run 3 to 5 ledgers through the new S3 and S6 rules, listing every decision that changes | tests pass; replay report written; a live run of at most 20 cards completes with every S1-passed card carrying `recall_strategy` and `card_type` |
| 6. Literature | seed group, citation snapshot, frontier registry hook in S2, Semantic Scholar demoted | `hybrid/literature/{seed_reading_list.toml,frontier_snapshot.py,prior_art.py,client.py}` | H.3 list | snapshot built with a receipt; S2 retrieves the right family paper for each renamed-family fixture card |
| 7. Real-text check | data prep on host, LM runner and evaluator, baseline curves | `hybrid/bench/lm/`, `configs/lm_tiny.toml`, `data/lm/` | G.4 list | loss-against-FLOPs curves for the four baseline families at the 2M size on 2 seeds; their ordering recorded; a candidate can be added with one config line |
| 8. Research runs | runs under the new gates, S5 on the calibrated frontier config, S6 classification per A.4 | `hybrid/config/run6.toml` | none new | each card that reaches S5 is classified as win, null, refutation or no result, with its point on the frontier plot |

`hybrid/config/bench_quick.toml` stays as a smoke test for plumbing only; S5 decisions use `configs/frontier.toml`.

---

## J. Risks and open decisions

### J.1 Risks

- **CPU budget may be too small for attention to learn MQAR at these sizes.** Then calibration fails at criterion 1. Response: smaller V, fewer grid points, more steps per run, or more VM CPUs, with the acceptance thresholds left as set.
- **Reimplementation errors read as architecture differences.** Pure-PyTorch versions of Gated DeltaNet and SSD may differ from the reference kernels in initialization, normalization or gating. Response: chunked-equals-recurrent tests, the B.5 criterion 5 review, and parameter and state counts reported beside every point.
- **Small-scale orderings may not transfer.** DeltaNet beat GLA at 340M and fell behind at 1.3B on recall-heavy tasks because of state-size scaling (2406.06484). A frontier gain at a few million parameters justifies a larger test (A.1) and does not establish a result at scale.
- **MQAR is simple.** The Zoology README calls it "useful but simplistic". A search tuned to MQAR can find tricks that do not help language. Response: copy, lookup, few-shot rules and the real-text check in every evaluation.
- **Timing noise.** Wall slopes move between runs on the shared VM, so state floats and analytic FLOPs are the primary cost measures.
- **The frontier moves.** Several 2025 and 2026 papers (MesaNet, Log-Linear Attention, the hybrid analysis 2507.06457) already target this gap. The citation snapshot and a dated registry keep the bar current, and a card that matches a recent paper is a rediscovery even when it beats the 2024 baselines.
- **Judges remain in S3 for learnability.** E.2 limits what they can pass; it does not remove them.
- **Brief facts not verified.** The Qwen3.8-Flash-Next layer counts come from the brief.

### J.2 Decisions for Andrew

1. **Target and thresholds.** Confirm A.1 as the question, and set the A.4 numbers (horizontal ratio 2, vertical gain 0.10, 2 of 3 settings, 3 seeds), or replace them.
2. **Compute.** (a) Give the Podman VM more of the host's 16 cores and 128 GB (it has 8 CPUs and 24 GB now)? (b) Allow untrusted host-MPS training for calibration sweeps or the LM check, with scoring kept in the container?
3. **Real-text data.** TinyStories first, then a FineWeb-Edu sample? Byte-level vocabulary or a host-trained BPE?
4. **Answer-only loss.** Change the candidate API so every entrant receives `loss_positions`, as Zoology trains? This changes the bench contract and invalidates every cached baseline.
5. **Known-family improvements.** Do they count as success for the project, or only as stepping stones toward a new mechanism?
6. **S3 judges.** Keep two judges for learnability, or move to one judge plus the new required checks?
7. **Pause runs.** Hold all research runs until Phase 3 exits (recommended), or keep running the loop on the old bench for generator data only, with S5 results marked uncalibrated?
8. **Explorer pacing.** The GPT route runs at 1 concurrent request, 10 s spacing and 60 per hour with effort capped at medium (project memory). With a narrower target, should both explorers stay on, or should the GPT route move to judge B and the falsifier designer?
