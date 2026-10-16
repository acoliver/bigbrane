# hybrid/bench: trusted evaluation bench

The bench compares a candidate model with five baselines on tasks whose next-token distribution or answer is known exactly. It measures accuracy against the oracle, sample efficiency across train sizes, and how predict time and memory grow with context length. A mechanism fingerprint then asks what the candidate is really doing: whether its recall capacity, scaling, length generalization, few-shot gain and predictive distributions set it apart from a GRU, a diagonal SSM, a linear-attention model, a transformer or an n-gram, or whether it is one of those under another name. Spec: [RESEARCH-SYSTEM-SPEC.md](../../RESEARCH-SYSTEM-SPEC.md) section 4. Current state and measured numbers: [STATUS.md](STATUS.md), [BASELINES.md](BASELINES.md).

## Using it

### Python API (stable interface)

```python
from hybrid.bench.evaluate import evaluate_candidate
from hybrid.executor.executor import Executor

result = evaluate_candidate(
    candidate_dir,            # directory with candidate.py, or None to run only the baselines
    bench_config,             # path to a TOML file (hybrid/bench/configs/*.toml) or a parsed dict
    run_dir,                  # e.g. runs/<run-id>/cards/card-1234abcd; <run_dir>/bench must not exist yet
    executor,                 # hybrid.executor.executor.Executor (trusted) or hybrid.bench.local.LocalExecutor (trusted: false)
    *, claimed_exponent=None, # the theory card's time-complexity exponent for the scaling verdict (0 = constant, 1 = linear, 2 = quadratic)
    baselines=None,           # subset of ["ngram", "gru", "transformer", "ssm", "linear_attention"]; None uses the config's list; [] runs none
    cache_dir=None,           # baseline cache root; default runs/_bench_cache
    log=<stderr logger>)      # callable(str) for progress lines; pass lambda _: None to silence
-> dict                       # the bench_result, also written to <run_dir>/bench/bench_result.json
```

It raises `FileExistsError` if `<run_dir>/bench` exists, `ValueError` for an unknown baseline, an invalid candidate directory (no `candidate.py`, symlinks or special files, over 16 MiB) or an invalid config, `TypeError` for any other executor type, and `RuntimeError` if the evaluator job fails. A candidate whose job fails (error, timeout, OOM) does not raise: its entry has `status` `job_failed`, no metrics, and `job_status` and `job_detail` from the receipt. A failed baseline job is treated the same way and is not cached.

The returned dict has this shape (schema `bigbrane.bench_result/1`):

| Key | Content |
| --- | --- |
| `schema`, `trusted`, `task_config_hash`, `config_hash` | `trusted` is false whenever a `LocalExecutor` was used |
| `tasks[task]` | `train_sizes`, `train_tokens`, `oracle_kind`, `group_name`, `group_labels`, `n_prefixes`, `target`, `vocab_size`, `length_probes` (for hmm, vom and compose: factor to `length`, `min_prefix`, `n_prefixes`) |
| `entrants[name]` | `name` is `candidate` or a baseline; `role`, `status` (`ok`, `job_failed`, or `invalid_describe: <reason>`), `job_status`, `job_detail`, `job_id`, `claimed_exponent`, `describe` (per task and train size), `runner_reported` (fit and predict seconds, platform), `tasks[task].sizes[n]` with `kl`, `acc`, `valid`, `n`, `train_tokens`, `groups[label]` (and `close_acc` for compose), `tasks[task].tokens_to_target`, `tasks[task].length_probes[factor]` (`kl`, `acc`, `valid`, `n`, `length`), `scaling` (lengths, median seconds, peak RSS deltas, `time_fit_tail`, `time_fit_all`, `slope_minus_claim`, `verdict`) and `fingerprint` (see [Mechanism fingerprint](#mechanism-fingerprint)) |
| `fingerprint` | `config` (every threshold used) and, when there is a candidate, `candidate.nearest_baseline` and `candidate.flags` |
| `comparison["candidate"][baseline][task]` | per train size `kl_delta` and `acc_delta` (candidate minus baseline) and `tokens_to_target_ratio` |
| `host` | `trusted`, `mode` (`podman` or `local`), `image`, the resolved `config`, every job `receipts`, `evaluator_receipt`, `timings_s` (`data`, `candidate_jobs`, `evaluator_job`, `total`), `evaluator_output_sha256` |

Beside `bench_result.json`, `<run_dir>/bench/manifest.json` lists the sha256 of every file under `bench/` (job inputs and outputs, oracle files, the result) for the research ledger.

### CLI (stable interface)

```sh
podman machine start podman-machine-default        # if stopped
.venv-hybrid/bin/python -m hybrid.bench eval --candidate <dir> --config hybrid/bench/configs/default.toml \
    --run-dir runs/<run-id> [--claimed-exponent 1.0] [--baselines ngram,gru] [--cache-dir <dir>] \
    [--markdown table.md] [--local]
.venv-hybrid/bin/python -m hybrid.bench baselines --config hybrid/bench/configs/default.toml --run-dir runs/<run-id> \
    [--baselines ...] [--cache-dir <dir>] [--markdown table.md] [--local]
```

The CLI prints the Markdown tables, then a JSON summary (`trusted`, path of `bench_result.json`, `timings_s`) on stdout, and progress lines on stderr. It uses the run directory's name as the executor run id (so the Podman labels read `bigbrane.run=<run dir name>`), which must match `[a-z0-9][a-z0-9_.-]*`; receipts go to `<run-dir>/receipts/`. It exits non-zero if `<run-dir>/bench` exists or if a job cannot be evaluated; the exit code is 0 when the evaluation completed, even if the candidate's own job failed (check `entrants.candidate.status`).

`--local` (or passing `hybrid.bench.local.LocalExecutor`) runs both jobs as host subprocesses with no isolation. It exists for development and tests; its results carry `"trusted": false` in `trusted` and `host.trusted`, and its baseline cache entries live under a separate `local` mode directory.

### Configs

| File | Train sizes | Budget | Probe lengths | Use |
| --- | --- | --- | --- | --- |
| `configs/default.toml` | 32, 128, 512 | 800 steps, 60 epochs | 2^6 to 2^12 | Full evaluation. Measured: 491.6 s for the five baselines on a cold cache (five jobs in parallel). A GRU candidate took 178.9 s with cached baselines before the fingerprint work; not remeasured since. |
| `configs/quick.toml` | 32, 128, 256 | 600 steps, 60 epochs | 2^6 to 2^11 | First pass in the loop: shorter sequences (hmm 32, vom 64, compose 48, recall K in {2, 8, 16}). Measured: 78.7 s for a GRU candidate with fingerprinting and cached baselines; 147.2 s for the five baselines on a cold cache. |
| `configs/tiny.toml` | 8, 32 | 30 steps | 2^6 to 2^9 | Tests only. |

Results from different configs are not comparable: each config has its own task config hash and its own baseline cache. Measured times are in [STATUS.md](STATUS.md) and the baseline numbers in [BASELINES.md](BASELINES.md). A candidate that is slower than a GRU takes longer; the candidate job cap is 1800 s (default) or 900 s (quick), after which the job is classified `timeout` and the entry has no metrics.

## Candidate API

A candidate is a directory with `candidate.py` defining `Candidate(vocab_size, seed, budget)` with `fit(train)`, `predict_proba(prefixes) -> (B, vocab)` and `describe() -> {"params", "state_floats_per_sequence", "notes"}`. The full contract is in [api.py](api.py); [candidate_template.py](candidate_template.py) is the file a builder copies. `api.check_causality(model, prefixes)` checks that each row depends only on its own prefix (alone vs batched, reversed batch, repeated call, extra longer prefix in the batch).

A fresh `Candidate` is built for every (task, train size) and fit on the first n training sequences of that task's pool. `budget` holds `max_steps`, `max_epochs`, `batch_size` and `lr`, the same for every entrant.

## Tasks

Every test prefix comes from its own independently sampled sequence, and the tokens after the scored position are never written anywhere. The candidate therefore cannot read a later prefix to learn an earlier answer.

| Task | Generator | Oracle at the scored position | Groups and extras |
| --- | --- | --- | --- |
| `hmm` | 4-state sticky HMM, 8 tokens, two-token emission support per state (math copied from `bigbrane/core.py`) | exact filtering distribution | |
| `vom` | random context tree over 6 tokens, depth up to 5, every node has a distribution | exact (deepest matching node) | `context_depth` |
| `recall` | K key-value pairs, SEP, then 16 query-answer pairs; K in {2, 8, 32} | the queried key's value | accuracy per K |
| `compose` | typed nested brackets (3 types, depth 6) with filler tokens whose distribution depends on the innermost open type | exact distribution from the parsed stack | `close_acc`: at depth >= 1, is the most likely closing token the matching one |
| `fewshot` | affine rules y = (a x + b) mod 13 shown as `x -> y ;` examples; train and test rules disjoint | answer for an input not among the shown examples | accuracy per number of shots (2, 4, 8) |

Default sequence lengths are 64 (hmm), 128 (vom), at most 97 (recall), 96 (compose) and at most 48 (fewshot). Train sizes are 32, 128 and 512 sequences.

For the length-generalization fingerprint, hmm, vom and compose also get one extra test set per factor in `[fingerprint] length_factors` (default 2 and 4): sequences `factor` times the training length, scored at a position in the upper half (between `factor/2` and `factor` times the training length), so every scored position lies beyond anything a training sequence showed. Each set has its own random stream, shares no sequence with the base test set, and is staged for the candidate as prefixes only (`data/<task>/prefixes_x<factor>.json`); its oracle exists only in the evaluator job (`oracle/<task>__x<factor>.npz`).

## Metrics (computed only in the evaluator job)

- `kl`: KL(oracle || prediction) in nats for distribution tasks, and -log q(answer) for answer tasks (the KL to the one-hot oracle). Predicted probabilities are floored at 1e-12 before the log, without renormalizing.
- `acc`: tie-aware top-1 credit. If the prediction's maximum is shared by m tokens, the credit is (correct tokens among them) / m. For distribution tasks, correct means in the oracle's argmax set.
- `tokens_to_target`: the first train size whose metric meets the task's target in the config, as train tokens, plus a log-linear interpolation between the bracketing sizes.
- Predictions must be finite, non-negative, of shape (prefixes, vocab), with rows summing to 1 within 1e-3; otherwise that file is marked invalid and gets no metrics.
- `comparison`: candidate minus each baseline per task and train size, and the ratio of tokens-to-target.
- `tasks[task].length_probes[factor]`: `kl` and `acc` on the longer test sets, computed with the same conventions at the largest train size.

## Scaling probe

After all predictions are saved, the runner takes the model fitted on the configured task and train size and times `predict_proba` on a batch of random prefixes at lengths 2^6 to 2^12. Each length gets one warmup call and several timed calls; the evaluator takes the median. In the container each length runs in a forked child that resets the kernel's peak-RSS counter (`/proc/self/clear_refs`), so the reported peak is per length. That delta has a floor of roughly 10 MB of fixed overhead, so memory slopes mean something only above it. The probe stops early if a length's median exceeds `stop_after_s`. The evaluator fits log(time) against log(n) over all lengths and over the longest `tail_points` lengths. The verdict compares the tail slope with the claimed exponent: `match` if within `tolerance`, otherwise `mismatch`.

## Execution path

```
host: build tasks (seeded) -> candidate_inputs/ (spec.json, train pools, prefixes)  +  oracle/*.npz
candidate job (executor, per entrant):  /in = runner.py, candidate/, spec.json, data/  -> /out = pred/*.npy, describe.json, scaling.json, runner.json
evaluator job (executor, once):        /in = code/hybrid/bench/{metrics,scaling,fingerprint,evaluator}.py, oracle/, entrants/<name>/{receipt.json,out/}  -> /out = bench_result.json
host: bench_result.json + host block (trusted, receipts, timings) + manifest.json (sha256 of every file under bench/)
```

The evaluator checks every entrant file against the sha256 in the receipt of the job that wrote it, loads arrays with `allow_pickle=False`, and uses nothing the entrant reports as a score. Baselines run through the same candidate path. Their job outputs are cached under `runs/_bench_cache/<task config hash>/<image digest prefix or local>/<baseline>-s<seed>/`. The task config hash covers the seeds, budget, task parameters, probe settings and the source of `tasks/`, `runner.py` and `baselines/`, so any change to those files recomputes the baselines.

## Baselines

All five share one torch training loop (Adam, gradient clip 1, next-token loss on every position, early stopping on a held-out eighth of the training sequences, checked ten times) and run with one thread.

| Name | Model | Claimed time exponent |
| --- | --- | --- |
| `ngram` | Witten-Bell interpolated 4-gram (NumPy) | 0 (reads the last three tokens) |
| `gru` | embedding 64, GRU hidden 128 | 1 |
| `transformer` | d 64, 2 layers, 4 heads, FFN 256, sinusoidal positions, causal SDPA | 2 |
| `ssm` | 2 LRU-style diagonal complex linear recurrence layers, state 64, computed as an FFT convolution | 1 |
| `linear_attention` | embedding 64, 2 blocks of single-head causal linear attention (feature map elu(x) + 1, no positional encoding) plus a 256-wide FFN. Each layer's state is a 64 x 64 matrix plus a 64-vector normalizer (8320 floats in all) whatever the context length. Evaluated in chunks of 64 positions (exact masked attention inside a chunk, running state across chunks), so cost per sequence is linear in length | 1 |

## Mechanism fingerprint

The loop's failure mode is a disguised RNN, SSM or linear-attention model reported as a new architecture. The fingerprint measures what an entrant does and compares it with the baselines. It is computed in the evaluator job for the candidate and for every baseline (so there is a reference), from prediction arrays and the scaling probe only, and written to `entrants[name].fingerprint` in `bench_result.json`. The candidate's `nearest_baseline` and `flags` are repeated under the top-level `fingerprint.candidate`, and every threshold used is recorded under `fingerprint.config`.

Why this catches a disguised architecture. A model that is a GRU, a state-space model or a linear-attention model under another name has the signatures of that family: it stores a bounded number of key-value pairs per state float, so recall accuracy falls with K at a knee that depends on the state size; its predict time grows linearly with context; it gains little from extra in-context examples on the rule-induction task; and, above all, it makes the same mistakes as that family on the same prefixes. Each of these is read from arrays and timings the trusted evaluator holds, so renaming the class, changing the describe notes or adding layers that compute the same recurrence does not change them. Recall capacity per state float and the few-shot slope ask whether the candidate has anything the fixed-state baselines lack; the time exponent rules out plain quadratic attention; length generalization shows whether it keeps working beyond the training length as the recurrent baselines do; and the behavioral similarity says outright which baseline it resembles.

| Part | What is measured | Output keys |
| --- | --- | --- |
| Recall capacity | Recall accuracy by K at the largest train size. The K at which accuracy falls to half its value at the smallest K, found by interpolation in log K between the bracketing points. Status `knee` (found), `no_knee_in_range` (accuracy never halves up to the largest K, so capacity is at least that K and `pairs_is_lower_bound` is true) or `not_learned` (accuracy at the smallest K is under `learned_ratio` times chance, so capacity is 0). Divided by `describe()["state_floats_per_sequence"]` to give `capacity_per_state_float`, `state_floats_per_pair` and `value_bits_per_state_float` | `recall_capacity` |
| Scaling | Tail and full log-log time slope from the probe, its fit quality, and whether peak memory grows with context: `flat` if the memory slope is under `memory_flat_slope` or the peak delta at the longest length is under `memory_floor_mb`, else `growing`. Peak RSS includes the activations of a whole batch, and has a floor of about 10 MB, so this is a weak signal for state size | `scaling` |
| Length generalization | KL and accuracy on hmm, vom and compose sequences 2x and 4x the training length, scored beyond the training length, with the change from the training-length test set (`kl_delta`, `kl_ratio`, `acc_delta`) | `length_generalization` |
| Few-shot slope | Fewshot accuracy at each number of shown examples (largest train size) and the least-squares slope per doubling of shots, with its standard error | `fewshot` |
| Behavioral similarity | For each baseline other than itself, per task at the largest train size: mean Jensen-Shannon divergence (nats, 0 to ln 2) between the two predictive distributions on the same test prefixes, mean KL(entrant, baseline), and the oracle-normalized ratio below. `nearest_baseline` names the baseline with the smallest mean ratio, plus the nearest by mean JS and per task | `similarity`, `nearest_baseline` |

Why a ratio as well as the JS: two good models are close to each other because both are close to the oracle, whatever their mechanism. With d(a, b) = sqrt(mean JS(a, b)), which obeys the triangle inequality, the ratio d(a, b) / (d(a, oracle) + d(b, oracle)) lies between 0 and 1. It is near 0 when the two models make the same errors, and near 0.7 when their errors are unrelated. A task counts as close only if the mean JS is at most `js_close` and the ratio at most `ratio_close`. A model equal to the oracle has no ratio (both distances are 0 against a perfect baseline), and no evidence either way. Answer tasks (recall, fewshot) use the one-hot answer as the oracle. Caveat: two models that have both learned nothing on a task (fewshot for most entrants) look alike there, and the ratio and JS cannot tell them apart.

### Flags

| Flag | Raised when | Threshold (`[fingerprint]`) |
| --- | --- | --- |
| `behaves_like:<baseline>` | the task is close to that baseline on more than `close_task_fraction` of the compared tasks | `js_close` 0.02, `ratio_close` 0.5, `close_task_fraction` 0.5 |
| `no_capacity_advantage` | `capacity_per_state_float` is at most `capacity_advantage_ratio` times the best of the other fixed-state baselines (`gru`, `ssm`, `linear_attention`). A candidate that did not learn recall has capacity 0 and is flagged | `capacity_advantage_ratio` 1.0, `learned_ratio` 2.0 |
| `quadratic_or_worse` | the tail time slope is above `quadratic_slope` | 1.5 |
| `no_fewshot_gain` | the few-shot slope per doubling of shots is below `fewshot_slope_min` | 0.03 |

The n-gram (state "window") and the transformer (state "growing") do not set the capacity bar: the n-gram reads three tokens and cannot store pairs, so its above-chance recall comes from token statistics, and its 3-float state would give a bar no storing mechanism could meet; the transformer's state grows with context. Flags are computed for baselines too (for reference, with the other baselines as the comparison set). `flag_details` holds the numbers behind each flag.

### Calibration and limits

The thresholds were fixed before looking at results, then checked against controls on `quick.toml` (see [BASELINES.md](BASELINES.md)): each of the GRU, SSM, linear attention and transformer code re-run with a different seed, and the GRU code unchanged. Three of the four different-seed controls were flagged `behaves_like` their own architecture (gru 4 of 5 tasks, ssm 3, transformer 3); linear attention with another seed was not flagged (2 of 5) and its nearest baseline by ratio was linear attention. No pair of different architectures among the quick-config controls and baselines reached 3 of 5 tasks, but on the `default.toml` baseline run linear attention and the transformer did (3 of 5 at ratio 0.30), so each is flagged `behaves_like` the other there. The unchanged GRU gets JS exactly 0. Linear attention and the transformer are hard to separate on these tasks: they sit at ratio 0.27 to 0.31 from each other, as close as a different seed of the same architecture. A disguised transformer-like model is therefore reported as close to both, and the flag alone cannot say which. The time-slope and memory labels move between runs on the shared VM (the transformer's tail slope read 1.45 on one quick run and 1.53 on the default run, either side of the 1.5 threshold), so `quadratic_or_worse` is a weak flag at these lengths.

Other limits: recall has three K values, so the half-capacity K is an interpolation from three points with a binomial standard error of about 0.03 to 0.05 per point; at these budgets the transformer baseline does not learn recall on `quick.toml`, so it has no capacity estimate there; the fewshot slope is zero for most entrants because they do not learn the task; a model that trains much longer than the budget may separate more than these controls show.

## Card falsifier tasks

`evaluate_candidate(..., falsifiers=[task dir, ...])` runs card-specific tasks next to the five standard ones. The research loop's `falsifier_designer` role writes the task programs from a theory card (never from the candidate code); the contract is in `falsifier.py`: `generate(seed)` returns `vocab_size`, `train`, `prefixes`, `next_tokens`, `oracle_kind` (`answer` or `dist`), `oracle`, `group` and `group_labels`, using NumPy and a short standard-library list only. Each task carries one to three pass rules, pre-registered and quoted from the card:

```
{"claims": ["falsifier.3"], "quote": "<card text>", "if_met": "supported" | "refuted",
 "conditions": [{"entrant": "candidate", "minus": "linear_attention" | null, "metric": "acc" | "kl",
                 "group": "K=64" | null, "op": ">=" | "<=" | ">" | "<", "value": 0.9}]}
```

Validation (`falsifier_harness.py`, one executor job per task, `falsifier.run_validation_job` or the research loop's CodeRunner): the generator runs in two child processes with different `PYTHONHASHSEED` values and must give the same sha256; an audit hook records and blocks file writes, reads outside the Python installation and `/in`, sockets, subprocesses and ctypes while the program runs (on top of the static AST checks in `falsifier.static_problems` and the container's `--network=none` and read-only root); structure and the caps in `falsifier.CAPS` are checked; the oracle must score itself perfectly (KL 0, accuracy 1, per group); every next token must agree with the oracle; the uniform and unigram predictors scored by `metrics.py` must equal their analytic levels computed from the definitions; rule groups must exist; and a rule that would support the card must not be met by a trivial predictor in the candidate's place. A passing job writes the data; `falsifier.write_task` turns it into a task directory with the unigram reference added.

Execution: one more candidate job per (entrant, task) runs `falsifier_runner.py` (a fresh model fitted on all training sequences, then `predict_proba` on the test prefixes). Cached baselines still run these jobs. The oracles are staged only into the evaluator job, where `falsifier_eval.evaluate` scores every entrant (overall and per group, the same `metrics.py`), adds the uniform and unigram references, and decides each rule. A rule is `inconclusive` when the candidate has no valid predictions, when it does not beat the unigram reference by 0.05 nats of KL or 0.05 accuracy (it did not train on the task), or when a named entrant has no valid predictions. Otherwise the claims are `if_met` when every condition holds and the opposite when not. Output: `bench_result.json` `falsifiers.<task id>` with `entrants`, `references` and `decisions`, and `host.falsifier_receipts`.

## Trust model and known gaps

- The candidate job has no network, a read-only root, one CPU and memory, pids and wall caps (see `hybrid/executor/README.md`). It receives train data and test prefixes only. The oracle exists only on the host and in the evaluator job. `hybrid/tests/bench/test_isolation.py` audits both: what is staged, and a snooping candidate that searches the container and probes the host oracle paths.
- Candidate code runs in the same process as the runner. It could read the test prefixes during `fit` (unlabeled data, transductive use) or tamper with the runner's timers and so with the scaling numbers. Predictions cannot be faked into good scores this way, because every score is computed by the evaluator from the arrays.
- The fingerprint is computed from the entrant's prediction arrays, the oracle and the scaling probe. The one self-reported number it uses is `describe()["state_floats_per_sequence"]`, as the denominator of recall capacity per state float. An entrant that understates its state looks more efficient on that one measure; the capacity curve, the behavioral similarity and the scaling exponent do not depend on it. The baselines' fixed or growing state is host knowledge (`baselines/__init__.py`), not taken from their `describe()`.
- Wall time is measured on a shared VM. With `parallel_jobs` above 1, concurrent jobs share the VM's 8 CPUs, which adds noise to the scaling probe.

## Test ladder, rung 1 (recall calibration)

DISCOVERY-PIPELINE-PLAN.md section 11 puts a fixed, calibrated ladder in front of every candidate result. Rung 1 is the set of small learned tasks from IMPROVEMENT-PLAN.md B and C: the bench must reproduce the published orderings of known models (B.5) before a candidate number from it counts. The current five-task bench above cannot do that (on `quick.toml` the transformer sits at chance on `recall`, linear attention and the transformer are flagged `behaves_like` each other on `default.toml`, and in `results/falsifier-20261011/recall-investigation.md` no entrant learned a card's recall task under the 200-step budget), so rung 1 is a separate harness with its own tasks, baselines, protocol and config. Calibration results: [BASELINES.md](BASELINES.md), section "Test ladder rung 1".

### Tasks (`tasks/mqar.py`, `tasks/copy_task.py`, `tasks/ngram_lookup.py`)

Each module is NumPy only and self-contained (the job imports it by file name), provides `generate(rng, n, ...)` returning `tokens` and `targets` (`-100` where a position is not scored) plus annotations, and `oracle(prefix, ...)`, an independent parser that recomputes every label from the visible prefix alone. Tests: `hybrid/tests/bench/test_ladder_tasks.py` (determinism, format invariants, oracle self-score 1.0).

| Task | Definition | Scored positions | Evaluation sets |
| --- | --- | --- | --- |
| `mqar` | Zoology `multiquery_ar` (commit d90de38, checked against the code): K pairs first, keys from 1 .. V/2 - 1 and values from V/2 .. V - 1 without repeats; every key queried once at a power-law slot (`power_a` 0.01); empty positions filled with uniform tokens over [0, V); the label sits at the query key | every query | one per K; per key-to-query distance bucket |
| `copy_task` | Repeat After Me copy: `BOS x1..xL COPY x1..xL EOS`, x uniform over D = 16 tokens, training L uniform in 1 .. 32. Named `copy_task.py` so it never shadows the standard library `copy` inside the job | COPY and the first L - 1 copied tokens (targets x1 .. xL); string accuracy = all L correct = greedy decoding reproduces x | one per L in {16, 32, 64, 128, 256} |
| `ngram_lookup` | Repeat After Me section 3.5, suffix key (`BOS x SEP g a`) and prefix key (`BOS g SEP x SEP a`), n = 3, k = 3, D = 16, the key unique in x, training L <= 30 | the token before a1 and a1 .. a(k-1) | one per L in {20, 30, 60, 100} |

Answer-only loss: training uses the cross-entropy at scored positions only (B.2 item 1). The old bench's candidate API is unchanged; the ladder trains its own baselines.

### Baselines (`baselines/bb_recall.py`)

Every family shares one two-layer backbone (embedding, then per layer RMSNorm, sequence mixer, residual, RMSNorm, SwiGLU MLP, residual; final RMSNorm and head) and differs only in the mixer: `transformer` (RoPE, or NoPE, or Hard-ALiBi heads as in Repeat After Me appendix A), `sliding_window` (each query sees itself and the previous w positions), `linear_attention` (elu+1, normalized, the old baseline's math), `deltanet` and `gated_deltanet` (delta rule, Gated DeltaNet Eq. 10, q and k L2-normalized, Mamba-2 style gate), `mamba2` (SSD, expand 2, head dimension 32, one B/C group). Fixed-state mixers get a kernel-4 causal depthwise convolution (B.4).

- Training forms are chunked (chunk 64): `chunk_gla` (exact decayed attention inside a chunk, running state across chunks; used by linear attention with a ones column for the normalizer, and by Mamba-2 with q = C, k = B, v = dt x, g = dt A) and `chunk_delta` (WY form, the triangular systems solved with `torch.linalg.solve_triangular`). Both equal their token recurrences (`recurrent_gla`, `recurrent_delta`) to 1e-10 in float64 at lengths that are and are not multiples of the chunk.
- Every mixer has a `step()` form. `LadderNet.stream()` runs the model token by token and equals the parallel forward to 1e-9 for every family; the floats `step()` carries after n tokens equal `state_formula(family, d, n)`, which the host uses for every reported state size (never `describe()`). The formula counts the convolution buffers; `include_conv=False` gives the B.3 table's numbers.
- `forward_flops_per_token` is the analytic forward cost at context n.
- `candidate_class(family, d, hyper)` wraps a family in the bench Candidate API, so `api.check_causality` runs on all of them.

Tests: `hybrid/tests/bench/test_baselines_recall.py` (C.4 list: chunked equals recurrent, gate 1 gives DeltaNet, beta 0 leaves the state unchanged, window >= N equals full attention, step form equals parallel form, state formula exact, state flat or growing, causality, finite gradients, parameter counts within a factor 1.6 of the transformer's, FLOPs ordering).

### Running it (`ladder.py`, `ladder_runner.py`, `configs/ladder.toml`)

```sh
.venv-hybrid/bin/python -m hybrid.diskguard --require run
.venv-hybrid/bin/python -m hybrid.bench.ladder microbench --config hybrid/bench/configs/ladder.toml --out results/phase10-<stamp> --threads 1,2
nohup .venv-hybrid/bin/python -m hybrid.bench.ladder run --config hybrid/bench/configs/ladder.toml --out results/phase10-<stamp> --slice first --workers 7 > results/phase10-<stamp>/slice-first.log 2>&1 &
.venv-hybrid/bin/python -m hybrid.bench.ladder report --config hybrid/bench/configs/ladder.toml --out results/phase10-<stamp>
```

One executor job per (task, model, learning rate, seed), one thread each (a second thread gained only 1.1x to 1.5x in the microbenchmark). The host writes the evaluation sets: tokens and the positions to report go to the job, the targets and annotations stay on the host. The job generates a fresh training batch every step from the train seed, keeps the parameters with the best accuracy on a validation stream from the training distribution (checked 20 times; it stops early after two checks at 0.995 or above), and returns the argmax at every listed position; the host scores. The job reads whole test sequences in one causal forward pass, which equals predicting each prefix alone for these baselines (their causality is tested); an untrusted candidate must be scored on prefixes instead.

Learning rates (B.4): the slice's first seed (seed 0) at each listed rate, then the remaining seeds at the learning rate with the best validation accuracy (`<out>/lr_choice/`). Slice `first` used seeds 0, 1, 2 (before decision B10); the default seeds are now 0 to 3 (`ladder.DEFAULT_SEEDS`), seed 0 sweeping and seeds 1, 2 and 3 being the acceptance seeds, as in slice `diag1`. Run ids hash the task, model, effective training settings, test seed and the code of the runner, baselines and tasks, plus the execution mode when it is not the container; a finished run (`<out>/runs/<id>/result.json`) is skipped, so rerunning the command resumes. `report` writes `<out>/calibration.json` with the B.5 criteria (`ladder_criteria.py`), the per-seed table, the one code hash all its results share (a directory mixing code hashes is an error), the devices they ran on, `trusted` (true only when every result ran in the Podman CPU container), `scope` (`acceptance`, or `diagnostic` when any result is a host MPS run) and the predeclared thresholds as `policy`.

Calibration gate: copy a report to `hybrid/bench/calibration/rung1.json`; `python -m hybrid.bench.ladder gate` exits 0 only when that report is `trusted: true` (host MPS reports are diagnostics and are refused), was judged on seeds 1, 2 and 3, belongs to the current code hash and records criteria 1, 2, 3, 4, 6, 7, 8 and 9 as passed (`fail`, `incomplete`, `unresolved` and `info` all keep rung 1 uncalibrated), and otherwise prints the reasons and exits 1 (a missing report is "not calibrated"). Calibration status is an experiment result, so the pytest suite tests the gate's rule on fixture reports (`hybrid/tests/bench/test_ladder_calibration.py`) and does not read the stored report.

Execution modes. A slice may set `execution`, `train` and `caps` overrides and `seeds`. `execution = "host_mps"` (configured by `[execution.host_mps]`: host python and worker count) trains on the Mac's GPU through `ladder_host.py`, which Andrew approved on 2026-10-13 for calibrating our own baselines only. It refuses any job whose staged files differ from the trusted runner, `bb_recall.py` and task modules, runs each job under `sandbox-exec` with network denied and writes confined to the job, records device, torch version, code hash and backend hash in every result, and refuses to run with `PYTORCH_ENABLE_MPS_FALLBACK` set. Candidate code never runs on the host. MPS is not bitwise deterministic and not bitwise equal to CPU: over 300 steps per-step losses stayed within 6e-4 relative of CPU (results/phase10-20261013/mps-probe). The device option left CPU training bitwise unchanged (results/phase10-20261013/cpu-equivalence).

```sh
nohup .venv-hybrid/bin/python -m hybrid.bench.ladder run --config hybrid/bench/configs/ladder.toml --out results/phase10-20261013 --slice diag1 > results/phase10-20261013/slice-diag1.log 2>&1 &
.venv-hybrid/bin/python -m hybrid.bench.ladder gate
```
