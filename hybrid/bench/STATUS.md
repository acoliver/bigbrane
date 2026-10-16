# hybrid/bench status

Spec: [RESEARCH-SYSTEM-SPEC.md](../../RESEARCH-SYSTEM-SPEC.md) section 4. Interface for the orchestrator: [README.md](README.md) ("Python API" and "CLI"). Baseline numbers: [BASELINES.md](BASELINES.md).

## Fingerprint pass (done)

The mechanism fingerprint is implemented, tested and measured through the Podman executor. Spec: stop the loop from reporting a disguised RNN, SSM or linear-attention model as a new architecture. Everything is computed in the evaluator job from prediction arrays, the oracle and the scaling probe; the only self-reported number is `describe()["state_floats_per_sequence"]`, the denominator of capacity per state float. Definitions, thresholds and limits: README "Mechanism fingerprint". Numbers: [BASELINES.md](BASELINES.md).

| Item | Where | Evidence |
| --- | --- | --- |
| Recall capacity versus K and per state float | `fingerprint.recall_capacity` | `test_fingerprint.py` (knee interpolation, no-knee lower bound, not-learned) |
| Scaling exponent and memory growth | `fingerprint.scaling_block` | `test_fingerprint.py` (flat, growing, unknown) |
| Length generalization at 2x and 4x | `evaluate.py` (longer fresh test sets), `fingerprint.length_generalization` | `test_length_probe_sets_are_fresh_and_beyond_the_training_length`, `test_length_generalization_reports_kl_at_two_and_four_times_the_training_length` |
| Few-shot slope | `fingerprint.fewshot_block` | `test_fewshot_slope_is_accuracy_gained_per_doubling_of_shots` |
| JS similarity to each baseline, nearest baseline | `fingerprint.similarity_block`, `nearest_baseline` | `test_similarity_normalizes_by_distance_to_the_oracle`, `test_a_copy_of_a_baseline_is_flagged_behaves_like_it` |
| Flags `behaves_like:<baseline>`, `no_capacity_advantage`, `quadratic_or_worse`, `no_fewshot_gain`, thresholds recorded in `fingerprint.config` | `fingerprint.flags_for`, `[fingerprint]` in each config | `test_each_flag_triggers_on_its_own`, `test_a_different_mechanism_raises_no_flags`, `test_fingerprint_thresholds_are_validated_and_recorded` |
| Linear-attention baseline (fixed 8320-float state, chunked, linear time) | `baselines/bb_neural.py` | `test_linear_attention_*` (equals the quadratic reference across chunk boundaries, equals a recurrent d x d state) |
| GRU submitted as a candidate is flagged `behaves_like:gru` | `evaluate_candidate` | `test_the_gru_submitted_as_a_candidate_is_flagged_behaves_like_gru` (local executor) and `..._through_podman` (Podman, labeled run, no containers or volumes left) |

Result keys: `entrants[name].fingerprint` (with `nearest_baseline`, `flags`, `flag_details`) and top-level `fingerprint` (`config`, `candidate.nearest_baseline`, `candidate.flags`). `evaluate_candidate`, the CLI and the existing keys are unchanged in shape; the fingerprint keys are additions.

Measured on `quick.toml` through Podman (`results/bench-final/`):

- The GRU's own code submitted as a candidate gets `behaves_like:gru`, `no_capacity_advantage`, `no_fewshot_gain`, nearest baseline `gru` with JS 0.0000 on 5 of 5 tasks. Wall time for that evaluation, fingerprint included: 78.7 s (candidate job 74.3 s, evaluator job 0.4 s, baselines cached).
- Four seed-changed controls: gru, ssm and transformer are flagged as themselves (4, 3, 3 of 5 tasks close); linear attention with a new seed is not flagged (2 of 5).
- Linear attention and the transformer are not separated well: ratio 0.29 apart on quick (2 of 5 close), and on the default baseline run each is flagged `behaves_like` the other (3 of 5). A disguised linear-attention or transformer-like entrant will be reported as close to both.

## State

Complete for the prototype. All items in the spec's section 4 are implemented and exercised through the Podman executor.

| Spec requirement | Where it is met | Evidence |
| --- | --- | --- |
| Candidate API (`fit`, `predict_proba`, `describe`) | `api.py`, `candidate_template.py`, `runner.py` | `test_causality.py` (batch independence for every baseline and the template; future tokens do not change past logits) |
| Tasks with exact oracles: HMM, variable-order Markov, associative recall (K varies), hierarchical composition, few-shot rule induction (disjoint train and test rules) | `tasks/` | `test_oracles.py`: filtering and tree walks and bracket-stack distributions checked against brute-force enumeration on tiny instances, samplers checked against the enumerated joint, recall and fewshot answers recomputed by parsing and by enumerating all consistent rules, train and test rules disjoint |
| Baselines: n-gram, GRU, tiny transformer, diagonal SSM, same budgets (plus linear attention for the fingerprint) | `baselines/` | all five run through the executor, [BASELINES.md](BASELINES.md) |
| Candidate job sees train data and test prefixes only; evaluator is a separate job | `evaluate.py`, `runner.py`, `evaluator.py` | `test_isolation.py`: staged files audited against the oracle hashes; a snooping candidate that searches the container finds no oracle |
| Metrics computed in the evaluator: KL to oracle, accuracy, tokens-to-target | `metrics.py`, `evaluator.py` | `test_evaluator.py`: perfect, uniform, tampered, invalid and failed entrants |
| Scaling probe with log-log slope compared to the claimed exponent | `scaling.py`, `runner.py` | `test_scaling.py`: exact on synthetic power laws; Podman runs measure GRU and SSM slopes near 1 and the transformer at 1.5 |
| Baseline results cached | `evaluate.py` (`runs/_bench_cache/<task config hash>/<image>/`) | `test_local_mode.py` and `test_e2e_podman.py`: the second evaluation reuses the first baseline job id |
| `--local` mode labeled `trusted: false` | `local.py`, `__main__.py` | `test_local_mode.py` (API and CLI) |
| `bench_result.json` with task metrics, baseline metrics, scaling slopes, receipts | `evaluate.py` | README "Python API" |

## Measured runtimes (Podman, image `7fa94b73...`, 8-CPU VM, one CPU per job)

Fingerprint-era measurements (five baselines, fingerprint computed in the evaluator job; logs in `results/bench-final/`):

| Measurement | Wall time | Log |
| --- | --- | --- |
| One quick candidate evaluation with fingerprinting, `quick.toml`, GRU as candidate, baselines cached | 78.7 s (candidate job 74.3 s, evaluator job 0.4 s) | `quick-gru-candidate.err.log`, `.time` |
| Five baselines, `quick.toml`, cold cache, five jobs in parallel | 147.2 s (ngram 0.6 s, gru 95.7 s, ssm 124.1 s, transformer 137.1 s, linear_attention 141.1 s) | `quick-baselines.err.log`, `.time` |
| Five baselines, `default.toml`, cold cache, five jobs in parallel | 491.6 s (ngram 0.7 s, gru 199.3 s, ssm 326.2 s, linear_attention 477.3 s, transformer 479.1 s) | `default-baselines.err.log`, `.time` |
| Five seed-control candidates, `quick.toml`, one at a time, baselines cached | 64.6 s to 115.2 s each | `control-*.time` |

Against the same GRU candidate before the fingerprint work, the evaluator job went from 0.3 s to 0.4 s and the candidate job from 64 s to 74.3 s (it now also predicts the 2x and 4x length test sets; the two runs were not controlled against each other). The five-way parallel runs are slower than the earlier four-way runs because a fifth job shares the VM. Measurements before the fingerprint work:

| Measurement | Wall time | Log |
| --- | --- | --- |
| One full candidate evaluation, `default.toml`, GRU as candidate, baselines cached (`python -m hybrid.bench eval`) | 178.9 s (candidate job 173.6 s, evaluator job 0.4 s) | `results/bench-20261009T233628Z/gru-candidate.err.log` |
| Four baselines, `default.toml`, cold cache, four jobs in parallel | 351.5 s (ngram 0.9 s, gru 198 s, ssm 280 s, transformer 344 s) | `results/bench-20261009T233628Z/baselines.err.log` |
| One full candidate evaluation, `quick.toml`, GRU as candidate, baselines cached | 71.0 s (candidate job 64 s, evaluator job 0.3 s) | `results/bench-finish/quick-gru-candidate.err.log` |
| Four baselines, `quick.toml`, cold cache | 114.7 s (ngram 0.5 s, gru 67.7 s, ssm 99.0 s, transformer 110.6 s) | `results/bench-finish/quick-baselines.err.log` |

The default evaluation is under ten minutes for a GRU-class candidate, so `quick.toml` was not required to bring it down. It exists as a cheaper first pass: shorter sequences, three train sizes (32, 128, 256), 600 steps, and a probe that stops at 2^11. A candidate that is slower per step than a GRU scales the candidate job time roughly in proportion; the transformer baseline needed 344 s (default) and 110.6 s (quick) in a four-way parallel run. Caps: candidate job 1800 s (default) and 900 s (quick).

The first candidate evaluation on `default.toml` was started by the previous builder and finished before this pass began. As a determinism check, the GRU code submitted as a candidate gave the same metrics as the GRU baseline to every printed digit in both configs.

## Phase 10 criterion revisions (decision B10, 2026-10-16)

`ladder_criteria.py` implements the revised B.5 criteria 2, 4, 6 and 9 and the acceptance execution policy; details, thresholds and changed assertions are in [B10-STATUS.md](B10-STATUS.md). Criterion 2 requires a learned K8 control and reports a model that never learns as `unresolved`. Criterion 4 measures beyond the layers × window receptive field and compares counted state including convolution buffers. Criterion 6 splits the empirical copy separation from the finite-state information bound (reported as `info` at the current grid). Criterion 9 splits meaningful order (predeclared ties) from same-seed repeatability. Reports carry `trusted`, `scope` and `policy`; the gate refuses host MPS (diagnostic) reports and any report not judged on seeds 1, 2 and 3. Read back, the stored diag3 results give criterion 2 five learned passes and four unresolved seeds (previously a failure) and criterion 9 the same three order reversals. No sweep has been run under the revised criteria; `hybrid/bench/calibration/rung1.json` remains absent and rung 1 uncalibrated.

## Phase 10 acceptance slice (decision B10t)

Slice `accept` in `configs/ladder.toml` holds every cell the calibration criteria read, a same-seed repeat attempt for every cell (`*-rep1` models, `lr_from` cells in `ladder.py`) and a long-copy task (`copy_long`, L = 76,000) that makes the criterion 6 information bound testable. 207 jobs, about 25 hours on 4 Podman slots (14 hours at the isolated microbenchmark rate). Counts, basis, launch command and tests are in [B10-SLICE-STATUS.md](B10-SLICE-STATUS.md). No sweep has been run.

## Tests

`.venv-hybrid/bin/python -m pytest hybrid/tests/bench -q` runs the suite; Podman tests run only when `BIGBRANE_BENCH_PODMAN=1` (VM running, executor image present). 
| Run | Result | Log |
| --- | --- | --- |
| Without Podman, before the new tests | 39 passed, 2 skipped | `results/bench-finish/pytest-local-1.log` |
| With `BIGBRANE_BENCH_PODMAN=1`, before the new tests | 41 passed (30.2 s) | `results/bench-finish/pytest-podman-1.log` |
| Without Podman, final (adds `test_local_mode.py`, hardened timing tests in `test_scaling.py`) | 47 passed, 2 skipped (8.9 s) | `results/bench-finish/pytest-final-local.log` |
| With `BIGBRANE_BENCH_PODMAN=1`, final | 49 passed (26.0 s); no labeled containers or volumes left | `results/bench-finish/pytest-final-podman.log` |
| Fingerprint pass, without Podman | 71 passed, 3 skipped (21.4 s) | `results/bench-final/pytest-local.log` |
| Fingerprint pass, with `BIGBRANE_BENCH_PODMAN=1` | 74 passed (43.0 s); `podman ps -a` and `podman volume ls` filtered on `bigbrane.owner=bigbrane` empty afterwards and after the baseline and candidate runs | `results/bench-final/pytest-podman.log`, `after-pytest-*.log`, `final-*.log` |

`test_scaling.py` timed synthetic busy-wait functions with margins small enough to fail under host load (one failure in each of two runs while other jobs were running: a fitted slope of 2.30 against 2.0 and 1.18 against 1.0, and a stop test whose 128-length median crossed its 5 ms limit). The work per call is now 5 to 100 times larger (linear and quadratic cases), repeats are higher, and six consecutive runs passed.

## Known gaps

- Fingerprint: linear attention and the transformer are hard to tell apart on these tasks (see above). Thresholds were fixed before the controls and not tuned on them, with five controls and five baselines as the only calibration data. Recall has three K values, so the knee is an interpolation. The tail time slope sits near the `quadratic_or_worse` threshold for the transformer (1.45 on quick, 1.53 on default), so that flag is unreliable at these context lengths. Memory-growth labels moved between runs for the GRU and the transformer. The default-config GRU-as-candidate evaluation was not remeasured after the fingerprint work.
- `describe()["state_floats_per_sequence"]` is self-reported and is the one fingerprint input an entrant controls; an understated state makes recall capacity per float look better. The baselines' state kind comes from host code, not from `describe()`.

- The fewshot task is not learned by any baseline at these budgets (chance 0.077; the best is the transformer at 8 shots with 0.26 on `default.toml`). Targets for it are set where the transformer crosses them, so tokens-to-target for fewshot separates the transformer from the others and nothing else.
- Scaling slopes of the transformer baseline read 1.5 against a claim of 2 at lengths up to 4096, because fixed and linear costs dominate for a model this small. The probe reports what the code does at the lengths it measures. A candidate claiming a different exponent is compared with the measured tail slope within `tolerance` (0.3 on default, 0.4 on quick), so the verdict can be `mismatch` for an honest claim when lengths are short.
- Peak memory per length has a floor of about 10 MB of fixed overhead in the container; memory slopes are informative only above it.
- Candidate code runs in the runner's process. It can read test prefixes during `fit` (unlabeled, transductive use) or interfere with the runner's timers and so with scaling numbers. It cannot change its scores, which the evaluator computes from the arrays it wrote. See the trust model in the README.
- Wall time and memory come from a shared VM; with `parallel_jobs` above 1 the concurrent jobs add noise to the scaling probe.
