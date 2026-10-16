# Fix recovery and freeze verification before the confirmatory run

Review date: 2026-09-30. This is a full implementation and design review of version 1, including RESEARCH.md, EXPERIMENT.md, README.md, pyproject.toml, every bigbrane/*.py module, all three test modules, both supplied pilots, and the verification logs.

## Verdict

**The confirmatory run should not proceed as specified until the three P1 findings below are fixed.** The supported resumable, frozen-input workflow can launch duplicate workers, reuse changed selection artifacts, and adopt inputs that differ from the manifest. These failures can invalidate selection, provenance, or resource measurements.

The mathematical implementation and the inspected scientific decision rules passed review and independent checks. An uninterrupted execution with unchanged inputs has no demonstrated mathematical blocker. Both supplied pilots passed their recorded hashes and reproduced their model tensors and predictions exactly. There is no evidence here that their artifacts were affected by the findings.

Those pilots are pipeline sanity checks only. No confirmatory advantage, fewer-token result, equivalent-quality representation economy, or stress-stage outcome has been established. Any eventual evidence concerns the eight specified generators, the three fixed learners, and their disclosed selection budgets.

Summary: **three high-priority workflow defects, zero demonstrated oracle/model/statistical calculation defects.** The limitations already disclosed in the protocol do not require expanding this study.

## Prioritized findings

### F1. P1: recovery can dispatch a second worker while the original worker is alive

**Locations:** bigbrane/runner.py:135-146; bigbrane/fitting.py:49-58; bigbrane/artifacts.py:34-40.

When a candidate lacks record.json, execute_candidate changes any prior running attempt to interrupted-driver and immediately launches another subprocess. It does not establish that the old worker has exited. The attempt records have no persisted process identity or exclusive ownership mechanism. Killing a driver does not by itself establish that its child has stopped.

Both workers can therefore own the same candidate directory. Their checkpoint, trace, resume, and JSON temporary filenames are identical. A live orphan and its replacement can race writes, overwrite each other's timing metadata, or fail during replacement of a shared temporary file. Even without a file collision, simultaneous execution violates the protocol's sequential CPU measurement condition. A second driver using the same output directory encounters the same ownership gap.

**Independent reproduction:** In an isolated workspace fixture, I paused an actual worker launcher with SIGSTOP and supplied the running-attempt state used by recovery. execute_candidate marked that attempt interrupted-driver and completed a replacement worker while the original process remained alive. The two worker PIDs were distinct. I killed and reaped the paused process before cleanup. This demonstrates duplicate dispatch; I did not deliberately run two active writers into a file collision.

**Required action:** Enforce exclusive ownership of the run/candidate directory for the driver's and worker's lifetimes. Recovery must refuse a replacement launch while an identified prior worker is alive. Persist sufficient process identity to distinguish a surviving worker from a reused PID, and recover completed work only after ownership is resolved. Parent signal cleanup alone cannot cover SIGKILL. Add an integration test that kills a driver with a surviving worker and verifies that resume neither creates a second writer nor proceeds to another concurrently running candidate. Cover simultaneous drivers on the same output directory too.

**Scientific impact:** A blocker for the advertised recovery and sequential resource-measurement workflow. It does not invalidate the supplied clean pilots.

### F2. P1: selection records and earlier checkpoints are not bound to their committed hashes during partial-run recovery

**Locations:** bigbrane/fitting.py:49-58, 76-79, 93-98, 185-203; bigbrane/runner.py:118-130, 167-175, 246-249.

There are two related integrity gaps:

1. A completed candidate's artifact_hashes protect checkpoint files and trace.json, but exclude record.json itself. Before the whole run finishes, the driver does not bind the candidate record to a previously committed digest. Reuse trusts its validation_nll, chosen checkpoint, counters, and search coordinates. The driver also skips the worker's identity.json comparison on this path.
2. An interrupted candidate verifies only the latest checkpoint referenced by resume.json. The best validation checkpoint may be an earlier file whose hash is no longer recorded in the resume pointer. At completion, the worker computes new hashes from all current checkpoint files, including any changed earlier file. Its original best_score and best_index survive in the latest resume state.

Final result_hashes protect a fully completed run, but are created after selection and evaluation. They cannot detect a changed partial-run artifact that has already been accepted and rehashed.

**Independent reproductions:**

- I fitted two tiny HMM candidates, changed only the losing candidate's record.json validation_nll to -100, then used execute_candidate and freeze_selections. Every existing checkpoint/trace digest still passed. Selection changed from candidate 1 to candidate 0 and froze the impossible negative validation NLL without an error.
- In a one-target RNN fixture, checkpoint 1 was the resume point and checkpoint 0 was the validation winner. After the committed interruption, I changed checkpoint 0's output bias while leaving checkpoint 1 and its resume hash untouched. Resume accepted the changed file and included its new digest in record.json. The saved validation score remained 2.073164435493247 nats/target, while the actual selected checkpoint scored approximately 1.4324891392619725e-08 on the same fixture. This used genuine model predictions and the existing cap/interruption test seams.

**Required action:** Record checkpoint digests at their first successful commit and retain a committed inventory through resume, including every checkpoint eligible for selection. Verify that inventory before continuation or reuse; do not bless earlier committed files by hashing their current bytes anew. Bind completed record.json to a committed completion digest before it becomes a selection input. Check its spec against the expected FitSpec and bind its selected score/index to the trusted trace/checkpoint state. Add negative tests for mutation of an earlier best checkpoint and for changes to a completed candidate's score, selected checkpoint, or identity before stage selection. Each must fail before test access.

**Scientific impact:** A blocker for the promised artifact-verification and validation-selection integrity of partially completed runs. The supplied completed pilots passed all checked digests.

### F3. P1: the freeze is not enforced when a running driver dispatches workers or evaluates outcomes

**Locations:** bigbrane/runner.py:79-85, 113-117, 132-146, 197-212, 244-250; bigbrane/evaluation.py:68-91; bigbrane/artifacts.py:90-106.

Source and stage-data verification occurs during resume initialization and stage preparation. Subsequent candidate dispatch computes new expected learner hashes from the files currently on disk. It never compares those hashes with the frozen stage inventory. Workers correctly compare their input files to the job, but the job can already contain replacement hashes.

Workers also execute `python -m bigbrane.cli` with cwd=ROOT, importing the live implementation. Frozen sources are copied and hashed, but are not the executed worker tree. The driver does not recheck the live/frozen source inventory between candidate subprocesses or before evaluation/finalization. Evaluation loads current evaluator files without comparing them to the stage inventory at that boundary. Thus drift after initialization can escape detection until a later resume, after outcomes may already have been evaluated.

**Independent reproductions:**

- I prepared a separate pilot fixture and changed its first eight training sequences after recording the stage hashes. Explicit verify_inventory correctly rejected the change. execute_candidate nevertheless generated a job containing the new training-file hash and completed the fit, while manifest.data_stages retained the original hash.
- I changed a copied core.py inside another fixture's frozen tree after freeze_sources. Explicit verify_sources rejected it, but execute_candidate still trained successfully. Inspection of the subprocess command confirms that workers execute the live tree. I did not modify the workspace implementation to test live-source drift.

**Required action:** Build learner jobs from the manifest's recorded hashes and fail if current files differ. Verify evaluator inputs against those same frozen inventories before any test/scaling access. Execute from a validated copied source tree and bind worker/evaluator provenance to the manifest; if execution continues to use the live tree, reject source drift before dispatch and before selection, evaluation, and finalization. Add orchestration-level tests for changes after initialization to training, validation, evaluator data, and sources. They must stop before the changed input is fitted or scored. A later resume detecting the change is insufficient.

**Scientific impact:** A blocker for reproducibility and the assertion that all candidates used the preregistered frozen inputs. It can also break shared-data comparisons between families.

## Full-scope checks and results

| Area | Review result |
| --- | --- |
| Research/design claims | The study tests a disclosed four-state prior and fixed tiny controls. Filtering, EM, predictive-state ideas, information bottleneck, MDL, and coarse-graining are presented with their limits. No architectural novelty or broad cross-theory victory is inferred. Literature URLs were read as documented references; I did not independently reacquire the papers. |
| Generator and splits | core.py:26-63 implements the specified stationary solution, dense persistence mixture, paired overlapping emissions, independent-sequence sampling, and second-order initialization. runner.py:27-61 uses the stated matrix, split, and scaling tuples. Saved pilot matrices, token arrays, and latent arrays regenerated exactly. Master training subsets are nested and boundaries are preserved. F3 concerns enforcement after freeze. |
| Seeds and grid | protocol.py:9-51 implements 384 matched and 192 conditional stress candidates, stress size indices 1/3, separate families/replicas, distinct HMM restart seeds, and candidate-zero neural initialization/shuffle pairing. seed_word uses the recorded uint64 conversion. Actual pilot initial tensors and every later tensor repeat exactly. |
| Prefix alignment/reset | neural.py:25-26 supplies BOS plus x1 through x63. Both neural definitions reset per sequence, attention has the upper-triangular future mask, and filtering stores probabilities before observing the target. Existing causality, current-shifted-input, BOS-only, and sequence-reset tests pass. |
| Matched/stress oracles | core.py:66-93 agrees with independent latent-path enumeration. Stress q1 is pi_i A_ij and emission uses the second pair coordinate. The existing lambda-zero tests correctly use pi A for a nonstationary fixture and pi for the stationary generator. |
| Forward-backward/MAP | core.py:96-140 has scaled likelihoods, correct posterior marginals, and pseudocount updates across independent sequences. Independent exhaustive counts agree. fitting.py:170-175 checks the penalized objective and uses three successive sub-threshold improvements per training token. A forced objective decrease fails before checkpoint commit. |
| Neural definitions/optimization | neural.py:29-77 matches the 280-scalar tanh RNN and 712-scalar attention equations, initialization, positions, layer norm epsilon, biases, and untied heads. fitting.py:87-88, 151-167 implements the specified Adam, clipping, full sequence backpropagation, batch sizing, and per-epoch permutation. All 992 scalar cross-entropy derivatives passed independent finite differences. |
| Stopping and raw selection | Neural tolerance-based progress is separate from strict raw validation minima. Initialization is eligible; earlier checkpoint then lower candidate wins exact ties. Existing tests cover sub-tolerance improvements and minimum-epoch stopping. Independent HMM fixtures confirm exactly three flat updates and initialization winning ties. F2 concerns trusting persisted selection artifacts. |
| Leakage and test freeze | LearnerData and the worker job expose training/validation tokens; fitting reads neither generator matrices nor oracle/test files. Every stage completes all fits and writes its selection file before test scoring. The stress trigger uses matched test decisions as explicitly preregistered. This is an audited code boundary, as disclosed, rather than an OS sandbox. F1-F3 undermine workflow guarantees under interruption or artifact drift. |
| Metrics | evaluation.py:12-38 computes full-distribution conditional KL, sampled NLL and paired differences, bits/token, entropy, initial/noninitial and block summaries, and per-sequence records. Neural logits are converted to float64 before log-softmax. Probability/nonfinite checks and negative-KL tolerance are present. Unigram smoothing and oracle-self checks match the protocol. Pilot selected predictions and KL were recomputed exactly. |
| Statistical decisions | decisions.py:6-24, 37-111 uses generator-level paired deltas, equal weights, the two-regime conjunction, six-positive-generator gate, 0.02 margin, and a 5,000-draw stratified bootstrap. An independent loop reproduced the interval exactly. Target crossing requires all larger sampled sizes to meet 0.05 and correctly handles nonmonotonicity and censoring. Stress reports retain matched same-size values and report reversal/retention against both controls. |
| Sample accounting | Observed training targets include the first symbol, exclude BOS, and are distinct from repeated optimization exposures. Accessible-token totals include the 2,048 validation tokens. Pilot epoch/iteration, exposure, diagnostic-pass, step, and validation counts agree with traces. Counts and dtype-aware parameter/optimizer bytes match 52 stored/43 free HMM entries, 280 RNN, and 712 attention parameters. |
| Resource measurement | CLI backend environment settings precede imports, candidates use fresh subprocesses, and CPU/Torch thread settings are recorded. Validation, training including diagnostics, checkpoint serialization timing, process-attempt wall time, RSS high-water marks, and tensor bytes are distinguished. Benchmarks implement two warmups and ten timed passes at batches 1/32 with all prediction heads/normalization included. Dense length scaling uses selected largest matched size, replica zero, evaluator-only arrays. F1 can violate sequential execution; F2/F3 can invalidate costs/provenance. |
| Persistence/manifest | JSON replacement, checkpoint replacement, optimizer state, stopping state, and committed-boundary numerical resume work in existing tests. Frozen source/environment/data and completed result digests pass for both pilots. Interrupted-driver recovery, partial-run integrity, and in-flight freeze enforcement remain incomplete as detailed in F1-F3. |
| CLI/dependencies/reporting | The declared NumPy dependency and Torch extra match installed usage. Python 3.12 is supported, roots are enforced for public study execution, and complete runs are explicitly excluded/included by pilot mode. The full report leads with both-control matched verdict and target comparisons; conditional stress follows the fixed trigger. No confirmatory report exists in the reviewed evidence. |

## Verification actually performed

1. Ran `.venv/bin/python -m unittest discover -s tests -v`: **18 tests passed**, exit code 0. The fresh subprocess and committed-checkpoint resume tests were included.
2. Independently enumerated a random K=3, V=3, L=4 fixture with nonstationary pi. Checked both oracle prefix distributions, matched observed likelihood, gamma, xi, and direct MAP expected-count updates. Maximum oracle discrepancy was **2.7755575615628914e-16**.
3. Checked cross-entropy finite differences for every scalar in both double-precision neural models on a separate short fixture: **280 RNN parameters**, maximum absolute discrepancy **4.2154021029014443e-10**; **712 attention parameters**, maximum **6.335049151928729e-10**.
4. Reimplemented the stratified bootstrap as a separate 5,000-iteration loop and obtained exactly the same percentile endpoints. Checked the regime conjunction and nonmonotonic/left-/right-censored target cases.
5. Verified both pilots' current/frozen source inventory, frozen environment digest, all stage-data digests, preflight-log digest, and all **245 final result-inventory entries per pilot**. Regenerated matrices and all splits/latent paths, compared every checkpoint's model tensors, traces, selections, and saved predictions across pilots, and recomputed selected-checkpoint predictions and KL. All checks passed exactly. Measured times/RSS are not expected to repeat numerically.
6. Independently checked HMM flat-objective stopping, initial-checkpoint ties, and rejection of an objective decrease before commit.
7. Reproduced F1, both F2 cases, and both F3 boundary cases in unique temporary directories under results/. All fixtures were removed. The paused worker was killed and reaped. Two exploratory length-64 checkpoint fixtures stopped on an unmet setup assertion because their best checkpoint was the latest one; the final one-target fixture established the earlier-checkpoint case and reproduced the integrity defect.

No implementation, tests, dependency configuration, or supplied experiment artifacts were changed. No confirmatory fits or outcomes were run. No OCR, installation, Git, or GitHub operations were used.

## Disclosed scope limits that are acceptable for version 1

Known K=4 and matched HMM generators favor the HMM family. The two regimes jointly change persistence and emission informativeness. Four generators per regime give limited population uncertainty, and the bootstrap is descriptive. Tiny fixed neural widths and two learning rates do not establish optimization adequacy or parameter-matched fairness. Validation assistance and unequal validation opportunities prevent a training-data-only sample-complexity theorem. The stress stage tests one restricted second-order mismatch, and its untriggered status after a failed matched advantage is an intended outcome. None of these restrictions requires widening the first harness.

CPU/library overhead and finite-length timing slopes limit compute conclusions. The inability to independently instrument Accelerate's internal thread count and unknown partial-epoch work after external interruption are disclosed measurement limits. Preserve those qualifications rather than presenting interrupted costs as exact. Coordinate counts do not imply byte savings: both the four float64 HMM filter entries and eight float32 RNN state entries use 32 bytes.

## Recommended disposition

Fix F1-F3 without changing the generator family, learners, search grid, thresholds, or statistical decision rules. Add the specified negative/integration tests, rerun correctness and independent-seed pilot reproducibility, and use a new frozen run directory for the changed implementation. A findings-only follow-up can verify these defects and the original review scope. Once they are resolved, the confirmatory study can proceed with its existing bounded interpretation. Its result must still establish the matched advantage and sampled target crossings from the full preregistered evidence.
