# Bigbrane experiment harness

The harness implements version 1 of [EXPERIMENT.md](EXPERIMENT.md). It fits a four-state MAP HMM, a 280-parameter tanh RNN, and a 712-parameter causal attention model. The true generator is available only to evaluation. No confirmatory study is included with the implementation.

## Local setup and correctness

Use Python 3.11 or newer with a compatible PyTorch wheel. This workstation uses Python 3.12 in the workspace-local environment. Install the declared neural extra without changing system packages:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[neural]'
.venv/bin/python -m unittest discover -s tests -v
```

The suite checks exhaustive independent latent-path fixtures for both oracles, MAP updates and objective monotonicity, shifted inputs, matrix equations and gradients, causality and state resets, parameter counts, seed pairing, selection ties, learner artifact access, candidate subprocesses, checkpoint resume, and resource accounting. A run performs the suite again before freezing its inputs. Numerical failures stop the run instead of substituting a new seed or omitting a candidate.

## Pilot, full run, and resume

The pilot uses root `20260931`, one generator, 512 training tokens, and one candidate per family. It uses the specified training caps and the full 2,048-token validation and 8,192-token test splits. Its losses are a pipeline check, excluded from study outcomes and from tuning decisions.

```sh
.venv/bin/python -m bigbrane.cli --pilot --root 20260931 --output results/pilot-check
.venv/bin/python -m bigbrane.cli --pilot --root 20260931 --output results/pilot-check --resume
```

To start the confirmatory study, use a new output directory. It schedules all 384 matched candidates. The 192 stress candidates run only if the prespecified matched advantage passes against both controls. Candidates run sequentially on CPU, each in a fresh subprocess with one configured backend thread. Training and validation seeds follow the protocol tuples; learning-rate pairs share initialization and epoch permutations.

```sh
.venv/bin/python -m bigbrane.cli --root 20260930 --output results/experiment-001/confirmatory-v1
.venv/bin/python -m bigbrane.cli --root 20260930 --output results/experiment-001/confirmatory-v1 --resume
```

`--prepare-only` freezes sources, environment, matrices, data arrays, and seed plans without fitting or evaluating outcomes. Continue the same directory with `--resume`. Roots other than the two specified roots are rejected. Existing runs require `--resume`, and changed sources, protocol documents, dependencies, or frozen artifact hashes fail verification. Start a separately labeled run when these inputs change.

Completed candidates are reused. Interrupted candidates resume the last committed EM iteration or neural epoch, including optimizer state and the separate stopping-progress record. All checkpoints are retained. Selection uses raw validation NLL only, with the specified earlier-checkpoint and candidate tie rules. Every selection in a stage is frozen before that stage's test scoring begins. Learner workers receive only observed training and validation arrays; generator matrices, latent paths, test arrays, and scaling arrays remain evaluator artifacts. This is an audited code boundary, not an OS security sandbox.

## Artifacts and interpretation

`manifest.json` records copied-source hashes, an environment hash, versions and backend settings, integer seed tuples, split hashes, all candidate plans and process attempts, completion state, and final result hashes. `frozen/` contains the source and environment snapshots. Each generator's `data/` directory separates `learner/` from `evaluator/` storage.

`candidates/` contains worker jobs and logs, per-iteration or per-epoch checkpoints, objective and NLL traces, resume pointers, and candidate resource records. `selections-stage-*.json` freezes chosen checkpoint hashes and includes both total search cost and retained-candidate cost. `evaluations/` contains raw model log probabilities, oracle distributions, KL and sampled NLL summaries, initial/noninitial and sixteen-target blocks, per-sequence records, unigram diagnostics, and batch-one/batch-32 scoring benchmarks. Selected largest-size matched models from replica zero also receive the specified evaluator-only length-scaling measurements. `report.json` records decisions, generator-level deltas, descriptive bootstrap intervals, learning curves, target crossings, stress status, and resource totals.

Candidate process RSS includes Python and library overhead. A separate pre-model RSS high-water measurement and actual parameter/optimizer tensor bytes prevent treating that overhead as model storage. Training diagnostics, validation, checkpoint I/O, and total candidate-process wall time are recorded separately. Optimizer/EM token exposures exclude separately counted diagnostic passes. Inference measurements use two warmup and ten timed passes, excluding file I/O and model construction. Analytical complexity and state-storage estimates are labeled separately from measurements.

An externally killed worker can consume part of an uncommitted epoch. Its process-attempt wall time is retained when the driver observes the exit, but exact partial-epoch exposure counters are unavailable. A killed driver can also lose the elapsed time for its outstanding attempt. Such attempts remain explicitly marked as unknown instead of being omitted. CPU backend environment settings and Torch's thread settings are recorded; the harness does not independently instrument Accelerate's internal thread count.

The report concerns the specified generators and optimization budgets. It does not establish optimization adequacy, width or search-budget sensitivity, unknown-state-count learning, broader sequence distributions, or retrieval/composition transfer. Sample counts include the shared validation access. The bootstrap is descriptive with four generators per regime. Measured finite-length rates do not prove asymptotic complexity.
