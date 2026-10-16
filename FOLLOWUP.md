# F1-F3 are fixed; the full run is ready to launch

Findings-only follow-up, 2026-09-30. This verifies REVIEW.md F1-F3 within the original protocol and implementation scope. It does not reopen the scientific design or constitute another full review.

**All three findings are resolved.** The fresh suite passed 29 tests, separate fixtures rejected the targeted ownership and integrity failures, and two new root-20260931 pilots reproduced their data, model tensors, choices and predictions exactly. No confirmatory candidates or scientific outcomes were run. Pilot results remain pipeline sanity checks, excluded from outcomes and tuning.

## Findings disposition

| Finding | Disposition | Implementation and independent evidence |
| --- | --- | --- |
| F1: surviving worker and simultaneous drivers | Fixed | `artifacts.py:115` provides exclusive run ownership whose descriptor is inherited by the worker. `runner.py:160` dispatches with that descriptor; `fitting.py:275` checks it and persists PID/token/source identity. A separate real-worker fixture stopped after checkpoint 1. While the driver was alive, a second CLI driver exited 1 with `directory is owned by a live driver or worker`. After SIGKILL of driver PID 47152, worker PID 47161 remained alive and the same refusal held. Direct dispatch of both HMM and a different RNN candidate also failed. Continuing the original worker permitted adoption of its completion with exactly one attempt, `complete-recovered`, and unknown driver elapsed time. The worker executed the frozen tree. Both fixture processes have exited. |
| F2: earlier checkpoints and completed selection records | Fixed | `fitting.py:55,76,105` retains original checkpoint digests, verifies all committed checkpoints before continuation, binds identity/record/resume/trace through completion, and checks selected score/index against committed state. The driver commits the completion digest and selection rechecks it in `runner.py:208`. A genuine one-target RNN fixture selected checkpoint 0 with checkpoint 2 as the committed resume point. Changing checkpoint 0 failed with `artifact hash mismatch: checkpoint-000.pt` before completion. Changes to completed score, checkpoint filename and root failed both reuse and selection with `artifact hash mismatch: record.json`. Replacing the completion pointer after rehashing the changed record failed with `artifact hash mismatch: completion.json`. Suite coverage also checks candidate identity, selected index and completion-pointer interruption. |
| F3: original frozen inputs at orchestration boundaries | Fixed | `runner.py:117,129,160,208,284`, `artifacts.py:144`, and `evaluation.py:68` use the recorded inventories and verify provenance through dispatch, selection, evaluation and finalization. Dispatch uses frozen source execution rather than the live tree. Separate post-freeze mutations of training, validation, test, generator, scaling arrays and frozen core/evaluator sources were rejected at all four boundaries. Dispatch launched no subprocess; evaluation constructed no predictor. A copied live-tree mutation was also rejected at all four boundaries. Injecting evaluator-source drift after the last evaluation caused the actual run finalization to raise `frozen source changed: bigbrane/evaluation.py`; no report was written. |

Primary logs are under `results/followup-20260931-independent-c482b1/`: `unit.log`, `independent-recheck.log`, `pilot.log`, `pilot-repeat.log`, `pilot-recheck.log`, and `resume-verified.log`. The independent scripts and persistent fixtures are retained there. Expected negative-case exceptions are successful checks, not failed fits.

## New pilots and resume

Workspace: `/Volumes/XS1000/acoliver/projects/bigbrane`.

- Pilot: `/Volumes/XS1000/acoliver/projects/bigbrane/results/pilot-20260931-followup-c482b1`
- Repeat: `/Volumes/XS1000/acoliver/projects/bigbrane/results/pilot-20260931-followup-c482b1-repeat`
- Resume: the first pilot path above, using `--resume`; no separate resumed run directory.

Both pilot paths were absent before launch. Neither empty `pilot-20260931-fixes-80f2f57c9ca8` directory was used. Both pilot preflights passed 29 tests. Each pilot completed three successful candidate attempts.

Independent hashing verified all 18 source entries against both live and copied sources, seven data entries, 254 final result entries per pilot, environment and preflight digests, and candidate completion/resume inventories. Source and data inventories agree across pilots. Matrices, permutation, every split's tokens and latent arrays regenerate exactly.

All 213 paired checkpoints were compared: 51 HMM, 81 RNN and 81 attention, including initialization. All model and optimizer tensors matched exactly, covering 243,404 tensor scalars. Traces, selected checkpoints, validation scores, counters, saved model/oracle predictions and evaluation summaries matched. Selected predictions were recomputed exactly using the evaluator's batch size. Timing, RSS, PIDs and serialized checkpoint digests need not match.

A completed-run resume returned exit 0 and left all 291 files unchanged, including the manifest and report. It added no attempt or fit. The before-resume inventory and `pilot-verification.json` are retained.

### Verification fixture corrections

Two checker issues were retained and corrected without changing implementation, tests, protocol or thresholds:

1. The first finalization fixture supplied costs for HMM only. It encountered `ValueError: max() iterable argument is empty` before the provenance check. That initial script caught the wrong ValueError. The corrected fixture supplies all three family cost inputs and asserts the exact frozen-source error; `independent-recheck.log` is the accepted evidence.
2. The first prediction checker used a 128-sequence HMM batch and failed its exact assertion at `pilot_checks.py:129`, as recorded in `pilot-verification.log`. Its maximum discrepancy was `8.881784197001252e-16`. `pilot-batch-diagnostic.log` shows that the evaluator's actual batch size of 32 matches exactly. The checker now uses that same batch size and retains the exact assertion; `pilot-recheck.log` passes.

No harness failure remains demonstrated by these checks. No scientific run was launched after either checker issue.

## Commands actually run

All commands were run from the workspace above. Pilot launch logs were redirected to the evidence directory, and the pilot, repeat and initial resume ran sequentially.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m bigbrane.cli --pilot --root 20260931 --output results/pilot-20260931-followup-c482b1
.venv/bin/python -m bigbrane.cli --pilot --root 20260931 --output results/pilot-20260931-followup-c482b1-repeat
.venv/bin/python -m bigbrane.cli --pilot --root 20260931 --output results/pilot-20260931-followup-c482b1 --resume
```

Independent scripts were invoked with the following environment and commands. Their internal subprocess commands are preserved in the scripts. The pilot checker also ran the completed resume with the absolute first-pilot path and captured `resume-verified.log`.

```sh
export PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
.venv/bin/python results/followup-20260931-independent-c482b1/independent_checks.py > results/followup-20260931-independent-c482b1/independent-recheck.log 2>&1
.venv/bin/python results/followup-20260931-independent-c482b1/pilot_checks.py > results/followup-20260931-independent-c482b1/pilot-recheck.log 2>&1
```

The independent script creates new fixture directories and intentionally fails if they already exist. For another execution, label new verification fixture paths rather than overwriting the retained evidence.

## Safe full-run command, not executed

The proposed output path was verified absent. The existing fixed root, candidate grid, thresholds, stress trigger and scientific qualifications remain unchanged. The full run performs its own preflight and freezes current inputs again.

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane
test ! -e results/experiment-001/confirmatory-v1-followup-c482b1 && \
  .venv/bin/python -m bigbrane.cli --root 20260930 \
  --output results/experiment-001/confirmatory-v1-followup-c482b1 \
  > results/followup-20260931-independent-c482b1/full-run.log 2>&1
```

Resume only that same run with:

```sh
.venv/bin/python -m bigbrane.cli --root 20260930 \
  --output results/experiment-001/confirmatory-v1-followup-c482b1 --resume
```

README.md does not describe the new ownership and completion-pointer files. EXPERIMENT.md still has its original documentation-stage status line. These are documentation gaps, not new review blockers; neither was edited. All 18 frozen source/document/test digests remained unchanged at the end of verification. No dependencies, OCR, Git/GitHub operations, or files outside the workspace were used.
