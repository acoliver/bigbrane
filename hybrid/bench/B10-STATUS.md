# B10 status: Phase 10 bench criterion revisions

Decision B10 (DECISIONS.md), 2026-10-16: adopt the criterion revisions proposed in
results/phase10-diag3-20261013T2009Z/summary.md ("Proposed criterion revisions") for criteria 2, 4, 6 and 9, and fix
the acceptance execution policy: pinned one-thread Podman CPU container, seeds 1, 2 and 3; MPS is for diagnostics.
No bench sweep is launched by this work (the Podman slots are in use by a calibration run).

## Progress

| Item | State |
| --- | --- |
| Read summary, B.5, ladder_criteria.py, ladder.toml, bench tests | done |
| Execution policy: `result_trusted`, report `trusted`/`scope`/`policy`, gate refusal, seeds 1,2,3 (`ladder.ACCEPTANCE_SEEDS`) | done |
| Criterion 2: learned K8 control, `unresolved` status | done |
| Criterion 4: receptive field layers x window, state-match ratio | done |
| Criterion 6: empirical part and information-bound part | done |
| Criterion 9: order part (ties) and repeatability part | done |
| Tests (synthetic pass/fail/incomplete/unresolved, stored diag3 check, policy) | done |
| IMPROVEMENT-PLAN.md B.5, BASELINES.md, STATUS.md, README.md | done |
| Bench test run | done: 334 passed, 4 skipped in 159.65 s (`results/b10-20261016/bench-full-1.log`); the skips are the Podman and MPS tests that need `BIGBRANE_BENCH_PODMAN=1` or Apple MPS |

No bench sweep was launched. No `rung1.json` exists, so rung 1 stays uncalibrated.

## Predeclared thresholds (hybrid/bench/ladder_criteria.py, stored in every report as `policy`)

| Criterion | Rule |
| --- | --- |
| 2 | Low-K control: K8 accuracy >= 0.25 (twice the 0.125 reached by guessing one of 8 stored values), per seed. Fixed-state models then need a K8 to K64 drop >= 0.2, attention <= 0.05. A model that misses the control on any seed, with no seed failing the drop, is `unresolved`; a learned seed that misses the drop is `fail`. |
| 4 | Window limit = layers x window = 64 tokens. Accuracy on queries wholly beyond 64 tokens (bucket starts at distance 65 or more) <= 0.1 for the window; the better rival above it on every seed; rival counted state (conv buffers included) within 1.25x of the window's. A bucket that straddles the field is excluded. No bucket beyond the field: `incomplete`. |
| 6 | Empirical part: transformer string accuracy at 2 x L_train above every fixed-state model on every seed, plus prefix-key lookup >= 0.9. Bound part: payload L log2(D) against counted FP32 state bits; tested only where the cap 2^(state bits - payload bits) is <= 0.05, otherwise `info`. Overall = combination, `info` does not block. |
| 9 | Order part: pairs with mean gap > 0.1 keep their order on every seed that is not a tie; a tie is a difference <= 0.01 or both models >= 0.95. Large reversals fail. Repeatability part: a cell whose seeds spread > 0.5 on some set needs a same-seed repeat attempt (model differing only by `attempt`, same lr) within 0.1 on every set; none = `incomplete`, disagreement = `fail`. Overall = combination. |
| Execution | Gate requires report `trusted: true` (all results ran in the container on CPU), `policy.seeds == [1, 2, 3]`, schema `bigbrane.ladder_calibration/2`. Reports holding MPS results get `scope: diagnostic` and are refused. |

## Stored diag3 results under the revised criteria (read only, same hash 8e4c6caba4c7)

See the stored-result test in `hybrid/tests/bench/test_ladder_calibration.py`. Criterion 2: LA 3/3 pass, GDN d32 seed 2 pass
(seeds 1 and 3 never learn), Mamba s16 d32 seed 3 pass (seeds 1 and 2 never learn), attention d32 from diag2 3/3 pass.
That is the summary's 5 of 9 passing checks; the four others become `unresolved` instead of failures. Criterion 9: the
same three eligible cross-family comparisons reverse order (GDN/LA at K4 and K16, LA/Mamba at K4), so order still fails.

## Assertions changed

| Test | Change | Reason |
| --- | --- | --- |
| `test_first_slice_settings_are_the_plain_config` | seeds `[0, 1, 2]` to `[0, 1, 2, 3]` | B10 acceptance seeds are 1, 2, 3 after the seed 0 sweep; `DEFAULT_SEEDS` follows the policy. |
| `test_gate_command_exit_status` | A report with devices cpu and mps no longer exits 0 | B10: the gate refuses MPS/diagnostic reports. Strengthened, not weakened. |
| `test_criterion_2_drop_rules`, `test_criterion_4_reads_distance_buckets`, `test_criterion_9_flags_a_seed_that_reverses_a_clear_ordering` | Rewritten for seeds 1, 2, 3 and the revised rules; the old pass/fail cases are kept | A model with K8 and K64 both near zero used to pass or fail the drop rules on noise; a 33-64 token bucket used to count as beyond a 32 window. |

## Notes for the acceptance sweep

- Slices `first` and `second` now default to seeds 0, 1, 2, 3 (`ladder.DEFAULT_SEEDS`). Run ids do not hash the seed list, so no run id changed.
- Update: slice `accept` (B10-SLICE-STATUS.md) now holds the repeat attempts and the long-copy design described in the next line, with the job count, estimate and launch command.
- `first` does not yet contain the criterion 9 repeat attempts (cells with unstable onset need a model that differs only by `attempt`, same learning rate) or the criterion 6 long-copy design; an acceptance sweep that has unstable cells and no repeats ends `incomplete` on criterion 9.
- The tie margin 0.01, ceiling 0.95, unstable spread 0.5, repeat margin 0.1, K8 control 0.25 and state ratio 1.25 are choices made here before any acceptance data exists. They were not tuned to make the diag3 results pass.
