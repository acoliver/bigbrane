# B10 acceptance slice status

Decision B10t (DECISIONS.md, 2026-10-16) fixes the B10 thresholds before any acceptance data and asks for two things in
the acceptance slice: same-seed repeat attempts for every cell, and a long-copy length at which the criterion 6 bound
is tested. This session builds that slice. **No sweep was run** and `hybrid/bench/calibration/rung1.json` still does
not exist, so rung 1 is uncalibrated.

## Result in one paragraph

The slice is `[slices.accept]` in `hybrid/bench/configs/ladder.toml`: 207 jobs (21 cells, 21 repeats, 3 long-copy
cells and their 3 repeats). The cost is about 358,000 job-seconds, which on 4 Podman slots is **about 25 hours**
using the rates measured for these cells under six-way contention, and about 14 hours if 4-way running is as fast as
the isolated microbenchmark. The long-copy length is L = 76,000, which makes the criterion 6 bound testable
(cap 2^-896) and is practical. The bench tests pass (numbers below).

## Progress

| Item | State |
| --- | --- |
| 1. Repeat attempts for every cell, run ids, ranking | done: `lr_from` cells in `ladder.py`, 11 `*-rep1` models, 27 follower cells in `accept` |
| 2. Long-copy length for criterion 6 | done: task `copy_long`, L = 76,000, bound part now `pass`/`fail`/`incomplete` instead of `info` |
| 3. Job count and runtime estimate | done: `hybrid/bench/ladder_estimate.py`, numbers below |
| 4. Tests | done: `hybrid/tests/bench/test_ladder_acceptance_slice.py` (new), `test_ladder_calibration.py` (criterion 6 and 9 additions) |
| 5. Launch command | below; not launched |

## What changed

| File | Change |
| --- | --- |
| `hybrid/bench/ladder.py` | A cell may name `lr_from = "task/model"`: no `lrs`, it runs at the rate chosen for that cell of the same slice, queued right behind the original's own post-selection seeds (`Scheduler._follower_runs`). `load_config` rejects an `lr_from` that has `lrs` or names no cell with its own rates (no chains). No change to runner, baselines or tasks, so the code hash is unchanged. |
| `hybrid/bench/ladder_criteria.py` | Criterion 6 bound part reads the long copy task `copy_long` (`LONG_COPY_TASK`) beside `copy`: `copy_length_tasks`, `counted_state_bits`, `bound_testable_length`. A testable length with missing runs is `incomplete`; with no long copy task configured the part is still `info`. `policy()` records `long_copy_task`. Thresholds are unchanged. |
| `hybrid/bench/ladder_estimate.py` | New. Plays a slice forward with the real `Scheduler` and measured durations to count jobs and estimate elapsed time. |
| `hybrid/bench/configs/ladder.toml` | New task `copy_long`, 11 `*-rep1` repeat models, new slice `accept`. `first`, `second` and the diagnostic slices are untouched. |

## 1. Repeat attempts (criterion 9)

- Every cell of `accept` has a same-seed repeat: a model that repeats its original's family, width, depth and
  hyperparameters and differs only by `attempt = "acc-rep1"`. The job spec is the original's (a test compares the
  built specs), so weights and data streams are the same; only the run id differs. 11 repeat models cover the 21 cells
  and the 3 long-copy cells (a repeat model serves every task its base runs on).
- A repeat runs on seeds 1, 2 and 3 at the learning rate chosen for its original on seed 0, so a repeat is never at a
  different rate (criterion 9 skips a repeat at another rate and would leave the cell `incomplete`). Seed 0 is not
  repeated; it only sweeps the rate. All three seeds are repeated because which seeds spread is not known in advance.
- Run ids are `r-` plus a hash of the key, which includes the model name, and of the model table, which includes
  `attempt`. A test checks that all 207 run ids of the slice are distinct, that every swept rate has its own id, and
  that an original and its repeat differ only by id.
- Ranking: `ladder_criteria._is_repeat` (a model with `attempt`) excludes a repeat from criterion 3, from the
  criterion 9 order part and from the unstable-cell scan; `_repeat_part` pairs it with the original of the same
  family, width, depth and hyperparameters at the same rate. Verified by reading the code and by a new test that gives
  the `*-rep1` models results that would reverse an order and shows no pair or violation is added.
- Cost: the 21 cell repeats and the long-copy repeats are 72 of the 207 jobs. Repeating one seed per cell would save
  48 jobs (two thirds of them), but then which seed is repeated decides what is checked.

## 2. Long copy length (criterion 6 bound)

Counted state bits at the current grid (`ladder_criteria.counted_state_bits`, FP32): linear attention d64 303,104;
Gated DeltaNet d64 299,008; Mamba-2 s32 d64 299,008. At D = 16 a symbol carries 4 bits. The cap 2^(S - P) is at most
0.05 once P exceeds S by log2(20) = 4.32 bits: L >= 75,778 (linear attention), 74,754 (the two others).

**L = 76,000** (payload 304,000 bits, 896 above the largest state, cap 2^-896). It is practical, so no small-width
fallback check was added:

| Measure (host, one thread, random-init two-layer d64 model, one 152,003-token row) | Linear attention | Gated DeltaNet | Mamba-2 s32 |
| --- | --- | --- | --- |
| Seconds per row | 1.89 | 1.42 | 1.78 |
| Peak resident MB | 1,126 | 1,217 | 1,762 |

Log: `results/b10-slice-20261016/longcopy-probe-75000-isolated.log`. 128 rows take about 3 to 4 minutes per job; the
prediction file is 128 x 76,000 int16 = 19 MB, under the 64 MB job output cap. Because Mamba-2 peaks near 1.8 GB and the
default job cap is 2,000 MB, the slice raises `memory_mb` to 4,096 (still one executor slot, `ceil(4096 / 4096)`).

`copy_long` trains exactly what `copy` trains (same train distribution, vocabulary, model and data seeds; a test compares
the built specs) and tests only at L = 76,000. It runs the three fixed-state copy models at the learning rate chosen on
`copy`, seeds 1, 2 and 3. Transformers are not run on it (quadratic attention at 152,003 tokens). The empirical 2 x L_train
part is unchanged. A deterministic CPU run should give the same weights as the `copy` job, but that is expected, not
checked, until the sweep exists.

Not verified: the Mamba-2 long row inside the container at the 4,096 MB cap (the figures above are host runs). The
first `copy_long` job of the sweep will show it; a memory kill would appear in `failures-accept.json`.

## 3. Job count and runtime estimate

**207 jobs**, counted by playing the real `Scheduler` forward (`python -m hybrid.bench.ladder_estimate`, output in
`results/b10-slice-20261016/estimate-accept-w4.json`) and by a test:

| Part | Cells | Jobs |
| --- | --- | --- |
| Cells (3 rates on seed 0, then seeds 1 to 3): mqar512 x 7, mqar256 x 5, copy x 5, lookup_prefix x 4 | 21 | 126 |
| Repeats of those cells (seeds 1 to 3) | 21 | 63 |
| Long copy (`copy_long`, seeds 1 to 3) and its repeats | 3 + 3 | 18 |
| Total | 48 | 207 |

Cells beyond `first` are the three mqar512 sizes criterion 3 needs (la-d64, gdn-d64, m2-s64-d64 beside the d32 models;
two state sizes per family). The d128 cells of `second` are left out: criterion 3 asks for two sizes, and by
extrapolation a d128 job at N 512 would pass the 9,000 s training budget.

**Basis.** Mean wall seconds per job of the same (task, model) cell in `results/phase10-20261012` (the `first` slice:
6,000 steps for mqar and copy, 4,000 for lookup, one thread per job in the pinned image, six workers on the 8-CPU Podman
VM, early stops included). This is the only measured one-thread Podman CPU run of these cells. The diag3 slice
(`results/phase10-diag3-20261013T2009Z`) was host MPS, so its seconds per step do not apply; from it I took only the
occupied-slot ratio, 0.9786 (63,426 job-seconds over 3 workers in 21,604 s), to cover launch, staging and tail gaps.

| Cell | Measured mean wall s | | Cell | Measured mean wall s |
| --- | --- | --- | --- | --- |
| mqar256 gdn-d64 | 2,955 | | copy gdn-d64 | 559 |
| mqar256 m2-s32-d64 | 3,002 | | copy m2-s32-d64 | 683 |
| mqar256 la-d64 | 2,262 | | copy la-d64 | 511 |
| mqar256 swa-w32-d64 | 1,810 | | copy tf-halibi-d64 | 284 |
| mqar256 tf-d64 | 1,147 | | copy tf-d64 | 170 |
| mqar512 gdn-d32 | 2,128 | | lookup gdn / m2 / la / tf | 132 / 163 / 155 / 98 |
| mqar512 m2-s16-d32 | 1,981 | | | |
| mqar512 la-d32 | 1,715 | | | |
| mqar512 tf-d32 | 1,694 | | | |

Derived, never measured: mqar512 gdn-d64, la-d64 and m2-s64-d64 are 2 x the mqar256 rate of the same family (N 512
carries twice the tokens per step; m2-s64 is costed as m2-s32, so it is a lower bound): 5,910, 4,524 and 6,004 s. Each
`copy_long` job is its `copy` cell plus 128 rows x 1.89 s x 1.5 container allowance = 363 s.

| Quantity | Value |
| --- | --- |
| Job-seconds | 358,017 (mqar512 215,595; mqar256 100,581; copy 19,859; copy_long 17,050; lookup 4,932) |
| Simulated elapsed on 4 workers, with the sweep-then-seeds dependencies | 89,548 s |
| Divided by the diag3 occupied-slot ratio 0.9786 | **91,504 s, about 25.4 hours** |
| If 4-way running matches the isolated microbenchmark (step times 0.54 to 0.60 of the six-way run for linear attention, Gated DeltaNet, Mamba-2 and the window) | about 51,500 s, 14.3 hours |

I take 25 hours as the planning figure and 14 hours as the floor: the microbenchmark excludes data generation and
evaluation, and four jobs may contend less than six. Wall time is dominated by mqar512 (60 percent of job-seconds). The
longest single job is about 6,000 s, inside the 9,000 s training budget and the 10,800 s wall cap; a job that hits the
budget records `stopped` in `train.json`, so check it after the sweep.

## 4. Tests

`.venv-hybrid/bin/python -m pytest hybrid/tests/bench -q -p no:cacheprovider`, run with nohup, polled every 60 s or less.

| Run | Result | Log |
| --- | --- | --- |
| Full bench suite without Podman or MPS | **362 passed, 4 skipped in 117.0 s** (the 4 skips are the Podman and MPS tests that need `BIGBRANE_BENCH_PODMAN=1` or Apple MPS; the previous session's run was 334 passed, 4 skipped) | `results/b10-slice-20261016/bench-full-1.log` |
| Slice and calibration files while developing | 112 passed, 1 failed (the criterion 6 `info` expectation changed under Assertions changed; the full run above is after that edit) | `results/b10-slice-20261016/pytest-slice-2.log` |


New tests (`test_ladder_acceptance_slice.py`): the slice builds on the acceptance seeds in the container with one slot
per job; its cells cover what criteria 1 to 4, 6, 8 and 9 read; 207 jobs; a repeat exists for each of the 21 cells and
the 3 long-copy cells, on seeds 1 to 3 at the original's chosen rate and queued behind its seeds; repeat models differ
from their base only by `attempt` and build the same job spec; all run ids are unique; `lr_from` validation; the long
copy length makes the cap at most 0.05 for every fixed-state copy model with a 896-bit margin and fits the output cap;
the estimate counts the Scheduler's jobs and costs unmeasured cells by the stated rules.
In `test_ladder_calibration.py`: criterion 6 now passes, fails, or is incomplete on the `copy_long` results, reports
`info` only without a long copy task, and criterion 9 is shown to compare `*-rep1` models with their original instead
of ranking them.

### Assertions changed

| Test | Change | Reason |
| --- | --- | --- |
| `test_criterion_6_passes_on_the_empirical_part_and_reports_the_unreached_bound` | Replaced by a pass test on `copy_long` results plus a separate `info` test on a config without the long copy task | B10t: with the shipped config the bound is tested, not reported |
| `test_criterion_6_empirical_part_fails_when_copy_does_not_separate_the_families` | Bound part expected `pass` instead of `info` | Same |
| Two bound-length tests | Run on a config copy without `copy_long` | They exercise a longer `copy` set and would otherwise also need long-copy results |

## 5. Launch command (not launched)

Prerequisites: the Podman VM is running with 4 free job slots (a calibration run held them at the time of writing; check
`ls var/slots` and `podman ps`), the executor image of `ladder.toml` is present, the disk guard has room (the long
evaluation set adds about 250 MB under `<out>/data/copy_long`).

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane && test ! -e results/phase10-accept-20261016a && mkdir -p results/phase10-accept-20261016a && nohup .venv-hybrid/bin/python -m hybrid.bench.ladder run --config hybrid/bench/configs/ladder.toml --slice accept --out results/phase10-accept-20261016a --workers 4 > results/phase10-accept-20261016a/slice-accept.log 2>&1 &
```

- Trusted Podman CPU: `accept` names no `execution`, so every job runs in the pinned image on CPU
  (`execution = "container"`, one thread); the report will carry `trusted: true`.
- Seeds 0 to 3: the slice default, `ladder.DEFAULT_SEEDS`; repeat cells list seeds 1, 2, 3. The CLI has no seeds flag.
- `--workers 4` is one worker per executor slot (`[math] max_parallel_jobs = "auto"` gives 4 on the 8-CPU VM). The run is
  resumable: rerun the same command and finished runs are skipped.
- Unique results dir: `results/phase10-accept-20261016a` (the command refuses an existing one). Pick another suffix for a
  second attempt.

After the sweep, on a quiet machine (the timings are the point):

```sh
.venv-hybrid/bin/python -m hybrid.bench.ladder microbench --config hybrid/bench/configs/ladder.toml --out results/phase10-accept-20261016a --threads 1
.venv-hybrid/bin/python -m hybrid.bench.ladder report --config hybrid/bench/configs/ladder.toml --out results/phase10-accept-20261016a
```

Criterion 7 reads the microbenchmark, so `report` needs it. Store the report as `hybrid/bench/calibration/rung1.json`
only if the whole directory is one code hash (`report` refuses a mixed one); then `python -m hybrid.bench.ladder gate`.

## Open points

- The acceptance outcome is not predicted. Earlier diagnostics (`results/phase10-diag3-20261013T2009Z/summary.md`) found
  unstable learning onset for Gated DeltaNet and Mamba-2 at these budgets, on MPS; B10 makes that `unresolved` for
  criterion 2, and the repeats now test whether CPU runs reproduce. Thresholds are not changed to fit.
- IMPROVEMENT-PLAN.md B.5 item 6 still says the bound part is `info` until a long-copy design exists. It sits outside
  the files this session may change; it should now say the bound is tested at L = 76,000 on the `copy_long` task.
- The Mamba-2 long row inside the container at 4,096 MB is untested (see section 2).
