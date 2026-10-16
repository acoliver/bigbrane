# Math tools: Lean, Z3, SymPy and the math library in the tools image

Phases 1 and 2 of [DISCOVERY-PIPELINE-PLAN.md](../../DISCOVERY-PIPELINE-PLAN.md). Phase 1 (this section through "Limits") is the tools image and its Lean, Z3 and SymPy checks. Phase 2 (from "Phase 2: the math library" on) is the library of conservation laws, the cost verifier, the construction harness, the magic-X harness and the check-hash manifest.

Phase 1. The tools image runs Lean 4 with Mathlib, z3-solver, SymPy and NumPy offline, through the same executor job API as every other Bigbrane job (fresh container, `--network=none`, read-only root, uid 10001, caps from the job). Results and logs of the build and smoke runs: `results/phase1-20261012T172040Z/` (start with `summary.md`).

## The image

| Item | Value |
| --- | --- |
| Reference | `localhost/bigbrane/tools@sha256:511da9751d13d9f2703ce41060932edebbc940a3156ca0dd5a3f121b35761ef7` (tag `localhost/bigbrane/tools:phase1-20261012T172040Z`, id `6e1a4b0613b3`), recorded in `hybrid/executor/image.tools.lock.json` |
| Labels | `bigbrane.owner=bigbrane`, `bigbrane.run=phase1-20261012T172040Z`, `bigbrane.image=tools` |
| Base | `python:3.12-slim-bookworm` pinned by its linux/arm64 digest `sha256:739ba32a...`, the same base as the executor image (glibc 2.36) |
| Lean | v4.34.1 (`lean-4.34.1-linux_aarch64.tar.zst`, sha256 `fdb974c2...de9e37` checked at build). This is the latest stable release; the aarch64 `leantar` fix (lean4 PR 13499) merged on 2026-04-22 and every stable release from v4.30.0 carries it. The build fails unless `lean`, `lake` and the bundled `leantar` are aarch64 ELF files, and all three are. |
| Mathlib | tag v4.34.1, commit `d13f23b723b8a846827a245b89c10fc7d3f11612`, in the Lake project `/opt/mathproj` (`hybrid/executor/tools/mathproj/`). Dependencies resolved by `lake update` from Mathlib's own manifest: batteries `f2effa3d`, aesop `355695d5`, Qq `6a489d9a`, proofwidgets `106ff4fa`, plausible `118aa17e`, importGraph `e928b725`, LeanSearchClient `ddf04cf3`, Cli `e92c9f15`. |
| Mathlib cache | fetched by `lake exe cache get` at build time (8908 files). `lake build --no-build Mathlib` then reported all 8923 jobs up to date, and a file importing Mathlib elaborated, so the prebuilt Linux arm64 cache loads with no rebuild. Package git checkouts and the downloaded archives are deleted after that check. |
| Python | z3-solver 4.15.4.0 (manylinux_2_34 aarch64 wheel; 4.15.5 and later need glibc 2.38), SymPy 1.14.0, NumPy 2.5.3, mpmath 1.3.0, the same versions as the executor image, installed with `--require-hashes` from `hybrid/executor/requirements.tools.txt` |
| Sizes | image 10.88 GB (10,882,644,629 bytes); Mathlib build output 6.70 GB; all Lake packages 7.15 GB; Lean toolchain 3.12 GB (from `/opt/bigbrane/tools.json` in the image) |

Build it (checks free space first, then labels the build and writes the lock):

```sh
.venv-hybrid/bin/python -m hybrid.executor.build tools --run phase1-<UTC stamp>
```

The build needs the network. Jobs never get it.

## Lean checks

A Lean obligation has two parts, following plan section 4.1:

- the **statement** file: imports and `def Claim : Prop := ...`. It is frozen before any prover call, and its sha256 after whitespace normalization (every run of whitespace becomes one space) is fixed then;
- the **proof**: `theorem claim_holds : Claim := ...`, plus helper lemmas if needed, with no imports.

```python
from hybrid.executor.executor import Executor
from hybrid.executor.lock import read_lock
from hybrid.math.lean_check import LeanCheck, statement_hash
from hybrid.math.runner import MathRunner

statement = "import Mathlib\n\ndef Claim : Prop := ∀ a b : ℝ, 2 * a * b ≤ a ^ 2 + b ^ 2\n"
frozen_hash = statement_hash(statement)          # recorded when the statement is frozen

runner = MathRunner(Executor("run-x", "runs/run-x/jobs/receipts"), read_lock("tools")["reference"], "runs/run-x")
result = runner.check_lean(LeanCheck(statement, frozen_hash,
                                     "theorem claim_holds : Claim := by\n  intro a b\n  nlinarith [sq_nonneg (a - b)]\n"))
print(result.status, result.decided, result.axioms, result.reasons)
# pass True ['Classical.choice', 'Quot.sound', 'propext'] []
```

What `check_lean` does:

1. On the host, before anything runs, it rejects the check on a statement hash mismatch (a restated statement file), or when the source scan finds a construct that `#print axioms` cannot see: `unsafe`, `implemented_by`, `extern`, `export`, `#eval`, `#exit`, `#reduce`, `run_cmd` and the other `run_*` commands, `elab`, `macro`, `syntax`, `notation` and the infix forms, `initialize`, `namespace`, `section`, `end`, `import` in the proof, or a `set_option` outside the allowlist (`maxHeartbeats`, `maxRecDepth`, `synthInstance.*` limits, `autoImplicit`, `linter.*`, `pp.*`). Comments and string literals are ignored by the scan.
2. It writes one file, `import Lean`, then the statement, the proof and a trailer the checker writes with a fresh 128-bit nonce. The trailer declares `theorem bigbrane_check_<nonce> : Claim := claim_holds` (a kernel re-check under a name the prover cannot know), runs `#print axioms claim_holds`, and prints the declaration kind of `claim_holds`, whether its elaborated type is exactly the constant `Claim`, whether `Claim` is a definition of type `Prop`, and the axioms from the environment, all tagged with the nonce.
3. It runs that file with `lean-check /in/Check.lean` (the image's `lean` with Mathlib on `LEAN_PATH`) in a fresh container, twice, each run with its own nonce.
4. A run passes only if Lean exits 0 with no error, `claim_holds` is a theorem, its type is exactly `Claim`, the `#print axioms` list equals the environment's list, and every axiom is `propext`, `Classical.choice` or `Quot.sound`. `sorryAx`, native evaluation (`Lean.ofReduceBool`, `Lean.trustCompiler`, or the auxiliary axiom Lean 4.34 creates for `native_decide`, for example `claim_holds._native.native_decide.ax_1_1`) and any user axiom reject, each with its own reason. The result is decided only when both runs agree; an executor failure or a disagreement gives `unknown`.

`LeanResult` holds `status` (`pass`, `reject`, `unknown`), `decided`, `reasons`, the statement hash, the axioms and both runs (job id, nonce, exit code, check time, parsed facts, output tail). Every job also leaves its receipt under the executor's receipts directory and its staged `Check.lean` under `<work_dir>/jobs/<job id>/in/`.

Caps: `LEAN_CAPS` is 900 s wall, 8 GB memory, 512 pids, 1 CPU, 64 MB `/tmp`. Mathlib's `.olean` files are memory-mapped from the read-only image.

## Z3 and SymPy programs

`runner.check_program(tool, program)` with `tool` `"z3"` or `"sympy"` runs `python3 /in/main.py` under `PROGRAM_CAPS` (300 s, 2 GB, 64 pids, 1 CPU) and applies the obligation-check contract and replay rule from `hybrid/research/stages/obligation_checks.py` unchanged: the last stdout line is `{"status": "discharged" | "violated" | "unknown", "evidence": str, "counterexample": ...}`, a violation needs a counterexample, and a deciding first run is replayed and counts only when the replay prints the same status and counterexample. For a bounded Z3 claim, assert the negation within the bound: `unsat` is `discharged` (proven up to the bound, which the evidence states), and `sat` is `violated` with the model as the counterexample.

## Smoke tests

```sh
.venv-hybrid/bin/python -m hybrid.math.smoke --out results/phase1-<stamp>/smoke<n>
BIGBRANE_TOOLS_PODMAN=1 .venv-hybrid/bin/python -m pytest hybrid/tests/math/test_tools_image_live.py -q -p no:cacheprovider
```

Both were run against the image above, and every case gave its expected result in both:

| Case | Result | Check time per run (s) |
| --- | --- | --- |
| small lemma, `∀ n : Nat, n + 0 = n` | pass, no axioms | 0.98, 0.89 |
| Mathlib lemma, `∀ a b : ℝ, 2 * a * b ≤ a ^ 2 + b ^ 2` by `nlinarith` | pass, axioms propext, Classical.choice, Quot.sound | 3.34, 3.27 |
| proof by `sorry` | reject: sorryAx | 0.83, 0.83 |
| user axiom | reject: non-standard axiom `bigbrane_magic` | 0.83, 0.82 |
| `native_decide` | reject: native evaluation (`claim_holds._native.native_decide.ax_1_1`) | 0.82, 0.83 |
| proof restates the claim as a weaker proposition | reject: type mismatch, type is not exactly `Claim` | 0.82, 0.83 |
| statement file changed after its hash was fixed | reject on the host, no job | none |
| Z3 satisfiable system | discharged, `sat: x=7 y=3`, replayed | 0.23 to 0.28 |
| Z3 bounded claim `x*x != 2*x + 3` on [0, 1000] | violated, counterexample `{"x": 3}`, replayed | 0.23 to 0.28 |
| SymPy identity `(x+1)^3`, symbolic plus 20 rational spot checks | discharged, replayed | 0.50 to 0.56 |

Check time is the container's run time from the receipt. With staging in and out, a Lean check takes 1.8 to 2.2 s of wall time per run for a file without Mathlib and about 4.4 s for a file importing Mathlib. Every job ran with `network_mode` `none`, and no labeled container or volume was left behind.

## Tests

- `hybrid/tests/math/test_lean_check.py`: hashing, the source scan, assembly and verdicts on Lean output copied from the Podman runs.
- `hybrid/tests/math/test_runner.py`: the runner with a fake executor (staging, replay, host-side rejection, program checks).
- `hybrid/tests/math/test_tools_image_live.py`: all smoke cases in Podman; skipped unless `BIGBRANE_TOOLS_PODMAN=1`.

## Limits

- The source scan is a denylist over tokens. It covers the constructs the plan names and those that change name resolution in the trailer; a construct outside the list that weakens checking without showing in `#print axioms` would get through. The kernel re-check and the exact-type check do not depend on the scan.
- An external replay checker (`lean4checker` style) is not run.
- The statement file is trusted once its hash matches. It is scanned, but nothing checks that it says what the card means; that is the statement-match role of Phase 3.

## Phase 2: the math library

Plan sections 4.4 (P1, P2, P5), 5.1, 5.2, 5.3 and 5.5. Results and logs: `results/phase2-20261012T180235Z/` (start with `summary.md`).

### Files

| File | What it holds |
| --- | --- |
| `conservation.py` | the conservation laws, one function per law, `check_claims` for a card's numbers, `report`, `find_state_collision` |
| `cost.py` | the op counter (`OpCounter`, `Counted`), the line tracer and `verify_cost` |
| `constructions.py` | generators, solvers and the harness for copy, recall, Dyck-2, composition and variable binding |
| `magic_x.py` | the magic-X harness and its n-gram baseline |
| `tiers.py` | the tool tiers of section 4.2 and the words a verdict may use for each |
| `references.py` | reference candidates with hand-derived cost formulas, the planted errors, hand-set constructions and the magic-X examples |
| `selfcheck.py` | the battery of known numbers and planted errors, run on the host and in the tools image |
| `jobmain.py` | the driver staged as `/in/main.py` in a library job |
| `architecture.py` | the shared architecture contract of P1, P2 and P4 (FIX-4): width formulas, the array survey and its violations; imported lazily, only by requests that carry a contract |
| `findings.py` | the host run of a stage C finding's `verify(witness)` code against the recorded sources (FIX-4); imported only by `finding` requests |
| `manifest.py`, `MANIFEST.json` | the check-hash manifest |
| `library_smoke.py` | the live run in Podman |

The check modules use only the standard library, NumPy and SymPy, and import each other relatively, so the same files run on the host and, staged as the package `bbmath`, in the tools image. A test checks the imports.

### Running a check in the tools image

```python
from hybrid.executor.executor import Executor
from hybrid.executor.lock import read_lock
from hybrid.math.runner import MathRunner

runner = MathRunner(Executor("run-x", "runs/run-x/jobs/receipts"), read_lock("tools")["reference"], "runs/run-x")
record = runner.check_library({"kind": "conservation",
                               "claims": {"state_bits": 256, "recall": [{"K": 64, "V": 256, "accuracy": 0.99}]}})
print(record["status"], record["counterexample"]["smallest_failing_K"], record["check_hashes"]["bbmath/conservation.py"])
# violated 33 49e58e2f...
```

`check_library(request)` does the following:

1. It verifies every file under `hybrid/math` against `MANIFEST.json` and raises `ManifestMismatch` before anything is staged when a file changed, disappeared or is not listed.
2. It stages the driver as `main.py`, the check modules as `bbmath/`, a `manifest.json` with the verified hashes of the staged files, and `request.json`.
3. The driver re-hashes every staged file in the container and refuses to import anything on a difference (status `unknown`, exit 2).
4. It runs the request under `LIBRARY_CAPS` (1800 s, 4 GB, 64 pids, 1 CPU) with `--network=none`, and applies the obligation-check contract and replay rule of `check_program`: a `discharged` or `violated` result counts only when a second run prints the same status and counterexample.
5. The record carries `check_hashes`, the sha256 of every staged check file, as `falsifier.code_hashes` does for falsifiers.

Request kinds: `conservation` {claims}; `cost` {source, claim, ns, tolerance}; `construction` {source, task, sizes, count, seed, params, threshold}; `magic_x` {claim, assumption}; `selfcheck` {quick}. The library functions can also be called directly on the host; that is how the unit tests use them.

A `learnability` request (Phase 4, profile obligation P4) goes through `check_learnability`, which runs a timing probe first (request kind `learnability_probe`, job id prefix `mathprobe-`): the first listed seed on min(m, 300) examples with the learner's full schedule and both data controls, every phase timed, under a wall cap of 80% of the full cap divided by the number of seeds. The probe prints status `unknown` and never decides anything; the result records it under `probe`. `learnability.projected_seconds` scales the phases that grow with m (training data, the true training, the wrong-answer control) by m over the probe's examples, keeps the fixed ones, and multiplies by the seeds. When that projection exceeds 80% of the cap (1440 s of 1800 s), or the probe hits its own cap or memory, the full job is not run and the result is `unknown` with a `resource` record and an evidence line that states the cap, the measured phases and the projection. A full job stopped by the cap gets the same kind of record. Stage M also refuses, for the same frozen experiment (statement and definition hashes), an unchanged source after a resource failure and, after a full timeout, a new source whose probe does not project at most 80% of the timed-out source's projection. Results and reasons: `results/phase4-20261013T073748Z/FIX-2.md`.

### Conservation laws

Each law records its statement, its assumptions and its source in `conservation.LAWS`. A check returns `discharged`, or `violated` with a counterexample that names the numbers and the smallest size at which the claim already fails, and the kill code `conservation:<law>`.

| Law | Function | Bound | Source |
| --- | --- | --- | --- |
| storage | `storage_bound`, `storage_check` | K facts of b bits need K·b bits of state | counting (pigeonhole) |
| copy | `copy_error_lower_bound`, `copy_check` | per-string error above 1 − 2^S / D^L | Jelassi et al. 2024, Theorem 2.7 |
| Dyck-2 | `dyck_error_lower_bound`, `dyck_check` | per-sequence error above 1 − 2^S / 2^D; exact needs S ≥ D | counting, plan P5 |
| INDEX | `streaming_index_bound` | S ≥ n·(1 − h2(e)) for one pass with per-query error e; S ≥ n at e = 0 | Kremer, Nisan and Ron 1999; constant from Fano |
| associative recall | `associative_recall_required_bits`, `associative_recall_check` | S ≥ K·(log2 V − h2(e) − e·log2(V − 1)) | reduction to INDEX, Fano per key |
| sample information | `sample_bound`, `sample_check` | m ≥ (log2 \|H\| − P) / c exact; m ≥ ((1 − e)·log2 \|H\| − 1 − P) / c with error e | counting and Fano |
| reads | `read_bound`, `read_check` | at least n operations | trivial |

The plan marks the exact statement of Kremer, Nisan and Ron as unverified in detail. The library therefore uses the Ω(n) result with the constant 1 − h2(e) from the standard information argument, derived in the docstring, and says so in the law's source field.

A card's numbers go in as one dict: an optional default `state_bits` and lists under `storage`, `copy`, `dyck`, `index`, `recall`, `sample` and `reads`. An unknown section or a missing field raises, since a claim the library cannot read cannot be discharged. `find_state_collision(encode, inputs, answers)` gives an executed pigeonhole witness when the card's state code exists: two inputs with different answers and the same state.

Known numbers in the tests: 1000 bits for 100 facts of 10 bits, and 999 bits failing at the 100th fact; Jelassi's bound 1 − 2^−10 for 10 bits and binary strings of length 20, 0 for 64 bits and 16 hex symbols, 0.9375 for 17; INDEX at 1024 bits needs 1024 bits exact and 512.09 at error 0.11 (h2(0.11) = 0.499916); recall of K = 64 values over V = 256 at accuracy 0.99 needs 501.71 bits, so a 256-bit state fails from K = 33 (the planted capacity overrun); 10 examples for 1024 hypotheses with binary labels (binary search attains it); 7 for a permutation of 10 items; 89 at Fano error 0.1 for 2^100 hypotheses.

### Cost verifier

The candidate is Python source without imports that defines `init(sizes, ops) -> (params, state)`, `update(params, state, token, ops) -> state` and `predict(params, state, ops)`. Its only array functions are in `ops.np`, and every array they return is a `Counted` wrapper whose arithmetic is charged to the counter. The raw array is out of reach: the source audit refuses imports, attributes starting with an underscore, and builtins such as `eval`, `getattr` and `open`. Mixing a plain NumPy array into counted arithmetic raises.

Counting rules, one unit each: a matrix product counts k multiply-adds per output element; an elementwise ufunc counts one per output element, as `arith`, `compare` or `transcendental`; `ops.np.sigmoid`, the logistic sigma(x) = 1/(1+exp(-x)), counts one `transcendental` per element like `tanh` (FIX-3: card k2 froze a unit-cost sigmoid the counter could not run); a reduction over k elements counts k − 1. Creating, indexing, slicing and concatenating add nothing to the op total and show up in the state size instead; the touched-word tally below charges the elements they write. Python scalar arithmetic is not counted. Every executed line of candidate code is counted with `sys.settrace`, so a Python loop doing uncounted work still shows its growth.

`verify_cost(source, CostClaim(ops, state, sizes, exponent), ns=..., tolerance=0.05)` runs the candidate at every n (the plan's 64 to 4096 by default) crossed with every listed value of the other sizes, each of which needs two values. It requires:

1. counted operations equal to the SymPy formula at every point, or within the declared remainder;
2. the least-squares log-log slope in n within 0.05 of the formula's slope over the same points, and likewise the two-point slope in every other size; with a declared remainder, within 0.05 of the band of slopes that counts inside the remainder can have;
3. the formula's asymptotic exponent, lim log f / log n, equal to the claimed exponent;
4. the slope of executed lines in n at most the formula's slope plus 0.05;
5. peak state words equal to the state formula at every point;
6. parameters that do not grow during the sequence;
7. the growth of the touched-word tally (next section) in n and in every other size at most the order of the claimed op formula (checks `touched_exponent_<size>`).

#### Costed work (`cost.COSTED_WORK`)

The op counter sees `ops.np` arithmetic and the line tracer sees executed lines. Before the 2026-10-16 review fix, work done inside a builtin, a container operator or an array copy was visible to neither: `state['h'] + [token]` per token with `sum(state['h'])` per prediction is quadratic and verified as linear. Of seven hidden-quadratic probes claimed linear, six were discharged (`results/review-fixes-20261016/probe_cost_before.json`); only the list comprehension was caught, by the line slope.

Inside a costed call (update and predict of a cost candidate, x of a magic-X assumption) the candidate source is rewritten at load time so that these operations go through guards. A guard charges the elements it touches, exactly, to a separate tally that is kept apart from the op total, so no cost formula changes:

| Work | Touched words |
| --- | --- |
| `sum`, `max`, `min`, `any`, `all`, `list`, `tuple`, `dict` over k items | k |
| `in` on a list, tuple or string | elements scanned (dict, set and range membership: 1) |
| slicing a list, tuple or string; `+`, `*`, `\|` that build one | result length |
| `list.copy`, `count`, `reverse`, `clear` | length |
| `list.index`, `remove`; `insert`, `pop(i)`, `del` | elements scanned and moved |
| `extend`, `+=` | items added |
| array creation, `ops.np.concatenate` and `stack`, `.copy()`, a reshape or index that copies, array item assignment | elements written |

Array views (basic slices, `.T`, a reshape that does not copy) are free. Outside a costed call (init, conformance runs) the guards charge nothing and refuse nothing.

Some work cannot be counted exactly, so it is refused with `CandidateError`: `sorted` and `list.sort`; `==`, `!=`, `<`, `<=`, `>`, `>=` between lists, tuples or dicts; `in` on an array or an iterator; set algebra; string methods; formatting a container in an f-string. `*`-unpacking and `**`-unpacking, and a chained comparison whose middle operand is an expression, are refused in the source audit, before anything runs.

`touched_exponent_<size>` requires the touched slope to be at most max(the formula's fitted slope, its asymptotic exponent) plus the tolerance. A design whose data movement outgrows its counted work fails; one that appends and reads the last item touches O(1) per token and passes. The calibration cards' own cost sources (K1 to K4) give the same verdicts as before the change (`results/review-fixes-20261016/known-cards-cost.json`).

A remainder (`ops_remainder`, `state_remainder`) must come with the sizes in which it is of strictly lower order than its formula (`ops_remainder_lower_in`, `state_remainder_lower_in`); `remainder_problems` refuses, before freezing and again in `verify_cost`, a remainder without that list, one that is negative or not below its formula at a grid point, and one whose asymptotic exponent in a listed size is not below the formula's. The first failed check in that order becomes the counterexample. Wall time is recorded and never decides.

Reference implementations with formulas derived by hand in `references.py`: an Elman RNN, n(d² + 2d + dV + V − 1); single-head softmax attention, n(3d² + dV + V − 3) + (2d + 5)·n(n + 1)/2 with state 2nd + d; a decayed unigram cache, 2nV; an HMM forward pass, n(2k² + kV + 3k + V − 2). All four match exactly at every point, and their measured slopes equal the formula's to four decimals. The planted exponent error claims n(2d − 1), linear, but re-applies a decay t times at step t; the verifier reports `exponent_n` (measured slope 1.98 against 1.0 on n from 64 to 512), `lines_exponent_n` and `ops_exact`, and accepts the true formula d·n(n + 1)/2 + n(d − 1). The same code claimed as d·n·log n fails too. A second planted candidate matches its linear op claim exactly while running an uncounted Python loop; only the line slope (1.93) catches it.

### Construction harness

`run_construction(source, task, sizes, count=100, seed=0, params=None, threshold=0.99)` runs a hand-set mechanism (`init(layout)`, `update(state, token)`, `predict(state)`, NumPy available as `np`) on generated instances and reports exact accuracy per size against the generator's answers. The result is the `executed` tier, and its evidence says "observed on the listed sizes", never proven.

| Task | Size | Instance | Metric |
| --- | --- | --- | --- |
| `copy` | L | uniform string over D symbols, separator, the string again; the instance stores the true string, but the harness feeds the mechanism its own previous prediction after the separator (self-fed continuation) | per string |
| `recall` | K | MQAR format: K distinct keys with values over V, a separator, every key once as a query | per query |
| `dyck` | depth D | Dyck-2 word that opens D brackets, wanders between depth 1 and D, then closes; predict the legal closer after every token that leaves the stack non-empty | per sequence |
| `composition` | T | tables g and f given as pairs, then every x as a query for f(g(x)) | per query |
| `binding` | hops | two interleaved assignment chains v1 = c, v2 = v1, ...; query the last variable of each | per query |

Generators are deterministic in task, size, seed and params. Each task also has a solver that recomputes the answers from the tokens alone, and `oracle_self_score` checks that the two agree: 1.0 on every task in the tests. The harness also records peak state words per size, which feed P5. The hand-set constructions in `references.py` pass at every listed size when unbounded. With their memory capped at 8 slots (copy and recall) or 4 stack levels (Dyck), they fail just above the cap: recall at K = 9 drops below 0.99 while K = 8 stays at 1.0, and at K = 16 it reads 0.5375.

### Magic-X harness

`magic_x_test(AdvantageClaim(task, size, value, tolerance, examples), Assumption(id, kind, source, status, cost_in_p1))` gives the assumption's code to a plain baseline and measures whether baseline plus X alone reaches the card's claimed per-query accuracy on the card's own task, after the same number of training instances. If it does, the card fails with kill code `magic_x`, unless X is `proven` or `measured` and its cost is in P1. The assumption is `def x(prefix, layout, ops)`: it sees only the tokens read so far, and it runs under the op counter and line tracer, so its cost is measured. Code that cannot load or run gives `unknown`, which leaves the assumption `open`.

The plan names an n-gram, an LSTM or a small transformer as the baseline. The tools image has NumPy and no torch, so only the n-gram is implemented: counts keyed on the last two tokens and X's feature, backing off to the feature alone, the context alone, then the most common answer. Every assumption kind maps to it.

The example in the tests is the plan's T3 case of a perfect retrieval index with no counted cost. A card claims 0.95 per-query recall at K = 8 over V = 16 from 64 training instances. The n-gram alone scores 0.064; with the retrieval oracle it scores 1.0, so the card fails magic-X. Position parity as X leaves the baseline at 0.056, and the card passes. The same oracle marked `measured` with its cost in P1 is exempt.

### Check-hash manifest

`MANIFEST.json` holds the sha256 of every Python file directly under `hybrid/math` (14 files). `python -m hybrid.math.manifest` verifies and lists every difference; `--write` records the current hashes and is an operator action after a reviewed change. `test_committed_manifest_matches_every_math_file` fails on any difference, and `check_library` refuses to stage when one exists. Model roles never get file access, so the manifest catches a changed check whatever made the change.

### Tests

- `hybrid/tests/math/test_conservation.py`: every law against the known numbers above, the planted storage and recall overruns, card claims, fail-fast on unreadable claims, the collision witness.
- `test_cost_hidden_work.py`: the seven hidden-quadratic probes (six discharged before the fix), the refusals, the exact touched tally, honest candidates that still pass and the references.
- `test_cost.py`: counting rules, the audit, the four references against their formulas and a by-hand count at one point, the planted exponent error, an n log n claim on a nested loop, the uncounted loop, state and parameter checks, remainders, exponent claims.
- `test_constructions.py`: determinism, oracle self-score 1.0, task formats, every hand-set construction, failure exactly above capacity.
- `test_magic_x.py`: the oracle that alone gives the advantage, the uninformative assumption, the exemption rule, unrunnable code, prefix-only access, the n-gram backoff.
- `test_library_manifest.py`: the committed manifest, edited, added and removed files, the staged driver refusing an edited file, imports of the staged modules, `check_library` end to end with the driver run by the host Python, and the selfcheck battery.
- `test_library_live.py`: the library cases in Podman; skipped unless `BIGBRANE_TOOLS_PODMAN=1`. Run with that variable on 2026-10-12: 2 passed in 101.75 s (`results/phase2-20261012T180235Z/live-pytest.log`).

Results: `pytest hybrid/tests/math` gave 154 passed and 2 skipped; the full suite gave 1585 passed and 11 skipped (`results/phase2-20261012T180235Z/full-suite.log`).

Live runs in the tools image (`python -m hybrid.math.library_smoke --out results/phase2-<stamp>/live<n>`), two passes, every case decided and replayed with network `none`:

| Case | Expected | Result | Check time per run, pass 2 (s) |
| --- | --- | --- | --- |
| selfcheck battery, plan grid n = 64 to 4096 | discharged | 49 of 49 cases as expected | 41.9, 77.2 |
| card with a recall capacity overrun | violated | `smallest_failing_K` 33 | 1.7, 1.6 |
| planted exponent error | violated | `exponent_n`, slope 1.98 against 1.0 | 9.4, 5.0 |
| Elman RNN on the plan grid | discharged | exact at all 28 points | 1.3, 1.6 |
| recall construction capped at 8 slots | violated | 0.5375 at K = 16 | 1.1, 0.5 |
| retrieval oracle as X | violated | baseline 0.064, baseline + X 1.0 | 0.6, 0.7 |

Pass 2 ran while the full test suite was running, which explains the slower selfcheck.

### Limits

- Only the n-gram baseline exists for magic-X. An LSTM or transformer baseline needs torch, which the tools image lacks; the executor image has it, and the test ladder (Phase 10) owns those baselines.
- The op counter sees NumPy work through `ops.np` and Python control flow through executed lines. Loops and comprehensions emit a line event per iteration, so their growth shows (a test checks a comprehension). Work done inside one call to a builtin is invisible: `sum(range(t))`, `sorted(xs)`, `xs.index(v)` or `v in xs` over a growing list count as one line. A candidate could hide a linear scan per token that way and pass a linear claim with quadratic work. Closing this needs either a narrower builtin set for candidates or charging such calls by argument length; neither is done.
- Data movement is not counted as operations; it is visible only through state size.
- The construction harness checks behaviour on generated instances. It is the executed tier, and nothing in it is a proof.
- The INDEX and recall constants come from the information argument, not from a fetched copy of Kremer, Nisan and Ron.
