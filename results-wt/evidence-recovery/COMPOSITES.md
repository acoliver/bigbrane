# Three composite investigations are reproducible; exact codes remain 7/19

No additional historical exact code is supported by a full independent mapping. The GCD/exemplar, OMP/coupled-value/FIFO and LFU/binding composites now have byte-identical original programs, observer-only complete source executions, compact specimen-bound evidence and concrete negative witnesses. The original Phase7 assertions still fail. These results identify limits of the historical mappings; they do not establish novelty or rule out every other published composition.

## Scope and provenance

All writes are confined to `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, branch `r2-bar`, starting HEAD `a7bb1e0`. Main remains read-only on `p4-diagnosis` at `09dd756`. No commit, push, merge, integration or independent review is part of this work. The other worker's diagnostic tools, seed projection, transport and cached-provenance implementations are untouched.

`composites_sources.json` binds the original run's `purpose=reduction_check` executor ledger event, artifact SHA256, full embedded program SHA256, observer SHA256, draw digest and complete state/output trace digest. `composites_specimens.json` retains the complete original replay card and critic contents, their normalized hashes and the original `card_version` and `reduction_critic` artifact receipts. The original job input, executor embedded source and packaged program have identical bytes. No comparator verdict or expected code selects the evidence. Changed card summaries or critic mappings cannot borrow it.

| Card | Untouched source execution | Captured CARD state and outputs |
| --- | --- | --- |
| `14429dbc` | Toy plus 400 seeded streams, 5,212 tokens, 63,981 probe readouts | Every retained ordered episode, consolidation clock, modulus, complete residue table, validity flag, original extra probes and output; 2,037 scheduled/explicit compilations |
| `68612a2a` | Original 99-step seeded stream | Complete C, E, beta, thresholds and draw values; all keys, values and norms after each step; all 187 pursuit support/refit/residual iterations; prediction, log probabilities, allocation flag and size; 75 writes |
| `8dc4db49` | Original 60-token seeded stream | Complete raw and transformed E/F, beta, background logits/distribution and pair draws; every LFU record including count, increment timestamp and cached affinity; presence, free concentrations, all 180 sweeps and every output; 29 evictions and seven gated-out bigrams |

`sys.settrace` observes the complete programs without source insertion, replacement, altered seeds or altered comparators. The original comparators still execute. Their stdout is retained as diagnostic evidence only. GCD probes are the original driver's explicit post-step read API calls, including calls after non-query tokens; they must not be confused with proof of every token-level interface convention in the prose card.

Historical raw input draws and the process hash seed were not saved. These captures are **source reexecutions with `PYTHONHASHSEED=0`**, not recovered historical input bytes. The programs use seeded NumPy draws and do not randomly sample set-to-list conversions. The local NumPy version and invocation are recorded. No claim of reproduction on every BLAS/NumPy platform is made.

## GCD/exemplar: closure is classical, but forgetting and delayed reads change the learner

[Helmbold, Sloan and Warmuth, Learning Integer Lattices (1992), p.241 and the closure discussion on p.242](https://mwarmuth.bitbucket.io/pubs/J21.pdf) describes the smallest lattice containing **all positive examples seen so far**. In one dimension,

\[
L=\operatorname{span}_{\mathbb Z}\{d_i\}=\gcd(d_1,\ldots,d_k)\mathbb Z.
\]

Applying this identity to the CARD's retained within-label translations verifies the compilation arithmetic. Independent all-pairs differences reproduce all 2,037 compilation results, including zero generators and multiclass collisions. Separate state checks verify the entire observed FIFO, clock and read precedence across all 63,981 readouts. These checks validate the CARD's specified components; they do not make the CARD's own scheduling and cache rules a published host definition.

Program paths in this report are relative to `hybrid/tests/research/fixtures/reference_replays`. The original CARD's compiler is at `programs/composites-card-14429dbc.py:20-33`, scheduled mutation at `:34-44` and newest-exact-match/quotient read at `:45-56`. The saved `Known` class at `:62-106` already supplies the extra FIFO, clock, multiclass table and read precedence. Its numeric agreement cannot independently source those additions.

Two small witnesses isolate the missing composition:

* **Delayed consolidation:** capacity 2, batch 10, demonstrations `(0,A),(6,A)`. A query at 12 before COMMIT is uniform; after COMMIT it returns A. An immediately updated closure of translation 6 already accepts 12 before the commit. Thus the commit schedule affects predictions.
* **Forgetting:** with the same capacity, demonstrate `(0,A),(6,A)`, COMMIT, then demonstrate `(2,A)` and COMMIT. The retained records are `(6,A),(2,A)`, giving modulus 4 and coset `2+4Z`. Query 8 is uniform. A cumulative learner retains translation generators 6 and 2, giving `2Z`, and accepts 8. Eviction changes the learned subgroup.

A third witness stores contradictory labels at coordinate 0. Before recompilation the newest exact record overrides the old compiled table. After recompilation the table is invalidated. Recency, clock and FIFO mutants each change witness outputs.

**Unresolved:** a full independent composition must specify bounded demonstration retention, anchor migration, delayed/explicit rebuilds, multiclass conflicts, exact-cache override, missing-residue abstention, parser timing and token distribution conventions. HSW supplies lattice learning, not this entire stream operation. The witnesses refute dropping these wrappers as prediction-irrelevant; they do not refute a separately sourced bounded-window composition.

## OMP: read components are supported; RAN/ALD allocation is not the same rule

[Tropp and Gilbert, Algorithm 3](https://tropp.caltech.edu/papers/TG07-Signal-Recovery-preprint.pdf) specifies residual correlation, deterministic tie resolution, support-restricted least squares and residual recomputation:

\[
i_j=\arg\max_i|\langle \phi_i,r^{j-1}\rangle|,\qquad
\alpha^j=\arg\min_a\|q-\Phi_{S_j}a\|_2,\qquad r^j=q-\Phi_{S_j}\alpha^j.
\]

Normalize stored keys to obtain the CARD's score `abs(K_i dot r)/norm(K_i)`; transfer coefficients back to raw-key scale. [Yang et al., CVPR 2008, section 2.1, equations 4-8](https://people.eecs.berkeley.edu/~yima/recognition/Files/CVPR2008_superres.pdf) supplies paired dictionaries and reconstruction `x = D_h alpha`. It solves sparse coding/relaxed l1 objectives with patch compatibility and offline sampled dictionaries. It does not define this stream's adaptive dictionary.

Every one of the 187 observed CARD refits passes the correlation selection, orthogonal residual and coupled value checks. All 99 predictions, output-code probabilities, write decisions and FIFO states are checked. The independent hand answer stores `a+b -> 7` and `a -> 3`, then answers b with 4 through coefficients `(1,-1)`. At capacity 2, storing `c -> 5` evicts the composite and b is answered with 0. The demonstration predictions occur **before** the writes. Signed-weight and FIFO mutants change outputs. Additional handchecks vary scalar output codes, feature codes and beta, rather than requiring identity codes for agreement.

[Platt (1991), section 2.2, equations 2.7-2.8](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/plat1991.pdf) allocates a new unit when nearest-centre distance exceeds its threshold **and** prediction error exceeds its threshold. It installs an error correction and uses LMS on non-allocation steps. [Engel, Mannor and Meir, section 2.1, equations 2.2-2.5](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_446-1.pdf) admits a dictionary atom when its **dense** feature-space projection residual exceeds the ALD threshold. Output error does not independently add an atom.

The two-demonstration witness repeats exactly the same key with values 1 and 2. The CARD writes both because its key-residual **or** value-error gate sees the output error. With an existing centre at that key, Platt's nearest-centre novelty is zero, so its conjunction cannot allocate another unit. ALD's dense projection residual is also zero. This compares allocation decisions only; it does not claim to simulate every RAN or KRLS parameter update. The CARD's rules are at `programs/composites-card-68612a2a.py:41-69`.

### The old comparator also has a floating-point counterexample here

The saved original stdout reported maximum difference `1.4210854715202004e-14`. Untouched local reexecution reports `1.4279215152635878`, with one differing value prediction, at zero-based stream step 32. The CARD chooses support `[9,1]` and predicts `13.427921515263586`; the original `Coupled` class chooses `[0,9]` and predicts `11.999999999999998`.

The cause in this execution is visible in the captured scores. Scalar CARD evaluation gives key 0 score `1.610699078090977` and key 9 score `1.6106990780909771`. The comparator's normalized-matrix evaluation gives both `1.610699078090977`, so its deterministic argmax takes index 0. The first selected atom changes the residual and later support. Source draws, original code and the near-tie remain unchanged. No score tolerance, retries or tie repair were added. The saved record includes prior keys, values, norms, query, both score arrays and both supports. Historical platform details are insufficient to identify which library arithmetic produced the older near-zero verdict.

**Unresolved:** a full independent operation needs the sparse-residual OR output-error write rule, raw coupled values, FIFO capacity, threshold/degenerate-key conventions, output timing and Gaussian decoding. The saved program is scalar-valued and untrained; it does not execute the card's general vector-valued pretraining or differentiate through selected supports. Those learning claims cannot be certified by this capture. Task-driven training, a different allocation rule and a budget policy cannot be inferred from numeric agreement on OMP alone.

## LFU/binding: pinned slack is not SuperGlue's prescribed dustbin marginal

[Sarlin et al., SuperGlue, section 3.2, equations 8-9](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf) augments a score matrix with a shared dustbin score and scales the augmented matrix to margins `[1_M,N]` and `[1_N,M]`. Both dustbin potentials participate in scaling.

The CARD instead alternates

\[
y\leftarrow a/(1+K^Tx),\quad x\leftarrow a/(1+Ky),\quad
p(v)=x_qK_{qv}y_v+x_qu_v.
\]

Its original comparator pins both dustbin potentials at 1. The pinning reproduces the `+1` monomer terms algebraically, but imposes no fixed dustbin marginal. This is an additional operation. Calling it SuperGlue does not supply that definition.

A one-active-pair witness uses positive prior kernel 1 and strictly positive background `(1/4,3/4)`. SuperGlue's augmented 2-by-2 kernel is all ones, with margins `(1,1)`, so its plan is all `1/2`. The pinned operation has free concentration `f=(sqrt(5)-1)/2` and plan

\[
\begin{pmatrix}f^2&f\\f&1\end{pmatrix}.
\]

The real margin is 1, while the dustbin margin is `1+f`, not 1. Thus its real matched mass is approximately `0.381966`, rather than `0.5`. Starting fully free, the unchanged CARD's **three** y-then-x sweeps give free mass `13/21` and matched mass `8/21`. Its probability vector is exactly `(15/28,13/28)` up to floating arithmetic. Routing SuperGlue's free mass through the same background gives `(5/8,3/8)`. The witness distinguishes the fixed point, the finite iteration schedule and the prescribed-marginal algorithm separately.

The full original trace retains the complete LFU table and all 180 sweeps. A dense arithmetic diagnostic reproduces every sweep and output, with maximum output difference `2.220446049250313e-16`. This dense check is explicitly conditional on captured table state, so it is not an independently published full learner. Separate count-state assertions validate all increments, threshold gates, LFU frequency/last-increment tie decisions and presence injections. Restarting concentrations from fully free values at every token, while keeping the same table states, changes maximum output TV by `0.29206361271804654`. Frequency, gate and iteration mutants change outputs. The original implementation is at `programs/composites-card-8dc4db49.py:23-57`; pinned comparator updates are at `:66-93`.

**Unresolved:** an independent full definition must combine LFU-capped exponentiated counts, last-increment eviction ties, positive learned low-rank priors, species injection, pinned slack, finite warm-start y-before-x sweeps and background routing. It must also specify the learning path: the saved program does not run pretraining, implicit differentiation or the card's CG termination rule. The witness refutes the direct identification with SuperGlue's margins, not every unbalanced or slack matrix-scaling formulation. Plain outer-product counts alone do not cover these state-selected evictions and nonlinear iterations.

## Verification and packaging

The current-tree targeted historical/reference/control run produced **181 passed, two failed**, exit 1. All existing mutation and negative-control tests in the selected eight-file research target passed. The failures are the unchanged Phase7 assertions at `test_phase7_replay.py:29` and `:36`. The new evidence file has **31 passing tests** on the final local files. No assertion, expected code, tolerance, family policy, preregistration or schema was changed, and no host operation was added.

The first new-test execution caught a test-authoring error: it selected the first sorted probe instead of probe 12 in the schedule witness. The corrected assertion selects the named input; the expected mathematical answer is unchanged. The first isolated run produced 180 passes and three failures because the new protection inventory accidentally made the local full `historical.json` capture mandatory. The new inventory now protects only the 88 tracked source/fixture files, leaving the raw capture locally preserved and optional. Existing protection lists, tests and controls were not edited. The final cache-free focused run produced **33 passed, two failed**, exit 1: all 31 new evidence cases pass, with the same two unchanged Phase7 failures. The final current-tree new-evidence run also has 31 passes. The broader current run preceded the strictly positive background witness adjustment and protection-inventory packaging correction; the final focused runs test both corrections. All 21 old/current/isolated exact-code lists are identical.

`composites_replays.full_capture()` always reconstructs complete traces in memory from the packaged original source programs and hash-bound observer. It has no local full-capture fast path. The compact fixture contains five complete episodes: three GCD streams, the OMP stream and the binding stream, plus witnesses. Publication JSON contains six short equation/excerpt snapshots with requested/final URLs, retrieval timestamps and PDF/text hashes. Full PDFs, extracted texts and the HSW OCR material remain preserved locally; publication bodies are not needed for tests. HSW text was OCR-extracted from PDF pages 1-10; the full PDF remains unchanged.

Verification uses main's `.venv-hybrid/bin/python` with sibling/package cwd, PYTHONPATH and explicit confcutdir. Network-prohibiting audit hooks are inherited by source-reexecution subprocesses. The isolated package excludes raw captures, publication bodies and the literature cache, masks external original-run existence and prohibits opening that original-run tree. Original artifact checks are conditional on local availability; packaged program, observer, draw and trace hashes are always mandatory. All required new files are listed in `composites-SHA256SUMS` and `composites-INVENTORY.json`; local-only capture/source hashes have a separate inventory.

## Exact-code rows from the unchanged Phase7 fixture

All 21 rows are preserved in `composites-current-phase7-rows.json`. Required exact matches remain **7/19**, with 12 gaps. The two original S2-pass cards still pass. `none` means an empty code list.

| Card | Required exact code | New actual replay |
| --- | --- | --- |
| `card-1184f7a7` | excluded_family:linear_attention | excluded_family:linear_attention |
| `card-14429dbc` | reduces_to_known:other_known | none |
| `card-2c5dd003` | reduces_to_known:other_known | none |
| `card-36725a7c` | reduces_to_known:other_known | none |
| `card-46dbb349` | none | none |
| `card-537e161c` | excluded_family:delta_rule | excluded_family:delta_rule |
| `card-582f8abf` | reduces_to_known:other_known | reduces_to_known:other_known |
| `card-68612a2a` | reduces_to_known:other_known | none |
| `card-6de11b01` | excluded_family:delta_rule | none |
| `card-6fc7efec` | not_radical:R2 | none |
| `card-7ec5e5c2` | reduces_to_known:other_known | reduces_to_known:other_known |
| `card-8dc4db49` | reduces_to_known:other_known | not_radical:R2 |
| `card-a36f1a3a` | reduces_to_known:other_known | none |
| `card-bc3935ac` | reduces_to_known:other_known | reduces_to_known:other_known |
| `card-c9615069` | none | none |
| `card-d1d383d5` | reduces_to_known:hopfield_or_kernel_memory | not_radical:R2 |
| `card-d86dca4b` | reduces_to_known:other_known | reduces_to_known:other_known |
| `card-dc69816b` | reduces_to_known:other_known | none |
| `card-ddb071bd` | not_radical:R2 | none |
| `card-ff8f801a` | not_radical:R2 | not_radical:R2 |
| `card-ffdbe331` | reduces_to_known:other_known | none |

No new exact-code row is added. Full composite references remain unsupported for these three cards. The other nine existing gaps, including tagged-memory policy, prior SUGS/HMM/NGE evidence, nonlinear PPGN, complex ART, tensor calibration, DAM and concept support, are unchanged. No literature search was repeated for the previously refuted mappings. The next tractable work should source a complete bounded composite or retain these witnesses, rather than replacing the CARD with another copy under a reference name.

## Reproduction and delivery

Run the new evidence tests from the sibling with the existing Python environment:

```sh
PYTHONHASHSEED=0 BIGBRANE_LIVE=0 \
PYTHONPATH="$PWD/results-wt/evidence-recovery/composites-network-guard:$PWD" \
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest \
hybrid/tests/research/test_composites_historical_evidence.py -q \
--confcutdir="$PWD" --basetemp=results-wt/evidence-recovery/composites-reproduce
```

`composites_verify.py <label> focused` also runs the actual original Phase7 assertions and saves all 21 real fixture rows; exit 1 is the recorded result, not a new success convention. Omitting `focused` runs the eight selected historical/control targets. `composites_package.py <unique-label>` creates a new cache-free package and runs the focused targets under network and original-run isolation. Existing output directories are rejected rather than overwritten.

The final compact fixture was regenerated byte-for-byte inside the isolated package using the final `composites_measure.py`, without a raw capture or accessible original run. This generation check follows the final isolated tests and changes no test or fixture bytes. `composites-tests.json` records each execution separately; counts are not added across reruns. SHA256 receipts establish byte integrity, not a publisher signature or an independently reproduced historical environment.

Required source/report files under the locally ignored `results-wt` path are explicitly included in the new-file inventory. Raw captures, full publication bodies, OCR images, logs, JUnit files and isolated working trees remain local and are excluded from the required delivery file set. Main remains clean, all 88 protected tracked files match, and all changes are left uncommitted.
