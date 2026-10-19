# Two historical exact-code gaps are closed; fourteen remain

The unchanged Phase7 replay now matches **5 of 19 required S2 codes**, up from 3. Exact-code misses fell from **16 to 14**. Original specimens `card-582f8abf` and `card-d86dca4b` now produce `reduces_to_known:other_known` through the real S2 host path. All 19 other replay rows retain their previous codes. Both original S2 passes still pass its reduction gates.

Worktree: `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, branch `r2-bar`, HEAD `fe264b5996b7177d2fbcca5e9274ca5414210578`. Main remains `p4-diagnosis` and received no edits from this task. Earlier `HISTORICAL.md` and `OPERATIONS.md` retain the previous schema-frozen recovery history; this report records the separately authorized binding.

## Contract changes and evidence selection

Only `parityunionfind582f8abf` and `finitePGLversionspaced86dca4b` were admitted to the schema enum and host registry. The 32 admitted operations exactly match the registry and policy advertisement. Both new identifiers dispatch to the already independently specified parity and projective functions. Their arithmetic, domains and protocols are unchanged.

Diagnostic names `parity_union_find`, `pgl_version_space`, `nested_size_posterior` and `sugs_regression_assignments` remain available to offline arithmetic tests, but are absent from the production registry, critic enum and policy advertisement. Neither refuted nor conditional arithmetic becomes a supported acceptance operation.

`historical_declarations.json` retains the recovered published-component citations, source and ledger provenance, and the same 16 parity and 24 projective sample episodes. Each declaration binds its base family, complete original critic parameter mapping, reference parameters, components, empty transformation list and `limit_only: false`. Sample parameters cover dimensions 2, 4, 6 and 8, plus primes 5, 7 and 11 in active/passive modes. These cards have no learned parts to freeze. The existing root answer keeps `fixed_learned_parts: []`.

The helper selects by original card id and SHA256 of the complete card and recorded critic JSON. Changed cards or mappings cannot borrow the original evidence. Expected codes, saved kill codes and recorded outcomes do not select evidence. The envelope preserves the original reported difference as diagnostic data and supplies actual captured CARD-side samples. The host recomputes the difference. No new verdict, equality claim or citation is inferred from the expected code. Unknown fields still fail validation.

The schema differs from HEAD only in the two enum additions. Hash checks confirm no changes to S2 acceptance code, `contracts.py`, `families.toml`, the original Phase7 test/fixture or existing reference/review-gate tests. Root tests, math, preregistration, tolerances, gates and expected codes remain frozen. The obsolete positive schema-freeze test now asserts the authorized two-operation boundary; historical negative and mutant test bodies are unchanged.

## Provenance and compact packaging

Parity equations come from [CP-Algorithms DSU parity](https://cp-algorithms.com/data_structures/disjoint_set_union.html#support-the-parity-of-the-path-length-checking-bipartiteness-online). Projective inference uses [Mitchell chapter 6](https://www.cs.cmu.edu/afs/cs/project/theo-20/www/mlbook/ch6.pdf) and [Steif section 4.1](https://www.math.chalmers.se/~steif/group.lectures.pdf). Retrieval/source hashes and the six publication snapshots remain local under `sources/`. Projective empty-space uniform predictions and query zero remain explicit abstention protocol, not a posterior on impossible evidence.

All four original program fixtures and the full observer capture remain byte-identical to the binding-start hashes. The full matrix/state capture remains at its original path, locally ignored. No original was deleted. `LOCAL-SHA256SUMS` records its hash. The source manifest retains source/ledger attribution and original/reexecuted stdout. When the local full capture is absent, the test helper can reproduce it in memory from the immutable program fixtures and original observer. A test simulates absence without moving or deleting the file, then checks equality of the entire reproduced capture.

Packaged fixtures are small: positive declarations are 291,268 bytes, source manifest 7,892 bytes, and negative controls 5,020 bytes. The negative fixture preserves an actual SUGS counterexample, an empty-support concept query and a roundoff-domain query, with episode/step attribution and source provenance. Every packaged fixture has a checksum. Original source fixtures are small Python programs; full traces and publication snapshots are local evidence, not packaged matrices. Compact contract regressions and Phase7 replay do not need publication snapshots. The original historical diagnostic test also checks the locally retained snapshots.

## Tests

Final targeted run: **89 passed, two failed; exit 1**. No full suite or review was run. Logs: `binding-delivery.log`, `binding-delivery.exit`, `binding-delivery-junit.xml`. `binding_results.py` retains rows from the actual unchanged Phase7 pytest fixture, without running another replay to manufacture the accounting.

| Module | Passed | Failed |
| --- | ---: | ---: |
| `test_historical_binding.py` | 18 | 0 |
| `test_historical_references.py` | 19 | 0 |
| `test_reference_replays.py` | 6 | 0 |
| `test_review_gate_evidence.py` | 44 | 0 |
| `test_phase7_replay.py` | 2 | 2 |

The two failures remain `test_phase7_replay.py:29`, requiring every exact code, and `:36`, requiring tagged-memory delta-rule exclusion. Both remain failures without xfail, weakened assertions or changed expected codes.

All 300 original parity trials, 13,580 tokens, match. All 720 original projective episodes, 10,108 consumed tokens plus 720 final predictions, match. Every compact sample was reexecuted against untouched original CARD definitions. The full capture was reproduced in memory and matched in full. These are sampled executable results within the documented operation domains, not a general conformance proof.

Deliberate perturbations are detected. Adding one to a CARD output gives a measured difference of 1 and a refuted comparison for each operation, despite reported difference zero. Unsupported operation names, extra parameter fields, undeclared ids, unknown report/declaration fields and changed specimens are rejected. Existing source mutants still differ: dropped parity labels by 1, suppressed projective updates by more than 0.5. Boolean enumeration, projective cardinality/three-anchor controls and all 44 review-gate evidence tests pass.

SUGS remains refuted on **1,174/2,418** accepted allocations. Concept evidence remains conditional: **680** feasible queries match, **84** empty-support cases and **two** roundoff-domain cases remain unsupported. Neither receives a Phase7 declaration. The tagged-memory policy mismatch remains unresolved.

Final command, from the worktree:

```sh
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/evidence-recovery /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest hybrid/tests/research/test_historical_binding.py hybrid/tests/research/test_historical_references.py hybrid/tests/research/test_reference_replays.py hybrid/tests/research/test_review_gate_evidence.py hybrid/tests/research/test_phase7_replay.py -p binding_results -q --basetemp=results-wt/evidence-recovery/pytest-binding-delivery --junitxml=results-wt/evidence-recovery/binding-delivery-junit.xml
```

## Exact-code accounting

All 21 replay rows are retained in `binding-phase7-rows.json`; the following table lists all 19 required S2 codes. `none` means no S2 kill code. The fourteen unresolved inputs also remain in the earlier per-card classifications.

| Card | Expected | Before binding | After binding |
| --- | --- | --- | --- |
| `card-1184f7a7` | `excluded_family:linear_attention` | `excluded_family:linear_attention` | `excluded_family:linear_attention` |
| `card-14429dbc` | `reduces_to_known:other_known` | `none` | `none` |
| `card-2c5dd003` | `reduces_to_known:other_known` | `none` | `none` |
| `card-36725a7c` | `reduces_to_known:other_known` | `none` | `none` |
| `card-537e161c` | `excluded_family:delta_rule` | `excluded_family:delta_rule` | `excluded_family:delta_rule` |
| `card-582f8abf` | `reduces_to_known:other_known` | `none` | `reduces_to_known:other_known` |
| `card-68612a2a` | `reduces_to_known:other_known` | `none` | `none` |
| `card-6de11b01` | `excluded_family:delta_rule` | `none` | `none` |
| `card-6fc7efec` | `not_radical:R2` | `none` | `none` |
| `card-7ec5e5c2` | `reduces_to_known:other_known` | `not_radical:R2` | `not_radical:R2` |
| `card-8dc4db49` | `reduces_to_known:other_known` | `not_radical:R2` | `not_radical:R2` |
| `card-a36f1a3a` | `reduces_to_known:other_known` | `none` | `none` |
| `card-bc3935ac` | `reduces_to_known:other_known` | `not_radical:R2` | `not_radical:R2` |
| `card-d1d383d5` | `reduces_to_known:hopfield_or_kernel_memory` | `not_radical:R2` | `not_radical:R2` |
| `card-d86dca4b` | `reduces_to_known:other_known` | `none` | `reduces_to_known:other_known` |
| `card-dc69816b` | `reduces_to_known:other_known` | `none` | `none` |
| `card-ddb071bd` | `not_radical:R2` | `none` | `none` |
| `card-ff8f801a` | `not_radical:R2` | `not_radical:R2` | `not_radical:R2` |
| `card-ffdbe331` | `reduces_to_known:other_known` | `none` | `none` |

## File inventory and tested hash

Tested-tree SHA256: `82278025dc8176e1ce4cb408b232122a95dffc10b3a42488fe74096fc9de50e3`. Complete hashes and the reproducible digest format are in `binding-tested-tree.json`. The fingerprint covers owned code/fixtures and frozen acceptance/configuration/assertion files. It excludes report timestamps and concurrent literature edits.

| Owned source/test file | Bytes | SHA256 prefix |
| --- | ---: | --- |
| `hybrid/research/schemas/reduction_critic.schema.json` | 7988 | `1c44d09cbf38bccb` |
| `hybrid/research/stages/published_references.py` | 26561 | `6b7aaf0bf1daeeee` |
| `hybrid/tests/research/fixtures/reference_replays/.gitignore` | 111 | `c7b197f8c4cc6b9f` |
| `hybrid/tests/research/fixtures/reference_replays/SHA256SUMS` | 2469 | `644c74fd0277c22d` |
| `hybrid/tests/research/fixtures/reference_replays/historical_declarations.json` | 291268 | `8a8476373a0dbdbb` |
| `hybrid/tests/research/fixtures/reference_replays/historical_negative_controls.json` | 5020 | `842ee6242df9185e` |
| `hybrid/tests/research/fixtures/reference_replays/historical_sources.json` | 7892 | `b6b9ae3e2ce9da89` |
| `hybrid/tests/research/fixtures/reference_replays/programs/historical-card-36725a7c.py` | 5669 | `fe866a2315eea970` |
| `hybrid/tests/research/fixtures/reference_replays/programs/historical-card-582f8abf.py` | 6167 | `287c53c95710ae1a` |
| `hybrid/tests/research/fixtures/reference_replays/programs/historical-card-d86dca4b.py` | 7130 | `436d3e3309b27d43` |
| `hybrid/tests/research/fixtures/reference_replays/programs/historical-card-ffdbe331.py` | 6515 | `86d3bf41fb759a79` |
| `hybrid/tests/research/phase7_replay.py` | 13249 | `8035c475a41fdcf8` |
| `hybrid/tests/research/reference_replays.py` | 5639 | `05677159ef0be255` |
| `hybrid/tests/research/test_historical_binding.py` | 10556 | `92e9a05bd208c66a` |
| `hybrid/tests/research/test_historical_references.py` | 11338 | `5b9fd48305cc217e` |
| `results-wt/evidence-recovery/capture_historical.py` | 6796 | `49f829d532717a71` |

`binding-inventory.json` also lists complete sizes/hashes for local retained captures and publication snapshots. Reporting files include BINDING.md, STATUS.md, binding-summary.json, binding-tested-tree.json, binding-baseline.json, binding-phase7-rows.json, binding_results.py, package_binding.py, LOCAL-SHA256SUMS and test logs/XML. Earlier measurements, observer/classification scripts and source provenance remain available. The old measurement script describes the prior offline recovery; it is not the binding fixture generator.

The arxiv HTTPS/parsing edits visible in the worktree belong to the other worker and were not modified or included in this binding fingerprint. No main-code edit, integration, commit, push, merge, live LLM call, system-temp output or review was performed.
