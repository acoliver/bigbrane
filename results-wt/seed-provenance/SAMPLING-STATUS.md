# Default sampling works without the ignored cache

Default literature sampling now consumes the validated shipped seed projection. The unchanged review-gate regression passes in a fresh cache-free source export. Explicit raw-log requests still fail when missing and retain the broader historical pool when supplied. The two Phase7 acceptance failures remain.

## Cause and scoped repair

Work started on sibling `r2-bar` at `0393537eadac363381ac5da14c79a5d8dd768d47`. Main `p4-diagnosis` remains clean at `82580a6630c4b5c9155f00d603f96682d5818e7c`.

`sample_for_lens` previously passed its default `BUILD_LOG_PATH` explicitly to `cached_search_papers`, whose own default also selected that raw log. The repaired seed reader correctly rejected its absence. Local historical cache files hid this dependency.

`hybrid/literature/sampling.py:17` and `hybrid/literature/seed.py:163` now default their optional raw paths to `None`. At `seed.py:165`, omitted selection invokes existing `read_build_log()` and its projection validation. Explicit paths still stream every raw JSONL row through `iter_build_log`, filtering only by lens. There is no existence fallback, swallowed error, copied cache, compatibility wrapper, or change to sampling order, deduplication or random-seed handling. Remaining `BUILD_LOG_PATH` defaults belong to raw-log writing and seed compilation, where raw history is intentional.

The default projected pool contains the 109 shipped paper identities across 24 lenses, with complete metadata checked against the seed. It does not claim to ship the broader historical search pool. Five added test cases exercise omitted and explicit-`None` defaults across every lens, reject any default raw-log open, check complete paper equality, require explicit missing paths to raise, preserve non-seed hits across multiple historical rows, and require corrupt projection data to raise.

## Verification

| Execution | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Targeted sibling: sampling, unchanged seed, provenance, unchanged failing review-gate case | 46 | 0 | 0 |
| Same targets in fresh cache-free export | 46 | 0 | 0 |
| Prior selected archive suite, including five added sampling cases | 426 | 2 | 1 |

The selected suite also reports three passing subtests. Its original testcase set now has 421 passes, two failures and one skip, versus 420/3/1 before this repair. Comparison of every original testcase finds exactly one changed outcome: `test_review_gate_evidence::test_accepted_excerpt_gaps_scope_the_final_persisted_card` now passes. The five additions all pass. All seven original seed assertions and 29 provenance tests pass unchanged.

Both remaining failures are unchanged assertions in `hybrid/tests/research/test_phase7_replay.py`:

- `test_every_s2_death_gets_its_expected_phase7_code`, line 29.
- `test_the_linear_attention_and_delta_rule_reductions_are_excluded_as_families`, line 36.

The Phase7 rows are byte-identical to the prior isolated run, SHA256 `259ec1b33cfaed911dfdd2f7034159bee4351eaf9a0f58b5cdd06bf8f025f72b`. Exact historical code matches remain 7/19. PA2 remains FAIL 2/4 against 3/4; the named-gate guard remains FAIL 9/10 and counted-reductions-stay-dead remains FAIL 3/6. No acceptance improvement is claimed for these historical predicates.

All runs used main's existing `.venv-hybrid/bin/python`, `BIGBRANE_LIVE=0`, sibling or export cwd/PYTHONPATH, and the project conftest. An inherited audit hook prohibited connections, DNS and datagrams; each run proved these prohibitions before pytest. Import proofs pin seed, sampling, client and measurement modules to the selected cwd. The prior export's optional absolute original-run existence checks were again masked absent. No live HTTP or LLM call was made. Only the selected archive suite was repeated, not the full research suite.

## Preserved evidence and delivery

Local proof is under `results-wt/seed-provenance/sampling-verification/`: `verify.py`, per-run invocation JSON, logs and JUnit XML, `verification.json`, `comparison.json`, `preservation.json`, and the fresh export. Invocation JSON records exact commands and environment. Before/after checks prove the export has no `hybrid/literature/cache`, raw captures, publication-source directories or original runs. The tested source bytes match the sibling. No `/tmp` output was used.

Hash checks preserve all 75 original seed inputs, 274 saved inputs, 44 main transport originals, 62 protected files and 65 fixture-checksum targets. The original seed tests, seed TOML, projection, review-gate test and Phase7 test are unchanged. No tolerance, preregistration, expected-code fixture, protected document or mathematical code was edited.

Delivery is limited to the two reader source files, sampling tests and this report, with commit subject `Use shipped provenance for default literature sampling`. Logs, caches, raw history and verification exports remain local. GitHub and existing SSH authentication were checked as `acoliver`; no global authentication setting changed. The pushed hash and final worktree states are recorded separately in local `sampling-verification/delivery.json` after delivery. No main push, branch switch, merge or integration is authorized. This work is unreviewed; no further review was run after the two completed cycles.
