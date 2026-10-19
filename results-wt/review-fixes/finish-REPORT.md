# Twenty-four historical failures are repaired; two Phase7 assertions still fail

This tree is not ready for integration or the findings-only follow-up. The exact original 26 failing test nodes now produce **24 passed, two failed**. Both failures are unchanged aggregate assertions in `test_phase7_replay.py`. The final affected-file run, including FIX10/FIX12, KD8/KD4, literature and B1/B2/B3/B4/B7 regressions, produces **420 passed, two failed, one skipped**. The narrower focused run produced 299 passed and one skipped. Final combined verification is running separately; no combined pass is claimed.

## Scope and changes

All changes are uncommitted in `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, branch `r2-bar`, over `fa08b93750c19f59808ae1ac18af979e4d8ac310`. Earlier workers' edits were preserved. No live LLM/gateway calls, OCR, environment changes, staging, commits, pushes or merges were performed by this finish worker. Protected-path and staging checks are empty in `finish-frozen-paths.log` and `finish-staged.log`.

**B4 integration:** `s2_prior_art.py:158` honors explicit unassessed coverage even for nonempty candidates without retrying. Its artifact at `:621` retains coverage, listing, coverage assessment, the computed unassessed flag and reasons. The critic listing at `:641` includes source-qualified identities and supplied excerpts. `difference.py:171,209,336` reserves classifier evidence and critic citations, balances the remaining fields, keeps the existing 80/35-paper limits and sends actual listing receipts to the critic and both difference participants. Excess required citations fail explicitly rather than disappearing. The rendered receipt also includes the classifier's coverage assessment. Tests cover an unrelated nonempty hit, usable alternate-source coverage despite an arXiv error, retained reasons, required late-ranked citations, actual positions and both participants.

**Independent comparisons:** host equations were moved out of S2 into `published_references.py`, with the specified SEQUITUR scheduler in `sequitur_reference.py`. References now cover the preserved Jeffrey/IPF, Dyck-2, FIFO copy automaton, two-counter recognizer, Inside, Earley, Viterbi, selective diagonal SSM and Mamba ZOH controls, delta rule and gated additive attention. Explicit permitted repairs cover the FIX12 phase normalizer, MesaNet polynomial multi-read and weighted bimap. The original Phase7 epsilon normalizer and MesaNet logits have separate operations matching their actual saved programs. Numeric samples may carry each seeded draw's mapped parameters; the host validates the exact parameter-field contract and compares returned finite numeric traces at the external boundary.

**Historical evidence:** `fixtures/reference_replays/` contains separate raw card traces, mapped parameters, unchanged saved source programs, source provenance, instrumented-source hashes and fixture checksums. `capture_saved_traces.py` re-executes only the enumerated saved local programs and adds trace snapshots at explicit boundaries. The published host algorithms do not import or execute those programs. `reference_replays.py` joins the new evidence to copies of historical replay inputs. Original fixture labels, scalar differences, verdicts, original guard assertions and expected kill codes remain unchanged. FIX10 uses its own original job's trace, not a later bimap job's output. The bimap reductions remain refuted; a matching limit-only trace remains unverified.

This is bounded numeric evidence. It does not prove equivalence for all parameters or independently establish that an authored card-side implementation faithfully implements every card equation. Unsupported operations and repairs stay unverified. No confidence fallback, tolerance change, fabricated source, copied comparator output or declaration-only success adapter was added.

## Verification and provenance

All new checks use main `.venv-hybrid/bin/python`, cwd and PYTHONPATH set to the sibling worktree. Logs and pytest directories are under `results-wt/review-fixes/`.

| Check | Result | Evidence |
| --- | --- | --- |
| Exact original 26 nodes | 24 passed, two failed | `finish-targeted26.log`, exit 1 |
| Literature, receipt wiring, independent-reference/hash tests, gate regressions, known-work, difference, local-index and evidence tools | 299 passed, one skipped | `finish-focused.log` |
| KD8 and KD4 file-level check | 46 passed | `finish-kd8-kd4.log` |
| FIX12 file-level check | 33 passed | `finish-fix12-2.log`; confirmed with corrected FIX10 provenance in later focused reference run |
| References, FIX12 and B1/B2/B3/B7 regressions | 82 passed | `finish-reference-tests.log` |
| Original Phase7 plus reference trace/hash tests | Eight passed, two failed | `finish-phase7.log` |
| Final affected files, including FIX10/FIX12, KD5/KD7 and the original Phase7 assertions | 420 passed, two failed, one skipped | `finish-final-tree-targeted.log` |
| FIX10 saved bimap and explicit synthetic routing specimens | 25 passed | `finish-fix10.log` |
| Final combined research, literature and evidence tools | Running | `finish-combined.log`, `finish-combined.pid`, `finish-combined.exit` when complete |
| Whitespace, protected paths and staged files | No output | `finish-diff-check.log`, `finish-frozen-paths.log`, `finish-staged.log` |

Final tested source manifest: `finish-tested-manifest.json`, content-set SHA-256 `383c1061d18f6a8c8faa325f12fdf0c86aacdd992d251ebf96a934a9083f524f`. `finish-manifest-check.json` confirms no source/test/fixture changes after final verification began. The tracked diff is preserved in `finish-tested.diff`; source and fixture hashes also cover untracked implementation files. `finish-phase7-source-manifest.json` additionally verifies the original Phase7 executor artifacts and copied program byte identity. `finish-files.txt` lists delivery paths. Three partial combined runs predated the last changes. The third, `finish-combined-pre-kd5-kd7.log`, exposed the remaining KD5 synthetic comparator and KD7 source-relevance/two-entry fixture migrations. Those fixtures now use the supported synthetic reference, real requirement construction and two relevant entries. The original assertions remain unchanged. KD7 also exposed receipt metadata preceding the required-first source listing; the listing now precedes its receipt without changing either participant's evidence. The KD5/KD7, B4 and difference follow-up slice passes all 42 tests. Their logs are marked `finish-combined-superseded.log` and `finish-combined-pre-fix10.log`; neither is final evidence. The latter exposed nine FIX10 fixture failures. Those were repaired by binding the original saved bimap traces and using explicit synthetic numeric routing specimens. The FIX10 assertions remain unchanged. The extra host repairs are L1 read normalization and a direct sum with a zero state channel.

## Remaining original findings

B1's genuine-control preservation is incomplete for the older Phase7 population. Of its 19 registered S2 deaths, 16 still miss their exact expected code, although the original phase-normalizer and MesaNet repairs now recover their required exclusions. The two failing assertions and all per-card mismatches are recorded in `finish-PHASE7-REMAINING.md` and `finish-phase7-rows.json`.

In particular, card-6de11b01 needs a tag equality gate read from a second memory and a prediction-error-dependent overwrite gate. Its saved job reports a 1.23 output difference when the tag is removed. A plain delta-rule reference therefore cannot stand in for that mechanism. The existing allowed gate definition is fixed, learned or input-dependent. No policy-conformance argument and independent operation for those state-dependent decisions was supplied, and the policy was not broadened to force the historical assertion to pass. Other unsupported published comparisons also require their own specifications and actual original traces before preservation can be claimed.

The saved PA2 retrieval component of B4 remains **FAIL 2/4 against 3/4**. New receipt wiring does not change a historical retrieval result. The corrected preregistered scoring remains **FAIL 9/10** and **FAIL 3/6**; all-dead 10/10 and 6/6 remain secondary diagnostics. No new retrieval experiment or learning/performance result is claimed.

B2, B3 and B7 have focused passing implementation regressions; B4's downstream wiring does too. These are implementation results, not an independent review verdict. No follow-up review was run while original tests remain failing. The one findings-only follow-up remains available after recovery.

## Recovery commands

Monitor the single final combined process without launching another worker:

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
cat results-wt/review-fixes/finish-combined.pid
cat results-wt/review-fixes/finish-combined.exit 2>/dev/null
tail -80 results-wt/review-fixes/finish-combined.log
```

If the final process is no longer running and no exit file exists, preserve its log before resuming `bash results-wt/review-fixes/finish-combined.sh`. The final launch has shell PID 90531 and pytest PID 90533. The managed monitor job is `shell_a3f744d002d6`; the test process also runs under nohup and survives this session.

Reproduce the remaining original assertions:

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
PYTHONPATH="$PWD" /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest hybrid/tests/research/test_phase7_replay.py -q --basetemp=results-wt/review-fixes/recovery-phase7-pytest
```

Rebuild the new evidence from the saved local sources, then recheck fixture integrity, only after deliberately changing a supported replay implementation:

```sh
PYTHONPATH="$PWD" /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python results-wt/review-fixes/capture_saved_traces.py
PYTHONPATH="$PWD" /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest hybrid/tests/research/test_reference_replays.py -q
```
