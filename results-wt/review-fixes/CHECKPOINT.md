# INCOMPLETE checkpoint: do not integrate

This checkpoint packages the existing review fixes on `r2-bar` over `fa08b93750c19f59808ae1ac18af979e4d8ac310` for a draft PR. It has no integration acceptance and establishes no new theory, novelty, learning result or performance advantage. The completed findings-only follow-up verdict is **REQUEST CHANGES**: B1 and B4 remain partial. The two review cycles are complete; this packaging performs no further review or source implementation.

## Final recorded verification

| Evidence | Actual result |
|---|---|
| Existing combined research, literature and evidence-tool run | **1,668 passed, two failed, three skipped; exit 1** |
| Final affected-file run | **420 passed, two failed, one skipped** |
| Exact original 26 failing nodes | **24 passed, two failed; exit 1** |
| Saved PA2 retrieval | **FAIL: 2/4 against required 3/4** |
| Original named-gate guards | **FAIL: 9/10 and 3/6** |
| Follow-up | **REQUEST CHANGES; B1 and B4 partial** |

The combined result is read from local `finish-combined.log` and `finish-combined.exit`; no suite was repeated for packaging. Both failed nodes are in `hybrid/tests/research/test_phase7_replay.py`: `test_every_s2_death_gets_its_expected_phase7_code` and `test_the_linear_attention_and_delta_rule_reductions_are_excluded_as_families`. Sixteen of 19 historical S2 deaths still miss their exact required code. The tagged-memory case cannot be preserved by substituting a plain delta rule for its state-dependent gates. Original assertions, labels, thresholds and acceptance predicates remain unchanged. Empty check output is not a test pass.

This record supersedes earlier handoff statements that the combined run or follow-up was still active, or that the worktree had not been packaged. Earlier reports and both reviews retain their snapshot context. See `finish-REPORT.md`, `finish-PHASE7-REMAINING.md`, `../final-review/FOLLOWUP.md`, `../kd8/REPORT.md` and `../r2-bar/SCORE.md` for limitations and blocker evidence.

## Selective contents

The package contains 56 changed or new hybrid files: literature/research sources, prompts and schemas, regression tests, and separately saved reference evidence. The 20-file `fixtures/reference_replays/` set totals 6,012,074 bytes, including four compact trace JSONs, unchanged saved source programs, provenance and checksums required by the fixture-integrity tests. All 19 listed fixture checksums match. Adjacent Jeffrey source/provenance and saved top-K evidence support the targeted gate regressions.

Also included are the dated root `DECISIONS.md` clarification, corrected KD8 and r2-bar reports/scores, changed extraction/scoring scripts, the tested KD8 report generator and evidence-tool tests, both completed review reports, compact repair reports/status records, this checkpoint and the reusable saved-trace capture script. Historical worker reports remain attributable to their own snapshots.

Logs, raw model calls, stores, cache, suite scratch, temporary output, full captured result manifests and the large tested diff stay local. Scratch `add_references.py` and `dbg.py` are excluded. A scoped `results-wt/.gitignore` protects generated output while allowing selected reports and tools. `kd8/census.jsonl` and `kd8/replay.json` are removed only from Git's index; both local files retain their exact original bytes. Their prior versions remain in Git history.

## Packaging checks and branch boundary

Staged content is checked for whitespace and credential patterns; no credential patterns were found. Frozen-path diff checks cover `hybrid/math`, root tests, `bigbrane`, `EXPERIMENT.md`, `README.md`, `expected.json`, root status/defect documents and the family configuration. No source implementation is changed by this packaging task.

The sibling branch is prepared for push to `git@github-acoliver:acoliver/bigbrane.git`, ref `refs/heads/r2-bar`, using verified GitHub account `acoliver`. The coordinator creates the draft PR. No PR is opened or merged here. No integration into `p4-diagnosis` occurs. Main's separate dirty `STATUS.md` and `KNOWN-DEFECTS.md` remain untouched. B1/B4 and the recorded acceptance failures remain follow-up work under the existing rules.
