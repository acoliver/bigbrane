> Finish-worker update: The finish worker completed S2 downstream coverage propagation and bounded receipt/excerpt exposure for the critic and both difference participants. The final focused literature/research/evidence-tool slice yields 299 passed and one skipped. Saved retrieval stays FAIL 2/4 against 3/4, and preregistered scores stay FAIL 9/10 and FAIL 3/6. Final affected-file verification is 420 passed, two failed and one skipped (finish-final-tree-targeted.log). Final combined verification is running. See finish-REPORT.md; the original worker report below describes its earlier ownership handoff.

# Literature and reporting fixes implemented; B4 still needs research-stage wiring

This implementation changes only the assigned literature module, its tests, and saved-evidence tools/reports. It makes no new searches or live model calls. No commits, pushes, merges, root decision/status edits, math changes, frozen inputs or acceptance changes were made. The concurrently edited research-stage files belong to the other worker.

## Original findings

| Finding | Implementation disposition | Evidence or remaining work |
|---|---|---|
| B4 coverage and bounded exposure | Literature payload FIXED; research-stage integration REMAINING | Result retains search/related failures, per-field coverage, query support, aliases, ranks, actual classifier positions, truncation and explicit novelty_unassessed. Helpers select_candidates, listing_receipt and render_receipt expose the same information to bounded downstream consumers. S2 must honor and persist those fields and use the helpers in critic/difference prompts. |
| B4 registered PA2 retrieval | FAIL preserved | Exact any-registered-work matching gives 2/4, below required 3/4. No tuning or new attempt was made. |
| B5 registered guard scoring | FIXED | Restored exact verdict/named-gate predicates; 9/10 and 3/6 remain NOT MET. All-dead 10/10 and 6/6 are separate secondary diagnostics. Reports no longer call these guards met. |
| B6 appearance-based ranking and stub retention | FIXED | Distinct normalized query sets, related-only support capped at one, cross-source identity merge, aliases and richer metadata retention. Repeated reference fanout cannot outrank two distinct queries. |
| N3 proposal diversity | FIXED | Boundary validation requires four distinct field tags and case/whitespace-normalized unique query texts. No synonym or topic keyword rules. Schema states the same contract and rejects exact duplicate objects. Local-index test now uses four fields. |
| N5 live extraction attribution and fuzzy matches | FIXED | Per-card and per-attempt spans, exact primary card id attribution, full normalized title or exact arXiv id, missing matches retained, stored candidate ranks separated from actual prompt positions. |
| N6 saved evidence/reporting | FIXED within historical scope | Added kd8 REPORT.md and live.json. All six runs complete with zero stage/terminal and model errors, but arXiv failures remain explicit. Recovered suite3 totals and checksums. Added protection for REFINE_SYSTEM alongside the other two prompt copies. Final combined suite remains NOT TESTED. |

## Decisions and limits

Usable transport coverage means at least one source answered a query in each proposed field; a complete local index counts as usable. One source failure, or one unanswered query when another query covers that field, does not automatically invalidate the search. All failures still reach the classifier. The classifier must separately state whether its available evidence covers the mechanism's update and readout steps. A single unrelated candidate cannot supply adequacy by existence alone. A novel result is explicitly unassessed when transport coverage is missing, no candidates exist, or the classifier reports inadequate mechanism coverage. Neither a completed query nor a model assessment proves exhaustive novelty.

Candidate ranking is distinct-query support followed by discovery order. Bounded selection reserves the highest-ranked representative of each requested field, then fills remaining positions by rank. Stored rank and exposure position remain different fields. The limits remain 60 classifier candidates, 80 critic titles and 35 difference papers; only the classifier and refinement callers are inside this worker's production-file ownership. Existing historical rankings are not rewritten.

The three prior-art model calls per classification are query proposal, refinement and classification. That is one more than the old two calls. HTTP request attempts cannot be recovered from source-search receipts, which include local-index, cache and skipped sources; the extractor reports HTTP requests as missing. Every card's three literature model request attempts are attributable from saved transcripts. The extracted evidence contains no raw prompts, responses or secrets.

PA2 retrieval is exactly the generous 2/4 reported in the original finding: Hwang/Srinivas/Silver missing for 3485a13f; Rossi found at rank 128 but Rendle missing for f5ee7edb; SciPy/Rehn missing for a4f684fb; Sacramento at rank 27 and Zhou at 258, with Jiang/Stoklosa missing for 89248ee5. Rossi and Zhou were not shown to any deciding listing. Sacramento reached classifier/critic position 27 and difference position 3. This patch does not establish that the new ranking reaches 3/4, nor identify a single cause for every missed work. No retry was used to conceal that uncertainty.

The census stores 64 matching rows from 51 artifacts, with artifact-level outcomes 47 verified, 13 unverified and four refuted. It lacks per-row within_tolerance, so those numbers are not comparator-level verification counts. Four identified bad comparator restatements across the three KD8 cards and the Jeffrey/IPF example are described in kd8 REPORT.md. The remaining population is not exhaustively classified. The 14-record historical replay uses supplied annotations and only tests label interpretation.

Historical suite3 is 1589 passed, three skipped, one prompt-copy failure in g3. Per-shard code revisions were not saved. Those results predate this repaired tree, and no final combined full-suite success is claimed. Tests run in this worker are listed below, independently of those historical counts.

## Verification logs

All commands used main `/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python`, cwd `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, and `PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`. Test temporary files for the final run were kept in worktree `tmp/literature-delivery-pytest`.

- `literature-tests-1.log`: 38 passed on the first literature/local-index slice.
- `evaluation-tests-1.log`: collection error from an incorrect test root parent index. Corrected test path construction; no assertion changed to hide it.
- `literature-evaluation-tests.log`: 48 passed after fixing that test path.
- `literature-suite.log`: 185 passed, one skipped before the last prompt/coverage regressions were added.
- `literature-final-tests.log`: 205 passed, one skipped, one failed. The stricter scorer exposed an inconsistent synthetic fixture using verdict=fail alongside reduction-gate reasons. Corrected the fixture to emit the required reduces_to_known verdict for reduction-gate deaths; assertions and acceptance thresholds were retained.
- `literature-delivery-tests.log`: 206 passed, one skipped. Delivery command `python -m pytest hybrid/tests/literature hybrid/tests/research/test_local_index.py results-wt/kd8/tools/tests/test_evidence_tools.py -q --basetemp=tmp/literature-delivery-pytest`.
- `r2-score-final.log`: regenerated exact-rule SCORE.md; Rule 1 NOT MET 9/10, Rule 2 NOT MET 3/6.
- `extract-live-final.log`: saved PA2 extraction; retrieval FAIL 2/4, published guard PASS 4/4.
- `kd8-report.log`: initial report command lacked PYTHONPATH and failed import; `kd8-report-2.log` and `kd8-report-final.log` show successful regeneration with the required environment. No evidence input changed.

Raw verification logs, runtime runs and stores must not be included in commits. The Markdown reports, source tools, tests and derived live.json are the delivery artifacts. No review cycle was run.

## Research-stage integration handoff

Outside this worker's ownership, the S2 worker or driver must:

1. Honor a nonempty literature result's `novelty_unassessed` flag without a broader-search retry. The current original wrapper only checks empty candidates.
2. Persist coverage, listing, coverage_assessment, novelty_unassessed and its reasons in the prior-art artifact. Preserve them in decision prompts, without weakening the existing card schema.
3. Use `select_candidates` within the existing critic/difference bounds, respecting required cited/evidence entries. Call `render_receipt(result, actual_shown_keys, limit)` for each actual bounded listing so the critic and both difference participants see omissions, ranks and coverage failures.
4. Delegate final combined verification only after that wiring and the independent gate worker's changes finish. Root STATUS, KNOWN-DEFECTS and DECISIONS corrections remain with the driver/docwriter.

The saved retrieval miss, historical named-gate misses, missing HTTP-attempt data, incomplete census adjudication and absent final-tree experiment are evidence limitations, not successes produced by this patch.

## Final targeted confirmation

`literature-confirm-tests.log`: **207 passed, one skipped** in the literature suite, local-index tests and evidence-tool tests. The added regression retains an incomplete attempt without absorbing the next attempt's searches. `extract-live-confirm.log` and `kd8-report-confirm.log` regenerate the same saved-evidence outcomes after that change. `literature-diff-check.log` is empty: git diff --check found no whitespace errors in this worker's tracked files. These checks are implementation verification, not a review cycle or a combined research-suite result.
