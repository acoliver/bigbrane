# PA2 retrieval failed; the published-mechanism guard passed

The saved PA2 experiment retrieves at least one registered work for 2/4 incremental cards. The registered requirement is 3/4, so retrieval FAILS even under this generous interpretation. Four published-mechanism controls die at S2. These results describe historical snapshots, not the final repaired tree. No live searches, retries or model calls were made to prepare this report.

## Criteria

| Criterion | Status | Evidence and limits |
|---|---|---|
| PA2 registered retrieval, at least 3/4 incremental cards | FAIL | 2/4 using any registered work per card; exact id or full normalized title only |
| PA2 registered guard, all four selected published mechanisms die at S2 | PASS | 4/4; Rule 110 is other_known, not a nonstandard RNN activation |
| PA2 not tuned to referee outcomes | PASS for this implementation | No new live runs or attempts; historical missing works remain missing |
| KD8 saved annotated label-interpreter replay | PASS within its synthetic scope | Historical replay.json has 14 records, including Jeffrey/IPF; declarations were supplied by hand or mechanically |
| KD8 live Jeffrey/IPF preservation after repair | NOT TESTED | No new live Jeffrey run in the saved PA2 experiment |
| KD8 executable declarations bound to published operations | NOT TESTED by these historical runs | Other worker owns the production fix and its targeted execution tests |
| Census fully classified for restatements | NOT TESTED | Four identified comparator restatements below; remaining population is not exhaustively adjudicated |
| Historical suite3 all green | FAIL | 1589 passed, three skipped, one prompt-copy failure |
| Full combined suite on final repaired tree | NOT TESTED | Driver will delegate after both implementation workers finish |
| R2-bar original Rule 1 and Rule 2 named-gate guards | FAIL | Historical 9/10 and 3/6; see ../r2-bar/SCORE.md |
| Novel theory, subquadratic learning and sample efficiency | NOT TESTED | Runs stop at S2, with no mathematical or learning evidence |

## All saved live runs

Complete means run_end reason=complete and no queued unfinished cards. Source failures are shown separately; completion is not scientific acceptance.

| Run | Completion | Terminal/stage error events | Failed model calls | Saved/ledger calls | Run wall s |
|---|---|---:|---:|---:|---:|
| flipA | complete | 0 | 0 | 38/38 | 1335.8 |
| flipB | complete | 0 | 0 | 28/28 | 1118.6 |
| probe-card-17100001 | complete | 0 | 0 | 7/7 | 263.4 |
| probe-card-17100003 | complete | 0 | 0 | 7/7 | 224.4 |
| probe-card-17100006 | complete | 0 | 0 | 7/7 | 225.5 |
| probe-card-17100008 | complete | 0 | 0 | 7/7 | 238.3 |

All saved call transcripts were attributable to an explicit primary card id. A history reference to another card is not an attribution. Attempt spans are paired only inside the same card.

## Per-card literature cost and outcomes

Wall covers query proposal start through classification result. The parenthesized value starts at the query-proposal receipt, after that proposal model call. Search receipts count source-level searches, including the local index, cache hits and skips. Actual HTTP request/attempt counts are NOT AVAILABLE. Related lookups can perform multiple HTTP requests and are counted separately.

| Card | Run | Search / related | Source receipts | Literature model calls / requests | Wall s (after proposal) | Source errors | Related errors | Candidates / fields | Saved S2 |
|---|---|---:|---:|---:|---:|---|---|---:|---|
| card-3485a13f | flipA | 18 / 5 | 72 | 3 / 3 | 98.65 (95.138) | arxiv:17 | none | 282 / 8 | reduces_to_known |
| card-8154d3b4 | flipA | 16 / 5 | 64 | 3 / 3 | 124.368 (120.947) | arxiv:16 | none | 243 / 8 | pass |
| card-f5ee7edb | flipA | 18 / 5 | 72 | 3 / 3 | 140.568 (137.008) | arxiv:18 | none | 297 / 8 | pass |
| card-89248ee5 | flipB | 18 / 5 | 72 | 3 / 3 | 169.901 (165.735) | arxiv:11 | none | 303 / 8 | fail |
| card-a4f684fb | flipB | 18 / 5 | 72 | 3 / 3 | 118.062 (114.867) | none | none | 269 / 7 | pass |
| card-17100001 | probe-card-17100001 | 18 / 5 | 72 | 3 / 3 | 177.169 (173.714) | none | none | 271 / 8 | reduces_to_known |
| card-17100003 | probe-card-17100003 | 18 / 5 | 72 | 3 / 3 | 131.045 (127.428) | none | none | 264 / 7 | reduces_to_known |
| card-17100006 | probe-card-17100006 | 16 / 5 | 64 | 3 / 3 | 143.128 (140.369) | none | none | 251 / 8 | reduces_to_known |
| card-17100008 | probe-card-17100008 | 16 / 5 | 64 | 3 / 3 | 155.634 (152.064) | arxiv:4 | none | 272 / 7 | reduces_to_known |

Each classification has three prior-art model calls and three recorded request attempts: query proposal, refinement and classification. The old flow had two model calls, so the increase is one, not two. This corrects the accounting claim in DECISIONS.md without editing that document. flipA has 51/52 arXiv search failures; flipB has 11/36. The other sources, including a complete local arXiv snapshot, sometimes provide usable coverage; a failed source is neither evidence of absence nor by itself a reason to discard all results.

## Exact registered retrieval and actual decision exposure

Any named work counts toward the generous card-level retrieval score. Missing works are retained. Decision positions come from saved prompt listings, not the candidate rank or today's selection rule.

| Card | Work | Candidate rank(s) | Actual exposed decision positions |
|---|---|---|---|
| card-3485a13f | Hwang | MISSING | not shown |
| card-3485a13f | Srinivas | MISSING | not shown |
| card-3485a13f | Silver | MISSING | not shown |
| card-f5ee7edb | Rendle | MISSING | not shown |
| card-f5ee7edb | Rossi | 128 | not shown |
| card-a4f684fb | SciPy QAP 2-opt | MISSING | not shown |
| card-a4f684fb | Rehn | MISSING | not shown |
| card-89248ee5 | Jiang | MISSING | not shown |
| card-89248ee5 | Zhou | 258 | not shown |
| card-89248ee5 | Sacramento | 27 | classifier:27; difference_checker:3; critic:27; difference_writer:3; difference_writer:3; difference_checker:3 |
| card-89248ee5 | Stoklosa | MISSING | not shown |

Rossi is stored at rank 128 but not exposed to the classifier, critic or difference listing. Zhou is stored at rank 258 and likewise not shown. Sacramento at rank 27 is shown to the classifier and critic at position 27 and to the difference writer/checker at position 3. The underlying per-call ids and exact matches are in live.json. Nothing here reclassifies a missing source as retrieved.

## Census and annotation limits

The saved census has 64 counted comparison rows in 51 artifact paths: {'other_known': 56, 'rnn': 8}. Its recorded artifact-level outcomes are {'refuted': 4, 'verified': 47, 'unverified': 13}. The prior collection log reports 327 examined artifacts; the census only stores matching rows and cannot independently reconstruct that denominator. It omits per-row within_tolerance and executable verification data, so artifact-level verified must not be read as every row being verified.

Four identified counted-and-verified restatements are fixed_quadratic_rnn on card-8f2e71cf, elman_rnn_identity_w and rnn_diff_sum_basis on card-136a1e9b, and block_sequential_ca_moore on card-383bd914. They insert the card's nonlinear operation into a purported published template. Jeffrey/IPF on card-ea50345f is a named published comparator in the historical record. The remaining census is not a reviewed population-wide classification, and these examples cannot establish a prevalence estimate. The historical 14-record annotated replay verifies how labels are interpreted, not that a live critic chooses or executes the correct published operation.

## Recovered historical suite3

| Log | Passed | Skipped | Failed | SHA-256 |
|---|---:|---:|---:|---|
| suite3/g0.log | 326 | 1 | 0 | 3e72ef80ae8bed20af6a7a196b800674cbdb35fdda37bad9a1ba3d47f1647304 |
| suite3/g1.log | 284 | 1 | 0 | eba2e4b46a4e75678e5c1e2d11b3cac60f9904febe0113b98f9ff3875e1fed00 |
| suite3/g2.log | 163 | 0 | 0 | a88c576fc5725805353c5f555464abf111569c3fdb46891e97ff3d638cf0f19f |
| suite3/g3.log | 219 | 0 | 1 | 2053b8e301cc87adac83f36a82bb1b4e2e573e5c5b86e74a4f669ef53dc14aa9 |
| suite3/g4.log | 162 | 0 | 0 | ab010141495790f270b73ae99a82fd1cbdf7d32cb9afe5860712ee375e62c836 |
| suite3/g5.log | 254 | 0 | 0 | c00f04a26403d86e2fd4c73dfbfa4c71416846f2d7ce6eac47755feead799b14 |
| suite3/g6.log | 181 | 1 | 0 | 401f5ee6fac7d1338432998da362cb38426ddd695dec3494d8f2375bb949725a |

Total: 1589 passed, 3 skipped, 1 failed. g3 failed test_orchestrator_units.py::test_prior_art_prompt_copy_matches_the_literature_module. These logs predate the repaired tree; no per-shard code hash was captured. The review observed that all three current prompt strings already matched the current document at fa08b93. A current targeted check and a historical suite failure can both be true. Neither is a final full-suite success.

## Provenance and remaining work

Inputs are the six saved runs under results-wt/kd8/runs, census.jsonl, replay.json and suite3/g0.log through g6.log. extract_live.py reads card-tagged ledger events and saved transcripts; report.py reads its derived live.json. Raw runs, stores, prompts and logs were not rewritten. Historical flip artifacts lack query_fields/round2_queries added in later snapshots, while the query events retain the fields. live.json lists missing artifact fields separately rather than claiming no tagged queries ran. No saved run contains the new coverage/listing payload from this implementation.

The final-tree retrieval target is NOT TESTED, and the saved 2/4 remains FAIL. Do not tune or retry to force 3/4. The distinct-query and bounded-exposure defects are addressed in source and targeted tests, but their effect on retrieval has no new live measurement. Final combined verification is delegated by the driver after the two workers finish. No result establishes complete novelty.
