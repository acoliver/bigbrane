# Relocation complete; PA2 remains FAIL 2/4

The HTTPS endpoint fix and its tests now reside only in sibling `r2-bar` at `fe264b5`. Main `p4-diagnosis` at `9c01df3` has clean tracked code and index. No commit, push, merge, root-document edit, host-reference/schema edit, research-test run or acceptance rerun occurred.

All 71 original diagnostic files, totaling 6,462,401 bytes, were copied and hash-verified. Main keeps every original evidence file. The exact source diff, HTTPS source and new test are preserved under placement-originals/ in both trees. Main's untracked test was removed only after validating these copies. Repository-local info/exclude ignores /results-wt/; nothing was staged. PLACEMENT.md explains the receipt, shared worktree exclusion and exact scope.

## Tests and provenance

Sibling full literature suite: **190 passed, 1 live-only test skipped**. Diagnostic test_diagnose.py: **11 passed**. Both use main's .venv-hybrid, sibling cwd/PYTHONPATH and explicit sibling pytest config. Exact commands and exit codes are in placement-verification/tests.json. Full saved replay with socket connections prohibited reproduces identical offline.json bytes and all historical ranks. All 274 saved replay inputs verify unchanged; configured-key scan and redaction regression pass.

Production scope is the arXiv HTTPS URL. The sibling parsing test now asserts that same URL without changing term escaping or result checks. The copied MockTransport regression proves actual requests need no redirect and preserve pacing/cache behavior. Diagnostic-only historical cache pinning keeps the original HTTP-keyed inputs available without production fallback. PLACEMENT-VERIFY.json and FILEMANIFEST.json hold current hashes; COMMIT-INVENTORY.md excludes raw HTTP, caches and output datasets from a future checkpoint.

## Unchanged retrieval losses

Original PA2 remains **FAIL 2/4 against required 3/4**. Saved pools contain 3/11 registered works across 2/4 cards. Current 35/60/80 bounds expose only Sacramento, rank 25. Rossi is acquired but rank 128; Zhou is acquired but rank 256. Hwang's raw-bibliography leads rank 65/66 before the five-seed selection and are never expanded. The other acquisition misses remain as documented in REPORT.md; diagnostic targeted results were never inserted into pools or acceptance.

Current exact-ID availability is not original-query coverage. The direct HTTPS endpoint removes the measured extra HTTP-to-HTTPS exchange only. Its contribution to the historical 429s is unknown. Initial 429 cause, historical attempt count, S2 availability, provider matching beyond saved windows and documentation/thesis indexing remain unresolved. No new live calls or diagnostic retries were made during placement. This package does not claim integration readiness or a 3/4 retrieval pass.
