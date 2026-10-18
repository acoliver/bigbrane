# HTTPS fix relocated; main code restored

The endpoint fix and its tests belong to sibling `r2-bar` at `fe264b5996b7177d2fbcca5e9274ca5414210578`, rooted at `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`. Main remains `p4-diagnosis` at `9c01df3afa58c971e8745ec0344aafd1bc972b27`, rooted at `/Volumes/XS1000/acoliver/projects/bigbrane`. No commit, push or merge was performed.

## Preservation and restoration

All **71 original diagnostic files (6,462,401 bytes)** were copied to the identical sibling relative path, `results-wt/retrieval-diagnosis`, and verified byte-for-byte using SHA-256 before restoring main. `PLACEMENT-COPY.json` lists every original file, byte count and hash. Main retains all original evidence, including request receipts, caches, saved public sources, test failure logs and the original report/status. Neither the original failed PA2 rows nor replay inputs were rewritten.

Before correction, main's complete source bytes had to equal HEAD with exactly the single `BASE_URL` HTTP-to-HTTPS substitution. Its hash also had to match the earlier PRESERVATION.json. The exact `git diff --binary` was saved as `placement-originals/arxiv-main.patch` in both trees, together with the HTTPS source and new test. Only then was main's exact one-line change restored. No checkout/reset was used. The untracked main test was removed only after validating its identical sibling test and diagnostic copies in both trees.

| Preserved object | SHA-256 |
|---|---|
| Exact main source diff | `0399e817803362033fd45fcd63641605feedea56c4ea7782968d2875c6a69232` |
| HTTPS source copied to sibling | `be20d9f3f0ceb52b2b2317ea0f18f4256f031c171390ffc3374a9411f6d61c11` |
| New HTTPS test, copied without alteration | `97136a8cc659e1f1ce20b2680461dbaad96887d210c4ab2540cc440fd68eaa5c` |
| Restored main source, identical to HEAD | `5b09ce5f154b665aa0259a1f9b8a5c286b06705e8b3daf87a44b6e1e9bf447c2` |

The sibling report/status and offline reader have current placement information. Their original bytes are preserved under placement-originals/; original main diagnostics remain untouched. `PRESERVATION.json` describes the earlier investigation state. `PLACEMENT-VERIFY.json` describes the completed relocation, saved-input checks and current source/tool hashes. `FILEMANIFEST.json` inventories package files except itself; COMMIT-INVENTORY.md is a separate allowlist for a small future checkpoint.

The only ignore change is repository-local `.git/info/exclude`, adding `/results-wt/`. Worktrees share that exclude file, so runtime results are ignored in both trees. No tracked or global ignore was changed. No runtime evidence is staged. Other workers can checkpoint the listed human reports and tools by explicit path without including cache bodies, HTTP logs or output datasets.

## Scope of implementation

Production/test changes are limited to sibling `hybrid/literature/sources/arxiv.py`, the URL assertion at `hybrid/tests/literature/test_parsing.py:37`, and the copied `hybrid/tests/literature/test_arxiv_https.py`. Term escaping, query conjunctions, result limits, retry/cooldown policy, ranking, PA2 predicates and thresholds are unchanged. No source identifier was added to production. Existing host-reference, schema, research-test and root documentation files were not edited.

The regression observes actual HTTPX request URLs under MockTransport. HTTP would receive a fake 301, so the two-exchange assertion fails if a redirect is reintroduced. Search and lookup start directly on HTTPS, remain three seconds apart, and repeated search stays cached. Existing query-construction assertions remain intact.

Historical replay needs its original HTTP-keyed entries. Only the diagnostic SavedLiterature reader pins those keys, without trying HTTPS or network if one is missing. Its new regression proves it selects historical bytes over a distinct HTTPS entry and fails if the historical entry is absent. There is no production cache fallback or migration.

## Verification and clean-main receipt

Main's `.venv-hybrid/bin/python` ran with sibling cwd and `PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, `BIGBRANE_LIVE=0` and `PYTHONDONTWRITEBYTECODE=1`. Exact argv, cwd, environment and exit codes are in `placement-verification/tests.json`:

```text
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest -c /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/pyproject.toml --confcutdir /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2 -q --basetemp /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/retrieval-diagnosis/placement-verification/pytest-literature hybrid/tests/literature
# 190 passed, 1 skipped; exit 0
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest -c /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/pyproject.toml --confcutdir /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2 -q --basetemp /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/retrieval-diagnosis/placement-verification/pytest-diagnostic results-wt/retrieval-diagnosis/test_diagnose.py
# 11 passed; exit 0
```

The one skipped literature test is live-only. No research tests were run. Full saved replay was also run once with socket connections prohibited and no --plan. It reproduced all historical ranks and identical offline.json bytes: `fff61a1ce0c6a41e1423378875dd93c74c7716a1ad0ca69340f11289151563a3`. All 274 original replay-input hashes and configured-key redaction checks are verified in PLACEMENT-VERIFY.json. No new live calls, diagnostic retries, model calls, PA2 rerun or review took place.

Clean-main commands and their complete output are captured in `placement-verification/git-receipt.txt` and PLACEMENT-VERIFY.json: `git status --short`, `git diff HEAD --name-status` and `git diff --cached --name-only` all produce empty output. `git check-ignore -v results-wt/retrieval-diagnosis/http.jsonl` resolves to the repository-local exclude. Both worktree HEADs are unchanged. Sibling files outside this task were not edited. Their before/after hashes record concurrent research/reference work, so they are not claimed byte-unchanged. Those observations are in PLACEMENT-VERIFY.json; no concurrent changes were reverted.

## Retrieval remains failed

Original PA2 remains **FAIL 2/4, required 3/4**. Current bounded exposure is a separate measure: only Sacramento, at rank 25, appears under each 35/60/80 bound. Rossi remains rank 128; Zhou moves from 258 to 256, still outside every bound. Their losses are ranking and bounded selection after successful acquisition, not missing source records.

Hwang has no direct saved paper record; two acquired works contain its opaque ID in raw bibliographies, but their initial seed-selection ranks 65/66 exceed the five-seed bound. Neither bibliography lead was expanded. Hwang, Srinivas, Silver, Jiang and Stoklosa are all present locally and by current exact-ID APIs, but every local all-term query rejects its target, and any-term eligibility does not put it inside the six returned results. Historical remote cooldown also limits coverage; Jiang/Stoklosa were already missed by successful searches before the later rate limit. Rendle is missing from the original source windows; targeted title lookup establishes availability only. SciPy documentation and Rehn's thesis exist at exact public URLs, while coverage by the configured scholarly-query windows remains unestablished. REPORT.md and QUERIES.md retain the per-work evidence and limits.

The endpoint change removes a measured redundant exchange: sixteen old arXiv operations each had HTTP 301 then HTTPS 200; the two saved direct-HTTPS validation operations each had one HTTPS 200. It does not explain the initial historical 429s, repair the eight acquisition misses, raise Rossi/Zhou into bounds, or satisfy the 3/4 PA2 requirement. Exact-ID source availability does not establish original-query coverage. S2 availability and provider behavior outside saved windows remain unknown. No retry increase or acceptance change is included.
