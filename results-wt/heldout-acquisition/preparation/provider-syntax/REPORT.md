# Both arXiv forms parsed; two helper contract checks remain pending

This preparation establishes documented grammar and neutral-request parse acceptance for Q0 and Q1. The complete helper contract remains **PENDING** because the frozen builder accepts empty ASCII-token queries and unknown policy names. Neither gap was changed. These direct-helper gaps are constrained by existing higher-level metadata and arm validation; they did not affect the preregistered pair.

## Frozen scope

Main remains clean on `p4-diagnosis` at `3ce1516b406f8053763033bcc1362a2f5c8889bd`; sibling HEAD remains `8cad9a1cf600c53d58e65889e36d8248342c7984`. All writes are inside this provider-syntax directory. STATUS was persisted first, followed by PLAN and its SHA before any network.

Validation PLAN SHA256: `3dfb76418382d0c8f51816cadc8256a43fd214fdd87771807181a2b7293779c5`.

`FROZEN-UNCHANGED.json` confirms all 20 pinned input/source/shared-pacing files unchanged, including root PLAN, policy, exclusions, their pins and harness. No production/policy changes, commits, pushes, integration, study acquisition, independent review, full suite, LLM, gateway or credential access occurred. File tools rejected the sibling write path; shell commands wrote only the authorized directory, with sibling cwd/PYTHONPATH and the main virtualenv Python. Bytecode writes were disabled.

## Grammar and exact request contract

The fixed [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html) was fetched once. Raw snapshot SHA256: `a4f8c339c191e35addd5a658114015f3a44000de89ec15ba633836a6c5db57b9`. `docs/EVIDENCE.json` pins excerpts and line references in `docs/user-manual.txt`.

The manual labels `ti` as Title, `abs` as Abstract and `all` as “All of the above”, and states that `all:` searches the listed fields simultaneously. It documents AND/OR and parenthesized Boolean groups. It specifies `%28`/`%29` for parentheses and `%22` for double-quoted multiword phrases. `sortBy=relevance` is documented. No claim follows about exact provider-internal fulltext scope, analyzer behavior or ranking.

The neutral phrase was fixed before fetching: `electron proton`, directly illustrated in the manual quickstart. Q0 used production `arxiv.search_request`, HTTPS direct with no redirect: `all:electron AND all:proton`. Q1 used the exact frozen expression: `(ti:electron OR abs:electron) AND (ti:proton OR abs:proton)`. Both kept `start=0`, `max_results=6`, `sortBy=relevance`. All ASCII tokens, case, duplicates and order are preserved. No double quotes were added to single tokens; URI escaping was round-trip checked. This preserves P0 formatting without phrase or operator retuning.

`REQUEST-SPEC.json` records exact public parameters, encoded URLs and production request hashes. Actual frozen harness output was compared offline before the probe. Q1 was transcribed locally for live execution because importing the harness installs the network prohibition. OpenAlex/S2 raw query strings and all builder parameters/fields remained byte-identical, including punctuation, via direct and actual mock-Transport differential checks. No provider selection behavior was added.

## Criterion results

| Criterion | Status |
| --- | --- |
| Documented arXiv fields, operators, parameters and URI escaping | Passed |
| Q0 nonempty request parity with production | Passed, seven generic cases |
| Q1 every-token retention and exact grouping | Passed, seven generic cases |
| OA/S2 unchanged parameters and mock request/association identity | Passed, fourteen cases in each layer |
| Foreign parameters/metadata, source/card/call/query ownership and hash tampering | Passed |
| HTTP200 API errors excluded; missing total remains unknown | Passed |
| Safe transport secret projection and request identity | Passed |
| Bounded neutral Q0/Q1 live Atom parse | Passed: syntax-valid |
| Frozen helper empty-token rejection | Pending: six failing assertions |
| Frozen helper unknown-policy rejection | Pending: three failing assertions |
| Full request-contract entry criterion | Pending, not study accepted |

Final offline run: **68 passed, 9 expected failures, 0 unexpected failures**, 77 cases. Expected failures expose the two frozen gaps; they do not count as passed contract checks. Focused tests block sockets, DNS and datagrams through the existing process audit hook. Mock HTTP exercises the existing retry/redirect/cooldown policy with fake time. A read-only lease test verifies exclusion against another file handle, ingress/egress waits and unchanged shared stamp bytes. No live request ran in pytest or CI.

## Exact bounded runtime

One pair only: **2 arXiv logical operations, 2 attempts, 2 HTTP requests, 2 responses, 0 redirects, 0 retries, 0 cooldown**, plus one documentation request/response. Combined observed HTTP dispatches: **3**; preregistered combined bound: 189. No OpenAlex/S2 query was dispatched. Both responses were valid Atom feeds with six parsed records and no API error entries; no conclusion relies on HTTP200 alone. Missing totals are represented as unknown by the preparation metadata extractor, never zero. Stored totals were not used for query choice or semantic comparison.

The production Transport, source Request, provenance operation/event hooks and empty API-key mapping were used. Existing limits stayed at four attempts, 30-second timeout, backoff 5 with cap 120, cooldown 600, and at most 20 redirects per attempt. None of the retry/cooldown allowance was needed. No extra pair filled the unused allowance.

Read-only exclusive leases on both main/sibling existing arXiv locks surrounded the batch. Actual file-backed Pacer state was writable only here. Dispatch starts were at least three seconds apart; Q1 measured 2.431528 seconds of pacer wait. Both shared leases remained held for 3.007722 seconds after the batch; shared timestamps/locks stayed byte-identical. `runtime/RECEIPT.json`, `runtime/events.jsonl` and `RUNTIME-COMPACT.json` retain measured facts, distinct call IDs and request hashes. Raw bodies and docs remain local and unstaged.

## Exact helper gaps for coordinator delegation

1. `results-wt/heldout-acquisition/harness.py:128-134`: `search_request('', 'arxiv', 'Q0')` returns `search_query=''`; production `hybrid/literature/sources/arxiv.py:16-19` raises ValueError. Q1, punctuation-only and non-ASCII-only inputs share this gap. Existing `queries()` at lines 109-124 rejects such metadata, so this is direct-helper parity rather than a demonstrated valid-replay escape. A delegated implementation can align helper rejection with production; this worker made no fix.
2. `harness.py:128-139`: `search_request('alpha', 'arxiv', 'FOREIGN')` returns `(ti:alpha OR abs:alpha)` rather than rejecting the policy. OA/S2 also accept unknown policies while ignoring them. `replay()` validates arm membership first, so valid replay arms constrain this path. A delegated implementation can reject unknown policy names at the helper boundary. No fallback/provider-selection feature is warranted.

`DEFECTS.json` records complete minimal calls, actual outputs and expected behavior. The frozen harness stores projected request/record blobs, not raw provider error bodies or provider totals; raw-source parse evidence and unknown-total metadata belong to this separate preflight evidence. It would be incorrect to infer that offline receipt replay alone validates a live provider response.

## Exclusion handoff and remaining limits

**12 diagnostic identities** were returned and copied, with IDs, titles, supplied DOI, source URL/version aliases and body hashes, into `PROBE-EXCLUSIONS.json`. Its registry-ready entries use `other_used_study_data`; the frozen registry and other worker inventories were not touched. SHA256: `1dad768a0a4afff9d1e3823749590aa163f70134d59279658bf57e17b0c2b4ca`. Future mechanism/reference identity checks must consume this handoff and resolve remaining aliases. Reports contain counts only; identities and titles stay in the handoff/raw bodies/cache.

No held-out cards, labels, provenance or corpus choices were acquired or selected. Provider grammar/parse acceptance does not establish arbitrary-query acceptance, exact all-field scope, fulltext indexing, rank equivalence, recall, or global Q1 benefit. Top-six results may differ. OA/S2 live authentication and availability remain untested. The original PA2 result remains **FAIL 2/4** and the initial 429 cause remains **unknown**.

## Reproduction

Use main `.venv/bin/python` with sibling cwd/PYTHONPATH and `PYTHONDONTWRITEBYTECODE=1`. Run only `run_offline.py` for repeatable offline checks; it isolates pytest config/cache/temp here. Tests are in `test_preflight.py`, output in `tests.log`/`tests.xml`. The live stages `preflight.py docs` and `preflight.py probe` were executed once in separate processes outside pytest; durable STARTED markers prohibit repeating them. Do not delete markers or rerun live stages under this plan. No remaining live probe is needed for the reported syntax evidence.

## Implementation correction v2: both diagnosed helper contracts pass

The preceding report records the original implementation and its 77-case run of 68 passed and nine expected failures. That report text, its original logs, RESULTS, DEFECTS, input hashes, and live receipts are retained as historical evidence. The current implementation is v2; its receipts are separate.

The helper now rejects a policy outside Q0/Q1 before source validation or dispatch. After source normalization, the arXiv branch rejects an empty token list using the same `[A-Za-z0-9]+` grammar as metadata validation and the production adapter. These are two added `require` calls. Request formatting, every-token retention, source parameters, source aliases, budgets, retries, and schemas are unchanged. OpenAlex/S2 tokenless text is still passed through unchanged. Unknown sources still raise for valid policies.

The two xfail decorators were removed from the newly authored diagnostic tests, promoting exactly six empty-token cases and three unknown-policy cases. No test predicate was changed. The 44 original held-out tests remain byte-identical.

The new isolated run passed **244 tests: 44 original held-out, 77 provider preflight, and 123 focused correction tests**, with no failures, skips, or expected failures. The focused checks cover policy ordering, whitespace/non-ASCII arXiv rejection, unchanged OA/S2 text, source aliases, version pins, and AF_UNIX socket pairs. Existing audit checks continue to reject network sockets, DNS, and datagrams. The request expression and parameter construction are unchanged after the new checks; 72 valid combinations also matched the retained original source through 144 mock dispatches, including exact encoded request bytes and request/association identifiers. No provider HTTP traffic occurred.

Old harness SHA256: `1546667435c4206f384ded06862da66cfcc2b8105bf0828423f1a2bcaf73b867`.

New harness SHA256: `2c1d366af78edd96cb12048b86c4b5656a11f396a91ebb8b2a7094c1c00cc5f0`.

`INPUT-HASHES-v2.json` differs from the retained original input manifest only at the harness hash. The preparation freeze loader now selects that explicit version, with no fallback. Root `SOURCE-HASHES-v2.json` and `ARTIFACTS-v2.json` identify the corrected implementation; the previous manifests are unchanged. `helper-fix-v2/IMPLEMENTATION-v2.json` ties the versions, frozen pins, promoted cases, test results, and valid-request comparison together. `FROZEN-UNCHANGED-v2.json` records that the other 19 of 20 base pins, all production files, original test records, and original live artifacts remained unchanged. Inventory v1 is a historical pin to the old harness and was not rewritten by this correction.

The two diagnosed invalid-input helper checks are resolved. The corpus and broader inventory remain PENDING. The unchanged corpus has no cards, so no corpus query classification or semantic measurement is claimed. No independent corpus, labels, arm acquisition, contrasts, acceptance rerun, or policy benefit was established. There were no additional live queries or documentation fetches. The prior exact-once neutral pair is unchanged; PA2 remains FAIL 2/4 and the initial 429 cause remains unknown. No independent review was performed in this scoped implementation.

### Reproduce the v2 offline checks

From sibling cwd, use the main virtualenv Python with `PYTHONDONTWRITEBYTECODE=1`, sibling `PYTHONPATH`, and `TMPDIR` under `helper-fix-v2/temp`. Run `preparation/provider-syntax/run_helper_fix.py` relative to the held-out directory. The runner selects the root held-out `pytest.ini`, an explicit test list and `confcutdir`, disables plugin autoloading, and installs the unchanged offline audit guard before importing pytest. Cache, temporary paths, JUnit XML, parity evidence, and summary output stay under `helper-fix-v2/`. Exact arguments, cwd, interpreter, and temporary path are in `TEST-SUMMARY-v2.json`. Do not rerun the historical live stages or the old finalizers to regenerate historical receipts.
