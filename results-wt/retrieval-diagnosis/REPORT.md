# B4 retrieval still fails; current ranking exposes only Sacramento

The unchanged saved pools contain 3 of 11 registered works across 2 of 4 incremental cards. Current ranking and field selection still expose only Sacramento under the 35/60/80-paper bounds. The original retrieval result remains **FAIL, 2/4 against the unchanged 3/4 requirement**. This investigation identifies acquisition limits, two downstream selection losses, and a transport defect. It does not establish integration readiness or a new theory.

All eight registered arXiv papers are available by exact ID from both arXiv and OpenAlex. Their absence from a saved pool therefore cannot be explained by those papers being unavailable today. Local metadata also contains all eight. For the five missing arXiv works, local all-term matching rejects every original query; eligible any-term results then fall outside the six returned records. Historical remote arXiv coverage was separately curtailed by rate-limit cooldowns. OpenAlex/S2 query behavior cannot be reconstructed beyond their saved returned windows.

## Scope and evidence

Main is `p4-diagnosis` at `9c01df3`; the sibling is `r2-bar` at `fe264b5`. The endpoint fix and current diagnostic package now reside in the sibling. Paths below are relative to the sibling unless explicitly labeled main. Original main evidence is preserved; see PLACEMENT.md for relocation receipts. Another worker's historical-reference edits were left untouched.

Read inputs include sibling `results-wt/review-fixes/LITERATURE.md`, `results-wt/final-review/FOLLOWUP.md` B4, and `results-wt/kd8/{REPORT.md,live.json,STATUS.md}`. The saved ledgers, source-cache bodies and model transcripts supply the detailed trace. No original report, ledger, transcript, predicate, preregistration, expected file, guard or failed row was rewritten. No model request, review cycle, PA2 rerun, merge, commit or push was made.

Evidence files:

- [QUERIES.md](QUERIES.md): every original proposal/refinement query for all four cards, field/round, per-source result counts, failures, exact matches, all twenty expansion seeds and the thirty titles actually shown to each refinement call.
- [offline.json](offline.json): parsed source records before deduplication, complete returned citation/reference identities, used edges, query support, aliases, original ranks, all actual prompt mentions/listings and current bounded positions. It includes missing FTS components for each original query against each indexed registered work.
- [MEASUREMENTS.json](MEASUREMENTS.json): denominators, errors, request totals and local eligibility counts.
- [raw-audit.json](raw-audit.json): raw API records before parsing, bibliography identifiers in search hits, their owning cards and reconstructed initial seed-selection ranks. Live diagnostic identities are used only to identify saved bibliography links, never to change replay inputs.
- [source-plan.json](source-plan.json), [primary-plan.json](primary-plan.json), their `*-results.json`, and [http.jsonl](http.jsonl): diagnostic-only requests, parameters, status codes and redacted receipts. These results never enter replay or acceptance.
- [postfix-results.json](postfix-results.json) and [postfix-http.jsonl](postfix-http.jsonl): direct-HTTPS validation and exact public-source fetches.
- [PRESERVATION.json](PRESERVATION.json): original investigation checksums and pre-relocation state. [PLACEMENT-COPY.json](PLACEMENT-COPY.json) and [PLACEMENT-VERIFY.json](PLACEMENT-VERIFY.json) record preserved bytes, current code hashes and branch restoration. The original report and status are retained under placement-originals/.

### Counting rules

The four incremental cards each ran fourteen tagged initial queries plus four untagged refinements: **72 queries, 288 source-search receipts and 20 related lookups**. Distinct initial fields per card are 8, 8, 7 and 8. The complete flipA/flipB runs additionally include card-8154d3b4, giving **88 queries and 352 source-search receipts**. Counts below do not silently mix those populations.

Source-search receipts include local-index reads, cached results and cooldown skips. They are not HTTP attempt counts. Historical HTTP exchanges, redirects, per-attempt waits and intermediate statuses were not saved. Three exhausted receipts declare four retry iterations each, but do not establish twelve observed HTTP 429 responses or the total network count.

## First provable loss for each registered work

“Acquisition” means no exact registered identity/title appears in any saved source window or any returned edge of the selected seeds, before pool ranking. Where an external API's internal matching/ranking is unobservable, the table says so. No paper's absence alone is treated as proof that a vocabulary query could not find it.

| Card / work | First provable loss or successful path | Evidence and limits |
|---|---|---|
| 3485a13f / Hwang, 1110.5667 | Direct acquisition miss plus a demonstrable unexpanded bibliography path: seed-selection ranks 65/66 exceed five | Present locally and by both live exact-ID lookups. Any-term eligible in 17/18 local queries, all-term in zero. No saved direct paper record or returned seed edge contains it. However, the original DreamCoder query returns two works whose raw OpenAlex bibliographies identify Hwang; neither is selected as a seed. Its paired original query answers arXiv 200/zero and OpenAlex six other works. Provider internal rank/matching remains unknown. |
| 3485a13f / Srinivas, 2204.09837 | Acquisition: local all-term rejection / six-record acquisition bound; remote arXiv cooldown | Present locally and by both exact-ID APIs. Any-term eligible in 15/18 queries, all-term eligible in zero. The proposal never uses “experts” or “memory bounds”; it does search hypothesis pruning and other memory vocabulary. The original beam-search pair returns arXiv 200/zero and OpenAlex five other works. This establishes that paired query's result, not that every possible query fails. |
| 3485a13f / Silver, 1904.06317 | Acquisition: local all-term rejection / six-record bound; remote arXiv cooldown | Present locally and by both exact-ID APIs. Any-term eligible in 13/18 queries; all-term in zero. Original program-induction query returns arXiv 200/zero and OpenAlex six other works. No saved direct or returned edge match. Source availability, deduplication and later prompt bounds do not explain this missing pool entry. |
| f5ee7edb / Rendle, Factorization Machines | Acquisition from supported source windows; observed proposal vocabulary omits factorization machines | Exact local title absent. OpenAlex targeted title search returns the exact 2010 paper first, DOI `10.1109/ICDM.2010.127`. None of the eighteen original queries names factorization machines; graph/higher-order vocabulary is present. No saved raw source or expansion edge contains Rendle. The paired graph query returns Rossi, not Rendle. The evidence does not distinguish OpenAlex/S2 internal query matching from ranking beyond the returned six. |
| f5ee7edb / Rossi, 1906.05059 | Retrieved, then ranking/bounded deciding selection | The original “graph motif two-path counting link prediction higher-order” query returns Rossi from both local arXiv and OpenAlex. Search-level dedup keeps the local arXiv identity. That single query supplies one vote, not two. Stored rank 128; no refinement title, deciding listing or other saved prompt mention. Current rank remains 128, outside 35/60/80. Live OpenAlex repeats the hit at position 2; availability is not the loss. |
| a4f684fb / SciPy QAP 2-opt | Acquisition/source-type coverage gap, not a deciding-list truncation | Actual SciPy documentation is publicly retrievable by exact URL. No exact local title or saved direct/edge match. The original query does search QAP, but not SciPy or 2-opt. OpenAlex's targeted descriptive-title search returns other QAP papers, not the documentation. All eighteen historical arXiv searches answered without source errors. Neither OpenAlex nor S2 global indexing of this page is established. |
| a4f684fb / Rehn thesis | Acquisition from the configured literature windows; external thesis exists | Author publication page and exact thesis PDF answer HTTP 200. Exact local title absent; targeted OpenAlex title search yields six other works. Original queries include permutation groups and conjugacy, but not stabilizer/BSGS terminology. No returned seed edge contains the thesis, even before the six-edge cutoff. S2's present cooldown prevents a targeted availability conclusion there. Do not describe this as proof that all scholarly indexes exclude the thesis. |
| 89248ee5 / Jiang, 1409.0334 | Acquisition: local all-term rejection / six-record bound; arXiv later loses coverage | Present locally and by both exact-ID APIs. Any-term eligible in 12/18 queries, all-term in zero. The saved sparse-binary sequence query answers remote arXiv successfully before the later cooldown, but does not return Jiang. Its live pair answers arXiv 200/zero and OpenAlex six other works. No saved raw or returned edge match. The miss cannot be attributed solely to the later 429. |
| 89248ee5 / Zhou, 0901.0553 | Retrieved, then ranking/bounded deciding selection | Local arXiv returns Zhou only for the original refined Adamic-Adar query. Stored rank 258, no deciding listing or other prompt mention. Current rank 256, still outside every requested bound. No metadata loss or dedup disappearance. Live exact-ID APIs succeed, while the paired original query returns no arXiv results and six other OpenAlex works. |
| 89248ee5 / Sacramento, 1207.7196 | Successful acquisition and exposure | The original Willshaw query retrieves it locally. Stored rank 27. Refinement title position 21; classifier/critic position 27; two writer and two checker listings each position 3. One additional critic prompt mentions the ID outside those listings. Current rank and bounded position 25. Live exact-ID APIs succeed; a zero-result remote keyword query does not erase the saved local retrieval. |
| 89248ee5 / Stoklosa, 2411.14480 | Acquisition: local all-term rejection / six-record bound; arXiv later loses coverage | Present locally and by both exact-ID APIs. Any-term eligible in all eighteen original queries, all-term in zero. The sparse-binary sequence query succeeds historically before the 429 but omits Stoklosa. Live pair returns arXiv 200/zero and OpenAlex six other works. No saved direct/edge match. This is an observed bounded acquisition miss, not evidence that the paper is unindexed. |

The five missing arXiv works are Hwang, Srinivas, Silver, Jiang and Stoklosa. The other three missing entries are Rendle, the SciPy documentation and Rehn. Rossi, Zhou and Sacramento are the three retrieved works.

### What the local query trace establishes

The local source uses title/abstract FTS with stemming, all terms first and an any-term BM25 fill when fewer than six all-term hits exist. All eight registered arXiv records have **zero** all-term matches across their own card's eighteen original queries. Any-term eligibility is measured directly against each target's indexed row, not inferred from a missing result. The fallback is reached in 17/18 Hwang-card queries and 18/18 queries for the other indexed cards. Returned IDs then show which eligible works make the six-record window.

For the paired original queries, the following components are absent under that exact local FTS tokenizer:

| Work | Saved query | Missing all-term components |
|---|---|---|
| Hwang | Bayesian program learning library abstraction DreamCoder | library; DreamCoder |
| Srinivas | beam search pruning top-K hypotheses sequence model | beam; search; pruning; top; k; hypotheses; sequence |
| Silver | in-context program induction from input-output examples | context; input; output |
| Rossi | graph motif two-path counting link prediction higher-order | two; path; counting |
| Jiang | sparse binary codes sequence memory lag-one lag-two transitions | sparse; codes; memory; lag; one; two; transitions |
| Zhou | Adamic-Adar common neighbours inverse log degree link prediction | neighbours; inverse; log; degree |
| Sacramento | Willshaw binary associative memory clipped Hebbian outer product | clipped; Hebbian; outer; product |
| Stoklosa | sparse binary codes sequence memory lag-one lag-two transitions | binary; codes; lag; two |

These are local title/abstract measurements. The remote arXiv adapter searches `all:` fields, so local component failures alone do not prove remote query behavior. The paired live requests provide the remote observations separately. No query was shortened or replaced to improve a referee outcome.

### Expansion, identity and actual exposures

All twenty saved seeds, complete returned edge lists and the first six references/citations actually used were reconstructed. Across the four cards, those calls returned 235 reference records and 113 citation records, including repeated identities across seeds. **None of the eleven registered works appears anywhere in those returned expansion windows.** Consequently, the six-edge expansion cutoff did not discard an otherwise returned registered work. The upstream reference cap of 100 and citation cap of 50 remain unobserved beyond their windows.

A raw-body audit adds a more specific Hwang path. The original query “Bayesian program learning library abstraction DreamCoder” returns `openalex:W3174357804`, DreamCoder, and `openalex:W4310557242`, Top-Down Synthesis for Library Learning. Their saved `referenced_works` arrays contain Hwang's independently resolved OpenAlex identity at positions 3 and 8. Reconstructing the initial acquisition order reproduces the actual five selected seeds and places these two leads at initial ranks **66 and 65**, outside that bound. Hwang was therefore one untraversed reference away from two returned works. It was never a returned paper record, candidate or prompt listing. The full-title query was not needed to establish those saved links; targeted lookup was used only to identify the opaque reference IDs. Expanding the source windows or selecting different seeds was not performed.

The same audit finds eleven raw bibliography occurrences pointing to Zhou: nine belong to papers retrieved for f5ee7edb, not Zhou's registered card, and two belong to 89248ee5's refined-query hits. Those refined hits arrive after seed expansion and are not expanded. Zhou already enters that card directly from the local source, so its deciding loss remains rank/bounded selection. No bibliography link to the other missing works is identified by the resolved identities available here.

Raw external paper records in the audited cache envelopes contain titles and IDs; none of the missing registered works appears as a raw paper record discarded by parsing. The only matching external paper record is Rossi. Local hits are audited separately through saved IDs and the read-only index. Bibliography IDs are not relabeled as retrieved papers.

Ten of the twenty seeds have empty reference and citation results. The seeds are held fixed during replay. Re-selecting seeds would require different acquisitions and is outside an unchanged-pool replay.

No registered raw hit disappears in search deduplication or pool merging. OpenAlex's live exact-ID records omit an arXiv identifier for five of the eight works, but their full titles still identify them. No corresponding saved raw exact hit was found and then lost because of missing metadata. Rossi's full title contains a subtitle; its exact arXiv ID establishes the registered match without fuzzy title scoring.

All saved card-attributed prompts were checked for exact ID and full-title mentions, in addition to listing positions. Only Sacramento appears. Its refinement call is `call-835065a47d00`; the six deciding listings and the additional ID mention are individually recorded in `offline.json`. Rossi and Zhou are not hidden in another prompt section.

The registered SciPy label, “Quadratic assignment problem (2-opt),” differs from the actual page heading `quadratic_assignment(method='2opt')`. The registration and matcher were left unchanged. This identity limitation would need explicit treatment before claiming a future fetched page satisfies that exact-title predicate; a related SciPy paper would not substitute for the registered documentation.

## Current offline exposure on unchanged acquisitions

The tool reconstructs saved per-source results, asserts exact returned-ID agreement and result counts, and reproduces **every historical candidate key and rank** for each card. Used cache envelopes predate each saved search completion; their hashes are retained. The local index is opened read-only. Original source results, selected seeds, returned edges and query texts are pinned. Current `_Pool` and `select_candidates` come from the sibling repaired code. Original stored candidate display metadata is retained.

| Card | Saved / replay merged pool | Saved registered works | Saved deciding exposure | Current registered ranks | Current 35 / 60 / 80 exposure |
|---|---:|---|---|---|---|
| 3485a13f | 282 / 278 | none | none | none | none / none / none |
| f5ee7edb | 297 / 295 | Rossi | none | Rossi 128 | none / none / none |
| a4f684fb | 269 / 268 | none | none | none | none / none / none |
| 89248ee5 | 303 / 301 | Zhou; Sacramento | Sacramento | Zhou 256; Sacramento 25 | Sacramento / Sacramento / Sacramento |
| Total | 1151 / 1142 | 3/11 works, 2/4 cards | 1/11 works, 1/4 cards | 3/11 works retained | 1/11 works, 1/4 cards at each bound |

The nine-record reduction is current cross-source identity merging, not new retrieval or removal of a registered work. Rossi has one supporting query: two multi-query records and 125 earlier single-query records precede it. Its optimization field's reserved representative is rank 127, `arxiv:1902.06679`, “Link Prediction via Higher-Order Motif Features,” rather than Rossi at 128. Zhou has one untagged refined-query hit, so it has no field reservation; six multi-query, 205 single-query and 44 related-only records precede it. Those ties use discovery order. The 30-title default field selection also exposes only Sacramento. This replay measures ranking and field selection, not a counterfactual model judgment or new citation choice. Required-evidence reservations can change downstream selection order; the saved evidence never requires Rossi or Zhou, so those reservations do not rescue them here.

| Card | Queries / fields | Source receipts | Source errors | Related errors |
|---|---:|---:|---|---|
| 3485a13f | 18 / 8 | 72 | arXiv 17 | 0 |
| f5ee7edb | 18 / 8 | 72 | arXiv 18 | 0 |
| a4f684fb | 18 / 7 | 72 | none | 0 |
| 89248ee5 | 18 / 8 | 72 | arXiv 11 | 0 |

All proposed fields have at least one answered alternate source in the saved transport receipts. That transport fact does not establish topical recall or novelty. The original guard and named-gate results were not rescored or altered: saved PA2 published guard 4/4, named-gate failures 9/10 and 3/6 remain their existing results.

## Rate limiting: shared cooldown explains most reported failures

| Saved run | arXiv query receipts | Exhausted 429 receipts | Cooldown skips | Successful receipts |
|---|---:|---:|---:|---:|
| flipA, including 8154d3b4 | 52 | 2 | 49 | 1 |
| flipB | 36 | 1 | 10 | 25 |

The process-shared client is cached by local-index path in sibling `hybrid/literature/__init__.py:46-50`. Its transport checks cache before cooldown and sets a 600-second source-wide cooldown after exhausted HTTP 429 retries. Pacing is shared across processes using the same worktree's `var/slots`; cooldown itself is in process memory.

In flipA, f5ee7edb's first query starts at 07:16:06 UTC and its exhausted receipt finishes at 07:16:46.532. The 8154d3b4 query starts at 07:16:29 before that first failure is recorded and finishes exhausted at 07:17:14.355. Hwang's card begins literature searches at 07:17:38, inside the already active cooldown, and completes them by 07:19:04.986. Its seventeen failures are skips, not seventeen independently observed overload responses. The one successful refined query is compatible with the cache-before-cooldown path and cannot be counted as cooldown recovery.

In flipB, 89248ee5's eighth query starts at 07:14:04 and finishes exhausted at 07:14:43.527. The remaining ten searches finish by 07:15:47.609, all within the cooldown. Earlier queries, including the sparse-binary query relevant to Jiang/Stoklosa, had answered successfully. Back-to-back work inside one process therefore loses an entire source after one exhausted request. Resetting clients or extending attempts would hide that measured behavior, so neither was done.

The ledgers do **not** retain actual attempt times, intermediate statuses, Retry-After values, competing request logs or redirect chains. They cannot establish whether the initial historical 429s arose from aggregate local traffic, retry timing, an upstream quota, an edge limiter or service overload. Current source-only arXiv calls all answered at measured paced attempt gaps of at least 3.0003 seconds. That establishes present availability, not the cause of the historical rate limits.

## Independent source diagnostics

Credentials were loaded only through project `load_api_keys()`, from environment/private files. The actual environment names are `BIGBRANE_S2_API_KEY` and `BIGBRANE_OPENALEX_API_KEY` (`hybrid/literature/keys.py:32-35`). OpenAlex uses an Authorization header and S2 uses `x-api-key`. No macOS credential store, key suffix, raw token or authorization header was recorded.

The bounded plans performed **54 operations**: 36 reached the transport and 18 were cooldown skips. HTTP hooks recorded **52 exchanges: 35 HTTP 200, 16 HTTP 301 and one HTTP 429**. All are explicitly `diagnostic=true`, `acceptance_input=false`. There were no retries in diagnostics; `max_retries=0` bounds measurement without changing production retry policy. The normal shared pacer preserves arXiv 3 seconds and S2 1.1 seconds. Only one S2 exchange occurred, so there is no live S2 interval estimate.

- Eight exact arXiv lookups and eight OpenAlex DOI-alias lookups returned the registered papers. The request form was project `get_paper('arxiv', id)` or `get_paper('openalex', 'doi:10.48550/arXiv.' + id)`.
- Eight **original saved queries**, unchanged, were paired across arXiv/OpenAlex. Seven arXiv queries returned zero; the QAP query returned one different paper. OpenAlex returned the registered Rossi paper at position 2 for its original graph query; the other paired returned windows contained none of their registered targets. Exact query strings and every response identity are saved.
- Rendle's targeted exact-title OpenAlex search returned the correct paper. Targeted SciPy/Rehn title searches returned other works. This is a bounded title-search observation, not proof of global index exclusion.
- The first S2 exact-ID request included the configured key and returned HTTP 429 without Retry-After. The other eighteen S2 operations were skipped by the existing cooldown. Key configuration does not prove authorization, available quota or index coverage. No S2 availability claim is made for any registered work.
- Exact public URLs independently answered HTTP 200: [SciPy QAP 2-opt](https://docs.scipy.org/doc/scipy/reference/optimize.qap-2opt.html), [Rehn's publication list](https://www.math.uni-rostock.de/~rehn/publications.html), and [Rehn's thesis PDF](https://www.math.uni-rostock.de/~rehn/docs/diploma-thesis-cs-rehn.pdf). Their bytes and hashes are preserved. They were fetched after source diagnosis, not inserted into a candidate pool.

## Production change and verification

The measured defect is the configured arXiv endpoint `http://export.arxiv.org/api/query`. All sixteen initial live arXiv operations produced HTTP 301 followed by HTTPS 200. HTTPX follows that redirect without another project-pacer slot, so every paced operation caused two exchanges. **The historical contribution of those redirects to rate limiting is unknown.**

The production change is sibling `hybrid/literature/sources/arxiv.py:11`, now using the HTTPS endpoint. The misplaced main edit was restored to its exact HEAD bytes after hash and diff validation. Query construction, result limits, retries, cooldown, ranking, prompts and acceptance rules are unchanged. This removes a measured extra exchange without asserting a retrieval improvement. The URL change also gives requests a different cache key; old HTTP-keyed entries remain untouched and are not migrated or used as a fallback.

Added `hybrid/tests/literature/test_arxiv_https.py`: both request builders must use HTTPS; search and lookup each use one exchange with unchanged three-second pacing; repeated searches remain cached. Two fresh source-only validation operations then produced **two HTTPS 200 responses and zero redirects**, with the original Hwang query still empty. Three public-source fetches bring that validation file to five HTTP 200 exchanges total.

The sibling URL assertion in `hybrid/tests/literature/test_parsing.py:37` now requires HTTPS. Its term escaping, conjunction and result-limit assertions are unchanged. This matches the intentional transport endpoint change; no acceptance predicate or threshold was changed. The copied HTTPS regression exercises actual HTTPX requests through MockTransport: an HTTP request would redirect and add an exchange, so the exact two-exchange assertion detects regression. It also preserves three-second pacing and search-cache behavior.

| Current verification | Result |
|---|---|
| Sibling full literature suite | **190 passed, 1 skipped**; the live test is skipped |
| Sibling diagnostic test_diagnose.py | **11 passed**, including historical-cache pinning and credential redaction |
| Full offline replay with socket connections prohibited | All four historical key/rank sequences reproduced; offline.json SHA-256 unchanged |
| Original replay inputs | All **274** saved input hashes unchanged |
| Main tracked worktree/index | Clean after the placement correction; exact receipt in PLACEMENT-VERIFY.json |

The current suites use main's `.venv-hybrid/bin/python`, sibling cwd and PYTHONPATH, explicit sibling config/confcutdir, and `BIGBRANE_LIVE=0`. Exact commands and logs are in placement-verification/tests.json. No research tests or new live diagnostics were run during relocation. The original main failure, **168 passed, 1 failed, 1 skipped**, remains in main-literature-tests.log as historical evidence; its old HTTP assertion is now corrected only in the sibling. Original earlier verification counts and collection errors remain in placement-originals/REPORT.md and the preserved logs.

`SavedLiterature` explicitly pins arXiv's historical HTTP cache key in the offline diagnostic reader. It does not consult newly fetched HTTPS entries or fall through to network. The regression populates distinct HTTP/HTTPS bodies, requires the old body and original hash, then requires failure when that old entry is absent. Production has no HTTP fallback or cache migration.

## Remaining evidence and next action

Keep the saved PA2 failure. The current ranking fix cannot retrieve eight absent works and does not raise Rossi or Zhou into deciding bounds. No referee identifiers, targeted searches or new vocabulary were added to production prompts or ranking.

The next justified production investigation is **per-HTTP-attempt literature provenance**: cache versus network versus cooldown, attempt/redirect status and timing, pacer wait and Retry-After, without credentials. Existing search receipts cannot distinguish the causes of the original rate limits. More retry attempts, shorter cooldowns, larger windows or a new PA2 run are not justified by this evidence. The HTTP-to-HTTPS repair is the only production change made here.

S2 remains unmeasured beyond its 429; OpenAlex's internal match/rank for missing works remains unobserved outside six returned results; supported connectors do not establish coverage of documentation or author-hosted theses; older cache/local metadata cannot establish a historical full-text search response beyond the stored windows. Resolve those source-evidence limits separately from acceptance. The sibling HTTP-URL assertion conflict is resolved by the authorized HTTPS expectation change; the remaining retrieval and source-evidence limits are unchanged.

### Reproduction

Use main `/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python`. For offline replay, cwd and `PYTHONPATH` must be `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`:

```text
python /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/retrieval-diagnosis/diagnose.py --worktree /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
python /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/retrieval-diagnosis/summarize.py
python /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/retrieval-diagnosis/audit_raw.py
```

Only `diagnose.py --plan <saved-plan.json>` performs live literature calls. It writes into isolated diagnostic caches; it is not needed to reproduce the saved-pool replay. `validate_transport.py` is the preserved live measurement tool; any future use must have sibling cwd/PYTHONPATH to exercise the relocated endpoint. It was not rerun. Do not rerun either live mode as an acceptance experiment. Use only the saved replay and offline tests to verify this package.
