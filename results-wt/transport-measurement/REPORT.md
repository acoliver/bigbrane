# Fresh requests validate the measured provenance; PA2 still fails

All twelve preregistered source operations completed through the instrumented literature client. They produced twelve source receipts and thirteen observed HTTP request/response exchanges: twelve HTTP 200 responses and one HTTP 429. The S2 request for card-f5ee7edb received the 429, then answered 200 after the existing retry policy's five-second wait. No operation was repeated outside that policy.

The four exact original queries returned fifty source-result positions, with no exact registered-work matches for their respective cards. This is a small, fixed diagnostic of four first queries, not a recall measurement over all saved queries. **Original PA2 remains FAIL, 2/4 against the unchanged 3/4 predicate.** No acceptance, ranking, deciding exposure, or novelty result was regenerated.

## Plan and original inputs

[PLAN.json](PLAN.json) was written at 2026-10-18T22:20:50.827362+00:00, before network execution. Its SHA-256 is:

```text
22a3b10aa88d985ce1e1b6bd1ff2eb9d2887f574df7eb75b869049c6c7abeadc
```

Main was `82580a6630c4b5c9155f00d603f96682d5818e7c`; the runtime sibling was `e0e2c5a5a5b8d2541428ae978c226a7f8f8f9b77`. The new provenance implementation is in the sibling. No production file was changed for this measurement.

Selection was mechanical: for each incremental card, find its initial proposal queries, then take the first matching `literature_search` event in original ledger order. There was no semantic selection, query rewriting, target-paper lookup, or added referee identifier. Each selected query ran once against arxiv, openalex, then s2, serially, with the original six-result limit.

| Card | Exact saved query | Query SHA-256 |
|---|---|---|
| card-3485a13f | exact match key value cache induction head copying | `c8b2f1f2dcf5ad14f55bccffc75b558acf0ba180c1e976c43cd436d0d276e250` |
| card-f5ee7edb | successor automaton learning from positive examples | `1322705b4452867b65b886a6c8d054e14bf1a2ef4642b703ecc7bb9f0888ac17` |
| card-a4f684fb | permutation automata group-structured state transitions | `bbec48499a57395574e94f2c6606abcd7f356dace1f5c4ec0011941029bb13b4` |
| card-89248ee5 | sparse distributed memory hard location activation Kanerva | `db0ee97f491634c907a5151f3ea45c8dcc752488dca743b39ac61923a753e9d2` |

The plan pins source order, adapter URLs and public parameters, request IDs, query hashes, original receipt sequence and SHA-256, original ledger hashes, runtime implementation hashes, and diagnostic runner/test hashes. Original receipt SHA-256 hashes cover the JSONL line bytes excluding the terminating newline. [FIXTURES.json](FIXTURES.json) retains those events and expected callback card IDs; its hash is `4b5874f955a921b473c7927c10f06baaeb7348e92022d9990ef37fffb146de67`.

[PRESERVATION-before.json](PRESERVATION-before.json) and [PRESERVATION-after.json](PRESERVATION-after.json) verify all **274 original saved input hashes unchanged**. Original runs, source caches, model transcripts, failed rows, scoring, and predicates were not written. Other workers own seed and reference work. After this measurement finished, the seed worker changed `seed.py` and `build_seed.py`; their filesystem timestamps and before/after hashes are recorded in [RUNTIME-HASH-CHANGES.json](RUNTIME-HASH-CHANGES.json). Transport, client, adapters, sink, ledger, and the pinned diagnostic runner/tests still match the plan. Main's tracked working tree remains clean.

## Actual runtime and settings

The single live run started at 2026-10-18T22:23:11.435066+00:00 and ended at 22:23:28.088626+00:00. [RUN-ONCE.json](RUN-ONCE.json) is an exclusive sentinel that prevents repeats, including partial repeats. A prior managed shell launch exited before creating that sentinel, runtime directory, or any source record; its empty log and the corrected command are documented in [COMMANDS.md](COMMANDS.md). No source operation was repeated because of that shell-launch failure.

`Literature.search_with_receipt` used its real HTTPX transport, a new empty cache at `raw-cache`, and explicit remote-only source tuples. The default local index was not consulted, and no default configuration was modified. Historical local counts and identities are kept separately in [COVERAGE.json](COVERAGE.json), with `fresh_local_measured=false`.

Existing settings were retained: four attempts maximum, thirty-second HTTPX timeout, redirects enabled, five-second initial backoff, 120-second cap, and 600-second source cooldown after exhausted 429s. The process-shared pacer used the sibling's existing `var/slots`, with arxiv 3 seconds, openalex 0.1 seconds, and s2 1.1 seconds. No retry limit, timeout, header policy, or cooldown was changed; there was no alternate key/provider fallback. No network call followed completion.

Observed minimum gaps between this process's attempt starts were 3.003088 seconds for arxiv, 2.568762 seconds for openalex, and 2.423394 seconds for s2. These meet their configured intervals. Total measured pacer wait, including coordination overhead, was 0.729835 seconds. The one retry requested 5 seconds and measured 5.000308 seconds of sleep. Its Retry-After field was absent. These observations do not enumerate unrelated processes' traffic.

## Outcomes and coverage by card

| Card | Fresh arxiv / openalex / s2 records | Source receipts | Attempts / HTTP requests / responses | Observed statuses | Terminal source errors | Exact registered matches |
|---|---:|---:|---:|---|---:|---:|
| 3485a13f | 0 / 6 / 6 | 3 | 3 / 3 / 3 | 3 x 200 | 0 | 0/3 |
| f5ee7edb | 0 / 6 / 6 | 3 | 4 / 4 / 4 | 3 x 200; 1 x 429 | 0 | 0/2 |
| a4f684fb | 2 / 6 / 6 | 3 | 3 / 3 / 3 | 3 x 200 | 0 | 0/2 |
| 89248ee5 | 0 / 6 / 6 | 3 | 3 / 3 / 3 | 3 x 200 | 0 | 0/4 |
| Total | 2 / 24 / 24 | 12 | 13 / 13 / 13 | 12 x 200; 1 x 429 | 0 | 0/11 |

Fifty counts source-result positions before any cross-source deduplication. The diagnostic uses the existing exact arXiv-ID or normalized full-title matcher offline, after acquisition. These matches were never used to choose a query or change an acceptance input. A missing exact match here does not prove that a source cannot retrieve the work under another saved query.

[RESULTS.json](RESULTS.json) retains each query/source result's public identity, title, year, abstract, arXiv ID, DOI, safe errors, and full transport record. [SUMMARY.json](SUMMARY.json) retains request/call identities and per-card counts. Successful raw API envelopes remain local in `raw-cache`, with no raw commit or upload. [ARTIFACTS.json](ARTIFACTS.json) pins the finished artifact bytes, including those twelve raw envelopes, while excluding itself and generated bytecode.

All twelve returned source-ID lists, including empty lists, equal the corresponding saved receipt's ordered IDs. Empty identity-list equality must not be read as equal transport outcomes. The previously failed arxiv searches now answer successfully but still return nothing:

| Exact query's card | Saved arxiv receipt | Fresh arxiv observation |
|---|---|---|
| 3485a13f | Cooldown skip, zero records | One observed HTTPS 200, zero records |
| f5ee7edb | HTTP 429 after four declared attempts, zero records | One observed HTTPS 200, zero records |
| a4f684fb | Successful receipt, two records | One observed HTTPS 200, the same two identities |
| 89248ee5 | Successful receipt, zero records | One observed HTTPS 200, zero records |

The two returned arxiv works for a4f684fb are `2107.04683`, Decomposing Permutation Automata, and `1205.0682`, Groups Synchronizing a Transformation of Non-Uniform Kernel. Neither is one of that card's registered works. The twelve-source diagnostic does not reproduce the older eight-query paired plan and is not presented as its rerun.

## Receipt counts, HTTP counts, and attribution

The real `CardLiteratureSink` forwarded events to a new hash-chained `Ledger` at [runtime/ledger.jsonl](runtime/ledger.jsonl). It contains twelve `literature_search`, twelve `literature_receipt`, and twelve `literature_transport` events. The chain verifies. The extractor joins summaries and transports by call ID and checks card ID, exact query, source, limit, query association hash, expected request ID, endpoint origin, and the source receipt's result IDs. S2 transport source `s2` is explicitly mapped to receipt source `semantic_scholar`.

There were twelve network outcomes, zero cache hits, zero cooldown skips, zero local outcomes, and zero redirects. Arxiv and openalex each account for four attempts/exchanges. S2 accounts for five. Attempt counts are project retry iterations, HTTP request counts are hook-observed dispatches, and responses are observed response headers. They are not packet counts or proof of server receipt. Source receipts remain logical source-operation records, not HTTP counts.

The historical 288 source-search receipts for the four incremental cards remain historical receipts. Their total HTTP count remains unknown. This fresh run supplies no missing historical intermediate statuses, wait durations, redirect chains, or competing traffic logs.

## Tests and security

Seven scoped offline tool tests pass in [tests-final.log](tests-final.log). Synthetic records come through the real client, sink, and ledger using MockTransport and a fake clock. Tests interleave event delivery, distinguish two retry attempts from three HTTP exchanges across a redirect, retain a separate zero-network cached card, measure an exhausted four-attempt failure plus a zero-network cooldown, and reject cross-card ownership, missing/duplicate transports, and corrupted exchange counts. Credential and URL-query exclusion checks reject deliberately unsafe synthetic records. No full literature suite was duplicated.

The earlier unittest module invocation selected main's older package and failed import before tests ran; its traceback remains in tests.log. The direct-script invocation selects the sibling modules. The first offline post-run analysis also detected the concurrent seed changes rather than claiming all files were unchanged; analysis.log retains that failure, while analysis-final.log records the source-specific validation and completed offline comparison. No live request was repeated during either correction.

Keys came only through existing `load_api_keys()` environment/private-file handling, using `BIGBRANE_OPENALEX_API_KEY` and `BIGBRANE_S2_API_KEY`. Both were configured; no key value, suffix, header, or origin was printed. There was no OS credential-store access, gateway call, model call, alternate provider, or secret export. HTTP debug logging was disabled for the live run.

[SECURITY.json](SECURITY.json) records an in-memory scan for actual configured key values across diagnostic files, including raw envelopes. The scan found none. Transport records also pass recursive exclusion of headers, URL paths/queries, bodies, and redirect Location values; endpoint fields contain origins only. Public query text is retained in the planned fixtures and summary receipts, separately from transport payloads. These checks cover this diagnostic's artifacts, not arbitrary external service logs.

## Limits and next justified action

Fresh S2 availability is now observed for these four queries, including one normal-policy 429 retry. That does not establish why an earlier S2 diagnostic was rate limited, prove key authorization beyond the successful operations, or establish wider index coverage. Current arxiv successes do not explain the initial historical 429s or establish that the HTTPS endpoint change caused a recovery. Their cause remains unknown.

The current first-query searches add no registered work and no source-window identity beyond their corresponding saved receipts. They do not settle missing-work coverage for the other original queries, provider ranking beyond six results, documentation/thesis indexing, seed expansion, or deciding selection. No PA2 pass, integration readiness, or production acceptance is claimed.

The next justified step is to use these records as bounded evidence that request identities, actual counts, and card attribution are observable on real source calls. Keep the original PA2 failure and the remaining coverage investigation separate. A future acquisition investigation should preregister its scope and consume the new provenance; this evidence does not justify more retries, tuned queries, larger windows, or a regenerated PA2 result. No production defect was repaired in this task. The module-selection and background-launch friction is recorded for the coordinator in COMMANDS.md.
