# Fresh measurement complete; original PA2 still fails

Twelve preregistered exact-query/source operations completed through sibling e0e2c5a's instrumented real Literature client. Runtime: 2026-10-18T22:23:11.435066+00:00 to 22:23:28.088626+00:00. No additional source operations were made after completion.

Measured totals: twelve source receipts, thirteen retry attempts, thirteen HTTP dispatches, thirteen responses, twelve HTTP 200s and one HTTP 429. The S2 operation for card-f5ee7edb used one existing-policy five-second retry, then answered 200. Zero redirects, cache hits, cooldown skips, local calls, or terminal source errors. Twelve transport records reached the real per-card sink and a verified 36-event hash-chained ledger.

Fresh per-card arxiv/openalex/s2 result counts are 0/6/6, 0/6/6, 2/6/6, and 0/6/6. Fifty source-result positions yield zero exact registered-work matches under the existing diagnostic matcher. All returned ID lists equal their corresponding original receipts, but the two formerly failed arxiv operations now have successful empty responses. This establishes fresh availability, not the cause of historical rate limits or an acquisition improvement.

Plan SHA-256: 22a3b10aa88d985ce1e1b6bd1ff2eb9d2887f574df7eb75b869049c6c7abeadc. All 274 original saved-input hashes verify unchanged before and after. Transport/client/adapters/sink/ledger and pinned runner/test hashes are unchanged. Other-worker seed.py/build_seed.py changes have timestamps after the measurement and are recorded separately. They were not modified by this task.

Seven scoped tool tests pass. Interleaved attribution, redirect/retry versus exchange counts, cached and cooldown zero-network outcomes, count corruption, and secret/URL-query exclusion are covered. Actual configured key values are absent from scanned diagnostic artifacts, including raw API envelopes. Tests.log preserves an initial module-selection failure; tests-final.log records the successful direct-script invocation. No full literature suite was duplicated.

REPORT.md, PLAN.json, FIXTURES.json, SUMMARY.json, COVERAGE.json, RESULTS.json, runtime/ledger.jsonl, SECURITY.json, VALIDATION.json, preservation manifests, commands and local raw-cache retain the evidence. The first shell background launch never reached its sentinel or runtime directory; the actual managed launch ran once. COMMANDS.md records that friction and the offline analysis's detection of subsequent seed edits.

Original PA2 remains FAIL 2/4 against the unchanged 3/4 predicate. Historical initial 429 causes and historical HTTP counts remain unknown. Missing-work coverage outside these four fixed queries, provider windows, documentation/thesis coverage, expansion, and deciding selection remain unresolved. No production/scoring/test edits, acceptance rerun, LLM/gateway calls, review, commit, push, integration, or merge occurred.
