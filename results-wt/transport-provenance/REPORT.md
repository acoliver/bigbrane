# Transport attempts are recorded; historical retrieval still fails

The literature boundary now records logical retry attempts and HTTPX request/response exchanges, including redirects, under call-scoped ownership. Existing per-card sinks forward these events to the ledger. This does not improve or rescore retrieval. Original PA2 retrieval remains **FAIL 2/4 against 3/4**, with named guards **9/10 and 3/6**. The cause of the initial historical 429 remains unknown.

Worker verification produced **79 targeted passed**, **212 literature passed/one live-only skipped**, and **12/12 unchanged offline differential cases** against `eeaba07`. The final combined suite and selected-package check are recorded in [NEXT-CHECKPOINT.md](../evidence-recovery/NEXT-CHECKPOINT.md). No live API/model calls, PA2 rerun, new review, integration or acceptance change occurred. The latest work has not been independently reviewed.

## Ownership and delivery

`provenance.py` owns invocation ContextVars and HTTPX hooks. `transport.py` measures the existing pacer, GET and sleep boundaries. `client.py` binds search, lookup and related operations to the caller's existing sink. Local-index calls receive zero-network records. Public method and subclass signatures remain unchanged.

`SearchReceipt.transport`, `Fetched.provenance` and failed direct calls' `SourceError.provenance` expose records. At invocation completion, including exceptions, `literature_transport` events go through the same sink. Summary events receive `transport_call_id`. The existing `CardLiteratureSink` forwards arbitrary event types with the card ID; a regression uses the actual sink and hash-chained ledger with deliberately interleaved calls. No research-file wiring change is needed.

Each invocation has a random `call_id` and SHA-256 `association_id` of the exact query or source-qualified paper identity. Each transport request uses its existing cache-key SHA-256 as `request_id`. Graph lookup, references and citations remain separate requests within one invocation. Context is reset in `finally`; neither transport nor client stores a mutable last receipt. Records are retained only for the active invocation, with existing retry/redirect bounds. An abrupt process termination can lose that unfinished invocation.

## Measurement boundaries and schema

`attempt_count` counts project retry iterations, each containing one `httpx.Client.get`. `http_request_count` counts request-hook dispatches, including redirected hops and sends that fail before a response. `http_response_count` counts observed response headers. These boundaries do not prove server receipt or packet counts. Cache, cooldown and local operations have zero attempts and HTTP requests.

Attempt start follows the pacer. `pacer_wait_s` includes lock contention; attempt duration covers GET/body processing and excludes retry sleep. Exchange duration ends at received headers, or the failure boundary if no headers arrive. UTC timestamps use the injected wall clock; elapsed values use the injected elapsed clock, monotonic by default. Absolute elapsed-clock readings cannot be compared across unrelated processes.

| Record level | Fields |
|---|---|
| Identity | `source`, origin-only `endpoint`, `operation`, `association_id`, `request_id`, `call_id` |
| Call timing/outcome | `started_at`, `started_s`, `duration_s`, `outcome`, `terminal`, nullable `error_type`, `cooldown_remaining_s` |
| Counts | `attempt_count`, `http_request_count`, `http_response_count`, bounded `attempts` |
| Attempt | One-based `ordinal`, `started_at`, `started_s`, `duration_s`, `pacer_wait_s`, nullable `http_status`, `exchanges`, `redirect_chain`, `retry_after_kind`, nullable `retry_after_s`, `retry_delay_s`, `retry_wait_s`, `terminal`, nullable `error_type`, HTTP request/response counts |
| Exchange | One-based `ordinal`, origin-only `endpoint`, `started_s`, `duration_s`, nullable `http_status` |
| Redirect entry | Responding hop's origin-only `endpoint` and `http_status`; next exchange gives destination origin |

Call `outcome` is `network`, `cache_hit`, `cooldown` or `local`. Call `terminal` is `success`, `skipped`, `http_error`, `exhausted`, `parse_error` or `error`. Attempt terminal is `success`, `retry` or `error`. A null exchange status means no response headers were observed. Source names retain adapter spelling, including `s2`. `CardLiteratureSink` adds `card_id`.

`retry_after_kind` is `absent`, `delta`, `date` or `invalid`. Parsed HTTP dates are measured against injected UTC time while retaining the existing exponential-backoff policy. Delta values keep the existing 120-second cap. `retry_delay_s` records requested sleep; `retry_wait_s` measures actual elapsed sleep. Early-returning sleeps are not extended. Minimum intervals remain arXiv 3 seconds, OpenAlex 0.1 seconds and S2 1.1 seconds.

## Safe synthetic examples

[examples.json](examples.json) retains complete fake-clock records, without raw query text, URL paths, headers or bodies. [behavior-comparison.json](behavior-comparison.json) contains the twelve differential results and pinned baseline hash.

| Synthetic case | Attempts | HTTP requests | Responses | Measured behavior |
|---|---:|---:|---:|---|
| 429 with delta 7, then 200 | 2 | 2 | 2 | Seven-second requested and elapsed retry wait |
| 301 then HTTPS 200 | 1 | 2 | 2 | One pacer slot; no additional project redirect wait |
| Four exhausted 429s | 4 | 4 | 4 | Starts 0/5/15/35, sleeps 5/10/20, 600-second cooldown |
| Distinct request during cooldown | 0 | 0 | 0 | `cooldown`, `skipped` |
| Cached request during cooldown | 0 | 0 | 0 | Cache checked first |
| Timeout then 200 | 2 | 2 | 1 | No invented first status; unchanged retry sleep |
| HTTP-date 40 seconds ahead | 2 | 2 | 2 | Parsed value 40; policy sleep remains 5 |

Handlers that do not advance the fake clock produce zero durations. Other tests advance it inside response handling and assert nonzero measurements.

## Verification and reproduction

The 22 added cases cover success, Retry-After delta/date/invalid/cap, exhaustion, cooldown/cache ordering, timeout, redirect followed by timeout, 404, parse failure, local/missing index, real graph calls, failed lookup delivery, shared-client reuse, fail-fast excessive redirects, early-returning sleep and interleaved per-card ledger attribution. Original client/key assertions and the existing HTTPS endpoint remain unchanged. The broader cache-free isolated package has one additional baseline failure at `test_seed.py:34`: the unchanged seed-list provenance test requires the ignored historical seed build log. All 22 new instrumentation cases and the client/key cases pass in that archive. No raw log was added or assertion suppressed.

`verify_behavior.py` loads the pinned pre-change transport from Git and drives both versions through identical MockTransport cases. It compares request/header hashes, request starts, sleep arguments, returned results, final fake-clock values and successful cache-envelope hashes. Each invocation uses fresh synthetic caches. It does not call a live service or change a production setting. `verification.json` records the worker's tested hashes and commands; its no-combined-suite fields describe that worker run, not the later checkpoint verification.

From the checkout root, with main's existing virtual environment:

```sh
export PYTHONPATH="$PWD/results-wt/evidence-recovery/next-network-guard:$PWD"
export BIGBRANE_LIVE=0
export TMPDIR="$PWD/tmp/transport-reproduce"
mkdir -p "$TMPDIR"
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest \
  -c "$PWD/pyproject.toml" --confcutdir="$PWD" hybrid/tests/literature \
  --basetemp="$TMPDIR/pytest" -q
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python \
  results-wt/transport-provenance/verify_behavior.py
```

Raw worker logs remain local: `targeted-complete.log`, `full-literature.log` and `behavior-comparison.log`. They are excluded from the package, along with generated caches. Earlier exploratory logs are preserved without overwriting them.

## Security and historical limits

New transport records omit URL paths/queries, redirect Location values, headers, response bodies, account/contact identifiers and credential origins/values. `SourceError.url` removes URL user information. Transport and parse errors record class names rather than arbitrary external messages. Project search warnings use hashed query identity. Identity hashes permit correlation; they are not encryption and do not prevent dictionary matching.

Existing search/get/related summaries and response-cache envelopes retain their prior contracts, including query text and configured contact parameters. Third-party raw HTTP debug logging is outside the new payload. No raw HTTP logger was added. Normal exceptions remain fail-fast. Downstream inspectors would need separate work to summarize the new event type; historical null HTTP counts must remain null.

The original four incremental cards produced 72 queries and 288 source-search receipts, not 288 measured HTTP attempts. Saved flipA evidence has two exhausted arXiv receipts plus 49 cooldown skips; flipB has one exhausted receipt plus ten skips. Neither those receipts nor this implementation establishes intermediate historical statuses, redirect counts, Retry-After values, actual waits, competing traffic or the initial rate-limit cause. Measurements apply to future instrumented calls only. They establish no improved acquisition, ranking, source availability or novelty assessment.
