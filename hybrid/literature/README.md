# hybrid/literature

Paper search, caching, the seed reading list, and prior-art classification for the research loop.
Sources are arXiv (keyless), OpenAlex and Semantic Scholar (both take an API key, see "API keys"). A local arXiv metadata index
(`arxiv_local`, harvested over OAI-PMH) sits in front of them so a rate-limited remote source no longer leaves a search empty.

## What it provides

| Name | Where | Purpose |
| --- | --- | --- |
| `Paper` | `paper.py` | One record per work: source, source_id, title, authors, year, abstract, url, venue, citation_count, fetched_at, plus arxiv_id and doi for matching. |
| `search(query, sources, limit, sink)` | `__init__.py`, `client.py` | Searches the requested sources, dedupes across them, and returns `list[Paper]`. |
| `Literature.search_with_receipt(...)` | `client.py` | The same search, returning a `SearchReceipt` with per-source hits and errors. |
| `get_paper(source, source_id, sink)` | `__init__.py`, `client.py` | Looks up one work by its source id. |
| `related(paper, sink)` | `__init__.py`, `client.py` | References and citations through OpenAlex, then Semantic Scholar. Best effort. |
| `sample_for_lens(lens, k, seed)` | `sampling.py` | Draws papers for one lens from the seed list and the logged search hits. |
| `classify_prior_art(card_text, llm_complete, sink)` | `prior_art.py` | Model proposes queries, searches run, model labels the card from the hits. |
| `python -m hybrid.literature keys [--live]` | `__main__.py`, `key_report.py`, `keys.py` | Shows which API keys are configured and where they come from; `--live` tests each one. |
| `python -m hybrid.literature harvest` | `__main__.py`, `harvest.py`, `oai.py` | Resumable arXiv OAI-PMH harvest into the local SQLite index. |
| `arxiv_local` source | `sources/arxiv_local.py`, `arxiv_index.py` | BM25 search over the local index. |
| JSON schemas | `schemas.py` | Query proposal and classification outputs for the model calls. |

The module-level functions use one shared `Literature` configured from the environment. Tests build their own `Literature` with a mocked HTTP client and a fake clock.

## Event sink

Pass any object with `append(type, actor, data)` as `sink`. The research ledger satisfies this. The module emits:

- `literature_search`: query, sources, ts, limit, `per_source_ids`, `errors`, `result_count`, and `local_index` when `arxiv_local` was requested. Emitted once per search.
- `literature_get`: source, source_id, ts, from_cache.
- `literature_related`: paper key, via, reference and citation ids, errors.
- `prior_art_queries`: the queries the model proposed.
- `prior_art_result`: label, grounded evidence, notes, queries, candidate keys, rejected evidence.

Failures are recorded in the `errors` field of the event. A source that fails does not drop the others.

## Sources and rate limits

| Source | Endpoint | Pacing | Notes |
| --- | --- | --- | --- |
| arXiv | `http://export.arxiv.org/api/query` (redirects to https) | One request every 3 seconds | Atom XML parsed with `xml.etree`. Version suffixes are stripped from ids. |
| OpenAlex | `https://api.openalex.org/works` | 0.1 seconds between requests | Abstracts are rebuilt from `abstract_inverted_index`. Needs a free API key since February 2026 (sent as `Authorization: Bearer`). `mailto` is added only when `BIGBRANE_CONTACT_EMAIL` is set; OpenAlex ignores it now. |
| Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/...` | One request per second | Without a key the pool is shared and strict. A key (sent as `x-api-key`) gives a dedicated 1 request per second. |

On HTTP 429 or 5xx, the transport waits (honoring `Retry-After` when numeric, else 5, 10, 20 seconds) and retries up to 3 times. When the retries run out on 429, the source enters a 600-second cooldown. During the cooldown, requests to that source are recorded as `skipped` errors and make no network call. Cache hits are still served during a cooldown.

## API keys

OpenAlex has required a free API key since February 2026 (docs: https://developers.openalex.org/api-reference/authentication). Semantic Scholar issues keys on request (docs: https://api.semanticscholar.org/api-docs/). Bigbrane looks for each key in this order:

1. The environment variable: `BIGBRANE_OPENALEX_API_KEY` or `BIGBRANE_S2_API_KEY`.
2. The file `~/.bigbrane/keys/openalex` or `~/.bigbrane/keys/semantic_scholar`, with surrounding whitespace stripped.

No keychain is used. Create a key file like this:

```
mkdir -p ~/.bigbrane/keys && chmod 700 ~/.bigbrane/keys
printf '%s
' 'YOUR-KEY' > ~/.bigbrane/keys/openalex && chmod 600 ~/.bigbrane/keys/openalex
```

A key file that group or others can read stops the program with an error that names the file and asks for `chmod 600`. This applies to `default_literature()` (so the shared search functions, `build_seed` and the research loop) and to `hybrid.research run`, which checks before it creates the run directory. The check also applies when the environment variable is set, since the file is exposed either way.

What is sent: OpenAlex gets `Authorization: Bearer <key>`, Semantic Scholar gets `x-api-key: <key>`. Both are headers, never query parameters, so the key is absent from URLs, from the cache (its keys and envelopes hold only the URL and the query parameters), from `SourceError` texts, and from the events and receipts. `ApiKey` hides its value in `repr` and `str`, and error text from the HTTP library is passed through `ApiKey.redact` before it is recorded. Cached responses are shared between keyed and keyless runs.

Pacing: requests to one source are spaced by a `Pacer` (in `transport.py`) that is thread-safe, because the prior-art searches run in worker threads. `default_literature()` gives every instance the same process-wide `Pacer`, so the Semantic Scholar spacing of 1 request per second holds across the whole process, and with a key it matches the dedicated rate. A `Literature` built directly (tests, ad hoc use) takes the keys and pacer it is given and reads no environment or files.

Check what is configured:

```
.venv-hybrid/bin/python -m hybrid.literature keys          # presence, origin, last 4 characters
.venv-hybrid/bin/python -m hybrid.literature keys --live   # plus one search request per found key, prints the HTTP status
```

`--live` exits 1 if any status is not 200, and 2 if a key file is too open. Without it no network call is made. Only the last 4 characters of a key are printed.

## Local arXiv index

The remote sources fail under shared-IP rate limits (Semantic Scholar failed 102 of 108 searches in the last runs, OpenAlex failed all 99 in run 2). The local index holds arXiv titles, abstracts, authors, categories, ids and dates for the fields the research loop searches, and answers without a network call.

### Harvest

```
.venv-hybrid/bin/python -m hybrid.literature harvest --data-dir data/arxiv-index --log-dir results/literature-20261011
.venv-hybrid/bin/python -m hybrid.literature info    --data-dir data/arxiv-index
```

The harvest uses arXiv's OAI-PMH endpoint `https://oaipmh.arxiv.org/oai` (the March 2025 rewrite; the old `export.arxiv.org/oai2` is retired) with `metadataPrefix=arXiv`, one `ListRecords` request at a time, following resumption tokens. It sleeps 5 seconds between requests (`--min-interval`), waits out `Retry-After` on 503 and 429, backs off 30 to 600 seconds on other 5xx and transport errors, and gives up after 8 failed attempts at one request (the saved state still resumes). The User-Agent names the tool and carries no address.

Sets harvested: `cs`, `stat`, `q-bio`, `physics:nlin` as whole archives; `math:math:{ST,PR,OC,DS,IT,CT,AT}`, `physics:cond-mat:{dis-nn,stat-mech}` and `physics:physics:{bio-ph,data-an}` as single categories. arXiv publishes one OAI set per category, so the narrow sets are requested directly instead of streaming `math`, `physics:cond-mat` and `physics:physics` (about 1.4 million records) to filter them afterwards. `SetPlan(..., categories=...)` can still filter a wide set after harvest. A paper listed in several sets is stored once.

Resuming: the resumption token is stored in the same transaction as the page's records, so a stop at any time (SIGINT, SIGTERM, a crash, a reboot) resumes at the first page that was not stored. SIGINT and SIGTERM stop cleanly after the page in flight; a second signal stops at once. Run the same command again to resume. Finished sets are skipped. arXiv says tokens expire daily; if the server rejects a stored token (`badResumptionToken`) the set restarts from its first page, which is safe because records are upserted by arXiv id. `--from YYYY-MM-DD` limits sets that have not started to records modified since that date (for refreshing a finished index into a new data directory), `--sets` selects a subset, `--max-pages` bounds a trial run.

Deleted records (`status="deleted"`) are removed from the index. A record without an id or title is logged and counted as malformed, not fatal.

Files:

| Path | Content |
| --- | --- |
| `data/arxiv-index/arxiv.sqlite` | `papers` table, `papers_fts` FTS5 index (porter stemming; title, abstract, authors, categories), `harvest_state` per set. SQLite WAL mode, so searches can run during a harvest. |
| `data/arxiv-index/manifest.json` | Harvest start and update times, run status, snapshot timestamp, record count, `created` and `datestamp` ranges, category filters, per-set pages, seen, kept, filtered, deleted, malformed counts and token. A copy goes to the log directory. |
| `results/literature-<date>/harvest.log` | One line per page: counts, records per minute, server response date. |

OAI-PMH reports no list size, so the harvest cannot show a percentage. Expected counts per set come from the arXiv API (`results/literature-20261011/expected-counts.json`, see `STATUS.md`).

### Search

`arxiv_local` takes the keyword queries the prior-art step proposes: bare words and optional `"quoted phrases"`. Terms match title and abstract with stemming (not authors or categories). Results containing all terms rank first by BM25 (title weighted 3, abstract 1); if fewer than `limit` match, results containing any term fill the rest. Stopwords are dropped, hyphens split words, and every term is quoted before it reaches FTS5, so words like `NOT` and `NEAR` are searched as words.

The papers it returns have `source="arxiv"` and the arXiv id, the same identity as the remote arXiv source, so dedupe merges them and `arxiv:<id>` grounding works. The source name `arxiv_local` appears in `per_source`, `per_source_ids`, errors, and receipts. `Paper.fetched_at` is the snapshot timestamp (the server's response date for the latest stored page).

`Literature(local_index_path=...)` enables it, and `Literature.default_sources` then starts with `arxiv_local`. `classify_prior_art` searches `default_sources`, so S2 queries the local index first and merges with arXiv, OpenAlex and Semantic Scholar, deduping by arXiv id with the local record winning. Remote failures and cooldowns stay in `errors` but no longer empty the search when the index has hits.

If the file is missing, the source is skipped: `errors["arxiv_local"]` reads `skipped: local index not found at <path>` and the event carries `local_index: {"path": ..., "status": "missing"}`. An unreadable file is recorded as `local index unreadable`. A schema version mismatch raises.

Config: `literature.local_index_path` in the run config (default `data/arxiv-index/arxiv.sqlite`, relative to the repository root). `hybrid/research/stages/s2_prior_art.py` resolves it and passes it to `classify_prior_art`.

Receipts: the search event's `local_index` holds `path`, `status`, `record_count`, `snapshot_date` and `complete` (true only when every harvested set finished). S2 turns each `arxiv_local` search into a `literature_receipt` with `source: "arxiv_local"`, `result_ids`, `snapshot_date`, `record_count`, `index_complete`, and a `note` holding the skip reason when the index was not searched.

## Cache

Raw responses are stored under `hybrid/literature/cache/<source>/<sha256>.json`. The key is the sha256 of the source, URL, and sorted query parameters. Each file records the status, the original `fetched_at`, and the raw body. A cache hit makes no network call and does not wait for the rate limiter.

Failures are never cached, so a rate-limited query is retried on a later run.

The cache is committed on purpose. It is the frozen record that lets the seed list be rebuilt offline with the same results.

## The seed reading list

`seed_reading_list.toml` holds 109 papers across 24 lenses. Each entry has the lens, title, year, source, source_id, url, authors, fetched_at, a one-line relevance note, and the abstract as returned by the source.

The file is generated. Do not edit it by hand. To change it:

1. Edit `LENS_QUERIES` or `PICKS` in `build_seed.py`.
2. Run the build from the repository root:

   ```
   .venv-hybrid/bin/python -m hybrid.literature.build_seed
   ```

   Pass `--searches-only` to run only the searches and write the log.

The build runs every lens query through the clients. Each raw hit goes into `cache/seed_build_log.jsonl`, one row per query, rewritten on every build. A pick is accepted only if its `(source, source_id)` appears among the raw hits for one of its lens's logged queries. The build fails otherwise. Metadata is copied from that logged hit. `test_seed.py` checks the same trace from the shipped files.

Lens names are fixed in `seed.LENSES`. The spec's 23 lenses plus `known_llm_alternatives_to_avoid_reinventing` make 24.

The build runs about 78 queries. arXiv costs about 3 seconds per uncached query. Cached queries cost nothing.

## Prior-art classification

`classify_prior_art(card_text, llm_complete, sink)` takes an injected async callable with the signature `(role, system, user, json_schema)` and returning an object with `.parsed`. It does not import `hybrid/llm`.

1. The model proposes 3 to 5 queries, validated against `QUERY_PROPOSAL_SCHEMA`.
2. Each query runs through `search_with_receipt` with all three sources and is recorded through the sink.
3. The model receives the unique hits (title, year, abstract truncated to 600 characters) and returns a label and evidence, validated against `CLASSIFICATION_SCHEMA`.
4. Evidence is grounded. An evidence entry is kept only if its `(source, source_id)` is among the hits. Entries that name anything else go to `rejected_evidence`, and the title and url are taken from the hit, not from the model.

Labels: `novel`, `prior_art_tested`, `prior_art_undertested`, `known_disproven`.

If the model returns no parsed output, or output that fails its schema, the call raises. Nothing is silently defaulted.

## Tests

Run from the repository root:

```
.venv-hybrid/bin/python -m pytest hybrid/tests/literature -q
```

Live tests are skipped unless `BIGBRANE_LIVE=1` is set:

```
BIGBRANE_LIVE=1 .venv-hybrid/bin/python -m pytest hybrid/tests/literature -m live -q
```

Local index tests (`test_oai.py`, `test_harvest.py`, `test_arxiv_local.py`, and `hybrid/tests/research/test_local_index.py`) use no network. They build a five-paper index through the real writer and serve OAI pages from an `httpx.MockTransport`. `fixtures/oai_list_records.xml` is a real arXiv OAI response trimmed to three records.

Fixtures under `hybrid/tests/literature/fixtures/` are real responses. The arXiv and OpenAlex fixtures were captured with curl. The Semantic Scholar search fixture is a cached response from the seed build. Semantic Scholar references, citations and single-paper lookups have no saved real response yet. See `STATUS.md`.

## Environment

- `BIGBRANE_CONTACT_EMAIL`: optional. Sent as `mailto` to OpenAlex only when set. OpenAlex has ignored `mailto` since February 2026.
- `BIGBRANE_OPENALEX_API_KEY`, `BIGBRANE_S2_API_KEY`: optional API keys; see "API keys".
