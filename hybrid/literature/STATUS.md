# hybrid/literature status

Date of the last build: 2026-10-09 (UTC). Build and tests ran in `.venv-hybrid` (Python 3.12.12, httpx 0.28.1, pytest 9.1.1) on the M4 Max workstation.

## Result

The module is built and its offline suite passes. The seed list holds 109 papers across 24 lenses. Each entry comes from a logged keyless search, and the build verifies that. Semantic Scholar answered 3 of the 78 logged queries, all in the reservoir computing lens. Its hits are in the log and the cache, but the seed list picks none of them. Semantic Scholar's citation and reference endpoints have no saved real response yet, so the `related()` S2 path has not run against live data.

## Local arXiv index (added 2026-10-11)

Why: in the previous runs Semantic Scholar failed 102 of 108 searches with rate-limit cooldowns and OpenAlex failed all 99 in run 2. arXiv's search API was the only source that answered, so S2 labels rested on thin evidence.

What was built:

- `python -m hybrid.literature harvest` and `info` (`__main__.py`, `harvest.py`, `oai.py`). arXiv OAI-PMH at `https://oaipmh.arxiv.org/oai`, `metadataPrefix=arXiv`, resumption tokens, 5 second pause between requests, `Retry-After` honored, SIGINT and SIGTERM stop cleanly, token stored with each page so a restart resumes.
- `arxiv_index.py`: SQLite with an FTS5 index (`porter unicode61`) over title, abstract, authors and categories, plus per-set harvest state.
- `sources/arxiv_local.py`: the `arxiv_local` search source (BM25, title weighted 3, abstract 1; all-terms results first, any-term results fill the rest).
- `Literature(local_index_path=...)`, `Literature.default_sources`, and `classify_prior_art` use the local index first. S2 reads `literature.local_index_path` (default `data/arxiv-index/arxiv.sqlite`) from `run.toml` and `run2.toml`. A missing file is skipped and noted in the receipt.
- The `literature_receipt` schema accepts `source: "arxiv_local"` and the optional fields `snapshot_date`, `record_count`, `index_complete`, `note`.

Design choices worth knowing:

- OAI-PMH has a set per category, so math, cond-mat and physics are harvested as `math:math:{ST,PR,OC,DS,IT,CT,AT}`, `physics:cond-mat:{dis-nn,stat-mech}` and `physics:physics:{bio-ph,data-an}`. The first plan streamed the wide sets and filtered afterwards; it was replaced after the first pages showed the server returns about 1300 records per page at roughly 20 seconds per page for `cs`, which would have spent hours on about 1.4 million records that were then discarded. `SetPlan(categories=...)` remains for filtering a wide set.
- Local papers have `source="arxiv"` and the arXiv id, not `source="arxiv_local"`. That keeps `arxiv:<id>` candidate keys, evidence grounding and cross-source dedupe unchanged. `arxiv_local` is the name of the search source in receipts.
- Counts that arXiv publishes (arXiv API `totalResults`, fetched 2026-10-11, in `results/literature-20261011/expected-counts.json`): cs 957,374; stat 134,614; q-bio 55,696; nlin 48,005; the seven math categories 220,523; cond-mat dis-nn and stat-mech 102,038; physics bio-ph and data-an 29,954. These sum to 1,548,204 before removing papers that appear in several sets. OAI-PMH itself reports no list size.

Harvest state: see `data/arxiv-index/manifest.json` (or `python -m hybrid.literature info`). To resume after any stop, run the harvest command again with the same `--data-dir`. The harvest log is `results/literature-20261011/harvest.log`.

Known limits:

- arXiv says resumption tokens expire daily. If the server rejects a stored token the set restarts from its first page. That has not happened yet in a real run, so the restart path is covered by the fake-server test only.
- The index is a partial snapshot until `complete` is true in the manifest. Receipts carry `index_complete` so a reader can tell.
- No ranking beyond BM25: no citation counts, no recency weighting.
- `--from` applies only to sets that have not started. Refreshing a finished index means harvesting into a new data directory or clearing the set's state.

## Semantic Scholar key and cross-process pacing (added 2026-10-15)

The Semantic Scholar key is configured at `~/.bigbrane/keys/semantic_scholar` (mode 600; `python -m hybrid.literature keys` finds it). Its limit is 1 request per second, cumulative across all endpoints, and Semantic Scholar asks clients to stay below it.

- `MIN_INTERVAL_S["s2"]` is now 1.1 seconds. arXiv stays at 3.0 seconds, which is a per-client policy across processes. OpenAlex stays at 0.1 seconds.
- Research runs start several processes at once (`calibrate --parallel 3`, `planted --parallel 2`), and each used to pace only itself. The production pacer (`PROCESS_PACER` in `client.py`) now keeps a lock file and a last-request time file per source under `var/slots/` (`literature-<source>.lock` and `literature-<source>.last`), the same directory the job and LLM lanes use. A process takes an exclusive `fcntl.flock` on the source's lock file, sleeps until the wall-clock time in the file plus the interval, records its own start time, and releases the lock. Waiting happens while holding the lock, so processes queue up. The OS drops the lock if a process dies.
- Every attempt goes through the pacer, including retries and the request after a 429 backoff, so a retry in one process cannot land inside another process's interval.
- `Pacer(clock, sleep)` still keeps its times in memory, which is what the fake-clock tests use. `Pacer(time.time, slots_dir=...)` is the file-backed form, used only by `PROCESS_PACER`. `wait_for_slot` now returns the start time it recorded.
- A recorded time in the future (the wall clock was stepped back) is clamped to now, so a stale file never makes a process wait longer than one interval.

Live check (2026-10-15, production client, key loaded from the file): one paper search ("liquid time-constant networks", limit 3) and one paper lookup of the first hit. Both returned HTTP 200, with 3 results and the matching paper. Two requests in total. The pacer's slot log put the two requests 1.105 seconds apart. Both cache envelopes hold only `source`, `url`, `params`, `status`, `fetched_at` and `body`, and neither the key nor an `x-api-key` header appears in them, in the URLs or in the log.

Tests: `.venv-hybrid/bin/python -m pytest hybrid/tests/literature -q -p no:cacheprovider` gives 160 passed, 1 skipped. New `test_pacing.py` covers two spawned processes never starting S2 slots closer than 1.1 seconds, arXiv slots 3.0 seconds apart across processes, independent sources, a second pacer on the same directory, threads, the future-time clamp, a 429 retry waiting for its slot, and the key being absent from the pacer files, cache envelopes, URLs and log. The S2 expectations in `test_api_keys.py` changed from 1.0 to 1.1 seconds.

## Tests

Command: `.venv-hybrid/bin/python -m pytest hybrid/tests/literature -p no:cacheprovider -v`

```
106 passed, 1 skipped in 0.34s
```

With the research suite (`hybrid/tests/literature hybrid/tests/research`, 2026-10-11): 341 passed, 1 failed, 1 skipped in 59.46s. The failure is `test_orchestrator_units.py::test_run_config_overrides_and_validation`, which asserts `bench_quick.toml` baselines are `["ngram", "gru"]` while the file now lists four. It failed the same way before the local index work.

The skipped test is `test_live_arxiv_and_openalex_search_return_titled_papers`. It is skipped unless `BIGBRANE_LIVE=1` is set.

Live run: `BIGBRANE_LIVE=1 .venv-hybrid/bin/python -m pytest hybrid/tests/literature -m live -q`

```
1 passed, 57 deselected in 1.06s
```

The live test ran a real arXiv and OpenAlex search for "Mamba selective state spaces" and found a Mamba-titled paper.

Coverage by file:

| File | Covers |
| --- | --- |
| `test_parsing.py` | arXiv and OpenAlex parsers against saved real responses. S2 search parser against a saved real response. S2 references against a synthetic payload, because no real references response was saved. Request construction and id normalization. |
| `test_dedupe.py` | Merge by arXiv id, merge by normalized title, fill-missing behavior, distinct papers kept. |
| `test_client.py` | Cache hit makes no network call (mocked transport counts calls). arXiv 3-second spacing. 429 with `Retry-After`. Exhausted 429 leads to cooldown, then retry after the window. Cooldown does not affect other sources. 404 is not retried. Transport errors retried then recorded. Unparseable bodies recorded. Search event contents. `mailto` only when configured. Related via OpenAlex, and fallback to S2. |
| `test_prior_art.py` | Fake model callable. Evidence grounded in hits, invented ids rejected and reported. Sink records every query and result. Prompt contents. Schema violations raise. |
| `test_sampling.py` | Pool is seed plus logged hits for the lens only. Deterministic for a given seed. `k` capped at pool size. Unknown lens and negative `k` rejected. |
| `test_oai.py` | OAI page parsing against a real trimmed arXiv response: authors, categories, dates, whitespace. Resumption token and list size, deleted records, malformed records counted not fatal, `<error>` codes, malformed XML. |
| `test_harvest.py` | Token following, pause between requests, category filtering, one row per paper across sets, deletions, `Retry-After` on 503, exponential backoff, give-up after 8 attempts, no retry on 400, stop between pages and during a retry wait, resume from the stored token, finished sets skipped, expired token restart, `noRecordsMatch`, page limit, manifest contents, User-Agent. |
| `test_arxiv_local.py` | Query translation (phrases, stopwords, hyphens, FTS5 operators neutralized), BM25 order, stemming, no match on authors or categories, limit, Paper identity, snapshot facts, threads, receipt fields, missing and unreadable index, schema version, local hits with every remote source rate limited, merge by arXiv id. |
| `hybrid/tests/research/test_local_index.py` | Run config default and validation, `arxiv_local` receipt fields and skip note against the schema, `prior_art()` end to end on the configured index with all remote sources returning 429. |
| `test_seed.py` | 80 to 120 entries. Per-lens counts in range. Known-alternatives lens names the architectures. Every entry traces to a logged search hit for its lens, with matching title and url. Metadata present. Relevance notes are one line with no em dash. No duplicates. |

## Seed list: papers per lens

Each entry is a pick from hits that a logged keyless search returned. The build checks this, and `test_seed.py` checks it again from the shipped files. `cache/seed_build_log.jsonl` has one row per query with the hits per source and every error.

| Lens | Papers |
| --- | --- |
| reservoir_computing | 5 |
| vector_symbolic_hyperdimensional | 5 |
| holographic_reduced_representations | 4 |
| modern_hopfield_networks | 5 |
| predictive_coding | 5 |
| active_inference | 4 |
| dendritic_spiking_computation | 4 |
| neural_odes_koopman | 5 |
| tensor_networks_language | 5 |
| renormalization_multiscale | 5 |
| kernel_gaussian_processes | 4 |
| bayesian_program_induction | 4 |
| cellular_automata | 4 |
| optimal_transport | 4 |
| sheaves_category_models | 4 |
| compression_mdl | 4 |
| algorithmic_information | 4 |
| sparse_distributed_memory | 5 |
| complementary_learning_systems | 4 |
| hyperbolic_geometry | 4 |
| quantum_inspired | 4 |
| chemical_reaction_networks | 4 |
| morphogenesis | 4 |
| known_llm_alternatives_to_avoid_reinventing | 9 |
| **Total** | **109** |

Sources of the 109 entries: 87 OpenAlex, 22 arXiv, 0 Semantic Scholar. Entries from 2024 or later: 35, across 20 lenses. Years range from 1988 to 2026.

The known-alternatives lens covers linear attention (Transformers are RNNs), Mamba, RWKV, RetNet, Hyena, xLSTM, BitNet, Large Concept Models, and test-time-training layers (Learning to (Learn at Test Time)).

Search log: 78 query rows (23 lenses at 3 queries each, the known-alternatives lens at 9). Errors in the final log:

- arXiv: 0 errors. Two queries returned zero results without error: "fast and slow learning memory" and "linear attention transformers are RNNs".
- Semantic Scholar: 75 errors. 74 were cooldown skips and 1 was an HTTP 429 after retries. The other 3 queries returned 200 and their responses are cached.
- OpenAlex: 0 errors.

## Build and reproduction

- Build command (repository root): `.venv-hybrid/bin/python -m hybrid.literature.build_seed`.
- A warm cache makes no arXiv or OpenAlex request. Semantic Scholar is contacted only for queries not yet cached, and at most once per cooldown window. The log and TOML are rebuilt from the cache.
- The cache is committed on purpose. It is the frozen record for reproducing the seed list offline.
- The build stdout is kept in `cache/seed_build_stdout.txt`.

## Incidents

1. **Breaker that was too strong (fixed).** The first version disabled a source for the rest of the process after one 429 exhaustion. A single arXiv 429 burst disabled arXiv for 25 of the 78 queries, and the log hid this because the summary filtered out `skipped` errors. Replaced with a 600-second cooldown. The cooldown is tested. The 25 queries were rerun and cached.
2. **arXiv rate limiting from this machine.** During the rerun, arXiv answered 429 to every request for several minutes, including fresh probes with curl. Several sibling sessions on this machine share the outbound IP. It recovered on its own, and the rerun completed with arXiv hits for 76 of 78 queries.
3. **Semantic Scholar mostly rate limited.** S2 answered 200 for "echo state network", "reservoir computing" and "liquid state machine", at different times during the build. Every other S2 request got 429, and most were skipped by the cooldown. The three responses are cached and are the source of the saved S2 fixture.

## Deviations from the spec

- `Paper` has two extra optional fields, `arxiv_id` and `doi`, used for dedupe and for resolving citation lookups. The spec's field list is otherwise intact.
- `search` is synchronous to match the spec's signature. `classify_prior_art` is async and runs searches in a worker thread (`asyncio.to_thread`) so the event loop is not blocked.
- The seed TOML also carries the abstract for each entry, so explorers can read an entry without a network call.
- The S2 rate-limit policy is a cooldown, not a permanent breaker (see Incident 1).
- Evidence the model cites that is not among the search hits is dropped from `evidence`, reported in `rejected_evidence`, and recorded in the `prior_art_result` event. It is not an exception. The label stands. The caller can decide whether an empty evidence list invalidates it. The spec did not say which.
- Relevance notes describe each paper's bearing on the target learner. They are not summaries.

## Open issues

1. **Semantic Scholar references, citations and single-paper lookups have no saved real response.** The search response is real (`fixtures/s2_search.json`). The references parser is tested against a synthetic payload that follows the documented field names, and the test says so. The citations and `get_paper` S2 paths have no test with real data. S2 contributed no papers to the seed list.
2. **Real model not exercised.** `classify_prior_art` was tested with a fake callable only. The prompts and schemas have not been run through `hybrid/llm`, which is not built yet.
3. **Citation graph is thin in OpenAlex for some works.** For example, the Mamba record in OpenAlex has an empty `referenced_works`. `related()` returns whatever OpenAlex has and falls back to S2 only when OpenAlex fails outright. An empty but successful OpenAlex result does not fall back.
4. **Some OpenAlex records are noisy.** Several 2026 OpenAlex records with ids in the W7 range have zero citations and duplicate arXiv entries. The pick list avoids most of them, but the search pool includes them, and `sample_for_lens` may return them.
5. **Recency is uneven.** 35 of 109 entries are from 2024 or later. Four lenses have no 2024–2026 entry: reservoir computing, tensor networks for language, complementary learning systems, and morphogenesis. Tensor networks has no entry after 2021.
6. **Abstracts carry source typography.** Some abstracts in the TOML contain em dashes and the word "canonical" because they are copied verbatim from the source. The relevance notes were written here and contain neither.
7. **Picks are a judgment.** The 109 picks were chosen by reading the logged hits. The build proves each pick came from a logged search. It does not prove the pick is the best available paper for its lens.
