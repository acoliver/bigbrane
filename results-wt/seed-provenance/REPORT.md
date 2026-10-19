# All 109 seed entries have durable original-source provenance

The cache-free source export passes the unchanged seed assertions. All 109 shipped entries match complete paper metadata from original logged search hits and re-parsed original HTTP responses: 87 OpenAlex entries and 22 arXiv entries. No unsupported entry or unknown metadata was found. PA2 remains **FAIL 2/4**. The two Phase7 failures and remaining research evidence work are outside this task and were not rerun.

## Scope and root cause

Work was restricted to `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`. Its actual branch is `r2-bar`, at `e0e2c5a5a5b8d2541428ae978c226a7f8f8f9b77`. Main remains clean at `82580a6630c4b5c9155f00d603f96682d5818e7c`. No branch switch, commit, push, merge, protected document edit, acceptance edit, source-schema change, transport change, or other worker's implementation edit occurred.

Previously, `read_build_log()` read ignored `hybrid/literature/cache/seed_build_log.jsonl` and returned an empty list if absent. The local cache concealed missing delivered history. It now reads `hybrid/literature/seed_provenance.json` explicitly and validates its integrity, source request identities, complete paper metadata and coverage against the shipped TOML. Missing or altered data raises rather than falling back to a cache or synthesized seed history.

The existing explicit `read_build_log(path)` JSONL contract is retained because sampling supports caller-supplied logs. `cached_search_papers` streams raw history and filters the requested lens. This does not ship the broader sampling cache. Explicit missing raw paths fail fast.

## Original evidence and projection

The original log contains 78 search rows and 2,346,413 bytes. Its SHA256 is `5a256f4a78b16415e85dd6ed553529f1d5361d4804d8f2bba206215077036e26`. Both original-worktree and sibling log copies are included in the preservation manifest. They remain unchanged.

Recovery validated 71 relevant original cache envelopes, 51 OpenAlex and 20 arXiv. The arXiv envelopes record `http://export.arxiv.org/api/query`; the current source request builder uses HTTPS. Recovery supplies those exact historical cache paths via `receipt-paths.json`. Production compilation does not try alternate URLs or choose a plausible cache entry. Each envelope's request parameters, source, successful status, request hash and filename are checked before the existing source parser reconstructs the hit. Every chosen logged paper must equal that reconstruction, including title, URL, authors, abstract and fetched timestamp. The seed list is used only to select identities and check equality, never to create historical hits.

The projection retains the last logged occurrence for each selected identity, preserving original lens, query, timestamp, requested sources, limit, errors, per-source paper metadata, original line number and raw-line SHA256. It also carries original full-log SHA256/size/row count and each retained response's request identity, fetched timestamp, cache-envelope SHA256/size and response-body SHA256. It has 59 rows, 68 retained source receipts and exactly 109 hits. The artifact is 204,726 bytes, 8.7% of the original log. Its payload SHA256 is `80b87a4a77132383e00c1e648c38371346aedecb861f39f6447822341bba30fa`.

`compile_seed_provenance` hashes and scans the original log once. It keeps one current row/HTTP response and at most one selected occurrence per seed identity, then groups those selected occurrences. It does not retain full history or repeatedly reread the log. Unrelated rows do not trigger cache reads. The builder now streams raw search rows to its existing JSONL writer and retains only configured picks. It validates the original inputs before writing a pending seed/provenance pair, verifies the pair, then replaces both outputs. The two replacements are consecutive, not a multi-file filesystem transaction; an interrupted replacement produces a visible metadata mismatch rather than a fallback.

The payload checksum detects corruption of the shipped artifact. These are unkeyed integrity hashes, not publisher signatures. A cache-free reader verifies the delivered projection, not unavailable original response bodies. Raw re-authentication is the compiler's job and was completed here; `recover.py` reproduced the delivered artifact byte-for-byte from those retained originals. Replacing evidence and recomputing every digest is outside this integrity check's guarantees.

## Verification

| Execution | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Initial unchanged seed plus new tests, before two additional negative cases | 34 | 0 | 0 |
| Final cache-free isolated export: unchanged seed plus 29 new tests | 36 | 0 | 0 |
| Complete literature suite, once | 241 | 0 | 1 |

The final tests exercise changed source ID/title/URL/lens/source/fetched timestamp, altered query, raw-log hash, raw-row hash and cache hash. Recomputed artifact checksums cannot hide metadata changes from the seed equality validation. Compiler cases reject manufactured logged metadata, wrong source requests, missing receipts and unsupported seed metadata. Builder tests confirm a validated pair is emitted and old outputs stay unchanged if input validation fails. A 2,001-row fixture verifies a single log open and projection of only the relevant final row. This is a behavioral and source-code bound, not a measured peak-memory benchmark.

Both final executions use main's existing `.venv-hybrid/bin/python`, sibling or isolated cwd/PYTHONPATH, and `BIGBRANE_LIVE=0`. An inherited `sitecustomize.py` audit hook rejects actual network connection, DNS and datagram events. No live HTTP or LLM call was made. `import-proof.log` confirms imports resolve inside the isolated export and the network-denial hook is active. The export contains tracked source files plus the two new deliverable files, with no `hybrid/literature/cache` before or after execution.

The seven existing seed tests are byte-identical to HEAD. Their SHA256 remains `435a6e98d633e9dd9a9cb26b17afcaaf552a14b43a3b26c523c75df8411e4b50`. The seed TOML SHA256 remains `a72d06d7d1e49531c59cfd89b3ca9a18d3477430fe5a23ad85b6a131949cce3f`.

Commands, environments and cwd values are recorded in `verification.json`. The driver is `verify.py`; XML and stdout are retained separately for each run. From the sibling, these scripts reproduce receipt authentication and the verification procedure:

```sh
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/seed-provenance/offline:/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2 BIGBRANE_LIVE=0 /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python results-wt/seed-provenance/recover.py
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python results-wt/seed-provenance/verify.py
```

`verify.py` creates a fresh export directory, so use a new output directory before another execution rather than deleting retained proof. Its isolated command runs only `test_seed.py` and `test_seed_provenance.py`; its literature command runs `hybrid/tests/literature` once. `git diff --check` passed. No full research suite was duplicated. Review skipped per Andrew 2026-10-09.

## Delivery inventory

| File | Bytes | SHA256 |
|---|---:|---|
| `hybrid/literature/seed.py` | 13435 | `1c4cdc0063f23af56109f4bc2eb2d1ac2a11950e0360a90b1e75d744669c35f0` |
| `hybrid/literature/build_seed.py` | 25597 | `53a63934964d3ffc2a651c2b6848a967f64a9b5dd5400c7ab84ea34b6820042f` |
| `hybrid/literature/seed_provenance.json` | 204726 | `25c11dd6cdb8de9c7f5179b8b9998bd8d34ce36a415a9dd406551758585ee3f9` |
| `hybrid/tests/literature/test_seed_provenance.py` | 7729 | `b75a9651ae9dddf026da072907476390d33c1c487d99b1867658807f77d713c1` |

`raw-baseline.json` identifies all 75 checked originals by full local path, bytes and SHA256: 71 relevant HTTP receipts, two raw log copies, the original test file and the seed TOML. Every recorded original was rehashed after verification and matches. Raw logs and caches remain local at their existing paths; none was added to delivery. `receipt-paths.json` records the original search-to-receipt mapping. `EVIDENCE.json` records counts and inventory hashes. Results, temporary exports and verification logs stay under this sibling's `results-wt/seed-provenance/`; no `/tmp` output was used.

## Remaining work

No unsupported seed metadata remains. This resolves shipped-seed provenance only. PA2 remains FAIL 2/4, and neither Phase7 failure is claimed fixed. Other workers' research references and diagnostics remain separate. The repository's distribution configuration includes only `bigbrane*`, so this proves a cache-free source export, not a new promise that the existing wheel installs the separate `hybrid` tree. No wheel configuration was changed.
