# Three named sets are inventoried; broader used-data coverage remains pending

This preparation binds the 11 original PA2 registrations, 19 historical recovery mechanisms and 109 seen seed identities to local original evidence. The versioned registry also contains 340 other used card identities, including both historical passes, and structured paper metadata from the bounded discovery scan. It does not establish that every previously used study has been found. The unchanged frozen harness returns PENDING and reports no measured query effect or acceptance improvement.

## Counts and interpretation

| Scope | Logical entries | Paper metadata records | Known normalized aliases | Unresolved links | Status |
|---|---:|---:|---:|---:|---|
| `historical_recovery_19` | 19 | 172 | 335 | 91 | complete |
| `original_pa2_11` | 11 | 11 | 19 | 0 | complete |
| `other_used_study_data` | 340 | 15543 | 22443 | 2636 | pending |
| `seen_seed_109` | 109 | 109 | 317 | 0 | complete |

These are different denominators. Paper metadata records retain their source-qualified combinations of title, identifiers and evidence. The global paper union has 12,932 conditional identities. The combined mechanism-and-paper registry has 13,291 unions and 22,662 distinct aliases. Summing scope aliases would give 23,114 and double-count cross-scope overlap. No union count is a count of independently novel mechanisms. 388 unions have title identifiers only.

Each PA2 logical registration retains its own registration ID even if a later alias joins papers. Each seed entry retains the actual source paper identity, lens, original source, raw-row hash and receipt hashes. The shipped provenance projection was checked against its existing evidence digest and the original TOML metadata, including authors, timestamps, URL and abstracts, without copying those abstracts into this delivery. The two original providers account for 87 OpenAlex and 22 arXiv seed entries. Hashes here are unkeyed integrity checks, not publisher signatures.

Historical mechanisms retain the original card and critic digests, source paths, normalized source-derived semantic labels, closest recorded prior-art paper links, and discovered card versions. Every historical CID has an exact bound card-payload match among those versions. Different card versions are not counted as separate historical mechanisms. Passes `card-46dbb349` and `card-c9615069` remain in the other-used scope. A reduction mapping naming a verified publication title can add that source identity; it does not establish a scientifically verified reduction.

## Evidence and scan boundary

The builder selected 4,540 paths, streamed 179 ledgers once per build, and bound 4,948 source files. The 212 discovery roots include 191 enumerated roots, 20 excluded-pending generated/verification roots and the excluded held-out stage. The explicit metadata selection includes old runs, sibling `r2-bar` and `kd8`, Phase7, research fixtures and bound tests, plus milestone source cards. `root-manifest.v1.json` records every root, selection and exclusion reason. `TOP-LEVEL-BOUNDARIES.v1.json` also names unexamined top-level directories. An enumerated root means the stated file selection ran, not that every file in it was examined.

`source-hashes.v1.json` maps 316 duplicate byte-hash groups involving 754 source paths. `ledger-discovery.v1.json` retains card IDs, artifact paths, hashes and counts rather than full ledger events. Existing file/source inventories are bound in `prior-source-inventories.v1.json`; publication metadata is in `publication-sources.v1.json`. Twenty-two exact title aliases are additionally bound to retained publication-header line spans and body hashes in `source-title-bindings.v1.json`. No author/year description was substituted for an unknown title or public identifier. Unqualified hex hashes are not assumed to be Semantic Scholar IDs.

The preparation files contain small metadata projections, not raw model calls, response bodies, full ledgers, publication texts, credentials or blinded labels. The larger other-used projection reflects thousands of recorded candidate identities and their path/hash attribution. Its source papers were used or exposed in earlier study evidence; inclusion does not assert semantic relevance to a future card.

## Remaining gaps

The historical scope has 91 unresolved `known_relatives` descriptors. Named-scope completeness means the required source entries and recorded known identity aliases are present, not that their bibliographies or every cross-provider link are complete. Other-used has 2,636 distinct scope-qualified unresolved links, including references, candidate evidence and relative descriptions. These are ambiguity records, not 2,636 distinct missing papers. Repeated occurrences retain their source paths and occurrence counts.

Two named provenance gaps remain in other-used: the original prior-art row for `card-46dbb349` has no structured closest-source paper, and the retained Mitchell chapter-2 PDF has no verified exact title span or supported public paper ID. The latter URL is `https://www.cs.cmu.edu/afs/cs/project/theo-20/www/mlbook/ch2.pdf`; its source bytes and path remain bound. The complete list of unresolved IDs and descriptors is in the scope files. The 20 excluded-pending roots include prior pilot/follow-up and verification trees. Their study relevance and deduplication still require attribution. Deleted inputs, archives and unregistered external study roots are unknown. These gaps keep the broader scope PENDING.

Known positive exclusions were tested across all recorded paper aliases. Unicode title normalization, DOI URL/case normalization, arXiv version/URL normalization and provider aliases use the unchanged frozen harness. Its accepted URL spellings are not unlimited: uppercase `.PDF` arXiv suffixes and uppercase OpenAlex URL prefixes are rejected. Tests preserve these visible limits; this preparation does not silently change the normalizer. Future ingestion must respect its accepted syntax or use a separately documented stage change.

## Stage pin and frozen preservation

This is an explicit inventory-stage update implementing the already-frozen exclusion task. It changes no policy, budget, corpus design or preregistration pin. The original registry remains byte-identical at SHA256 `45f3753e8181ef784c77df11529fe9eea405a6c1f1f90b99c411bb2c69d8e433`. Nine frozen base files are byte-identical, including PLAN, policy, preregistration, original exclusions and their pin, corpus, harness, offline hook and original source-hash manifest.

The new registry SHA256 is `bd15b2e35324ac71c3d88a40b5b740396268d8b949003df37ab592e0ba9aae9c`. The stage SHA256 is `c76e12528108f07913f6928fc59e470ac28a20df9bc9c5021fa6e24bbe88565f`. `STAGE.v1.json` binds the frozen base, all four inventories, alias/source/root manifests, title bindings, builder and targeted test files. `exclusions.v1.sha256` and `STAGE.v1.sha256` are separate pins. Consumers must opt into these files explicitly, validate both pins and retain the PENDING scope status. Nothing repoints the original preregistration.

The 274 original replay inputs and 75 original seed inputs, 349 distinct paths, matched their existing manifests before and after extraction. `preservation-before.json` and `preservation-after.json` record the checks. Main remains clean at `3ce1516`; sibling HEAD remains `8cad9a1`. No production edits, commits, pushes, integration or review cycle occurred.

## Validation and reproduction

All 32 preparation tests pass with zero failures, errors or skips. Tests cover exact logical counts/CIDs, all recorded paper identity aliases, duplicate paper registrations, source and artifact tampering, missing inventory/source links, incomplete scopes, synthetic corpus intersection, semantic normalization and network denial for DNS, TCP and UDP. Compact replay tests use shipped metadata and synthetic fixtures; they do not reconstruct unavailable original response bodies. Full source verification additionally checked all 4,948 bound source hashes. A second complete local build reproduced identical stage bytes and artifact hashes.

Only the main `.venv-hybrid/bin/python` was used, with sibling cwd/PYTHONPATH, bytecode disabled, isolated pytest configuration, plugin autoload disabled and the frozen socket/DNS/datagram audit hook. Test work and receipts stay in this directory. No full suite, network query, live API call or LLM call was run. Run the following from the sibling checkout with `PYTHONDONTWRITEBYTECODE=1 BIGBRANE_LIVE=0 PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`:

```sh
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -B results-wt/heldout-acquisition/preparation/inventory/build_inventory.py
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -B results-wt/heldout-acquisition/preparation/inventory/run_tests.py
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -B results-wt/heldout-acquisition/preparation/inventory/build_inventory.py --validate --verify-sources
```

Retained results are `tests.log`, `tests.xml`, `source-validation.log`, `reproducibility.json`, `BUILD-RECEIPT.v1.json` and `harness-replay.json`. The last replay passes the new registry and its pin explicitly to the unchanged frozen harness. The harness reports PENDING, as required by the absent corpus and incomplete broader scope.

## Later independent acquisition

No corpus or reference labels were selected here. The future corpus worker must be separate from the policy implementer. Human provenance means independently authored publications or external original mechanism sources with citations; it introduces no owner-approval gate. Existing exclusion titles and public identifiers are allowed metadata under the frozen PLAN. They must not become policy words or blinded acquisition targets.

The acquisition worker receives only opaque mechanism IDs, mechanism concept phrases and the supplied concept lenses. It receives no target titles, authors, public identifiers or reference judgments in query metadata. An independent label worker maintains the separate pinned label vault. Both future mechanism identities and reference-work identities need the complete exclusion check before unblinding. Identifier disjointness is conditional on recorded aliases. Title inequality cannot establish semantic independence of a future unit.
