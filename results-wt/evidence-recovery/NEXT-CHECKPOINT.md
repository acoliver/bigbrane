# Seven exact codes match; twelve gaps and two Phase7 failures remain

This checkpoint recovers two more source-bound comparisons and records literature transport attempts. Required exact codes improve from **5 to 7 of 19**, and misses fall from **14 to 12**. This does not establish novelty success or integration acceptance. Original PA2 remains **FAIL 2/4 against 3/4**, and named guards remain **9/10 and 3/6**.

## Verification

| Execution | Passed | Failed | Skipped |
|---|---:|---:|---:|
| New current-tree combined research + literature + evidence tools | 1779 | 2 | 3 |
| New isolated selected archive, six research modules + literature + evidence tools | 360 | 3 | 1 |
| Prior worker research-targeted run, current tree and isolated separately | 126 | 2 | 0 |
| Prior worker complete literature run | 212 | 0 | 1 |
| Prior worker literature-targeted run | 79 | 0 | 0 |

The new combined totals are taken directly from its completed JUnit record, not inferred from targeted totals. Each table row describes its own execution. The transport differential remains **12/12 unchanged**. The combined suite ran once, with main's existing `.venv-hybrid/bin/python`, sibling cwd/PYTHONPATH, `BIGBRANE_LIVE=0`, an inherited network-prohibiting audit hook and sibling-only temporary paths. All assertions and exact expected codes remain unchanged. No live API/model call was made.

Both new executions retain the ordinary Phase7 failures at `test_phase7_replay.py:29` and `:36`: required exact-code matching and the tagged-memory family assertion. The isolated archive also fails the unchanged seed-list provenance test at `hybrid/tests/literature/test_seed.py:34`. Its loader reads the ignored `hybrid/literature/cache/seed_build_log.jsonl`; that historical log exists locally but is absent from the cache-free archive. The loader returns an empty list, so no logged source hit can match the seed. The seed loader, list and test are byte-identical to the base. No raw cache/log was added, test suppressed, assertion changed or suite retried. The complete literature suite is not self-contained without that baseline historical log. All owned historical/reference and new transport targets pass in the archive, apart from the two expected Phase7 assertions.

## Tested tree and delivery contents

Base is `r2-bar` at `eeaba07b93a8907cf6b1e4a6bef62b458b01ae5d`; main stays clean at `701f7d2e2a642714779cfce3d7321f6f9d0b12be`. The checkpoint commit containing this report records the selected overlay. Source/fixture/helper tree SHA256 is `a62b6e1929f61877024c51962184daeec70494257118ca979d257a94a87c220c`; its exact path/hash records and both test executions are in `NEXT-CHECKPOINT-TESTS.json`. Only report and result metadata changed after execution. The isolated archive's staged-tree hash and command are recorded separately.

The package contains the two independent host references, two schema enum additions, replay helpers, new tests, compact trace/diagnostic fixtures, three byte-identical original programs, specimen/ledger/publication metadata and the hash-bound observer needed for in-memory reproduction. It also contains literature provenance/transport/client changes, the new test file, compact reports, safe synthetic examples, the twelve-case behavior comparison and its reproduction script. `NEXT-CHECKPOINT-INVENTORY.json` lists selected paths, sizes and SHA256 values, excluding itself to avoid a self-hash. All 36 fixture-checksum targets are tracked and match exactly. Existing acceptance/family files, tests, reference arithmetic and prior fixture hashes were checked unchanged; 49 protected file hashes match.

Raw captures, publication bodies/text, HTTP/network caches, generated logs, XML and temporary trees stay local and are not committed. Relevant original main artifacts and both retained full captures keep their byte hashes. No implementation was changed during verification/packaging. Safe example endpoints contain origins only; no real credential material was found. The sole credential-bearing URL in the selected tests is an explicit synthetic redaction fixture.

## Source caveat and unresolved work

Forest draws are **untouched-source reexecution with `PYTHONHASHSEED=0`**, not byte-exact recovery of unsaved historical draws. The original forest generator depends on Python set iteration; its historical process hash seed was not retained. Programs and draw hashes remain separate. Cubic DAM arithmetic remains **diagnostic only**: the energy citation does not establish the complete finite synchronous stream operation, and no DAM reference identifier or declaration was admitted.

The twelve remaining cards are `card-14429dbc`, `card-2c5dd003`, `card-36725a7c`, `card-68612a2a`, `card-6de11b01`, `card-6fc7efec`, `card-8dc4db49`, `card-a36f1a3a`, `card-d1d383d5`, `card-dc69816b`, `card-ddb071bd`, `card-ffdbe331`. [NEXT.md](NEXT.md) gives the definitions and limits. [Transport REPORT.md](../transport-provenance/REPORT.md) gives measurement boundaries and reproduction; the initial historical 429 cause remains unknown and retrieval was not rescored.

The latest work has not been independently reviewed. No further review was run after the two completed review cycles. No integration, merge, main push or main documentation edit occurred. Delivery is only to the existing SSH `origin` branch `r2-bar`, using verified `acoliver` identity without global authentication changes.

Local raw verification is retained under `results-wt/evidence-recovery/next-checkpoint-verification/`: `combined.log`, `combined-junit.xml`, `combined.exit`, `combined.pid`, `status`, minute-spaced `heartbeat.log`, isolated results and original-report backups. Worker raw logs and full captures remain in their original locations. Both verification jobs completed; there is no running test handoff.
