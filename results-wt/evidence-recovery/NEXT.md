# Two more comparisons recovered; twelve exact-code gaps remain

The unchanged Phase7 replay matches **7/19 required exact codes**, up from 5/19. Misses fell from 14 to 12. Cards `7ec5e5c2` and `bc3935ac` now return `reduces_to_known:other_known` through independent host comparisons. The other nineteen replay rows keep their previous codes. Both aggregate Phase7 assertions still fail. This checkpoint does not establish novelty success or integration acceptance.

The evidence worker's final current-tree and isolated research-targeted run produced **126 passed, two failed**, exit 1. These are separate executions of the same 128 tests. The final combined verification and package accounting are recorded in [NEXT-CHECKPOINT.md](NEXT-CHECKPOINT.md). No additional review was performed; the latest work has not been independently reviewed.

## Recovered comparisons

| Card | Untouched-source reexecution | Independent comparison |
|---|---|---|
| `7ec5e5c2` | 200 trials, 30,693 tokens, dimensions 4/6/8/12 | All emitted distributions match exactly through `automatonMorphism7ec5e5c2` |
| `bc3935ac` | 200 streams, 18,234 tokens | All emitted distributions match exactly through `deweyForestbc3935ac`; final matrix labels also checked |
| `d1d383d5` | Six trials, 72 tokens, arbitrary symmetric and Hebbian tensors | Diagnostic recurrence difference at most 7.190914530497139e-12; no published full-operation reference admitted |

For automaton homomorphisms, [Transition Systems, section 7.1.2, pages 6-7](https://cs.nju.edu.cn/lixuandong/automata/fch7.pdf) gives preservation of labeled transitions. The host saturates a Boolean relation under `(x,y) -> (T[a,x],U[a,y])`, rather than copying the CARD's FIFO assignment loop. A row with multiple images certifies conflicting demonstrated closure. The full D/Q parser, resets, answer timing and uncertainty distributions remain fixed stream conventions. Unanchored states abstain; the operation does not claim global extendability. Exhaustive two-state checks cover 256 table pairs and four anchors per pair, alongside malformed-parser, noncommuting-word and mutant checks.

For forests, [Xu et al., DDE, SIGMOD 2009, section 2.2/P1, pages 720-721](https://www.comp.nus.edu.sg/~lingtw/papers/sigmod2009.xuliang.pdf) defines ancestry through Dewey path prefixes. [Stern-Brocot uniqueness](https://cp-algorithms.com/others/stern_brocot_tree_farey_sequences.html) supplies the matrix-word correspondence for `C(j)=U^j L`. The host walks parent links. Tests independently reconstruct matrix labels and compare every final node pair. Fresh roots arise only after accepted leaf insertions; rejection never reparents or creates a ghost root. Unknown-node abstention, reflexivity, smoothing and reserved-token behavior are included. No DAG or arbitrary-position insertion claim is made.

### Forest draw limitation

The untouched forest generator calls `rng.choice(list(placed))`. Its historical process hash seed and input streams were not saved. Reexecution uses **`PYTHONHASHSEED=0`**. These are source-reexecution draws, **not byte-exact recovery of unsaved historical draws**. Program bytes, random seed statements and draw bounds are unchanged. `next_sources.json` binds the observed draws, programs and observer separately; `next-capture-invocation.json` records the invocation.

### Cubic DAM remains unsupported

[Krotov and Hopfield, ICLR 2021, section 3.1, equations 5-10](https://arxiv.org/pdf/2008.06996) supports a polynomial energy and continuous dynamics. That citation does not establish the complete finite-temperature synchronous, K-step, warm-started token-stream operation. The diagnostic covers arbitrary symmetric W, cumulative evidence, previous tokens, warm starts, all relaxation steps and the bilinear softmax output. Its numerical agreement is diagnostic arithmetic only. No DAM identifier, enum entry or declaration was added. Negative controls retain warm-start/accumulation differences, binary diagonal-term distinctions and a synchronous step that increases the stated continuous-state energy.

## Binding and reproduction

Only the two recovered operations were added to the registry and schema. Complete original card and critic JSON hashes select each envelope; original expected codes and old numerical verdicts do not select it. `next_specimens.json` retains original artifact/ledger receipts. The embedded original code, job input and packaged programs are byte-identical. Changed specimens cannot borrow the evidence.

The compact fixtures contain eight homomorphism episodes, four forest episodes and two diagnostic DAM episodes. Tests also execute every original draw. `reference_replays.next_historical_capture()` uses the retained local full capture when present, or reconstructs all draws in memory from packaged programs and the hash-bound `next_capture.py` observer. The forest subprocess sets `PYTHONHASHSEED=0` and verifies the draw hash. Original-run and publication-file checks are optional when those local files are absent.

Raw captures, publication snapshots/text, caches and logs remain local and are excluded from the commit. `next_publications.json` retains four snapshot hashes, retrieval timestamps and section references. The fixture directory's `SHA256SUMS` lists only tracked fixture files; the checkpoint inventory hashes the selected delivery files. The isolated package check includes both workers' changes and excludes raw captures and snapshots. All owned historical/transport targets reproduce there. A broader isolated check also finds an existing dependency in the unchanged seed-list test: `hybrid/literature/cache/seed_build_log.jsonl` is absent, so `read_build_log()` returns no historical source hits and `test_seed.py:34` fails. The current-tree log exists. That raw cache/log is deliberately not packaged; the complete literature suite is therefore not self-contained in a cache-free archive. This limitation is reported without changing or suppressing the assertion. No assertion, expected code, tolerance, family policy or acceptance gate was weakened.

## Unchanged failures and remaining cards

The worker research-targeted runs pass 37 new historical-reference cases, 18 binding cases, 19 prior historical-reference cases, six replay cases, 44 gate-evidence cases and two Phase7 cases. The two failed Phase7 assertions remain ordinary failures: the required exact-code aggregate and the tagged-memory family assertion. Prior parity/projective comparisons, SUGS's 1,174 allocation disagreements out of 2,418 observations, and concept's 680 feasible/84 empty-support/two roundoff-domain cases remain unchanged.

The twelve remaining required-code gaps are `card-14429dbc`, `card-2c5dd003`, `card-36725a7c`, `card-68612a2a`, `card-6de11b01`, `card-6fc7efec`, `card-8dc4db49`, `card-a36f1a3a`, `card-d1d383d5`, `card-dc69816b`, `card-ddb071bd` and `card-ffdbe331`. The full worker accounting is in `next-summary.json` and `next-tests.json`.

Original PA2 retrieval remains **FAIL 2/4 against 3/4**; named guards remain **9/10 and 3/6**. [Transport provenance](../transport-provenance/REPORT.md) records future attempt measurements without claiming improved retrieval or explaining the initial historical 429. The transport differential remains 12/12 unchanged. No merge, integration, live API/model call or new review is part of this checkpoint.
