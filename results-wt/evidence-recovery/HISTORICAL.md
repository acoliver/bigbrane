# Two independent comparisons recovered; Phase7 still fails

**Zero of the 16 original exact-code gaps are closed.** Two cards now have independently specified host references and matching original CARD-side traces. The frozen critic schema does not admit their new operation ids, and this task forbids editing that schema or the acceptance path. The unchanged replay therefore still misses 16 of 19 required S2 codes. Both original aggregate assertions fail.

The completed subset covers four cards: parity union-find, projective version-space inference, the Bayesian size principle and SUGS allocation. It yields two matching standalone comparisons, one conditional comparison that cannot establish a full reduction, and one refuted allocation mapping. All sixteen missed inputs were inspected. Twelve remain without newly executed traces; no missing trace was inferred from an old scalar verdict.

## Scope and provenance

Worktree: `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, branch `r2-bar`, starting HEAD `fe264b5996b7177d2fbcca5e9274ca5414210578`. Source run: `/Volumes/XS1000/acoliver/projects/bigbrane/runs/run-20261010T181300Z`.

`ledger-index.json` identifies the original executor artifact for each of the sixteen cards, its ledger event and immutable artifact/source hashes. Attribution comes from an executor artifact event with the card id, followed by checking the artifact's `purpose=reduction_check`. Purpose is stored inside the executor artifact, not in a top-level ledger field. All sixteen matching artifact hashes were verified against their ledger events.

For the four captured cards, `capture_historical.py` checks byte identity among the executor's embedded `files/main.py`, the original `jobs/<job>/in/main.py`, and a new fixture copy. It executes those same bytes with `sys.settrace` observers. No insertion, replacement, changed seed, or altered draw is made to the original program. The observers collect only actual CARD-side calls and states, never comparator outputs. The original comparator still runs as part of the untouched program, but neither its result nor its `equivalent` declaration decides the independent measurements. Saved and reexecuted stdout are retained as separate diagnostics.

Published equations and their parameter domains are recorded in [OPERATIONS.md](OPERATIONS.md). Six accessible source snapshots, including Mitchell, Steif, Tenenbaum-Griffiths, Wang-Dunson and Cappé, are saved under `sources/` with URL, retrieval time and SHA256 provenance. These definitions and their algebra specify the host code; no saved comparator was copied into the host. The original card equations were also checked against the selected saved CARD-side implementations. This remains sampled executable evidence, not a general conformance proof.

## Completed comparisons

| Card | Original draws executed | Independent observation | Accepted exact code recovered? |
| --- | --- | --- | --- |
| `582f8abf` | 300 trials; 13,580 tokens; dimensions 2, 4, 6, 8; consistent and contradictory streams | Z2 union-find agrees at every token; maximum absolute difference 0 | No, new operation id is absent from the frozen schema |
| `d86dca4b` | 720 episodes; 10,108 consumed tokens and 720 final predictions; primes 5, 7, 11; active/passive and clean/noisy draws | Finite PGL version-space elimination agrees exactly; maximum absolute difference 0 | No, same schema boundary |
| `ffdbe331` | All 19 original episodes, including the toy; 6,405 sliding windows | 680 supported queries agree within 3.56e-15; 84 have empty posterior support; two have mapped score -4.44e-16 and are refused without numeric repair | No, the full mapping remains unsupported |
| `36725a7c` | All 120 original trials; 3,516 recurrence steps, of which 2,418 pass the original input gate | Normalized weighted-Gaussian SUGS makes 1,174 different allocations; first counterexample saved in `measurements.json` | No, the compared published allocation is refuted |

The concept program's historical 736 compared queries include 54 unseen-word queries answered by a stipulated 0.5 convention, plus the 680 feasible queries and two roundoff-domain queries above. Another 84 queries had been excluded by the original comparator because its Bayesian posterior was empty. They are now explicitly retained in the negative evidence. The host does not copy the card's point-mass fallback on contradictory data. The claimed exponential shadow-volume law is not proved for the arbitrary learned embedding; first-positive anchoring and ignored pre-anchor negatives are further mapping conditions.

SUGS equation 3.1 uses a normalized posterior predictive density. The host retains its determinant term, the precision weight of the new observation, and an uncapped new-cluster option. The original comparator omits the determinant and imposes a four-cluster cap. Thus the mismatch is against a specified published conjugate model, not a claim that no related approximation could resemble the card. A two-observation scalar test isolates the determinant's effect: with zero targets, unit ridge/noise and concentration 1.1, normalized SUGS chooses allocations `[0,0]`; omitting the determinant would choose `[0,1]`. No state-dependent corrective score, frozen gate, or limit-only repair was introduced.

The two matching operations have source-qualified standalone declarations in `historical_declarations.json`. Sixteen parity episodes and 24 projective episodes are selected across dimension/mode/contradiction strata for compact declarations; every original episode is checked separately by the tests. All input/output/state captures remain in `historical.json`, including the negative cases. Projective empty-space uniform predictions and query zero are explicit protocol abstentions, never described as a Bayesian posterior on impossible data. No arbitrary transition table is accepted as a published operation.

## Per-card exact-code accounting and input classification

`none` means no S2 death code, not no later outcome. Before codes come from the original B1 follow-up's `finish-phase7-rows.json`. Current codes were read from the real S2 replay ledger produced by the unchanged Phase7 tests; no second replay was run to manufacture this table. `phase7-current-rows.json` includes all 21 original replay rows.

| Card | Exact expected | Before | Current | Disposition and root cause |
| --- | --- | --- | --- | --- |
| `card-14429dbc` | `reduces_to_known:other_known` | none | none | **unverified.** Missing independent host and trace. GCD quotient compilation, newest-match exemplar lookup and scheduled commits need a published definition of the complete composition. |
| `card-2c5dd003` | `reduces_to_known:other_known` | none | none | **unverified.** Missing exact published lag-one estimator and trace. Permanent lag-one deposits differ from the full recursive online-EM auxiliary-statistic update; generic naming is insufficient. |
| `card-36725a7c` | `reduces_to_known:other_known` | none | none | **refuted allocation mapping.** CARD trace recovered. Normalized weighted-Gaussian SUGS disagrees on 1174/2418 accepted allocations. The old comparator drops the determinant and caps clusters. |
| `card-582f8abf` | `reduces_to_known:other_known` | none | none | **verified standalone.** Independent Z2 union-find matches all 300 original trials exactly. The new operation is outside the frozen critic-schema enum; no accepted kill is restored. |
| `card-68612a2a` | `reduces_to_known:other_known` | none | none | **unverified.** Missing independent host and trace. OMP is related, but residual/error-triggered allocation, coupled value storage and FIFO eviction need separately specified published components. |
| `card-6de11b01` | `excluded_family:delta_rule` | none | none | **unverified; original mapping outside policy.** Memory-tag equality gates the read and prediction error gates overwrite. Input-dependent gate policy does not cover these memory-dependent decisions. Removing the tag changes outputs by 1.23 in the saved program. |
| `card-6fc7efec` | `not_radical:R2` | none | none | **unverified.** Missing full published host and trace. Tensor contraction covers expression evaluation, but its entries are compiled by the card-specific calibration procedure. The full calibration reduction is not established. |
| `card-7ec5e5c2` | `reduces_to_known:other_known` | `not_radical:R2` | `not_radical:R2` | **unverified.** Missing independent host and trace. Worklist automaton-homomorphism closure is tractable; its parser, conflicting-map rejection and output protocol need explicit reference semantics. |
| `card-8dc4db49` | `reduces_to_known:other_known` | `not_radical:R2` | `not_radical:R2` | **unverified; incomplete old base mapping.** Missing independent host and trace. LFU-capped counts, state-selected eviction and warm-started dustbin Sinkhorn iterations are not covered by a plain outer-product update. |
| `card-a36f1a3a` | `reduces_to_known:other_known` | none | none | **unverified; incomplete old base mapping.** Missing independent host and trace. Repeated nonlinear masked matrix squaring with free alpha/beta is not covered by an outer-product table alone. The PPGN composition needs independent specification. |
| `card-bc3935ac` | `reduces_to_known:other_known` | `not_radical:R2` | `not_radical:R2` | **unverified.** Missing independent host and trace. Stern-Brocot labels and interval containment are a promising next subset; insertion, unknown nodes and output smoothing still need explicit semantics. |
| `card-d1d383d5` | `reduces_to_known:hopfield_or_kernel_memory` | `not_radical:R2` | `not_radical:R2` | **unverified; wrong existing host operation.** Missing cubic-DAM host and trace. Existing softmax memory is not a cubic energy/mean-field recurrence. Evidence accumulation and warm starts also need mapping, with arbitrary W free. |
| `card-d86dca4b` | `reduces_to_known:other_known` | none | none | **verified standalone.** Independent PGL elimination and information-gain policy match all 720 original episodes exactly, with explicit empty-space abstention. The frozen schema rejects this operation id. |
| `card-dc69816b` | `reduces_to_known:other_known` | none | none | **unverified.** Missing independent host and trace. NGE/Find-S is related, but residual matching, intersection consolidation, specificity/time conflict resolution and bounded eviction are not established by naming that family. |
| `card-ddb071bd` | `not_radical:R2` | none | none | **unverified; prior comparator already refuted.** Missing independent host and trace. The saved ART/leader comparison already differs by 0.444618. Phase-normalized complex merging and a protected last-merged slot are not established by the family name. |
| `card-ffdbe331` | `reduces_to_known:other_known` | none | none | **unverified full reduction; conditional arithmetic verified.** CARD trace recovered. Quadrature matches 680 feasible queries. All 84 empty-support queries remain unsupported; two tiny negative geometric scores are refused. Volume and anchoring assumptions remain unproved. |

All sixteen original saved programs exist. The missing evidence consisted of independently specified reference arithmetic and independently captured CARD-side traces. The table distinguishes that evidence deficit from mappings that add unsupported operations or already have counterexamples.

## The tagged-memory expected code cannot be justified by its original mapping

For original `card-6de11b01`, job `job-073666f82626`, `hit = (T[a] == kappa)` gates the residual read, while `norm(tgt-h) > tau` decides whether to overwrite the slot. Both decisions inspect current memory or its prediction. `contracts.py` permits a fixed, learned or input-dependent gate and separately permits a second same-family memory; it does not thereby permit equality-driven read gating or a prediction-error-driven write decision. The original program reports a 1.23 output change when the tag is removed.

Under the current written repair policy, the original proposed mapping cannot establish `excluded_family:delta_rule`. Preserving that exact expected kill through this mapping would require an unsupported repair. This is the known policy impossibility for this historical specimen, not permission to edit the expected code or expand the policy. It is not a theorem excluding every imaginable alternative reduction; none has been established here.

## Verification and failures retained

Final affected-file run, before a packaging-only rename of the standalone declaration fixture:

```sh
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2 /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest hybrid/tests/research/test_historical_references.py hybrid/tests/research/test_reference_replays.py hybrid/tests/research/test_review_gate_evidence.py hybrid/tests/research/test_phase7_replay.py -q --basetemp=results-wt/evidence-recovery/pytest-final-tree
```

Result: **71 passed, two failed; exit 1**, in `final-tree.log` and `final-tree.exit`. The failures are the two unchanged original aggregate assertions at `test_phase7_replay.py:29` and `:36`. A subsequent final-packaging run of the new historical tests and existing provenance/reference tests produced **25 passed; exit 0**, recorded in `packaging.log` and `packaging.exit`. It checks the renamed declaration fixture and final checksums without rerunning the full suite. Tests run in background with local logs and PID files, using the main checkout's `.venv-hybrid` and the worktree's cwd/PYTHONPATH.

The positive controls include all original draws, independent Boolean-assignment enumeration for parity, and projective group cardinalities plus three-anchor identification. Deliberately dropping the parity label in an isolated copy of the original CARD function produces an output difference of 1. Suppressing the original projective card's anchor-count update also produces a difference above 0.5. The originals and their fixture copies remain unchanged. Bayesian hand answers exercise the zero-exponent and exponential cases, and SUGS has a separately specified density-normalization counterexample. Free learned embeddings, gates and concept parameters come from all original draws; no trained component is frozen to gain a match.

An initial exploratory run produced 67 passes, one failing mutant test and three Phase7 setup errors. The errors exposed the frozen operation enum; the proposed replay wiring was removed rather than weakening that schema. The first mutant could create singular projective frames, so it was replaced by a dropped-update mutant that remains executable. Those logs are retained as `targeted.log`. The next affected-file run had 69 passes and the original two failures. The final run above adds source-snapshot and frozen-contract tests. These are implementation/debugging runs, not additional review cycles.

`protected-files.json` records hashes for 68 protected tracked files and confirms no protected diff. The original Phase7 helper, its assertions, the critic schema and all existing expected-code fixtures are unchanged. Root tests, `bigbrane`, literature, root STATUS/DECISIONS, README, EXPERIMENT, math, MANIFEST and scoring predicates were not edited. No full suite, live LLM call, OCR, provider retry, commit, push, merge or integration was performed.

## What remains actionable

The numeric reference deficit is closed for two cards; **the exact historical acceptance deficit is closed for zero cards**. All sixteen exact misses remain. The standalone operations are deliberately omitted from the live reference-policy advertisement because the frozen schema cannot admit their names.

A separately authorized contract task could admit the two verified operation ids and wire their compact declarations through the historical helper, without changing expected codes or family policy. It must still run the original assertions; matching standalone arithmetic does not itself establish the missing source/component contract or the entire historical population. This recovery does not make that unauthorized edit.

The next evidence subset is automaton-homomorphism worklist closure (`7ec5e5c2`), Stern-Brocot/Dewey containment (`bc3935ac`) and the specified cubic-DAM recurrence (`d1d383d5`). Each needs its own published equations and original CARD-side capture. For recursive HMM (`2c5dd003`), first identify a published finite-lag estimator with precisely the card's permanent deposition schedule; Cappé's full auxiliary-statistic recursion is not a substitute. Preserve the SUGS refutation and concept empty-support cases rather than adding score corrections, retries or contradictory-evidence fallbacks. Retrieval remains owned by the separate literature worker and is not re-evaluated here.
