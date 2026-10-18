> Finish-worker update: The finish worker added independent host references and actual saved-program card traces for the KD8/KD4/FIX12 controls. The exact original 26 nodes now yield 24 passed and two failed. The remaining original Phase7 aggregate assertions are unchanged; see finish-REPORT.md and finish-PHASE7-REMAINING.md. B4 downstream wiring is now implemented. Final affected-file verification is 420 passed, two failed and one skipped (finish-final-tree-targeted.log). Final combined verification is running, not yet a pass.

# Gate bypasses repaired; historical acceptance evidence is still incomplete

The assigned gate fixes for B1, B2, B3 and B7 are implemented, with N1, N2 and N4 addressed in the same paths. The focused tests pass. This is not a successful historical acceptance replay: the broader affected-test run retains 26 failures where old assertions require outcomes unsupported by independent reference evidence. No novelty, PA2 success or later-stage performance result is claimed.

Worktree: `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, branch `r2-bar`, uncommitted changes over `fa08b93750c19f59808ae1ac18af979e4d8ac310`. File hashes and the exact owned paths are in `gates-manifest.json` and `gates-files.txt`. The original review at `results-wt/final-review/REVIEW.md` and all of `DECISIONS.md` were read. No project-local AGENTS.md or LLXPRT.md was found at the root or research instruction locations examined.

No commits, pushes, merges, live gateway calls, full-suite runs or edits under `hybrid/math` were made. Frozen root tests, `bigbrane`, `EXPERIMENT.md`, `README.md`, `expected.json`, `DECISIONS.md` and root `STATUS.md` were not edited. Literature and scoring changes visible in the shared worktree belong to the other worker and are outside this report's verification.

## Changes and evidence

### B1: execute the published comparator independently

`hybrid/research/stages/s2_prior_art.py:279-448` specifies and executes host-owned reference operations. A comparator's `reference_spec` selects an operation and numeric mapped parameters before execution; its report supplies executed card input/output traces. The host computes the reference outputs and replaces the model's reported difference with its own measurement. It retains the original number as `reported_max_abs_diff`. Model-written comparator equality is diagnostic, never independent validation.

The supported references are Elman tanh and identity cells, fixed linear state-space recurrence, input-table-selected state-space recurrence with a binary input-only mask, additive linear attention, the delta rule, Jeffrey conditioning/IPF, modern Hopfield softmax retrieval, the fixed Dyck-2 pushdown recognizer, synchronous periodic Rule 110 and Viterbi. These are specified operations, not source-code scans or names treated as proof. A different published operation, unsupported repair, missing trace or malformed reference remains `unverified_reference` and cannot verify or refute the reduction. Implemented repairs must agree with the host operation's repair set. Family labels alone cannot provide a reference.

The input-selected SSM uses immutable parameter tables indexed by the input token. There is no state-selected top-K operation or eviction/reset repair. The original saved `topk_selective_diag_ssm` declaration and report are retained unchanged in `hybrid/tests/research/fixtures/review_gate_state_topk.json`; the new regression confirms that they cannot count. No prose keyword is used to reinterpret the mask as evidence.

`judge_comparisons` at `s2_prior_art.py:451-515` now measures the independent reference. `counted_comparisons` at `s2_prior_art.py:518-529` recomputes from reference evidence, including for loaded reports; it does not trust saved `eligibility` or `within_tolerance` labels. The equivalence tolerance remains `1e-6`, and no confidence, sample-count or acceptance threshold was lowered.

The square-copy regression actually executes `(h + x) ** 2`, reports its zero self-difference, and is refuted against host tanh. Genuine tanh traces verify under both `Elman RNN` and `Elman recurrent neural network (1990)` names. Independent, hand-specified answer vectors test the other listed controls.

Jeffrey preservation also uses actual saved execution. `review_gate_jeffrey_saved.py` is a byte-for-byte copy of original job `job-481031acb7bf`; its SHA-256 and source path are in the adjacent provenance JSON. The test executes the original code, then calls its original `card_run` for the original 50 seeded parameter/stream draws and compares with the new host Jeffrey operation. It never uses the saved model-written `ipf_run` as the independent reference. The old annotated KD8 replay is still not treated as reference proof.

**Rule and limits:** a small fixed reference registry is the chosen implementation, rather than inspecting Python source or trusting another declaration flag. Numeric sampling is not a general equivalence proof. The card-side implementation, the correctness of its mapping and the claimed freedom of learned parameters remain authored inputs requiring the existing adversarial/conformance checks. This patch does not independently prove those inputs faithful to the card. The registry does not cover every genuine published algorithm, LSTM/GRU variant or permitted repair. Those comparisons stay unverified until an independent reference and executed samples exist; a missing reference is not evidence of novelty or refutation. Historical scientific outcomes must not be described as preserved merely because routing fixtures pass.

### B2: failed execution never verifies

`s2_prior_art.py:577-619` returns unverified for null code, unsuccessful jobs and missing JSON reports, regardless of confidence. `declaration_problem` at `:259-267` blocks mappings that admit fixed learned parts. The source-evidence judgment applies the same fixed-part restriction before counted evidence.

The regressions cover confidence 1.0 with null/crashed/missing-report checks crossed with restatement, unresolved, limit-only and fixed-learned-part declarations. End-to-end confidence tests also confirm that no counted family or excluded-family kill is manufactured. No retries were added. The configured confidence field remains available for recording but cannot substitute for execution.

### B3: identity does not establish mechanism coverage

`hybrid/research/stages/known_work.py:35-92` preserves exact source-qualified paper keys. A bare legacy id resolves only when exactly one candidate source matches it; a different qualified prefix never falls through to suffix matching. `difference.py:168-217,229-279` retains that identity through listing, requirements and citation checks.

Resolved paper identities and unvalidated textbook/algorithm names are leads for the existing difference challenger. Without independently referenced counted evidence, `known_work.judge` returns a non-killing `coverage_unchecked` or `no_evidence` result. `covers` prose is not parsed as proof. An arbitrary algorithm name is explicitly not a mandatory, potentially impossible closest-prior citation, and cannot itself force a difference rejection. The tests include fabricated names, a source-prefix collision, ambiguous bare ids and a real source claiming only initialization coverage.

The separately documented `textbook_transplant` judgment gate remains unchanged. This report does not turn that exception into a source-verified claim.

### B7: scope accepted stored content

`hybrid/research/stages/difference.py:416-456,467-504` stores a scoped `accepted_statement` in the artifact and a scoped `difference_statement` on the final card when the accepted checker result has excerpt gaps. Raw author prose remains in the artifact's original `statement` for provenance.

Because recorded gaps have no field addresses, the host conservatively replaces literature comparison narratives and aggregate comparison/change prose with excerpt-limited statements. It retains source refs, titles, the changed equation and addressed cause ids. `evidence_scope` carries the full, untruncated gap list and writer/check call ids. The model writer cannot supply this host-owned field. No unsupported paper assertion is retained as an accepted comparison merely because a ledger warning exists.

This can remove more comparison detail than the exact unsupported clause. That is the documented cost of scoping recorded unaddressed gaps without another model call or unreliable substring parsing. It does not create an automatic rejection for a missing excerpt detail. End-to-end tests inspect the final stored card and artifact's scoped accepted content, while verifying that the raw statement remains available.

### N1, N2 and N4

- **N1:** two distinct relevant-source/outcome/family entries are required in total, with family entries filling missing relevant-source slots. Source slots count classifier-cited evidence, resolved critic paper leads and own-history records. Unrelated retrieved papers and duplicated aliases do not fill required slots. Schema, prompt, host checks and synthetic writer fixtures reflect R2D; no two-prior rule was removed.
- **N2:** the universal-class regex recognizes the bare recurrent-network class without rejecting a specified Elman cell merely because its name spells out “recurrent neural network.” The paired names use the same host reference and execution traces.
- **N4:** `critic_answers.novelty_fields` no longer invents a Hopcroft citation from an old `radical_enough` answer. The Phase 7 adapter no longer adds tanh/update components or substitutes 0/1 differences when the recording has no number. Pure routing tests use explicitly named synthetic specimens with independently specified constants; they are not claimed as evidence about a recorded card. Actual saved execution has a separate unchanged fixture and provenance. Historical registration assertions and recorded expected kill codes remain unchanged.

## Targeted verification

All commands used the requested interpreter, `/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python`, with cwd and PYTHONPATH set to the worktree. The interpreter initially lacked pytest, jsonschema and httpx; these were installed into that venv, recorded in `test-dependencies.log`. No gateway endpoint was contacted.

| Check | Outcome | Log |
| --- | --- | --- |
| Initial regression file before production changes | 25 failed | `gates-red.log` |
| Final focused regressions, known-work tests and difference tests | 82 passed | `gates-targeted.log` |
| Broader affected gate tests plus orchestrator units | 168 passed, 26 failed | `gates-affected-final.log` |
| Owned-file `git diff --check` | Passed | Shell verification; no output |
| Frozen-area tracked diff check | No changes | Shell verification; no output |

Focused command:

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2 \
  /Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python -m pytest \
  hybrid/tests/research/test_review_gate_evidence.py \
  hybrid/tests/research/test_known_work.py \
  hybrid/tests/research/test_difference.py -q
```

The broader command adds `test_kd8_restatement.py`, `test_fix12_phase7_live.py`, `test_kd4_counted_family.py`, `test_phase7_replay.py` and `test_orchestrator_units.py`. It used the same final production source, before the final test-only persisted-artifact assertions were strengthened and confirmed in the focused run. No full suite was run; combined verification belongs to the coordinating worker.

## Remaining original findings and exact handoff

The gate bypasses named above have code and focused regression coverage, but historical evidence preservation is incomplete. All 26 remaining failures are listed in `gates-failures.txt`:

- Ten KD8 assertions require annotated legacy reports to count without independent references. Keep the original recorded expectations. For any mechanism whose preservation is claimed, implement its published reference independently and execute the original card-side code to capture actual input/output samples. The saved Jeffrey code already has this separate executed control; that does not retroactively validate its old annotation-only fixture.
- Ten FIX-12 assertions require verification/refutation and old kill codes from model-written phase-normalizer, MesaNet hop-mixture or bimap checks without the new reference evidence. Add independently specified reference operations and actual samples for those repairs before claiming those outcomes established. Do not attach a reference label to their old scalar differences.
- Four KD4 saved-report assertions rely on old counted labels. Their original reports need independent reference/sample evidence before current revalidation can count them.
- Two Phase 7 historical acceptance assertions miss their original kill codes. They remain misses. Neither a later textbook-transplant kill nor overall dead/alive agreement substitutes for the registered reduction/exclusion gate.

B4, B5 and B6, plus N3, N5 and the broader N6 reporting work, belong to the other worker and are not certified here. The original review's saved PA2 retrieval result remains at most 2/4 against 3/4, and its registered gate misses remain misses unless new evidence under the unchanged predicates establishes otherwise. This patch did not rescore those experiments or change their tools.

The coordinator should retain these misses, combine the two workers' verification, and arrange the one permitted follow-up against the full original review scope and findings. No OCR review was run. No new review requirement, approval gate, tolerance change or acceptance relaxation was introduced.
