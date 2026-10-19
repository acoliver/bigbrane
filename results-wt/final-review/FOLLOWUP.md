# Do not integrate: historical preservation and retrieval still fail

**Verdict: REQUEST CHANGES.** The original bypasses have been addressed in source, but B1 and B4 remain **PARTIAL**. Two unchanged Phase7 aggregate assertions still fail, with 16 of 19 historical S2 deaths missing their exact required code. Saved PA2 retrieval remains **FAIL, 2/4 against 3/4**. The corrected registered named-gate scores remain **FAIL, 9/10 and 3/6**. Passing bypass regressions do not establish integration readiness.

This is the findings-only follow-up to `REVIEW.md`, covering B1-B7 and N1-N6 only. It completes the second review cycle without introducing another requirement. Eleven findings are FIXED within their original scope; two are PARTIAL; none is classified OPEN. FIXED denotes the identified defect or reporting discrepancy, not acceptance of the experiment or a claim that its historical gates now pass.

## Reviewed tree and method

All paths are relative to `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, branch `r2-bar`. The reviewed changes are uncommitted over `fa08b93750c19f59808ae1ac18af979e4d8ac310`. I read the original review, the four requested repair reports, the relevant production/test/report diffs and the added reference, regression and provenance files.

I independently compared all 1,254 files in `results-wt/review-fixes/finish-tested-manifest.json` with their recorded hashes: no differences. Its content-set hash is `383c1061d18f6a8c8faa325f12fdf0c86aacdd992d251ebf96a934a9083f524f`. This identifies the tested uncommitted snapshot more precisely than HEAD alone.

Verification used existing worker logs and bounded in-memory reproductions with bytecode writing disabled. I checked fixture hashes and original/copy byte identity, exercised the original bypass inputs and re-evaluated saved comparisons. I also recomputed saved extraction and named-gate scores without writing their outputs. No code edits, live model calls, searches, OCR, duplicate pytest suite, commits or acceptance changes were made. This review writes only this file.

## Dispositions

| ID | Status | Result within the original finding |
|---|---|---|
| B1 | PARTIAL | Independent reference measurements replace self-comparison; square/tanh and state-selected top-K cases are refused. Historical preservation and card-side conformance remain limited. |
| B2 | FIXED | Missing code, failed execution and missing reports cannot verify through confidence. |
| B3 | FIXED | Source-qualified identity is retained; names and coverage assertions alone cannot trigger a checked known-work kill. |
| B4 | PARTIAL | Coverage, excerpts and actual bounded listings propagate through classifier, critic and difference participants. Saved retrieval still fails. |
| B5 | FIXED | Registered exact-gate scoring is restored; both saved guard misses remain failures. |
| B6 | FIXED | Ranking uses distinct normalized queries; related fanout cannot multiply votes; identity and metadata merge. |
| B7 | FIXED | Accepted stored content is scoped, with full gap annotations and call provenance. |
| N1 | FIXED | Schema, host and prompt require two distinct relevant-source/outcome/family entries. |
| N2 | FIXED | Published Elman naming works with both spellings. |
| N3 | FIXED | Query uniqueness and at least four fields are validated at the model-output boundary. |
| N4 | FIXED | Synthetic routing inputs are identified; separately captured saved-program traces carry provenance. |
| N5 | FIXED | Extraction attributes card/attempt costs and matches exact registered works; missing costs remain missing. |
| N6 | FIXED | Missing report artifacts exist; historical and final-tree verification are distinguished without a green claim. |

## Blocking findings

### B1. PARTIAL: independent reference binding works; preservation remains incomplete

**File evidence:** `hybrid/research/stages/s2_prior_art.py:281-321,325-403`; `hybrid/research/stages/published_references.py:10-23,39-169,177-336`; `hybrid/research/stages/sequitur_reference.py:1-98`; `hybrid/tests/research/test_review_gate_evidence.py:31-64,176-184,195-274`; `hybrid/tests/research/test_phase7_replay.py:26-41`.

The host now selects a separately implemented operation from `reference_spec`, checks family, parameter fields and repair types, computes its outputs for each sample, and measures against `card_outputs`. The model's scalar remains diagnostic under `reported_max_abs_diff`. Loaded reports are remeasured rather than trusted for their saved eligibility flags. Missing reference evidence cannot count.

The original declared-tanh square copy is refuted in the bounded reproduction: its reported self-difference is zero, while the independent maximum difference is approximately **0.3415792712**. Genuine tanh traces verify under both Elman names. The unchanged saved state-selected top-K declaration/report is unverified and supplies no counted family. The supported input-SSM operation uses an input-indexed fixed binary mask; there is no state-selected top-K/reset operation in that reference.

Re-evaluating all 14 KD8 records with the separate trace evidence gives ten verified controls and four unverified records: the three original bad restatements and P08's old nonstandard-RNN mapping. Jeffrey/IPF, Dyck-2, FIFO copy, two-counter, Inside, Earley, Viterbi, SEQUITUR and both excluded-family controls have supported comparisons. The old P08 mapping is not counted as evidence for the different Rule 110 mapping in the saved PA2 run. A separate Rule 110 operation and hand-specified answer test exist at `hybrid/research/stages/published_references.py:128-136` and `hybrid/tests/research/test_review_gate_evidence.py:221-245`.

**Residual blocker:** this does not preserve the full historical Phase7 population. `results-wt/review-fixes/finish-phase7-rows.json` contains 19 expected S2 deaths and **16 exact-code misses**, independently counted in this follow-up. The two original aggregate assertions remain unchanged and fail. The supported phase-normalizer and MesaNet controls recover their expected exclusions; card-6de11b01 does not. Its second-memory tag equality and prediction-error overwrite gates are not established by a plain delta-rule reference. `results-wt/review-fixes/finish-PHASE7-REMAINING.md` records the unsupported mapping and its saved 1.23 output difference when the tag is removed. No repair policy was broadened to force agreement.

The original conformance limitation also remains: the host receives authored mapped parameters, transformed inputs and card-side output traces. It does not independently establish that the reported card execution faithfully implements the card's equations or general free-parameter mechanism. Independent reference arithmetic removes the demonstrated comparator-copy bypass, but sampled agreement is not a general equivalence proof. The code and repair reports acknowledge this limit.

**Disposition:** retain unsupported cases as unverified and keep the original failures visible. Historical preservation cannot be claimed from synthetic routing success. This follow-up requests no broader repair type or weaker assertion.

### B2. FIXED: confidence no longer supplies verification

**File evidence:** `hybrid/research/stages/s2_prior_art.py:262-270,392-403,451-489`; `hybrid/tests/research/test_review_gate_evidence.py:66-100,277-288`; `hybrid/tests/research/test_kd4_counted_family.py:171-184`.

Null code, unsuccessful jobs and missing printed JSON return unverified regardless of confidence. Fixed learned parts stop the declaration before execution, and no-report counted-family synthesis has been removed. I exercised all twelve combinations of null/crashed/missing execution with restatement, unresolved, limit-only and fixed-learned-part declarations at confidence 1.0: all remained unverified and uncounted. The existing eligibility/judgment paths also keep a matching limit-only comparison and a restatement from verifying. No retries or tolerance changes were introduced.

### B3. FIXED: identity and claimed coverage are not sufficient evidence

**File evidence:** `hybrid/research/stages/known_work.py:35-101`; `hybrid/research/stages/difference.py:171-207,221-235,247-278,345-358`; `hybrid/tests/research/test_review_gate_evidence.py:103-122,304-323`.

Qualified paper references resolve only to that qualified candidate key. An ambiguous bare id does not resolve. The known-work judge requires an independently referenced counted comparator for an automatic checked-evidence kill; a resolved source or a textbook/algorithm name instead yields unchecked coverage or no evidence. It does not parse `covers` as proof.

The original fabricated-name, wrong-prefix and initialization-only examples no longer kill through checked known work. The difference context explicitly treats unvalidated names as leads, without forcing an impossible citation or rejection. The separately documented opinion-based `textbook_transplant` gate is unchanged and is not represented here as source verification.

### B4. PARTIAL: coverage wiring works; the registered retrieval result remains a miss

**File evidence:** `hybrid/literature/prior_art.py:75-141,226-289,358-375`; `hybrid/research/stages/s2_prior_art.py:158-174,605-643,699-703`; `hybrid/research/stages/difference.py:171-207,211-235,328-358`; `hybrid/tests/literature/test_prior_art_receipts.py:85-128`; `hybrid/tests/research/test_review_coverage_wiring.py:19-91`.

Search/related failures, per-field usable coverage, the classifier's mechanism-coverage assessment and actual listing positions now survive classification. S2 persists them and exposes them with supplied excerpts to the critic. The difference writer and checker receive the same required-source-aware bounded listing and receipt. Bounds remain 60 classifier candidates, 80 critic papers and 35 difference papers. Required late-ranked evidence is reserved; overflow is explicit rather than silently dropped.

The bounded reproduction confirms that an inadequate nonempty result remains unassessed, with **one** wrapper call and no broader-search retry. A usable alternate source can still supply coverage; a single arXiv failure does not automatically reject it. Both difference participants' receipt/excerpt wiring is covered by the existing focused integration regression and its saved affected-file run.

**Residual blocker:** read-only extraction exactly reproduces `results-wt/kd8/live.json`: **FAIL 2/4**, required **3/4**, using the generous any-registered-work interpretation. Rossi is stored at rank 128 but not exposed to deciding listings; Zhou is at 258 and not exposed; Sacramento is at rank 27 and reaches the saved classifier/critic and difference participants. Hwang/Srinivas/Silver and SciPy/Rehn remain absent. New ranking and exposure code does not change those historical observations or demonstrate a new retrieval result.

**Disposition:** the propagation defect is fixed, but the original retrieval finding is unresolved as an acceptance result. Keep it failed; no new searches or retries were made or inferred.

### B5. FIXED: scoring again uses the registered named gates

**File evidence:** `results-wt/r2-bar/tools/score.py:65-109`; `results-wt/r2-bar/SCORE.md:39-70`; `results-wt/r2-bar/REPORT.md:5-10,62-75,171-178`; `results-wt/kd8/tools/tests/test_evidence_tools.py:88-116`.

The scorer now requires both the reduction verdict and the specified gate, and marks the rules unmet when named-gate counts fall short. I independently evaluated the saved S2 ledgers: Rule 1 is **9/10**, Rule 2 is **3/6**. Their all-dead totals, 10/10 and 6/6, are secondary diagnostics only. SEQUITUR's later prior-art death and the three later-gate deaths do not satisfy the registered reduction predicates. The scoring/report defect is fixed; these acceptance failures remain and prevent a green integration conclusion.

### B6. FIXED: query support and metadata survive acquisition

**File evidence:** `hybrid/literature/prior_art.py:145-223,292-305`; `hybrid/tests/literature/test_prior_art_receipts.py:13-52,131-138`.

Each merged paper stores a set of normalized query texts. Related-only support contributes one ranking hit regardless of seed fanout. Source/arXiv/DOI/title identity merging preserves aliases, query support and richer abstracts/metadata, including a late identity bridge between previously separate groups. Repeated refinement queries are not searched twice. Bounded reproductions confirm the original two-query paper ranks ahead of repeated related appearances, and richer later metadata is retained. This repairs the ordering rule without claiming it alone resolves the saved retrieval miss.

### B7. FIXED: limitations persist in the accepted statement

**File evidence:** `hybrid/research/stages/difference.py:431-479,501-509`; `hybrid/research/schemas/difference.schema.json:32-44`; `hybrid/tests/research/test_review_gate_evidence.py:137-173,326-338`.

On acceptance with excerpt gaps, the final card receives a scoped statement. Literature narratives and aggregate comparison prose are replaced with excerpt-limited wording, while refs, equation location and cause ids remain. `evidence_scope` carries the full gap list and writer/check call ids. The artifact stores both the original raw `statement` and the scoped `accepted_statement`; the writer cannot author the host-owned scope field.

The persisted-card regression inspects the final stored card and artifact, rather than schema acceptance alone. The bounded deterministic reproduction confirms full, untruncated gap retention, provenance and nonmutation of the original. Scoping can remove supported detail along with unsupported prose because gaps lack field addresses; that documented tradeoff does not leave the original unsupported assertion presented as accepted content.

## Nonblocking findings and evidence limits

### N1. FIXED: two-prior contract alignment

**Evidence:** `hybrid/research/schemas/difference.schema.json:10-14`; `hybrid/research/stages/difference.py:221-235,247-278`; `hybrid/research/prompts/difference.md:11`; `hybrid/tests/research/test_review_gate_evidence.py:125-134,291-301`.

The schema requires at least two entries, and host checks require two distinct relevant-source/outcome/family entries. Family entries fill missing relevant-source slots; duplicated aliases and unrelated papers do not fill them. The bounded checks refuse one entry and a duplicate family pair, and accept two distinct valid families. This aligns the implementation with the original recorded contract without requiring unrelated citations.

### N2. FIXED: the published Elman cell is not rejected for spelling out RNN

**Evidence:** `hybrid/research/stages/contracts.py:61-62`; `hybrid/tests/research/test_review_gate_evidence.py:47-57`.

The bare recurrent-network class remains excluded without rejecting the specified Elman name. Both names verify against the same independent tanh operation and numeric traces in the bounded reproduction.

### N3. FIXED: field diversity and query uniqueness are checked

**Evidence:** `hybrid/literature/schemas.py:27-51`; `hybrid/literature/prior_art.py:292-305`; `hybrid/tests/literature/test_prior_art_receipts.py:55-69`; `hybrid/tests/research/test_local_index.py:140-146`.

The external-output boundary enforces four distinct field tags and unique case/whitespace-normalized query texts. The single-field fixture now uses multiple fields. Single-field and duplicate-query reproductions are rejected. No topical keyword classifier or additional search requirement was introduced.

### N4. FIXED: routing assumptions and executed evidence are separated

**Evidence:** `hybrid/tests/research/critic_answers.py:1-14`; `hybrid/tests/research/test_orchestrator_fakes.py:106-182`; `hybrid/tests/research/phase7_replay.py:1-19,124-157,187-196`; `hybrid/tests/research/reference_replays.py:1-47`; `hybrid/tests/research/test_reference_replays.py:13-36`; `results-wt/review-fixes/capture_saved_traces.py:20-39,44-124`.

A legacy `radical_enough` answer no longer manufactures a Hopcroft source, tanh components or absent zero/one differences. Synthetic routing specimens are named as synthetic and are not claimed as saved-card evidence. Separate saved-program input/output traces and mapped parameters support the recovered reference comparisons. The original Phase7 fixture and expected-code assertions were not changed.

I verified all 19 fixture-file checksums, ten KD8 original/copied-program byte matches, and three Phase7 original-executor/copied-program byte matches. For all four trace groups, applying the recorded capture insertions to the saved program reconstructs its recorded instrumented-source hash. This is reproducible provenance for captured saved computations, not proof of card-equation conformance. The sixteen Phase7 misses remain reported under B1.

### N5. FIXED: exact matches and attributable costs replace fuzzy global counts

**Evidence:** `results-wt/kd8/tools/extract_live.py:61-231`; `results-wt/kd8/tools/tests/test_evidence_tools.py:23-85,125-138`; `results-wt/kd8/REPORT.md:34-72,94-98`.

Extraction filters events by card, pairs attempts within that card and attributes calls to the explicit primary card JSON id. Registered works match an exact arXiv id or full normalized title, not a word such as Willshaw. Saved prompt positions are recorded separately from stored candidate ranks. Missing artifact fields and incomplete attempts remain visible.

The read-only extraction exactly equals the saved `live.json`. Each of the nine saved card classifications has three attributable literature model calls and three request attempts. Per-card search counts are 16 or 18 with five related lookups, rather than every card receiving every run span. HTTP-attempt counts are unavailable and remain null. The report correctly states an increase of one model call over the old two-call flow.

### N6. FIXED: report completion and exact-tree status are stated accurately

**Evidence:** `results-wt/kd8/REPORT.md`, `results-wt/kd8/live.json`; `hybrid/tests/literature/test_prior_art_receipts.py:149-157`; `results-wt/review-fixes/finish-REPORT.md`; `results-wt/review-fixes/finish-tested-manifest.json`; `results-wt/review-fixes/finish-final-tree-targeted.log`; `results-wt/review-fixes/finish-combined.log` and its existing process/monitor.

The two missing artifacts now exist. The report distinguishes historical suite3's **1,589 passed, three skipped, one failed** from current checks and records that per-shard revisions were not captured. A new test protects all three prompt copies, including `REFINE_SYSTEM`; direct membership checks pass. The final affected-file result is attributable to the hashed uncommitted snapshot and retains the two failures. The combined run is described below as running, without an inferred success. N6's reporting discrepancy is fixed; it provides no basis to disregard the failed acceptance evidence.

## Observed verification status

| Evidence | Observed result | Interpretation |
|---|---|---|
| `results-wt/review-fixes/finish-targeted26.log` and `.exit` | 24 passed, two failed; exit 1 | Original aggregate failures remain. |
| `results-wt/review-fixes/finish-final-tree-targeted.log` | **420 passed, two failed, one skipped** | Final affected-file result; both failures are in `test_phase7_replay.py`. |
| Original exact-code rows | **16/19 expected S2 deaths miss their exact code** | Counted from `results-wt/review-fixes/finish-phase7-rows.json`, not inferred from dead/alive totals. |
| Existing combined research/literature/evidence-tools run | **RUNNING at review close** | PID 90533 and managed monitor `shell_a3f744d002d6`; no exit file observed. No final total or pass is asserted. |
| Bounded read-only follow-up reproductions | Original square copy refuted; top-K refused; 12 confidence bypass cases unverified; identity, coverage, ranking, scoping, diversity and prompt-copy checks behaved as described | No pytest suite or live service invoked by this review. |
| Saved evidence recomputation | `live.json` reproduced exactly; retrieval FAIL 2/4; named gates FAIL 9/10 and 3/6 | Historical results, not a new experiment on the repaired tree. |
| File/provenance checks | 1,254 tested files unchanged; fixture and copied-source checks match; tracked `git diff --check` clean | Snapshot integrity and whitespace checks, not scientific acceptance. |

The failing original tests are:

- `hybrid/tests/research/test_phase7_replay.py::test_every_s2_death_gets_its_expected_phase7_code` at line 29.
- `hybrid/tests/research/test_phase7_replay.py::test_the_linear_attention_and_delta_rule_reductions_are_excluded_as_families` at line 36.

## Unresolved original scope and integration verdict

B1 remains partial because broad historical known-mechanism preservation is not established and card-side conformance is not independently established. B4 remains partial because the registered retrieval evidence still misses its threshold. B5's scoring correction makes the two existing guard failures explicit; it does not repair their recorded outcomes. These facts are sufficient to withhold integration approval regardless of the eventual combined-run total.

Keep the original repair policy, assertions and thresholds. Unsupported comparisons must remain unverified, and the saved misses must remain failures. No broader repair type, lowered assertion, substituted all-dead rule or additional review cycle is requested. This follow-up establishes neither a new theory nor subquadratic learning, sample efficiency or language-model utility.
