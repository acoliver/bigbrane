# Request changes: S2 can still count restatements and unsupported novelty decisions

**Verdict: REQUEST CHANGES.** Reviewed and exercised HEAD `fa08b93750c19f59808ae1ac18af979e4d8ac310` on `r2-bar`, against merge base `1f9d51b82f6f9f12d8e5baf9f18aec098b7241c4` with `p4-diagnosis`. There are seven blocking findings and six nonblocking findings or evidence limitations below.

The changes improve the questions asked of the models, and the saved controls recognize several published mechanisms. They do not yet establish an evidence-based novelty screen. Executable comparators remain independent of their component declarations. Confidence can bypass even a correctly declared restatement or limit. The cited-source path accepts an assertion of coverage without checking coverage. The available PA2 runs also miss their registered retrieval target.

This review makes no claim that any candidate is a new theory. The independent audit identifies four incremental cards and one narrow, unvalidated lead. None of the supplied S2 evidence establishes the user's combined goal of a new sequence-learning theory, subquadratic compute and sample efficiency.

## Scope and method

I reviewed the complete 56-file branch diff, including production code, schemas, prompts, changed tests, the KD8 replay fixture, decision records, replay configurations, scripts and report claims. I read `DECISIONS.md`, `hybrid/config/families.toml`, the main repository's `KNOWN-DEFECTS.md`, and `results/flip-audit-20261018/AUDIT.md`. The family file is under `hybrid/config`, not the repository root.

All paths below are relative to `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2`, unless explicitly marked main repository. Physical line numbers refer to the reviewed files. Long JSON string fields sometimes occupy one physical line.

Verification was read-only except for this requested report. I made no code changes, live calls, full-suite runs, commits, issue posts or OCR calls. I inspected the other worker's existing results and ran small in-memory reproductions with the sibling worktree inserted first in Python's import path and bytecode writing disabled. I also evaluated all 14 saved KD8 fixture records through the production comparison judge without rerunning their programs.

## Blocking findings

### B1. P1: component declarations do not constrain the executable comparator

**Locations:** `hybrid/research/stages/s2_prior_art.py:197-232,348-379,419-440`; `hybrid/research/stages/contracts.py:99-125`; `hybrid/research/prompts/reduction_check.md:31`; `DECISIONS.md:42`.

`component_problem` examines names, origins and citations in the declaration. The execution path then runs unrestricted model-written `check_code`. `report_problem` binds only the id, family, transformation type names and `limit_only` value. Neither the report contract nor a separate host computation establishes that the code implements the declared activation, update or readout.

**Reproduced:** declare an Elman cell with the published activation `tanh`, but execute these functions:

```python
def card_update(h, x):
    return (h + x) ** 2

def comparator(h, x):
    return card_update(h, x)
```

A report of the actual zero difference on two inputs, carrying the declared id and family, passes `component_problem` and `report_problem`. `judge_comparisons` returns `verified`. The comparator is a copy of the card, not the declared tanh cell. Declaring only the activation also passes even though the other component roles are absent.

This contradicts the explanation in `DECISIONS.md:42` that numerical matching will necessarily expose a card function falsely labelled as a standard function. Both executable sides are still selected by the same author. Adding a sentence to the prompt does not establish that relationship.

There is a related production example outside the three component-checked families. In `results-wt/kd8/runs/flipA/artifacts/reduction_critic/card-3485a13f-v7-dfbdce.json:20-28,89-107`, `topk_selective_diag_ssm` declares a **state-selected** top-K mask as `sparsity`. The policy permits only fixed or input-selected sparsity (`contracts.py:42`). Its executable sorts the current hidden scores and resets evicted channels in `results-wt/kd8/runs/flipA/jobs/job-38540869dffb/in/main.py:137-145`. The host counts this comparator and kills the card as `excluded_family:ssm` at `results-wt/kd8/runs/flipA/ledger.jsonl:551`.

The numerical equality in that saved run is not disputed. It establishes equality to the state-selected beam rewrite. It does not establish membership in the input-dependent SSM form that the declaration names. Whether that card is independently incremental remains a different question.

**Required remediation:** bind counted comparisons to independently specified published operations and enforce the existing repair definitions, rather than accepting their labels as proof. This does not call for a brittle scan for particular function names. Add an executable declaration/code mismatch test and retain the saved state-selected-mask example as a regression. The copied nonlinear comparator must not verify, while genuine tanh cells, Jeffrey/IPF and independently specified published automata must remain recognized.

### B2. P1: the no-report confidence path bypasses KD8, limits and learned-part restrictions

**Locations:** `hybrid/research/stages/s2_prior_art.py:258-264,335-345,393-427,613-634`; `hybrid/research/stages/known_work.py:63-70`.

Before execution, `declaration_problem` checks repeated ids only. When no code is supplied, or execution crashes, times out or produces no report, confidence at the configured threshold returns `verified` with `report=None`. `counted_comparisons` then manufactures a counted family from `reduces_to`. It never applies the new component eligibility rule to that declaration.

**Reproduced through `run_reduction_check`:** with confidence `0.95` and `check_code=None`, a comparator classified by `eligibility` as `restatement` returns `verified`; so does a comparator classified as `limit_only`. Both become:

```text
[{'declared_id': None, 'family': 'rnn', 'comparator': ''}]
```

The same critic can explicitly list a learned part fixed at zero. The early reduction gate still kills before `known_work.judge` could apply its cited-source protection. This is an inherited fallback interacting with the new contract, not a newly introduced confidence mechanism.

**Required remediation:** apply the existing comparator eligibility and general-mechanism requirements on every verification path. A missing or failed check must not turn a declaration already known to be a restatement or limit into a counted published reduction. Add focused tests for null code, a crashed job and a missing report, using restatement, unresolved and limit-only declarations. Preserve their non-verifying status; do not lower tolerances or add retries.

### B3. P1: the new known-work evidence gate accepts names and coverage assertions as checked evidence

**Locations:** `hybrid/research/stages/known_work.py:35-47,60-76`; `hybrid/research/schemas/reduction_critic.schema.json:25-42`; `hybrid/research/stages/s2_prior_art.py:653-663`.

For textbooks and standard algorithms, any nonempty `ref` is accepted. For papers, the host checks membership in the candidate id set, after discarding the source prefix. `covers` is never examined or independently checked. The critic sees at most 80 titles, without source excerpts, on the path that can immediately kill the card. A cited-source kill occurs before the difference challenger, so that challenger cannot correct it.

**Reproduced with schema-valid critic outputs and no counted comparator:**

- `standard_algorithm`, ref `not a real algorithm`, covers `does not cover the card update`: `kills=True`, basis `cited_source`.
- `paper`, ref `s2:abc`, covers `only initialization, not transition or readout`, with only candidate `arxiv:abc`: also `kills=True`.

The prefix example additionally shows that a wrong source is accepted as the same citation. More importantly, even an explicit statement of partial coverage is reported as a source covering the mechanism.

This follows the permissive identity rule recorded in R2C, but that rule does not deliver the asserted checked-evidence property. It remains a design limitation of the implementation against the user's goal and the guiding evidence rule in `DECISIONS.md:5`. The separate, explicitly opinion-based `textbook_transplant` exception is documented; this finding concerns the path represented as evidenced known work.

**Required remediation:** distinguish source identity from checked support for the general mechanism. Use the existing unevidenced outcome until the claimed source and mapping are supported. Keep source-qualified identity, and test fabricated names, mismatched source prefixes and a real source that covers only initialization. Do not substitute host keyword parsing of the `covers` prose for verification.

### B4. P1: PA2 loses search-coverage failures before classification, and its saved retrieval target is not met

**Locations:** `hybrid/literature/prior_art.py:76-102,132-136,197-210`; `hybrid/research/stages/s2_prior_art.py:152-168`; `results-wt/kd8/STATUS.md:16-19`.

Search receipts retain source errors, and related lookups retain their errors, but PA2 consumes only the returned papers. Coverage status is absent from the classifier prompt and returned classification. The existing empty-search check marks only a `novel` result with zero candidates as unassessed. One unrelated hit is enough to bypass that check even when other coverage is missing. A read-only reproduction with a nonempty unrelated candidate returned `novelty_unassessed=False`.

This is observable in the saved production results:

- `results-wt/kd8/runs/flipA/ledger.jsonl:196,201` record HTTP 429 and cooldown skips for arXiv. Across this run, 51 of 52 searches record an arXiv error. Its `card-f5ee7edb` and `card-8154d3b4` verdicts nevertheless pass at lines 530 and 541 with prior-art labels `novel`.
- `results-wt/kd8/runs/flipB/ledger.jsonl:284,289` record the same failure modes. Eleven of its 36 searches have arXiv errors.
- The system retrieves Rossi's `1906.05059` for `card-f5ee7edb` at candidate rank 128 (`results-wt/kd8/runs/flipA/artifacts/prior_art/card-f5ee7edb-v7-2bd18a.json:1146-1154`) but labels the card `novel` at line 2691. Rank 128 is outside the classifier's 60, critic's 80 and ordinary difference listing's 35. Storage alone does not make a paper available to the model deciding the claim.

The saved candidate sets also miss the registered retrieval target. The following is a generous score, counting a card as retrieved when **any** of its listed referee works is present:

| Card | Found registered referee works | Missing registered works |
| --- | --- | --- |
| `3485a13f` | None | Hwang 1110.5667, Srinivas 2204.09837, Silver 1904.06317 |
| `f5ee7edb` | Rossi 1906.05059, rank 128 | Rendle Factorization Machines |
| `a4f684fb` | None | SciPy/QAP 2-opt, Rehn thesis |
| `89248ee5` | Sacramento 1207.7196, rank 27; Zhou 0901.0553, rank 258 | Jiang 1409.0334, Stoklosa 2411.14480 |

That is at most **2 of 4**, below the registered **3 of 4**. Matches were checked by title and the supplied arXiv ids. The two `flip` candidate directories cited above are the evidence; no new searches were performed. A stricter interpretation requiring all named works would not improve the score.

These observations do not prove that the cards are repeats or that every partial-source search must fail. They show that missing coverage is currently invisible to the novelty judgment, and that the proposed retrieval repair has not demonstrated its own stated target.

**Required remediation:** carry coverage and related-lookup failures into the existing novelty assessment and user-facing evidence, and preserve the distinction between retrieved, shown and assessed sources. Correct the retrieval/listing behavior that omits the known leads. Score the current experiment as not meeting its registered retrieval criterion, without changing the threshold or interpreting more candidates as success.

### B5. P1: acceptance scoring was relaxed after the results

**Locations:** `results-wt/r2-bar/STATUS.md:5-13`; `results-wt/r2-bar/tools/score.py:78-94`; `results-wt/r2-bar/SCORE.md:39-64`; `results-wt/r2-bar/REPORT.md:5-13,66-87`.

The registration requires all eight published probe mechanisms to die at the reduction gate, and the two controls at the excluded-family gate. It separately requires every previously counted reduction to die at the reduction gate, explicitly excluding a later-gate kill.

The scorer instead marks each rule `MET` when all cards die for any reason. It computes named-gate counts but does not use them to decide success. The report calls this compliance with the rules' intent.

The supplied results show:

- Rule 1: 9 of 10 at the required gates. SEQUITUR dies on `prior_art_tested`, after its numeric reduction is refuted.
- Rule 2: 3 of 6 at the reduction gate. `440f0204`, `62785c10` and `d6dfeac2` die through other checks.

Under the written registration, both rules are **not met**. A later kill can hide a broken reduction check. Commit `d929ba1` changes the scoring to intent-based dead/alive reporting after commit `885e8dd` recorded the replay score.

**Required remediation:** restore reporting against the registered predicates and mark the existing results accurately. Investigate the recorded mismatches against those predicates. Preserve the useful secondary dead/alive counts, but do not use them to replace the acceptance conditions or describe the guards as met. This requests no new gate.

### B6. P2: ranking counts appearances, not distinct queries or a single related-only hit

**Locations:** `hybrid/literature/prior_art.py:69-79,108-136`; `hybrid/literature/schemas.py:27-48`.

The pool increments a scalar for every appearance, with no query identity. A paper reached from several seed reference/citation lists receives several votes, although the docstring says related-only papers count as one hit. Repeated identical query proposals also receive separate votes. Source-qualified ids can split the same paper across different queries, and `setdefault` discards later richer metadata instead of merging it.

**Reproduced:** add paper A in two searches and paper B once from each of five related lists. The pool ranks B first with five hits, ahead of A with two. This directly contradicts the specified distinct-query ranking. It matters because ranking controls all three bounded listings and the five citation-follow-up seeds.

No claim is made that this particular synthetic ordering caused the saved rank-128 miss. The ordering rule itself is incorrect independently of that miss.

**Required remediation:** track distinct query support separately from related discovery, deduplicate repeated query text, and merge paper identity and available metadata across acquisitions. Add tests for repeated citations across seeds, duplicate proposals, the same work under multiple source ids, and a later result that supplies an initially missing abstract. Keep the documented listing bounds.

### B7. P2: unsupported prior-paper claims are stored unchanged as accepted difference statements

**Locations:** `hybrid/research/stages/difference.py:398-402,430-457`; `hybrid/research/prompts/difference_check.md:9`; `results-wt/r2-bar/runs/batch8/artifacts/difference/card-8154d3b4-v7-37833f.json:10-21,40-51`.

The checker can accept a statement while identifying unsupported paper claims in `excerpt_gaps`. Send-back text scopes them only when some other failure triggers a send-back. On acceptance, the host stores the original statement first, then appends a clipped warning to the stage reasons. The stored `difference_statement` contains neither the gap annotations nor a revised version of those claims.

The tensor-card artifact is an actual example: the checker says the abstract does not establish the claim that `1605.05775` is a fixed classifier with no sequential mechanism. The accepted statement still says `1605.05775 is a fixed supervised classifier` in its `why_it_failed` field. Similarly, the motif-card artifact `card-f5ee7edb-v7-a51b33.json:12-17,48-52` identifies an unsupported claim about infini-gram while retaining it in the accepted statement.

A stage-reason warning can inform a reader who sees that warning. It does not scope the statement subsequently carried by the card, and the 600-character combined warning can omit later gaps.

**Required remediation:** preserve the limitations with the accepted statement, or scope its unsupported assertions before it is presented downstream as accepted. Do not turn an absent abstract detail into either proof of absence or a new automatic rejection. Add an end-to-end test with `accepted=True` and nonempty `excerpt_gaps` that inspects the final stored card, not just checker-schema validation.

## Nonblocking findings and coverage gaps

### N1. P2: the two-entry difference contract was removed without updating the decision

`DECISIONS.md:41` says the total closest-prior count must remain two, with family entries filling missing source slots. `difference.py:206-244` enforces only the relevant literature/outcome count; the unchanged difference schema permits one entry. `hybrid/tests/research/test_known_work.py:104-109` explicitly accepts one family entry. A schema-valid one-entry statement returned no host problems in the read-only reproduction.

Commit `f74f2f2` and `results-wt/r2-bar/REPORT.md:35-36` disclose the removal, so it is an intentional divergence rather than a hidden implementation accident. It still disagrees with the recorded contract and prompt. Restore the stated source/family count and its test without forcing unrelated paper citations. This is a narrow contract discrepancy, not evidence that any saved flip depended on the removed check.

### N2. P2: the universal-class regex also rejects a named published Elman cell

`contracts.py:61-62` matches `recurrent neural network` anywhere in the extracted head. At `s2_prior_art.py:208-210`, a published comparator named `Elman recurrent neural network (1990)` with a standard tanh activation is classified as a universal class. The same cell named `Elman RNN` passes. This was reproduced.

Distinguish a specified published cell from a bare universal class instead of making the outcome depend on abbreviation. Add paired-name tests for the same executable published cell. Recognition of genuine named mechanisms should remain intact.

### N3. P2: cross-field coverage is requested in prose but not checked at the external boundary

`QUERY_PROPOSAL_SCHEMA` checks 10 to 14 items and field enum membership, but not four distinct fields or distinct query text (`hybrid/literature/schemas.py:27-48`). Ten copies of the same query tagged `optimization` validate. The changed fixture in `hybrid/tests/research/test_local_index.py:140-146` uses one field for all ten queries, so the tests normalize a proposal that violates the production prompt.

Validate the already stated proposal contract and add single-field and duplicate-query cases. The receipt field tags should describe actual requested coverage, not merely item count.

### N4. P2: replay fakes manufacture the evidence that the new checks are meant to evaluate

`hybrid/tests/research/critic_answers.py:2-11` converts every old `radical_enough=False` into known work with the same Hopcroft textbook source, regardless of the recorded mechanism. `phase7_replay.py:124-145` adds tanh and published-update declarations and synthesizes reports from saved differences. `test_orchestrator_fakes.py:109-114` similarly gives unrelated comparators tanh and a generic textbook update. These changes keep routing tests usable but do not establish a factual migration from resemblance to known work.

The KD8 fixture is more explicit about its assumptions: `results-wt/kd8/tools/replay_kd8.py:1-8,48-65` adds the rejection labels by hand to the three bad examples, and marks other mechanisms published mechanically. All 14 records produce the expected host outcomes, including Jeffrey/IPF. That verifies the label interpreter under those annotations. It does not verify that a live critic will declare or execute the right components. The new tests do not cover B1 or B2, and the saved PA2 live run does not replay the three KD8 problem cards.

Retain useful routing tests, but identify synthetic evidence as synthetic and add cases with the original missing evidence and declaration/code mismatches. Do not present the unchanged replay kill codes as proof of unchanged scientific behavior.

### N5. P2: the new live extractor cannot report accurate per-card costs or exact source matches

`results-wt/kd8/tools/extract_live.py:42-55` collects every query/result span in the run and attaches the full cost list to every card. Its comment says literature events lack card ids, but `CardLiteratureSink` adds card ids at `s2_prior_art.py:105-106`. The `flipA` and `flipB` runs contain multiple cards with interleaved activity, so pairing global starts and results by position can mix cards and misattribute their costs.

The retrieval regexes at `extract_live.py:8-20,35-37` are also broader than the registered works. For example, any title containing `willshaw` can count as Sacramento, and `storing sequences` can count as Jiang. Match the named work by id or full normalized title before scoring it. Filter cost events by card id and attempt. Report current missing artifact fields as missing rather than as evidence that no tagged queries ran.

The module calls rise from two to three per classification, an increase of one, while `DECISIONS.md:43` says two additional model calls. Correct that accounting from the existing receipts rather than extrapolating it from candidate counts.

### N6. Evidence/report completion and HEAD provenance remain incomplete

At review close, `results-wt/kd8/REPORT.md` and `results-wt/kd8/live.json` were absent, despite references in `DECISIONS.md:43` and `results-wt/kd8/STATUS.md:11`. The other worker was preparing reporting; this review does not infer completion or success from exit codes.

The older report's research-suite count is supported by the four saved tails: 408 + 394 + 248 + 341 = 1391 passed, with two skips. Those runs predate KD8/PA2. The newest available `results-wt/kd8/suite3/g0.log` through `g6.log` show 1589 passed, three skipped and one failure. The failure is the prior-art prompt-copy assertion in `g3.log`. At the reviewed HEAD, direct membership checks find all three current prompt strings in the current document, so that saved failure is consistent with a pre-`fa08b93` snapshot. No new full-suite success at exactly this HEAD was established by this review. The existing copy test at `test_orchestrator_units.py:214-216` checks only two prompts, despite the prompt document claiming three are protected, and names a nonexistent test file in that document.

Record the final worker result with its actual code revision and distinguish repaired-current-source checks from historical failing logs. Include a check for the third prompt. This is a reporting/provenance limitation, not a claim that the current prompt-copy test still fails.

## Verification results and limits

The following small checks were actually executed against the reviewed source, without network access or file writes:

| Check | Observed result |
| --- | --- |
| Published tanh declaration, executed square-update copy | Component check accepts; report check accepts; reduction verified |
| Declared restatement, null code, confidence 0.95 | Verified with synthetic counted family |
| Declared limit-only comparator, null code, confidence 0.95 | Verified with synthetic counted family |
| Fabricated algorithm reference with explicitly absent coverage | Cited-source kill |
| Wrong-source paper prefix with partial coverage | Cited-source kill |
| Nonempty unrelated candidate under the existing empty-search guard | Not marked unassessed |
| Single family closest-prior entry with no relevant source | Schema accepted; no host problems |
| Five related appearances versus two query appearances | Related-only paper ranks ahead |
| Ten identical queries in one field | Schema accepted |
| Fully spelled published Elman name | Classified as universal-class restatement |
| Current QUERY_SYSTEM, CLASSIFY_SYSTEM and REFINE_SYSTEM in current prompt document | All present |
| All 14 saved KD8 replay records through production judge | Three KD8 problem records unverified; P08's old RNN mapping unverified; Jeffrey, P01 to P07 and both controls verified under the supplied annotations |

The four saved PA2 published-mechanism controls all die at S2: Dyck-2, two-counter automaton, Viterbi and Rule 110. Rule 110 is now recognized as `other_known`, with a counted cellular-automaton comparator, rather than as a nonstandard RNN activation. This supports preservation of those named mechanisms. Jeffrey/IPF preservation is supported by the annotated replay; a new live Jeffrey check is not present in the reviewed PA2 experiment.

The PA2 flip runs show three passes (`f5ee7edb`, `8154d3b4`, `a4f684fb`), one difference rejection (`89248ee5`) and one excluded-SSM reduction (`3485a13f`, subject to B1). These are observations from saved runs, not new novelty verdicts. Most of the saved prior-art artifacts in the two flip runs lack the fields added by `fa08b93`; later probe artifacts include them. The run set therefore spans code snapshots. The query and related receipts still provide evidence of the search operations.

The source-error handling in `hybrid/literature/client.py` records external failures rather than hiding them, and the new lookups still pass through the existing transport/pacer. The problem in B4 is their loss from the decision input. The fully empty, `novel` search path still retries broadly and then becomes unassessed under the default setting. That behavior was not removed.

The changes do not alter the math profile, target exponents, learning thresholds, bench assertions or family data. The replay configurations deliberately stop at S2 and therefore supply no evidence about later mathematical validity, subquadratic performance, sample efficiency or language-model utility. No additional acceptance gate is proposed here.

## Coverage by work item

| Work item | Disposition |
| --- | --- |
| R2A: general learned mechanism | Prompt and fixed-part schema implemented; cited-source fixed-part branch works; early verification paths still defeat the intended restriction (B1, B2) |
| R2C: known work, evidence, recorded surprise | Fields and routing implemented; surprise is not gated; source-coverage evidence is not established (B3) |
| R2D: scoped difference evidence | Checker-schema consistency and mandatory cited-paper entries implemented; persisted scoping and two-entry contract incomplete (B7, N1) |
| KD8: refuse universal-template reductions | Declared bad components rejected; executable binding and confidence bypass remain; naming false positives and synthetic-fixture limits remain (B1, B2, N2, N4) |
| PA2: cross-field and refined search, citation follow-up, ranking, receipts, bounds | Operations and caps implemented; distinct-query ranking, coverage semantics and registered retrieval outcome incomplete (B4, B6, N3, N5) |
| Tests and report claims | Existing evidence inspected; registered gates scored under a weaker predicate; final KD8/PA2 report and exact-HEAD suite result missing (B5, N6) |

## Data, secrets and maintainability

The changed committed files include research text, saved comparator declarations/results, a census of local artifact paths and experiment scripts/configurations. A scan of all 56 changed files found no private-key blocks or long literal token/password assignments. This is a bounded scan, not a credential audit of every ignored live-run file. API keys still travel through the existing header/redaction path; no new credential storage was introduced in this diff.

The worktree already contained untracked census logs, suite directories, copied stores and `results-wt/r2-bar/tools/dbg.py`. They were not modified or staged. Copied stores and raw logs should remain separate from the requested review artifact when preparing an integration commit. Several result scripts hardcode the local main/worktree paths and one obsolete patch script rewrites production source; they are experiment scaffolding, not a reusable verification interface.

The production helpers are small and the new model-output schemas fail fast on structural errors. The principal maintainability concern is that prose, declaration labels and freely authored programs can disagree while the host reports a checked result. Fix those deciding interfaces rather than adding fallback wrappers or repeated attempts.

## Integration disposition

Do not integrate this as a verified fix for novelty/reduction and prior-art coverage. Address B1 through B7 under the existing stated rules, retain the successful published-mechanism controls, and report the current acceptance misses as misses. The saved tensor-circulation candidate can remain an unvalidated research lead. Nothing in this review establishes a new learning theory or a sample-efficiency advantage.
