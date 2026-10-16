# AI mathematics claims: failures and safeguards for Bigbrane

**OpenAI's October 6 mathematics release already has acknowledged proof failures, and Lean coverage does not establish every claim in its papers.** OpenAI withdrew three manuscripts after finding a sign error, revised 14 others, and logged the changes publicly. Thomas Bloom also identified an October paper claiming a stronger quantitative bound than its accompanying formal statement covers. Separately, a critique of the September Navier-Stokes release documents differences between written lemmas and Lean lemmas. These findings justify separate checks of novelty, statement fidelity, proof validity and search accounting. They do not justify dismissing the entire collection or declaring its remaining theorems false.

Research snapshot: **October 9, 2026**. Read-only `date` returned `Fri Oct 9 17:34:25 -03 2026`. The report path was checked absent, then an initial report was written before completing the source checks. No code, installations, Lean builds, OCR or GitHub mutations were performed.

## Episode register

| Date | Claim and claimant | What scrutiny found, and who found it | Status | Failure mode |
| --- | --- | --- | --- | --- |
| October 17-18, 2025, background | OpenAI VP Kevin Weil: GPT-5 found solutions to ten previously unsolved Erdős problems, open for decades; other staff amplified this | Thomas Bloom explained that GPT-5 found existing papers missing from his database. Sébastien Bubeck acknowledged that only literature solutions were found and deleted his post | Confirmed misleading novelty framing; useful literature search remains | Prior-art status conflated with database status; retrieval described as discovery |
| February 14-20, 2026, background | OpenAI initially considered its First Proof problem 2 attempt likely correct | OpenAI changed its assessment to incorrect after official organizer commentary and community analysis | Acknowledged incorrect proof attempt | Premature confidence before expert review |
| August 1, 2026, background | OpenAI: ten advances from internal Astra; initial announcement said the problems had seen no progress on the main result for at least a decade; selected solution tokens priced at roughly $2,000 | Specialists identified recent antecedents; Scientific American documented an announcement wording change. Stephen Miller alleged inadequate attribution. Andreas Thom explained a new combination of older results. A later audit withdrew its own chapter 6 error allegation | Historical framing corrected; attribution disputed; no surviving principal-result error in the audit corpus examined | Prior art and credit; success-only cost framing; an erroneous review caused by text extraction |
| September 8, 2026, background; follow-ups September 17 and October 6 | OpenAI: finite-time Navier-Stokes blow-up, with written proof and Lean artifacts | Constantin, Ignatova and Vicol proved a limitation for the construction's class under analytic forcing. Bastounis, Circelli and Hansen documented lemma-level translation differences | Scope limitation and formalization mismatch; these findings do not disprove the final formal theorem | Forced case generalized in public interpretation; stronger written lemmas paired with weaker or different formal lemmas |
| October 6, 2026; corrections logged October 7 | OpenAI: 722 manuscripts in 372 result families from an internal model | OpenAI's audit found one sign error invalidating three related proofs; 14 further manuscripts revised | Three acknowledged withdrawals; additional repairs and statement corrections | Proof error; dependency cascade; overstrong local claim |
| October 6 release; Bloom's October 8 post, updated by October 9 | October arithmetic-progression paper: quasipolynomial bounds; geometric similarity paper: all ratios in `(0,1)` | Bloom identified a weaker arithmetic-progression formal bound and similarity formal coverage only at ratio `1/2`. The opened Lean files confirm those scope differences | Formal coverage limitation, with proof correctness still requiring review | Partial or weaker formalization mistaken for verification of the whole paper |
| October 6 release; complaints October 7-9 | OpenAI manuscripts presented as research contributions | Jialong Deng reports missing references in families 335 and 336. Chenglong Ma says an OpenAI proof shares strategy and notation with a draft he uploaded to ChatGPT | Attribution allegations, without an established misconduct finding | Missing credit and provenance; possible prior-art overlap |

## 1. October 6 batch: confirmed corrections

### Claim and release procedure

OpenAI's [announcement, *Sharing AI progress in mathematics*](https://openai.com/index/sharing-ai-progress-in-mathematics/), described new results from an internal frontier model. The [repository README](https://github.com/openai/math) says the model was posed approximately 4,000 problems, that some outputs build on earlier model results, and that outputs were grouped into families and filtered for significance. It describes different verification stages and warns that some unformalized results could have issues. The catalogue initially contained 722 manuscripts and now contains 719, while retaining 372 families.

The README gives an average compute equivalent in ChatGPT Pro thinking units. It identifies exceptions to the fixed procedure for the zeta zero-free region and Hodge conjecture for CM abelian varieties, and human editing of one zeta write-up. This procedure cannot be reconstructed from a manuscript count alone.

[Scientific American's October 6 report](https://www.scientificamerican.com/article/openai-unleashes-hundreds-more-math-results-upon-a-field-already-in-shock/) quotes an OpenAI spokesperson saying almost every result came from a single prompt to a single agent, while acknowledging that some results might have taken multiple attempts. MIT mathematician Andrew Sutherland asks for receipts and treats the one-shot claim as unverified until outsiders can replicate it. The inspected evidence establishes incomplete disclosure of attempts. It does not establish that OpenAI secretly used many attempts for each result.

### Three withdrawn proofs

OpenAI's [history file](https://raw.githubusercontent.com/openai/math/main/history.md), under October 7, names three withdrawals:

1. [Algebraicity of Weil classes on split abelian eightfolds](https://raw.githubusercontent.com/openai/math/main/preprints/Algebraicity-of-Weil-classes-on-split-abelian-eightfolds-September-18-2026/README.md).
2. [Algebraicity of Kuga-Satake Correspondences for K3 Surfaces](https://raw.githubusercontent.com/openai/math/main/preprints/Algebraicity-of-Kuga-Satake-Correspondences-for-K3-Surfaces-October-3-2026/README.md).
3. [The rational Hodge conjecture for products of K3 surfaces](https://raw.githubusercontent.com/openai/math/main/preprints/The-rational-Hodge-conjecture-for-products-of-K3-surfaces-October-4-2026/README.md).

The first notice explains the mechanism. Starting from signed double-point count `I(f₁) = -m`, the proof assigned each reverse stabilization trace sign `+1`, making `m` traces appear to cancel the count. Correct orientation accounting gives `-1` for each trace. The resulting count is `-2m`, whereas the invoked Eliashberg-Murphy cancellation theorem requires zero. The construction therefore lacks its required hypothesis. Both dependent papers adapt that construction and leave their claimed theorems unproved.

Each notice explicitly limits the withdrawal to the proof and leaves the truth of the mathematical statement unresolved. The notices link archived originals at initial revision `adc7f1241b42e322a6451854ab7e4b4c146bf78a`.

**Who found it and how:** [Retraction Watch, October 8](https://retractionwatch.com/2026/10/08/openai-withdraws-preprints-722-manuscripts-unsolved-math-problems/), reports OpenAI's statement that its own audit identified the errors. The inspected record does not identify a particular auditor. Retraction Watch quotes Cornell mathematician Alex Townsend asking for separate announcements for verified work and work needing verification, and Sutherland welcoming prompt withdrawals while retaining broader concerns.

### Further repairs and an overstrong statement

The same history entry records **14 revised manuscripts**, with proof repairs, corrected statements, clearer hypotheses and dependencies, and one obsolete-citation correction:

- Four on Lipschitz heights and Ashkin-Teller currents: crossing, boundary-attachment, conditioning and convergence arguments repaired.
- Six on Kähler minimal model programs and abundance: positivity and contraction arguments expanded; inputs and hypotheses clarified.
- Two on taming and hypersymplectic deformation: the cone-equality claim in *Taming implies compatibility* corrected, a strict-inclusion example added, and a cone-comparison dependency removed from the companion paper.
- *Incompressible Box Transport and Finite Computation*: torus-projection and common-clock estimates revised.
- *Exact Birch-Swinnerton-Dyer Formula from Low Selmer Corank*: obsolete introductory citation removed.

The cone-equality correction supplies a specific case of a claim being too strong. The aggregate of 14 includes citation and exposition changes as well as proof repairs; it would be misleading to count all 14 as false main theorems. A further 13 manuscripts received companion-reference and version-date updates.

### Commit history, removals and complaint access

The opened [commit API](https://api.github.com/repos/openai/math/commits?per_page=100) returned three commits:

| Revision | UTC timestamp | Description |
| --- | --- | --- |
| `adc7f1241b42e322a6451854ab7e4b4c146bf78a` | October 6, 21:58:50 | Initial release |
| `301488868beec11bfd897168433b0a64f5258559` | October 8, 05:03:50 | Dan Roberts: update manuscripts and Lean formalizations |
| `fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb` | October 8, 05:20:00 | James R. Lee: merge update PR #1 |

The update is publicly described in [PR #1](https://github.com/openai/math/pull/1). Its validation section says Lean builds were not run for that synchronization. This is a limit on that update's validation evidence; it does not establish that the proofs fail to build.

There is a dating discrepancy worth preserving: the withdrawal notices say October 6; the history entry says October 7; the update reached main on October 8 UTC, corresponding to October 7 Pacific. Public history and notices disclose the withdrawals and preserve archived versions. No evidence of concealed withdrawals was found in the inspected history. A read-only issue-list request confirmed that **issues are disabled**. The full PR diff exceeded GitHub's 300-file limit, so this research inspected the history and individual artifacts rather than every changed file.

## 2. October batch: what Lean covers

### Arithmetic progressions: weaker formal bound

In [his October 8 post](https://www.erdosproblems.com/forum/thread/blog:10), Thomas Bloom records the paper's bound for progression-free subsets:

`|A| ≤ N exp(-(log N)^cₖ)` for every fixed progression length `k ≥ 3` and sufficiently large `N`.

He observes that the comparator target covers Erdős's reciprocal-sum conjecture, while the actual formal development appears to use a weaker bound. The opened [Lean Model.lean](https://raw.githubusercontent.com/openai/math/main/lean/OAI/Combinatorics/Progressions/Model.lean) defines `QuantitativeDensityBound` with:

`extremalNumber(k,N) ≤ C N exp(-c (log(log N))^(1+η))`.

The different nested logarithms are present in the source. The formal target described by Bloom can still settle the reciprocal-sum conjecture. It does not certify the stronger quasipolynomial bound in the PDF. This is a documented statement-coverage difference, without a demonstrated disproof of the PDF bound.

### Similarity: one parameter value

Bloom records the geometric similarity paper's claim for each ratio `q ∈ (0,1)`, but says its formalization covers `q = 1/2`. The opened [DyadicAvoidance challenge](https://raw.githubusercontent.com/openai/math/main/lean/ComparatorChallenges/DyadicAvoidance.lean) defines the sequence as `(2 : ℝ)⁻¹ ^ n`. That target cannot by itself verify a result for every geometric ratio. Its measure guarantee is expressed differently and can be stronger; the limitation discussed here is the ratio domain.

The challenge file contains a `sorry` placeholder. A comparator challenge states the target against which a separate proof is checked. Its placeholder alone is insufficient evidence of a gap in the submitted proof.

### Coverage and external scrutiny

OpenAI's history reports **300/719 top-line results formalized, approximately 42%**. Preserve that denominator as the company's own figure. A family with some Lean material, a manuscript with one formalized theorem, and a checked main theorem are different counting units.

The opened [Comparator instructions](https://raw.githubusercontent.com/openai/math/main/lean/ComparatorChallenges/README.md) explicitly list five setups covering supporting results rather than corresponding main theorems. The [formalization catalogue](https://raw.githubusercontent.com/openai/math/main/lean/formalization.yaml) also supplies a paper-to-artifact map. Treating every linked Lean artifact as certification of all claims would contradict this documentation.

Bloom explicitly says his post is a summary of claims, without serious proof review or independent compilation. At access it lists 25 Erdős problems, 21 with formalizations, and warns that novelty and attribution need literature review. He also says some linked proofs appeared online before October 6, mostly in papers disclosing AI use, without making a judgment about their novelty or attribution. His remarks are evidence of scrutiny and scope concerns, rather than blanket endorsement.

There is also positive checking evidence: [Plasma AI's October 8 account](https://www.plasma.ai/research/erdos) says its team rebuilt 12 full-statement Erdős proofs from the release and used separate AI sessions to audit statement correspondence. Its reviewers are AI sessions in separate contexts. This supports a limited report of third-party checking, without establishing human specialist review of all 12 statements or acceptance of the entire batch.

**Answer to the requested formal-statement check:** the October collection already has publicly identified weaker or partial formal targets, most concretely the two cases above. The inspected evidence does not show that an October top-level Lean proof was invalid or that Lean accepted a false theorem. The September lemma mismatches below provide a separate example of written and formal arguments diverging.

## 3. October batch: attribution complaints remain unresolved

[Jialong Deng's MathOverflow question](https://api.stackexchange.com/2.3/questions/515845?site=mathoverflow&filter=withbody), posted October 8 and accessed through the public API, says families 335 and 336 generalize results from his September 2 preprint without mentioning it. He reports that OpenAI Support could provide neither a correspondence address nor forwarding to the responsible researchers.

[Chenglong Ma's October 7 preprint, §1 after Theorem 1.1](https://arxiv.org/html/2610.10072v1), says that an OpenAI paper contains the same result, with proof strategy and specific notation identical to a preliminary draft he uploaded to ChatGPT on September 8. He also acknowledges an independent result by Ningyi Li using a different method.

These are direct, attributable complaints. This research did not compare the complete proofs or obtain private-draft access, model training records or an OpenAI response settling them. They support a provenance review and correction channel. They do not establish plagiarism, training-data reuse, or that the whole October result was already available in published literature. No adjudicated October prior-art duplication was found in the sources inspected.

## 4. Background: October 2025 Erdős claims

**Claim:** Kevin Weil's deleted post said GPT-5 found solutions to ten previously unsolved Erdős problems and progress on eleven others, describing them as open for decades. [The Decoder, October 18](https://the-decoder.com/leading-openai-researcher-announced-a-gpt-5-math-breakthrough-that-never-happened/) and [TechCrunch, October 19](https://techcrunch.com/2025/10/19/openais-embarrassing-math/) preserve and attribute that wording.

**Finding and discovery:** [Bloom's October 17 reply](https://x.com/thomasfbloom/status/1979254235075059732) explains that GPT-5 found references solving problems whose answers he personally did not know. A database's open designation represented the maintainer's knowledge. On [October 18, Bubeck](https://x.com/SebastienBubeck/status/1979539604522127746) acknowledged that only solutions in the literature were found and said he deleted his post.

**Failure mode:** the achievement was communicated with an unsupported novelty claim. The literature retrieval itself was useful. The checked sources do not show ten fabricated proofs or a Lean failure in this episode. They show a promotion from database update to purported resolution of previously unsolved mathematics.

## 5. Background: First Proof, February 2026

OpenAI released attempts on ten First Proof questions on February 14. Its [February 20 announcement](https://openai.com/index/first-proof-submissions/) says it initially considered problem 2 likely correct, then judged it incorrect following official First Proof commentary and community analysis. The admission was available in the search-index text of OpenAI's announcement; direct fetch was blocked. The opened [Sienicki and Sienicki audit, §1](https://arxiv.org/html/2608.14673v3) independently cites and describes the reassessment.

The supported classification is an acknowledged incorrect attempt following expert scrutiny. This research did not reconstruct the failed representation-theoretic step or verify the identities of every contributor to that correction, so it makes no further technical claim about the failure. The relevant safeguard is to retain a provisional label until proof review is complete. This episode also illustrates a correction made explicitly by the claimant.

## 6. Background: August 1 ten advances

### Historical framing and attribution

OpenAI announced ten results produced by internal Astra, including new bounds, conjecture resolutions and counterexamples. It said humans prepared manuscripts with the model and that the arguments were subsequently formalized. The announcement priced the tokens for finding the selected solutions at roughly $2,000 using Sol API rates.

[Scientific American, August 6, with later editorial corrections](https://www.scientificamerican.com/article/openais-latest-math-breakthroughs-commit-research-misconduct-experts-say/), documents removal of the initial claim that these problems had seen no progress on the main result for at least a decade. The opened article identifies **Stephen Miller**, rather than the Steven spelling appearing in some secondary accounts, as the Yeshiva mathematician alleging inadequate credit for a sphere-packing argument from his 2016 work with Henry Cohn. The article reports the allegation and OpenAI's response promising updates. It supplies no adjudication of misconduct.

For non-sofic groups, Francesco Fournier-Facio and colleagues identified antecedents from 2016 and 2019. [Andreas Thom's MathOverflow answer, accessed through its API](https://api.stackexchange.com/2.3/questions/513866/answers?site=mathoverflow&filter=withbody), explains how Proposition 2.3 overcomes an obstacle to combining Kun's theorem with Kun and Thom's earlier theorem. He describes a new construction and says he had sought such a mechanism since 2019. This is evidence that antecedents and a new theorem can coexist. The older ingredients do not establish that the final result was already known.

### Costs and mathematical audit

[Senén Barro's August 4 comments to Science Media Centre Spain](https://sciencemediacentre.es/en/openai-announces-results-ten-mathematical-research-problems-using-its-astra-artificial-intelligence) identify missing unsuccessful attempts, human input and cost denominators. His objection concerns what the selected-success figure can tell us about the search's total efficiency. It does not establish a false token bill.

The opened [September 9 version of *A Human Audit of OpenAI's AI-Generated Mathematical Proofs*](https://arxiv.org/html/2608.14673v3), by Mikołaj and Krzysztof Sienicki, reviews 18 chapter-specific assessments. It says no confirmed substantive error in a principal result survives in that examined corpus. It leaves chapter 8's compressed analytic transitions incompletely verified. This is an audit of a review record, without a claim to independently reconstruct all ten proofs.

Its chapter 6 account is an instructive **review failure**: an apparent polarity error disappeared when the typeset source showed an overbar lost during PDF extraction. The reviewers withdrew the objection. Bigbrane must verify criticism against source mathematics as carefully as it verifies proposed discoveries. The inspected August evidence supports corrected historical framing, disputed credit and incomplete verification, with no basis for calling all ten results erroneous.

## 7. Background: September Navier-Stokes

### Forced versus unforced scope

The opened [OpenAI repository](https://github.com/openai/NavierStokesAndEuler) claims smooth initial data and forcing that prevent a global smooth solution, corresponding to alternatives C and D of the Clay formulation. Its stated Navier-Stokes scope includes forcing.

[Scientific American, September 21](https://www.scientificamerican.com/article/did-openai-solve-the-wrong-navier-stokes-problem/), quotes Luis Silvestre distinguishing this result from the unforced question many fluid researchers prioritize. The article accepts eligibility under Clay's formulation while describing disagreement about its meaning for the field.

[Constantin, Ignatova and Vicol, first posted September 17, revised September 29](https://arxiv.org/abs/2609.20803), prove regularity for a specified class sharing two structural features of OpenAI's construction under real-analytic forcing. Their conclusion requires those features and stated bounds. It rules out simply removing the force or making it analytic within that construction class. It supplies no general impossibility result for every approach to unforced blow-up and no refutation of the claimed forced theorem.

**Failure mode:** a valid result in one formulation can acquire a broader interpretation than its assumptions support. Calling this a proven wrong theorem would overstate the available evidence.

### Written versus formal lemmas

[Alexander Bastounis, Fabian Circelli and Anders C. Hansen, October 6 preprint, §3](https://arxiv.org/html/2610.08144), compare written estimates with Lean declarations at revision `f9e8bc5`:

- **Example 3.1, Lemma 8.6 / equation 8.19:** the written estimate uses input derivatives through order `m+4`; the closest formal estimate requires `m+5`. The extra derivative weakens that estimate. Some other comparisons also involve different derivative types.
- **Example 3.3, equation 10.19:** the written pressure-flux bound uses `B_R`; the formal bound also uses `A_R`, and employs a different analytic argument. The preprint explains why the cited relationship between the quantities does not directly recover the written estimate.

The authors explicitly disclaim a judgment on whether the written Navier-Stokes proof is correct. The findings concern intermediate statements and proof correspondence. They do not establish that the final formal theorem encodes a different top-level Clay problem.

**How discovered:** [New Scientist's October 8 article](https://www.newscientist.com/article/2592824-openai-mistranslated-mathematics-into-code-for-its-navier-stokes-proof/) reports that the researchers asked ChatGPT for discrepancies and then checked them manually, rejecting many suggested discrepancies. Its indexed text reports OpenAI acknowledging the mismatch while denying that it invalidates either proof. Direct article retrieval returned HTTP 406; [TechCrunch's opened October 8 follow-up](https://techcrunch.com/2026/10/08/openais-math-solutions-arent-meeting-the-fields-standards-yet/) independently covers the critique and its limits.

**Failure mode:** a generated formal proof can establish a related statement or use a different argument while leaving stronger statements in the written explanation unsupported by that certificate. A correct formal proof of an independently fixed target can still be a mathematical success. Any claim that the entire accompanying paper was verified requires more evidence.

## 8. Concrete safeguards for Bigbrane

The existing [README](README.md) documents exact-oracle evaluation, frozen inputs, validation-only selection, retained process attempts and labeled unknown resource costs. [HYBRID-PLAN.md](HYBRID-PLAN.md) places the claim schema, ledger, freeze and confirmation authority outside the research controller. Those boundaries address empirical selection and evaluator gaming. The historical episodes require additional evidence about meaning, novelty and proof scope.

The following are recommended requirements, without a claim that they have already been implemented.

| Safeguard | Failure modes and episodes addressed | Required evidence or enforcement |
| --- | --- | --- |
| **S1. Prior-art receipt before novelty language** | 2025 Erdős retrieval; August historical framing; October missing-reference complaints | Record exact queries, dates, databases, returned sources, citation chains and closest theorem statements. Compare assumptions and conclusions against retrieved papers. A negative search supports only “no matching prior result found in this search.” Refresh the search before publication. Archived-packet search alone supports only packet-relative conclusions. |
| **S2. Independent target before proof search** | October progression and similarity scope; September lemma mismatch | A human specialist or separate agent sees the research question and sources, writes the formal statement before seeing the candidate proof, and reviews quantifiers, domains, constants, regularity, asymptotics and definitions. Check for empty domains, inconsistent hypotheses and assumptions that already contain the conclusion. Freeze its hash. Proof search cannot edit it. A weaker target becomes a new claim with an explicit implication relation and fresh review. Separate-agent review stays labeled as such. |
| **S3. Separate statement checking from kernel checking** | A compiling proof certifying a different target | The trusted checker pins Lean, Mathlib, source hashes and the target, builds the proof independently, and records the theorem declaration and transitive axioms. Reject unapproved axioms and proof holes in the theorem's dependency closure. Keep target-template placeholders separate from submitted proofs. Record semantic-review evidence independently of compilation. |
| **S4. Paper-to-proof coverage manifest** | October weaker bounds and single-ratio case; September differing arguments | For every advertised theorem and lemma, record the exact formal declaration, assumptions and status: equivalent, weaker, special case, supporting only, or absent. Uncovered consequences retain an unverified label. Claiming a faithful translation additionally requires a reviewed correspondence between the written and formal arguments. |
| **S5. Complete search denominator** | August selected-success price; October one-shot claims | Extend the attempt ledger to problem selection, every model call, branches, retries, critique calls, formalization and human edits. Report problems attempted, attempts per problem, failures, withdrawals and deduplicated accepted claims. Publish retained-candidate cost and total search cost separately. Keep unknown costs visible. |
| **S6. Define “one-shot” before measuring it** | October single-prompt interpretation | Distinguish one user prompt, one agent run, one candidate proof, and zero retries or proof-feedback loops. An agent run can contain many internal calls. Require a transcript-backed receipt for the exact definition; otherwise describe the observed run without one-shot wording. |
| **S7. Preserve empirical scope through confirmation** | Forced/unforced interpretation; stronger claim inferred from weaker evidence | Freeze the claim, model, candidate, distributions, metric, baseline and acceptance rule before protected confirmation. Exact-oracle accuracy concerns the declared evaluation distribution. A finite empirical learning curve cannot establish universal learnability, a theorem or an asymptotic rate. Any scope expansion needs a new claim and new evidence. |
| **S8. Dependency suspension and public errata** | One sign error invalidating three October papers | Store parent theorem and experiment hashes. A refuted or withdrawn parent suspends descendants. Recheck them before restoring status. Append corrections, changed statements and withdrawal reasons without overwriting original receipts or moving stable links to different claims. |
| **S9. Independent check before announcement** | First Proof reassessment; October unreviewed batch | Public discovery wording requires reviewed prior art, statement fidelity, independently reproduced relevant checks and a bounded claim. Human specialist review is required before a public mathematical novelty claim. Pending items can be released as labeled candidates or requests for review. Give a named contact and accessible correction channel. |
| **S10. Verify objections and require explanation** | August lost-overbar audit error; September model-generated discrepancy suggestions | Check alleged errors against original TeX or source expressions, record the exact failed inference or counterexample, and independently validate it. Require an explanation of assumptions, proof structure, dependencies and limitations before acceptance. Preserve withdrawn criticisms as well as corrected proofs. |

### Claim-strength wording rules

Use **“retrieved an existing proof”** for literature retrieval. Use **“candidate theorem, unreviewed”** for a generated argument awaiting checks. Use **“kernel-checked proof of [exact statement and assumptions]”** only after trusted independent checking and statement review, with uncovered parts named. Use **“confirmed on [specified generators and budget]”** for empirical confirmation. Use **“new result”** only after prior-art and independent mathematical review; search evidence still cannot prove exhaustive absence from the literature.

Avoid “solved mathematics problem X” when only a special case, supporting lemma, weaker bound or empirical surrogate has passed. Report formalized theorem counts, manuscript counts and result-family counts separately. Do not compute an autonomous solve rate from `372/4,000`: the numerator groups selected outputs, and the denominator does not expose attempts, repeated questions or the acceptance decision for each input.

## 9. Source access and remaining uncertainty

This research opened primary repository notices, history, commit records, a PR, Lean target files, Bloom's blog and 2025 response, Bubeck's correction, arXiv critiques, and MathOverflow content via its public API. It opened follow-ups from Retraction Watch, Scientific American and TechCrunch. Links throughout point to those records. OpenAI announcement pages and AGMAI's site blocked direct retrieval; announcement content was available through indexed search text and the repository. The original August Wayback capture returned HTTP 429, so the announcement edit is attributed to Scientific American's reporting rather than presented as an independently compared archive. The First Proof admission and New Scientist response are explicitly identified above as indexed text.

The opened [Lean Zulip public archive](https://leanprover-community.github.io/archive/) displayed a last-update date of February 28, 2026. Targeted searches did not produce an accessible October correction thread. This archive cannot support a conclusion that the live Lean community found no errors. MathOverflow's browser pages returned HTTP 403, but its public API supplied the two discussions cited above.

No Lean build or proof audit was run here, and the complete October corpus was not independently refereed. Remaining needs include specialist review of the corrected papers, full statement audits of formal targets, provenance responses to attribution complaints, and complete attempt records for the one-shot claims. Search did not establish a representative error rate for the collection. No additional other-lab failure is asserted without a verified incident within the requested year.

The practical policy for Bigbrane is to keep novelty, mathematical validity, statement fidelity, empirical confirmation and search efficiency as separate acceptance records. An exact oracle and a Lean kernel each check a defined object. Review must establish that the checked object supports the published claim.
