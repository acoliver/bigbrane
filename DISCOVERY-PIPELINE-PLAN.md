# Bigbrane discovery pipeline: build plan

Date: 2026-10-12. Status: plan only. Writing it changed no code, config, run or result file; the only new file is this one. Sources are named beside each claim. Paper URLs were checked with live searches on 2026-10-12 (section 21); anything not confirmed is marked UNVERIFIED.

## Summary

The current loop (S0 to S6 in `hybrid/research/ORCH-STATUS.md`) generates cards that mostly rediscover known families (122 of 196 proposals by Andrew's count, `reports/2026-10-creativity/research-llm-creativity.md`), and its S3 feasibility gate still rests on two model judges. This plan replaces S0 and S3, keeps S1, S2, S4, S5, S6, the ledger, the store and the executor, and adds four things the loop lacks:

1. **A math stage that a model cannot argue past.** A separate agent fixes the formal statement before any proof attempt. Claims are checked with Lean 4 + Mathlib (rejecting `sorry` and non-standard axioms through `#print axioms`), Z3, SymPy and executed simulation, and each claim records the tier it reached. Every card must discharge a fixed "transformer profile": exact cost verified by running code, executed constructions for copying, recall, nesting and composition, derived limits, toy learnability, and information bounds. A challenger from a different model family runs a magic-X test, conservation-law checks from a shared library, and a statement-match check by a third agent.
2. **Generation as measured arms over a quality-diversity archive**, with a written, checkable definition of "radical" and a hard exclusion of the transformer, SSM, linear-attention and delta-rule families.
3. **An outcome store** keyed by a normalized mechanism signature, so a variant of a dead idea must name the recorded cause of death and change it.
4. **Two acceptance tests before live use:** calibration of the math stage on four known models presented without names, and a planted-card test of the whole system with 49 cards of seven types whose correct outcomes are sealed in advance.

Phases, in dependency order (section 17): 0 disk hygiene and decisions; 1 tools image (Lean, Mathlib, Z3, SymPy) with a smoke test; 2 math library; 3 math roles and the assumption ledger; 4 transformer profile and calibration on known models; 5 outcome store; 6 generation arms and archive; 7 refinement dialogue and difference statement; 8 budgets and stall detection; 9 planted-card test; 10 test ladder (runs alongside 2 to 9); 11 live run of 30 ideas; 12 the six-year-old question set, built only when a survivor needs it.

Two findings from the verification pass change the image build: current z3-solver wheels for Linux aarch64 need glibc 2.38 while the executor base (Debian bookworm) has 2.36, and Lean v4.30.0-rc1 shipped an x86 `leantar` in its Linux aarch64 build (fixed upstream). Both are handled in Phase 1.

Decisions needed from Andrew are listed in section 18. The ones that block work: the compute target numbers (cost exponent ceiling and state growth rule), whether the S3 judges leave the gate entirely, the allocation of the paced GPT route, and whether a base model is available for the base-model arm.

---

## 1. Goal, scope and the definition of "radical"

### 1.1 Goal

Agreed with Andrew: generate radical, non-derivative ideas for language learners that beat LLMs on data efficiency (learning from about a child's language input) and on compute (well below quadratic, no ever-growing GPU memory). Check ideas with real mathematical tools before any training, and train only the survivors.

Numbers the plan uses, from `reports/2026-10-creativity/research-radical-substrates.md` section 5: a six-year-old hears about 7M to 100M words, central estimates 27M (LENA) to 39M (Hart and Risley working class). BabyLM's 10M-word track is the low end of one child; its 100M track is about four times the LENA mean.

Measurable form of the compute target, proposed here and pending decision D1:

- total training and inference cost over a sequence of length n is O(n^a) with a at most 1.5 (the verifier checks the exponent from executed op counts, section 4.4);
- inference state grows at most as O(log n) bits per token position processed, or is fixed; a store that grows with the corpus (not the context) is allowed only if its size and lookup cost are declared and verified.

### 1.2 Out of scope as a category

Evolutions of the transformer, state-space models (S4, Mamba, SSD), linear attention and the delta-rule family (DeltaNet, gated DeltaNet, fast weights, RLS-style and MesaNet-style in-context least squares, Titans). Today S2 lets a reduced card through when `difference_if_any` starts with `LEARNING_RULE:` (`stages/s2_prior_art.py`). Under the new scope, a verified reduction into one of these families fails the card regardless of that prefix. The list lives in one config table (`hybrid/config/families.toml`, new) that the reduction critic, the archive and the planted-card generator all read.

### 1.3 "Radical", as three checks

A card is radical when all three hold. Each clause is decided by a check that the proposer cannot edit.

| Clause | Check | Decided by |
|---|---|---|
| R1. It changes at least one foundational commitment: representation substrate, learning mechanism, unit, objective, or where knowledge lives. | A classifier agent (not the proposer) assigns the card to one value on each of the five archive axes (section 6.2) from its equations and code. At least one axis differs from every excluded-family reference tuple. An executed ablation replaces the differing component with the reference family's component; the card's toy result must change. A commitment that can be swapped out without changing the toy result is decoration. | classifier role plus executed ablation |
| R2. It does not reduce to a known family under the existing equivalence tests. | The S2 reduction critic and its executed numeric check, unchanged, plus the family exclusion in 1.2. | existing S2 |
| R3. If it reuses a known-failed idea, it names the recorded cause of failure and changes exactly that cause. | Outcome-store lookup by mechanism signature (section 9). On a hit, the difference statement must cite the cause id, and the challenger checks that the changed equation is the one the cause names. | outcome store plus challenger |

---

## 2. What exists, and what happens to each part

Read for this plan: `hybrid/research/ORCH-STATUS.md`, the stage modules in `hybrid/research/stages/` (4,643 lines with the research package), the prompts in `hybrid/research/prompts/` (16 files), `hybrid/config/run.toml`, `hybrid/config/models.toml`, `hybrid/executor/` and `IMPROVEMENT-PLAN.md`.

| Part | Today | Plan |
|---|---|---|
| `ledger.py`, `store.py` | Hash-chained ledger, immutable cards and artifacts | **Keep.** New event and artifact kinds only. |
| `hybrid/executor/` | Podman jobs, `--network=none`, read-only rootfs, labels `bigbrane.owner`, `bigbrane.run`, `bigbrane.job`, receipts | **Keep.** Add a second image (tools image, Phase 1) through the same `Job` and lock-file path. |
| S0 `s0_explore.py`, lenses, `diversity.py` | Lens-driven explorer, TF-IDF duplicate check | **Replace** the explorer by generation arms and the archive (section 6). Keep TF-IDF as a cheap pre-filter. Lenses survive as one input to the cross-field arm. |
| `graveyard.py` | Up to 20 one-line outcomes in the explorer prompt | **Replace** by outcome-store queries (section 9). The renderer is reused for the prompt block. |
| Escape operator `escape.py` | Child card that must break the invariant a reduction relied on | **Change.** Becomes the "repair the recorded cause" operator, charged to the lineage budget (section 10). |
| S1 `s1_wellformed.py`, salad critic, toy example | Fatal and fixable findings, toy example run in the executor | **Keep** as the entry check of the refinement dialogue (section 7). The dialogue's challenger replaces the revision loop. |
| S2 `s2_prior_art.py`, reduction critic, numeric check, `prior_art_delta` | Literature classification, reduction with executed equivalence check | **Keep and extend.** Add own-history search, the difference statement, the family exclusion and the one send-back (section 8). |
| S3 `s3_feasibility.py`, formalizer, disprover, two judges, `obligation_checks.py` | Model-judged feasibility, executed obligation checks with replay | **Replace** by the math stage (sections 4 and 5). Reuse the executor pattern of `obligation_checks.py` (program, replay, decided only when both runs agree) and the disprover role. The judges leave the gate (decision D2). |
| S4 `s4_construct.py`, builder | Candidate code with static and executed checks | **Keep.** The builder also receives the math stage's construction code as a reference implementation. |
| S5 `s5_evaluate.py`, `falsify.py`, falsifier designer and gates P0-1 to P1-8 | Card-specific falsifier tasks, validated before use | **Keep** with every gate. The standard bench becomes the test ladder (section 11). |
| S6 `s6_verdict.py`, rule override | Claims ledger decided by pre-registered rules | **Keep.** Adds a comparison of stated predictions against results. |
| `plumb.py` | Operator override replay of single stages | **Keep.** Calibration (section 5) uses the same override banner. |
| `inspect.py`, report | Dossier, creativity block | **Extend** with archive map, tiers, assumption ledger, kill reasons. |
| Experiment 1 files: `bigbrane/`, `tests/`, `EXPERIMENT.md`, `README.md` | Frozen | **Untouched.** A test asserts their sha256 values are unchanged after each phase. |
| `IMPROVEMENT-PLAN.md` | Recall-frontier target and bench calibration | Bench calibration (B.2 to B.6, C) feeds the test ladder. The recall-frontier target (A.1, A.4) is superseded by section 1.1. |

---

## 3. The pipeline at a glance

Andrew's eleven steps map to stages as follows. The scheduler, queues, budgets and resume logic of `orchestrator.py` stay; stage names change.

| Step | Stage | Code | Leaves the stage when |
|---|---|---|---|
| 1 Idea | G (generate) | `stages/g_generate.py`, `archive.py` | card drafted and assigned to an archive cell |
| 2 Refinement dialogue | R (refine) | `stages/r_refine.py` | challenger has no more questions, or 3 rounds used; assumption list written |
| 3 Prior art and own history | P (prior art) | `stages/s2_prior_art.py` (extended) | difference statement accepted, or reduction found |
| 4 Refine or drop | P, one send-back | same | the send-back answer is judged, then pass or drop |
| 5 Math | M (math) | `stages/m_math.py`, `hybrid/math/` | every profile obligation has a tier and a status |
| 6 Challenge | C (challenge) | `stages/c_challenge.py` | magic-X, conservation and statement-match results recorded |
| 7 Revision | V (revise) | `stages/v_revise.py` | repair fixes the named failure, then back to P; or stall |
| 8 Budgets | all | `stages/common.py` `Budget`, `lineage.py` | budget or stall stops the loop |
| 9 Test design | T (test design) | `stages/t_design.py`, `falsify.py` | predictions pre-registered, tests ordered |
| 10 Test | X (execute) | S4, S5 | ladder rung results recorded |
| 11 Feedback | F (feedback) | `outcomes.py` | outcome record written for every terminal card |

The `_enqueue` invariant carries over: a card enters stage k only with a `pass` at stage k-1, and enters X only with both M and C passed.

### 3.1 Card fields added

The theory card schema (`schemas/theory_card.schema.json`, fields today: id, version, parent_ids, created_by_call, lens, title, summary, core_idea, source_transplant, non_reduction_argument, extensions, mechanism, claims, known_relatives, falsifiers, toy_example, references, status) gains:

```python
class Commitments(TypedDict):          # assigned by the classifier role, never by the proposer
    substrate: str                     # value from hybrid/config/archive.toml
    learning_rule: str
    unit: str
    objective: str
    structure_source: str

class Assumption(TypedDict):
    id: str                            # "A1", "A2", ...
    text: str
    status: Literal["proven", "cited", "measured", "open"]
    evidence: str | None               # Lean declaration, citation URL with quote, or artifact id
    used_by: list[str]                 # obligation ids whose conclusion depends on it
    added_by: Literal["dialogue", "statement_fixer", "challenger"]

class DifferenceStatement(TypedDict):
    closest_prior: list[PriorRef]      # literature receipts or outcome-store ids, at least two
    what_was_tried: str
    why_it_failed: str                 # quote or outcome cause id
    precise_change: str                # one sentence
    changed_equation: str              # pointer into card.mechanism
    cause_ids_addressed: list[str]     # required when the signature matched a dead idea

class Prediction(TypedDict):          # written at T, before any run
    claim_id: str
    task: str
    metric: str
    value: float
    tolerance: float
```

`archive_cell`, `lineage_id`, `mechanism_signature` and `arm` are orchestrator-filled.

---

## 4. The math stage (step 5)

### 4.1 Roles and the order of calls

1. **Statement fixer** reads the card and the assumption list and writes, for each profile obligation and each card claim, a formal statement in one of four forms: a Lean `def Claim : Prop`, a Z3 formula with an explicit bound, a SymPy identity or inequality, or a simulation spec (generator, metric, threshold, sizes). It also writes the dependency edges from assumptions to statements. The output is hashed and frozen (`math_statements` artifact) before any prover call.
2. **Prover** gets the frozen statements and writes proofs or programs. It cannot change a statement: the Lean statement file is staged byte-identical (sha256 checked), and the prover supplies only `theorem claim_holds : Claim := ...` in a separate file. The checker confirms the elaborated type of `claim_holds` is exactly the constant `Claim`.
3. **Checkers** run in the tools image (Phase 1) through the executor. Results are replayed as in `obligation_checks.py`: a result counts only when two runs agree.

The challenger (section 5) runs after the checkers.

### 4.2 Tool tiers

| Tier | Meaning | Accepted when |
|---|---|---|
| `lean` | proven in Lean 4 with Mathlib | compiles; `#print axioms claim_holds` lists only a subset of `propext`, `Classical.choice`, `Quot.sound`; no `sorryAx`, no `Lean.ofReduceBool` or `Lean.trustCompiler` (native evaluation), no user `axiom`, no `unsafe`, no `implemented_by` or `extern` in the proof's files, no `set_option` that weakens kernel checking; statement file hash matches |
| `smt_bounded` | proven by Z3 for all inputs up to a stated bound | `unsat` for the negation within the bound, bound recorded; `sat` gives a counterexample, which refutes |
| `sympy_exact` | exact symbolic identity or inequality | `simplify(lhs - rhs) == 0` or a SymPy proof of sign over the stated domain; numeric spot checks at 20 random rational points must agree |
| `executed` | observed on examples | simulation program meets its threshold on every listed size, two runs agree |
| `argued` | prose only | never satisfies an obligation |

Reporting rule: the verdict text says "proven" only for `lean` and `sympy_exact`, "proven up to bound B" for `smt_bounded`, and "observed on the listed sizes" for `executed`. The card's math verdict lists the tier of each obligation, so executed-on-examples is never reported as proven.

Lean rules come from the Lean reference manual: the three standard axioms are `propext`, `Classical.choice` and `Quot.sound`; `sorryAx` marks an incomplete proof; native evaluation adds `Lean.trustCompiler` and cannot be checked by external checkers ([Axioms](https://lean-lang.org/doc/reference/latest/Axioms/), [Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)).

### 4.3 Assumption ledger

Every assumption from the dialogue, the statement fixer and the challenger goes on the card's ledger with a status:

- `proven`: has a `lean` or `sympy_exact` result;
- `cited`: a published result, with URL and the quoted statement; the statement-match role checks the quote supports the use;
- `measured`: a measurement on real data, with the artifact id (for example a corpus statistic computed by a job);
- `open`: none of the above.

Gate rule: if any obligation the card's conclusion depends on has an `open` assumption among its dependencies, the card is `unproven` and cannot pass M. Dependencies are written by the statement fixer and can be added by the challenger; neither the proposer nor the prover can remove an edge.

### 4.4 The transformer profile (required obligations)

Each card must discharge all five. The reference implementation is the card's mechanism code (Python, NumPy, no torch autograd unless the learning rule needs it), run in the tools image.

**P1. Exact cost.** The statement fixer writes symbolic formulas for time per token, memory per token, total sequence time and inference state size, in n, d, V and the card's own size parameters. The cost verifier (`hybrid/math/cost.py`) instruments the reference code with an operation counter (multiply-adds, comparisons, memory words allocated and live), runs it at n in {64, 128, 256, 512, 1024, 2048, 4096} with two values of every other size parameter, and requires:

- counted operations equal the formula exactly at every point (SymPy evaluation), or within a stated lower-order remainder that the formula declares;
- the fitted log-log slope of total operations in n is within 0.05 of the formula's own slope over the same finite grid (finite-grid slope), and the formula's asymptotic exponent in n equals the claimed exponent (asymptotic exponent); lower-order terms depress a finite-grid slope, so the two are checked separately. With a declared remainder the formula's finite-grid slope widens to the band of slopes that counts within the remainder can have on the same points;
- peak live memory words match the state formula.

Counting units (decision FIX-3, 2026-10-13). The counter's units are fixed by `cost.py`, not by a definition: a contraction of length k written with `@` counts k multiply-add units per output element, the same product written as multiply then sum counts 2k-1, elementwise functions one per output element, reductions k-1. `init` does not receive n, so state that grows with n is measured as the peak of what the state holds as tokens arrive, and every Python number in the state is a word. The statement fixer and the prover both receive this convention as text measured from the counter when the prompt is built. In the calibration smoke the fixer froze a scalar-dot convention while the counter charged `@` as k units; card k2's reference code then failed `ops_exact` at every point, and the same source passes every exact check against the formula written in the counter's units. The checks stay exact and no tolerance was added; the fix is the stated convention.

Activations and remainders (decision FIX-3, 2026-10-13, from the k2 smoke diagnosis). The counter has `ops.np.sigmoid`, the logistic sigma(x) = 1/(1+exp(-x)) at one transcendental unit per element, as `tanh`. A definition that charges a sigmoid one operation is then executable as written; a sigmoid written out from `exp` costs its expanded units (negative, exp, add, divide) and the formula must count them. The convention text states the sign: 1/(1+exp(x)) is sigma(-x). Card k2 froze a unit-cost sigmoid that the counter could not execute; its reference code fell back to 1/(1+exp(x)), and its third attempt, which wrote scalar dots as the frozen definition counts them, missed the formula only by the expanded sigmoid (78784 against 74176 at n=64, d=8, V=4). With the unit sigmoid that source meets the frozen formula exactly at every point.

A remainder stays the only allowance and is part of the frozen statement: `ops_remainder` or `state_remainder` with `ops_remainder_lower_in` or `state_remainder_lower_in`, the sizes in which it is of strictly lower order than its formula. Before freezing, the host refuses a remainder that has no such list, is negative or not below its formula at some grid point, or whose asymptotic exponent in a listed size is not below the formula's; the verifier refuses the same claim if it ever reaches a job. A remainder can therefore admit lower-order implementation detail, such as k2's 9nd of bias and pointwise work against a leading n(16d^2+2dV), but never a leading-order difference, such as the 8nd^2 that MAC-counted dots leave against a scalar-dot formula, and it cannot be fitted after a failure. 9nd is lower order in d at fixed V, not in n, and the list must say so. The finite-grid slope checks use the band of slopes that counts within the remainder can have; the asymptotic exponent check is unchanged.

P1 passes when the formula is verified. A separate target gate (`hybrid/math/target.py`) then compares the verified exponent and state growth with the target in 1.1 (decision D1) and fails cards that miss it with `obligation_failed:target`. Keeping the two apart lets calibration (section 12) check that the profile describes a transformer correctly while the target gate still rejects it.

Wall time is recorded but does not decide (noise on the shared VM, as `IMPROVEMENT-PLAN.md` J.1 notes).

**P2. Constructions, executed.** A hand-set configuration of the mechanism (no learning) must solve four tasks at stated sizes:

| Task | Generator | Pass |
|---|---|---|
| Copy | string of length L over D symbols, separator, reproduce | per-string accuracy at least 0.99 for every listed L |
| Associative recall | MQAR format of Zoology: K key-value pairs, then queries ([arXiv 2312.04927](https://arxiv.org/abs/2312.04927v1)) | accuracy at least 0.99 for every listed K |
| Nesting | Dyck-2 prefixes up to depth D, predict the legal closers | accuracy at least 0.99 for every listed D |
| Composition | two-hop lookup f(g(x)) with f and g given in context, plus a variable-binding variant | accuracy at least 0.99 for every listed table size |

The card states, for each task, the resources its construction needs as a function of the size (for example state bits as a function of K). These resource curves feed P5.

Task contracts (FIX-3, 2026-10-13). The statement fixer and the prover receive, for every P2 and P4 task, a contract measured by running the generators in `constructions.py`: the layout fields at each size, the token order, whether the tables are shuffled, the chain structure, when predict is called, the pair count against the key universe, and the exact token-stream length per size (composition 5T+3, binding 4·hops+7, computed from the generator rather than written down). A construction that keeps rows per input token declares `token_store` (arrays, tokens, width); before the statements are frozen the host checks the declared tokens against the real streams of the instances the check will run, and the stores alone against the declared state words, and returns a mismatch to the fixer as a problem to fix. In the smoke, card k1 froze 4T+3 and 3·hops+3 tokens against streams of 5T+3 and 4·hops+7. Copy is run with self-fed continuation: after the separator the harness feeds the mechanism its own previous prediction, never the stored true symbol, which is what card k2's frozen copy definition promised and what the harness did not do (it fed the true string). The stored instance still holds the true continuation, which P4 training examples carry. For a deterministic mechanism and exact whole-string success the pass event is the same either way; the change removes the contract mismatch, and after a first error the mechanism reads its own wrong symbol as the task defines. Learners size their output layer from `layout['vocab']`, stated in the learnability contract and the prover prompt; the k1 and k2 smoke learners derived it from the integers in the layout and produced too many logits (k1 composition 20 for a vocabulary of 15, k2 copy 6 for 5).

Shared architecture contract (decision FIX-4, 2026-10-13). P1, P2 and P4 are held to one declared architecture, so a witness cannot run a model that P1 does not count. The statement fixer freezes `profile.architecture` = {`hidden`, `precision_bits`}: the feed-forward hidden width H as a formula in d ("d" for square matrices, "4*d" for a 4x expansion, only what the card states) and the float width of stored numbers (16, 32 or 64). Every P1, P2 and P4 statement declares `widths` = {`d`, `rounds`, optionally `positions`} as formulas in its own size names (n and the cost sizes for P1, the task size for P2 and P4); `positions` defaults to n for P1 and to the longest token stream the generator makes at that size for P2 and P4. The checks (`hybrid/math/architecture.py`, called from the cost, construction and learnability checks) survey every array the code holds: module-level data, the state `init` returns, the parameters of the cost reference, the model a learner returns, and every list, tuple or dict of arrays, whose length counts as an axis. At every point they require:

- no axis above the largest declared dimension, max(d, H, rounds, positions, vocab);
- P2 floats stored at no more than `precision_bits`;
- when H differs from d, the P1 reference holds an array of width H, so the counted cost is the cost of the declared architecture.

The fixer schema requires the contract. A statement without `widths` goes back to the fixer as a problem to fix; one that still lacks them is dropped and its slot left unproven. A witness that breaks the contract fails its statement however accurate it is. The check is a resource audit and does not show that a witness uses the card's update rule. Reasoning: in the calibration smoke card k1's nesting witness used a decoder of hidden width H = N·K·2, 2176 at depth 16 against d = 224, while P1's reference counted square d-by-d matrices; nothing compared the two, and the witness scored 1.0 at every depth (`results/phase4-20261013T073748Z/k1-c-diag/REPORT.md`, C3 and C10). Under a square contract that witness now fails P2.nesting at depths 8 and 16. Under a contract wide enough for it, the square P1 reference no longer realizes the declared H and P1 fails (`hybrid/tests/research/test_fix4_challenge.py`).

Construction memory and the contract's authority (decision FIX-6, 2026-10-14). Within one run, after a failed construction the prover receives that statement's best measured attempt of the run (its accuracy and state at every size and its full source) and the frozen contract evaluated at each size (rounds, d, H, positions, precision, state words). A prompt section states that the frozen contract is the authority: a claim that it is insufficient is argued as a finding with evidence and is not acted on by submitting a witness known to fail. Nothing is carried between runs and the checks are unchanged. Reasoning: in the 2026-10-14 calibration (card k1, run 1) the prover asserted that three rounds were needed and submitted a most-frequent-output composition witness and a lower-constant binding witness, while a two-round construction under the same frozen contract had passed in an earlier run and replays at 1.0 (`results/phase4-20261013T073748Z/k1-calib-diag/REPORT.md`, section 2).

**P3. Derived limits.** The card states at least one task family it provably cannot solve within its resources, with the derivation. The challenger checks that the stated limit is consistent with P2 and with the conservation library. A card that claims no limit fails P3: every finite mechanism has some, and Hahn 2020 and Merrill and Sabharwal show even the transformer has them.

Matching a P3 statement (decision FIX-4, 2026-10-13). The slot is existential: one task family the mechanism cannot solve. A statement instantiates it by naming the family (in the limit and its definitions) and proving that no setting of the mechanism within the stated resources solves every member. Its formal statement is then universal over settings or readouts and concrete in the task, which is the instantiation and not a quantifier difference. The statement-match comparison therefore receives the frozen limits record and the slot claim with the limit filled in (`instantiated_claim`); the blind back-translation still sees only the statements and definitions. The comparison judges the statement against the template and the limit, and does not add criteria such as depth or novelty. Reasoning: in the calibration smoke the matcher failed k1 on its P3 theorem (a single-token readout over a V-symbol vocabulary cannot report length V) by reading the witness's universal readout quantifier as a mismatch with the existential slot, and by requiring a non-trivial limit the slot does not state. Whether such an output-domain limit should satisfy P3 is a profile rule, open for Andrew (`results/phase4-20261013T073748Z/FIX-4.md`, proposal P3); until it is decided the literal slot text governs.

**P4. Toy learnability.** From its initial state, using only its own learning rule, the mechanism must learn the P2 construction from data in at least one solvable case per task (small V, small K, depth 2), with the number of training examples m reported per task, on three seeds. The card states m before the run. Reference learners (a two-layer transformer and an LSTM at matched parameter count) run on the same toy so the comparison is direct.

Training recipe (decision FIX-3, 2026-10-13). The prover owns the P4 training recipe, and the harness applies it identically to the true data and to both controls. The frozen statement fixes the model class (the card's architecture and learning rule, with the widths the card states), m, the seeds, the 0.99 threshold, both data controls and the job budget. The training recipe (initialization, optimizer, learning rate, epochs or passes, batch size) is chosen by the prover within the card's learning rule and that budget, declared in the learner's `RECIPE`, recorded with the result, and applied by the harness, through the same `learn`, to the true data and to both controls; a recipe that changes during any training stops the check as a prover fault. Reasoning: in the calibration smoke the statement fixer froze an untested recipe for card k1 (width 16, learning rate 0.01, 50 full-batch epochs) under which the seed-0 loss moved from 3.000009 to 2.987527 against 2.995732 for uniform guessing, and no prover could change the recipe, although nothing showed the architecture was at fault. The claim P4 tests is that the card's rule learns the task from m examples; the recipe is a search setting within that rule. The controls guard against a recipe that does the work itself: a recipe that reaches the threshold without the data (on no examples and on random answers) fails with `data_independent`, and the recipe cannot differ between the true run and the controls. One cost remains: the prover sees the deciding test accuracies and can adjust the recipe across at most the stage's four attempts. That is far less selection than a sweep on the test set, but it is some; a separate development split for recipe feedback would remove it and is not built.

**P5. Information bounds.** Two inequalities, computed by the conservation library from the card's numbers:

- storage: state bits S against bits that must be held to answer the P2 tasks at the claimed sizes (K log2 V for recall, L log2 D for copy, D bits for Dyck-2 depth D with two bracket types);
- sample information: m (from P4 and from the card's data-efficiency claim) against the bits needed to identify the target rule, log2 |H| for the hypothesis class the card names, divided by the information an example carries. Structure built into the mechanism counts as supplied bits and is added to the prior's account (this follows the accounting rule in `reports/2026-10-lineages/research-pre2017.md` section 18: count pretrained embeddings, hand-defined primitives and any programs that generate training data).

Target-rule class (decision FIX-4, 2026-10-13). Every P5 sample entry defines the class it counts: `hypothesis_class` = {`definition`, `size_justification`, `log2_size`}, where `log2_size` equals the entry's `log2_H` and the justification gives the count or derivation and the prior bits the mechanism's built-in structure supplies. The class is the family of target rules the task could follow (for example every readout table from 4 symbols to 4 answers, 4^4 = 256), not the set of every function the mechanism can represent. An entry without it goes back to the fixer as a problem to fix, and a P5 statement that still lacks it is dropped, leaving P5 unproven. For a record frozen before this rule, stage C raises the missing class itself as one finding on P5 (an open premise that keeps P5 unshown). Reasoning: card k1 copied "at most 8 bits" from the card into four sample bounds with no class defined; the conservation library decided the inequality but could not establish its input (diagnosis findings C9 and C11, one gap reported twice).

Card completeness (decision FIX-16, 2026-10-15). A frozen statement contains no value that is neither in the card nor set by the protocol. Every frozen field of a profile statement is one of three kinds (`profile.FIELD_KINDS`). Protocol values are set by the host (`profile.PROTOCOL` and the constants above): P2 sizes exactly the required sizes, the generator's default params, count 200 (the instances the task contracts are measured on), threshold 0.99, and `positions` equal to the generator's stream formula (left out for dyck); P4 seeds 0, 1 and 2, test_count 500 and threshold 0.99; P1 lengths 64 to 4096 and every other cost size at 8 and 32, with conformance points on that grid. Card values come from the card's new structured `resources` section (theory_card schema, optional for older cards): the shared architecture (hidden width, precision, carried-state bits), the P1 cost size names with d and rounds, and for each P2 task d, rounds and state words as formulas in the task's size, for each P4 task the solvable size, the example count m, d and rounds. Derived values are computed by the host: each P5 copy, recall and dyck entry's state bits (the carried-state budget at the task's d, else state words times precision) and its D or V. The host fills in every value the fixer leaves out and returns as a problem every value that differs (formulas compared after SymPy simplification); a statement that still carries one after the retry is refused at freeze (`hybrid/research/stages/card_resources.py`, called from `math_contracts.fixer_problems` and `freeze`). A completeness gate runs before the statement fixer: when `resources` leaves an item unstated (or a mechanism shape symbol uncovered), stage M asks the card's author for the missing items under the `math_revision` budget (stage V's revision step, run inline because there is no V stage and a card cannot return to R), and ends the card `undetermined` with kill code `card_incomplete` and the list when the author does not supply them; no math check runs on such a card. Stage R asks once after its dialogue, and the explorer, refine and fixer prompts tell authors to state resources. In calibration an incomplete fixture is refused before any model call. Reasoning: calibration card K2 never stated its store width, so the fixer chose one per run; at d = 64 the recall task at K = 64 draws from 128 possible keys, the witness shared cells and scored 0.757, and all six such runs failed, while the three runs at d = 128, 4K+8 or 8K+32 passed (`results/phase4-20261013T073748Z/STATUS.md`, "K2 recall: cause found"). The audit of three calibrations found the same freedom in widths, rounds, positions, hidden width, state bits, state words, count, the P1 grid and the threshold (`results/phase4-20261013T073748Z/fix16/audit-varying-frozen-values.txt`). The four calibration cards now state their resources (`results/phase4-20261013T073748Z/FIX-16.md`). The gate also refuses a stated architecture the contract check refuses (a carried-state budget above 32 bits). The planted T5 override, which runs M and C without stage R, gives the sealed card the resources its author stated in the main line's stage R. The first test run after the change had 28 failures: 25 were fakes that relied on the fixer inventing widths, counts or state bits, and were fixed by having the fakes copy the card's resources or by stating the tested value in the card; 3 were the T5 override ending every sealed card `card_incomplete`, fixed in `planted.py`.

Capability coupled to cost (decision C17, FIX-17, 2026-10-15; installed by a watcher after calibration PID 42807 exits). The target gate reads the verified P1 formula with every size but n held fixed, while P2 constructions are proven at widths that grow with the task size, so a card could pass the target at a fixed d and prove recall at a d that the target never saw. For each P2 task the host now substitutes the card's stated resources (`resources.tasks.<task>`: d, rounds and state words in the task size s) into the verified P1 cost and state formulas with SymPy (`hybrid/math/coupled.py`) and reports the skill scale b: the largest exponent with s = n^b, total cost at most n^1.5 and state at most n^0.5 (`run.toml` [target] `coupled_max_cost_exponent` and `coupled_max_state_exponent`). The state bound is sublinear, not D1's fixed or O(log n), because recalling K pairs needs K log V bits in any mechanism, and under D1 every card would get b = 0. A task whose rounds differ from the rounds P1 counted is `uncovered`; a task whose needed bits (conservation library) outgrow the card's carried bits at d(s) is `bounded`. A P2 slot takes the worst of its tasks. Stage M records the block in every verdict (`target_coupled`); when the target gate applies, a card with b > 0 on no P2 slot fails with `obligation_failed:target_coupled`. Calibration records b and never scores it; the fixed-size target verdict (D1) is unchanged. Measured on the calibration cards with their verified P1 formulas: K1 (transformer) none, since even a constant size costs n^2; K2 (LSTM, recall width 4K+8) b = 1/4 on every slot, set by the n d^2 cost (the state bound alone allows 1/2); K3 (HMM) and K4 (n-gram) bounded on every slot, their state holds recall only up to K = 1 and K = 3.

Two related rules from the same fix. P3 (decision P3): each stated limit names the part of the mechanism it follows from (`basis`: state, update or readout, with the card's anchor and a quote of its text there); a limit that rests only on the output vocabulary or the task encoding, or whose anchor or quote is not the card's mechanism text, goes back to the fixer and is refused at freeze, and when every limit is refused P3 is not formalized. At stage C the challenger receives the card's resources and may report a `resource_contradiction` (for example "fixed state" against a width that grows with the task, or a claimed size the stated width and precision cannot hold); it is settled only by runnable code over `witness.resources` and `witness.conservation` or by a SymPy derivation, and a confirmed one fails the claims it names with kill code `resource_contradiction`. Statement-match disputes go to a third judge (decision XC, section 5.4).

The theory the profile rests on is listed in section 21.1.

### 4.5 Why these five

Each obligation corresponds to a result about transformers, so a candidate is held to what is known about the model it must beat: universal approximation (Yun et al.) and Turing completeness under unbounded precision (Pérez et al.) set what a construction can show; the limits in Hahn 2020 and the TC0 bound of Merrill and Sabharwal show what a derived limit looks like; RASP and Tracr show constructions as executable programs; induction heads (Olsson et al.), Birth of a Transformer (Bietti et al.), Zhang, Frei and Bartlett, and Nichani, Damian and Lee show that gradient training finds such constructions in solvable cases, which is the form P4 asks for; Zhang et al. 2017 and Lotfi et al. are the reason P5 asks for counted bits rather than parameter counts.

Each slot asks for a finite, executed or derived result, and the challenge (section 5) holds a card to that and no more (decision FIX-4, 2026-10-13). P1 counts inference work; training is outside it. P2 allows hand-set parameters, so a table set by hand is not a premise; a P2 witness is challenged for using something the mechanism does not have (an external oracle or index, a wider hidden layer, more rounds or more precision than declared). P4 asks that one prover-chosen recipe within the card's learning rule reaches the threshold from m examples on three seeds while failing both data controls; it does not ask for a guarantee over every optimizer, recipe or length, or for asymptotic learnability. An objection to the card's explanation of how its mechanism works attaches to the card's own capability claims, never to a profile slot. The challenger receives these scopes as a prompt section (`prompt_sections/challenge_profile.md`).

Settlement of challenger findings (decision FIX-4). Before FIX-4 every component the challenger listed without a cost statement of its own became an open premise "component X works as the card says", and the challenger's own premises had no field for a plan or a test; in card k1 that turned eleven objections, eight of them restating executed results, into permanent blockers (diagnosis, section 2). Now a challenger allegation is settled before it can block:

- Components. Each component is "mechanism" or "external". It is runnable when a function of that name is defined in the source of an executed statement that held. Its cost is covered when that statement is the P1 cost statement, or when it is a P2 or P4 witness that held under the frozen architecture contract while P1 held under the same contract. A runnable mechanism component without covered cost is an accounting finding on P1; its executed result stands. A training procedure (`learn` in a P4 source) is recorded as outside P1's scope. A component named only in prose, or an external component without covered cost, is an open premise on the claims that use it. A runnable component called external inside a covered witness is an allegation that only a magic-X run can substantiate.
- Findings. Each finding has a subject, a kind (missing assumption, witness violation, accounting, explanation), an exact evidence anchor (card text, the text of a stage M check, a frozen definition or a frozen statement), the dependency it claims, the obligations it affects, and a way to settle it: runnable `verify(witness)` code run by the host against the recorded sources (`hybrid/math/findings.py`), a SymPy derivation, or a citation with a quoted sentence of the record. The id is a hash of the subject and the claims it names, so the same gap raised twice, or in a later revision, is one finding.
- Dispositions. An anchor that is not exact text, a code-testable finding without code, or code that does not run is `unsupported`; a verification that says no is `rejected`; both are recorded and block nothing. A confirmed missing assumption is `substantiated` and becomes an open premise on the claims it names, with its measurement plan when it gives one (only then can the claim advance as schedulable). A confirmed witness violation fails those claims. A verification the executor could not finish is `unknown` and blocks as an open premise, since a stopped job is not a result. An explanatory finding aimed only at profile slots is `out_of_scope`.
- Magic-X runs on every substantiated doing-work dependency that comes with X code, original or newly found; without X code a substantiated dependency stays open and blocks. The exemption for an X that is obtainable at a cost included in P1 (`cost_in_p1`) applies only to a premise implemented by a component the verified P1 covers.
- Host findings. Stage C itself raises a P5 sample entry without a defined target-rule class, and a profile frozen without an architecture contract while runnable mechanism components lack covered cost (P1's coverage cannot be verified).

Where the fixer's own premises are settled (decision FIX-6, 2026-10-14). Every premise the statement fixer writes states its subject, and the host classifies it by subject and testability:

- `frozen_definitions`: a claim about the card's own code, equations or frozen definitions, for example that the cost reference executes the card's equations with a definition's product spellings and retained buffers. The host also puts in this class any premise whose text or measurement plan names one of the record's frozen definitions, whatever label it carries. Such a premise is settled at stage M only. It needs an obligation with no claim id that discharges it, frozen with the statements and checked in M; a profile slot or card claim that lists it does not count, because its statement is compared with its own claim and a count check would pass as equation conformance. For the cost reference the discharging statement is a `conformance` statement: the fixer freezes the card's update and readout as `expected(params, sizes, tokens)`, and after the prover rounds the host runs the reference that held for P1 and those equations on the same parameters over repeated, distinct and random token streams, comparing every retained array and the output after every token (`spec_checks.reference_conformance`). Equal operation and state counts do not settle it. A premise with no discharger is refused at freeze (`refused`); a refused premise, or one whose discharging check did not hold, stays open, never advances and is never deferred to test design, so the claims that use it are not shown. A citation cannot settle it.
- `mechanism` and `world` (a capacity the mechanism relies on; external data, hardware, a published result): unchanged. Magic-X runs on mechanism premises at stage C, a code-testable premise with a measurement plan may still be deferred to test design, and a cited premise is verified at stage C.

Reasoning: in the 2026-10-14 calibration (card k1, run 1) the fixer created A1, "the reference executes the card's equations with D1's product spellings and D2's retained buffers", created no obligation to discharge it, and its plan was deferred to test design, which calibration never runs. The three complexity claims and P3 stayed conditional on it while P1 itself held (`results/phase4-20261013T073748Z/k1-calib-diag/REPORT.md`, section 3). Under this rule A1 is refused at freeze unless the fixer adds the conformance statement, and the recorded P1 reference meets such a statement on the card's equations (`hybrid/tests/research/test_fix6_settlement.py`).

The conditional-to-not_shown cell map of calibration stays strict: a slot with any open finding is not shown.

---

## 5. The challenge stage (step 6)

The challenger is a different agent in a different model family from both the proposer and the prover (routing in section 16). It sees the card, the frozen statements, the proofs and the checker outputs. The checker outputs go complete, in batches of at most 60000 characters with one call per batch and the replies merged; no source is cut (FIX-4: the k1 smoke prompt was sliced at 60000 characters inside one source and left another check out). It runs four checks; any one fails the card.

### 5.1 Magic-X test

For each assumption on the ledger, the challenger writes "baseline + X": a plain baseline (n-gram, LSTM or a small transformer, chosen by the harness from the assumption's kind) given the assumption as an extra input or module. Example: if the card assumes an oracle that segments input into constituents, the baseline gets the same oracle. The harness (`hybrid/math/magic_x.py`) runs it on the card's own P2 and P4 tasks and on the claimed advantage. If baseline + X alone reaches the claimed advantage within the card's tolerance, the advantage is attributed to X and the card fails, unless X is `proven` or `measured` to be obtainable from the data at a cost that is included in P1.

The assumption must be runnable code with a measured cost: an assumption the harness cannot implement is `open`, and section 4.3 then blocks the card.

FIX-4 (2026-10-13): the test also runs on dependencies the challenger finds, once a finding is substantiated (section 4.5), not only on the fixer's original premises; a substantiated dependency without X code stays open. The P1 exemption is granted only through a verified cost link (section 4.5, components), never by default.

### 5.2 Conservation-law checks

`hybrid/math/conservation.py`, a reusable library with one function per law, each with unit tests against published numbers:

| Law | Function | Source |
|---|---|---|
| Storage (pigeonhole): a state of S bits cannot distinguish more than 2^S contexts | `storage_bound(state_bits, required_bits)` | counting argument |
| Copy lower bound for fixed-state models: error above 1 − 2^S / D^L on uniform strings | `copy_error_lower_bound(state_bits, D, L)` | Jelassi et al., Theorem 2.7 ([arXiv 2402.01032](https://arxiv.org/abs/2402.01032v1)); already used in `IMPROVEMENT-PLAN.md` B.5 |
| Sample information: identifying one of |H| hypotheses needs at least log2 |H| bits, so m ≥ (log2 |H| − prior bits) / bits per example | `sample_bound(log2_H, prior_bits, bits_per_example)` | counting and Fano-style argument |
| One-way communication: a single left-to-right pass with an s-bit state solves INDEX on n items only if s is Ω(n) | `streaming_index_bound(n, state_bits)` | Kremer, Nisan and Ron 1999 ([doi 10.1007/s000370050018](https://doi.org/10.1007/s000370050018)); associative recall with keys before queries reduces to INDEX |
| Input reading: any model that uses every token does at least n work | `read_bound(n)` | trivial |

A card whose claimed performance violates any of these at its stated sizes fails with the law named. These checks need no model judgment: the inputs are the card's numbers from P1, P2 and P4.

Provenance (FIX-4, 2026-10-13). Each number the challenger passes to the library is quoted from the card, a frozen definition or a frozen statement, or is the value of a derivation (an expression in + - * / **, ceiling, floor, log and sqrt) whose literals and inputs are all quoted; the host re-evaluates every derivation. Invented or altered numbers keep the check from running. In the smoke, card k1's challenge quoted precision from definition D5 and state formulas from frozen statements, and the old card-text-only rule skipped the check.

### 5.3 "X must be runnable code with a measured cost"

Every component the card relies on (an oracle, a parser, a retrieval store, a search procedure, a prior) must exist as code in the card's reference implementation and must appear in P1's counted cost. A component named only in prose is an `open` assumption.

Implementation and cost are separate evidence (FIX-4, 2026-10-13). A component can be implemented in a P2 or P4 witness and counted by P1 when both held under the same frozen architecture contract (section 4.4); a missing cost link is an accounting finding on P1 and leaves the executed capability result standing. External components without covered cost stay open premises (section 4.5).

### 5.4 Statement match

A third agent, in a family different from both the statement fixer and the prover, gets only the card's natural-language claims and the frozen formal statements (no proofs). It translates each formal statement back into plain language without seeing the claim it came from, then a fixed rubric compares the two: quantifiers, domains and ranges, constants, exponents, hypotheses, and direction of inequalities. For statements over finite domains, the harness also evaluates the formal predicate and a Python predicate written from the natural-language claim on 1,000 random instances; any disagreement is a mismatch. A mismatch fails the card with `statement_mismatch`.

This check exists because a Lean proof that compiles can prove a different statement from the one claimed: Bastounis, Circelli and Hansen document the OpenAI Navier-Stokes Lean formalisation proving weaker or different estimates than the natural-language paper, for example m+5 derivatives in Lean against m+4 claimed ([arXiv 2610.08144](https://arxiv.org/html/2610.08144); [New Scientist report](https://www.newscientist.com/article/2592824-openai-mistranslated-mathematics-into-code-for-its-navier-stokes-proof/)).

Mechanism definitions, anchored objections and the cross-check (decision FIX-7, 2026-10-14). A definition that faithfully restates the card's own state, update, readout or learning rule defines the object the claims are about and is part of the claim, not a hidden premise; the fixer labels it kind `mechanism`. Accuracy, capacity, cost, data and world assertions in a definition stay substantive, as does a definition that changes the card's mechanism. The comparison, never the blind back-translation, reads the card's mechanism with anchors and the definitions' provenance. Every mismatch objection is anchored: a statement objection quotes the claim text and the formal statement or definition text that differ, and a definition objection quotes the clause and either the card text it contradicts or what it asserts beyond the card. The host discards an objection whose quotes are not exact text, and one that only restates the card's state, update, readout or learning rule. A P4 statement is compared with the selected solvable case it instantiates: P4 selects one size at or below the ceiling, and P2 is the slot held to every listed size. The local matcher stays the third-family judge, and a gateway Sonnet judge cross-checks every anchored objection in a separately recorded call (role `statement_crosscheck`, which may share the prover's family but not the matcher's or the fixer's). Both agree: the statement is mismatched and its claim blocked, as before. They disagree: the claim is `unresolved`, keeps the status its checks give, and stage C lists `match_unresolved:<claim>`; a card with unresolved claims and no failure is undetermined. Calibration scores an unresolved cell from its checks and records it, and the run is not reproduced until the dispute is settled. The predicate check and the precedence of `blocked` over `failed` are unchanged. Reasoning: in the 2026-10-14 calibration (card k3, run 1) the matcher called D1, which restates the card's sixteen states and its equations, a substantive premise, and read three P4 ceilings ("L at most 4") as ranges. Sixteen statements were mismatched and all seven scored cells hidden, though neither finding quoted anything that differed (`results/phase4-20261013T073748Z/match-diag/REPORT.md`; `hybrid/tests/research/test_fix7_match.py`).

Statement tiebreak (decision XC, FIX-17, 2026-10-15). A dispute (the matcher upheld an anchored objection and the cross-check rejected it) blocks the claim until it is settled. A third judge from a family other than both, role `statement_tiebreak` (gateway ChatGPT `chatgpt/gpt-6.1-sol`, effort medium, no fallback), reads both readings and decides: upheld makes the statement mismatched, rejected lets the objection block nothing. A failed tiebreak call settles nothing, so the claim stays unresolved and calibration scores it not shown. Every dispute is recorded in the stage C verdict (`disputes`: both readings, the tiebreak's and the outcome) and in a `statement_dispute` event.

### 5.5 The generator never edits checks

- Check code lives in `hybrid/math/`, `hybrid/bench/` and the tools image. Model roles return JSON only; no role has tools or file access (the llxprt backend already restricts tools to `web_tools` on calls that set `allow_web`, `models.toml`).
- Each M and C verdict records the sha256 of every staged check file, as the falsifier provenance does today (`falsifier.code_hashes`). A test fails if any hash differs from the committed manifest `hybrid/math/MANIFEST.json`.
- Obligation lists are fixed by the statement fixer. A revision that removes or edits an obligation, a statement or an assumption edge is rejected before any check runs (`v_revise.diff_problems`).

---

## 6. Generation (step 1)

### 6.1 Arms

Each arm is a prompt section plus a required output structure. Arms run as measured arms: every card records its arm, and `inspect` reports per arm the share reaching each stage, the share that is radical (1.3), cells newly filled, and calls and tokens per card.

| Arm | Required output, checked before R |
|---|---|
| A1 Cross-field mapping | A map from named objects in the source field to objects in the mechanism, one equation of the source field and its image under the map, and one prediction derived through the map that P2 or P4 can test. A card whose image equation does not appear in its mechanism fails the arm check. |
| A2 Break an assumption of a known limit | Names a result from `hybrid/config/limits.toml` (Jelassi copy bound, Hahn 2020, Merrill and Sabharwal TC0, the finite-state argument of Chomsky 1956, INDEX, others added over time), quotes the assumption broken, and states how the break is realized and what it costs in P1 terms. |
| A3 Invert a transformer commitment | Picks one entry of a fixed list (dense real vectors, subword tokens, next-token objective, knowledge in weights, end-to-end backprop, fixed compute per token, flat context, softmax retrieval) and builds on its inverse. |
| A4 Distant random concepts | Two concepts drawn from a concept list unrelated to sequence modelling; the card must still pass the A1 structure (map, equation, prediction). Random text earns nothing on its own (`research-llm-creativity.md` section 5 recommends this rule). |
| A5 Base or less post-trained model | Drafts from a base model, then a separate instruction-tuned formatter turns the draft into a card without adding mechanism (the formatter's diff is checked: new equations are rejected). Availability of a base model on LM Studio is UNVERIFIED; decision D4. |

### 6.2 Quality-diversity archive

MAP-Elites (Mouret and Clune 2015, [arXiv 1504.04909](https://arxiv.org/abs/1504.04909v1)) over five axes with controlled vocabularies in `hybrid/config/archive.toml`:

| Axis | Values (first version) |
|---|---|
| Representation substrate | dense real vector; sparse or binary hypervector; symbolic graph or term; program; probability distribution held as samples or particles; discrete automaton or grammar; geometric or topological object; continuous physical dynamics |
| Learning rule | gradient descent with backprop; local or Hebbian; Bayesian posterior update; search or program induction; compression or minimum description length; closed-form algebraic; evolutionary; none (fixed) |
| Unit | byte or character; subword token; morpheme; word; chunk or phrase; event or sentence; learned segment |
| Objective | next-unit prediction; masked or denoising; contrastive; description length; latent prediction; constraint satisfaction or consistency; reward or prediction error |
| Where structure comes from | learned from the data; built-in prior with declared bits; search at inference; external store; interaction |

Rules:

- The classifier role assigns the cell from equations and code. A deterministic consistency check overrides obvious contradictions (code calling torch autograd cannot be in a "none" or "Hebbian" learning-rule cell).
- Cells matching an excluded-family reference tuple are closed.
- Quality of an elite: furthest stage reached, then the number of profile obligations at tier `lean` or `sympy_exact`, then P4 example count (fewer is better).
- Parent selection for mutate and cross: probability proportional to 1 / (1 + visits) of the cell, plus a bonus for each empty neighbouring cell (one axis different). An empty cell filled by a card that passes R gets a reward recorded for its arm.
- Per-axis coverage and pairwise coverage are reported, because 8 × 8 × 7 × 7 × 5 = 15,680 full cells is too many to fill and most combinations are meaningless.

FunSearch and AlphaEvolve are the precedent for an LLM proposer paired with an executable evaluator and a persistent population ([Nature 2023](https://www.nature.com/articles/s41586-023-06924-6), [arXiv 2506.13131](https://arxiv.org/abs/2506.13131)). Their evaluators score programs; ours scores cards through M and C first, so the population is filtered by proofs and executed constructions before any training.

---

## 7. Refinement dialogue (step 2)

- Entry: the S1 salad critic's fatal check and the toy example run, unchanged. A fatal finding ends the card here.
- Rounds: a challenger agent (other family from the proposer) asks at most 5 questions per round; the proposer answers by revising the card. At most 3 rounds; the dialogue ends early when the challenger returns no questions.
- Questions must target one of: an undefined object, an unstated assumption, a claim without a test, a mechanism step whose cost is not stated, or a commitment that looks like decoration (R1). Free-form critique is not accepted by the schema.
- Output: the revised card and the assumption list (section 3.1), each assumption tagged `added_by: dialogue`.
- Card growth limit: the parsimony counter already in `common.parsimony` must not grow by more than 25% over the dialogue; the first live run showed revision answered with new helper objects (`ORCH-STATUS.md`, first live run).

## 8. Prior art, own history, refine or drop (steps 3 and 4)

1. Literature: existing `classify_prior_art` and the local arXiv index, unchanged.
2. Own history: outcome-store query (section 9) by exact signature and by nearest neighbours (Jaccard on signature tokens, TF-IDF on core idea).
3. Reduction critic and executed equivalence check: unchanged, plus the family exclusion of 1.2.
4. Difference statement (schema in 3.1), written by the proposer, checked by the prior-art role: at least two closest prior items with receipts or outcome ids; when the signature matched a dead idea, `cause_ids_addressed` must list the cause, and `changed_equation` must point at the equation the cause names.
5. Refine or drop: a card that fails the difference check gets one send-back carrying the specific required difference ("closest prior X used Y; state what replaces Y and where"). The send-back costs a slot from the lineage budget. A second failure drops the card with `no_difference`.

## 9. Outcome store (step 11)

- File: `results/outcomes/outcomes.sqlite` plus an append-only `outcomes.jsonl` with the same records (so the store can be rebuilt). Written only by the orchestrator at a card's terminal verdict.
- Record: card id, run, lineage id, arm, archive cell, mechanism signature, difference statement, assumption ledger, terminal stage, kill code, kill detail, evidence pointers (artifact ids, job ids), cost.
- Mechanism signature: written by the statement fixer as a skeleton over a fixed operator vocabulary (outer-product write, similarity read, softmax read, argmax, merge of equivalence classes, stack push and pop, sampling step, posterior update, program enumeration, and so on), plus the five commitments. Normalization (`outcomes.signature`) sorts and deduplicates operators, drops dimension names and hashes the result. The proposer never writes it.
- Kill codes, an enumeration: `reduces_to_known:<family>`, `excluded_family:<family>`, `no_difference`, `not_radical:<R1|R2|R3>`, `obligation_failed:<P1..P5>`, `unproven:<assumption id>`, `magic_x:<assumption id>`, `conservation:<law>`, `statement_mismatch`, `stall`, `budget`, `falsified:<task>`, `predictions_missed`. No free-text-only kill reason is accepted.
- Backfill: import the 241 existing card directories under `runs/*/cards/` read-only, with their S2 reductions mapped to kill codes; signatures for these are written by the statement fixer in a one-off batch job.
- Use: G receives the recorded causes for the target cell and its neighbours; P enforces R3.

## 10. Budgets and stall detection (step 8)

Per idea (defaults proposed, all in `run.toml` under `[budgets.idea]`):

| Loop | Cap |
|---|---|
| Refinement dialogue | 3 rounds |
| Prior-art send-back | 1 |
| Prover attempts per obligation | 4 |
| Lean wall time per attempt | 300 s; memory 6 GB (to be set from the Phase 1 measurement) |
| Z3 timeout per query | 60 s |
| Math revision rounds (V) | 3 |
| Model calls per idea, all stages | 80 |

Per lineage (root idea plus every mutate, escape and repair child): at most 4 ideas and 200 model calls. A child inherits the lineage id; a card whose signature matches a lineage at its cap is dropped at G with `budget`.

Stall: after each V round, the set of failing obligation ids and the hash of each counterexample are compared with the previous round. Same set and same counterexamples ends the loop with `stall`. A repair that changes anything other than what the failure names is rejected without running checks.

## 11. Test design and the test ladder (steps 9 and 10)

### 11.1 Fixed, calibrated ladder

| Rung | Content | Baselines | Calibration source |
|---|---|---|---|
| 0 | The M-stage constructions (already executed) | none | section 4.4 |
| 1 | The P2 tasks learned from data at small scale: copy, MQAR, Dyck-2, two-hop composition, plus a nonce-word morphology task and a synthetic fast-mapping task | n-gram, LSTM, two-layer transformer | `IMPROVEMENT-PLAN.md` B.2 to B.5: generator determinism, oracle self-score, answer-only loss, criteria 1, 2, 4, 6, 7 and 9 reused as acceptance |
| 2 | Child-scale probes: train on 1M and then 10M words of the BabyLM strict-small corpus; Zorro, a BLiMP subset, an EWoK subset, entity tracking | BabyBERTa-size model, small GPT-2 | published BabyLM numbers as sanity checks (`research-radical-substrates.md` section 7) |
| 3 | BabyLM 10M full evaluation | GPT-2 small, GPT-BERT | same |
| 4 | 50M and 100M words | same | requires the compute decision D6 |

The ladder is calibrated once, before any candidate result counts: each baseline must reproduce its published ordering on rungs 1 and 2. Measured rates set the run sizes: the default transformer job ran at about 0.063 s per step on one thread (`IMPROVEMENT-PLAN.md` B.6), and the substrates report extrapolates GPT-2 small on 100M words seen to about 58 days on one thread at that rate, so rungs 3 and 4 need the MPS throughput measurement before they are sized.

### 11.2 Card-specific tests

The falsifier designer and its validation gates (P0-1 to P1-8, provenance hashes, builder contract) stay as they are. New: the card states numerical predictions (schema `Prediction`) at T, before any run, hashed into the ledger; S6 compares results against them and records `predictions_missed` when a prediction falls outside its tolerance.

### 11.3 Order: cheapest likely kill first

Each candidate test gets an estimated cost (executor seconds from measured rates) and a kill probability (the challenger's estimate, later replaced by the observed kill rate of that test type in the outcome store). Tests run in decreasing order of kill probability per unit cost; the first kill stops the card.

---

## 12. Calibrating the math stage on known models

Four cards describe known models in neutral notation, with symbols renamed, no family names, no references and equations rewritten (attention as "weighted retrieval over stored pairs with exponential scores", and so on). They run through M and C only, under the plumb operator-override banner, because P would recognize them.

| Card | Model | P1 cost (expected formula) | P2 expected | P3 expected limit | Expected profile verdict |
|---|---|---|---|---|---|
| K1 | Transformer, 2 layers | total Θ(n² d); state 2·L·d·n | copy, recall, composition pass at listed sizes; nesting passes for bounded depth with layers growing with depth (RASP Dyck-k construction) | Hahn 2020 (no robust Dyck-2 or parity at fixed depth as n grows); TC0 bound | passes all five obligations; fails the compute target of 1.1 |
| K2 | LSTM | total Θ(n d²); state Θ(d) | copy for L·log2 D below state bits; recall for K·log2 V below state bits; nesting with counters to bounded depth; composition passes | Jelassi bound for copy beyond state; finite precision | passes all five obligations |
| K3 | HMM with k states | total Θ(n k²); state k probabilities | nesting fails beyond depth log2 k; composition fails beyond table sizes that fit in k states | finite-state: Dyck-2 needs at least 2^D states at depth D | fails P2 on nesting and composition at listed sizes chosen above its capacity, with those reasons |
| K4 | n-gram, order c | total Θ(n); table V^c | copy fails beyond c; recall fails beyond window; nesting and composition fail | window c | fails P2 on nesting and composition (and copy and recall) |

Acceptance: every expected cell above is reproduced (16 obligation verdicts on P2 plus P1, P3, P5 for each card), the failure reasons for K3 and K4 name nesting and composition, and three independent runs (different seeds, model sampling) agree on every verdict. A disagreement fixes the stage, not the expected table, unless the expected entry is shown wrong with a citation.

The P4 learnability results for K1 and K2 also become the reference example counts that later cards are compared against.

## 13. The planted-card test

### 13.1 Design

A separate agent writes the planted cards from specifications, and the specifications and expected outcomes are sealed (sha256 committed in the ledger) before the run. Cards enter the pipeline mixed and stripped of any origin field. A second set of 21 cards (3 per type, same specifications, different instances) is written and sealed at the same time and used only for the re-run after a fix, so a fix tuned to the first set is caught.

| Type | Count | Examples | Expected stop | Expected kill code |
|---|---|---|---|---|
| T1 Known method renamed | 10 | two each of linear attention, gated delta rule, a Mamba-style SSM, a transformer variant, an n-gram or cache model, written in new vocabulary | P | `reduces_to_known` or `excluded_family` |
| T2 Known-failed idea without a real change | 6 | an idea from the outcome store reworded, with a difference statement that changes something other than the recorded cause | P | `no_difference` or `not_radical:R3` |
| T3 Magic-X proof | 8 | valid proofs that rest on an oracle segmenter, a collision-free hash in O(1) bits, a learner told the hypothesis class, a perfect retrieval index with no counted cost | C | `magic_x` or `unproven` |
| T4 Subtle math error | 9 | three sign errors in an update or bound; three cost-exponent errors (claims O(n log n), nested loop gives n²); three capacity overruns (claims recall of K log2 V bits from a smaller state) | M or C | `obligation_failed:P1`, `obligation_failed:P2`, `conservation:storage`, or a Lean or Z3 refutation |
| T5 Valid known mechanism | 6 | the four calibration models in fresh notation, a stack-augmented RNN, a CKY chart parser | passes M and C; flagged known at P | `reduces_to_known` at P, with M and C both `pass` (M and C run under override for these cards) |
| T6 Trivial variant of a killed idea | 6 | surface changes to cards killed in the planted run's own earlier batch or in the backfilled store | P | `no_difference` via signature match |
| T7 Valid, not known (added by this plan) | 4 | mechanisms built by the planting agent, correct by construction and checked by hand, outside the excluded families | reaches T | none |

Total: 49 cards. T5 and T7 together give 10 valid cards for the false-kill rate.

### 13.2 Metrics and acceptance thresholds

| Metric | Definition | Threshold |
|---|---|---|
| Catch rate per type | share of T1, T2, T3, T4, T6 cards killed | T1 at least 9/10; T2 at least 5/6; T3 at least 7/8; T4 at least 8/9; T6 at least 5/6 |
| Stop-point accuracy | killed at the expected stage | at least 85% of caught cards |
| Reason accuracy | kill code matches the expected code family | at least 80% of caught cards |
| False-kill rate | valid cards (T5, T7) killed at M or C | at most 1 of 10 |
| T7 survival | T7 cards reaching T | at least 3 of 4 |
| Loop termination | every card reaches a terminal verdict within its budgets | 49 of 49, zero budget overruns |
| Cost per card | model calls, tokens, executor seconds | median and maximum reported; maximum within the per-idea cap |
| Repeatability | whole planted run executed twice | verdict agreement at least 90% |
| Check integrity | hash audit of check files across the run | zero differences |

Failure on any threshold: fix, then re-run on the held-back set of 21 plus the failing type's original cards. Thresholds on the held-back set are the same rates, rounded down (for example at least 2 of 3 per caught type, at most 0 of 6 valid cards killed).

## 14. Live run of about 30 ideas

30 ideas, 6 per arm, run after the planted test passes.

| Criterion | Threshold |
|---|---|
| Radical by definition (R1 to R3) among ideas that leave R | at least 40% |
| Radical among ideas that enter M | 100% (enforced by the gate; checked as an audit) |
| Repeat rate (reduces to an excluded or known family, or matches an outcome signature without addressing its cause) | at most 30%, against 62% (122 of 196) in the earlier runs |
| Specific kill reasons | 100% of kills carry a kill code other than `budget`, and `budget` kills at most 10% |
| Loop termination | 100% |
| Archive | at least 5 cells filled that were empty before the run |
| Check integrity | zero hash differences |

The number of survivors has no threshold: zero survivors with specific kill reasons is a valid outcome and is reported as such.

## 15. The six-year-old question set

Built late, only when a survivor reaches rung 2, and used only as the held-out test for survivors. It covers abilities existing benchmarks miss, in text form:

1. a word defined once in one sentence, used correctly later in a different context (delayed fast mapping);
2. novel combinations of known words and relations;
3. everyday why and what-if questions, answered by generation.

Overlap check before any item is kept: each candidate item is tagged with an ability and a format, and compared against tagged samples of BLiMP (grammatical minimal pairs, [TACL 2020](https://aclanthology.org/2020.tacl-1.25/)), Zorro (BLiMP-style with child-directed vocabulary, [CoNLL 2021](https://aclanthology.org/2021.conll-1.49/)), EWoK (world-knowledge plausibility, [arXiv 2405.09605](https://arxiv.org/abs/2405.09605)), DevBench (multimodal lexical, syntactic and semantic tasks against child response patterns, [arXiv 2406.10215](https://arxiv.org/abs/2406.10215)), and the BabyLM 2025 additions (entity tracking, WUG, COMPS; [BabyLM 2025 findings](https://aclanthology.org/2025.babylm-main.28/)) plus MEWL ([ICML 2023](https://proceedings.mlr.press/v202/jiang23i.html)). An item whose ability and format already appear in one of these is dropped; a near-duplicate test by n-gram and embedding similarity runs as well. The overlap report is part of the set's documentation.

Other rules from `research-radical-substrates.md` section 6.3: every word appears at least twice in the training corpus; 300 to 500 items; generated answers judged by two raters blind to the system. The set is stored with its hash, never shown to any generator or designer role, and read only by the evaluator. Until it is checked against child norms, results on it are reported as engineering measures.

## 16. Model routing

Routes available today (`hybrid/config/models.toml`, project memory):

- Claude family through the gateway Anthropic route (Opus, Sonnet, Haiku); the gateway maps `effort` to adaptive thinking on Claude 5.5 ids.
- GPT family through Sign in with ChatGPT on the gateway (`chatgpt/gpt-6.1-sol`), paced at 1 concurrent request, 10 s spacing, 60 requests per hour, effort capped at medium.
- Local Qwen (`qwen3.6-35b-a3b-mlx`) on LM Studio with a 131,072-token context and concurrency 1.

Three families, five roles that must be separated. Proposed assignment:

| Role | Family | Constraint it satisfies |
|---|---|---|
| Proposer | Claude (Opus); GPT for one arm's alternate drafts if quota allows | |
| Refinement challenger | the family the proposer is not | proposer ≠ challenger |
| Statement fixer | GPT | ≠ prover |
| Prover | Claude (Opus for Lean, Sonnet for Z3 and SymPy programs) | |
| Disprover and challenger (C) | GPT | ≠ prover, ≠ proposer for Claude-drafted cards |
| Statement-match checker | local Qwen | ≠ statement fixer, ≠ prover |
| Classifier (archive cell, R1) | local Qwen | ≠ proposer |
| Falsifier designer | GPT, as today | ≠ builder (Claude) |

Quota arithmetic from the pacing limit: a card that reaches C uses about 1 statement-fixer call, 2 challenge calls and up to 3 dialogue calls on GPT, so 6 GPT calls; at 60 per hour that caps throughput at about 10 such cards per hour on the GPT route, before falsifier-designer calls. The local Qwen route at concurrency 1 is the other bottleneck for the statement match and classifier roles; its throughput is unmeasured and Phase 3 measures it. If GPT is paced out, the fallback for GPT roles is the local Qwen model, never Claude, so no card has its prover and checker in one family; the run records each fallback.

## 17. Phases

Phases are in dependency order. Phase 10 depends only on Phase 0 and runs alongside Phases 1 to 9. Each phase ends with its tests green under `.venv-hybrid/bin/python -m pytest hybrid/tests -q -p no:cacheprovider` and the Experiment 1 hash test passing.

| Phase | Work | Files likely touched | Tests | Exit criterion |
|---|---|---|---|---|
| 0. Hygiene and decisions | Free-space precheck, retention script, label audit; Andrew's decisions in section 18 | `hybrid/hygiene/{space,retention}.py` (new), `scripts/hygiene.py` (new), `scripts/podman-teardown.sh` (read only, reused), `hybrid/research/__main__.py`, `hybrid/executor/podman.py` | precheck refuses below threshold (fake `statvfs`); retention keeps newest N and never touches ledgers, cards or artifacts; every podman argv built by `policy.py` carries the owner label; no code path issues an unfiltered prune | precheck wired into `run`, `plumb` and image builds; decisions D1, D2, D3 recorded |
| 1. Tools image | `Containerfile.tools`: pinned Lean toolchain (a release after the aarch64 `leantar` fix, lean4 PR #13499), Mathlib pinned by commit with `lake exe cache get` at build, z3-solver, SymPy, NumPy; same user, policy and labels as the executor image; `image.tools.lock.json` | `hybrid/executor/{Containerfile.tools,requirements.tools.txt,image.tools.lock.json,lock.py}`, `hybrid/math/lean/{Statement.lean,lakefile.toml}` | smoke: a small lemma (for example `∀ n : ℕ, n + 0 = n` and one Mathlib lemma) proves and `#print axioms` lists only standard axioms; a proof using `sorry` is rejected via `sorryAx`; a proof using a user `axiom` is rejected; a `native_decide` proof is rejected; a prover file that restates `Claim` differently is rejected by the statement-hash check; a Z3 query returns `unsat` and a counterexample query returns a model; a SymPy identity holds; image built with `--network` only at build time, jobs run `--network=none` | smoke passes in Podman twice; image size and Mathlib cache size measured and recorded; disk precheck passed before and after build |
| 2. Math library | Conservation laws, cost verifier with op counter, construction harness for the P2 tasks, magic-X harness | `hybrid/math/{conservation,cost,constructions,magic_x,tiers}.py`, `hybrid/math/MANIFEST.json` | each conservation function against published numbers (Jelassi Theorem 2.7 values, INDEX bound); cost verifier on four reference implementations with known formulas, including one planted exponent error; construction harness generators deterministic and oracle self-score 1.0; magic-X harness on a toy where an oracle alone gives the advantage | all tests pass in the tools image; manifest hashes committed |
| 3. Math roles and ledger | Statement fixer, prover, challenger, statement-match roles; assumption ledger; tier reporting; replay | `stages/{m_math,c_challenge}.py`, `prompts/{statement_fixer,prover,challenger,statement_match}.md`, `schemas/{math_statement,assumption,math_verdict}.schema.json`, `models.toml` routes and role settings | fake-LLM tests: statement frozen before prover; prover cannot alter statement; open assumption blocks; tiers reported correctly; statement mismatch detected on a planted mistranslation (a strengthened exponent, as in the Navier-Stokes example); family separation per card enforced by a routing test | one live card from an earlier run passes through M and C under override, with every artifact recorded; Qwen throughput measured |
| 4. Profile and calibration | P1 to P5 obligations; anonymized K1 to K4 cards | `hybrid/math/profile.py`, `hybrid/fixtures/calibration/{k1,k2,k3,k4}.json`, `results/calibration-math-<date>/` | expected table of section 12 as tests over the stored result | section 12 acceptance met on three runs |
| 5. Outcome store | Store, signature normalization, kill codes, backfill of 241 cards | `hybrid/research/outcomes.py`, `schemas/outcome.schema.json`, `results/outcomes/` | signature invariance under renaming and dimension changes; distinct mechanisms get distinct signatures on a fixture set; rebuild from JSONL equals SQLite; backfill reads runs without writing (byte comparison) | backfill complete; query latency recorded |
| 6. Generation arms and archive | Five arms, classifier role, archive, closed cells | `stages/g_generate.py`, `hybrid/research/archive.py`, `prompts/{arm_cross_field,arm_limit_break,arm_invert,arm_random,arm_base}.md`, `hybrid/config/{archive,families,limits}.toml` | arm structure checks; cell assignment consistency rule; closed cells; parent selection favours empty neighbours (seeded test); per-arm metrics in `inspect` | 10 live drafts per arm generated and classified with no schema failure above 10% |
| 7. Refinement and difference | Dialogue stage, difference statement, send-back, family exclusion in S2 | `stages/{r_refine,s2_prior_art}.py`, `prompts/{refine_challenger,difference}.md`, `schemas/difference.schema.json` | 3-round cap; early stop; growth limit; send-back once; cause-id rule against a fixture store; reduction into an excluded family fails despite `LEARNING_RULE:` | fake-LLM line from G through P passes; replay of run-20261010T181300Z reductions gives the expected kill codes |
| 8. Budgets and stall | Per-idea, per-lineage caps; stall detector; repair diff check | `stages/{common,v_revise}.py`, `hybrid/research/lineage.py`, `run.toml` | each cap stops the loop with `budget`; identical failure set ends with `stall`; repair touching an unnamed obligation rejected; lineage cap across escape children | all loop paths terminate in tests with randomized fake failures (property test) |
| 9. Planted-card test | Planting specs, sealed answers, runner, report | `hybrid/fixtures/planted/`, `scripts/planted_run.py`, `results/planted-<date>/` | sealing and unsealing; origin stripping; metric computation on a scripted fake run | section 13.2 thresholds met |
| 10. Test ladder | Rung 1 tasks and calibration (from `IMPROVEMENT-PLAN.md` B and C), rung 2 data and probes, predictions schema, test ordering | `hybrid/bench/tasks/*`, `hybrid/bench/baselines/*`, `hybrid/bench/configs/ladder.toml`, `stages/t_design.py` | `IMPROVEMENT-PLAN.md` B.5 criteria 1, 2, 4, 6, 7, 9 as stored-result tests; rung 2 baseline sanity checks; ordering by kill probability per cost | rung 1 calibrated on three seeds; rung 2 baselines within the published ranges |
| 11. Live run | 30 ideas | `hybrid/config/run-discovery.toml`, `results/discovery-live-<date>/` | none new | section 14 criteria met, or a written failure analysis with kill codes |
| 12. Six-year-old set | Section 15, only when a survivor reaches rung 2 | `data/childqa/`, `hybrid/bench/childqa/` | overlap report; vocabulary filter; evaluator-only access | set frozen with hash and overlap report |

## 18. Decisions for Andrew

1. **D1 Compute target numbers.** Total cost exponent ceiling (proposed 1.5) and the state-growth rule (fixed or O(log n), corpus-scale stores allowed when declared). Blocks Phase 4.
2. **D2 S3 judges.** Remove the two feasibility judges from the gate entirely (proposed), or keep one as an advisory note in the dossier.
3. **D3 Disk thresholds.** Proposed: refuse runs below 20 GB free on XS1000; refuse image builds below 30 GB free on the internal disk or 20 GB inside the Podman VM's `/var`; keep the newest 3 `results/bench-tests-*`.
4. **D4 Base-model arm.** Is a base (not instruction-tuned) model available or wanted on LM Studio? Without one, A5 is dropped and its 6 live-run slots move to A1 and A2.
5. **D5 GPT route allocation.** Keep GPT for the statement fixer, the challenger and the falsifier designer (proposed), which leaves no GPT explorer, or accept a lower card rate.
6. **D6 Compute for rungs 3 and 4.** Measure MPS throughput and allow untrusted host training with scoring in the container, or cap the ladder at rung 2.
7. **D7 Planted-card authorship.** An agent writes the planted cards (proposed, from specs Andrew can read), or Andrew writes or approves the T3 and T7 sets himself, since those two decide the false-kill and magic-X numbers.
8. **D8 Excluded families list.** Confirm section 1.2, including whether Hopfield-style energy memories and exact-match caches are excluded as a category or only reduced case by case.

## 19. Risks and open questions

- **Statement fixer as a single point of failure.** If it writes a weak statement, every later check passes a weak claim. The statement-match role checks translation, but not whether the obligation was the right one to ask. Mitigation: the profile obligations P1 to P5 are templates with fixed shapes; the fixer fills parameters, not the shape.
- **Lean proving yield.** Model provers may fail to close many Mathlib proofs in the attempt budget, leaving cards at `executed` tier. The gate counts tiers but does not require `lean` for every obligation; whether that is strict enough is open until Phase 4 data exists.
- **Tool versions on aarch64.** z3-solver Linux aarch64 wheels from 4.15.5 are tagged `manylinux_2_38` (glibc 2.38), while the executor base is Debian bookworm (glibc 2.36) ([vera issue 701](https://github.com/aallan/vera/issues/701), [z3 issue 7232](https://github.com/Z3Prover/z3/issues/7232)). Options: base the tools image on Debian trixie, or pin z3-solver at 4.15.4 or earlier. Lean v4.30.0-rc1 bundled an x86 `leantar` in its Linux aarch64 build, which breaks `lake exe cache get`; fixed in lean4 PR #13499 and backported to v4.30.0 ([issue 13306](https://github.com/leanprover/lean4/issues/13306), [PR 13499](https://github.com/leanprover/lean4/pull/13499)). The current z3 wheel tag and the exact Lean release to pin are checked at build time.
- **Mathlib cache on Linux arm64.** The cache tool supports `aarch64-unknown-linux-musl` for `leantar` and downloads a static curl on Linux aarch64 (`Cache/IO.lean` in mathlib4). Whether cached `.olean` files load on Linux arm64 without a rebuild is UNVERIFIED in this pass; the Phase 1 smoke test settles it. The cache key covers the toolchain and manifests (`Cache/README.md`). The size of the unpacked Mathlib cache is UNVERIFIED; Phase 1 measures it against the VM's 75 GB free.
- **Magic-X harness coverage.** Some assumptions (for example "a prior over grammars") have no obvious baseline + X form. Those stay `open` and block the card, which may be too strict for ideas whose whole point is a prior. Measured in the live run by the share of `unproven` kills.
- **Archive axes are model-assigned.** The classifier can be wrong or gamed by vocabulary. The executed R1 ablation is the guard; its false-negative rate is unknown until the planted test.
- **Known families keep coming back under new vocabulary.** The history (`ORCH-STATUS.md`, graveyard section) shows explorers return to the linear-attention and delta-rule basin. The outcome store and the closed cells push against it; the live run's repeat rate measures whether it works.
- **Calibration leak.** Calibration cards may be recognized by the models despite renaming. M and C do not depend on recognition for their verdicts (they execute), but the prover could import known proofs. That is acceptable for calibration and is noted.
- **Toy learnability may not predict child-scale learning.** P4 shows a rule can find a construction; rung 2 and later decide whether that matters. The plan treats P4 as a necessary filter only.
- **Child-scale evaluation is mostly comprehension.** BabyLM metrics compare probabilities; the question set adds generation. Production is harder than comprehension for models and children (Hu, Tan, Feng and Frank 2025, cited in the substrates report; details UNVERIFIED there).
- **Quota.** GPT pacing caps M and C throughput (section 16). A paced-out GPT route moves roles to Qwen, whose reasoning quality for the statement-fixer role is untested.
- **Open question:** whether the S2 literature search can find prior art for ideas from distant fields (arm A4). OpenAlex and Semantic Scholar keys are not installed yet (`ORCH-STATUS.md`, literature API keys).

## 20. Disk hygiene

State measured on 2026-10-12: `/Volumes/XS1000` 57 GiB free (97% used, shared with other projects); internal disk 147 GiB free; the Podman machine `podman-machine-default` (applehv, 8 CPUs, 24 GB) stores images under `/var/home/core/.local/share/containers/storage` on a 100 GB virtual disk with 75 GB free; Podman images total 12.09 GB, of which the Bigbrane executor image is 1 GB. In the workspace: `runs/` 891 MB, `results/` 1.1 GB, `tmp/` 672 KB, one `results/bench-tests-*` directory.

Rules:

1. **Toolchain location.** Lean, Mathlib and the Mathlib build cache live only inside the tools image in Podman storage (VM disk on the internal drive). Jobs never bind-mount `/Volumes`; the executor policy already forbids it. Per-job Lean scratch goes to the job's tmpfs or its VM-local volume, which the executor removes after the job.
2. **Free-space precheck** (`hybrid/hygiene/space.py`, `require_free(path, min_gb)`): `run`, `plumb`, the planted runner and every image build call it first and exit with status 2, printing the measured and required numbers, when free space is below the thresholds of decision D3. Image builds also check the VM's `/var` through `podman machine ssh df`.
3. **Retention** (`scripts/hygiene.py`, dry run by default, `--apply` to act, run at the start of each `run` invocation in dry-run mode with a summary):
   - `tmp/pytest-*`: the existing `conftest.py` removes passing sessions; failed-session directories older than 3 days are deleted.
   - `results/bench-tests-*`: keep the newest 3.
   - Run artifacts: ledgers, cards, artifacts and reports are never deleted. Bulk outputs (`runs/*/falsifiers/*` data, `runs/*/bench_runs`, `runs/_bench_cache`) of runs older than 14 days are packed into one `tar.zst` per run, verified by listing, then the originals removed; ledger artifact hashes keep pointing at the packed members.
   - Lean caches: tools images are kept newest 2 by the `bigbrane.owner` label and image creation time; per-run Lean volumes are removed by the executor; no Lean cache is written to XS1000.
4. **Labels.** Every Podman image, build, container, volume and staging container carries `bigbrane.owner=bigbrane` (plus `bigbrane.run` and `bigbrane.job` where they apply), as `policy.py` does now; image builds pass `--label bigbrane.owner=bigbrane` and `--layers=false` as in `hybrid/executor/README.md`.
5. **Never prune unlabeled images.** Cleanup goes only through `scripts/podman-teardown.sh` and the new retention script, both label-filtered. No `podman system prune`, `podman image prune -a` or `rmi` without a `label=bigbrane.owner=bigbrane` filter. A test scans the code for podman calls and fails on any prune or removal without that filter. The 11 GB of other projects' images (llxprt sandbox images, rust, node) are never touched.

## 21. Citations

### 21.1 Transformer theory the profile rests on

| Work | Claim used | URL | Status |
|---|---|---|---|
| Yun, Bhojanapalli, Rawat, Reddi, Kumar (ICLR 2020), "Are Transformers universal approximators of sequence-to-sequence functions?" | universal approximation of continuous sequence-to-sequence functions on compact domains with positional encodings | [arXiv 1912.10077](https://arxiv.org/abs/1912.10077), [OpenReview PDF](https://openreview.net/pdf?id=ByxRM0Ntvr) | verified |
| Pérez, Barceló, Marinkovic, "Attention is Turing-Complete", JMLR 22(75), 2021 | Turing completeness with hard attention and arbitrary precision | [JMLR](https://www.jmlr.org/papers/v22/20-302.html) | verified |
| Hahn, "Theoretical Limitations of Self-Attention in Neural Sequence Models", TACL 8, 2020 | cannot model parity or Dyck-2 robustly unless layers or heads grow with input length | [ACL Anthology](https://aclanthology.org/2020.tacl-1.11/) | verified |
| Merrill and Sabharwal, "The Parallelism Tradeoff: Limitations of Log-Precision Transformers", TACL 2023 | log-precision transformers are simulated by constant-depth logspace-uniform threshold circuits (uniform TC0) | [arXiv 2207.00729](https://arxiv.org/abs/2207.00729) | verified (abstract and theorem via alphaXiv and DeepAI mirrors; TACL volume details not fetched) |
| Weiss, Goldberg, Yahav, "Thinking Like Transformers", ICML 2021 (RASP) | programs for histograms, sorting, Dyck-k; layer and head counts | [PMLR v139](https://proceedings.mlr.press/v139/weiss21a) | verified |
| Lindner et al., "Tracr: Compiled Transformers as a Laboratory for Interpretability", 2023 | compiles RASP programs into decoder-only transformer weights | [arXiv 2301.05062](https://arxiv.org/abs/2301.05062) | verified |
| Olsson et al., "In-context Learning and Induction Heads", 2022 | induction heads form with a sudden rise in in-context learning; causal in small attention-only models | [transformer-circuits.pub](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads), [arXiv 2209.11895](https://arxiv.org/abs/2209.11895) | verified |
| Zhang, Frei, Bartlett, "Trained Transformers Learn Linear Models In-Context", JMLR 25(49), 2024 | gradient flow on one linear self-attention layer reaches a global minimum that matches the best linear predictor | [JMLR](http://jmlr.org/papers/v25/23-1042.html) | verified |
| Bietti, Cabannes, Bouchacourt, Jégou, Bottou, "Birth of a Transformer: A Memory Viewpoint", NeurIPS 2023 | global bigrams learned first, induction head later; weight matrices as associative memories; induction head via three gradient steps | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/0561738a239a995c8cd2ef0e50cfa4fd-Paper-Conference.pdf), [arXiv 2306.00802](https://arxiv.org/abs/2306.00802) | verified |
| Nichani, Damian, Lee, "How Transformers Learn Causal Structure with Gradient Descent", ICML 2024 | gradient descent on a two-layer transformer encodes a latent causal graph; induction head as a special case | [PMLR v235](https://proceedings.mlr.press/v235/nichani24a.html), [arXiv 2402.14735](https://arxiv.org/abs/2402.14735) | verified |
| Zhang, Bengio, Hardt, Recht, Vinyals, "Understanding deep learning requires rethinking generalization", ICLR 2017 | networks fit random labels; parameter counts do not explain generalization | [Google Research](https://research.google/pubs/understanding-deep-learning-requires-rethinking-generalization/), [arXiv 1611.03530](https://arxiv.org/abs/1611.03530) | verified |
| Lotfi, Finzi, Kuang, Rudner, Goldblum, Wilson, "Non-Vacuous Generalization Bounds for Large Language Models", ICML 2024 | compression bounds on unbounded log-likelihood, non-vacuous up to 849M parameters | [PMLR v235](https://proceedings.mlr.press/v235/lotfi24a.html), [arXiv 2312.17173](https://arxiv.org/abs/2312.17173) | verified |

### 21.2 Other sources

| Work | URL | Status |
|---|---|---|
| FunSearch, Romera-Paredes et al., Nature 2023 | [Nature](https://www.nature.com/articles/s41586-023-06924-6) | verified |
| AlphaEvolve, 2025 white paper | [arXiv 2506.13131](https://arxiv.org/abs/2506.13131) | verified |
| MAP-Elites, Mouret and Clune 2015 | [arXiv 1504.04909](https://arxiv.org/abs/1504.04909v1) | verified |
| Bastounis, Circelli, Hansen, "Navier-Stokes lost in translation", October 2026 | [arXiv 2610.08144](https://arxiv.org/html/2610.08144), [author PDF](http://www.damtp.cam.ac.uk/research/afha/anders/NavierStokes_LostInTranslation_Final_Anders.pdf), [New Scientist](https://www.newscientist.com/article/2592824-openai-mistranslated-mathematics-into-code-for-its-navier-stokes-proof/) | verified |
| Lean 4 `#print axioms` and the standard axioms | [Axioms](https://lean-lang.org/doc/reference/latest/Axioms/), [Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/) | verified |
| Mathlib cache on Linux arm64 | [Cache/IO.lean](https://github.com/leanprover-community/mathlib4/blob/master/Cache/IO.lean) (aarch64 Linux supported for `leantar` and curl), [Cache/README.md](https://github.com/leanprover-community/mathlib4/blob/bc135ce2/Cache/README.md), [lean4 issue 13306](https://github.com/leanprover/lean4/issues/13306) | tool support verified; olean loading on Linux arm64 UNVERIFIED (Phase 1 smoke) |
| z3-solver aarch64 wheel tags | [PyPI](https://pypi.org/project/z3-solver/), [vera issue 701](https://github.com/aallan/vera/issues/701) | verified as reported by a third party; current release tag checked at build |
| Kremer, Nisan, Ron, one-round communication complexity, 1999 | [doi 10.1007/s000370050018](https://doi.org/10.1007/s000370050018) | paper verified; the INDEX lower bound statement is from memory of the paper's content, not fetched text: UNVERIFIED in detail |
| Jelassi et al., "Repeat After Me", ICML 2024 | [arXiv 2402.01032](https://arxiv.org/abs/2402.01032v1) | verified in `IMPROVEMENT-PLAN.md` on 2026-10-11 |
| Zoology, MQAR | [arXiv 2312.04927](https://arxiv.org/abs/2312.04927v1) | verified in `IMPROVEMENT-PLAN.md` |
| BLiMP, Warstadt et al., TACL 2020 | [ACL Anthology](https://aclanthology.org/2020.tacl-1.25/) | verified |
| Zorro / BabyBERTa, Huebner et al. 2021 | [ACL Anthology](https://aclanthology.org/2021.conll-1.49/) | from the substrates report reference list |
| EWoK, Ivanova et al. | [arXiv 2405.09605](https://arxiv.org/abs/2405.09605) | from the substrates report |
| DevBench, Tan et al. 2024 | [arXiv 2406.10215](https://arxiv.org/abs/2406.10215) | from the substrates report |
| BabyLM 2025 findings | [ACL Anthology](https://aclanthology.org/2025.babylm-main.28/) | from the substrates report |
| MEWL, Jiang et al. 2023 | [PMLR v202](https://proceedings.mlr.press/v202/jiang23i.html) | from the substrates report |
| Chomsky 1956, finite-state argument | cited in `research-radical-substrates.md` section 0 | from that report |
