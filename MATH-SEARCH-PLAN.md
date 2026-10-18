# Math search plan: generate mechanisms as typed programs, score them with host checks, and send only screened survivors to the card stages

> **Status 2026-10-18: on hold.** An independent review (results/math-search-review-20261018/REVIEW.md) found 4 blocking problems. The main one: S2's binding gate is the reduction critic's radicality judgment, which came back true on only 3 of 39 recorded calls, and nothing in this plan's fitness tracks it. The plan's own example program had already been killed at S2. Decisions T7b and T7c also record that on the protocol's task alphabets, textbook structures are optimal for every P2 skill. A probe of about 350 calls (results/s2-probe-20261018) runs first. It tests whether S2 can pass anything at all, and whether steered generation does better than the 40-card batch. This plan is revised or dropped on its result.


Draft 1, 2026-10-18, branch p4-diagnosis. Design only; nothing here is built yet.

**Verdict.** Stop asking explorers for prose theory cards. Generate candidates as small programs in a typed sequence-mechanism language (BBIR, section 3), score every program with host checks that cost no model call (section 4), and keep the best in a quality-diversity archive whose cells come from the program itself (section 6). A cheap model (Haiku or Sonnet through the gateway) mutates and crosses programs, prompted with elites from distant cells and with the evaluator's reason each parent failed or reduced (section 7). Only programs that pass every screen are rendered into theory cards and enter S0 admission, S1, R, S2, M and C unchanged (section 8). The first experiment (section 10) spends about the model-call budget the 40-card batch spent to reach its 28 S2 verdicts, and it stops the direction if the rendered cards repeat known work at the batch's rate.

## 1. What the evidence says, and what it rules out

| Record | Number | What follows for generation |
|---|---|---|
| results/discovery-live-20261017/SCORE.md | 0 of 28 cards that left R passed S2; 31 of 40 repeats (78%); 12 of 28 textbook transplants | Prose drafting from lenses, arms and operators lands on known mechanisms. |
| same, kill codes | not_radical:R2 12, reduces_to_known:other_known 9, not_radical:R1 7, excluded_family:ssm 5 | The two largest kill codes are a radicality judgment and a counted reduction to a known non-family mechanism (Willshaw memory, pushdown automaton, Kuramoto, RotatE, CHP stabilizers). A screen built only from families.toml would have missed most of them. |
| results/defect-recheck-20261017/RESULT.md | 11 of 11 defect-affected cards die again on fixed code, 121 calls | The gates are not the cause; the inputs are. |
| results/novelty-probe-20261017/RESULT.md | 8 of 8 published mechanisms reduced to the right source by exact executed checks; 2 of 2 disguised controls killed | S2 can be trusted as the final authority, so the new generator should aim at S2, never replace it. |
| same, KD4 | the one false kill came from a lifted one-hot state that grows with n | Any screen in the loop must refuse lifted-state simulations as reductions, exactly as S2 now does. |
| run-discovery.toml budget notes and the batch | 415 calls and 12.2M tokens for 28 S2 verdicts: 14.8 calls and 0.44M tokens per S2 verdict; recheck 121 calls for 11 cards, 11 per verdict | This is the comparison point for calls per S2 verdict. |
| Si et al. (arXiv 2409.04109) | LLM prose ideas collapse to duplicates as samples grow | Matches the batch. |

The direction (programs plus an automatic evaluator plus an archive) is decided. What has to be designed carefully is the representation, because AutoML-Zero (arXiv 2003.03384) shows a search over vector primitives rediscovers linear models first, and the fitness, because every quantity the loop rewards will be gamed.

## 2. The loop at a glance

```
            elites from distant cells + failure reasons
                         |
                 [LLM operator: mutate / cross]           1 cheap call -> 2 to 4 BBIR programs
                         |
   E0 parse, type, derive cost and state (static)          host, milliseconds, no model
   E1 constructions at growing sizes (hard mode)           host, seconds
   E2 hacking probes (held-out sizes, precision, ablation) host, seconds
   E3 family screen: structural normal form + numeric fit  host, tens of CPU seconds
   E4 learnability proxy (soft mode, one task, one seed)   host, at most 60 CPU s
                         |
                 [archive update: cell, elite rank]
                         |
         every K generations: render top elites -> theory card -> S0 admit, S1, R, S2, M, C (unchanged)
                         |
         S2 counted comparators and R2 reasons flow back into the library and the prompts
```

Each tier is a gate: a program that fails a tier is recorded with its kill reason and never runs the next tier. Cheapest likely kill first (plan section 11.3).

## 3. The representation: BBIR, a typed sequence-mechanism language

### 3.1 Why a data language and no Python

The LLM writes a BBIR program as JSON (an S-expression form is the alternative; JSON wins because schema validation, hashing and diffing already exist in hybrid/research/schemas). The host interprets it. Three reasons:

1. **Cost and state are derived.** Every operation has a declared cost formula in the widths of its operands, and every state object has a declared size in bits. Per-token cost and state size are computed from the program, so the non-quadratic constraint is checked by arithmetic over the types. The cost verifier in hybrid/math/cost.py has to trace executed lines because Python can hide work in built-ins (STATUS.md open problem 3); BBIR has no built-ins outside the primitive table, so that gap does not exist.
2. **Programs run on the host safely.** The program is data read by trusted code, so no LLM-written code executes. Every loop tier runs on the 16-CPU host in NumPy. The container is reserved for tiers that run generated Python (the M and C stages, and the bench). This matters because NumPy matrix math in the pinned tools container is 12 to 17 times slower than on the Mac (KD1).
3. **The search space is enumerable and diffable.** Two programs can be compared structurally, which the novelty screen needs (section 6.1).

### 3.2 Widths and n

A program declares named widths (`d`, `k`, `c`, ...) with integer values fixed for a run of the program. The sequence length n never appears in a type, with one exception: `Counter[log]`, which holds a position or count in ceil(log2 n) + 1 bits and is charged that much. So state is fixed in n by construction unless the program uses counters, and then it is O(log n), which is the D1 target. Capability at growing task size is measured by substituting widths, as decision C17 does (skill scale b with state at most n^0.5).

### 3.3 State types

| Type | Size in bits (charged) | Operations (cost per call) | Family it can fall into on its own |
|---|---|---|---|
| `Vec[d]` | 32 d | affine, elementwise nonlinearity, add (d or d*e) | ssm, rnn |
| `Mat[d,e]` | 32 d e | matvec (d e), rank-1 write (d e) | linear_attention, delta_rule |
| `Sym[V]` | ceil(log2 V) | equality, table index (1) | none |
| `Int[m]`, `Bool` | ceil(log2 m), 1 | arithmetic mod m, compare (1) | none |
| `Counter[log]` | ceil(log2 n) + 1 | inc, dec, reset, compare (1) | counter machine |
| `Stack[c, T]` | c * bits(T) + log2 c | push, pop, top, depth (bits(T)); overflow drops the bottom | pushdown automaton |
| `Queue[c, T]`, `Deque[c, T]` | same as Stack | push, pop at either end allowed by the type | queue automaton |
| `Set[u]` | u (bitset over a declared universe) | add, remove, member (1), union (u) | none |
| `Table[k, K, T]` | k * (bits(K) + bits(T)) | exact put with declared eviction (LRU or overwrite), exact get (k compares, or 1 with a declared hash) | exact_match_cache |
| `Perm[k]` | ceil(log2 k!) | compose with a transposition or a generator (k), apply to an index (1), invert (k) | permutation tracker |
| `Graph[k, L]` | k^2 log2(L+1) | add or drop a labelled edge (1), neighbors (k), bounded reach (k^2), union-find merge (k) | union-find and graph automata |

Every container has a declared capacity, and the overflow behaviour is part of its semantics. A program that wants more capacity must raise a width, and the width shows up in the state formula. Floats are stored at the declared precision (float32 by default) and rounded after every update, matching the architecture contract (`precision_bits`, hybrid/math/architecture.py).

### 3.4 Learned parameters and control

Parameters are separate from state and are trained:

- `Emb[V, d]`, `Lin[d, e]`, `MLP[d, h, e]`, `Score[d, a]` (scores for a actions).
- `choose(score, action_1, ..., action_a)`: a discrete control decision. At inference (hard mode) the arg-max action runs. In training (soft mode) the host uses a relaxation defined per primitive: a convex mixture of the resulting states for containers that have a mixture semantics (the soft stack and queue of Grefenstette et al. 2015 and Joulin and Mikolov 2015), and straight-through estimation otherwise. Every primitive must declare its soft rule; a primitive with none cannot sit under a `choose` that the trainer reaches.
- Keys, values and addresses may be computed from the input token, from parameters, or from the current state of another object. The last case is where BBIR leaves the S2 repair policy: contracts.UNRESOLVED_NOTE lists "keys or queries generated from the memory state, a discrete search over the state, or a new write rule" as changes no repair type covers. The language makes these easy to write on purpose (section 3.6).

Parameter shapes may depend on V and the widths only. Parameter tables may never be indexed by position, so a hand-set table cannot hold one row per test position.

### 3.5 Control flow

- No recursion and no unbounded loops. `repeat r { ... }` takes a width expression for r, and `while` takes a width bound. Every loop bound is therefore a formula, so cost stays derivable.
- One `update(state, token)` and one `readout(state) -> logits over V`. There is no access to earlier tokens except through state, which keeps the mechanism causal by construction.
- Total per-token cost is the sum of the costs of the executed operations, each a polynomial in the widths, so cost over n tokens is n times a width polynomial plus counter terms. A program can therefore never be quadratic in n. Quadratic behaviour can only come back through widths that the C17 substitution grows with the task size, and the C17 check is exactly where that is measured.

### 3.6 Why this primitive set leaves the excluded families without falling back into them

AutoML-Zero found linear regression and then gradient descent first because those are the shortest programs over its vector primitives that score well. BBIR changes what is short:

1. **The shortest high-scoring programs in BBIR are already known automata.** A stack with a finite control solves Dyck; a table solves recall; a queue solves copy. These are textbook structures, and S2 kills them as other_known (the novelty probe proved it on 8 of 8). So the loop must treat the textbook programs as the known floor: the family library (section 6.1) contains each of them as a BBIR program, and a candidate that normalizes to one of them is recorded as a rediscovery and closes its cell. This is a positive control: if the loop cannot rediscover the stack automaton from an empty archive within the first generations, its operator or evaluator is broken.
2. **Vector-only programs are cheap to identify and are excluded from parenthood.** A program whose only state is `Vec` and `Mat` is tagged by the static normal form (section 6.1) as ssm, rnn, linear_attention or delta_rule when its update matches the family form under the repair types. Such a program can stay in the archive as a recorded rediscovery, but its cell is closed and it is never chosen as a parent. This removes the gradient towards SSMs that a vector-heavy space has.
3. **Fitness rewards material coupling between objects of different kinds.** A candidate earns the coupling descriptor only if at least two state objects of different types are both material (E2 ablation) and at least one address, key or control score is computed from another object's state. Single-structure programs score at most as well as their textbook counterpart and are killed by E3 as reductions.
4. **The archive descriptor includes the coupling kind** (section 6.3), so the search is pushed to fill combinations the S2 repair policy cannot express.

The risk that remains is that a coupling of two textbook structures is itself textbook (two stacks make a Turing machine; a stack plus a table is a known memory-augmented automaton). E3 cannot know every such combination. That is why the R2 screen (section 6.2) and the S2 feedback into the library (section 6.4) exist, and why the experiment measures the leak rate between loop and S2.

### 3.7 Example (sketch)

A program that keeps a stack of symbols and a table whose key is the symbol on top of the stack, so the table's address comes from another object's state:

```json
{"widths": {"c": 16, "k": 32, "d": 16},
 "state": {"S": {"type": "Stack", "cap": "c", "elem": "Sym"}, "T": {"type": "Table", "slots": "k", "key": "Sym", "val": "Sym", "evict": "lru"}},
 "params": {"E": {"type": "Emb", "shape": ["V", "d"]}, "A": {"type": "Score", "shape": ["d", 3]}},
 "update": [{"choose": {"score": ["A", ["E", "x"]], "actions": [["push", "S", "x"], ["pop", "S"], ["put", "T", ["top", "S"], "x"]]}}],
 "readout": ["onehot", ["get", "T", ["top", "S"]]]}
```

The host derives: state bits = c log2 V + log2 c + 2k log2 V (the stack's 16 symbols and depth, the table's 32 key and value pairs); per-token cost = 3d for the scores, 1 for the chosen container operation, and up to 2k compares for the table writes and reads; state fixed in n. Whether this program reduces to a known automaton is E3's question; the example only shows the shape.

## 4. Fitness: tiers, order and gates

All tiers write one `eval` record per program (section 8.2). Each tier is a gate; the fitness used for elite ranking inside a cell is computed only for programs that pass every gate.

| Tier | Check | Gate | Where it runs | Reused code |
|---|---|---|---|---|
| E0 static | JSON schema, type check, width declarations, derived per-token cost formula and state formula, soft rule present for every reachable `choose` | Total cost exponent in n at most 1.5 with widths fixed (always true by section 3.5, checked anyway) and state fixed or O(log n) (D1, hybrid/math/target.py `asymptotic_exponent` on the derived formula) | host | target.py, cost.py `parse_formula` |
| E1 constructions | The program with its witness parameters (the LLM supplies a parameter setting per task, as a P2 witness) on copy, recall, dyck, composition and binding at three sizes from the hybrid/math/constructions.py generators, 50 instances each | At least one task at 0.99 exact accuracy at all three sizes with skill scale b > 0 under C17 width substitution | host | constructions.py generators, solvers and `oracle_self_score`, via an adapter from BBIR to the `init(layout)/update/predict` contract |
| E2 hacking probes | Section 5 | Every probe passes | host | architecture.py survey rules (as the model for the audit) |
| E3 family screen | Structural normal form against the library, then numeric fit of each library comparator at matched state budget | No structural reduction under repair types; no comparator fitted to agreement at or above 0.99 at the largest size | host | contracts.REPAIR_TYPES and families.toml forms as the rule source; the comparator library of section 6.1 |
| E4 learnability proxy | Soft-mode training from data on the program's best E1 task: m = 1000 examples, one seed, at most 60 CPU s, then hard-mode test on 100 fresh instances | Test accuracy at or above 0.9, and not `data_independent` under the two learnability.py data controls (training on no examples, and on the same streams with random answers) | host | learnability.py data controls and the constructions.py generators |

Fitness inside a cell, lexicographic: (number of E1 tasks passed with b > 0, E4 accuracy, minus the largest E3 comparator agreement, minus per-token cost at the reference widths, minus program size in nodes). Program size last, as a parsimony pressure (FunSearch and AlphaEvolve both prefer shorter programs at equal score).

Why this order: E0 kills malformed or over-budget programs for free; E1 is the main signal and costs seconds; E2 must run before E3 and E4 so that a hacked construction never earns the expensive tiers; E3 is the novelty gate, more expensive because it fits comparators; E4 is the most expensive and only matters for programs that are already novel and capable.

Why E4 exists at all: construction existence does not show learnability. The P4K record shows the transformer constructs composition and still fails to learn it at toy sizes (1 of 12 seeds at m = 4000), so a loop that rewarded constructions alone would select for mechanisms nobody can train. E4 is a proxy at one seed and a short budget; the full P4 at three seeds stays in stage M.

### 4.1 Entry points reused, and where each runs

Checked against the code on p4-diagnosis. The math modules are plain Python with relative imports: the container stages them as the package `bbmath` (hybrid/math/runner.py `check_library`, driver jobmain.py), and the host can import them as `hybrid.math` (the test fakes already run jobmain `main()` in the pytest process). They need the container only when the code they run was written by a model.

| Entry point | What the loop uses | Host or container | Why |
|---|---|---|---|
| hybrid/math/constructions.py `generate`, `solve`, `layout`, `oracle_self_score` | E1 task instances and answers | host | host-written generators; the BBIR interpreter is host code and the program is data |
| hybrid/math/constructions.py `load_mechanism` and the P2 audited construction | P2 at stage M on the compiled reference code | container (unchanged stage M) | runs generated Python |
| hybrid/math/cost.py `parse_formula`, `asymptotic_exponent`; hybrid/math/target.py | E0 exponent of the derived cost and state formulas | host | formulas only |
| hybrid/math/cost.py counting verifier | cross-check of the IR compiler in tests; P1 at stage M | host in tests (compiled code from our own compiler), container at stage M | at M it runs card code |
| hybrid/math/learnability.py data controls and `Result` shape | E4 controls and record format | host | the trainer is host code |
| hybrid/math/architecture.py survey rules | the model for the E2 width and precision audit | host | rules only |
| hybrid/research/stages/contracts.py `REPAIR_TYPES`, `UNRESOLVED_NOTE`; s2_prior_art.py `eligibility`, `counted_comparisons`, `EQUIVALENCE_TOL`; families.toml | E3 rewrite tags, the counted-family rule, library family ids | host | policy data and pure functions |
| S2 reduction check programs stored in runs/*/artifacts | harvested comparators (6.1 item 4) | container once per comparator, to reproduce the stored numbers before translation | they are model-written Python |
| hybrid/research/signature.py `card_signature` | outcome-store signature of a rendered card | host | deterministic reading of the equations |
| hybrid/research/stages/s0_explore.py `admit`; the injector pattern of planted.py and live_check.py; plumb.py | entry of rendered cards; M and C for S2 passers | host orchestrator; M and C jobs in the 4 executor slots | unchanged stages |
| `python -m hybrid.bench eval --candidate ... --config hybrid/config/bench_quick.toml`; hybrid/bench/fingerprint.py `attach`, `flags_for`, `nearest_baseline` | E5 descriptor on rendered cards | bench container (one slot per job) | trains the candidate with torch; `--local` is untrusted and marks results `trusted: false` |

## 5. Preventing fitness hacking

| Hack | How it would score | Defence (tier) |
|---|---|---|
| Lookup table sized to the test | A witness table with one entry per test instance or position | Parameter shapes depend only on V and widths (E0). E1 sizes are drawn by the host from a schedule the operator never sees, and E2 reruns at a held-out size 4 times the largest E1 size with widths substituted by C17; a witness that does not generalize fails. Tables indexed by position are refused at E0. |
| State that grows with n disguised as constant | A float used as an unbounded store (digits as a stack, the Siegelmann construction), or a counter that holds more than a count | Floats are rounded to the declared precision after every update. E2 adds a precision probe: rerun E1 with state rounded to 8 fewer mantissa bits; a mechanism whose accuracy collapses relies on precision it did not declare and fails. Counters are charged log n bits and may only be compared, incremented, decremented or reset, never used as an address into a parameter table. |
| Witness that only works below a container's capacity | Sizes chosen just below the stack or table capacity | E2 runs one size above the declared capacity and records the limit; the card renderer states that limit (the KD2 lesson: a table limit is a stated limit, never a hidden one). The b > 0 requirement uses C17 substitution, so capacity has to grow with the task through widths, which the state formula charges. |
| Lifted state (KD4) | A screen that maps the candidate to an ssm through a one-hot state over reachable configurations would kill everything | The structural normal form only maps a candidate to a family through size-preserving rewrites: the comparator's state bits must stay within a factor 4 of the candidate's at every width, and must not grow with n. A simulation that needs a larger lift is recorded as "simulation by lifting, not counted", the same rule S2 now applies. This rule also stops the opposite error (a candidate that hides a growing state) because state size is a type fact. |
| Decoration | An unused or inert state object to fill a descriptor | E2 materiality ablation: replace each state object's writes with no-ops (and, for structured objects, with a `Vec` of equal bits); a descriptor counts only objects whose ablation drops E1 accuracy by at least 0.2 on some passing task. |
| Witness hand-coding the answer through the readout | Readout that reads a parameter table keyed by the query alone | The readout may depend on parameters only through `Emb`, `Lin`, `MLP` or `Score` applied to state values; E2 runs each passing task with the in-context pairs permuted, and a readout that ignores the context fails. |
| A learner that ignores its data | A hand-set witness behind a trainer that changes nothing | E4 runs the two learnability.py data controls with the same trainer and recipe, as P4 does, and fails the program when both reach the threshold. |

The defences are fixed host rules. The operator prompt states them, and a program the host refuses gets the refusal text back as feedback, but no model judges them.

## 6. Novelty inside the loop, and its link to S2

### 6.1 The comparator library

S2 has no fixed comparator library today. The reduction critic declares comparators per card, and the host decides equivalence from the numbers its check program prints (`judge_comparisons`, EQUIVALENCE_TOL = 1e-6, eligibility by REPAIR_TYPES). The loop needs a library it can run with no model. It is built in BBIR, so the same interpreter runs the families and the candidates:

1. **Excluded families** from families.toml: diagonal and input-gated linear recurrence (ssm), Elman, GRU and LSTM cells (renamed RNN), linear attention with and without decay and normalizer, delta rule and gated delta rule, and a windowed softmax attention (the transformer family, run at a fixed window so it stays inside the language's cost rules).
2. **Known in-scope families** from families.toml: Hopfield and kernel memory, exact-match cache.
3. **Textbook structures** from the novelty probe and the batch: pushdown automaton, queue automaton, Minsky counter machine, n-gram and variable-order Markov, Viterbi-style max-product recursion, union-find, permutation tracking under transpositions (arXiv 2306.04470), Willshaw and Lernmatrix memory, Kuramoto phase coupling, RotatE-style rotation memory, and the CHP stabilizer update.
4. **Harvested S2 comparators.** Every counted comparator S2 has used (the `counted_comparisons` records and the reduction check programs stored as artifacts in runs/) is translated into BBIR once, with a check that the BBIR version reproduces the stored check's numbers on its stored inputs. A comparator that cannot be expressed in BBIR is recorded as "outside BBIR" and listed, since it is evidence the language is too narrow or the mechanism is outside the search space.

Each library entry carries its family id and free parameters.

**Structural screen.** The host rewrites a candidate to a normal form (inline, constant-fold, drop dead state, merge objects of the same type) and asks, for each library entry, whether the candidate equals the entry's form after a sequence of rewrites each tagged with one REPAIR_TYPES name, under the size-preserving rule of section 5. A match with every rewrite inside the policy is a structural reduction and fails E3, recorded with the family and the rewrite list. This is the same rule S2 counts by, applied by code with no critic. Rewrites outside the policy (the `other` type: a key from another object's state, a search over the state, a new write rule) are recorded as the candidate's distance to that family, as a list of named differences.

**Numeric screen.** For each library entry whose state can be matched within the factor-4 budget, the host fits the entry's free parameters to reproduce the candidate's hard-mode outputs on 200 random streams and on the E1 instances (soft-mode gradient fit with the E4 trainer, a fixed budget of 20 CPU s per comparator), then measures agreement on 200 held-out streams at the largest E1 size. Agreement at or above 0.99 fails E3 as a behavioural reduction. This is stricter than S2's exact equivalence check, on purpose: the loop should over-reject rather than send S2 repeats.

### 6.2 The R2 screen

The largest kill code in the batch (not_radical:R2, 12 of 28) is a model judgment: "would surprise a researcher who knows those families" (prompts/reduction_critic.md). No numeric screen reproduces it. So the last screen before rendering is one call of the existing reduction critic prompt, routed to Sonnet, on the rendered card with no literature search, asking only for `reduces_to`, `radical_enough` and `textbook_transplant`. A card the screen calls not radical or textbook goes back into the archive with the critic's reason as feedback, and is not rendered further. This screen gates rendering only; S2 still runs in full on every card that passes it, so the screen cannot pass a card S2 would kill. It costs one call per rendered candidate against 11 to 15 calls for a full trip to an S2 verdict.

### 6.3 Archive cells and the novelty descriptor

A loop cell is computed from the program and its evaluation, never from a model label:

| Axis | Values |
|---|---|
| Material state kinds | the sorted set of state types that passed the E2 ablation, capped to the two most material |
| Coupling | none; key or address from another object's state; control score from another object's state; search over a state object |
| Skill profile | the set of E1 tasks passed with b > 0 |
| Cost class | per-token cost constant in the reference widths, linear in one width, or a product of widths |

Cells whose material state kinds are only `Vec` and `Mat`, and cells that contain a structural reduction, are closed (they mirror families.toml `cells`). Two elites per cell, as archive.toml `elites_per_cell`.

The behaviour descriptor for novelty distance is the vector of E1 accuracies at every task and size plus the E3 agreement with each library entry. Parent selection weights cells by 1 / (1 + visits) plus a bonus for empty neighbouring cells (archive.toml `neighbour_bonus`), and within the archive a novelty term: mean distance to the 5 nearest archive descriptors (ELM, arXiv 2206.08896). The novelty term chooses parents; it never passes a gate.

The bench fingerprint (hybrid/bench/fingerprint.py: Jensen-Shannon and oracle-normalized distance to the five trained baselines, `behaves_like:<baseline>` flags) needs the program trained on the bench tasks and the bench baselines, which run in the bench container at 95 to 141 s per baseline job on the quick config (BASELINES.md). It is too expensive per candidate. It runs as tier E5 on rendered cards only (section 8.3), its `behaves_like` flags go into the card's `known_relatives`, and its descriptor becomes an archive field for those elites.

### 6.4 Keeping the loop's novelty consistent with S2

What the loop calls novel is checked against S2 in three ways:

1. **Same rules.** The structural screen uses REPAIR_TYPES and the size-preserving lift rule that S2 counts by, so any reduction S2 can count through a policy repair is one the loop can find, provided the target is in the library.
2. **Library growth from every S2 verdict.** When S2 kills a rendered card with a counted comparator, the comparator is harvested (6.1 item 4) and the candidate's cell is closed. When S2 kills on R2, the critic's reason is attached to the cell and shown to the operator as a reason the parent failed.
3. **The leak rate is measured.** Leak rate = rendered cards killed at S2 as repeats / rendered cards that reached S2. It is the loop's false-novelty rate, it is reported per generation of rendering, and the stop rule in section 10 is written in terms of it.

The rendered card also carries the program's own outcome-store signature (hybrid/research/signature.py `card_signature`) so S2's own-history and R3 rules see it like any other card.

## 7. The LLM as mutation and crossover operator

**Roles.** Two new roles in models.toml, routed in both tables:

| Role | Gateway route | Fallback | Use |
|---|---|---|---|
| `program_operator` | anthropic/claude-haiku-5-5 | anthropic/claude-sonnet-5-5 | mutate and cross; 2 to 4 programs per call |
| `program_renderer` | anthropic/claude-sonnet-5-5 | anthropic/claude-opus-5-5 | the prose fields of a rendered card |

The R2 screen reuses `reduction_critic` with a Sonnet-first entry in a new routing table, or the role itself if Andrew prefers Opus there; the design does not depend on it. The local Qwen (LM Studio :1234) is not used for the operator: it generates 20 to 34 tokens/s and is the serial statement_match judge in stage M (LT decision), so loop traffic would slow the math stages it feeds. It stays an option for a later run measured against Haiku on the same seeds.

**Prompt.** Each operator call gets:

- the BBIR grammar and primitive table (fixed text, versioned with the interpreter);
- two or three parents drawn from different cells, each with its program, its descriptor and its evaluation record in short form: tiers passed, E1 accuracies, the E3 result (the nearest library entry, its agreement, and the structural rewrite list or the named differences), E4 accuracy;
- for a parent that failed or reduced, the reason in host words ("normalizes to pushdown automaton through gate + sparsity", "E2 precision probe: accuracy 1.0 to 0.31 with 8 fewer mantissa bits", "R2 screen: a stack automaton with a cache keyed by the top symbol is a textbook memory-augmented PDA");
- the list of closed cells and the five most frequent library reductions so far;
- the instruction to return 2 to 4 programs as JSON, each with a one-paragraph rationale naming which parent property it keeps and which reduction it is trying to escape.

Mutation takes one parent; crossover takes two from different material state kinds. The operator may also be asked for a "fresh" program with no parent at a small rate (one call in eight) to keep the archive from collapsing onto its early elites.

**Malformed output.** A program that fails the schema or the type check is recorded at E0 with its error and returned to the same lineage once as feedback. There is no repair loop beyond that one feedback (fail fast).

## 8. How it plugs in

### 8.1 A new generation mode, arm A6

- generation.toml gains `[arms.A6] name = "program_search"`, with a mode flag `generation.mode = "program_search"` read by g_generate.build_generation. In this mode the explorer tasks are replaced by an injector of rendered cards, the pattern hybrid/research/planted.py and live_check.py already use: the orchestrator with one injector in place of the explorers. A6 can also run beside A1 to A4 with a weight, once the experiment justifies it.
- A new package hybrid/search (BBIR, interpreter, evaluator tiers, library, archive, loop), with no imports from hybrid/research except the stage entry it injects into, so dependencies point from research to search and the evaluator stays testable alone.
- The rendered card enters through s0_explore.admit unchanged (TF-IDF duplicate check, lineage, outcome signature). Its `arm` is A6, its `arm_structure` holds the program hash, the cell and the evaluation summary. The S0 arm classifier still assigns the five archive axes, so the card archive stays comparable with earlier runs.

### 8.2 Ledger, outcome store, inspect, resume

- **Ledger.** New event types, hash-chained like the rest: `program_proposed` (operator call id, parents, program hash), `program_eval` (tier results, kill reason), `archive_update`, `program_rendered` (card id, program hash), `program_feedback` (S2 verdict returned to the program). The program JSON and every evaluation record are artifacts, hashed.
- **Outcome store.** A killed program gets no outcome record (it was never a card). A rendered card gets the ordinary outcome record from its stage verdict, plus its program hash, so the outcome store links an S2 kill to the program that produced it.
- **Inspect.** `inspect` gets a program-search section: programs per tier and kill reasons, archive cells filled and closed, library reductions by family, rendered cards and their S2 verdicts, leak rate, calls per S2 verdict.
- **Resume.** The archive and the generation counter are rebuilt from `archive_update` and `program_eval` events, as the orchestrator rebuilds budgets from records today. A program whose evaluation has no record is re-evaluated (evaluation is deterministic in program, seed and interpreter version); an operator call with no `program_proposed` is not replayed, so a resume never pays for a call twice. Evaluation records carry the interpreter and library versions; a resume on different versions refuses to continue (fail fast).

### 8.3 Rendering

The renderer fills every theory card field the host can compute and asks one model call for the rest:

- **Host fields.** `mechanism` (state, update and readout equations printed from the IR), `resources` (widths, state bits and per-token cost formulas), `limits` (the capacity limits E2 measured, stated per decision L), `toy_example` (inputs and expected outputs computed by the interpreter, so S1's toy check reproduces them exactly), `known_relatives` (the nearest library entries with their agreement and the named differences; the E5 `behaves_like` flags when run), `archive_cell`.
- **Model fields** (`program_renderer`): title, summary, core_idea, claims, falsifiers, references. The prompt includes the program and its evaluation, and the rule that the prose may not claim anything the evaluation does not show.
- **Reference code for stage M.** The renderer compiles the IR to Python under the cost.py contract (`init(sizes, ops)`, `update`, `predict`) and under the constructions.py contract, so P1 and P2 run on the same mechanism the loop evaluated, and it attaches the E4 recipe (optimizer, rate, epochs, relaxations) as a card-supplied P4 protocol. Decision P4P allows card-supplied protocols for calibration cards; extending it to A6 cards is part of the decision row in section 13. The protocol must be timed in the pinned container before it is attached (the KD1 lesson).

## 9. Budget on this machine

Measured numbers, then the estimates the experiment will check.

| Item | Measured basis | Figure used |
|---|---|---|
| Calls per S2 verdict, prose cards | batch: 415 calls / 28 verdicts; recheck: 121 / 11 | 14.8 (batch), 11 (recheck) |
| Tokens per S2 verdict | batch: 12.2M / 28 | 0.44M |
| M and C per card | calibration card-runs, 3 at once: median 19 calls, 0.59M tokens, 8800 s; max 35 calls, 15093 s | 19 calls, 8800 s |
| Container NumPy speed | KD1: 12 to 17 times slower than the host | loop tiers stay on the host |
| Bench quick baselines | 95 to 141 s per baseline job, 5 in parallel on the VM | E5 at about 150 s per card on one slot |

Estimates for the loop, to be replaced by measurements in the verification of phase MS5:

| Per unit | Model calls | Host CPU | Container |
|---|---|---|---|
| Candidate program | 0.3 operator calls (3 programs per call) | E0 under 0.1 s; E1 2 to 5 s (5 tasks, 3 sizes, 50 instances of streams up to a few hundred tokens in a NumPy interpreter); E2 about 3 times E1; E3 about 15 library entries with a matched budget at 20 s each, about 300 s for programs that reach it; E4 at most 60 s | none |
| Generation (24 programs, 8 operator calls) | 8 Haiku calls | most programs stop at E0 to E2; with a guess of 4 reaching E3 and 2 reaching E4, about 1500 CPU s, which 12 host workers finish in a few minutes of wall time | none |
| Rendered card | 1 renderer call + 1 R2 screen call, then about 11 calls to an S2 verdict | negligible | E5 one bench job when run; M and C for S2 passers in the 4 executor slots |

The host has 16 CPUs; the loop uses at most 12 worker processes so the gateway, LM Studio and the Podman VM keep headroom. The executor slots (hybrid/slots.py, 4) are untouched by the loop and stay available for M, C and the bench.

## 10. First experiment

**Question.** At about the same model-call budget the batch spent to reach 28 S2 verdicts, do cards rendered from screened programs repeat known work less often and pass S2 more often than prose cards?

**Setup.**
- One run of the loop from an empty archive, seeded with three trivial programs (a single `Vec` with an affine update, a single `Stack`, a single `Table`), with the library of 6.1 items 1 to 3 plus every harvested S2 comparator from runs/discovery-live-20261017 and runs/defect-recheck.
- 40 generations of 8 operator calls (320 Haiku calls, Sonnet fallback), 24 programs per generation.
- Rendering: after generations 20 and 40, render the best elite of each open cell, at most 2 cells per material state kind, at most 20 cards in total; each goes through the R2 screen, and those that pass go through live_check (S0 admit, S1, R, S2) with the run-discovery.toml stage settings and a copy of the current outcome store. Cards that pass S2 go to M and C through `plumb`, at most 3 at once.
- Model calls counted per phase: operator, renderer, R2 screen, S1 to S2, M and C.
- Expected size: 320 operator calls, at most 20 renderer and 20 R2 screen calls, about 11 calls per card that reaches S2 (at most 220), and 19 to 35 calls per card in M and C. The upper case before M is 580 calls, against the batch's 415 for 28 S2 verdicts, and most of the 580 are Haiku calls. Host compute, from the section 9 estimates: about 1500 CPU s per generation, 60,000 CPU s for 40 generations, which 12 workers run in under 1.5 hours of wall time if the estimates hold; MS5 measures the real rate on 3 generations first, and the run is resized from that measurement before launch.

**Measures, against the batch.**

| Measure | Batch (prose) | Definition |
|---|---|---|
| Repeat rate | 31 of 40 = 78% | plan section 14: reduces to an excluded or known family, or matches an outcome signature without addressing its cause; over rendered cards with an outcome record, as SCORE.md criterion 3 counted it (phase MS0 reproduces the 31) |
| S2 pass rate | 0 of 28 | S2 passes / cards that left R |
| Calls per S2 verdict | 14.8 | all model calls of the run, operator calls included, / S2 verdicts |
| Leak rate | not defined | rendered cards killed at S2 as repeats / rendered cards that reached S2 |
| Positive control | not defined | the loop rediscovers the pushdown automaton (structural reduction recorded) within its first 10 generations |

**Stop rule, written before the run.**

- **Stop the direction** if the repeat rate of rendered cards is at least 60% **and** no card passes S2. With 20 cards, 12 repeats has a 95% Wilson interval of about 39% to 78%, which reaches the batch's rate, so at that result the loop cannot be shown to repeat less than prose, and it produced nothing S2 accepts. That result means program search lands on known work as prose does.
- **Stop and fix the evaluator, then rerun once,** if the positive control fails (the loop never finds the stack automaton) or the leak rate is above 80%: the loop is not seeing what S2 sees, and the experiment did not test the direction.
- **Continue** if the repeat rate is at most 40% (8 of 20 has an interval of about 22% to 61%, clear of 78%), or at least one card passes S2. A card that passes S2 then goes through M and C; the milestone is that card surviving its P1 cost and P2 construction obligations at stage M and C.
- **Between the two** (repeat rate above 40% and below 60% with no S2 pass): one more run with the library grown from this run's S2 kills, then the same rule. No third run without a new decision row.

Calls per S2 verdict is reported but does not gate: a lower number is a cost result, and a direction that produces no survivors cheaply has not succeeded.

## 11. Phases

Each phase leaves the test suite green (`.venv-hybrid/bin/python -m pytest hybrid/tests -q -p no:cacheprovider --ignore=hybrid/tests/test_probes.py`) and ends with its verification step. Sizes are lines of code, implementation plus tests.

| Phase | Content | Tests first | Verification | Size |
|---|---|---|---|---|
| MS0 harvest and baseline | Script that reads the batch and recheck ledgers: extracts every counted comparator and its check artifact, the repeat-code definition behind SCORE.md criterion 3, and calls per S2 verdict; writes results/math-search-baseline/ | Counts on a fixture ledger; reproduces 31 of 40, 28 verdicts and 415 calls on the real batch | The script's numbers equal SCORE.md | 250 + 150 |
| MS1 BBIR core | Schema, parser, type checker, width substitution, derived cost and state formulas, hard-mode interpreter, hashing | Type errors (position-indexed tables, missing soft rule, unbounded loop); cost and state formulas for each primitive; interpreter results on hand-checked streams; a vector-only affine program has the cost formula of a diagonal SSM | Derived cost equals the cost.py count on 6 compiled programs | 1200 + 800 |
| MS2 constructions and hacking probes (E1, E2) | Adapter from BBIR to the constructions.py contract; size schedules; held-out size; precision probe; materiality ablation; context permutation; capacity limit report | Hand-written hacking programs, one per row of section 5, each refused by the right probe; a correct stack program for Dyck and a correct table program for recall pass | The 6 hacking programs fail and the 2 correct ones pass, with the recorded reasons | 500 + 450 |
| MS3 soft mode and E4 | NumPy reverse-mode tape over the primitives (runs on the host and, being NumPy only, can be staged as a bbmath module), soft rules per primitive, trainer, learnability.py controls | Gradient checks per primitive against finite differences; the soft stack learns Dyck depth 4 from data on one seed; the two data controls do not both reach the threshold | E4 on the correct programs of MS2 passes; a trainer that ignores its data fails as `data_independent` | 900 + 500 |
| MS4 library and E3 | Library programs (6.1 items 1 to 3), harvested comparators translated to BBIR with a reproduction check, normal form, repair-tagged rewrites, size-preserving lift rule, numeric fit | Each library entry, renamed and reparametrized, reduces to itself; the 8 novelty-probe mechanisms reduce to their sources; a one-hot lift of a counter machine into an ssm is refused as not counted (the KD4 case); a program with a state-generated key is reported with an `other` difference | The novelty-probe set and the two disguised controls give the same family as the probe's S2 verdicts | 1000 + 650 |
| MS5 search loop | Archive (cells, elites, closed cells, novelty term), parent selection, operator prompt assembly, `program_operator` role, ledger events, resume, worker pool | Fake operator returning fixed programs: archive updates, closed cells, feedback text, resume after a kill rebuilds the same archive and repays no call | A 3-generation live run with Haiku: every program has an eval record; the archive rebuilds from the ledger; measured per-tier CPU replaces the section 9 estimates | 900 + 600 |
| MS6 rendering and A6 injection | Renderer (host fields, `program_renderer` call, R2 screen), IR to Python compiler for the cost.py and constructions.py contracts, P4 protocol attachment, injector into s0_explore.admit, `inspect` section, outcome-store program hash | A rendered card validates against theory_card.schema.json; S1's toy check reproduces the interpreter's values exactly; the compiled code's cost.py count equals the derived formula; a fake S2 kill flows back as library entry and cell closure | One rendered card through live_check to an S2 verdict with full ledger and inspect output | 700 + 500 |
| MS7 experiment driver and scorer | Config, driver, scorer for the section 10 measures and stop rule | Scorer on a fixture run | The experiment of section 10 | 300 + 150 |
| MS8 E5 bench descriptor (after a continue verdict) | Compile rendered programs to the bench candidate template, run `python -m hybrid.bench eval` in a slot, fold `behaves_like` flags into `known_relatives` and the archive | One compiled library program (the GRU) is flagged `behaves_like:gru` | Fingerprint of 3 rendered cards recorded | 300 + 200 |

Order: MS0 and MS1 first, since every later phase uses the interpreter; MS2 before MS3 (the correct programs E4 trains come from the MS2 tests); MS3 before MS4 (the numeric fit uses the trainer); MS5 and MS6 after MS4. Total about 6050 implementation and 4000 test lines.

## 12. Risks and open questions

- **The language may be too narrow or too wide.** Too narrow: every expressible mechanism is a library entry or a known combination, and the experiment's stop rule catches it. Too wide: the operator writes programs that pass E1 by brute capacity. The mitigation is the C17 substitution and the material-coupling descriptor; more primitives would widen the space further.
- **Textbook combinations.** A coupling of two textbook structures can still be textbook. The R2 screen and S2 feedback are the defence; the leak rate measures how well.
- **Numeric fit as a reduction test is approximate.** Agreement at 0.99 after a fixed fit budget can miss a comparator that a longer fit would match (false novelty, caught later at S2) or flag a comparator that matches on the probe streams only (false reduction, a lost candidate). Both directions are reported through the leak rate and through the numeric-only kills, which can be audited by rerunning the fit at a larger budget on a sample.
- **Learnability proxy.** E4 at one seed and 60 CPU s can pass mechanisms that fail P4 at three seeds. That is acceptable, because stage M runs the full P4; the proxy only removes mechanisms with no trainable path.
- **Stage M cost for A6 cards.** The compiled reference code and the card-supplied P4 protocol reduce what the prover invents (decisions R, P4P). If the M statement fixer still invents sizes or learners for A6 cards, that is a stage M defect to log, never a reason to change the loop.
- **Operator collapse.** FunSearch-style loops can converge on one lineage. The cell structure, the novelty term for parents and the fresh-program rate address it; the per-generation count of new cells is reported.

## 13. DECISIONS.md row (draft, not yet entered)

| Id | Decision | Reason |
| --- | --- | --- |
| MS | Generation moves to program search (MATH-SEARCH-PLAN.md). Candidates are programs in a typed sequence-mechanism language (BBIR) whose cost and state are derived from the types; host checks score them with no model call (constructions at growing sizes under C17 width substitution, hacking probes, a structural and numeric screen against a comparator library built from families.toml, textbook structures and every counted S2 comparator, and a one-seed learnability proxy); a cheap model (Haiku first, Sonnet fallback) mutates and crosses archive elites with the evaluator's failure reasons. Only screened programs are rendered into cards (arm A6), after one R2 screen call, and they enter S0 admission, S1, R, S2, M and C unchanged. Rendered cards carry compiled reference code and a card-supplied P4 protocol timed in the pinned container, extending P4P to A6 cards. First experiment: 320 operator calls, at most 20 rendered cards. Stop the direction if the rendered repeat rate is at least 60% and no card passes S2; fix the evaluator and rerun once if the loop cannot rediscover the pushdown automaton or more than 80% of rendered cards are reduced at S2. Rejected: more prose arms and prompt changes (the batch and recheck show the cards die for what they propose, and Si et al. arXiv 2409.04109 shows prose ideation collapses to duplicates); generated Python as the candidate (cost hides in built-ins, STATUS.md problem 3, and every evaluation would need the container, 12 to 17 times slower, KD1); a vector-only primitive set (AutoML-Zero arXiv 2003.03384 rediscovers linear models first, and here those are excluded families); replacing S2 with the loop's screen (the novelty probe shows S2 identifies published mechanisms correctly, so it stays the authority). | results/discovery-live-20261017/SCORE.md: 0 of 28 S2 passes, 78% repeats, 14.8 calls per S2 verdict; results/defect-recheck-20261017/RESULT.md: 11 of 11 re-checked cards die again on fixed gates; results/novelty-probe-20261017/RESULT.md: 8 of 8 published mechanisms reduced correctly, 2 of 2 controls killed, the one false kill from a lifted state (KD4). |
