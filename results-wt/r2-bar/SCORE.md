# Replay score

| card | run | S2 verdict | class | earlier kill | gate reason |
|---|---|---|---|---|---|
| card-084a2510 | batch8 | fail | R2 | R2 | gate: not radical enough: textbook transplant, the source field's construction unmodified: The core is the Todd-Coxeter/HLT scan-and-deduce step on a coset table, applied unchanged to demonstration words with given endpoints. A permutation-automaton lookup readout is added. The card only removes coset definition and coincidence processing and adds the standa |
| card-0fe3943c | batch8 | fail | no_difference | R2 | gate: no_difference: difference statement rejected: The textbook Hopcroft–Motwani–Ullman pushdown matcher is incorrectly identified as literature source arxiv:2103.04079, which is a different work on determinizing event-clock input-driven pushdown automata; the textbook comparator needs its own closest_prior entry.; The standard balanced-bracket matching/fre |
| card-17100001 | probe-card-17100001 | reduces_to_known | reduces_to_known | probe | gate: update rule reduces to other_known: Card state (q_t, z_t) maps to the DPDA configuration (control state, stack contents) of Ginsburg and Greibach (1966). Control: q in {live, dead} is the DPDA finite control, with dead as an absorbing reject sink. Stack: z_t is the pushdown store over Gamma = {(, [}, starting empty. Transition delta(q, top(z), x_t) wor |
| card-17100002 | probe-card-17100002 | reduces_to_known | reduces_to_known | probe | gate: update rule reduces to other_known: The card is the queue automaton (Post-machine style FIFO automaton) with nothing renamed. Card q_t in {store, compare, dead} is the finite-control state. Card Q_t is the FIFO queue contents. delta is the queue-automaton transition: in store phase on a or b, enqueue x_t; on #, move to compare; in compare phase on x_t, |
| card-17100003 | probe-card-17100003 | reduces_to_known | reduces_to_known | probe | gate: update rule reduces to other_known: Card state (q, c1, c2) is the finite control state and the two registers of a Minsky two-counter machine (Minsky 1967, ch. 14). Restricted to one move per input symbol, it is a deterministic real-time 2-counter automaton (Fischer-Meyer-Rosenberg; Merrill's real-time counter automata, arxiv:2004.06866). The card's del |
| card-17100004 | probe-card-17100004 | reduces_to_known | reduces_to_known | probe | gate: update rule reduces to other_known: Card symbols map one-to-one onto the SCFG inside algorithm of Baker (1979) and Lari and Young (1990), as presented in Manning and Schuetze ch. 11. Card chart I[A,i,j] is the inside probability beta_A(i,j) = P(x_{i..j-1} / A). Card e_A(x_i) is the lexical rule probability P(A -> x_i). Card p(A->BC) is the binary rule  |
| card-17100005 | probe-card-17100005 | reduces_to_known | reduces_to_known | probe | gate: update rule reduces to other_known: Card state C[0..t] (sets of items (A, rhs, dot, origin)) maps to Earley's state sets S_0..S_n of dotted productions with origin pointers (Earley 1970). The card's 'prediction' rule maps to Earley's Predictor. 'scanning' maps to Scanner. 'completion' maps to Completer. Saturating prediction and completion to a least f |
| card-17100006 | probe-card-17100006 | reduces_to_known | reduces_to_known | probe | gate: update rule reduces to other_known: The card is the Viterbi algorithm as published, with no change. Card v_t(j) is the Viterbi score delta_t(j), the probability of the best path ending in state j at time t. Card A(i,j) is the HMM transition matrix a_ij, card e(j,x) is the emission b_j(x), and card pi(j) is the initial distribution pi_j. The initializat |
| card-17100007 | probe-card-17100007 | fail | prior_art | probe | gate: prior_art_tested and no untested element (The card's update rules (append, digram uniqueness, rule utility with inlining of single-use rules) are the published SEQUITUR mechanism, as the evidence states: cs/9709102 is labeled same_mechanism and reports the same incremental, digram-uniqueness and rule-utility constraints. The card's own summ) |
| card-17100008 | probe-card-17100008 | reduces_to_known | reduces_to_known | probe | gate: update rule reduces to rnn: Card h_{t-1} (binary vector of length W+2(t-1)) is the RNN hidden state h_{t-1}. The RNN state grows by two cells per step, so the RNN uses a lifted, growing state. The RNN recurrent matrix W_t is a (W+2t) x (W+2t-2) banded Toeplitz matrix. Row j (card cell i = j - t) carries weights 4, 2, 1 on the stored cells i-1, i, i+1.  |
| card-17100009 | probe-card-17100009 | reduces_to_known | excluded_family | probe | gate: excluded_family:ssm: the update rule reduces to ssm (check verified; counted comparators: mamba_zoh_n1, selective_diag_ssm_gated), which is in the out-of-scope family State-space models (S4, Mamba, SSD, LRU, RetNet) and linear recurrences (families.toml, decision D8); a LEARNING_RULE: difference does not apply: Card h_t = f_t ⊙ h_{t-1} + (1-f_t) ⊙ u_t, |
| card-1710000a | probe-card-1710000a | reduces_to_known | excluded_family | probe | gate: excluded_family:delta_rule: the update rule reduces to linear_attention (check verified; counted comparator: deltanet_delta_rule), which is in the out-of-scope family Delta rule, fast weights and in-context least squares (DeltaNet, gated DeltaNet, TTT, RLS, MesaNet, Titans) (families.toml, decision D8); a LEARNING_RULE: difference does not apply: Card  |
| card-3485a13f | batch8 | pass | pass | R2 | prior art: prior_art_undertested. Close prior work exists for the components: abductive program learning from few demonstrations (arxiv:1904.06317), library learning and program abstraction (arxiv:2106.11053, arxiv:2211.16605), abduction integrated with induction (openalex:W178818641), and few-shot  |
| card-440f0204 | recheck | fail | no_difference | reduces_to_known | gate: no_difference: difference statement rejected: Relative to arxiv:2306.04470, update[1] remains exactly the supported update Pi <- tau Pi with tau = (pi(i), j); selecting that transposition using token locations and the current permutation is a different application of the same form, not a change to its update equation.; The claim that an externally supp |
| card-62785c10 | recheck | fail | R3 | excluded_family | gate: not_radical:R3: difference statement rejected: the challenger finds the cause not addressed: The cause is reduction to a known algorithm. update[2] changes only a known error nonlinearity, the exact flow of the Bhat-Bernstein law, inside the LMS-with-error-nonlinearity form. The remaining equations (RESET, APPLY, READ) are textbook phase-accumulator fo |
| card-76ee768c | recheck | reduces_to_known | reduces_to_known | reduces_to_known | gate: update rule reduces to other_known: theta is the point on the flat torus T^d. Active rows of Y give integer constraint vectors a_{ijl} = b[i,j,l,:], with +1 at i, +1 at j and -1 at l, multiplicities combined. Stacked, these form the incidence A, whose free-column block is B; the anchor columns give h. The objective E(theta) = 0.5*sum_{active} dist_{S^1 |
| card-7e993aab | recheck | fail | no_difference | R2 | gate: no_difference: difference statement rejected: no real difference from ssm: Without the reset, the update is exactly the input-dependent diagonal SSM identified by the critic. The reset adds a state-conditioned gate selecting between the masked state and fresh evidence, which does not qualify under the prohibition on gate-only differences. |
| card-8154d3b4 | batch8 | pass | pass | R2 | prior art: novel. No listed paper addresses the proposed mechanism: a learned, state-dependent Schmidt circulation of a bipartite amplitude tensor, followed by quadrature reinjection inside a support mask. The closest listed items are general tensor-network learning (arxiv:1605.05775) and Schmidt-de |
| card-83bd879a | recheck | reduces_to_known | reduces_to_known | reduces_to_known | gate: update rule reduces to hopfield_or_kernel_memory: Base form: the clipped heteroassociative Willshaw net (Steinbuch Lernmatrix). Its write is W <- max(W, x y^T) and its read is y_hat = 1[W^T x >= /x/], over sparse binary codes.

Card to base:
- Code: the SDM activation a(y), the s nearest hard locations of E[y], is the sparse binary code. It is a fixed  |
| card-89248ee5 | recheck | pass | pass | R2 | prior art: prior_art_undertested. Sparse hard-location memory (Kanerva SDM) and sparse-graph sequence storage (Associative Knowledge Graphs) are close prior work, and an SDM has been used for sequence-based navigation. None of the retrieved papers uses lag-two Boolean graphs with a degree-taxed tria |
| card-8ad6004f | recheck | fail | R2 | R2 | gate: not radical enough: known work (cited source covering the mechanism: standard_algorithm Relaxation labelling (Rosenfeld, Hummel & Zucker 1976) / mean-field Potts graph matching (Peterson & Söderberg 1989; Gold & Rangarajan 1996 graduated assignment / softassign); standard_algorithm Seeded / percolation graph matching (seed-and-extend propagation of cor |
| card-909ad773 | batch8 | fail | no_difference | R2 | gate: no_difference: difference statement rejected: no real difference from exact_match_cache: Both reads retain the family's key-equality form: position keys address B, and token keys address tau. Affine query arithmetic, per-line tables and vote-based selection change how existing lookups are composed and selected, not the retrieval equation. |
| card-97baae43 | recheck | reduces_to_known | reduces_to_known | reduces_to_known | gate: update rule reduces to other_known: Explicit deterministic pushdown stack plus a stack-top-conditioned count table, with the stack stored as an injective integer encoding. The card's symbols map onto this form as follows:
- Stack S_t, a word over {0..K-1}, maps to M_t = Phi(S_t) = W_{s_1} W_{s_2} ... W_{s_D}. Phi is the monoid homomorphism into the fre |
| card-a4f684fb | recheck | pass | pass | no_difference | prior art: novel. None of the 64 results addresses the core mechanism: installing in-context bindings by conjugating a carried permutation with a transposition read from its own state, so that cycle type is preserved and forced unseen bindings follow. The permutation-related hits concern combinatori |
| card-a775f78d | batch8 | fail | R2 | R2 | gate: not radical enough: known work (cited source covering the mechanism: paper openalex:W2910040435; paper arxiv:1804.01186; standard_algorithm Greedy hill-climbing / local search with a learned heuristic (append/delete edit neighbourhood, first-index tie-breaking); paper arxiv:2402.06990): The escape argument holds against the parent's reduction. The card |
| card-c5524960 | recheck | fail | R3 | R2 | gate: not_radical:R3: difference statement rejected: the challenger finds the cause not addressed: The recorded cause concerns the earlier card's nearest-past-key attention readout and equivalent wavefront transport equations, not this card's binary merge equation update[1]. The present mechanism avoids that reduction, but changing its segment merge does not |
| card-d6dfeac2 | recheck | fail | R2 | reduces_to_known | gate: not radical enough: textbook transplant, the source field's construction unmodified: The construction is Gottesman's Pauli-measurement update in Aaronson-Gottesman destabilizer-tableau form (CHP), with its deterministic-measurement sign readout and the Born rule. The card substitutes a token-compiled stream of postselected Pauli-product measurements an |
| card-ebf0e99d | batch8 | fail | no_difference | R2 | gate: no_difference: difference statement rejected: The claimed change in update[0] is exactly the standard token-bucket equation with candidate evaluations substituted for traffic, so it is a different application of an unchanged known form rather than an equation-level difference.; The statement that a correctly predicted token spends zero compute is false |
| card-f5ee7edb | batch8 | pass | pass | R2 | prior art: prior_art_undertested. The closest found work is higher-order motif link prediction (1902.06679, 1906.05059), which scores edges using two-path and higher-order motif statistics, a family related to the proposed connected-witness term Q. Long Sequence Hopfield Memory (W4379958458) uses He |

## Rule 1 (novelty probe guard), registered named-gate predicate: NOT MET

10 of 10 dead; 9 of 10 died at the gate the rule named (reduces_to_known for the 8 published, excluded_family for the 2 controls).

- card-17100001: dead, code reduces_to_known (named gate)
- card-17100002: dead, code reduces_to_known (named gate)
- card-17100003: dead, code reduces_to_known (named gate)
- card-17100004: dead, code reduces_to_known (named gate)
- card-17100005: dead, code reduces_to_known (named gate)
- card-17100006: dead, code reduces_to_known (named gate)
- card-17100007: dead, code prior_art (other gate)
- card-17100008: dead, code reduces_to_known (named gate)
- card-17100009: dead, code excluded_family (named gate)
- card-1710000a: dead, code excluded_family (named gate)

## Rule 2 (counted reductions stay dead), registered reduction-gate predicate: NOT MET

6 of 6 dead; 3 of 6 died at the reduction gate itself (reduces_to_known or excluded_family); the rest died at another gate (no_difference, R3, R2) and FAIL this registered rule.

- card-76ee768c: dead, code reduces_to_known (named gate)
- card-440f0204: dead, code no_difference (other gate)
- card-62785c10: dead, code R3 (other gate)
- card-83bd879a: dead, code reduces_to_known (named gate)
- card-97baae43: dead, code reduces_to_known (named gate)
- card-d6dfeac2: dead, code R2 (other gate)

## Secondary all-dead diagnostic (not an acceptance rule)

Probe: all dead; counted-before: all dead. Later-gate kills do not satisfy either registered named-gate guard.


## Rule 3 (flips to pass): card-3485a13f, card-f5ee7edb, card-8154d3b4, card-a4f684fb, card-89248ee5

