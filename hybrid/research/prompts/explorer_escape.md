# section: explorer_escape

## Operator `escape`

This section applies only when the user message names the operator `escape`. It adds to the instructions above; everything above still holds (parsimony, the card contents, the toy example, citing only papers you were given).

The parent card was reduced to a known architecture at S2. The user message gives you the parent card, the reduction critic's output (the known family in reduces_to, the symbol-by-symbol mapping, what was left over in difference_if_any) and the numeric check: a program that ran the parent's equations next to the known form built through the mapping and found them equal to floating-point precision, or the critic's confidence where no check could run. Treat the parent as a known model: renaming it, adding a gate, a schedule, a nonlinearity on the readout, a different feature map, a bigger state or a second memory beside it will be reduced the same way, because the mapping absorbs every one of those.

Your task is a mechanism the mapping cannot reach. Do this in order:

1. Read the mapping and the check code, and name the one property of the parent that the mapping relies on. Common ones:
   - the state update is linear in the state, or additive (S_t = A_t S_{t-1} + write_t), so the state is a weighted sum of per-token writes;
   - the update is associative, so a parallel scan or chunked form computes it and the order of writes enters only through decay factors;
   - the write is linear in the key or bilinear in key and value (k v^T, or a delta rule beta (v - S k) k^T);
   - the state has a fixed size that does not depend on what was stored, and capacity comes only from its dimension;
   - retrieval is exact-match or a similarity lookup over stored items (one-hot keys, hashing, softmax over patterns);
   - the procedure is a textbook algorithm or data structure (union-find, a worklist closure, a lookup table) run on the token stream.
2. Change the mechanism so that this property fails, in the core, not in an extension. Examples of the kind of change that counts: an update whose result depends on the order of writes in a way no diagonal or matrix decay reproduces; a state whose shape or structure (number of components, graph, rank, support) is decided by the data and changes during the sequence; retrieval by a computation over the stored structure (composition, solving, inference) rather than by match or similarity; a write that is not a sum of per-token terms.
3. Check yourself before you answer: write your update in the known family's form and confirm it does not fit. If it fits after a change of basis, a larger state, a learned feature map or a gate, you have not escaped; pick another change. Breaking linearity with an elementwise nonlinearity, h_t = f(W h_{t-1} + U x_t), gives an RNN, which is also a known family; breaking it with a softmax over stored items gives attention or a Hopfield layer.

The change must still serve the target: cost that does not grow quadratically in n, and few-shot use of a rule shown a handful of times. Keep the core within the parsimony limits. Do not escape by making the mechanism vague: an underspecified update fails S1, and it does not make the reduction wrong.

Fill the field non_reduction_argument, which escape cards must have:

- known_family: the family the parent reduced to, named as the critic named it (for example "linear_attention: preconditioned delta rule / MesaNet", "other_known: labeled union-find").
- broken_invariant: the property of that family the parent's mapping relied on, stated precisely, and the equation or step of your mechanism that breaks it. Example shape: "The family's state is a sum of per-token rank-one writes, S_t = sum_i w_{t,i} k_i v_i^T. In my update u[2] the write at step t depends on the stored graph through a shortest-path solve, so S_t is not a weighted sum of terms that depend on one token each."
- why_unrepairable: why the old mapping cannot be repaired: why no change of basis, larger or lifted state, learned feature map, extra gate or decay, or second memory makes your update and readout equal to a member of the family. Name the strongest repair you can think of and say what it would have to compute that it cannot. A critic will try exactly that repair in code.

State the escape card's own P3 limits in `limits` as the instructions above describe (FIX-18): the escaped mechanism has its own boundary, so do not copy the parent's limits unless its state, update or readout still implies them.

Return only the JSON card draft that matches the schema, with non_reduction_argument filled in.
