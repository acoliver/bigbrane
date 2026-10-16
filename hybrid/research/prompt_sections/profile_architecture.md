ARCHITECTURE CONTRACT (FIX-4). The frozen profile states the feed-forward hidden width H as a formula in d and the precision of stored numbers; each P1, P2 and P4 statement states its `widths` (d, rounds, optionally positions) in its own size names. The checks survey every array your code holds (module-level data, the state `init` returns, the parameters of a cost reference, the model a learner returns) and every list, tuple or dict of arrays:

- no axis may exceed the largest declared dimension at that size, max(d, H, rounds, positions, vocab); a list of arrays counts its length as an axis;
- a P2 construction stores floats at no more than the declared precision (cast float64 work to float32 before storing it when the contract says 32 bits);
- a P1 cost reference must hold an array of width H when H differs from d, so the counted cost is the cost of the declared architecture.

- module-level data is fixed once `init` returns: everything carried from one token to the next must be in the state `update` returns. A write after `init` to module data, a function default, a closure or a class attribute is hidden storage and fails the size;
- when the contract declares `state_bits` (the carried-state budget of a finite-state mechanism, separate from the float precision), the state `init` and every `update` return must be integers: one index z with 0 <= z < 2**state_bits, or several integers that pack into state_bits bits together. The check counts the distinct whole configurations at each size and fails a size that exceeds 2**state_bits. One Python integer is one state word, but it does not hold more configurations than the budget allows.

A construction that needs a wider hidden layer than the contract declares, or more carried configurations than its state_bits, fails its statement, however accurate it is. A size that breaks a resource check never counts as a P2 success. Build the witness at the declared widths and budget, or report that they are not enough.
