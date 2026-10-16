# prompt: card_resources

You wrote this theory card (FIX-16, card resources). The math stage formalizes it only from what the card states, and the card does not yet state some of its own resources: the sizes the mechanism uses to solve the standard tasks. Without them the statement writer would have to invent a width or a round count, and that invented number would decide the result. State them now in the card's `resources` section.

What to state:

- `architecture`: `hidden`, the feed-forward hidden width as a formula in d (`"d"` when there is no wider layer); `precision_bits`, the float width of stored numbers (16, 32 or 64); `state_bits`, the bits of the whole carried state when it is a finite discrete set (a state index, a window of symbols), as a number or a formula in d, else null.
- `stream_symbols`: the shape symbols that count tokens read so far (for example `t` in a store with one row per position).
- `dimensions`: other dimensions of the equations that do not depend on the task size (for example the number of states).
- `cost`: `sizes`, the size parameters besides n that the cost formula of the card's complexity claim uses (for example `["d", "V"]`), and `d` and `rounds` as formulas in them.
- `tasks`: for copy (size L), recall (size K, keys drawn from max(2K, 16) possible keys), dyck (size depth), composition (size T) and binding (size hops): `d`, the model width; `rounds`, the rounds or layers; `state_words`, the peak carried state in words; each as a formula in that size, plus any other shape symbol under `symbols`. Choose the width the mechanism needs to solve the task at that size, as the card's capability claims argue it; a card that claims a task fails within fixed resources states those fixed resources.
- `learning`: for copy, recall, dyck and composition (or binding): the solvable `size` at which the card states its example count, `examples`, the count m the card states, and `d` and `rounds` there.

- `limits` (FIX-18, a top-level card field returned next to `resources`): the card's own P3 limits, at least one. Each states a task family the mechanism cannot solve within stated resources: `task` (copy, recall, dyck, composition, binding, or other), `size` (the task size from which it holds, or null), `resources` (what it holds within, for example the width at a stated size), `why` (the derivation, for example a counting argument on the bits the state can hold) and `kind` (state, update or readout, the part of the mechanism it follows from). A limit that follows only from the output vocabulary or the task encoding does not count. A limit on a task the card claims to solve must sit beyond the sizes or resources the claim covers, or the card contradicts itself. The statement writer formalizes these limits and invents none.

Every value must agree with the card's own text: its capability claims, its limits, its complexity claim and its stated example counts. Do not change the mechanism. If a value is a choice the card had left open, make the standard choice for this kind of mechanism and say why in the entry's `why`.
