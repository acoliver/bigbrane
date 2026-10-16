# prompt: refine_proposer

You proposed the theory card below. A challenger from another model family asked questions about it in the Bigbrane refinement dialogue. Answer every question and revise the card where an answer changes it.

How to answer:

- Answer each question by its id in `answers`: `answer` is what is true of the card, `change` says what you changed in the card (empty when nothing changed, for example when the answer is a definition or a premise).
- Revise by tightening: define a symbol where it is used, merge or drop objects, move a secondary part into `extensions`. The core (state objects plus update and readout equations) may not grow past the limit stated in the user message; a revision past it is rejected and the card stays as it was.
- Keep the core idea. A question is not a request to change the mechanism into something else.
- Keep the toy example's `expected` values true of the revised code; the toy example is run again after the dialogue and a mismatch fails the card.

Resources (FIX-16). Keep the card's `resources` section complete and consistent with every answer: the shared architecture (hidden width, precision, carried-state bits of a finite-state mechanism), the size names and widths of the cost formula, and for each standard task (copy, recall, dyck, composition, binding) the width d, the rounds and the state words as formulas in the task's size, plus the solvable size, example count m, d and rounds of each learned case. The math stage formalizes only what the card states: a resource the card leaves open stops the card at the math stage with `card_incomplete`. When an answer changes a size, change the resources with it.

The assumption list. Return the complete current lists every time (not only what changed), in the vocabulary of the Bigbrane math stage:

- `definitions` (ids D1, D2, ...): conventions and protocol choices the card's claims are stated under: how inputs are sampled (`sampling`), how the model is evaluated (`evaluation`), how cost is counted (`accounting`), how a baseline is built (`baseline`), notation (`notation`), or `other`. A definition has no truth value.
- `premises` (ids A1, A2, ...): substantive claims about the mechanism, the data or the world that a claim relies on but the card does not prove: an oracle the mechanism consults (`oracle`), a store it reads (`store`), a prior it assumes (`prior`), or `other`. For each give `why_needed` (which claim needs it), `does_work` (true when the claimed advantage depends on it), and `testable_by`: `code` with a `measurement_plan` (measure, data, procedure, accept_if) or `citation` with a source (`arxiv` id or `openalex` work id) and a quote. Do not list conventions as premises, and do not restate a claim as a premise.

Return only the JSON object matching the schema: `card` (the full revised card draft, without id, version, parent_ids, created_by_call, lens or status), `answers`, `definitions`, `premises`.
