# prompt: arm_base_formatter

You format a research note into a Bigbrane theory card (generation arm A5). The note was drafted by a base language model. Your job is transcription, not invention.

Rules:

- Every state object, equation and learning step of the card must come from the note. Do not add a mechanism, an operator, a memory, a gate or a learning rule the note does not state. Where the note is vague, write the vague part as stated and keep it out of the core; do not repair it with your own design.
- You may add what a card needs that is not mechanism: names and shapes of the objects the note uses, the toy example that computes the note's equations on a tiny input, claims and falsifiers that follow from the note, known relatives.
- The general card rules follow this section; the parsimony limits and the card contents apply.

Fill `arm_structure` with:

- `note_excerpt`: the sentence or sentences of the note that state its core idea, quoted exactly.
- `added_by_formatter`: a list of everything you added that the note did not state (definitions, shapes, the toy example's inputs), so a reviewer can see it.

A later check compares the operators in your card's equations with the operators in the note; an operator that appears only in the card fails the arm.

Return only the JSON card draft with `arm_structure` filled in.
