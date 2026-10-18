# prompt: reduction_check

You are the S2 reduction critic, asked to back your claimed reduction with a numeric check. You receive the theory card, your reduction claim (the known form, the family in `reduces_to`, the mapping and the comparators you declared in `comparators`) and the family policy the host judges the check by.

Write check_code: a self-contained Python program (standard library and numpy only, no files, no network, under 20 seconds) that

1. implements the card's update and readout exactly as the card writes them, on small random inputs with a fixed seed (for example n = 12 tokens, small d), with the card's parameters drawn at random where the card leaves them free;
2. implements every comparator you declared, exactly as declared: its base form with its declared `update` and `read` equations and every declared transformation, with parameters built from the card's parameters through its `parameter_mapping`;
3. runs each comparator and the card on the same inputs and compares the outputs at every step;
4. prints exactly one JSON object on its last line, reporting every comparison it ran as data:

```
{"comparisons": [{"declared_id": "<the comparator's id from your declaration>" | null,
                  "comparator": "<what this form is>",
                  "family": "<the declared base_family>",
                  "transformations": ["<each declared transformation type>", ...],
                  "limit_only": <the declared limit_only>,
                  "reference": "<the published form and its source>",
                  "inputs_tested": <number of input sequences>,
                  "max_abs_diff": <largest absolute output difference over all tested inputs and steps>}],
 "notes": "<short>"}
```

The host decides equivalence from these numbers, with its own fixed tolerance (1e-6 on max_abs_diff). Do not print an `equivalent` field or any verdict of your own: an object with extra fields is refused and the check counts as unverified. Compare outputs on the scale the card's readout produces (probabilities or normalized outputs); if the outputs are not of order one, divide both by the largest absolute card output before taking the difference, and say so in notes.

What counts. The host reads your declaration, not labels in the program's output:

- Report every declared comparator exactly once, under its id, with `family`, `transformations` (the declared types, without the descriptions) and `limit_only` exactly as declared. A declared id that is missing or repeated, an id you did not declare, or values that differ from the declaration make the whole report unverified.
- Implement the comparator you declared, not a simpler one. If you declared your strongest repair (a normalizer, a hop mixture, a lifted state), the comparison under that id must include it; replacing it by the plain published form, or by a weaker repair, misreports the claim.
- A declared comparator of your `reduces_to` family whose transformation types are all in the family policy counts toward that family: within the tolerance it verifies the reduction, outside it counts against the claim. A comparator with type `other` is unresolved and never verifies. A `limit_only` comparator never verifies; measured at the card's finite parameters, its difference counts against the claim.
- Implement each comparator from its declared `components` only. An activation, transition, update or readout the declaration marks as published must be coded as that published function (tanh, sigmoid, the cited algorithm's own step), never as a call to the card's function. A comparator that reuses the card's own update or readout code is a restatement and the host does not count it, however small the difference.
- Anything else you computed (the card's equations rearranged into another notation, a sanity check) goes in the list with `declared_id: null`, and the host never counts it.
- No correction terms. A comparator may not add a term whose only purpose is to cancel the card's difference from its base form; such a term is not a repair of any type. If you computed one, report it with `declared_id: null`.

Every comparison you compute must appear in the list, with its true max_abs_diff; do not leave out one that disagrees.

The check must be honest: do not build a comparator from the card's outputs, and do not compare a quantity with itself. If the card's equations are too underspecified to implement, or the declared equality holds only in a limit that cannot be computed, return check_code as null and explain why in explanation.

Return only the JSON object matching the schema.
