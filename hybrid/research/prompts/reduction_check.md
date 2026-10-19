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

The host decides equivalence from executed card traces and its independently implemented published reference, with its fixed tolerance (1e-6 on max_abs_diff). The difference you report between model-written implementations is diagnostic only. Do not print an `equivalent` field or any verdict of your own: an object with extra fields is refused and the check counts as unverified. Report card outputs on their original readout scale. Do not divide them by a sample maximum or otherwise rescale the reference samples.

What counts. The host reads your declaration, not labels in the program's output:

- Report every declared comparator exactly once, under its id, with `family`, `transformations` (the declared types, without the descriptions) and `limit_only` exactly as declared. A declared id that is missing or repeated, an id you did not declare, or values that differ from the declaration make the whole report unverified.
- Implement the comparator you declared, not a simpler one. If you declared your strongest repair (a normalizer, a hop mixture, a lifted state), the comparison under that id must include it; replacing it by the plain published form, or by a weaker repair, misreports the claim.
- A declared comparator of your `reduces_to` family whose transformation types are all in the family policy is eligible, but counts only with a supported independently executed host reference. That reference measurement, not your self-comparison, verifies within the tolerance or counts against the claim outside it. A comparator with type `other` is unresolved and never verifies. A `limit_only` comparator never verifies; measured at the card's finite parameters, its difference counts against the claim.
- Implement each comparator from its declared `components` only. An activation, transition, update or readout the declaration marks as published must be coded as that published function (tanh, sigmoid, the cited algorithm's own step), never as a call to the card's function. A self-comparison that reuses the card's own update or readout code supplies no independent evidence, however small its reported difference.
- Anything else you computed (the card's equations rearranged into another notation, a sanity check) goes in the list with `declared_id: null`, and the host never counts it.
- No correction terms. A comparator may not add a term whose only purpose is to cancel the card's difference from its base form; such a term is not a repair of any type. If you computed one, report it with `declared_id: null`.

Every comparison you compute must appear in the list, with its true max_abs_diff; do not leave out one that disagrees.

The check must be honest: do not build a comparator from the card's outputs, and do not compare a quantity with itself. If the card's equations are too underspecified to implement, or the declared equality holds only in a limit that cannot be computed, return check_code as null and explain why in explanation.

Return only the JSON object matching the schema.

Independent reference evidence (required to count a comparison): the critic's declaration must include `reference_spec: {"operation": ..., "parameters": ...}` for one of the host operations listed in your request. In the comparison report add `reference_samples: [{"inputs": [...], "card_outputs": [...]}]`, with the card's actual output at every step for each input stream. These arrays must be on their original output scale; do not divide, normalize or otherwise alter them. The host computes the published reference outputs and the numeric difference itself. Your `max_abs_diff` remains diagnostic only. A declaration or a small difference between two programs you wrote cannot verify a reduction. No samples, missing code, a crash or an unsupported reference stays unverified regardless of confidence.

Supported references and their conventions:
- `elman_tanh`, `elman_identity`: W, U, b, C, h0; h = activation(W h + U x + b), y = C h. Inputs are vectors. Declare the matching activation component.
- `linear_ssm`: A, B, C, h0; h = A h + B x, y = C h. Inputs are vectors.
- `input_ssm`: A[token], B[token], C[token], h0, mask[token]; h = (A[token] h + B[token]) * mask[token], y = C[token] h. Inputs are integer token indices. The binary mask is a fixed table selected only by the input. This supports the `sparsity` repair only, not a mask selected by hidden-state scores or a reset of evicted channels.
- `linear_attention`, `delta_rule`: S0; each input is an object with key, value, query vectors (and scalar beta for delta_rule). S += outer(value, key), or S += beta outer(value - S key, key); y = S query.
- `jeffrey_ipf`: p0, masks (binary partitions indexed by token), targets (interior probabilities), labels (nonnegative integer label per atom), label_count (the output vocabulary size). Inputs are integer token indices. Scale each partition cell to its target mass. Read the conditional label marginal within the selected cell. This is Jeffrey conditioning (1965), also one binary-partition IPF step (Csiszar 1975).
- `softmax_memory`: keys (pattern rows), values (corresponding value rows), beta. Each input is a query vector; output softmax(beta keys query) weighted values, the modern Hopfield memory read. No state update or additional repair is implied.
- `dyck2`: no parameters; input 0 or 1 opens its bracket, 2 or 3 closes the corresponding bracket. Read 1 if the prefix is balanced and valid, else 0, as a one-element vector. Invalidity persists.
- `rule110`: h0, a binary ring; each input is one clock tick, with periodic boundaries and synchronous updates. Output the full ring after each tick.
- `viterbi`: initial, transition (from-state rows), emission (state rows, token columns). Inputs are token indices. At the first token multiply initial by emission; thereafter take max over predecessor score times transition, then multiply emission. Output the score vector at every token, without normalization.

These references cover only the listed operations and repairs. A genuine published algorithm outside this list is not a restatement merely because it lacks a host implementation, but its model-written check remains unverified. Do not select a different operation, freeze a learned part or invent reference samples to make it count.
