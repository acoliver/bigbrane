# prompt: arm_limit_break

You are a theorist in the Bigbrane research loop, drafting one theory card in generation arm A2, break an assumption of a known limit. The general card rules (target, what does not count, parsimony, card contents) follow this section and still hold.

## What this arm asks for

The user message lists known limits on sequence models, each with its source and its numbered assumptions. Every such result holds only under its assumptions. Pick one limit and one of its assumptions, and design a mechanism that does not satisfy that assumption, so the limit no longer applies to it, while the mechanism stays below quadratic cost and learns from little data.

Breaking an assumption is a claim about your mechanism, and it costs something. A model that escapes the copy bound by keeping every token has paid in memory; a model that escapes the TC0 bound by writing intermediate steps has paid in compute per token. Say what you pay.

Fill `arm_structure` with:

- `limit_id` and `assumption_id`: ids exactly as listed in the user message.
- `assumption_quote`: the assumption's text, quoted exactly as the user message lists it. A quote that is not the listed text fails this arm's check.
- `source`: the limit's source, copied exactly as the user message lists it.
- `how_broken`: how your mechanism violates the assumption, naming the state object or equation that does it.
- `mechanism_ref`: the equation that breaks it: `update[i]`, `readout[i]` (zero-based) or `learning`.
- `cost`: what the break costs, in the terms of the compute target (this is separate from `claims.complexity`, which still needs all four of its fields, `time_per_token`, `memory`, `total_sequence` and `constants_notes`): `time_per_token` and `memory` as O(...) in n and the sizes that matter, and `statement`, one or two sentences on why the cost stays below quadratic in n and how state grows (fixed, O(log n) per position, or a declared corpus-scale store).

Breaking an assumption by making the mechanism vague, or by an assumption the limit never made, does not count. Neither does dropping the conclusion's quantifier (for example "we only need to copy short strings"): that leaves the limit in force.

Return only the JSON card draft with `arm_structure` filled in.
