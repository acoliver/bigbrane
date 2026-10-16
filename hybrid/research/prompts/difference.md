# prompt: difference

You proposed the theory card below. Before it goes on, write its difference statement: what the closest earlier work did, why it failed or fell short, and the precise change this card makes. Rule R3 of the Bigbrane plan applies: a card that reuses a known-failed idea must name the recorded cause of failure and change exactly that cause.

Fields:

- `closest_prior`: the closest earlier works, closest first. Each has a `kind` and a `ref`:
  - `literature`: a source id from this card's prior-art search, exactly as listed in the user message;
  - `outcome`: a record id (`oc-...`) of the Bigbrane outcome store from the OWN HISTORY list: an earlier card of ours with this mechanism or a near one, and how it died;
  - `family`: a family id from the known-family list (for example `hopfield_associative_memory`, `exact_match_cache`, `linear_attention`).
  For each give `title`, `what_was_tried` and `why_it_failed` (a quote from the work, or for an outcome record its cause id and what it says). Give at least the number of literature and outcome entries the requirements state. Do not invent ids: an id that is not in the lists is rejected.
- `what_was_tried`: across those works, the approach this card is closest to, in one or two sentences.
- `why_it_failed`: why that approach failed or fell short, as a quote or an outcome cause id.
- `precise_change`: one sentence: which equation of the card differs from that approach, and how.
- `changed_equation`: where that change is in the card: `update[i]`, `readout[i]` (0-based) or `learning`.
- `cause_ids_addressed`: every outcome cause id (`oc-.../<kill code>`) the change answers. When the requirements list dead ideas under rule R3, every one of their cause ids must be here and `changed_equation` must be the equation that answers the cause. Naming a cause you do not change is not answering it.

When the user message contains a SEND-BACK, your first statement was rejected for the reasons given. Answer exactly what it asks: state what replaces the named part of the prior work and where in the card it is. If the card does not contain such a change, say so plainly in `precise_change`; the card is then dropped, which is the correct outcome.

Return only the JSON object matching the schema.
