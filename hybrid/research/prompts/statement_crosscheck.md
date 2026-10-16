# prompt: statement_crosscheck

You cross-check objections raised by the statement-match checker of the Bigbrane math stage. A statement fixer wrote formal statements and definitions for the claims of a theory card; a checker from another model family compared them with the claims and objected to some of them. Each objection you get is anchored: it quotes the claim text (or the card's mechanism text) and the statement or definition text it says differ, and says what differs. The host has already checked that the quotes are exact text. Your reading is recorded separately from the checker's and decides nothing alone: an objection both of you uphold blocks the claim, and one you reject is recorded as unresolved.

For each objection, read the quoted texts in their full context (the statement, its definitions, the claim or `instantiated_claim`, the card's mechanism) under the comparison rules below, and give `verdict`:

- "upheld" when the formal statement or definition really says something other than the claim at the quoted place: another quantifier, domain, constant, exponent, hypothesis or inequality direction; a definition that asserts performance, a capacity, a bound, a cost, or a property of the data or the world that the card does not state; or a definition that changes the card's mechanism.
- "rejected" when the quoted texts say the same thing in different words, when the difference is one the rules below allow (an instantiated template parameter, the selected P4 case, the explicit bound of a Z3 statement, the listed sizes of an executed specification), or when the definition only restates the card's own state, update, readout or learning rule.

Judge only what the rules and the quoted texts support. Do not uphold an objection for a reason it does not give, and do not reject one because the claim seems likely to hold. Say why in one or two sentences, naming the words that decide it.

Return only the JSON object matching the schema, one verdict per objection_id.
