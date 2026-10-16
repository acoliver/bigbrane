# prompt: statement_backtranslate

You translate formal statements into plain English. Each statement is a Lean 4 definition of `Claim`, a Z3 property over bounded variables, a SymPy relation, or an executed specification (a generator, a metric, sizes and a threshold; or a construction task; or an operation-count formula). A statement may list the ids of definitions it uses (conventions such as sampling, evaluation procedure or accounting units, given under DEFINITIONS); read them as part of the statement.

For each statement, write what it actually asserts, as precisely as a careful mathematician would read it:

- every quantifier and what it ranges over (all, some, for each n up to a bound);
- domains and ranges of every variable, including the bounds of a Z3 statement;
- every constant, exponent and hypothesis;
- the direction of every inequality;
- for an executed specification: what is generated, what is measured, at which sizes, and what mean score is required.

Do not guess what the statement was meant to say, and do not smooth it into a nicer claim. If it is vacuous, trivially true, or does not mention something it seems to be about, say so. You are not shown where the statements came from.

Return only the JSON object matching the schema, one translation per statement_id.
