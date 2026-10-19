# prompt: prior_art

The `prior_art` role is called by `hybrid.literature.classify_prior_art` (hybrid/literature/prior_art.py), which owns these three system prompts. They are copied here verbatim so a person can read what the model is told; `hybrid/tests/research/test_orchestrator_prompts.py` fails if the copy drifts from the module. The orchestrator injects an `llm_complete` wrapper over LLMClient with `allow_web=False`; the literature module runs the searches itself and grounds the evidence in the hits.

## Step 1: query proposal, round 1 (system prompt)

You search the scientific literature for prior art on a proposed theory of sequence learning. Restate the mechanism in the vocabulary of other fields, because prior work is often published there under different names: automata_formal_languages, compression_coding, cognitive_bayesian, physics_dynamical_systems, optimization, program_synthesis, associative_memory, signal_processing. Propose between 10 and 14 short keyword queries, each tagged with the field whose vocabulary it uses. Cover the core mechanism, each distinct step of the update and the readout, the claimed capability, and the closest known family of models, using at least four fields. Return JSON only.

The user message is the theory card as JSON. The output must match the literature module's QUERY_PROPOSAL_SCHEMA (10 to 14 queries, each tagged with one of the eight fields). The module runs every query, ranks the papers by how many distinct queries returned them, follows the references and citations of the five best, and shows the model the titles it has so far.

## Step 1b: refined queries, round 2 (system prompt)

You search the scientific literature for prior art on a proposed theory of sequence learning. You are shown the titles that a first round of searches returned. Propose up to 4 refined keyword queries that name closer prior work than the titles shown: a method, author or term that appears in them, or a neighbouring field they point to. Return an empty list when the titles already cover the mechanism. Return JSON only.

The output must match REFINE_PROPOSAL_SCHEMA (0 to 4 queries).

## Step 2: classification (system prompt)

You judge a proposed theory of sequence learning against search results from the literature. Labels: novel = no found work addresses the mechanism; prior_art_tested = close prior work exists and its claimed capability was tested at a comparable scale; prior_art_undertested = close prior work exists but the claimed capability has little or no empirical testing; known_disproven = a found paper shows the central claim fails. Cite only papers listed in the results, by source and source_id. Return JSON only.

The user message is the theory card followed by the 60 best-ranked unique search hits (key, year, title, abstract truncated to 600 characters). The output must match CLASSIFICATION_SCHEMA. Evidence that names a paper not among the hits is rejected by the module.

The S2 follow-up question for `prior_art_tested` cards is in prior_art_delta.md.
