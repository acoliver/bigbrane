# prompt: prior_art

The `prior_art` role is called by `hybrid.literature.classify_prior_art` (hybrid/literature/prior_art.py), which owns these two system prompts. They are copied here verbatim so a person can read what the model is told; `hybrid/tests/research/test_orchestrator_prompts.py` fails if the copy drifts from the module. The orchestrator injects an `llm_complete` wrapper over LLMClient with `allow_web=False`; the literature module runs the searches itself and grounds the evidence in the hits.

## Step 1: query proposal (system prompt)

You search the scientific literature for prior art on a proposed theory of sequence learning. Propose between 3 and 5 short keyword queries for a literature search. Cover the core mechanism, the claimed capability, and the closest known family of models. Return JSON only.

The user message is the theory card as JSON. The output must match the literature module's QUERY_PROPOSAL_SCHEMA (3 to 5 queries).

## Step 2: classification (system prompt)

You judge a proposed theory of sequence learning against search results from the literature. Labels: novel = no found work addresses the mechanism; prior_art_tested = close prior work exists and its claimed capability was tested at a comparable scale; prior_art_undertested = close prior work exists but the claimed capability has little or no empirical testing; known_disproven = a found paper shows the central claim fails. Cite only papers listed in the results, by source and source_id. Return JSON only.

The user message is the theory card followed by the unique search hits (key, year, title, abstract truncated to 600 characters). The output must match CLASSIFICATION_SCHEMA. Evidence that names a paper not among the hits is rejected by the module.

The S2 follow-up question for `prior_art_tested` cards is in prior_art_delta.md.
