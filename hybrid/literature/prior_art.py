"""Prior-art classification: model-proposed queries, recorded searches, model judgment grounded in hits."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Protocol

import jsonschema

from hybrid.literature.client import Literature
from hybrid.literature.events import EventSink
from hybrid.literature.paper import Paper, paper_key_of
from hybrid.literature.schemas import CLASSIFICATION_SCHEMA, PRIOR_ART_LABELS, QUERY_PROPOSAL_SCHEMA

ROLE = "prior_art"
ACTOR = "literature"
RESULTS_PER_SOURCE_PER_QUERY = 6
ABSTRACT_CHARS_IN_PROMPT = 600

QUERY_SYSTEM = (
    "You search the scientific literature for prior art on a proposed theory of sequence learning. "
    "Propose between 3 and 5 short keyword queries for a literature search. Cover the core mechanism, "
    "the claimed capability, and the closest known family of models. Return JSON only."
)

CLASSIFY_SYSTEM = (
    "You judge a proposed theory of sequence learning against search results from the literature. "
    "Labels: novel = no found work addresses the mechanism; prior_art_tested = close prior work exists "
    "and its claimed capability was tested at a comparable scale; prior_art_undertested = close prior "
    "work exists but the claimed capability has little or no empirical testing; known_disproven = a found "
    "paper shows the central claim fails. Cite only papers listed in the results, by source and source_id. "
    "Return JSON only."
)


class LLMCompletion(Protocol):
    parsed: dict[str, Any] | None


LLMComplete = Callable[[str, str, str, dict], Awaitable[LLMCompletion]]


async def classify_prior_art(
    literature: Literature,
    card_text: str,
    llm_complete: LLMComplete,
    sink: EventSink,
) -> dict[str, Any]:
    proposal = await llm_complete(ROLE, QUERY_SYSTEM, card_text, QUERY_PROPOSAL_SCHEMA)
    queries = _validated(proposal, QUERY_PROPOSAL_SCHEMA)["queries"]
    sink.append("prior_art_queries", ACTOR, {"queries": queries})

    candidates: dict[str, Paper] = {}
    for query in queries:
        receipt = await asyncio.to_thread(
            literature.search_with_receipt, query, literature.default_sources, RESULTS_PER_SOURCE_PER_QUERY, sink
        )
        for paper in receipt.papers:
            candidates.setdefault(paper_key_of(paper), paper)

    judgment = await llm_complete(ROLE, CLASSIFY_SYSTEM, _candidate_prompt(card_text, candidates), CLASSIFICATION_SCHEMA)
    verdict = _validated(judgment, CLASSIFICATION_SCHEMA)
    evidence, rejected = _ground_evidence(verdict["evidence"], candidates)

    result = {
        "label": verdict["label"],
        "evidence": evidence,
        "notes": verdict["notes"],
        "queries": queries,
        "candidates": sorted(candidates),
        "rejected_evidence": rejected,
    }
    sink.append("prior_art_result", ACTOR, result)
    return result


def _validated(completion: LLMCompletion, schema: dict) -> dict[str, Any]:
    """Model output is an external boundary. A missing or off-schema answer stops the classification."""
    if completion.parsed is None:
        raise ValueError("prior-art model call returned no parsed JSON")
    jsonschema.validate(completion.parsed, schema)
    return completion.parsed


def _ground_evidence(
    cited: list[dict[str, Any]], candidates: dict[str, Paper]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only citations that name a paper the searches actually returned. Citations to
    papers outside the results are rejected and returned so the caller can see them."""
    grounded, rejected = [], []
    for item in cited:
        key = f"{item['source']}:{item['source_id']}"
        paper = candidates.get(key)
        if paper is None:
            rejected.append(item)
            continue
        grounded.append(
            {
                "source": paper.source,
                "source_id": paper.source_id,
                "title": paper.title,
                "url": paper.url,
                "relation": item["relation"],
            }
        )
    return grounded, rejected


def _candidate_prompt(card_text: str, candidates: dict[str, Paper]) -> str:
    lines = [
        "THEORY CARD",
        card_text.strip(),
        "",
        f"SEARCH RESULTS ({len(candidates)} unique papers). Labels: {', '.join(PRIOR_ART_LABELS)}.",
    ]
    for key in sorted(candidates):
        paper = candidates[key]
        abstract = (paper.abstract or "(no abstract)")[:ABSTRACT_CHARS_IN_PROMPT]
        year = paper.year if paper.year is not None else "n.d."
        lines.append(f"- [{key}] ({year}) {paper.title}\n  {abstract}")
    return "\n".join(lines)
