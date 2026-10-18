"""Prior-art classification: model-proposed queries, recorded searches, model judgment grounded in hits."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Protocol

import jsonschema

from hybrid.literature.client import Literature
from hybrid.literature.events import EventSink
from hybrid.literature.paper import Paper, paper_key_of
from hybrid.literature.schemas import (CLASSIFICATION_SCHEMA, PRIOR_ART_LABELS, QUERY_FIELDS, QUERY_PROPOSAL_SCHEMA,
                                       REFINE_PROPOSAL_SCHEMA)

ROLE = "prior_art"
ACTOR = "literature"
RESULTS_PER_SOURCE_PER_QUERY = 6
ABSTRACT_CHARS_IN_PROMPT = 600

# PA2 limits. Round 1 is 10 to 14 queries; round 2 follows the references and citations of the SEED_PAPERS best
# round-1 hits (RELATED_PER_SEED of each) and runs at most 4 refined queries. The classifier sees the
# CLASSIFIER_CANDIDATES best-ranked papers with excerpts; the stored candidate list keeps every paper, best first.
SEED_PAPERS = 5
RELATED_PER_SEED = 6
ROUND2_TITLES_SHOWN = 30
CLASSIFIER_CANDIDATES = 60

QUERY_SYSTEM = (
    "You search the scientific literature for prior art on a proposed theory of sequence learning. "
    "Restate the mechanism in the vocabulary of other fields, because prior work is often published there under "
    "different names: " + ", ".join(QUERY_FIELDS) + ". Propose between 10 and 14 short keyword queries, each "
    "tagged with the field whose vocabulary it uses. Cover the core mechanism, each distinct step of the update "
    "and the readout, the claimed capability, and the closest known family of models, using at least four fields. "
    "Return JSON only."
)

REFINE_SYSTEM = (
    "You search the scientific literature for prior art on a proposed theory of sequence learning. You are shown "
    "the titles that a first round of searches returned. Propose up to 4 refined keyword queries that name "
    "closer prior work than the titles shown: a method, author or term that appears in them, or a neighbouring "
    "field they point to. Return an empty list when the titles already cover the mechanism. Return JSON only."
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
    tagged = _validated(proposal, QUERY_PROPOSAL_SCHEMA)["queries"]
    queries = [item["query"] for item in tagged]
    sink.append("prior_art_queries", ACTOR, {"queries": queries, "fields": [item["field"] for item in tagged]})

    pool = _Pool()
    for query in queries:
        await _search(literature, query, pool, sink)

    seeds = pool.best(SEED_PAPERS)
    for seed in seeds:
        related = await asyncio.to_thread(literature.related, seed, sink)
        pool.add_all((*related.references[:RELATED_PER_SEED], *related.citations[:RELATED_PER_SEED]))

    shown = "\n".join(f"- ({paper.year or 'n.d.'}) {paper.title}" for paper in pool.best(ROUND2_TITLES_SHOWN))
    refined = await llm_complete(ROLE, REFINE_SYSTEM, f"{card_text.strip()}\n\nTITLES FOUND SO FAR\n{shown}",
                                 REFINE_PROPOSAL_SCHEMA)
    round2 = _validated(refined, REFINE_PROPOSAL_SCHEMA)["queries"]
    sink.append("prior_art_round2_queries", ACTOR, {"queries": round2, "seeds": [paper_key_of(p) for p in seeds]})
    for query in round2:
        await _search(literature, query, pool, sink)

    candidates = pool.ranked()
    judgment = await llm_complete(ROLE, CLASSIFY_SYSTEM, _candidate_prompt(card_text, candidates), CLASSIFICATION_SCHEMA)
    verdict = _validated(judgment, CLASSIFICATION_SCHEMA)
    evidence, rejected = _ground_evidence(verdict["evidence"], candidates)

    result = {
        "label": verdict["label"],
        "evidence": evidence,
        "notes": verdict["notes"],
        "queries": queries,
        "query_fields": [item["field"] for item in tagged],
        "round2_queries": round2,
        "candidates": [_candidate_record(key, paper) for key, paper in candidates.items()],
        "rejected_evidence": rejected,
    }
    sink.append("prior_art_result", ACTOR, result)
    return result


class _Pool:
    """The papers found so far. Rank: the number of distinct searches that returned the paper (a paper several
    differently worded queries agree on is likely central), then the order of first discovery. Papers reached
    only through references and citations count as one hit."""

    def __init__(self) -> None:
        self.papers: dict[str, Paper] = {}
        self.hits: dict[str, int] = {}

    def add_all(self, papers: Any) -> None:
        for paper in papers:
            key = paper_key_of(paper)
            self.papers.setdefault(key, paper)
            self.hits[key] = self.hits.get(key, 0) + 1

    def ranked(self) -> dict[str, Paper]:
        order = {key: index for index, key in enumerate(self.papers)}
        keys = sorted(self.papers, key=lambda key: (-self.hits[key], order[key]))
        return {key: self.papers[key] for key in keys}

    def best(self, count: int) -> list[Paper]:
        return list(self.ranked().values())[:count]


async def _search(literature: Literature, query: str, pool: _Pool, sink: EventSink) -> None:
    receipt = await asyncio.to_thread(
        literature.search_with_receipt, query, literature.default_sources, RESULTS_PER_SOURCE_PER_QUERY, sink
    )
    pool.add_all(receipt.papers)


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
    papers outside the results are rejected and returned so the caller can see them.

    The results are listed to the model by key (`s2:712c...`), so it often answers with the key
    as the source_id. A source_id is taken as a key only when it starts with the cited source and
    names a returned paper of that source; no other prefix handling applies."""
    grounded, rejected = [], []
    for item in cited:
        paper = candidates.get(_cited_key(item["source"], item["source_id"]))
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


def _cited_key(source: str, source_id: str) -> str:
    prefix = f"{source}:"
    return source_id if source_id.startswith(prefix) else prefix + source_id


def _excerpt(paper: Paper) -> str:
    return (paper.abstract or "(no abstract)")[:ABSTRACT_CHARS_IN_PROMPT]


def _candidate_record(key: str, paper: Paper) -> dict[str, Any]:
    """What the later stages need to show a returned paper: its identity, title and the abstract excerpt the
    classifier saw. Records stay in the order the searches returned the papers."""
    return {
        "key": key,
        "source": paper.source,
        "source_id": paper.source_id,
        "title": paper.title,
        "year": paper.year,
        "url": paper.url,
        "excerpt": _excerpt(paper),
    }


def _candidate_prompt(card_text: str, candidates: dict[str, Paper]) -> str:
    shown = list(candidates)[:CLASSIFIER_CANDIDATES]
    lines = [
        "THEORY CARD",
        card_text.strip(),
        "",
        f"SEARCH RESULTS (the {len(shown)} best-ranked of {len(candidates)} unique papers, most agreed-on first). "
        f"Labels: {', '.join(PRIOR_ART_LABELS)}.",
    ]
    for key in shown:
        paper = candidates[key]
        year = paper.year if paper.year is not None else "n.d."
        lines.append(f"- [{key}] ({year}) {paper.title}\n  {_excerpt(paper)}")
    return "\n".join(lines)
