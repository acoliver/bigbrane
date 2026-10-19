"""Prior-art classification: model-proposed queries, recorded searches, model judgment grounded in hits."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any, Awaitable, Callable, Protocol

import jsonschema

from hybrid.literature.client import Literature
from hybrid.literature.events import EventSink
from hybrid.literature.paper import Paper, normalize_arxiv_id, normalize_title, paper_key_of
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
    searches, related_receipts = [], []
    for item in tagged:
        searches.append(await _search(literature, item["query"], pool, sink, field=item["field"], round=1))

    seeds = pool.best(SEED_PAPERS)
    for seed in seeds:
        related = await asyncio.to_thread(literature.related, seed, sink)
        pool.add_all((*related.references[:RELATED_PER_SEED], *related.citations[:RELATED_PER_SEED]))
        related_receipts.append({
            "paper": related.paper_key, "via": related.via, "errors": dict(related.errors),
            "references_returned": len(related.references), "citations_returned": len(related.citations),
            "reference_keys_used": [paper_key_of(p) for p in related.references[:RELATED_PER_SEED]],
            "citation_keys_used": [paper_key_of(p) for p in related.citations[:RELATED_PER_SEED]],
            "references_used": min(len(related.references), RELATED_PER_SEED),
            "citations_used": min(len(related.citations), RELATED_PER_SEED),
            "truncated": len(related.references) > RELATED_PER_SEED or len(related.citations) > RELATED_PER_SEED,
        })

    first_round = [dict(_candidate_record(key, paper), **pool.provenance(key))
                   for key, paper in pool.ranked().items()]
    shown = "\n".join(f"- ({item['year'] or 'n.d.'}) {item['title']}"
                      for item in select_candidates(first_round, ROUND2_TITLES_SHOWN))
    refined = await llm_complete(ROLE, REFINE_SYSTEM, f"{card_text.strip()}\n\nTITLES FOUND SO FAR\n{shown}",
                                 REFINE_PROPOSAL_SCHEMA)
    round2 = _validated(refined, REFINE_PROPOSAL_SCHEMA)["queries"]
    sink.append("prior_art_round2_queries", ACTOR, {"queries": round2, "seeds": [paper_key_of(p) for p in seeds]})
    repeated = [query for query in round2 if _query_key(query) in {_query_key(q) for q in queries}]
    round2 = [query for query in round2 if query not in repeated]
    for query in round2:
        searches.append(await _search(literature, query, pool, sink, round=2))

    candidates = pool.ranked()
    records = [dict(_candidate_record(key, paper), **pool.provenance(key), rank=rank)
               for rank, (key, paper) in enumerate(candidates.items(), 1)]
    shown = select_candidates(records, CLASSIFIER_CANDIDATES)
    positions = {item["key"]: position for position, item in enumerate(shown, 1)}
    for item in records:
        item["classifier_position"] = positions.get(item["key"])
    coverage = _coverage(searches, related_receipts)
    listing = listing_receipt(records, list(positions), CLASSIFIER_CANDIDATES)
    judgment = await llm_complete(ROLE, CLASSIFY_SYSTEM,
                                 _candidate_prompt(card_text, candidates, shown, coverage, listing), CLASSIFICATION_SCHEMA)
    verdict = _validated(judgment, CLASSIFICATION_SCHEMA)
    aliases = {alias: candidates[key] for key in candidates for alias in pool.aliases[key]}
    evidence, rejected = _ground_evidence(verdict["evidence"], aliases)
    reasons = list(coverage["inadequate_reasons"])
    if not candidates:
        reasons.append("No candidates returned.")
    if not verdict["coverage_assessment"]["adequate_for_novelty"]:
        reasons.append(verdict["coverage_assessment"]["reason"])
    unassessed = verdict["label"] == "novel" and bool(reasons)

    result = {
        "label": verdict["label"],
        "evidence": evidence,
        "notes": verdict["notes"],
        "queries": queries,
        "query_fields": [item["field"] for item in tagged],
        "round2_queries": round2,
        "candidates": records,
        "coverage": coverage,
        "listing": listing,
        "coverage_assessment": verdict["coverage_assessment"],
        "novelty_unassessed": unassessed,
        "novelty_unassessed_reasons": reasons if unassessed else [],
        "duplicate_round2_queries": repeated,
        "rejected_evidence": rejected,
    }
    sink.append("prior_art_result", ACTOR, result)
    return result


class _Pool:
    """Rank by distinct normalized query support, then discovery order. Related-only support is one.
    Identity aliases and richer metadata survive later searches and repeated reference fanout."""

    def __init__(self) -> None:
        self.papers: dict[str, Paper] = {}
        self.queries: dict[str, set[str]] = {}
        self.fields: dict[str, set[str]] = {}
        self.aliases: dict[str, set[str]] = {}
        self.identities: dict[str, str] = {}

    def add_all(self, papers: Any, *, query: str | None = None, field: str | None = None) -> None:
        for paper in papers:
            identities = _identities(paper)
            matches = {self.identities[i] for i in identities if i in self.identities}
            key = next((k for k in self.papers if k in matches), paper_key_of(paper))
            if key not in self.papers:
                self.papers[key] = paper
                self.queries[key], self.fields[key], self.aliases[key] = set(), set(), set()
            for other in matches - {key}:
                self.papers[key] = _richer(self.papers[key], self.papers.pop(other))
                self.queries[key].update(self.queries.pop(other))
                self.fields[key].update(self.fields.pop(other))
                self.aliases[key].update(self.aliases.pop(other))
                for identity, owner in list(self.identities.items()):
                    if owner == other:
                        self.identities[identity] = key
            self.papers[key] = _richer(self.papers[key], paper)
            self.aliases[key].add(paper_key_of(paper))
            for identity in identities:
                self.identities[identity] = key
            if query is not None:
                self.queries[key].add(_query_key(query))
            if field is not None:
                self.fields[key].add(field)

    def provenance(self, key: str) -> dict:
        return {"query_support": sorted(self.queries[key]), "query_hits": len(self.queries[key]),
                "query_fields": sorted(self.fields[key]), "aliases": sorted(self.aliases[key]),
                "related_only": not self.queries[key]}

    def ranked(self) -> dict[str, Paper]:
        keys = sorted(self.papers, key=lambda key: -max(1, len(self.queries[key])))
        return {key: self.papers[key] for key in keys}

    def best(self, count: int) -> list[Paper]:
        return list(self.ranked().values())[:count]


def _query_key(query: str) -> str:
    return " ".join(query.casefold().split())


def _identities(paper: Paper) -> set[str]:
    keys = {paper_key_of(paper), "title:" + normalize_title(paper.title)}
    arxiv_id = normalize_arxiv_id(paper.arxiv_id or (paper.source_id if paper.source == "arxiv" else ""))
    if arxiv_id:
        keys.add("arxiv:" + arxiv_id)
    if paper.doi:
        keys.add("doi:" + paper.doi.lower().removeprefix("https://doi.org/"))
    return keys


def _richer(kept: Paper, other: Paper) -> Paper:
    changes = {name: getattr(other, name) for name in ("year", "venue", "citation_count", "arxiv_id", "doi")
               if getattr(kept, name) is None and getattr(other, name) is not None}
    if len(other.abstract or "") > len(kept.abstract or ""):
        changes["abstract"] = other.abstract
    if not kept.authors and other.authors:
        changes["authors"] = other.authors
    if paper_key_of(kept) == paper_key_of(other) and len(other.title) > len(kept.title):
        changes["title"] = other.title
    return replace(kept, **changes)


async def _search(literature: Literature, query: str, pool: _Pool, sink: EventSink,
                  *, field: str | None = None, round: int) -> dict:
    receipt = await asyncio.to_thread(
        literature.search_with_receipt, query, literature.default_sources, RESULTS_PER_SOURCE_PER_QUERY, sink
    )
    pool.add_all(receipt.papers, query=query, field=field)
    return dict(receipt.event_data(), field=field, round=round)


def _coverage(searches: list[dict], related: list[dict]) -> dict:
    missing = []
    fields: dict[str, dict] = {}
    for search in searches:
        successes = [source for source in search["sources"] if source not in search["errors"]]
        usable = [source for source in successes if source != "arxiv_local"
                  or (search.get("local_index") or {}).get("complete") is True]
        search["successful_sources"], search["usable_sources"] = successes, usable
        if not usable:
            missing.append(search["query"])
        if search["field"]:
            entry = fields.setdefault(search["field"], {"queries": [], "answered_queries": [], "hit_queries": []})
            entry["queries"].append(search["query"])
            if usable:
                entry["answered_queries"].append(search["query"])
            if search["result_count"]:
                entry["hit_queries"].append(search["query"])
    return {"searches": searches, "related": related, "fields": fields, "unanswered_queries": missing,
            "inadequate_reasons": ["No usable source answered any query in field: " + field
                                   for field, entry in fields.items() if not entry["answered_queries"]],
            "limits": "Bounded searches and excerpts do not establish complete novelty. Successful empty results "
                      "record a completed query, not proof that no prior work exists."}


def select_candidates(candidates: list[dict], limit: int) -> list[dict]:
    """Reserve the highest-ranked representative of each queried field, then fill by query rank.
    This keeps a single field's many hits from occupying every bounded decision slot."""
    selected: dict[str, dict] = {}
    fields = dict.fromkeys(field for item in candidates for field in item.get("query_fields", []))
    for field in fields:
        if len(selected) >= limit:
            break
        item = next(item for item in candidates if field in item.get("query_fields", []))
        selected[item["key"]] = item
    for item in candidates:
        if len(selected) >= limit:
            break
        selected.setdefault(item["key"], item)
    return [item for item in candidates if item["key"] in selected]


def listing_receipt(candidates: list[dict], shown_keys: list[str], limit: int) -> dict:
    positions = {key: position for position, key in enumerate(shown_keys, 1)}
    return {"total_candidates": len(candidates), "limit": limit, "shown_count": len(shown_keys),
            "truncated": len(shown_keys) < len(candidates),
            "shown": [{"key": item["key"], "rank": item.get("rank", rank),
                       "position": positions[item["key"]], "query_fields": item.get("query_fields", [])}
                      for rank, item in enumerate(candidates, 1) if item["key"] in positions],
            "omitted_keys": [item["key"] for item in candidates if item["key"] not in positions]}


def render_receipt(result: dict, shown_keys: list[str], limit: int) -> str:
    return "SEARCH COVERAGE AND LISTING RECEIPT\n" + json.dumps({
        "coverage": result.get("coverage", {"status": "missing; coverage was not recorded"}),
        "listing": listing_receipt(result["candidates"], shown_keys, limit),
        "coverage_assessment": result.get("coverage_assessment"),
        "novelty_unassessed": result.get("novelty_unassessed"),
        "novelty_unassessed_reasons": result.get("novelty_unassessed_reasons", []),
    }, ensure_ascii=False)


def _validated(completion: LLMCompletion, schema: dict) -> dict[str, Any]:
    """Model output is an external boundary. A missing or off-schema answer stops the classification."""
    if completion.parsed is None:
        raise ValueError("prior-art model call returned no parsed JSON")
    jsonschema.validate(completion.parsed, schema)
    if schema is QUERY_PROPOSAL_SCHEMA or schema is REFINE_PROPOSAL_SCHEMA:
        items = completion.parsed["queries"]
        queries = [item["query"] for item in items] if schema is QUERY_PROPOSAL_SCHEMA else items
        normalized = [_query_key(query) for query in queries]
        if any(len(query) < 3 for query in normalized) or len(set(normalized)) != len(normalized):
            raise jsonschema.ValidationError("Queries must be distinct after normalizing case and whitespace")
        if schema is QUERY_PROPOSAL_SCHEMA and len({item["field"] for item in items}) < 4:
            raise jsonschema.ValidationError("Query proposal must cover at least four distinct fields")
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


def _candidate_prompt(card_text: str, candidates: dict[str, Paper], shown: list[dict],
                      coverage: dict, listing: dict) -> str:
    lines = [
        "THEORY CARD", card_text.strip(), "",
        "SEARCH COVERAGE AND LISTING RECEIPT", json.dumps({"coverage": coverage, "listing": listing}),
        "Assess whether the completed queries, available sources and shown excerpts cover the mechanism's "
        "distinct update and readout steps. Return coverage_assessment with adequate_for_novelty and a reason. "
        "An unrelated hit alone does not make coverage adequate. A source failure alone does not make coverage "
        "inadequate when other usable searches cover the mechanism. Unshown candidates have not been assessed. "
        "The novel label means no prior work found within this search, never complete novelty.",
        f"SEARCH RESULTS (the {len(shown)} best-ranked of {len(candidates)} unique papers, with field representatives). "
        f"Labels: {', '.join(PRIOR_ART_LABELS)}.",
    ]
    for item in shown:
        paper = candidates[item["key"]]
        year = paper.year if paper.year is not None else "n.d."
        lines.append(f"- [{item['key']}] ({year}) {paper.title}\n  {_excerpt(paper)}")
    return "\n".join(lines)
