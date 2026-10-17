"""Semantic Scholar graph API (keyless, strict shared rate limit; 429s are retried by the transport)."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from hybrid.literature.paper import Paper, normalize_arxiv_id
from hybrid.literature.transport import Request

NAME = "s2"
BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,externalIds,url,title,abstract,year,authors,venue,citationCount"


def search_request(query: str, limit: int) -> Request:
    return Request(f"{BASE_URL}/paper/search", {"query": query, "limit": str(limit), "fields": FIELDS})


def get_request(paper_id: str) -> Request:
    """`paper_id` is an S2 paperId or a prefixed id such as `ARXIV:2312.00752` or `DOI:10.x/y`."""
    return Request(f"{BASE_URL}/paper/{quote(paper_id, safe=':')}", {"fields": FIELDS})


def references_request(paper_id: str, limit: int) -> Request:
    return Request(
        f"{BASE_URL}/paper/{quote(paper_id, safe=':')}/references",
        {"fields": FIELDS, "limit": str(limit)},
    )


def citations_request(paper_id: str, limit: int) -> Request:
    return Request(
        f"{BASE_URL}/paper/{quote(paper_id, safe=':')}/citations",
        {"fields": FIELDS, "limit": str(limit)},
    )


def parse_search(body: str, fetched_at: str) -> list[Paper]:
    payload = _load(body)
    data = payload.get("data")
    if data is None and payload.get("total") == 0:
        # A search with no hits answers {"total": 0, "offset": 0} and omits "data" (seen 2026-10-17).
        return []
    if not isinstance(data, list):
        raise ValueError("Semantic Scholar search response has no 'data' array")
    return [paper for paper in (parse_paper(item, fetched_at) for item in data) if paper is not None]


def parse_references(body: str, fetched_at: str) -> list[Paper]:
    return _parse_edges(body, "citedPaper", fetched_at)


def parse_citations(body: str, fetched_at: str) -> list[Paper]:
    return _parse_edges(body, "citingPaper", fetched_at)


def parse_get(body: str, fetched_at: str) -> Paper:
    paper = parse_paper(_load(body), fetched_at)
    if paper is None:
        raise ValueError("Semantic Scholar paper has no title")
    return paper


def parse_paper(item: dict[str, Any], fetched_at: str) -> Paper | None:
    """Return None for records without a title or paperId. S2 returns null entries for unknown papers."""
    if not item or not item.get("title") or not item.get("paperId"):
        return None
    external = item.get("externalIds") or {}
    arxiv_raw = external.get("ArXiv")
    arxiv_id = normalize_arxiv_id(arxiv_raw) if arxiv_raw else None
    return Paper(
        source=NAME,
        source_id=item["paperId"],
        title=" ".join(item["title"].split()),
        authors=tuple(author["name"] for author in item.get("authors") or [] if author.get("name")),
        year=item.get("year"),
        abstract=item.get("abstract") or None,
        url=item.get("url") or f"https://www.semanticscholar.org/paper/{item['paperId']}",
        venue=item.get("venue") or None,
        citation_count=item.get("citationCount"),
        fetched_at=fetched_at,
        arxiv_id=arxiv_id,
        doi=external.get("DOI") or None,
    )


def _parse_edges(body: str, key: str, fetched_at: str) -> list[Paper]:
    data = _load(body).get("data")
    if not isinstance(data, list):
        raise ValueError("Semantic Scholar edge response has no 'data' array")
    papers = []
    for edge in data:
        paper = parse_paper(edge.get(key) or {}, fetched_at)
        if paper is not None:
            papers.append(paper)
    return papers


def _load(body: str) -> dict[str, Any]:
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("Semantic Scholar response is not a JSON object")
    return payload
