"""arXiv export API (Atom feeds)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from hybrid.literature.paper import Paper, normalize_arxiv_id
from hybrid.literature.transport import Request

NAME = "arxiv"
BASE_URL = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ABS_URL = re.compile(r"arxiv\.org/abs/(.+)$")


def search_request(query: str, limit: int) -> Request:
    terms = re.findall(r"[A-Za-z0-9]+", query)
    if not terms:
        raise ValueError("arXiv query has no searchable terms")
    return Request(
        BASE_URL,
        {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "start": "0",
            "max_results": str(limit),
            "sortBy": "relevance",
        },
    )


def get_request(source_id: str) -> Request:
    arxiv_id = normalize_arxiv_id(source_id)
    if arxiv_id is None:
        raise ValueError(f"not an arXiv id: {source_id!r}")
    return Request(BASE_URL, {"id_list": arxiv_id, "max_results": "1"})


def parse_search(body: str, fetched_at: str) -> list[Paper]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError(f"malformed Atom feed: {exc}") from exc
    papers = []
    for entry in root.findall("atom:entry", _NS):
        entry_id = _text(entry, "atom:id")
        if "/api/errors" in entry_id:
            raise ValueError(f"arXiv API error: {_text(entry, 'atom:summary')}")
        match = _ABS_URL.search(entry_id)
        if match is None:
            raise ValueError(f"unexpected arXiv entry id {entry_id!r}")
        arxiv_id = normalize_arxiv_id(match.group(1))
        if arxiv_id is None:
            raise ValueError(f"unexpected arXiv entry id {entry_id!r}")
        title = _collapse(_text(entry, "atom:title"))
        if not title:
            continue
        published = _text(entry, "atom:published")
        doi = _text(entry, "arxiv:doi") or None
        journal_ref = _text(entry, "arxiv:journal_ref") or None
        papers.append(
            Paper(
                source=NAME,
                source_id=arxiv_id,
                title=title,
                authors=tuple(
                    _collapse(name.text or "") for name in entry.findall("atom:author/atom:name", _NS)
                ),
                year=int(published[:4]) if published else None,
                abstract=_collapse(_text(entry, "atom:summary")) or None,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                venue=journal_ref,
                citation_count=None,
                fetched_at=fetched_at,
                arxiv_id=arxiv_id,
                doi=doi,
            )
        )
    return papers


def parse_get(body: str, fetched_at: str) -> Paper:
    papers = parse_search(body, fetched_at)
    if not papers:
        raise ValueError("arXiv returned no entry for this id")
    return papers[0]


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path, _NS)
    return (found.text or "") if found is not None else ""


def _collapse(text: str) -> str:
    return " ".join(text.split())
