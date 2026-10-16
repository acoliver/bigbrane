"""OpenAlex works API (keyless; contact email is added by the client when configured)."""
from __future__ import annotations

import json
import re
from typing import Any

from hybrid.literature.paper import Paper, normalize_arxiv_id
from hybrid.literature.transport import Request

NAME = "openalex"
BASE_URL = "https://api.openalex.org"
SELECT = (
    "id,doi,title,display_name,publication_year,authorships,primary_location,"
    "cited_by_count,abstract_inverted_index,referenced_works"
)
_WORK_ID = re.compile(r"^W\d+$")
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_ARXIV_LANDING = re.compile(r"arxiv\.org/abs/(.+)$", re.IGNORECASE)


def search_request(query: str, limit: int) -> Request:
    return Request(f"{BASE_URL}/works", {"search": query, "per-page": str(limit), "select": SELECT})


def get_request(work_id: str) -> Request:
    """`work_id` is an OpenAlex id (W123...) or a lookup such as `doi:10.48550/arXiv.2312.00752`."""
    return Request(f"{BASE_URL}/works/{work_id}", {"select": SELECT})


def references_request(work_ids: list[str]) -> Request:
    if not 0 < len(work_ids) <= 100:
        raise ValueError("OpenAlex OR-filters take between 1 and 100 ids")
    return Request(
        f"{BASE_URL}/works",
        {"filter": "openalex:" + "|".join(work_ids), "per-page": "100", "select": SELECT},
    )


def citations_request(work_id: str, limit: int) -> Request:
    return Request(
        f"{BASE_URL}/works",
        {
            "filter": f"cites:{work_id}",
            "sort": "cited_by_count:desc",
            "per-page": str(limit),
            "select": SELECT,
        },
    )


def parse_search(body: str, fetched_at: str) -> list[Paper]:
    payload = _load(body)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("OpenAlex list response has no 'results' array")
    return [paper for paper in (parse_work(item, fetched_at) for item in results) if paper is not None]


def parse_get(body: str, fetched_at: str) -> Paper:
    paper = parse_work(_load(body), fetched_at)
    if paper is None:
        raise ValueError("OpenAlex work has no title")
    return paper


def parse_work(work: dict[str, Any], fetched_at: str) -> Paper | None:
    """Return None for records without a title, which OpenAlex does return for some withdrawn works."""
    title = work.get("title") or work.get("display_name")
    if not title:
        return None
    work_id = work_id_of(work)
    doi = _strip_doi_prefix(work.get("doi"))
    landing = ((work.get("primary_location") or {}).get("landing_page_url")) or None
    arxiv_id = _arxiv_id_from(doi, landing)
    source = (work.get("primary_location") or {}).get("source") or {}
    url = f"https://doi.org/{doi}" if doi else (landing or f"https://openalex.org/{work_id}")
    authors = tuple(
        authorship["author"]["display_name"]
        for authorship in work.get("authorships") or []
        if (authorship.get("author") or {}).get("display_name")
    )
    return Paper(
        source=NAME,
        source_id=work_id,
        title=" ".join(title.split()),
        authors=authors,
        year=work.get("publication_year"),
        abstract=rebuild_abstract(work.get("abstract_inverted_index")),
        url=url,
        venue=source.get("display_name") or None,
        citation_count=work.get("cited_by_count"),
        fetched_at=fetched_at,
        arxiv_id=arxiv_id,
        doi=doi,
    )


def work_id_of(work: dict[str, Any]) -> str:
    work_id = str(work.get("id", "")).rsplit("/", 1)[-1]
    if not _WORK_ID.match(work_id):
        raise ValueError(f"unexpected OpenAlex work id {work.get('id')!r}")
    return work_id


def referenced_work_ids(work: dict[str, Any]) -> list[str]:
    return [str(ref).rsplit("/", 1)[-1] for ref in work.get("referenced_works") or []]


def rebuild_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words: dict[int, str] = {}
    for word, positions in index.items():
        for position in positions:
            words[position] = word
    return " ".join(words[position] for position in sorted(words)) or None


def _strip_doi_prefix(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def _arxiv_id_from(doi: str | None, landing: str | None) -> str | None:
    if doi:
        match = _ARXIV_DOI.match(doi)
        if match:
            return normalize_arxiv_id(match.group(1))
    if landing:
        match = _ARXIV_LANDING.search(landing)
        if match:
            return normalize_arxiv_id(match.group(1))
    return None


def _load(body: str) -> dict[str, Any]:
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("OpenAlex response is not a JSON object")
    return payload
