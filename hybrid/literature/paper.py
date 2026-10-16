"""Paper records shared by all literature sources, and cross-source deduplication."""
from __future__ import annotations

import dataclasses
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

SOURCES = ("arxiv", "openalex", "s2")

_ARXIV_ID = re.compile(
    r"^(?:arxiv:)?((?:[a-z\-]+(?:\.[a-z\-]+)?/\d{7})|(?:\d{4}\.\d{4,5}))(?:v\d+)?$",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9]")


@dataclass(frozen=True)
class Paper:
    source: str
    source_id: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    abstract: str | None
    url: str
    venue: str | None
    citation_count: int | None
    fetched_at: str
    arxiv_id: str | None = None
    doi: str | None = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown literature source {self.source!r}")
        if not self.source_id or not self.title:
            raise ValueError(f"paper from {self.source} needs a source_id and a title")


def normalize_arxiv_id(raw: str) -> str | None:
    """Return the bare arXiv id (no 'arXiv:' prefix, no version), or None if not an arXiv id."""
    match = _ARXIV_ID.match(raw.strip())
    return match.group(1) if match else None


def paper_key_of(paper: Paper) -> str:
    return f"{paper.source}:{paper.source_id}"


def normalize_title(title: str) -> str:
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub("", folded.lower())


def paper_keys(paper: Paper) -> list[str]:
    keys = [f"title:{normalize_title(paper.title)}"]
    if paper.arxiv_id is not None:
        keys.insert(0, f"arxiv:{paper.arxiv_id}")
    return keys


def dedupe(papers: Iterable[Paper]) -> list[Paper]:
    """Merge records describing the same work. The first record wins; its empty fields
    are filled from later duplicates. Matching is by arXiv id or normalized title."""
    merged: list[Paper] = []
    index: dict[str, int] = {}
    for paper in papers:
        hit = next((index[key] for key in paper_keys(paper) if key in index), None)
        if hit is None:
            position = len(merged)
            merged.append(paper)
        else:
            position = hit
            merged[hit] = _fill_missing(merged[hit], paper)
        for key in paper_keys(merged[position]):
            index[key] = position
    return merged


def _fill_missing(kept: Paper, other: Paper) -> Paper:
    fills: dict[str, Any] = {}
    for field in ("year", "abstract", "venue", "citation_count", "arxiv_id", "doi"):
        if getattr(kept, field) is None and getattr(other, field) is not None:
            fills[field] = getattr(other, field)
    if not kept.authors and other.authors:
        fills["authors"] = other.authors
    return dataclasses.replace(kept, **fills) if fills else kept


def paper_to_dict(paper: Paper) -> dict[str, Any]:
    data = dataclasses.asdict(paper)
    data["authors"] = list(paper.authors)
    return data


def paper_from_dict(data: dict[str, Any]) -> Paper:
    fields = dict(data)
    fields["authors"] = tuple(fields["authors"])
    return Paper(**fields)
