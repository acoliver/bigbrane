"""Verification of a cited premise against the literature record the pipeline already has (stage C,
DISCOVERY-PIPELINE-PLAN.md section 4.3).

A premise the statement fixer marks ``testable_by: citation`` names a source: an arXiv id (looked up in the local
arXiv index, ``literature.local_index_path``) or an OpenAlex work id or lookup such as ``doi:10.48550/...``
(through ``hybrid.literature``, cached and paced). Stage C resolves every source before the challenger call and
shows the challenger the record it resolved to (title, abstract, year). The premise counts as ``cited`` only when

1. the source resolved,
2. the challenger says the record supports the premise as the card uses it, and
3. the sentence the challenger quotes for that is text of the resolved record (title or abstract).

Anything else leaves the premise open, with the reason recorded. Resolution is external I/O (a SQLite file, a web
API), so a lookup that cannot answer raises ``LookupFailed`` with the reason, and the stage records it.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable

import httpx

from hybrid.literature import SourceError, literature_for, resolve_index_path
from hybrid.literature.arxiv_index import ArxivIndexReader
from hybrid.literature.paper import normalize_arxiv_id
from hybrid.research.stages.falsify import SYMBOLS

MIN_RECORD_QUOTE = 20

Lookup = Callable[[str, str], dict]


class LookupFailed(RuntimeError):
    """A cited source could not be resolved in the literature record."""


def default_lookup(cfg: dict) -> Lookup:
    """``lookup(source, id)`` over the run's local arXiv index and OpenAlex. Returns {source, id, title, abstract,
    year, url, via}; raises LookupFailed."""
    index = ArxivIndexReader(resolve_index_path(cfg["literature"]["local_index_path"]))

    def lookup(source: str, ident: str) -> dict:
        if source == "arxiv":
            arxiv_id = normalize_arxiv_id(ident)
            if arxiv_id is None:
                raise LookupFailed(f"{ident!r} is not an arXiv id")
            if not index.exists():
                raise LookupFailed(f"the local arXiv index is not at {index.path}")
            try:
                record = index.get(arxiv_id)
            except sqlite3.Error as exc:
                raise LookupFailed(f"local arXiv index unreadable: {exc}") from exc
            if record is None:
                raise LookupFailed(f"arXiv {arxiv_id} is not in the local index")
            return {"source": "arxiv", "id": arxiv_id, "title": record.title, "abstract": record.abstract or "",
                    "year": int(record.created[:4]) if record.created else None,
                    "url": f"https://arxiv.org/abs/{arxiv_id}", "via": "arxiv_local"}
        if source == "openalex":
            try:
                paper = literature_for(None).get_paper("openalex", ident)
            except (SourceError, httpx.HTTPError, ValueError) as exc:
                raise LookupFailed(f"OpenAlex lookup of {ident!r} failed: {exc}") from exc
            return {"source": "openalex", "id": paper.source_id, "title": paper.title,
                    "abstract": paper.abstract or "", "year": paper.year, "url": paper.url, "via": "openalex"}
        raise ValueError(f"unknown citation source {source!r}")

    return lookup


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(SYMBOLS)).strip().lower()


def quote_in_record(quote: str, record: dict) -> bool:
    """The challenger's quote is text of the resolved record's title or abstract (whitespace and case
    normalized), and long enough to say something."""
    needle = _normal(quote).strip(" .;:'\"")
    return len(needle) >= MIN_RECORD_QUOTE and needle in _normal(f"{record['title']} {record['abstract']}")


def verify(premise_id: str, record: dict | None, lookup_problem: str | None, answer: dict | None) -> dict:
    """The verification of one cited premise: {premise_id, verified, evidence or reason, record, answer}."""
    base = {"premise_id": premise_id, "record": record, "answer": answer, "verified": False, "evidence": None,
            "reason": None}
    if record is None:
        return dict(base, reason=f"source not resolved: {lookup_problem}")
    if answer is None:
        return dict(base, reason="the challenger gave no verification for this source")
    if not answer["supports"]:
        return dict(base, reason=f"the challenger found the source does not support the premise: {answer['why']}")
    if not quote_in_record(answer["record_quote"], record):
        return dict(base, reason="the challenger's quote is not text of the resolved record's title or abstract")
    evidence = (f"{record['source']}:{record['id']} ({record['url']}, via {record['via']}): "
                f"\"{answer['record_quote'][:300]}\"")
    return dict(base, verified=True, evidence=evidence)
