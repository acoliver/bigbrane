"""Local arXiv index (SQLite FTS5, harvested through OAI-PMH) as a search source.

Papers it returns carry `source="arxiv"` and the arXiv id, the same identity the remote arXiv source
uses, so deduplication across sources and grounding of cited ids work unchanged. The source name
`arxiv_local` appears in receipts and per-source results.

Queries use the keyword form the prior-art step proposes: bare words, optionally `"quoted phrases"`.
Every term is matched against title and abstract, with stemming. Results with all terms come first;
when they number fewer than the limit, results with any term fill the rest, ranked by BM25.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hybrid.literature.arxiv_index import ArxivIndexReader, ArxivRecord, IndexInfo
from hybrid.literature.paper import Paper

NAME = "arxiv_local"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_PATH = "data/arxiv-index/arxiv.sqlite"

_TERM = re.compile(r'"([^"]*)"|([^\W_]+)')
_WORD = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it of on or that the this to using via with without".split()
)
_COLUMNS = "{title abstract}"


@dataclass(frozen=True)
class LocalResult:
    papers: tuple[Paper, ...]
    info: IndexInfo


def resolve_index_path(configured: str | Path) -> Path:
    """A configured path is absolute or relative to the repository root."""
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def match_expressions(query: str) -> list[str]:
    """FTS5 expressions for a keyword query: all terms first, then (if there are several) any term.
    Raises ValueError when the query has nothing to search for."""
    terms: list[str] = []
    for phrase, word in _TERM.findall(query):
        if phrase:
            words = [w.lower() for w in _WORD.findall(phrase)]
            if words:
                terms.append('"' + " ".join(words) + '"')
        elif word.lower() not in _STOPWORDS:
            terms.append(f'"{word.lower()}"')
    if not terms:
        # A query made only of stopwords still means something to the caller; search it as written.
        terms = [f'"{w.lower()}"' for w in _WORD.findall(query)]
    if not terms:
        raise ValueError("arxiv_local query has no searchable terms")
    unique = list(dict.fromkeys(terms))
    expressions = [f"{_COLUMNS} : ({' AND '.join(unique)})"]
    if len(unique) > 1:
        expressions.append(f"{_COLUMNS} : ({' OR '.join(unique)})")
    return expressions


class ArxivLocal:
    def __init__(self, path: Path) -> None:
        self._reader = ArxivIndexReader(path)

    @property
    def path(self) -> Path:
        return self._reader.path

    def exists(self) -> bool:
        return self._reader.exists()

    def search(self, query: str, limit: int, fallback_ts: str) -> LocalResult:
        """The index must exist; callers check `exists()` first. sqlite3.Error means the file is not a
        usable index and propagates."""
        expressions = match_expressions(query)
        info = self._reader.info()
        found: dict[str, ArxivRecord] = {}
        for expression in expressions:
            if len(found) >= limit:
                break
            for record in self._reader.search(expression, limit):
                found.setdefault(record.arxiv_id, record)
                if len(found) >= limit:
                    break
        fetched_at = info.snapshot_ts or fallback_ts
        return LocalResult(tuple(_to_paper(r, fetched_at) for r in found.values()), info)


def _to_paper(record: ArxivRecord, fetched_at: str) -> Paper:
    year = int(record.created[:4]) if record.created else None
    return Paper(
        source="arxiv",
        source_id=record.arxiv_id,
        title=record.title,
        authors=record.authors,
        year=year,
        abstract=record.abstract or None,
        url=f"https://arxiv.org/abs/{record.arxiv_id}",
        venue=record.journal_ref,
        citation_count=None,
        fetched_at=fetched_at,
        arxiv_id=record.arxiv_id,
        doi=record.doi,
    )
