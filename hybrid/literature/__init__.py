"""Paper search, caching, the seed reading list, and prior-art classification for the research loop.

The module-level functions use one shared `Literature` (cache under hybrid/literature/cache/).
Tests and callers that need another cache or transport construct `Literature` directly.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from hybrid.literature.client import (
    DEFAULT_SOURCES,
    LOCAL_SOURCE,
    Literature,
    Related,
    SearchReceipt,
    default_literature,
)
from hybrid.literature.events import EventSink
from hybrid.literature.paper import Paper
from hybrid.literature.prior_art import LLMComplete, classify_prior_art as _classify
from hybrid.literature.sampling import sample_for_lens
from hybrid.literature.sources.arxiv_local import DEFAULT_INDEX_PATH, resolve_index_path
from hybrid.literature.transport import SourceError

__all__ = [
    "DEFAULT_LOCAL_INDEX",
    "DEFAULT_SOURCES",
    "LOCAL_SOURCE",
    "Literature",
    "Paper",
    "Related",
    "SearchReceipt",
    "SourceError",
    "classify_prior_art",
    "get_paper",
    "literature_for",
    "related",
    "sample_for_lens",
    "resolve_index_path",
    "search",
]


@lru_cache(maxsize=None)
def literature_for(local_index_path: Path | None) -> Literature:
    """One shared `Literature` per local index path, so the transport's pacing and cooldowns persist
    across calls. `None` means no local index."""
    return default_literature(local_index_path)


DEFAULT_LOCAL_INDEX = resolve_index_path(DEFAULT_INDEX_PATH)


def _literature() -> Literature:
    return literature_for(DEFAULT_LOCAL_INDEX)


def search(
    query: str,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    limit: int = 10,
    sink: EventSink | None = None,
) -> list[Paper]:
    return _literature().search(query, sources, limit, sink)


def get_paper(source: str, source_id: str, sink: EventSink | None = None) -> Paper:
    return _literature().get_paper(source, source_id, sink)


def related(paper: Paper, sink: EventSink | None = None) -> Related:
    return _literature().related(paper, sink)


async def classify_prior_art(
    card_text: str,
    llm_complete: LLMComplete,
    sink: EventSink,
    *,
    local_index_path: Path | None = DEFAULT_LOCAL_INDEX,
) -> dict[str, Any]:
    """`local_index_path` (the run config's `literature.local_index_path`, resolved with
    `resolve_index_path`) adds the arxiv_local source ahead of the remote ones. A path whose file is
    missing is skipped and recorded in the search receipt. `None` searches the remote sources only."""
    return await _classify(literature_for(local_index_path), card_text, llm_complete, sink)
