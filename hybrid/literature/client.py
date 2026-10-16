"""Search, lookup, and citation-graph access over arXiv, OpenAlex and Semantic Scholar."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx

from hybrid.literature import sources
from hybrid.literature.cache import ResponseCache
from hybrid.literature.events import EventSink, utc_now
from hybrid.literature.keys import ApiKey, load_api_keys
from hybrid.literature.paper import SOURCES, Paper, dedupe, paper_key_of
from hybrid.literature.sources.arxiv_local import ArxivLocal
from hybrid.literature.transport import Fetched, Pacer, Request, SourceError, Transport
from hybrid.slots import SLOTS_DIR

ACTOR = "literature"
DEFAULT_SOURCES = ("arxiv", "openalex", "s2")
LOCAL_SOURCE = "arxiv_local"
SEARCH_SOURCES = (LOCAL_SOURCE, *SOURCES)
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / "cache"
CONTACT_ENV = "BIGBRANE_CONTACT_EMAIL"
RELATED_SOURCES = ("openalex", "s2")
RELATED_REFERENCE_LIMIT = 100
RELATED_CITATION_LIMIT = 50

_LOG = logging.getLogger(__name__)
# Production pacing: the last-request times live in the workspace's var/slots, so the spacing holds across
# every process of a research run (calibrate --parallel, planted --parallel), not only within this one.
PROCESS_PACER = Pacer(time.time, slots_dir=SLOTS_DIR)


@dataclass(frozen=True)
class SearchReceipt:
    query: str
    sources: tuple[str, ...]
    limit: int
    ts: str
    per_source: dict[str, tuple[Paper, ...]]
    errors: dict[str, str]
    papers: tuple[Paper, ...]
    # Snapshot facts about the local index when arxiv_local was searched: path, record_count,
    # snapshot_date, complete. None when arxiv_local was not requested.
    local_index: dict[str, Any] | None = None

    def event_data(self) -> dict[str, Any]:
        data = {
            "query": self.query,
            "sources": list(self.sources),
            "limit": self.limit,
            "ts": self.ts,
            "per_source_ids": {
                source: [paper.source_id for paper in found] for source, found in self.per_source.items()
            },
            "errors": dict(self.errors),
            "result_count": len(self.papers),
        }
        if self.local_index is not None:
            data["local_index"] = dict(self.local_index)
        return data


@dataclass(frozen=True)
class Related:
    paper_key: str
    via: str | None
    references: tuple[Paper, ...]
    citations: tuple[Paper, ...]
    errors: dict[str, str] = field(default_factory=dict)


class Literature:
    """Holds the cache, transport and configuration. Tests construct it with a mocked httpx client."""

    def __init__(
        self,
        *,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        now: Callable[[], str] = utc_now,
        contact_email: str | None = None,
        max_retries: int = 3,
        local_index_path: Path | None = None,
        api_keys: Mapping[str, ApiKey] | None = None,
        pacer: Pacer | None = None,
    ) -> None:
        http = http or httpx.Client(timeout=30.0, follow_redirects=True)
        transport_kwargs: dict[str, Any] = {"now": now, "max_retries": max_retries, "api_keys": api_keys, "pacer": pacer}
        if sleep is not None:
            transport_kwargs["sleep"] = sleep
        if clock is not None:
            transport_kwargs["clock"] = clock
        self._transport = Transport(ResponseCache(Path(cache_root)), http, **transport_kwargs)
        self._now = now
        self._local = ArxivLocal(Path(local_index_path)) if local_index_path is not None else None
        # The contact address is sent only when given, and only to OpenAlex. OpenAlex has required an
        # API key since February 2026 and ignores `mailto`, so this is harmless but no longer useful.
        # The key itself goes in a header (see Transport), never in these query parameters.
        self._extra_params: dict[str, dict[str, str]] = {
            "openalex": {"mailto": contact_email} if contact_email else {}
        }

    @property
    def default_sources(self) -> tuple[str, ...]:
        """The sources a prior-art search uses: the local index first when one is configured."""
        return (LOCAL_SOURCE, *DEFAULT_SOURCES) if self._local is not None else DEFAULT_SOURCES

    # Search

    def search(
        self,
        query: str,
        sources: tuple[str, ...] = DEFAULT_SOURCES,
        limit: int = 10,
        sink: EventSink | None = None,
    ) -> list[Paper]:
        return list(self.search_with_receipt(query, sources, limit, sink).papers)

    def search_with_receipt(
        self,
        query: str,
        sources: tuple[str, ...] = DEFAULT_SOURCES,
        limit: int = 10,
        sink: EventSink | None = None,
    ) -> SearchReceipt:
        _check_sources(sources)
        if not query.strip():
            raise ValueError("search query is empty")
        if limit < 1:
            raise ValueError("search limit must be at least 1")
        ts = self._now()
        per_source: dict[str, tuple[Paper, ...]] = {}
        errors: dict[str, str] = {}
        local_index: dict[str, Any] | None = None
        for source in sources:
            try:
                if source == LOCAL_SOURCE:
                    found, local_index = self._search_local(query, limit, ts)
                    per_source[source] = found
                else:
                    per_source[source] = tuple(self._search_source(source, query, limit))
            except SourceError as exc:
                per_source[source] = ()
                errors[source] = exc.reason
                if source == LOCAL_SOURCE:
                    missing = self._local is None or not self._local.exists()
                    local_index = {"path": self._local_path_text(), "status": "missing" if missing else "unreadable"}
        # Sources are merged in the order requested, so the local index's records win ties.
        papers = tuple(dedupe(paper for source in sources for paper in per_source[source]))
        receipt = SearchReceipt(query, tuple(sources), limit, ts, per_source, errors, papers, local_index)
        if errors:
            _LOG.warning("literature search %r: source errors %s", query, errors)
        if sink is not None:
            sink.append("literature_search", ACTOR, receipt.event_data())
        return receipt

    def _search_local(self, query: str, limit: int, ts: str) -> tuple[tuple[Paper, ...], dict[str, Any]]:
        if self._local is None:
            raise SourceError(LOCAL_SOURCE, "", "skipped: no local index is configured")
        if not self._local.exists():
            raise SourceError(LOCAL_SOURCE, str(self._local.path), f"skipped: local index not found at {self._local.path}")
        try:
            result = self._local.search(query, limit, ts)
        except sqlite3.Error as exc:
            raise SourceError(LOCAL_SOURCE, str(self._local.path), f"local index unreadable: {exc}") from exc
        except ValueError as exc:
            raise SourceError(LOCAL_SOURCE, str(self._local.path), str(exc)) from exc
        info = result.info
        return result.papers, {
            "path": info.path,
            "status": "ok",
            "record_count": info.record_count,
            "snapshot_date": info.snapshot_date,
            "complete": info.complete,
        }

    def _local_path_text(self) -> str | None:
        return str(self._local.path) if self._local is not None else None

    def _search_source(self, source: str, query: str, limit: int) -> list[Paper]:
        module = sources.MODULES[source]
        fetched = self._fetch(source, module.search_request(query, limit))
        return _parse(source, fetched, module.parse_search)

    # Lookup

    def get_paper(self, source: str, source_id: str, sink: EventSink | None = None) -> Paper:
        _check_sources((source,))
        module = sources.MODULES[source]
        fetched = self._fetch(source, module.get_request(source_id))
        paper = _parse(source, fetched, module.parse_get)
        if sink is not None:
            sink.append(
                "literature_get",
                ACTOR,
                {"source": source, "source_id": source_id, "ts": self._now(), "from_cache": fetched.from_cache},
            )
        return paper

    # Citation graph

    def related(self, paper: Paper, sink: EventSink | None = None) -> Related:
        """Best-effort references and citations. OpenAlex is tried first, then Semantic Scholar.
        Each failed attempt is recorded in `errors`; the result is empty when neither answers."""
        errors: dict[str, str] = {}
        result: Related | None = None
        for source in RELATED_SOURCES:
            try:
                if source == "openalex":
                    references, citations = self._related_openalex(paper)
                else:
                    references, citations = self._related_s2(paper)
            except SourceError as exc:
                errors[source] = exc.reason
                continue
            result = Related(paper_key_of(paper), source, references, citations, errors)
            break
        if result is None:
            result = Related(paper_key_of(paper), None, (), (), errors)
        if sink is not None:
            sink.append(
                "literature_related",
                ACTOR,
                {
                    "paper": result.paper_key,
                    "via": result.via,
                    "reference_ids": [p.source_id for p in result.references],
                    "citation_ids": [p.source_id for p in result.citations],
                    "errors": dict(errors),
                    "ts": self._now(),
                },
            )
        return result

    def _related_openalex(self, paper: Paper) -> tuple[tuple[Paper, ...], tuple[Paper, ...]]:
        module = sources.openalex
        if paper.source == "openalex":
            lookup = paper.source_id
        else:
            lookup = _openalex_lookup(paper)
        fetched = self._fetch("openalex", module.get_request(lookup))
        work = _load_json("openalex", fetched, dict)
        work_id = module.work_id_of(work)
        referenced = module.referenced_work_ids(work)[:RELATED_REFERENCE_LIMIT]
        references: tuple[Paper, ...] = ()
        if referenced:
            ref_fetch = self._fetch("openalex", module.references_request(referenced))
            references = tuple(_parse("openalex", ref_fetch, module.parse_search))
        cite_fetch = self._fetch("openalex", module.citations_request(work_id, RELATED_CITATION_LIMIT))
        citations = tuple(_parse("openalex", cite_fetch, module.parse_search))
        return references, citations

    def _related_s2(self, paper: Paper) -> tuple[tuple[Paper, ...], tuple[Paper, ...]]:
        module = sources.semantic_scholar
        lookup = _s2_lookup(paper)
        ref_fetch = self._fetch("s2", module.references_request(lookup, RELATED_REFERENCE_LIMIT))
        references = tuple(_parse("s2", ref_fetch, module.parse_references))
        cite_fetch = self._fetch("s2", module.citations_request(lookup, RELATED_CITATION_LIMIT))
        citations = tuple(_parse("s2", cite_fetch, module.parse_citations))
        return references, citations

    # Plumbing

    def _fetch(self, source: str, request: Request) -> Fetched:
        extra = self._extra_params.get(source, {})
        params = {**request.params, **extra}
        return self._transport.get(source, Request(request.url, params))


def default_literature(local_index_path: Path | None = None) -> Literature:
    """The production configuration: default cache, real HTTP, contact address and API keys from the
    environment and ~/.bigbrane/keys. Raises ApiKeyError when a key file is readable by group or others.
    Every instance shares one request pacer, whose files under var/slots make the per-source spacing hold
    across every process on this host."""
    return Literature(
        contact_email=os.environ.get(CONTACT_ENV) or None,
        local_index_path=local_index_path,
        api_keys=load_api_keys(),
        pacer=PROCESS_PACER,
    )


def _check_sources(sources_requested: tuple[str, ...]) -> None:
    for source in sources_requested:
        if source not in SEARCH_SOURCES:
            raise ValueError(f"unknown literature source {source!r}; expected one of {SEARCH_SOURCES}")


def _parse(source: str, fetched: Fetched, parser: Callable[[str, str], Any]) -> Any:
    """Parsing is the boundary where an unexpected upstream shape becomes a recorded SourceError."""
    try:
        return parser(fetched.body, fetched.fetched_at)
    except (ValueError, KeyError, TypeError) as exc:
        raise SourceError(source, fetched.url, f"unparseable response: {exc}") from exc


def _load_json(source: str, fetched: Fetched, expected: type) -> Any:
    try:
        payload = json.loads(fetched.body)
    except ValueError as exc:
        raise SourceError(source, fetched.url, f"invalid JSON: {exc}") from exc
    if not isinstance(payload, expected):
        raise SourceError(source, fetched.url, f"expected JSON {expected.__name__}")
    return payload


def _openalex_lookup(paper: Paper) -> str:
    if paper.doi:
        return f"doi:{paper.doi}"
    if paper.arxiv_id:
        return f"doi:10.48550/arXiv.{paper.arxiv_id}"
    raise SourceError("openalex", "", "paper has no DOI or arXiv id to resolve in OpenAlex")


def _s2_lookup(paper: Paper) -> str:
    if paper.source == "s2":
        return paper.source_id
    if paper.arxiv_id:
        return f"ARXIV:{paper.arxiv_id}"
    if paper.doi:
        return f"DOI:{paper.doi}"
    raise SourceError("s2", "", "paper has no S2 id, arXiv id or DOI to resolve in Semantic Scholar")


