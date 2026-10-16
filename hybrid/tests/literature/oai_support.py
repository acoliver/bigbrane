"""Builders for OAI-PMH pages and a small fixture index. No network."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Callable

import httpx

from hybrid.literature.arxiv_index import ArxivIndexWriter, ArxivRecord, PageCommit

RESPONSE_DATE = "2026-10-11T00:37:04Z"


def record(
    arxiv_id: str,
    title: str,
    abstract: str = "An abstract.",
    categories: tuple[str, ...] = ("cs.LG",),
    created: str = "2024-03-01",
    authors: tuple[str, ...] = ("Ada Lovelace",),
    datestamp: str = "2026-10-01",
    doi: str | None = None,
    journal_ref: str | None = None,
) -> ArxivRecord:
    return ArxivRecord(arxiv_id, title, abstract, authors, categories, created, created, datestamp, doi, journal_ref)


def _record_xml(rec: ArxivRecord) -> str:
    authors = "".join(
        f"<author><keyname>{escape(a.split()[-1])}</keyname><forenames>{escape(' '.join(a.split()[:-1]))}</forenames></author>"
        for a in rec.authors
    )
    doi = f"<doi>{escape(rec.doi)}</doi>" if rec.doi else ""
    jref = f"<journal-ref>{escape(rec.journal_ref)}</journal-ref>" if rec.journal_ref else ""
    return (
        f"<record><header><identifier>oai:arXiv.org:{rec.arxiv_id}</identifier><datestamp>{rec.datestamp}</datestamp>"
        f"<setSpec>cs:cs:LG</setSpec></header><metadata>"
        f'<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{rec.arxiv_id}</id><created>{rec.created}</created>'
        f"<updated>{rec.updated}</updated><authors>{authors}</authors><title>{escape(rec.title)}</title>"
        f"<categories>{' '.join(rec.categories)}</categories>{doi}{jref}"
        f"<abstract>{escape(rec.abstract)}</abstract></arXiv></metadata></record>"
    )


def oai_page(
    records: list[ArxivRecord],
    token: str | None = None,
    deleted: tuple[str, ...] = (),
    response_date: str = RESPONSE_DATE,
    list_size: int | None = None,
) -> str:
    gone = "".join(
        f'<record><header status="deleted"><identifier>oai:arXiv.org:{i}</identifier>'
        f"<datestamp>2026-10-01</datestamp></header></record>"
        for i in deleted
    )
    size = f' completeListSize="{list_size}"' if list_size is not None else ""
    token_xml = f"<resumptionToken{size}>{token}</resumptionToken>" if token else "<resumptionToken/>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        f"<responseDate>{response_date}</responseDate><request verb=\"ListRecords\">x</request>"
        f"<ListRecords>{''.join(_record_xml(r) for r in records)}{gone}{token_xml}</ListRecords></OAI-PMH>"
    )


def oai_error(code: str, message: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        f'<responseDate>{RESPONSE_DATE}</responseDate><request>x</request><error code="{code}">{message}</error></OAI-PMH>'
    )


class FakeOai:
    """Serves scripted responses. `handler` gets the request params and returns an httpx.Response or text.
    Every request is recorded."""

    def __init__(self, handler: Callable[[dict[str, str], httpx.Request], httpx.Response | str]) -> None:
        self._handler = handler
        self.requests: list[tuple[dict[str, str], httpx.Request]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.requests.append((params, request))
        result = self._handler(params, request)
        return result if isinstance(result, httpx.Response) else httpx.Response(200, text=result)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self), headers={"User-Agent": "test-agent"})

    @property
    def params(self) -> list[dict[str, str]]:
        return [p for p, _ in self.requests]


FIXTURE_RECORDS = (
    record("2401.00001", "Echo state networks for chaotic time series",
           "Reservoir computing with echo state networks predicts chaotic dynamics.",
           ("cs.LG", "nlin.CD"), "2024-01-02", ("Herbert Jaeger",)),
    record("2401.00002", "Hopfield networks and associative memory capacity",
           "Modern Hopfield networks store exponentially many patterns for associative recall.",
           ("cs.NE", "cond-mat.dis-nn"), "2024-01-03", ("John Hopfield", "Dmitry Krotov")),
    record("2401.00003", "State space models for long sequence modelling",
           "Selective state space models compete with attention on language modelling.",
           ("cs.LG",), "2024-01-04", ("Albert Gu",), journal_ref="Some Journal 12 (2024)"),
    record("2401.00004", "Reservoir computing hardware",
           "Physical reservoir computing with photonic hardware.",
           ("physics.data-an",), "2023-05-06", ("Some Author",), doi="10.1000/xyz"),
    record("2401.00005", "Sparse distributed memory",
           "Kanerva's sparse distributed memory revisited.",
           ("cs.AI",), "2022-09-10", ("Pentti Kanerva",)),
)


def build_fixture_index(path: Path, *, complete: bool = True) -> Path:
    """A small index through the real writer: five papers, one set, snapshot 2026-10-11."""
    with ArxivIndexWriter(path) as writer:
        writer.ensure_sets(["cs"])
        writer.commit_page(
            PageCommit("cs", FIXTURE_RECORDS, (), 0, 0, None if complete else "tok", RESPONSE_DATE)
        )
    return path
