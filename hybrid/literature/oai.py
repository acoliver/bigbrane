"""Parser for arXiv OAI-PMH `ListRecords` responses in the `arXiv` metadata format."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from hybrid.literature.arxiv_index import ArxivRecord
from hybrid.literature.paper import normalize_arxiv_id

_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_ARXIV = "{http://arxiv.org/OAI/arXiv/}"
_IDENTIFIER_PREFIX = "oai:arXiv.org:"


class OaiError(RuntimeError):
    """The server answered with an OAI-PMH `<error>` element."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"OAI-PMH error {code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class OaiPage:
    response_date: str
    records: tuple[ArxivRecord, ...]
    deleted_ids: tuple[str, ...]
    malformed_ids: tuple[str, ...]
    resumption_token: str | None
    complete_list_size: int | None


def parse_list_records(body: str) -> OaiPage:
    """Parse one page. A record that lacks an id or a title is reported in `malformed_ids` instead of
    stopping the harvest, because the feed is third-party data. A malformed document raises ValueError."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError(f"malformed OAI-PMH response: {exc}") from exc
    response_date = _text(root, f"{_OAI}responseDate")
    if not response_date:
        raise ValueError("OAI-PMH response has no responseDate")
    error = root.find(f"{_OAI}error")
    if error is not None:
        raise OaiError(error.get("code", "unknown"), (error.text or "").strip())
    listing = root.find(f"{_OAI}ListRecords")
    if listing is None:
        raise ValueError("OAI-PMH response has neither ListRecords nor error")

    records: list[ArxivRecord] = []
    deleted: list[str] = []
    malformed: list[str] = []
    for element in listing.findall(f"{_OAI}record"):
        header = element.find(f"{_OAI}header")
        identifier = _text(header, f"{_OAI}identifier") if header is not None else ""
        header_id = identifier.removeprefix(_IDENTIFIER_PREFIX)
        if header is not None and header.get("status") == "deleted":
            deleted.append(_bare_id(header_id) or header_id)
            continue
        record = _parse_record(element, header)
        if record is None:
            malformed.append(identifier or "(no identifier)")
        else:
            records.append(record)

    token_element = listing.find(f"{_OAI}resumptionToken")
    token = (token_element.text or "").strip() if token_element is not None else ""
    size = token_element.get("completeListSize") if token_element is not None else None
    return OaiPage(
        response_date=response_date,
        records=tuple(records),
        deleted_ids=tuple(deleted),
        malformed_ids=tuple(malformed),
        resumption_token=token or None,
        complete_list_size=int(size) if size and size.isdigit() else None,
    )


def _parse_record(element: ET.Element, header: ET.Element | None) -> ArxivRecord | None:
    meta = element.find(f"{_OAI}metadata/{_ARXIV}arXiv")
    if meta is None or header is None:
        return None
    arxiv_id = _bare_id(_text(meta, f"{_ARXIV}id"))
    title = _collapse(_text(meta, f"{_ARXIV}title"))
    datestamp = _text(header, f"{_OAI}datestamp")
    if arxiv_id is None or not title or not datestamp:
        return None
    return ArxivRecord(
        arxiv_id=arxiv_id,
        title=title,
        abstract=_collapse(_text(meta, f"{_ARXIV}abstract")),
        authors=tuple(name for name in (_author(a) for a in meta.findall(f"{_ARXIV}authors/{_ARXIV}author")) if name),
        categories=tuple(_text(meta, f"{_ARXIV}categories").split()),
        created=_text(meta, f"{_ARXIV}created") or None,
        updated=_text(meta, f"{_ARXIV}updated") or None,
        datestamp=datestamp,
        doi=_collapse(_text(meta, f"{_ARXIV}doi")) or None,
        journal_ref=_collapse(_text(meta, f"{_ARXIV}journal-ref")) or None,
    )


def _author(element: ET.Element) -> str:
    parts = (_text(element, f"{_ARXIV}{tag}") for tag in ("forenames", "keyname", "suffix"))
    return _collapse(" ".join(part for part in parts if part))


def _bare_id(raw: str) -> str | None:
    return normalize_arxiv_id(raw) if raw else None


def _text(element: ET.Element | None, path: str) -> str:
    found = element.find(path) if element is not None else None
    return (found.text or "").strip() if found is not None else ""


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
