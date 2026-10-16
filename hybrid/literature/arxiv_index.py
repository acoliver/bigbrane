"""SQLite store with an FTS5 index over harvested arXiv metadata.

One file holds three things: the `papers` table, the `papers_fts` index over it, and `harvest_state`,
the per-set progress (including the resumption token) that makes a harvest resumable. The writer is
used by the harvester, the reader by the `arxiv_local` search source.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

# Column order matters: bm25() weights below are positional (title, abstract, authors, categories).
BM25_WEIGHTS = (3.0, 1.0, 0.0, 0.0)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    arxiv_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    categories TEXT NOT NULL,
    created TEXT,
    updated TEXT,
    datestamp TEXT NOT NULL,
    doi TEXT,
    journal_ref TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, authors, categories,
    content='papers', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, abstract, authors, categories)
    VALUES (new.id, new.title, new.abstract, new.authors, new.categories);
END;
CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors, categories)
    VALUES ('delete', old.id, old.title, old.abstract, old.authors, old.categories);
END;
CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors, categories)
    VALUES ('delete', old.id, old.title, old.abstract, old.authors, old.categories);
    INSERT INTO papers_fts(rowid, title, abstract, authors, categories)
    VALUES (new.id, new.title, new.abstract, new.authors, new.categories);
END;
CREATE TABLE IF NOT EXISTS harvest_state (
    set_spec TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    resumption_token TEXT,
    from_date TEXT,
    pages INTEGER NOT NULL DEFAULT 0,
    records_seen INTEGER NOT NULL DEFAULT 0,
    records_kept INTEGER NOT NULL DEFAULT 0,
    records_filtered INTEGER NOT NULL DEFAULT 0,
    records_deleted INTEGER NOT NULL DEFAULT 0,
    records_malformed INTEGER NOT NULL DEFAULT 0,
    complete_list_size INTEGER,
    first_response_date TEXT,
    last_response_date TEXT,
    started_at TEXT,
    completed_at TEXT,
    restarts INTEGER NOT NULL DEFAULT 0
);
"""

STATUS_PENDING, STATUS_RUNNING, STATUS_DONE = "pending", "running", "done"
# A set whose server kept failing. Its resumption token stays, so the next pass resumes where it stopped.
STATUS_DEFERRED = "deferred"


@dataclass(frozen=True)
class ArxivRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    created: str | None
    updated: str | None
    datestamp: str
    doi: str | None = None
    journal_ref: str | None = None


@dataclass(frozen=True)
class SetState:
    set_spec: str
    status: str
    resumption_token: str | None
    from_date: str | None
    pages: int
    records_seen: int
    records_kept: int
    records_filtered: int
    records_deleted: int
    records_malformed: int
    complete_list_size: int | None
    first_response_date: str | None
    last_response_date: str | None
    started_at: str | None
    completed_at: str | None
    restarts: int


@dataclass(frozen=True)
class PageCommit:
    """Everything one harvested page changes, applied in a single transaction with the new token."""

    set_spec: str
    records: tuple[ArxivRecord, ...]
    deleted_ids: tuple[str, ...]
    filtered: int
    malformed: int
    resumption_token: str | None
    response_date: str
    complete_list_size: int | None = None

    @property
    def seen(self) -> int:
        return len(self.records) + len(self.deleted_ids) + self.filtered + self.malformed


@dataclass(frozen=True)
class IndexInfo:
    """What a search receipt says about the snapshot it ran against."""

    path: str
    record_count: int
    snapshot_ts: str | None
    complete: bool

    @property
    def snapshot_date(self) -> str | None:
        return self.snapshot_ts[:10] if self.snapshot_ts else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_schema(db: sqlite3.Connection) -> None:
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        raise RuntimeError(f"arXiv index schema version {version}, expected {SCHEMA_VERSION}")


class ArxivIndexWriter:
    """Single-writer handle used by the harvester. Not shared across threads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        if self._db.execute("PRAGMA user_version").fetchone()[0] == 0:
            self._db.executescript(_SCHEMA)
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._db.commit()
        _check_schema(self._db)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ArxivIndexWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Harvest state

    def ensure_sets(self, set_specs: Iterable[str], from_date: str | None = None) -> None:
        with self._db:
            for spec in set_specs:
                self._db.execute(
                    "INSERT OR IGNORE INTO harvest_state(set_spec, status, from_date) VALUES (?, ?, ?)",
                    (spec, STATUS_PENDING, from_date),
                )

    def state(self, set_spec: str) -> SetState:
        row = self._db.execute("SELECT * FROM harvest_state WHERE set_spec = ?", (set_spec,)).fetchone()
        if row is None:
            raise KeyError(f"set {set_spec!r} is not registered; call ensure_sets first")
        return _state_from_row(row)

    def states(self) -> list[SetState]:
        rows = self._db.execute("SELECT * FROM harvest_state ORDER BY rowid").fetchall()
        return [_state_from_row(row) for row in rows]

    def restart_set(self, set_spec: str) -> None:
        """Drop the token and the counts so the set is walked again from its first page. The records
        already stored stay; the upserts of the new pass overwrite them."""
        with self._db:
            self._db.execute(
                "UPDATE harvest_state SET status=?, resumption_token=NULL, pages=0, records_seen=0, records_kept=0,"
                " records_filtered=0, records_deleted=0, records_malformed=0, complete_list_size=NULL,"
                " first_response_date=NULL, completed_at=NULL, restarts=restarts+1 WHERE set_spec=?",
                (STATUS_PENDING, set_spec),
            )

    def mark_deferred(self, set_spec: str) -> None:
        """A set gave up after its retries. The token and counts stay, so the set resumes from its last stored page."""
        with self._db:
            self._db.execute("UPDATE harvest_state SET status=? WHERE set_spec=?", (STATUS_DEFERRED, set_spec))

    def mark_done(self, set_spec: str, response_date: str) -> None:
        """For a set that answered `noRecordsMatch`: nothing to store, nothing left to fetch."""
        with self._db:
            self._db.execute(
                "UPDATE harvest_state SET status=?, resumption_token=NULL, completed_at=?,"
                " first_response_date=COALESCE(first_response_date, ?), last_response_date=?,"
                " started_at=COALESCE(started_at, ?) WHERE set_spec=?",
                (STATUS_DONE, _utc_now(), response_date, response_date, _utc_now(), set_spec),
            )

    def commit_page(self, page: PageCommit) -> SetState:
        """Store the page's records and advance the set's token atomically, so a stop at any moment
        leaves the token pointing at the first page whose records are not yet stored."""
        finished = page.resumption_token is None
        now = _utc_now()
        with self._db:
            self._db.executemany(
                "INSERT INTO papers(arxiv_id, title, abstract, authors, categories, created, updated, datestamp,"
                " doi, journal_ref) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(arxiv_id) DO UPDATE SET title=excluded.title, abstract=excluded.abstract,"
                " authors=excluded.authors, categories=excluded.categories, created=excluded.created,"
                " updated=excluded.updated, datestamp=excluded.datestamp, doi=excluded.doi,"
                " journal_ref=excluded.journal_ref",
                [_record_row(record) for record in page.records],
            )
            self._db.executemany("DELETE FROM papers WHERE arxiv_id = ?", [(i,) for i in page.deleted_ids])
            self._db.execute(
                "UPDATE harvest_state SET status=?, resumption_token=?, pages=pages+1, records_seen=records_seen+?,"
                " records_kept=records_kept+?, records_filtered=records_filtered+?,"
                " records_deleted=records_deleted+?, records_malformed=records_malformed+?,"
                " complete_list_size=COALESCE(?, complete_list_size),"
                " first_response_date=COALESCE(first_response_date, ?), last_response_date=?,"
                " started_at=COALESCE(started_at, ?), completed_at=? WHERE set_spec=?",
                (
                    STATUS_DONE if finished else STATUS_RUNNING,
                    page.resumption_token,
                    page.seen,
                    len(page.records),
                    page.filtered,
                    len(page.deleted_ids),
                    page.malformed,
                    page.complete_list_size,
                    page.response_date,
                    page.response_date,
                    now,
                    now if finished else None,
                    page.set_spec,
                ),
            )
        return self.state(page.set_spec)

    # Summary

    def summary(self) -> dict[str, Any]:
        return _summary(self._db, self.path)


class ArxivIndexReader:
    """Read-only handle. Opens one connection per use, so it can be called from any thread."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        _check_schema(db)
        return db

    def info(self) -> IndexInfo:
        db = self._connect()
        try:
            states = db.execute("SELECT status, last_response_date FROM harvest_state").fetchall()
            count = db.execute("SELECT count(*) FROM papers").fetchone()[0]
        finally:
            db.close()
        dates = [row["last_response_date"] for row in states if row["last_response_date"]]
        complete = bool(states) and all(row["status"] == STATUS_DONE for row in states)
        return IndexInfo(str(self.path), count, max(dates) if dates else None, complete)

    def deferred_sets(self) -> list[str]:
        db = self._connect()
        try:
            rows = db.execute("SELECT set_spec FROM harvest_state WHERE status = ? ORDER BY rowid", (STATUS_DEFERRED,))
            return [row["set_spec"] for row in rows.fetchall()]
        finally:
            db.close()

    def search(self, match: str, limit: int) -> list[ArxivRecord]:
        """Records matching an FTS5 expression, best BM25 score first."""
        db = self._connect()
        try:
            rows = db.execute(
                "SELECT p.* FROM papers_fts JOIN papers p ON p.id = papers_fts.rowid"
                " WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts, ?, ?, ?, ?) LIMIT ?",
                (match, *BM25_WEIGHTS, limit),
            ).fetchall()
        finally:
            db.close()
        return [_record_from_row(row) for row in rows]

    def get(self, arxiv_id: str) -> ArxivRecord | None:
        """The record with this bare arXiv id (no version), or None when the index does not hold it."""
        db = self._connect()
        try:
            row = db.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        finally:
            db.close()
        return _record_from_row(row) if row is not None else None


def _record_row(record: ArxivRecord) -> tuple[Any, ...]:
    return (
        record.arxiv_id,
        record.title,
        record.abstract,
        "\n".join(record.authors),
        " ".join(record.categories),
        record.created,
        record.updated,
        record.datestamp,
        record.doi,
        record.journal_ref,
    )


def _record_from_row(row: sqlite3.Row) -> ArxivRecord:
    return ArxivRecord(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        abstract=row["abstract"],
        authors=tuple(row["authors"].split("\n")) if row["authors"] else (),
        categories=tuple(row["categories"].split()),
        created=row["created"],
        updated=row["updated"],
        datestamp=row["datestamp"],
        doi=row["doi"],
        journal_ref=row["journal_ref"],
    )


def _state_from_row(row: sqlite3.Row) -> SetState:
    return SetState(**{key: row[key] for key in row.keys()})


def _summary(db: sqlite3.Connection, path: Path) -> dict[str, Any]:
    states = [_state_from_row(row) for row in db.execute("SELECT * FROM harvest_state ORDER BY rowid")]
    count, created_min, created_max, stamp_min, stamp_max = tuple(db.execute(
        "SELECT count(*), min(created), max(created), min(datestamp), max(datestamp) FROM papers"
    ).fetchone())
    dates = [s.last_response_date for s in states if s.last_response_date]
    return {
        "index_path": str(path),
        "record_count": count,
        "snapshot_ts": max(dates) if dates else None,
        "complete": bool(states) and all(s.status == STATUS_DONE for s in states),
        "created_range": [created_min, created_max],
        "datestamp_range": [stamp_min, stamp_max],
        "sets": {s.set_spec: {k: v for k, v in asdict(s).items() if k != "set_spec"} for s in states},
    }


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
