"""Outcome store (DISCOVERY-PIPELINE-PLAN.md section 9, Phase 5).

One record per card outcome: the idea, its mechanism signature, its difference statement (when present), its
premise ledger and definitions, its stage verdicts, the exact reason it died (kill codes, each with evidence
pointers) and its cost.

Storage, in one directory (default `results/outcomes/`):
- `outcomes.jsonl`: append-only. Each line is `{"seq", "ts", "prev_hash", "hash", "record"}`; `hash` is the sha256
  of `prev_hash` plus the sorted-key JSON of the line without `hash`, as in the run ledger, so an edited, removed or
  reordered line breaks the chain at a specific seq. This file is the source; the store can be rebuilt from it.
- `outcomes.sqlite`: an index of the same records (one row per line, the record JSON included), rebuilt from the
  JSONL when it is missing or behind.
- `outcomes.lock`: an flock held for every append, so two processes can write.

Records are keyed by (run, card id). A later record for the same key supersedes the earlier one in queries (a
resumed run that moves a card on), and nothing is ever rewritten. `append` skips a record whose key and card file
hash are already stored, so the backfill and the orchestrator can both write without duplicates.

Queries:
- `nearby_dead` (step 1, generation): dead ideas near a task, by TF-IDF of the task text against each record's
  idea, Jaccard of signature tokens, and the lens; each with its kill codes and cause ids.
- `own_history` (step 3, prior art): records with the card's exact signature and nearest neighbours (Jaccard on
  signature tokens, TF-IDF on the core idea), each with its recorded kill reason, so a variant must address it.
- `r3_targets`, `r3_problems`: the R3 rule (plan section 1.3): an exact or near signature match with a dead idea
  (near: Jaccard of signature tokens at least `jaccard_min`; a TF-IDF-only neighbour shares words, not mechanism, and
  is shown but not enforced) must be answered by a difference statement whose `cause_ids_addressed` cites every cause
  of that record. Stage S2 enforces it with `outcomes.enforce_r3` (Phase 7 adds the difference statement and the
  challenger's check that the changed equation is the one the cause names).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hybrid.research.graveyard import Grave, clip
from hybrid.research.ledger import GENESIS_HASH, event_hash, sorted_json
from hybrid.research.schemas import validate
from hybrid.research.signature import jaccard
from hybrid.research.signature import signature as signature_of
from hybrid.research.signature import tokens as signature_tokens

JSONL = "outcomes.jsonl"
SQLITE = "outcomes.sqlite"
LOCK = "outcomes.lock"
LINE_KEYS = {"seq", "ts", "prev_hash", "hash", "record"}
OUTCOMES = ("dead", "finished", "open")

# Kill codes (plan section 9), one pattern per code family. The first block is the plan's enumeration (with the
# codes stage M and C record since Phase 3: magic_x_missing, circular). The second block names deaths in stages the
# plan replaces or keeps from the earlier loop, so no backfilled death is free text only.
KILL_CODES = {
    "reduces_to_known": r"reduces_to_known:[a-z0-9_]+",
    "excluded_family": r"excluded_family:[a-z0-9_]+",
    "no_difference": r"no_difference",
    "not_radical": r"not_radical:R[123]",
    "obligation_failed": r"obligation_failed:[\w.\-]+",
    "unproven": r"unproven:[\w.\-]+",
    "magic_x": r"magic_x:[\w.\-]+",
    "magic_x_missing": r"magic_x_missing:[\w.\-]+",
    "circular": r"circular:[\w.\-]+",
    "conservation": r"conservation:[\w.\-]+",
    "statement_mismatch": r"statement_mismatch",
    # FIX-16: stage M refused to formalize a card whose resources section leaves items unstated.
    "card_incomplete": r"card_incomplete",
    # FIX-17: stage C confirmed that the card's own stated resources cannot hold a claim it makes.
    "resource_contradiction": r"resource_contradiction",
    # Phase 8: the loop that stalled or ran out (`stall:s1_revision`, `budget:lineage_ideas`, ...). The bare forms
    # are the plan's enumeration and stay valid.
    "stall": r"stall(?::[a-z0-9_]+)?",
    "budget": r"budget(?::[a-z0-9_]+)?",
    "falsified": r"falsified:[\w.\-]+",
    "predictions_missed": r"predictions_missed",
    # Stages of the earlier loop.
    "duplicate": r"duplicate:card-[0-9a-f]{8}",
    "ill_formed": r"ill_formed:(fatal|fixable|toy)",
    "known_disproven": r"known_disproven",
    "novelty_unassessed": r"novelty_unassessed",
    "infeasible": r"infeasible:(disproof|judges)",
    "undetermined": r"undetermined:[A-Z0-9]+",
    "construct_failed": r"construct_failed",
    "bench_failed": r"bench_failed",
    # Phase 6, generation stage G: the arm's required output structure was missing or wrong.
    "arm_structure": r"arm_structure:A[1-5]",
}
KILL_CODE = re.compile("^(?:" + "|".join(f"(?:{p})" for p in KILL_CODES.values()) + ")$")


class OutcomeStoreError(RuntimeError):
    """The store's files are inconsistent (broken hash chain, malformed line). Nothing is repaired silently."""


def check_kill_code(code: str) -> str:
    if not isinstance(code, str) or not KILL_CODE.match(code):
        raise ValueError(f"not a kill code from the enumeration: {code!r}")
    return code


def kill_family(code: str) -> str:
    return code.split(":", 1)[0]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----- store -------------------------------------------------------------------------------------


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS outcomes (
    seq INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    run TEXT NOT NULL,
    card_id TEXT NOT NULL,
    card_sha256 TEXT NOT NULL,
    signature TEXT NOT NULL,
    outcome TEXT NOT NULL,
    terminal_stage TEXT,
    lens TEXT,
    legacy INTEGER NOT NULL,
    override INTEGER NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outcomes_signature ON outcomes(signature);
CREATE INDEX IF NOT EXISTS outcomes_card ON outcomes(card_id);
CREATE INDEX IF NOT EXISTS outcomes_key ON outcomes(run, card_id);
"""
SQL_COLUMNS = ("seq", "record_id", "hash", "run", "card_id", "card_sha256", "signature", "outcome",
               "terminal_stage", "lens", "legacy", "override", "record")


def record_id(line_hash: str) -> str:
    return "oc-" + line_hash[:12]


def cause_id(rec_id: str, code: str) -> str:
    """The id a difference statement cites in `cause_ids_addressed`: the record and the kill code."""
    return f"{rec_id}/{code}"


def _row(line: dict) -> tuple:
    record = line["record"]
    return (line["seq"], record_id(line["hash"]), line["hash"], record["run"], record["card_id"],
            record["card_sha256"], record["mechanism_signature"]["signature"], record["outcome"],
            record["terminal_stage"], record.get("lens"), int(record["legacy"]), int(record["override"]),
            sorted_json(record))


class OutcomeStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.jsonl = self.dir / JSONL
        self.sqlite = self.dir / SQLITE
        self._lock_path = self.dir / LOCK

    @contextmanager
    def _locked(self, mode: int):
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, mode)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # --- JSONL -----------------------------------------------------------------------------

    def _read_lines(self) -> list[dict]:
        if not self.jsonl.exists():
            return []
        raw = self.jsonl.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise OutcomeStoreError(f"{self.jsonl}: last line is unterminated (partial write?)")
        return [json.loads(line) for line in raw.decode("utf-8").splitlines()]

    def lines(self) -> list[dict]:
        with self._locked(fcntl.LOCK_SH):
            return self._read_lines()

    def verify(self) -> tuple[bool, int | None]:
        """(True, None) for an intact chain, else (False, seq of the first bad line)."""
        prev = GENESIS_HASH
        for position, line in enumerate(self.lines(), start=1):
            if set(line) != LINE_KEYS or line["seq"] != position or line["prev_hash"] != prev:
                return False, position
            body = {key: value for key, value in line.items() if key != "hash"}
            if event_hash(prev, body) != line["hash"]:
                return False, position
            prev = line["hash"]
        return True, None

    def append(self, record: dict) -> dict | None:
        """Append one record. Returns the stored line, or None when a record with the same (run, card id) and card
        file hash is already stored. Validates the record against schemas/outcome.schema.json and every kill
        code against the enumeration first."""
        validate("outcome", record)
        for code in record["kill_codes"]:
            check_kill_code(code)
        if record["outcome"] == "dead" and not record["kill_codes"]:
            raise ValueError(f"{record['run']}/{record['card_id']}: a dead record needs at least one kill code")
        if record["outcome"] != "dead" and record["kill_codes"]:
            raise ValueError(f"{record['run']}/{record['card_id']}: only a dead record carries kill codes")
        with self._locked(fcntl.LOCK_EX):
            lines = self._read_lines()
            key = (record["run"], record["card_id"], record["card_sha256"])
            if any((line["record"]["run"], line["record"]["card_id"], line["record"]["card_sha256"]) == key
                   for line in lines):
                return None
            prev = lines[-1]["hash"] if lines else GENESIS_HASH
            body = {"seq": len(lines) + 1, "ts": _utc(), "prev_hash": prev, "record": record}
            line = dict(body, hash=event_hash(prev, body))
            with open(self.jsonl, "ab") as handle:
                handle.write((sorted_json(line) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            self._sync_sqlite(lines + [line])
        return line

    # --- SQLite index ------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite)
        connection.executescript(SCHEMA_SQL)
        return connection

    def _sync_sqlite(self, lines: list[dict]) -> None:
        """Bring the index up to the JSONL. Rows already present must match their line; a mismatch is an error."""
        connection = self._connect()
        try:
            have = {seq: digest for seq, digest in connection.execute("SELECT seq, hash FROM outcomes")}
            for seq, digest in have.items():
                if seq > len(lines) or lines[seq - 1]["hash"] != digest:
                    raise OutcomeStoreError(f"{self.sqlite}: row seq {seq} does not match {JSONL}; rebuild it")
            new = [_row(line) for line in lines if line["seq"] not in have]
            if new:
                marks = ", ".join("?" for _ in SQL_COLUMNS)
                connection.executemany(f"INSERT INTO outcomes ({', '.join(SQL_COLUMNS)}) VALUES ({marks})", new)
                connection.commit()
        finally:
            connection.close()

    def rebuild_sqlite(self) -> int:
        """Write the SQLite index again from the JSONL. Returns the row count."""
        with self._locked(fcntl.LOCK_EX):
            lines = self._read_lines()
            if self.sqlite.exists():
                self.sqlite.unlink()
            self._sync_sqlite(lines)
        return len(lines)

    def sqlite_rows(self) -> list[tuple]:
        with self._locked(fcntl.LOCK_SH):
            self._sync_sqlite(self._read_lines())
            connection = self._connect()
            try:
                return list(connection.execute(f"SELECT {', '.join(SQL_COLUMNS)} FROM outcomes ORDER BY seq"))
            finally:
                connection.close()

    def jsonl_rows(self) -> list[tuple]:
        return [_row(line) for line in self.lines()]

    # --- records ---------------------------------------------------------------------------------

    def current(self) -> list["Stored"]:
        """The latest record per (run, card id), in append order."""
        latest: dict[tuple[str, str], Stored] = {}
        for line in self.lines():
            stored = Stored(record_id(line["hash"]), line["seq"], line["record"])
            latest[(stored.record["run"], stored.record["card_id"])] = stored
        return sorted(latest.values(), key=lambda s: s.seq)

    def by_signature(self, sig: str) -> list["Stored"]:
        return [s for s in self.current() if s.record["mechanism_signature"]["signature"] == sig]


@dataclass(frozen=True)
class Stored:
    record_id: str
    seq: int
    record: dict

    @property
    def dead(self) -> bool:
        return self.record["outcome"] == "dead"

    def causes(self) -> list[dict]:
        return [dict(cause, id=cause_id(self.record_id, cause["code"])) for cause in self.record["causes"]]

    def text(self) -> str:
        r = self.record
        return "\n".join([r.get("core_idea") or "", r.get("summary") or ""])


# ----- queries -----------------------------------------------------------------------------------


@dataclass
class Match:
    stored: Stored
    exact: bool
    jaccard: float
    tfidf: float
    score: float = 0.0

    def as_dict(self) -> dict:
        r = self.stored.record
        return {"record_id": self.stored.record_id, "card_id": r["card_id"], "run": r["run"], "title": r["title"],
                "exact_signature": self.exact, "jaccard": round(self.jaccard, 4), "tfidf": round(self.tfidf, 4),
                "outcome": r["outcome"], "terminal_stage": r["terminal_stage"], "kill_codes": list(r["kill_codes"]),
                "kill_detail": r["kill_detail"], "causes": self.stored.causes(), "override": r["override"],
                "legacy": r["legacy"]}


def _tfidf_scores(query: str, texts: list[str]) -> list[float]:
    # Imported here: hybrid.research.stages imports every stage, and S2 imports this module.
    from hybrid.research.stages.diversity import tfidf_matrix

    if not texts:
        return []
    matrix = tfidf_matrix(texts + [query])
    return [float(value) for value in matrix[:-1] @ matrix[-1]]


def comparable(a: dict, b: dict) -> tuple[dict, dict]:
    """Two mechanism signatures in a form that can be compared: as they are when both skeletons carry the five
    commitments, else both without them. A card classified at stage G has commitments, while every record of a card
    made before Phase 6 (all backfilled records) has none; comparing the two as they are can never give an exact
    match and dilutes the Jaccard similarity with five commitment tokens (found by the Phase 9 planted-card fixture:
    a near variant of a backfilled dead idea fell from Jaccard 0.88 to 0.54 and escaped R3)."""
    if "commitments" in a["skeleton"] and "commitments" in b["skeleton"]:
        return a, b

    def bare(sig: dict) -> dict:
        skeleton = {key: value for key, value in sig["skeleton"].items() if key != "commitments"}
        return {"signature": signature_of(skeleton), "tokens": signature_tokens(skeleton)}

    return bare(a), bare(b)


def own_history(store: OutcomeStore, card: dict, signature: dict, *, current_run: str, limit: int,
                jaccard_min: float, tfidf_min: float) -> list[Match]:
    """Step 3: earlier outcomes with the card's exact signature, then nearest neighbours (Jaccard on signature
    tokens at or above jaccard_min, or TF-IDF of core idea and summary at or above tfidf_min). Exact matches always
    come back; neighbours fill up to `limit`.

    Only the card's own record in `current_run` is left out. A record with the same card id from another run is an
    earlier outcome of that card (a card copied into a new run keeps its id, as the live check and plumb do), so it is
    found and R3 applies to it (FIX-10, defect D1 of the Phase 7 live check)."""
    candidates = [s for s in store.current()
                  if (s.record["run"], s.record["card_id"]) != (current_run, card["id"])]
    query = "\n".join([card.get("core_idea", ""), card["summary"]])
    scores = _tfidf_scores(query, [s.text() for s in candidates])
    matches = []
    for stored, tf in zip(candidates, scores):
        mine, theirs = comparable(signature, stored.record["mechanism_signature"])
        exact = mine["signature"] == theirs["signature"]
        jac = jaccard(mine["tokens"], theirs["tokens"])
        if exact or jac >= jaccard_min or tf >= tfidf_min:
            matches.append(Match(stored, exact, jac, tf, score=max(jac, tf)))
    exact = [m for m in matches if m.exact]
    near = sorted((m for m in matches if not m.exact), key=lambda m: (-m.score, m.stored.seq))
    return exact + near[:max(0, limit - len(exact))]


def nearby_dead(store: OutcomeStore, *, text: str, tokens: list[str] | None = None, lens: str | None = None,
                limit: int, exclude_runs: tuple[str, ...] = ()) -> list[Match]:
    """Step 1: dead ideas near a generation task, at most one per card id (the most recent record of that card).
    Score: TF-IDF of the task text (lens, papers, parent cards) against the record's idea, plus Jaccard of the
    parents' signature tokens when given, plus 0.25 for the same lens."""
    latest_per_card: dict[str, Stored] = {}
    for stored in store.current():
        if stored.dead and stored.record["run"] not in exclude_runs:
            latest_per_card[stored.record["card_id"]] = stored
    candidates = list(latest_per_card.values())
    scores = _tfidf_scores(text, [s.text() for s in candidates])
    out = []
    for stored, tf in zip(candidates, scores):
        jac = jaccard(tokens, stored.record["mechanism_signature"]["tokens"]) if tokens else 0.0
        same_lens = 0.25 if lens and stored.record.get("lens") == lens else 0.0
        out.append(Match(stored, False, jac, tf, score=tf + jac + same_lens))
    out.sort(key=lambda m: (-m.score, -m.stored.seq))
    return out[:limit]


def r3_targets(matches: list[Match], jaccard_min: float | None = None) -> list[Match]:
    """The matches R3 applies to: dead records with the card's exact signature or, when jaccard_min is given, a
    signature whose tokens have Jaccard similarity at least jaccard_min with the card's."""
    return [m for m in matches if m.stored.dead and (m.exact or (jaccard_min is not None and m.jaccard >= jaccard_min))]


def r3_problems(card: dict, matches: list[Match], jaccard_min: float | None = None) -> list[str]:
    """R3 (plan section 1.3): every exact (and, with jaccard_min, near) signature match with a dead idea must be
    answered by the card's difference statement, whose `cause_ids_addressed` lists each cause of that record. Returns
    the unmet requirements."""
    addressed = set((card.get("difference_statement") or {}).get("cause_ids_addressed") or [])
    problems = []
    for match in r3_targets(matches, jaccard_min):
        missing = [cause["id"] for cause in match.stored.causes() if cause["id"] not in addressed]
        if missing:
            how = "matches" if match.exact else f"is near (Jaccard {match.jaccard:.2f})"
            problems.append(f"signature {how} dead idea {match.stored.record['card_id']} "
                            f"({match.stored.record['run']}) killed by {', '.join(match.stored.record['kill_codes'])}; "
                            f"the difference statement does not address {', '.join(missing)}")
    return problems


# ----- prompt rendering --------------------------------------------------------------------------


def outcome_line(stored: Stored) -> str:
    """One line for a prompt: where the card stopped, its kill codes, the first cause id and the deciding detail."""
    r = stored.record
    if not r["kill_codes"]:
        return f"{r['outcome']} at {r['terminal_stage']}"
    first = cause_id(stored.record_id, r["kill_codes"][0])
    return f"killed at {r['terminal_stage']}: {', '.join(r['kill_codes'])} (cause {first}): {r['kill_detail']}"


def as_grave(stored: Stored) -> Grave:
    """The graveyard renderer's entry for a store record, so S0 prints store records like its own graves."""
    r = stored.record
    kind = "reduced" if any(kill_family(c) == "reduces_to_known" for c in r["kill_codes"]) else (
        "died" if stored.dead else r["outcome"])
    return Grave(r["card_id"], r["title"], r.get("core_idea") or "", kind if kind in ("reduced", "died") else "alive",
                 outcome_line(stored), stored.seq)


def render_nearby(matches: list[Match]) -> str:
    return "\n".join(as_grave(match.stored).line() for match in matches) or "(none)"


def render_history(matches: list[Match]) -> str:
    """The own-history block for the S2 reduction critic: each match with its exact kill reason and cause ids. Each
    cause is shown whole, as stored: the store already bounds it at outcome_records.CAUSE_CHARS with an explicit
    truncation marker, and a second cut here would hide the deciding argument the critic judges (FIX-12, F1)."""
    if not matches:
        return "(no earlier outcome matches this card's signature or its nearest neighbours)"
    rows = []
    for match in matches:
        r = match.stored.record
        how = "EXACT signature" if match.exact else f"near (Jaccard {match.jaccard:.2f}, TF-IDF {match.tfidf:.2f})"
        rows.append(f"- {r['card_id']} ({r['run']}) [{how}] {clip(r['title'], 80)}: {r['outcome']} at "
                    f"{r['terminal_stage']}")
        rows.append(f"  idea: {clip(r.get('core_idea') or '(none)', 300)}")
        for cause in match.stored.causes():
            rows.append(f"  cause {cause['id']}: {cause['detail']}")
    return "\n".join(rows)

