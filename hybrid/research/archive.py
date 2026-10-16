"""Quality-diversity archive for generation (DISCOVERY-PIPELINE-PLAN.md section 6.2, Phase 6).

MAP-Elites (Mouret and Clune 2015, arXiv 1504.04909) over five axes of foundational commitments: representation
substrate, learning rule, unit, objective and where structure comes from. Value ids and texts are in
`hybrid/config/archive.toml`; the classifier role assigns a card's values (never the proposer), and
`hybrid/research/stages/g_generate.py` applies the deterministic consistency check before the card reaches here.

Cells. A cell is one value per axis, keyed "substrate|learning_rule|unit|objective|structure_source". A cell is
closed when it matches the reference cells of an excluded family (`hybrid/config/families.toml`); a card that lands
in a closed cell dies at G with not_radical:R1. Visits count every classified card of the cell (admitted or not);
members are the admitted cards; elites are the best `elites_per_cell` members by quality: furthest stage passed,
then profile obligations at tier lean or sympy_exact (none are counted before stage M reports them), then fewer P4
examples (unknown before Phase 10).

Reward. A card admitted into a cell with no earlier visit filled an empty cell. When it later passes
`archive.reward_stage` (the plan's R; S1 stands in until Phase 7 builds R), an `archive_reward` event credits its
arm, once.

Parents. For mutate and cross tasks, the weight of a cell is 1 / (1 + visits) plus `neighbour_bonus` for each empty
neighbouring cell (one axis different, not closed); an eligible card is drawn with its cell's weight, elites only.

Duplicates. The archive indexes the mechanism signature of every card of the run and every outcome-store record,
with the commitments left out (`mechanism_signature`), so a classifier that places a renamed copy on other axes does
not hide it. A new card whose signature is already indexed is a duplicate.

Persistence and resume. The run ledger holds every assignment (`archive_assigned`) and reward (`archive_reward`);
`Archive.replay` rebuilds the archive from them and the card files, so a resumed run continues the same archive.
Outcome-store records that carry an archive cell (cards of earlier arms runs) seed the visits, so the archive also
persists across runs; their signatures seed the duplicate index whether or not they carry a cell.
"""
from __future__ import annotations

import random
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from hybrid.research.ledger import Ledger
from hybrid.research.signature import AXES, card_signature, signature

ACTOR = "orchestrator"
WILDCARD = "*"


# ----- config ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveConfig:
    values: dict[str, tuple[str, ...]]        # axis -> value ids, in file order
    texts: dict[str, dict[str, str]]          # axis -> value id -> text
    titles: dict[str, str]
    elites_per_cell: int
    neighbour_bonus: float
    reward_stage: str


@dataclass(frozen=True)
class Family:
    id: str
    name: str
    form: str
    reduces_to: tuple[str, ...]
    cells: dict[str, tuple[str, ...]] | None   # excluded families only: allowed values per axis, "*" for any


@dataclass(frozen=True)
class Families:
    excluded: tuple[Family, ...]
    known: tuple[Family, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(f.id for f in self.excluded + self.known)

    def get(self, family_id: str) -> Family | None:
        return next((f for f in self.excluded + self.known if f.id == family_id), None)

    def is_excluded(self, family_id: str) -> bool:
        return any(f.id == family_id for f in self.excluded)

    def is_known(self, family_id: str) -> bool:
        return any(f.id == family_id for f in self.known)


def load_archive_config(path: Path) -> ArchiveConfig:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    axes = data["axes"]
    if tuple(axes) != AXES:
        raise ValueError(f"{path}: axes must be exactly {AXES} in that order, got {tuple(axes)}")
    values, texts, titles = {}, {}, {}
    for axis in AXES:
        ids = [entry["id"] for entry in axes[axis]["values"]]
        if len(set(ids)) != len(ids) or not ids:
            raise ValueError(f"{path}: axis {axis} needs distinct value ids, got {ids}")
        bad = [value for value in ids if "|" in value or value == WILDCARD]
        if bad:
            raise ValueError(f"{path}: axis {axis} value ids may not contain '|' or be '*': {bad}")
        values[axis] = tuple(ids)
        texts[axis] = {entry["id"]: entry["text"] for entry in axes[axis]["values"]}
        titles[axis] = axes[axis]["title"]
    settings = data["archive"]
    elites = settings["elites_per_cell"]
    if isinstance(elites, bool) or not isinstance(elites, int) or elites < 1:
        raise ValueError(f"{path}: archive.elites_per_cell must be an integer >= 1, got {elites!r}")
    bonus = settings["neighbour_bonus"]
    if isinstance(bonus, bool) or not isinstance(bonus, (int, float)) or bonus < 0:
        raise ValueError(f"{path}: archive.neighbour_bonus must be a number >= 0, got {bonus!r}")
    return ArchiveConfig(values, texts, titles, elites, float(bonus), str(settings["reward_stage"]))


def load_families(path: Path, archive: ArchiveConfig) -> Families:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)

    def family(entry: dict, excluded: bool) -> Family:
        cells = None
        if excluded:
            cells = {}
            for axis in AXES:
                allowed = tuple(entry["cells"][axis])
                unknown = [v for v in allowed if v != WILDCARD and v not in archive.values[axis]]
                if not allowed or unknown:
                    raise ValueError(f"{path}: family {entry['id']} axis {axis} names unknown values {unknown}")
                cells[axis] = allowed
            extra = set(entry["cells"]) - set(AXES)
            if extra:
                raise ValueError(f"{path}: family {entry['id']} names unknown axes {sorted(extra)}")
        return Family(entry["id"], entry["name"], entry["form"], tuple(entry.get("reduces_to", [])), cells)

    out = Families(tuple(family(e, True) for e in data.get("excluded", [])),
                   tuple(family(e, False) for e in data.get("known", [])))
    ids = out.ids
    if len(set(ids)) != len(ids) or "none" in ids:
        raise ValueError(f"{path}: family ids must be distinct and not 'none': {ids}")
    return out


# ----- cells -------------------------------------------------------------------------------------


def cell_key(commitments: dict) -> str:
    return "|".join(commitments[axis] for axis in AXES)


def cell_values(key: str) -> dict:
    parts = key.split("|")
    if len(parts) != len(AXES):
        raise ValueError(f"not an archive cell key: {key!r}")
    return dict(zip(AXES, parts))


def closed_by(commitments: dict, families: Families) -> Family | None:
    """The excluded family whose reference cells contain this cell, or None."""
    for fam in families.excluded:
        if all(WILDCARD in fam.cells[axis] or commitments[axis] in fam.cells[axis] for axis in AXES):
            return fam
    return None


def neighbours(key: str, cfg: ArchiveConfig) -> list[str]:
    """Cells that differ from `key` on exactly one axis."""
    values = cell_values(key)
    out = []
    for axis in AXES:
        for other in cfg.values[axis]:
            if other != values[axis]:
                out.append(cell_key(dict(values, **{axis: other})))
    return out


def mechanism_signature(card: dict) -> str:
    """The card's mechanism signature with the commitments left out (the duplicate key)."""
    return card_signature({key: value for key, value in card.items() if key != "commitments"})["signature"]


def stored_mechanism_signature(record: dict) -> str:
    """The same key for an outcome-store record, from its stored skeleton."""
    skeleton = {key: value for key, value in record["mechanism_signature"]["skeleton"].items() if key != "commitments"}
    return signature(skeleton)


def quality(card: dict, order: tuple[str, ...]) -> tuple:
    """Elite quality (plan 6.2): furthest stage passed, then lean or sympy_exact obligations (0 until stage M reports
    them), then fewer P4 examples (0 until Phase 10). -1 for a card that has not passed S0."""
    passed = [order.index(stage) for stage, v in card["status"]["verdicts"].items()
              if v["verdict"] == "pass" and stage in order]
    return (max(passed, default=-1), 0, 0)


# ----- archive -----------------------------------------------------------------------------------


@dataclass
class Entry:
    card_id: str
    cell: str
    arm: str | None
    op: str
    admitted: bool
    new_cell: bool      # the card filled a cell no earlier card had visited
    seq: int
    gate: str | None = None


@dataclass
class Archive:
    cfg: ArchiveConfig
    families: Families
    order: tuple[str, ...]
    visits: Counter = field(default_factory=Counter)
    entries: dict[str, Entry] = field(default_factory=dict)      # card id -> its assignment in this run
    members: dict[str, list[str]] = field(default_factory=dict)  # cell -> admitted card ids, in arrival order
    seeded: Counter = field(default_factory=Counter)             # cell -> visits from the outcome store
    signatures: dict[str, str] = field(default_factory=dict)     # mechanism signature -> "run/card id"
    rewarded: set = field(default_factory=set)
    causes: dict[str, Counter] = field(default_factory=dict)     # cell -> kill codes recorded there (G gates, store)

    # --- building --------------------------------------------------------------------------

    def seed_from_store(self, stored: list, *, run_id: str) -> None:
        """Earlier runs' records: every signature joins the duplicate index; a record with an archive cell adds one
        visit to it. This run's own records are skipped (its ledger holds them)."""
        for item in stored:
            record = item.record
            if record["run"] == run_id:
                continue
            self.signatures.setdefault(stored_mechanism_signature(record), f"{record['run']}/{record['card_id']}")
            cell = (record.get("archive_cell") or {}).get("key")
            if cell:
                self.seeded[cell] += 1
                self.visits[cell] += 1
                self.causes.setdefault(cell, Counter()).update(record["kill_codes"])

    def replay(self, ledger: Ledger, store) -> None:
        """Rebuild this run's part of the archive from its ledger: assignments, then rewards. Every card of the run
        joins the duplicate index, including cards made before the run switched to arms (none can) or rejected at G."""
        for card_id in store.list_cards():
            self.signatures.setdefault(mechanism_signature(store.get_card(card_id)), card_id)
        for event in ledger.iter_events():
            data = event["data"]
            if event["type"] == "archive_assigned":
                self._apply(Entry(data["card_id"], data["cell"], data["arm"], data["op"], data["admitted"],
                                  data["new_cell"], event["seq"], data["gate"]))
            elif event["type"] == "archive_reward":
                self.rewarded.add(data["card_id"])

    def _apply(self, entry: Entry) -> None:
        self.entries[entry.card_id] = entry
        self.visits[entry.cell] += 1
        if entry.gate:
            self.causes.setdefault(entry.cell, Counter())[entry.gate] += 1
        if entry.admitted:
            self.members.setdefault(entry.cell, []).append(entry.card_id)

    # --- queries ---------------------------------------------------------------------------

    def is_empty(self, cell: str) -> bool:
        return self.visits[cell] == 0

    def is_closed(self, cell: str) -> bool:
        return closed_by(cell_values(cell), self.families) is not None

    def duplicate_of(self, card: dict) -> str | None:
        """The run/card the card's mechanism signature already belongs to, or None."""
        return self.signatures.get(mechanism_signature(card))

    def elites(self, cell: str, store) -> list[str]:
        members = self.members.get(cell, [])
        ranked = sorted(members, key=lambda cid: (quality(store.get_card(cid), self.order), -members.index(cid)),
                        reverse=True)
        return ranked[:self.cfg.elites_per_cell]

    def empty_neighbours(self, cell: str) -> int:
        return sum(1 for other in neighbours(cell, self.cfg) if self.is_empty(other) and not self.is_closed(other))

    def cell_weight(self, cell: str) -> float:
        return 1.0 / (1.0 + self.visits[cell]) + self.cfg.neighbour_bonus * self.empty_neighbours(cell)

    def choose_parents(self, eligible: list[str], k: int, rng: random.Random, store) -> list[str]:
        """k distinct parents from the eligible cards that are elites of their cells, each drawn with its cell's
        weight. Cards outside the archive (none in an arms run) are not drawn."""
        pool = [cid for cid in eligible if cid in self.entries and cid in self.elites(self.entries[cid].cell, store)]
        chosen: list[str] = []
        for _ in range(min(k, len(pool))):
            left = [cid for cid in pool if cid not in chosen]
            weights = [self.cell_weight(self.entries[cid].cell) for cid in left]
            chosen.append(rng.choices(left, weights=weights, k=1)[0])
        return chosen

    def coverage(self) -> dict:
        """Per-axis and pairwise coverage of admitted members (seeded visits from earlier runs are not members)."""
        filled = [cell_values(cell) for cell, ids in self.members.items() if ids]
        per_axis = {axis: {"filled": len({v[axis] for v in filled}), "values": len(self.cfg.values[axis])}
                    for axis in AXES}
        pairwise = {}
        for a, b in combinations(AXES, 2):
            pairwise[f"{a}x{b}"] = {"filled": len({(v[a], v[b]) for v in filled}),
                                   "cells": len(self.cfg.values[a]) * len(self.cfg.values[b])}
        return {"cells_filled": len(filled), "per_axis": per_axis, "pairwise": pairwise}

    # --- recording -------------------------------------------------------------------------

    def record(self, ledger: Ledger, *, card: dict, op: str, admitted: bool, classifier_call: str,
               overrides: list[str], gate: str | None) -> Entry:
        """Record the card's cell (`archive_assigned`). new_cell is true when the cell had no visit before and the
        card was admitted. The card's signature joins the duplicate index whatever the outcome."""
        cell = card["archive_cell"]["key"]
        entry = Entry(card["id"], cell, card.get("arm"), op, admitted, bool(card["archive_cell"]["new_cell"]) and admitted,
                      0, gate)
        fam = closed_by(cell_values(cell), self.families)
        event = ledger.append("archive_assigned", ACTOR, {
            "card_id": card["id"], "cell": cell, "commitments": dict(card["commitments"]), "arm": entry.arm,
            "op": op, "admitted": admitted, "new_cell": entry.new_cell, "visits_before": self.visits[cell],
            "closed_by": fam.id if fam else None, "classifier_call": classifier_call, "overrides": overrides,
            "gate": gate, "signature": mechanism_signature(card)})
        entry.seq = event["seq"]
        self._apply(entry)
        self.signatures.setdefault(mechanism_signature(card), card["id"])
        return entry

    def on_pass(self, ledger: Ledger, card_id: str, stage: str) -> bool:
        """After a pass verdict: a card that filled an empty cell earns its arm's reward when it passes the reward
        stage, once. Returns True when a reward was recorded."""
        entry = self.entries.get(card_id)
        if stage != self.cfg.reward_stage or entry is None or not entry.new_cell or card_id in self.rewarded:
            return False
        self.rewarded.add(card_id)
        ledger.append("archive_reward", ACTOR, {"card_id": card_id, "cell": entry.cell, "arm": entry.arm,
                                                "op": entry.op, "stage": stage})
        return True

    def snapshot(self, store) -> dict:
        cells = {}
        for cell in sorted(set(self.visits) | set(self.members)):
            fam = closed_by(cell_values(cell), self.families)
            cells[cell] = {"visits": self.visits[cell], "seeded_visits": self.seeded[cell],
                           "members": list(self.members.get(cell, [])), "elites": self.elites(cell, store),
                           "closed_by": fam.id if fam else None}
        return {"cells": cells, "coverage": self.coverage(), "rewards": sorted(self.rewarded),
                "elites_per_cell": self.cfg.elites_per_cell}
