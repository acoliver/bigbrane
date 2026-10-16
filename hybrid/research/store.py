"""Versioned theory cards, content-addressed artifacts and stage verdicts for one run.

Every write goes to disk first and then appends an event to the ledger with the sha256 of the
written bytes. Files are never overwritten: a new card version is a new file, and artifacts are
immutable.
"""
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from hybrid.research.ledger import Ledger
from hybrid.research.schemas import load_schema, validate

CARD_ID = re.compile(r"^card-[0-9a-f]{8}$")
ARTIFACT_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
VERSION_FILE = re.compile(r"^v(\d+)\.json$")
STAGES = tuple(load_schema("theory_card")["properties"]["status"]["properties"]["stage"]["enum"])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


class Store:
    def __init__(self, run_dir, ledger: Ledger):
        self.run_dir = Path(run_dir).resolve()
        self.ledger = ledger
        if ledger.run_dir.resolve() != self.run_dir:
            raise ValueError(f"ledger belongs to {ledger.run_dir}, not {self.run_dir}")

    def _check_card_id(self, card_id: str) -> str:
        if not isinstance(card_id, str) or not CARD_ID.match(card_id):
            raise ValueError(f"malformed card id: {card_id!r}")
        return card_id

    def _card_dir(self, card_id: str) -> Path:
        return self.run_dir / "cards" / self._check_card_id(card_id)

    def versions(self, card_id: str) -> list[int]:
        directory = self._card_dir(card_id)
        if not directory.is_dir():
            return []
        numbers = []
        for entry in directory.iterdir():
            match = VERSION_FILE.match(entry.name)
            if match:
                numbers.append(int(match.group(1)))
        return sorted(numbers)

    def latest_version(self, card_id: str) -> int:
        numbers = self.versions(card_id)
        return numbers[-1] if numbers else 0

    def next_version(self, card_id: str) -> int:
        return self.latest_version(card_id) + 1

    def list_cards(self) -> list[str]:
        cards_dir = self.run_dir / "cards"
        if not cards_dir.is_dir():
            return []
        return sorted(entry.name for entry in cards_dir.iterdir() if CARD_ID.match(entry.name))

    def get_card(self, card_id: str, version: int | None = None) -> dict:
        if version is None:
            version = self.latest_version(card_id)
            if version == 0:
                raise KeyError(f"no card {card_id} in {self.run_dir}")
        path = self._card_dir(card_id) / f"v{version}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _ancestors(self, card_id: str) -> set[str]:
        seen: set[str] = set()
        frontier = [card_id]
        while frontier:
            current = frontier.pop()
            for parent in self.get_card(current)["parent_ids"]:
                if parent not in seen:
                    seen.add(parent)
                    frontier.append(parent)
        return seen

    def put_card(self, card: dict) -> Path:
        validate("theory_card", card)
        card_id = card["id"]
        expected = self.next_version(card_id)
        if card["version"] != expected:
            raise ValueError(f"{card_id}: card version is {card['version']} but the next version is {expected}")
        for parent in card["parent_ids"]:
            if parent == card_id:
                raise ValueError(f"{card_id} lists itself as a parent")
            if not (self.run_dir / "cards" / parent).is_dir():
                raise ValueError(f"{card_id} names unknown parent {parent}")
        ancestors: set[str] = set()
        for parent in card["parent_ids"]:
            ancestors |= self._ancestors(parent) | {parent}
        if card_id in ancestors:
            raise ValueError(f"{card_id}: parent_ids would create a cycle")

        path = self._card_dir(card_id) / f"v{expected}.json"
        data = (json.dumps(card, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        _write_new(path, data)
        self.ledger.append("card_version", "store", {
            "card_id": card_id,
            "version": expected,
            "path": path.relative_to(self.run_dir).as_posix(),
            "sha256": _sha256(data),
            "stage": card["status"]["stage"],
        })
        return path

    def put_artifact(self, kind: str, name: str, content, card_id: str | None = None) -> dict:
        if not ARTIFACT_PART.match(kind) or not ARTIFACT_PART.match(name):
            raise ValueError(f"artifact kind and name must be plain file-name parts, got {kind!r}, {name!r}")
        if card_id is not None:
            self._check_card_id(card_id)
        if isinstance(content, Path):
            data = content.read_bytes()
        elif isinstance(content, (bytes, bytearray)):
            data = bytes(content)
        else:
            raise TypeError(f"artifact content must be bytes or a Path, got {content.__class__.__name__}")

        path = self.run_dir / "artifacts" / kind / name
        _write_new(path, data)
        sha = _sha256(data)
        event = {
            "kind": kind,
            "name": name,
            "path": path.relative_to(self.run_dir).as_posix(),
            "sha256": sha,
            "bytes": len(data),
        }
        if card_id is not None:
            event["card_id"] = card_id
        self.ledger.append("artifact", "store", event)
        return {"path": path, "sha256": sha}

    def set_verdict(self, card_id: str, stage: str, verdict: str, reasons: list[str], by_calls: list[str]) -> dict:
        current = self.get_card(card_id)
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        if STAGES.index(stage) < STAGES.index(current["status"]["stage"]):
            raise ValueError(f"{card_id}: verdict for {stage} after the card reached {current['status']['stage']}")
        record = {
            "card_id": card_id,
            "card_version": self.next_version(card_id),
            "stage": stage,
            "verdict": verdict,
            "reasons": list(reasons),
            "by_calls": list(by_calls),
        }
        validate("verdict", record)

        updated = copy.deepcopy(current)
        updated["version"] = record["card_version"]
        updated["status"]["stage"] = stage
        updated["status"]["verdicts"][stage] = {
            "verdict": verdict,
            "reasons": record["reasons"],
            "by_calls": record["by_calls"],
        }
        self.put_card(updated)
        self.ledger.append("verdict", "store", record)
        return record

    def lineage(self) -> dict:
        cards = {card_id: self.get_card(card_id) for card_id in self.list_cards()}
        children: dict[str, list[str]] = {card_id: [] for card_id in cards}
        for card_id, card in cards.items():
            for parent in card["parent_ids"]:
                if parent not in cards:
                    raise ValueError(f"{card_id} names unknown parent {parent}")
                children[parent].append(card_id)
        nodes = {
            card_id: {
                "title": card["title"],
                "stage": card["status"]["stage"],
                "parent_ids": sorted(card["parent_ids"]),
                "child_ids": sorted(children[card_id]),
            }
            for card_id, card in cards.items()
        }
        roots = sorted(card_id for card_id, card in cards.items() if not card["parent_ids"])
        return {"nodes": nodes, "roots": roots}

    def file_problems(self) -> list[str]:
        """Compare files on disk with the sha256 recorded in the ledger, in both directions."""
        recorded: dict[str, str] = {}
        for event in self.ledger.iter_events():
            if event["type"] in ("card_version", "artifact"):
                data = event["data"]
                recorded[data["path"]] = data["sha256"]

        problems = []
        for rel, sha in sorted(recorded.items()):
            path = self.run_dir / rel
            if not path.is_file():
                problems.append(f"recorded file missing: {rel}")
            elif _sha256(path.read_bytes()) != sha:
                problems.append(f"sha256 mismatch: {rel}")

        on_disk = list((self.run_dir / "cards").glob("*/v*.json"))
        artifacts_dir = self.run_dir / "artifacts"
        if artifacts_dir.is_dir():
            on_disk += [p for p in artifacts_dir.rglob("*") if p.is_file()]
        for path in sorted(on_disk):
            rel = path.relative_to(self.run_dir).as_posix()
            if rel not in recorded:
                problems.append(f"file not recorded in ledger: {rel}")
        return problems
