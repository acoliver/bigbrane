"""Planted cards: format, sealing and the leak check (DISCOVERY-PIPELINE-PLAN.md section 13, Phase 9).

A planted set is a directory ``hybrid/fixtures/planted/<set>/`` (sets ``main``, 45 cards, and ``held_back``, 18 cards)
holding:

- ``cards/<key>.json``: one theory card per file in the current ``theory_card`` schema, named by a neutral key
  (``pc-`` and eight hex digits). The card is what the models see, so it carries nothing about its origin: lens
  ``neutral``, no references, no source transplant, no generation or stage fields, status S0 with no verdicts, and it
  passes the Phase 4 anonymization check (hybrid/research/calibration.py) plus the planted terms below and the card's
  own source family name and aliases.
- ``SEAL.json``: the sha256 of every card file and of the sealed expectations file, written by ``seal``. It is the
  hash committed in the repository; every planted run also records it as its first ledger event.

The expectations (type, specification, source family, origin, batch) live in ``<sealed dir>/<set>.sealed.json``,
outside the workspace (default ``~/.bigbrane/planted``, or ``BIGBRANE_PLANTED_DIR``), readable by its owner only. No
model role reads files: models see prompts, and no prompt is built from this file. The driver reads it for three
things only (which cards run M and C under override, the batch order, and the origins it must seed into the run's
outcome store) and unseals it for scoring after every run has finished.

Sealed file format (``FORMAT``)::

    {"format": "bigbrane.planted/1", "set": "main",
     "cards": {"pc-1a2b3c4d": {"type": "T4", "subtype": "capacity", "spec": "...", "source_family": "...",
                               "aliases": ["..."], "batch": 1, "origin": null, "notes": "..."}}}

``origin`` is required for T2 and T6 and absent otherwise: ``{"run": "run-...", "card_id": "card-..."}`` for a dead
record of an earlier run (seeded into the run's store copy from ``runs/<run>`` when the store lacks it), or
``{"planted": "pc-..."}`` for a card of the same set in an earlier batch.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hybrid.research.calibration import anonymization_problems
from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.signature import card_signature, jaccard, signature, tokens
from hybrid.research.stages.common import card_text_for_similarity
from hybrid.research.stages.diversity import most_similar

ROOT = Path(__file__).resolve().parents[2]
# Sealed expectations must live outside this tree (tests point it at a subtree, since their temporary directories
# are inside the repository).
WORKSPACE = ROOT
PLANTED_DIR = ROOT / "hybrid" / "fixtures" / "planted"
FORMAT = "bigbrane.planted/1"
SEAL_FILE = "SEAL.json"
CARDS_DIR = "cards"
SEALED_ENV = "BIGBRANE_PLANTED_DIR"
DEFAULT_SEALED_DIR = Path.home() / ".bigbrane" / "planted"
KEY = re.compile(r"^pc-[0-9a-f]{8}$")
TYPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
T4_SUBTYPES = ("sign", "cost_exponent", "capacity")
# Plan section 13.1 counts; the held-back set has 3 per type (one T4 card of each subtype). DECISIONS T7c: no T7 card
# survived review, so both sets carry none.
SET_COUNTS = {"main": {"T1": 10, "T2": 6, "T3": 8, "T4": 9, "T5": 6, "T6": 6, "T7": 0},
              "held_back": {**{t: 3 for t in TYPES}, "T7": 0}}
SUBTYPE_COUNTS = {"main": {s: 3 for s in T4_SUBTYPES}, "held_back": {s: 1 for s in T4_SUBTYPES}}
ORIGIN_TYPES = ("T2", "T6")
LENS = "neutral"
ENTRY_KEYS = {"type", "subtype", "spec", "source_family", "aliases", "batch", "origin", "notes"}
# Fields a planted card may not carry: filled by generation, the classifier or a later stage, or naming a source.
FORBIDDEN_FIELDS = ("arm", "arm_structure", "commitments", "archive_cell", "difference_statement",
                    "non_reduction_argument", "source_transplant")
# Words that give away a planted card's type or test purpose, and names of known mechanisms the calibration list does
# not cover. Matched as whole words, case-insensitive, anywhere in the card (on top of the calibration list and the
# families.toml ids).
PLANTED_TERMS = (
    "planted", "renamed", "rename", "trivial variant", "magic", "magic-x", "oracle", "known-failed", "known failed",
    "subtle error", "sign error", "control card", "held back", "held-back", "expected outcome", "expected stop",
    "kill code", "sealed", "decoy", "trap", "rwkv", "retnet", "retention network", "hyena", "xlstm", "fast weight",
    "fast weights", "titans", "mesanet", "griffin", "s4d", "selective scan", "cky", "cyk", "chart parser",
    "stack-augmented", "stack augmented", "neural turing", "memory network", "kv cache", "key-value cache",
    "performer", "longformer", "bert", "gpt", "llama", "recursive least squares",
)
TYPE_TAG = re.compile(r"\bT[1-7]\b")


class SealError(ValueError):
    """A planted set, its seal or its sealed expectations are not usable. Nothing has run."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sealed_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(SEALED_ENV)
    return Path(env).expanduser().resolve() if env else DEFAULT_SEALED_DIR


def sealed_path(set_name: str, directory: Path | None = None) -> Path:
    return sealed_dir(directory) / f"{set_name}.sealed.json"


def set_dir(set_name: str) -> Path:
    return PLANTED_DIR / set_name


# ----- cards --------------------------------------------------------------------------------------------------------

def card_problems(card: dict, *, extra_terms: tuple[str, ...] = ()) -> list[str]:
    """Why a card cannot be planted: schema, fields that are filled later or name a source, status, lens, and every
    anonymization finding (family names, give-away terms, citations, attention-style part names in the mechanism,
    planted terms, a type tag such as T3, and the extra terms: the card's own source family and aliases)."""
    problems = []
    try:
        validate("theory_card", card)
    except SchemaViolation as exc:
        problems.append(f"not a valid theory card: {str(exc)[:300]}")
    for name in FORBIDDEN_FIELDS:
        if name in card:
            problems.append(f"field {name} must be absent")
    if card.get("status") != {"stage": "S0", "verdicts": {}}:
        problems.append("status must be {stage: S0, verdicts: {}}")
    if card.get("version") != 1 or card.get("parent_ids") != []:
        problems.append("version must be 1 and parent_ids empty")
    if card.get("lens") != LENS:
        problems.append(f"lens must be {LENS!r}")
    terms = tuple(t for t in (*PLANTED_TERMS, *extra_terms) if t and t.strip())
    problems += anonymization_problems(card, extra_terms=terms)
    tag = TYPE_TAG.search(json.dumps(card, ensure_ascii=False))
    if tag:
        problems.append(f"names a planted type: {tag.group(0)!r}")
    return sorted(set(problems))


@dataclass
class PlantedSet:
    name: str
    directory: Path
    cards: dict[str, dict]
    raw: dict[str, bytes]
    seal: dict | None

    def card_hashes(self) -> dict[str, str]:
        return {key: sha256_bytes(raw) for key, raw in sorted(self.raw.items())}


def read_cards(directory: Path) -> tuple[dict[str, dict], dict[str, bytes]]:
    cards_dir = Path(directory) / CARDS_DIR
    if not cards_dir.is_dir():
        raise SealError(f"{cards_dir} does not exist")
    cards, raw = {}, {}
    for path in sorted(cards_dir.glob("*.json")):
        key = path.stem
        if not KEY.match(key):
            raise SealError(f"{path.name}: card files are named pc-<8 hex digits>.json")
        raw[key] = path.read_bytes()
        try:
            cards[key] = json.loads(raw[key])
        except ValueError as exc:
            raise SealError(f"{path.name}: not JSON: {exc}") from exc
    if not cards:
        raise SealError(f"{cards_dir} holds no cards")
    return cards, raw


def load_set(directory: Path, *, name: str | None = None, require_seal: bool = True) -> PlantedSet:
    """The cards of a set and its SEAL.json; every card file must match the hash the seal recorded."""
    directory = Path(directory).resolve()
    cards, raw = read_cards(directory)
    seal = None
    seal_path = directory / SEAL_FILE
    if seal_path.is_file():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    elif require_seal:
        raise SealError(f"{seal_path} is missing: seal the set first (python -m hybrid.research planted seal ...)")
    planted = PlantedSet(name or directory.name, directory, cards, raw, seal)
    if seal is not None:
        problems = []
        recorded = seal.get("cards", {})
        found = planted.card_hashes()
        problems += [f"{key}: card file changed since sealing" for key in sorted(set(recorded) & set(found))
                     if recorded[key] != found[key]]
        problems += [f"{key}: sealed card file missing" for key in sorted(set(recorded) - set(found))]
        problems += [f"{key}: card file not in the seal" for key in sorted(set(found) - set(recorded))]
        if problems:
            raise SealError(problems)
    return planted


# ----- expectations --------------------------------------------------------------------------------------------------

def expectation_problems(data: dict, keys: set[str], set_name: str) -> list[str]:
    """Format, per-card entries and, for the main and held-back sets, the plan's counts per type and T4 subtype."""
    problems = []
    if data.get("format") != FORMAT:
        problems.append(f"format must be {FORMAT!r}")
    if data.get("set") != set_name:
        problems.append(f"set must be {set_name!r}, got {data.get('set')!r}")
    entries = data.get("cards")
    if not isinstance(entries, dict):
        return problems + ["cards must map card keys to entries"]
    problems += [f"{key}: no expectation for this card" for key in sorted(keys - set(entries))]
    problems += [f"{key}: expectation for a card that is not in the set" for key in sorted(set(entries) - keys)]
    for key, entry in sorted(entries.items()):
        if not isinstance(entry, dict):
            problems.append(f"{key}: entry must be an object")
            continue
        unknown = set(entry) - ENTRY_KEYS
        if unknown:
            problems.append(f"{key}: unknown fields {sorted(unknown)}")
        kind = entry.get("type")
        if kind not in TYPES:
            problems.append(f"{key}: type must be one of {TYPES}")
            continue
        for field in ("spec", "source_family"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                problems.append(f"{key}: {field} must be a non-empty string")
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
            problems.append(f"{key}: aliases must be a list of non-empty strings")
        batch = entry.get("batch", 1)
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            problems.append(f"{key}: batch must be an integer >= 1")
        if (kind == "T4") != (entry.get("subtype") is not None):
            problems.append(f"{key}: subtype is required for T4 and only for T4")
        elif kind == "T4" and entry["subtype"] not in T4_SUBTYPES:
            problems.append(f"{key}: subtype must be one of {T4_SUBTYPES}")
        origin = entry.get("origin")
        if kind in ORIGIN_TYPES:
            problems += [f"{key}: {p}" for p in origin_problems(origin, entries, key, batch)]
        elif origin is not None:
            problems.append(f"{key}: only {', '.join(ORIGIN_TYPES)} cards name an origin")
    if set_name in SET_COUNTS:
        counts = {t: sum(1 for e in entries.values() if isinstance(e, dict) and e.get("type") == t) for t in TYPES}
        if counts != SET_COUNTS[set_name]:
            problems.append(f"the {set_name} set needs {SET_COUNTS[set_name]} cards per type, got {counts}")
        subtypes = {s: sum(1 for e in entries.values() if isinstance(e, dict) and e.get("subtype") == s)
                    for s in T4_SUBTYPES}
        if subtypes != SUBTYPE_COUNTS[set_name]:
            problems.append(f"the {set_name} set needs T4 subtypes {SUBTYPE_COUNTS[set_name]}, got {subtypes}")
    return problems


def origin_problems(origin, entries: dict, key: str, batch: int) -> list[str]:
    if not isinstance(origin, dict):
        return ["origin is required: {run, card_id} of a dead record, or {planted: key} of an earlier batch"]
    if set(origin) == {"run", "card_id"}:
        if not (isinstance(origin["run"], str) and origin["run"] and re.match(r"^card-[0-9a-f]{8}$",
                                                                               str(origin["card_id"]))):
            return ["origin {run, card_id} needs a run name and a card id"]
        return []
    if set(origin) == {"planted"}:
        other = entries.get(origin["planted"])
        if other is None:
            return [f"origin names {origin['planted']}, which is not in this set"]
        if not isinstance(other, dict) or other.get("batch", 1) >= batch:
            return [f"origin {origin['planted']} must be in an earlier batch than this card"]
        return []
    return ["origin must be {run, card_id} or {planted}"]


def leak_problems(cards: dict[str, dict], entries: dict[str, dict]) -> dict[str, list[str]]:
    """Per card, the anonymization findings with its own source family name and aliases added to the banned terms."""
    found = {}
    for key, card in sorted(cards.items()):
        entry = entries.get(key, {})
        extra = tuple(t.lower() for t in (entry.get("source_family", ""), *entry.get("aliases", [])) if t)
        problems = card_problems(card, extra_terms=extra)
        if problems:
            found[key] = problems
    return found


def find_store_record(store, origin: dict):
    """The current record of an origin {run, card_id} in an outcome store, or None."""
    for stored in store.current():
        if stored.record["run"] == origin["run"] and stored.record["card_id"] == origin["card_id"]:
            return stored
    return None


def set_problems(cards: dict[str, dict], entries: dict[str, dict], *, store=None, jaccard_min: float = 0.6,
                 diversity_threshold: float = 0.6) -> list[str]:
    """Collisions that would make a card die for a reason its type does not intend, and origins that would not test
    what T2 and T6 test:

    - two cards with one mechanism signature, or a TF-IDF cosine at or above the diversity threshold, unless one is the
      T6 variant of the other (G would kill the second as a duplicate);
    - a card whose signature is already in the outcome store, unless it is a T6 card and that record is its origin;
    - a T6 card whose signature differs from its origin's (it must be a surface change);
    - a T2 card whose signature equals its origin's, or whose signature tokens are below ``jaccard_min`` from it (R3
      would not apply).
    Store origins are checked only when a store is given."""
    problems = []
    signatures = {key: mechanism_key(card_signature(card)["skeleton"]) for key, card in cards.items()}

    def planted_pair(a: str, b: str) -> bool:
        for one, other in ((a, b), (b, a)):
            entry = entries.get(one, {})
            if entry.get("type") == "T6" and (entry.get("origin") or {}).get("planted") == other:
                return True
        return False

    keys = sorted(cards)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if signatures[a]["signature"] == signatures[b]["signature"] and not planted_pair(a, b):
                problems.append(f"{a} and {b} share a mechanism signature")
    for i, a in enumerate(keys):
        others = {b: card_text_for_similarity(cards[b]) for b in keys[i + 1:]}
        while others:
            nearest, cosine = most_similar(card_text_for_similarity(cards[a]), others)
            if nearest is None or cosine < diversity_threshold:
                break
            if not planted_pair(a, nearest):
                problems.append(f"{a} and {nearest} have TF-IDF cosine {cosine:.2f}, at or above {diversity_threshold}")
            others.pop(nearest)
    for key in keys:
        entry = entries.get(key, {})
        origin = entry.get("origin") or {}
        sig = signatures[key]
        if store is not None:
            for stored in store.current():
                if stored_key(stored.record)["signature"] != sig["signature"]:
                    continue
                own = (entry.get("type") == "T6" and stored.record["run"] == origin.get("run")
                       and stored.record["card_id"] == origin.get("card_id"))
                if not own:
                    problems.append(f"{key}: mechanism signature equals that of the stored record "
                                    f"{stored.record['run']}/{stored.record['card_id']}")
        if entry.get("type") not in ORIGIN_TYPES:
            continue
        if "planted" in origin:
            if origin["planted"] not in cards:
                continue
            other = signatures[origin["planted"]]
        elif store is not None:
            stored = find_store_record(store, origin)
            if stored is None:
                continue
            other = stored_key(stored.record)
        else:
            continue
        same = sig["signature"] == other["signature"]
        if entry["type"] == "T6" and not same:
            problems.append(f"{key}: a T6 card must keep its origin's mechanism signature")
        if entry["type"] == "T2":
            if same:
                problems.append(f"{key}: a T2 card must change an equation, so its signature differs from its origin")
            elif jaccard(sig["tokens"], other["tokens"]) < jaccard_min:
                problems.append(f"{key}: signature tokens are below Jaccard {jaccard_min} from the origin, so R3 "
                                f"would not apply")
    return problems


def mechanism_key(skeleton: dict) -> dict:
    """Signature and tokens of a skeleton without the five commitments: the duplicate key stage G uses
    (hybrid/research/archive.py), since a planted card gets its commitments only from the classifier."""
    bare = {key: value for key, value in skeleton.items() if key != "commitments"}
    return {"signature": signature(bare), "tokens": tokens(bare)}


def stored_key(record: dict) -> dict:
    return mechanism_key(record["mechanism_signature"]["skeleton"])


# ----- sealing -------------------------------------------------------------------------------------------------------

def inside_workspace(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(WORKSPACE)
        return True
    except ValueError:
        return False


def check_unsealed(*, set_name: str, cards: dict[str, dict], data: dict, store=None, jaccard_min: float = 0.6,
                   diversity_threshold: float = 0.6, runs_root: Path | None = None) -> list[str]:
    """Every check sealing makes on a written set, without writing anything: the expectations' format and counts,
    the leak check, and (when those pass) the set checks and the store origins."""
    problems = expectation_problems(data, set(cards), set_name)
    entries = data.get("cards") if isinstance(data.get("cards"), dict) else {}
    for key, found in leak_problems(cards, entries).items():
        problems += [f"{key}: {p}" for p in found]
    if not problems:
        problems += set_problems(cards, entries, store=store, jaccard_min=jaccard_min,
                                 diversity_threshold=diversity_threshold)
        if store is not None:
            problems += store_origin_problems(entries, store, runs_root)
    return problems


def seal(*, set_name: str, directory: Path, expectations_path: Path, sealed_directory: Path | None = None,
         store=None, jaccard_min: float = 0.6, diversity_threshold: float = 0.6, runs_root: Path | None = None) -> dict:
    """Check a written set and seal it: the expectations file moves to the sealed directory (owner-only), and
    SEAL.json with every hash is written into the set directory. Refuses to overwrite an existing seal."""
    directory = Path(directory).resolve()
    cards, raw = read_cards(directory)
    target = sealed_path(set_name, sealed_directory)
    if inside_workspace(target):
        raise SealError(f"the sealed file must live outside the workspace, got {target}")
    if target.exists() or (directory / SEAL_FILE).exists():
        raise SealError(f"{set_name} is already sealed ({target} or {directory / SEAL_FILE} exists)")
    source = Path(expectations_path).resolve()
    data_bytes = source.read_bytes()
    data = json.loads(data_bytes)
    problems = check_unsealed(set_name=set_name, cards=cards, data=data, store=store, jaccard_min=jaccard_min,
                              diversity_threshold=diversity_threshold, runs_root=runs_root)
    if problems:
        raise SealError(problems)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data_bytes)
    record = {"format": FORMAT, "set": set_name, "sealed_file": target.name, "sealed_sha256": sha256_bytes(data_bytes),
              "cards": {key: sha256_bytes(value) for key, value in sorted(raw.items())},
              "sealed_at": datetime.now(timezone.utc).isoformat()}
    (directory / SEAL_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if source != target:
        source.unlink()
    return record


def store_origin_problems(entries: dict, store, runs_root: Path | None = None) -> list[str]:
    """A store origin must be a dead record in the store, or a dead card of a run directory that can seed it."""
    from hybrid.research.outcome_records import RunView, build_record

    problems = []
    runs_root = Path(runs_root or ROOT / "runs")
    for key, entry in sorted(entries.items()):
        origin = entry.get("origin") or {}
        if "run" not in origin:
            continue
        stored = find_store_record(store, origin)
        if stored is not None:
            if not stored.dead:
                problems.append(f"{key}: origin {origin['run']}/{origin['card_id']} is not dead in the store")
            continue
        run_dir = runs_root / origin["run"]
        if not (run_dir / "ledger.jsonl").is_file():
            problems.append(f"{key}: origin run {origin['run']} is neither in the store nor under {runs_root}")
            continue
        view = RunView(run_dir)
        if origin["card_id"] not in view.versions:
            problems.append(f"{key}: origin card {origin['card_id']} is not in {origin['run']}")
        elif build_record(view, origin["card_id"], "planted_seed")["outcome"] != "dead":
            problems.append(f"{key}: origin {origin['run']}/{origin['card_id']} did not die")
    return problems


@dataclass
class Sealed:
    path: Path
    sha256: str
    data: dict

    @property
    def entries(self) -> dict[str, dict]:
        return self.data["cards"]


def read_sealed(planted: PlantedSet, sealed_directory: Path | None = None) -> Sealed:
    """The sealed expectations of a set, after checking the owner-only mode and the hash SEAL.json committed."""
    if planted.seal is None:
        raise SealError(f"{planted.directory} has no {SEAL_FILE}")
    path = sealed_dir(sealed_directory) / planted.seal["sealed_file"]
    if not path.is_file():
        raise SealError(f"sealed expectations not found at {path}")
    if inside_workspace(path):
        raise SealError(f"sealed expectations must live outside the workspace, found at {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise SealError(f"{path} is readable by group or others (mode {oct(mode)}); run: chmod 600 {path}")
    data_bytes = path.read_bytes()
    digest = sha256_bytes(data_bytes)
    if digest != planted.seal["sealed_sha256"]:
        raise SealError(f"{path} differs from the hash in {planted.directory / SEAL_FILE}")
    data = json.loads(data_bytes)
    problems = expectation_problems(data, set(planted.cards), planted.seal["set"])
    if problems:
        raise SealError(problems)
    return Sealed(path, digest, data)
