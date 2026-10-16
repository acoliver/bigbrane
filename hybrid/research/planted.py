"""The planted-card test of the whole pipeline (DISCOVERY-PIPELINE-PLAN.md section 13, Phase 9).

``python -m hybrid.research planted --set main --reason "..."`` runs every card of a sealed planted set
(hybrid/research/planted_seal.py) through the pipeline with the live configuration, ``--runs`` times (default 2), and
scores the result against the sealed expectations (hybrid/research/planted_metrics.py).

Entry point. Planted cards replace the drafting call of generation stage G: each card enters at G's admission step
(the classifier places it in the archive; then the out-of-scope family, closed cell, known family without a real
difference and signature duplicate gates; then the TF-IDF duplicate check and the lineage caps of S0), exactly where
a drafted card would, and from there runs S1 (the entry check), R, S2 (family exclusion, own history, difference
statement and R3), M (with the transformer profile), C and the target gate. The card's proposer, who answers the
refinement dialogue and writes the difference statement, is the ``explorer`` role, as for a drafted card. A card
that passes C is stopped before S4 and recorded as having reached T (test design, Phase 10, is not built).

Live configuration. ``planted_config`` takes run.toml and switches on what section 13 tests: s3.mode math,
math.profile, the target gate, refine, the S2 family exclusion and difference statement, the outcome store with R3,
and generation mode arms (for the G admission gates). Escapes are off (p_escape 0), since they add cards that were not
planted; budgets per idea and per lineage are the config's. The run budget is raised so that only the per-idea caps
bind: max_cards is the number of planted cards and max_calls the per-idea call cap times that number.

Isolation. Every run gets its own copy of the outcome store (``run<i>-outcomes`` under the run root, copied from
``--store``, default results/outcomes), so planted cards never reach the live store and the second run does not see
the first run's records. Run directories sit one level below ``runs/planted-<UTC>/``, where the outcome backfill
(which reads ``runs/*/``) does not look.

T2 and T6 origins. A T6 card (a trivial variant of a killed idea) and a T2 card (a known-failed idea without a real
change) name the dead idea they vary in the sealed file. An origin ``{run, card_id}`` must be a dead record in the
run's store copy; when the copy lacks it, the driver builds the record from ``runs/<run>`` read-only with
hybrid.research.outcome_records (the same code as the backfill) and appends it to the copy, and refuses to start when
neither has it. An origin ``{planted: key}`` is a card of the same set in an earlier batch: cards are injected batch by
batch, the next batch only after every card of the previous one has ended, and the driver records whether the origin
died (its outcome record is in the copy by then).

T5. Cards of valid known mechanisms also run M and C under the operator override, with the profile on and the target
gate off (as in calibration), in ``run<i>-override/<key>``; the plan expects both to pass while P flags the card as
known in the main line. The override runs the card as it entered with the resources its author stated in the main
line's stage R (FIX-16): stage M refuses a card without them, and the override has no stage R of its own.

Mixing. Within a batch the cards are injected in a seeded shuffle (seed from the set name and the run index), stripped
of every origin field: a card enters with a fresh card id, lens ``neutral`` and nothing from the sealed file.

Preflights (exit 2, nothing created): the free-space check and the local-model check (in ``__main__``), the gateway
check, the set's SEAL.json against its card files, the sealed file's mode and hash, the leak check with each card's
source family terms, the set checks (signature and TF-IDF collisions, T2 and T6 origins), the check files against
hybrid/math/MANIFEST.json, and an output or run directory that is not empty.

Check integrity. The driver hashes the check files (hybrid/math, its MANIFEST, the family, archive and limit
configs, the schemas, the prompts and prompt sections, the planted cards, SEAL.json and the sealed file) before the
first run and after every run, and compares the check-file hashes each executed math check recorded with the manifest
at the start. Any difference fails acceptance.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hybrid.executor.job import Caps
from hybrid.math import manifest
from hybrid.research import calibration, lineage
from hybrid.research import planted_metrics as metrics
from hybrid.research import planted_seal as ps
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import Orchestrator
from hybrid.research.parallel import check_parallel, run_limited
from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.outcomes import JSONL, OutcomeStore
from hybrid.research.stages import m_math
from hybrid.research.stages.common import (Budget, CodeRunner, RunBudgetExhausted, StageContext, StageError,
                                           new_card_id)
from hybrid.research.store import Store
from hybrid.slots import VMUnavailable, job_slots_from_config, slot_line

ROOT = ps.ROOT
ACTOR = "planted"
OPERATOR = "operator"
PROPOSER_ROLE = "explorer"
STORE_SOURCE = ROOT / "results" / "outcomes"
RUNS_ROOT = ROOT / "runs"
OVERRIDE_STATEMENT = ("planted-card test: stages M and C run under the operator override for a card of a valid known "
                      "mechanism (plan section 13.1, T5); stages S0 to S2 are bypassed and results are not evidence "
                      "that the card passed any gate")
PLANTED_STATEMENT = ("planted-card test (plan section 13): the cards below enter at the admission step of generation "
                     "stage G instead of a drafting call; everything after admission is the live pipeline")
# Files that decide checks and gates, hashed before the first run and after each run (plan section 13.2, check
# integrity). Relative to the repository root.
AUDIT_GLOBS = ("hybrid/math/*.py", "hybrid/math/MANIFEST.json", "hybrid/config/families.toml",
               "hybrid/config/archive.toml", "hybrid/config/limits.toml", "hybrid/research/schemas/*.json",
               "hybrid/research/prompts/*.md", "hybrid/research/prompt_sections/*.md")
DEFAULT_WALL_S = 7 * 24 * 3600


class PlantedError(ValueError):
    """The planted test cannot start. Nothing has run."""


# ----- configuration ------------------------------------------------------------------------------------------------

def planted_config(cfg: dict, *, store_path: Path, n_cards: int, max_wall_s: int | None = None) -> dict:
    """run.toml with every switch section 13 tests turned on (see the module docstring)."""
    cfg = copy.deepcopy(cfg)
    cfg["s3"]["mode"] = "math"
    cfg["math"]["profile"] = True
    cfg["target"]["enabled"] = True
    cfg["refine"]["enabled"] = True
    cfg["s2"].update(family_exclusion=True, difference_statement=True)
    cfg["outcomes"].update(enabled=True, enforce_r3=True, path=str(Path(store_path).resolve()))
    cfg["generation"].update(mode="arms", draft_only=False, quota={})
    cfg["run"]["n_explorers"] = 1
    cfg["explore"].update(p_escape=0.0, escape_reserve=0.0, graveyard_runs=[])
    cap = cfg["budgets"]["idea"]["model_calls"]
    cfg["budgets"]["max_cards"] = n_cards
    cfg["budgets"]["max_calls"] = max(cfg["budgets"]["max_calls"], n_cards * (cap + 1))
    cfg["budgets"]["max_tokens"] = max(cfg["budgets"]["max_tokens"], n_cards * cap * 100_000)
    cfg["budgets"]["max_wall_s"] = max_wall_s or DEFAULT_WALL_S
    return cfg


# ----- the run's outcome store ------------------------------------------------------------------------------------

def prepare_store(source: Path, target: Path, origins: dict[str, dict], runs_root: Path) -> dict:
    """Copy the outcome store into ``target`` and make sure every store origin ({run, card_id}) is a dead record in
    the copy, building it read-only from ``runs_root/<run>`` when the copy lacks it. Returns what was seeded."""
    target = Path(target)
    if target.exists():
        raise PlantedError(f"{target} exists; every planted run starts from a fresh store copy")
    target.mkdir(parents=True)
    source_jsonl = Path(source) / JSONL
    if source_jsonl.is_file():
        shutil.copyfile(source_jsonl, target / JSONL)
    store = OutcomeStore(target)
    ok, bad_seq = store.verify()
    if not ok:
        raise PlantedError(f"the outcome store copied from {source} has a broken hash chain at seq {bad_seq}")
    present, seeded, problems = [], [], []
    for key, origin in sorted(origins.items()):
        stored = ps.find_store_record(store, origin)
        if stored is None:
            problems += [f"{key}: {p}" for p in ps.store_origin_problems({key: {"origin": origin}}, store, runs_root)]
            if problems:
                continue
            line = store.append(build_record(RunView(Path(runs_root) / origin["run"]), origin["card_id"],
                                             "planted_seed"))
            seeded.append({"key": key, **origin, "seq": line["seq"] if line else None})
        elif not stored.dead:
            problems.append(f"{key}: origin {origin['run']}/{origin['card_id']} is not dead in the store")
        else:
            present.append({"key": key, **origin})
    if problems:
        raise PlantedError("cannot seed the T2 and T6 origins: " + "; ".join(problems))
    return {"source": str(source), "records": len(store.current()), "present": present, "seeded": seeded}


# ----- admission at G -----------------------------------------------------------------------------------------------

def synthetic_call_id(card_id: str) -> str:
    """created_by_call for a planted card: no model call drafted it, so this id names no llm_call event."""
    return "call-" + hashlib.sha256(f"planted:{card_id}".encode()).hexdigest()[:12]


def entering_card(card: dict, card_id: str) -> dict:
    return dict(copy.deepcopy(card), id=card_id, version=1, parent_ids=[], created_by_call=synthetic_call_id(card_id),
                lens=ps.LENS, status={"stage": "S0", "verdicts": {}})


async def admit_planted(ctx: StageContext, card: dict, *, proposer_model: str | None) -> bool:
    """Stage G's admission for a card that was not drafted: classify, gate, store and give the S0 verdict."""
    from hybrid.research.stages.g_generate import classify_and_gate
    from hybrid.research.stages.s0_explore import ExploreTask, admit

    classified = await classify_and_gate(ctx, card)
    task = ExploreTask(op="new", lens=card["lens"], lens_sources=[], role=PROPOSER_ROLE, explorer=0, seed=0)
    admission = admit(ctx, task, card, call_ids=[classified.call_id], model=proposer_model, papers=[], nearby=None,
                      user=json.dumps(card, sort_keys=True), dropped=[], dropped_sources=[], gate=classified.gate,
                      created_extra={"archive_cell": card["archive_cell"], "family": classified.family},
                      extra_reasons=classified.reasons)
    ctx.generation.archive.record(ctx.ledger, card=card, op="new", admitted=admission.passed,
                                  classifier_call=classified.call_id, overrides=classified.overrides,
                                  gate=admission.kill_code)
    return admission.passed


class PlantedOrchestrator(Orchestrator):
    """The orchestrator with one injector in place of the explorers. The injector admits each batch's cards at G in
    the given order and waits until every card of a batch has ended before the next batch. A card that passes C is
    stopped before S4 (``planted_reached``)."""

    def __init__(self, cfg: dict, run_dir: Path, *, deps, batches: list[list[tuple[str, dict]]],
                 proposer_model: str | None, origins: dict[str, str], ledger: Ledger):
        super().__init__(cfg, run_dir, llm=deps.llm, executor=deps.executor, bench=deps.bench, image=deps.image,
                         ledger=ledger, classify_prior_art=deps.classify_prior_art, math=deps.math)
        if self.generation is None or not self.math_mode:
            raise ValueError("the planted driver needs generation.mode arms and s3.mode math (planted_config)")
        self.ctx.lookup_source = deps.lookup_source
        self.batches = batches
        self.proposer_model = proposer_model
        self.origins = origins
        self.card_ids = {key: card["id"] for batch in batches for key, card in batch}

    async def _explorer(self, explorer: int) -> None:
        try:
            for number, batch in enumerate(self.batches, 1):
                if number > 1:
                    await self._drained()
                    if self.stop_event.is_set():
                        break
                    self._check_origins(batch)
                for key, card in batch:
                    if self.stop_event.is_set():
                        return
                    reason = self.budget.exhausted_reason()
                    if reason is not None:
                        self._stop(reason)
                        return
                    self.emit("planted_admit", {"key": key, "card_id": card["id"], "batch": number})
                    self.budget.cards += 1
                    self.pending += 1
                    try:
                        if await admit_planted(self.ctx, card, proposer_model=self.proposer_model):
                            self._enqueue("S1", card["id"])
                        else:
                            self._record_outcome(card["id"])
                    except StageError as exc:
                        self.emit("planted_admit_failed", {"key": key, "card_id": card["id"], "error": str(exc)})
                    except RunBudgetExhausted as exc:
                        self._stop(str(exc))
                    finally:
                        self.pending -= 1
                        self._wake()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # internal bug: stop the run and re-raise from run()
            self._fail(exc)
        finally:
            self.explorers_live -= 1
            if not self.stop_event.is_set():
                self._check_idle()

    async def _drained(self) -> None:
        while self.pending > 0 and not self.stop_event.is_set():
            self.slot_wakeup.clear()
            await self.slot_wakeup.wait()

    def _check_origins(self, batch: list[tuple[str, dict]]) -> None:
        for key, _ in batch:
            origin = self.origins.get(key)
            if origin is None:
                continue
            origin_id = self.card_ids[origin]
            card = self.store.get_card(origin_id) if origin_id in self.store.list_cards() else None
            dead = False
            if card is not None:
                dead = build_record(RunView(self.run_dir), origin_id, ACTOR)["outcome"] == "dead"
            self.emit("planted_origin_check", {"key": key, "origin": origin, "origin_card_id": origin_id,
                                               "origin_dead": dead})

    def _enqueue(self, stage: str, card_id: str) -> None:
        if stage == "S4":
            verdicts = self.store.get_card(card_id)["status"]["verdicts"]
            if any(verdicts.get(gate, {}).get("verdict") != "pass" for gate in ("M", "C")):
                raise AssertionError(f"{card_id} reached S4 without a pass at M and C")
            self.emit("planted_reached", {"card_id": card_id, "stage": "T"})
            return
        super()._enqueue(stage, card_id)


# ----- T5: M and C under the override ------------------------------------------------------------------------------

def override_card(planted_card: dict, card_id: str, store: Store) -> tuple[dict, int | None]:
    """FIX-16: the card a T5 override runs, and the main-line version its resources come from. A sealed card that
    states no resources gets those of the latest main-line version that states them (the author's answer in stage R,
    which runs before P); stage M would end it with card_incomplete otherwise. FIX-18: the same for its stated P3
    limits. Everything else is the card as it entered. A card the main line never asked keeps no resources (None),
    and M refuses it."""
    card = entering_card(planted_card, card_id)
    found = None
    if not isinstance(card.get("resources"), dict):
        for version in reversed(store.versions(card_id)):
            stated = store.get_card(card_id, version).get("resources")
            if isinstance(stated, dict):
                card["resources"] = copy.deepcopy(stated)
                found = version
                break
    if not isinstance(card.get("limits"), list):
        # FIX-18: the card's P3 limits come from the same author answer in stage R as its resources.
        for version in reversed(store.versions(card_id)):
            stated = store.get_card(card_id, version).get("limits")
            if isinstance(stated, list):
                card["limits"] = copy.deepcopy(stated)
                break
    return card, found


async def override_mc(*, key: str, card: dict, run_dir: Path, cfg: dict, reason: str,
                      deps_factory: Callable[[Ledger], Any], target_rule: dict,
                      resources_from: int | None = None) -> dict:
    """Run a card through M and C under the operator override (profile on, target gate off in M) in a new run
    directory; return the verdicts, kill codes, where it was killed, the separate target result and the calls."""
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PlantedError(f"{run_dir} is not empty")
    cfg = calibration.calibration_config(cfg)
    raw = (json.dumps(card, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    ledger.append("operator_override", OPERATOR, {
        "statement": OVERRIDE_STATEMENT, "reason": reason, "planted": {"key": key},
        "card_id": card["id"], "source_card": {"version": 1, "sha256": hashlib.sha256(raw).hexdigest(),
                                               "resources_from_version": resources_from},
        "source_verdicts": {stage: None for stage in ("S0", "S1", "S2", "S3")}, "source_parent_ids": [],
        "stages_run": ["M", "C"], "math_note": "C runs after M whatever M's verdict (planted T5 override)",
        "s5_mode": cfg["s5"]["mode"], "falsifiers": False, "routing": None})
    b = cfg["budgets"]
    budget = Budget(b["max_cards"], b["max_calls"], b["max_tokens"], b["max_wall_s"])
    ledger.append("run_start", "orchestrator", {"run_id": run_dir.name, "planted_override": True, "plumb": True,
                                                "resume": False, "budget": budget.snapshot(), "stages": ["M", "C"],
                                                "s3_mode": "math", "profile": True, "target_in_m": False})
    store.put_card(card)
    deps = deps_factory(ledger)
    ex = cfg["executor"]
    caps = Caps(wall_s=ex["wall_s"], memory_mb=ex["memory_mb"], pids=ex["pids"], tmp_mb=ex["tmp_mb"])
    ctx = StageContext(run_id=run_dir.name, run_dir=run_dir, ledger=ledger, store=store, llm=deps.llm,
                       runner=CodeRunner(deps.executor, run_dir, store, deps.image, caps), bench=None, cfg=cfg,
                       budget=budget, lenses={}, math=deps.math, lookup_source=deps.lookup_source)
    stopped = await calibration.run_math_stages(ctx, card["id"])
    record = calibration.outcome_record(ctx, key, card, 1, target_rule)
    verdicts = {stage: entry["verdict"] for stage, entry in record["stages"].items()}
    killed_at = next((stage for stage in ("M", "C") if stage in verdicts and verdicts[stage] != "pass"), None)
    codes = [code for stage in ("M", "C") for code in record["stages"].get(stage, {}).get("kill_codes", [])]
    result = {"M": verdicts.get("M"), "C": verdicts.get("C"), "kill_codes": list(dict.fromkeys(codes)),
              "killed_at": killed_at, "stopped": stopped, "profile_passed": record["profile_passed"],
              "target": record["target"]["status"], "target_coupled": record["target_coupled"]["status"],
              "model_calls": record["model_calls"], "run_dir": str(run_dir)}
    ctx.emit("run_end", {"planted_override": True, "key": key, **{k: result[k] for k in ("M", "C", "stopped")}})
    return result


# ----- results per card ---------------------------------------------------------------------------------------------

def executor_seconds(view: RunView, card_id: str) -> float:
    """Executor time of the card's jobs: toy examples and checks (``executor`` artifacts) and math checks (both runs)."""
    total = 0.0
    for entry in view.artifacts.get(card_id, []):
        if entry["kind"] not in ("executor", "math_check"):
            continue
        record = json.loads((view.dir / entry["path"]).read_text(encoding="utf-8"))
        if entry["kind"] == "executor":
            total += float(record.get("run_s") or 0.0)
        else:
            total += sum(float(run.get("run_s") or 0.0) for run in record.get("runs", []))
    return round(total, 3)


def card_result(view: RunView, key: str, card_id: str, *, reached: set[str], cfg: dict) -> dict:
    """What the run did with one planted card: stop point, kill codes, cost and budget overruns."""
    base = {"key": key, "card_id": card_id, "notes": [], "override": None}
    if card_id not in view.versions:
        return dict(base, stop="none", stage=None, outcome="not_admitted", kill_codes=[], model_calls=0, tokens=0,
                    executor_s=0.0, over_budget=[])
    record = build_record(view, card_id, ACTOR)
    if card_id in reached:
        stop, outcome = "T", "reached"
    elif record["outcome"] == "dead":
        stop, outcome = metrics.STAGE_STOP[record["terminal_stage"]], "dead"
    else:
        stop, outcome = "open", "open"
    budget = record["budget"] or {"model_calls": 0, "loops": {}}
    charges = [e["data"] for e in view.events if e["type"] == "budget_charge"
               and e["data"].get("card_id") == card_id and e["data"]["loop"] == lineage.MODEL_CALLS]
    over = []
    cap = cfg["budgets"]["idea"]["model_calls"]
    idea_used = max((charge["used"] for charge in charges), default=0)
    if idea_used > cap:
        over.append(f"the idea's model calls reached {idea_used}, above the per-idea cap {cap}")
    for loop, used in budget["loops"].items():
        if loop in lineage.IDEA_LOOPS and used > lineage.loop_cap(cfg, loop):
            over.append(f"{loop} rounds {used} above the cap {lineage.loop_cap(cfg, loop)}")
    return dict(base, stop=stop, stage=record["terminal_stage"], outcome=outcome, kill_codes=record["kill_codes"],
                model_calls=budget["model_calls"], tokens=sum(view.tokens.get(c["call_id"], 0) for c in charges),
                executor_s=executor_seconds(view, card_id), over_budget=over)


# ----- check-hash audit ---------------------------------------------------------------------------------------------

def audit_snapshot(planted: ps.PlantedSet, sealed: ps.Sealed) -> dict[str, str]:
    """sha256 of every check file (AUDIT_GLOBS), every planted set's cards and seal, the given set's directory (when
    it lies elsewhere) and the sealed file."""
    paths = [path for pattern in AUDIT_GLOBS for path in ROOT.glob(pattern)]
    paths += [path for path in ps.PLANTED_DIR.rglob("*") if path.is_file()]
    paths += [path for path in Path(planted.directory).rglob("*") if path.is_file()]
    files = {}
    for path in sorted(set(p.resolve() for p in paths)):
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        files[name] = manifest.sha256_file(path)
    files[f"sealed:{sealed.path.name}"] = manifest.sha256_file(sealed.path)
    return files


def snapshot_differences(before: dict[str, str], after: dict[str, str], when: str) -> list[str]:
    out = [f"{when}: {name} changed" for name in sorted(set(before) & set(after)) if before[name] != after[name]]
    out += [f"{when}: {name} removed" for name in sorted(set(before) - set(after))]
    out += [f"{when}: {name} added" for name in sorted(set(after) - set(before))]
    return out


def math_check_differences(run_dir: Path, recorded: dict[str, str]) -> tuple[int, list[str]]:
    """Every executed math check of a run records the hashes of the check files it staged; each must equal the
    manifest at the start of the test."""
    from hybrid.math.runner import LIBRARY_FILES

    view = RunView(run_dir)
    count, out = 0, []
    for entries in view.artifacts.values():
        for entry in entries:
            if entry["kind"] != "math_check":
                continue
            count += 1
            check = json.loads((view.dir / entry["path"]).read_text(encoding="utf-8"))
            for staged, digest in (check.get("check_hashes") or {}).items():
                source = LIBRARY_FILES.get(staged, staged)
                if recorded.get(source) != digest:
                    out.append(f"{view.run_id}/{entry['path']}: {staged} hashed {digest[:12]}, manifest "
                               f"{(recorded.get(source) or 'missing')[:12]}")
    return count, out


# ----- the test -----------------------------------------------------------------------------------------------------

@dataclass
class Deps:
    llm: Any
    executor: Any
    bench: Any
    image: str
    math: Any
    classify_prior_art: Any = None
    lookup_source: Any = None


def mixed_batches(cards: dict[str, dict], entries: dict[str, dict], seed: str) -> list[list[str]]:
    """Card keys per batch, each batch in a seeded shuffle."""
    numbers = sorted({entries[key].get("batch", 1) for key in cards})
    rng = random.Random(seed)
    batches = []
    for number in numbers:
        keys = sorted(key for key in cards if entries[key].get("batch", 1) == number)
        rng.shuffle(keys)
        batches.append(keys)
    return batches


def preflight(planted: ps.PlantedSet, sealed: ps.Sealed, *, store_source: Path, runs_root: Path, cfg: dict) -> dict:
    """Every check that must hold before anything runs; raises PlantedError with the list of problems."""
    problems = []
    for key, found in ps.leak_problems(planted.cards, sealed.entries).items():
        problems += [f"{key}: {p}" for p in found]
    store = OutcomeStore(store_source) if (Path(store_source) / JSONL).is_file() else None
    problems += ps.set_problems(planted.cards, sealed.entries, store=store,
                                jaccard_min=cfg["outcomes"]["jaccard_min"],
                                diversity_threshold=cfg["diversity"]["threshold"])
    store_origins = {key: entry["origin"] for key, entry in sealed.entries.items()
                     if key in planted.cards and "run" in (entry.get("origin") or {})}
    if store_origins:
        if store is None:
            problems.append(f"T2 and T6 origins name stored records but {store_source} has no outcome store")
        else:
            problems += ps.store_origin_problems({k: {"origin": o} for k, o in store_origins.items()}, store,
                                                 runs_root)
    try:
        recorded = manifest.verify()
    except manifest.ManifestMismatch as exc:
        problems.append(str(exc))
        recorded = {}
    if problems:
        raise PlantedError("; ".join(problems))
    return {"manifest": recorded, "store_origins": store_origins}


async def run_once(*, index: int, planted: ps.PlantedSet, sealed: ps.Sealed, keys: list[str], run_root: Path,
                   cfg: dict, reason: str, deps_factory: Callable[[Ledger], Deps], store_source: Path,
                   runs_root: Path, store_origins: dict[str, dict], proposer_model: str | None,
                   max_wall_s: int | None, override_gate: asyncio.Semaphore | None = None
                   ) -> tuple[dict[str, dict], dict]:
    """One whole planted run: store copy, the pipeline run, the T5 override runs, and the per-card results. The
    override runs of the run start together, each when ``override_gate`` (shared by every whole-set run; None: one at
    a time) lets it, each in its own run directory with its own dependencies."""
    run_dir = (run_root / f"run{index}").resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PlantedError(f"{run_dir} is not empty")
    store_dir = run_root / f"run{index}-outcomes"
    seeded = prepare_store(store_source, store_dir, {k: o for k, o in store_origins.items() if k in keys},
                           runs_root)
    run_cfg = planted_config(cfg, store_path=store_dir, n_cards=len(keys), max_wall_s=max_wall_s)
    entries = {key: sealed.entries[key] for key in keys}
    order = mixed_batches({key: planted.cards[key] for key in keys}, entries, f"{planted.name}:{index}")
    ids = {key: new_card_id() for key in keys}
    batches = [[(key, entering_card(planted.cards[key], ids[key])) for key in batch] for batch in order]
    origins = {key: entries[key]["origin"]["planted"] for key in keys
               if "planted" in (entries[key].get("origin") or {}) and entries[key]["origin"]["planted"] in ids}
    ledger = Ledger(run_dir)
    ledger.append("planted_seal", ACTOR, {
        "statement": PLANTED_STATEMENT, "reason": reason, "set": planted.name, "run": index,
        "sealed_sha256": sealed.sha256, "seal": planted.seal, "cards": {key: ids[key] for key in keys},
        "batches": order, "store": seeded, "config": {key: run_cfg[key] for key in (
            "s3", "math", "target", "refine", "s2", "outcomes", "generation", "budgets", "explore")}})
    deps = deps_factory(ledger)
    orch = PlantedOrchestrator(run_cfg, run_dir, deps=deps, batches=batches, proposer_model=proposer_model,
                               origins=origins, ledger=ledger)
    summary = await orch.run()
    view = RunView(run_dir)
    reached = {e["data"]["card_id"] for e in view.events if e["type"] == "planted_reached"}
    results = {key: card_result(view, key, ids[key], reached=reached, cfg=run_cfg) for key in keys}
    for event in view.events:
        if event["type"] == "planted_origin_check" and not event["data"]["origin_dead"]:
            results[event["data"]["key"]]["notes"].append(
                f"origin {event['data']['origin']} had not died when this card entered")
        elif event["type"] == "planted_admit_failed":
            results[event["data"]["key"]]["notes"].append(f"admission failed: {event['data']['error'][:300]}")
    target_rule = {k: cfg["target"][k] for k in m_math.TARGET_RULE_KEYS}
    gate = override_gate or asyncio.Semaphore(1)

    main_store = Store(run_dir, ledger)

    async def override(key: str) -> None:
        card, resources_from = override_card(planted.cards[key], ids[key], main_store)
        async with gate:
            results[key]["override"] = await override_mc(
                key=key, card=card, run_dir=run_root / f"run{index}-override" / key, cfg=run_cfg, reason=reason,
                deps_factory=deps_factory, target_rule=target_rule, resources_from=resources_from)

    overrides = [key for key in keys if metrics.RULES[sealed.entries[key]["type"]].override]
    await run_limited([lambda key=key: override(key) for key in overrides], max(1, len(overrides)))
    meta = {"run": index, "run_dir": str(run_dir), "stop_reason": summary["reason"], "budget": summary["budget"],
            "store": seeded, "order": order}
    return results, meta


async def planted_test(*, planted: ps.PlantedSet, sealed: ps.Sealed, runs: int, run_root: Path, out_dir: Path,
                       cfg: dict, reason: str, deps_factory: Callable[[Ledger], Deps],
                       store_source: Path = STORE_SOURCE, runs_root: Path = RUNS_ROOT,
                       proposer_model: str | None = None, keys: list[str] | None = None,
                       max_wall_s: int | None = None, parallel: int = 1, job_slots: dict | None = None) -> dict:
    """Run the set ``runs`` times and score it. Writes outcomes.jsonl (one line per card and run, the finished runs
    in run order), report.json and report.md in ``out_dir``.

    ``parallel`` whole-set runs run at once, and at most ``parallel`` T5 override runs across them
    (hybrid/research/parallel.py). Every run has its own run directory, store copy and dependencies, so the runs
    stay independent; they share the host's job slots and LLM lanes. ``job_slots`` is recorded in the meta."""
    reason = reason.strip()
    if not reason:
        raise PlantedError("a planted run needs a non-empty reason")
    if runs < 1:
        raise PlantedError("runs must be at least 1")
    try:
        check_parallel(parallel)
    except ValueError as exc:
        raise PlantedError(str(exc)) from None
    out_dir, run_root = Path(out_dir).resolve(), Path(run_root).resolve()
    for directory in (out_dir, run_root):
        if directory.exists() and any(directory.iterdir()):
            raise PlantedError(f"{directory} is not empty")
    keys = sorted(keys or planted.cards)
    checked = preflight(planted, sealed, store_source=store_source, runs_root=runs_root, cfg=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    before = audit_snapshot(planted, sealed)
    override_gate = asyncio.Semaphore(parallel)
    finished: dict[int, dict[str, dict]] = {}
    audits: dict[int, tuple[int, list[str]]] = {}

    def thunk(index: int):
        return lambda: run_once(index=index, planted=planted, sealed=sealed, keys=keys, run_root=run_root, cfg=cfg,
                                reason=reason, deps_factory=deps_factory, store_source=store_source,
                                runs_root=runs_root, store_origins=checked["store_origins"],
                                proposer_model=proposer_model, max_wall_s=max_wall_s, override_gate=override_gate)

    def done(position: int, value: tuple[dict[str, dict], dict]) -> None:
        index = position + 1
        results, meta = value
        finished[index] = results
        with open(out_dir / "outcomes.jsonl", "w", encoding="utf-8") as handle:
            for number in sorted(finished):
                for key in keys:
                    handle.write(json.dumps(dict(finished[number][key], run=number), sort_keys=True,
                                            default=str) + "\n")
        found = snapshot_differences(before, audit_snapshot(planted, sealed), f"after run {index}")
        count = 0
        for directory in [Path(meta["run_dir"])] + sorted((run_root / f"run{index}-override").glob("*")):
            checked_count, problems = math_check_differences(directory, checked["manifest"])
            count += checked_count
            found += problems
        audits[index] = (count, found)

    pairs = await run_limited([thunk(index) for index in range(1, runs + 1)], parallel, done)
    all_results = [results for results, _ in pairs]
    metas = [meta for _, meta in pairs]
    math_checks = sum(audits[index][0] for index in sorted(audits))
    differences = [item for index in sorted(audits) for item in audits[index][1]]
    # Unseal: the sealed file must still hash to the seal (the audit snapshot covers it too); then score.
    if ps.sha256_bytes(sealed.path.read_bytes()) != sealed.sha256:
        differences.append(f"{sealed.path} changed during the test")
    types = {key: sealed.entries[key]["type"] for key in keys}
    audit = {"files": len(before), "math_checks": math_checks, "differences": differences}
    report = metrics.score(all_results, types, idea_calls_cap=cfg["budgets"]["idea"]["model_calls"], audit=audit)
    meta = {"set": planted.name, "cards": len(keys), "runs": runs, "started": started,
            "finished": datetime.now(timezone.utc).isoformat(), "reason": reason, "run_root": str(run_root),
            "sealed_sha256": sealed.sha256, "runs_meta": metas, "statement": PLANTED_STATEMENT,
            "parallel": parallel, "job_slots": job_slots}
    report["meta"] = meta
    report["cards"] = {key: {"type": types[key], **{f: sealed.entries[key].get(f) for f in ("subtype", "batch")},
                             "results": [run[key] for run in all_results]} for key in keys}
    (out_dir / "report.json").write_text(metrics.dumps(report), encoding="utf-8")
    (out_dir / "report.md").write_text(metrics.render_markdown(report, all_results, {k: sealed.entries[k]
                                                                                   for k in keys}, meta),
                                       encoding="utf-8")
    return report


# ----- the command --------------------------------------------------------------------------------------------------

def real_deps_factory(cfg: dict, routing: str | None, slots=None) -> Callable[[Ledger], Deps]:
    """New dependencies for every run directory; ``slots`` (hybrid.slots.JobSlots) are shared by every executor and
    every LLM client holds the host-wide lanes (hybrid/llm/lanes.py)."""
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.llm import LLMClient
    from hybrid.math.runner import MathRunner
    from hybrid.slots import SLOTS_DIR

    def build(ledger: Ledger) -> Deps:
        run_dir = ledger.run_dir.resolve()
        llm = LLMClient(str(ROOT / cfg["run"]["models_config"]), str(run_dir), ledger, routing=routing,
                        lanes_dir=SLOTS_DIR)
        executor = Executor(run_dir.name, run_dir / "jobs" / "receipts", slots=slots)
        return Deps(llm, executor, None, read_lock()["reference"],
                    MathRunner(executor, read_lock("tools")["reference"], run_dir))

    return build


def selected_keys(args, planted: ps.PlantedSet) -> tuple[ps.PlantedSet, ps.Sealed, list[str]]:
    """The set named by --set, plus the main-set cards of the --add-types types (the re-run after a fix: the held-back
    set and the failing type's original cards, plan section 13.2)."""
    sealed = ps.read_sealed(planted, args.sealed_dir)
    keys = sorted(planted.cards)
    if not args.add_types:
        return planted, sealed, keys
    add = [t.strip().upper() for t in args.add_types.split(",") if t.strip()]
    unknown = [t for t in add if t not in ps.TYPES]
    if unknown:
        raise PlantedError(f"--add-types takes types from {', '.join(ps.TYPES)}, got {unknown}")
    main = ps.load_set(ps.set_dir("main"), name="main")
    main_sealed = ps.read_sealed(main, args.sealed_dir)
    extra = sorted(k for k, e in main_sealed.entries.items() if e["type"] in add)
    if set(extra) & set(keys):
        raise PlantedError("the added main-set cards share keys with the selected set")
    merged = ps.PlantedSet(f"{planted.name}+main:{','.join(add)}", planted.directory,
                           {**planted.cards, **{k: main.cards[k] for k in extra}},
                           {**planted.raw, **{k: main.raw[k] for k in extra}}, planted.seal)
    for key in extra:
        entry = main_sealed.entries[key]
        if "planted" in (entry.get("origin") or {}) and entry["origin"]["planted"] not in merged.cards:
            raise PlantedError(f"{key} varies the main-set card {entry['origin']['planted']}, which is not selected")
    merged_sealed = ps.Sealed(sealed.path, sealed.sha256,
                              dict(sealed.data, cards={**sealed.entries, **{k: main_sealed.entries[k]
                                                                            for k in extra}}))
    return merged, merged_sealed, keys + extra


def planted_command(args) -> int:
    import sys

    from hybrid.llm import ConfigError, GatewayUnavailable, check_gateways, load_config
    from hybrid.research.orchestrator import load_run_config

    if args.action in ("seal", "check"):
        return seal_command(args)
    if args.action == "verify":
        return verify_command(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path("results") / f"planted-{args.set}-{stamp}"
    run_root = args.run_root or Path("runs") / f"planted-{args.set}-{stamp}"
    try:
        if not (args.reason or "").strip():
            raise PlantedError("--reason is required for a planted run")
        if args.runs < 1:
            raise PlantedError("--runs must be at least 1")
        parallel = check_parallel(args.parallel)
        for directory in (Path(out_dir), Path(run_root)):
            if directory.exists() and any(directory.iterdir()):
                raise PlantedError(f"{directory.resolve()} is not empty")
        cfg = load_run_config(args.config)
        models = load_config(ROOT / cfg["run"]["models_config"], args.routing)
        planted = ps.load_set(args.cards_dir or ps.set_dir(args.set), name=args.set)
        planted, sealed, keys = selected_keys(args, planted)
        proposer = models.roles[PROPOSER_ROLE][0].model
        preflight(planted, sealed, store_source=args.store, runs_root=RUNS_ROOT, cfg=cfg)
        slots, slot_record = job_slots_from_config(cfg)
        # Last, after every offline check: the only preflight that contacts a service.
        check_gateways(models)
    except (GatewayUnavailable, ConfigError, ValueError, VMUnavailable) as exc:
        print(f"cannot start the planted test: {exc}", file=sys.stderr)
        return 2
    print("PLANTED-CARD TEST: " + PLANTED_STATEMENT, flush=True)
    print(f"set {planted.name}: {len(keys)} cards; runs: {args.runs}; out: {Path(out_dir).resolve()}; runs under "
          f"{Path(run_root).resolve()}", flush=True)
    print(f"runs at once: {parallel}; " + slot_line(slot_record), flush=True)
    try:
        report = asyncio.run(planted_test(planted=planted, sealed=sealed, runs=args.runs, run_root=run_root,
                                          out_dir=out_dir, cfg=cfg, reason=args.reason,
                                          deps_factory=real_deps_factory(cfg, args.routing, slots),
                                          store_source=args.store, proposer_model=proposer, keys=keys,
                                          max_wall_s=args.max_wall_s, parallel=parallel, job_slots=slot_record))
    except PlantedError as exc:
        print(f"cannot start the planted test: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"accepted": report["accepted"], "agreement": report["agreement"]["rate"],
                      "checks": [[c["metric"], c["ok"]] for run in report["runs"] for c in run["checks"]]}, indent=1))
    return 0 if report["accepted"] else 1


def seal_command(args) -> int:
    """`seal` checks a written set and seals it; `check` makes the same checks and writes nothing. Both use the run
    config's own-history Jaccard and TF-IDF thresholds and the outcome store the runs will copy."""
    import sys

    from hybrid.research.orchestrator import load_run_config

    verb = "seal" if args.action == "seal" else "check"
    if not args.expectations:
        print(f"cannot {verb}: --expectations names the expectations file", file=sys.stderr)
        return 2
    store = OutcomeStore(args.store) if (Path(args.store) / JSONL).is_file() else None
    directory = args.cards_dir or ps.set_dir(args.set)
    try:
        cfg = load_run_config(args.config)
        thresholds = {"jaccard_min": cfg["outcomes"]["jaccard_min"],
                      "diversity_threshold": cfg["diversity"]["threshold"]}
        if args.action == "check":
            cards, _ = ps.read_cards(directory)
            problems = ps.check_unsealed(set_name=args.set, cards=cards,
                                         data=json.loads(Path(args.expectations).read_text(encoding="utf-8")),
                                         store=store, runs_root=RUNS_ROOT, **thresholds)
            if problems:
                raise ps.SealError(problems)
            print(f"planted set {args.set}: {len(cards)} cards pass every sealing check (nothing was written)")
            return 0
        record = ps.seal(set_name=args.set, directory=directory, expectations_path=args.expectations,
                         sealed_directory=args.sealed_dir, store=store, runs_root=RUNS_ROOT, **thresholds)
    except (ps.SealError, ValueError) as exc:
        print(f"cannot {verb}:\n- " + "\n- ".join(getattr(exc, "problems", [str(exc)])), file=sys.stderr)
        return 2
    print(json.dumps({"set": record["set"], "cards": len(record["cards"]), "sealed_sha256": record["sealed_sha256"],
                      "sealed_file": str(ps.sealed_path(args.set, args.sealed_dir))}, indent=1))
    return 0


def verify_command(args) -> int:
    """Check a sealed set without running it: seal hashes, sealed file, leak check and set checks."""
    import sys

    from hybrid.research.orchestrator import load_run_config

    try:
        cfg = load_run_config(args.config)
        planted = ps.load_set(args.cards_dir or ps.set_dir(args.set), name=args.set)
        planted, sealed, keys = selected_keys(args, planted)
        preflight(planted, sealed, store_source=args.store, runs_root=RUNS_ROOT, cfg=cfg)
    except ValueError as exc:
        print(f"planted set {args.set} does not verify: {exc}", file=sys.stderr)
        return 2
    print(f"planted set {planted.name}: {len(keys)} cards verify (seal, sealed file, leak check, set checks, "
          f"manifest)")
    return 0
