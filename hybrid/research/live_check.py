"""Phase 7 live check: stage R and the S2 difference statement on cards of an earlier run, stopping before S3/M.

    .venv-hybrid/bin/python -m hybrid.research.live_check --source-run runs/run-20261010T181300Z \
        --cards card-46dbb349,card-c9615069 --run-dir runs/phase7-live-<UTC> --store-copy <dir> \
        --routing gateway --reason "..."

No other entry point fits: `run` drafts new cards, `plumb` runs S3 to S6 or M and C on a copied card, and `planted`
needs a sealed set and continues through M and C. This driver is the orchestrator with one injector in place of the
explorers (as in hybrid/research/planted.py) and one stop:

- Entry. Each named card is read from the source run (plumb.load_source: plain reads, sha256 checked against the
  source ledger; the source run is never written), stored as version 1 with no parents and an empty status, and
  admitted at S0 through the real admission (s0_explore.admit: TF-IDF duplicate check, lineage and idea assignment,
  which joins the lineage of a recorded signature in the outcome store). It keeps its card id and its drafting model
  (from the source `card_created`), so stage R picks the challenger family exactly as for a drafted card. Keeping the
  id is safe for S2's own-history query, which leaves out only this run's record of the card (FIX-10): the card's
  deaths in the source run and other runs are found, and R3 applies to them. The
  proposer role is `--role` (default `explorer`), because on the gateway table `explorer_alt` is GPT first and would
  put the proposer in the challenger's family for a card a Claude model drafted. Parents are dropped (the store
  refuses unknown parents), so an escape card reaches S2 with its non-reduction argument but without its parent's
  reduction mapping. The source card file is stored verbatim as a `live_check_source` artifact.
- Stages S1 (the entry check), R and S2 run unchanged, with the configuration's switches; the outcome store is a copy
  (`--store-copy`, made from `--store`), so a card that dies here is recorded in the copy, never in the live store.
- Stop. A card that S2 passes is not queued for the next stage (S3, or M in math mode): `live_check_stopped` is
  recorded instead and the card stays alive at S2, with no outcome record.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from hybrid.research import graveyard
from hybrid.research.orchestrator import ACTOR, ROOT, Orchestrator, load_run_config
from hybrid.research.outcomes import JSONL, OutcomeStore
from hybrid.research.plumb import load_source
from hybrid.research.stages.common import RunBudgetExhausted, StageError
from hybrid.research.stages.s0_explore import ExploreTask, admit

LAST_STAGE = "S2"
SOURCE_KIND = "live_check_source"


class LiveCheckError(Exception):
    """The request cannot be run (bad source, card, store or run directory). Raised before anything is created."""


def synthetic_call_id(card_id: str) -> str:
    """created_by_call and the S0 call of a copied card: no model call drafted it in this run."""
    return "call-" + hashlib.sha256(f"live_check:{card_id}".encode()).hexdigest()[:12]


def source_created(source_run: Path, card_id: str) -> dict:
    for event in graveyard.read_events(source_run):
        if event["type"] == "card_created" and event["data"]["card_id"] == card_id:
            return event["data"]
    raise LiveCheckError(f"{card_id} has no card_created event in {source_run}")


def entering_card(card: dict) -> dict:
    return dict(copy.deepcopy(card), version=1, parent_ids=[], created_by_call=synthetic_call_id(card["id"]),
                status={"stage": "S0", "verdicts": {}})


def copy_store(source: Path, target: Path) -> Path:
    """A fresh outcome-store copy (the JSONL; the index is rebuilt from it), checked before use."""
    if target.exists():
        raise LiveCheckError(f"{target} exists; the live check starts from a fresh store copy")
    if not (source / JSONL).is_file():
        raise LiveCheckError(f"no outcome store at {source}")
    target.mkdir(parents=True)
    shutil.copyfile(source / JSONL, target / JSONL)
    ok, bad_seq = OutcomeStore(target).verify()
    if not ok:
        raise LiveCheckError(f"the store copied from {source} has a broken hash chain at seq {bad_seq}")
    return target


def live_config(cfg: dict, *, store_path: Path, n_cards: int, max_wall_s: int) -> dict:
    """The configuration's switches as they are, plus: the outcome store on (a copy), one injector, no escapes."""
    cfg = copy.deepcopy(cfg)
    cfg["outcomes"].update(enabled=True, path=str(Path(store_path).resolve()))
    cfg["run"]["n_explorers"] = 1
    cfg["explore"].update(p_escape=0.0, escape_reserve=0.0)
    cfg["budgets"]["max_cards"] = n_cards
    cfg["budgets"]["max_wall_s"] = max_wall_s
    return cfg


class LiveCheckOrchestrator(Orchestrator):
    """The orchestrator with an injector of copied cards in place of the explorers, stopping every card after S2."""

    def __init__(self, cfg: dict, run_dir: Path, *, source_run: Path, card_ids: list[str], role: str, **kwargs):
        super().__init__(cfg, run_dir, **kwargs)
        if self.generation is not None:
            raise ValueError("the live check admits cards at S0; run it with generation.mode lenses")
        self.source_run = Path(source_run).resolve()
        self.card_ids = list(card_ids)
        self.role = role
        self.stop_after = self.order.index(LAST_STAGE)

    async def _explorer(self, explorer: int) -> None:
        try:
            for card_id in self.card_ids:
                if self.stop_event.is_set():
                    return
                self.budget.cards += 1
                self.pending += 1
                try:
                    self._admit(card_id)
                except (StageError, RunBudgetExhausted) as exc:
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

    def _admit(self, card_id: str) -> None:
        source = load_source(self.source_run, card_id)
        created = source_created(self.source_run, card_id)
        card = entering_card(source.card)
        self.store.put_artifact(SOURCE_KIND, f"{card_id}-source-v{source.version}.json", source.raw, card_id=card_id)
        self.emit("live_check_admit", {"card_id": card_id, "source_run": source.run_id, "source_path": source.path,
                                       "source_version": source.version,
                                       "source_sha256": hashlib.sha256(source.raw).hexdigest(),
                                       "source_role": created["role"], "source_model": created["model"],
                                       "source_op": created["op"], "source_parents": created["parents"],
                                       "role": self.role})
        task = ExploreTask(op="new", lens=card["lens"], lens_sources=[], role=self.role, explorer=0, seed=0)
        admission = admit(self.ctx, task, card, call_ids=[card["created_by_call"]], model=created["model"], papers=[],
                          nearby=None, user=json.dumps(card, sort_keys=True), dropped=[], dropped_sources=[],
                          extra_reasons=[f"live check: copied from {source.run_id}/{source.path} (no model call "
                                         f"drafted it in this run)"])
        if admission.passed:
            self._enqueue("S1", card_id)
        else:
            self._record_outcome(card_id)

    def _enqueue(self, stage: str, card_id: str) -> None:
        if self.order.index(stage) > self.stop_after:
            self.emit("live_check_stopped", {"card_id": card_id, "next_stage": stage,
                                             "why": f"the live check stops every card after {LAST_STAGE}"})
            return
        super()._enqueue(stage, card_id)


def card_results(orch: Orchestrator) -> dict:
    """Per card: the verdict and its gate reason at each stage this run decided, and where the card ended."""
    out = {}
    for card_id in orch.store.list_cards():
        verdicts = [e["data"] for e in orch.ledger.iter_events("verdict") if e["data"]["card_id"] == card_id]
        stopped = [e["data"]["next_stage"] for e in orch.ledger.iter_events("live_check_stopped")
                   if e["data"]["card_id"] == card_id]
        gates = {}
        for v in verdicts:
            gate = [r for r in v["reasons"] if r.startswith("gate: ")]
            gates[v["stage"]] = {"verdict": v["verdict"], "gate": gate[0][:300] if gate else None}
        out[card_id] = {"verdicts": gates, "stopped_before": stopped[0] if stopped else None}
    return out


def build(args) -> LiveCheckOrchestrator:
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.literature.keys import load_api_keys
    from hybrid.llm import LLMClient, check_gateways, load_config
    from hybrid.slots import SLOTS_DIR, job_slots_from_config

    load_api_keys()
    card_ids = [c.strip() for c in args.cards.split(",") if c.strip()]
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise LiveCheckError(f"{run_dir} is not empty")
    for card_id in card_ids:
        load_source(Path(args.source_run), card_id)
        source_created(Path(args.source_run), card_id)
    store = copy_store(Path(args.store), Path(args.store_copy))
    cfg = live_config(load_run_config(args.config), store_path=store, n_cards=len(card_ids),
                      max_wall_s=args.max_wall_s)
    models = load_config(ROOT / cfg["run"]["models_config"], args.routing)
    gateways = check_gateways(models)
    slots, slot_record = job_slots_from_config(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    from hybrid.research.ledger import Ledger

    ledger = Ledger(run_dir)
    ledger.append("live_check_start", ACTOR, {"reason": args.reason, "source_run": str(Path(args.source_run).resolve()),
                                              "cards": card_ids, "role": args.role, "store_source": str(args.store),
                                              "store_copy": str(store.resolve()), "last_stage": LAST_STAGE})
    llm = LLMClient(str(ROOT / cfg["run"]["models_config"]), str(run_dir), ledger, routing=args.routing,
                    lanes_dir=SLOTS_DIR)
    executor = Executor(run_dir.name, run_dir / "jobs" / "receipts", slots=slots)
    routing = {"table": models.routing, "source": models.routing_source, "gateways": gateways}
    return LiveCheckOrchestrator(cfg, run_dir, source_run=args.source_run, card_ids=card_ids, role=args.role,
                                 llm=llm, executor=executor, bench=None, image=read_lock()["reference"], ledger=ledger,
                                 config_bytes=Path(args.config).read_bytes(), routing=routing, job_slots=slot_record)


def main(argv: list[str] | None = None) -> int:
    from hybrid import diskguard, localmodels
    from hybrid.llm import ConfigError, GatewayUnavailable, load_config

    parser = argparse.ArgumentParser(prog="python -m hybrid.research.live_check", description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--cards", required=True, help="comma-separated card ids of the source run")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs") / ("phase7-live-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")))
    parser.add_argument("--store", type=Path, default=Path("results/outcomes"), help="outcome store to copy; never written")
    parser.add_argument("--store-copy", type=Path, required=True, help="new directory for the store copy")
    parser.add_argument("--config", type=Path, default=Path("hybrid/config/run.toml"))
    parser.add_argument("--routing", default=None)
    parser.add_argument("--role", default="explorer", choices=("explorer", "explorer_alt"),
                        help="proposer role for the dialogue and the difference statement")
    parser.add_argument("--max-wall-s", type=int, default=28800)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)

    if not diskguard.guard_cli(diskguard.require_run_space, "the phase 7 live check"):
        return 2
    try:
        models = load_config(ROOT / load_run_config(args.config)["run"]["models_config"], args.routing)
    except ConfigError as exc:
        print(f"cannot start the live check: {exc}", file=sys.stderr)
        return 2
    if not localmodels.guard_cli(models, ("S1", "R", "S2"), "the phase 7 live check", report=print):
        return 2
    try:
        orch = build(args)
    except (LiveCheckError, GatewayUnavailable, ConfigError) as exc:
        print(f"cannot start the live check: {exc}", file=sys.stderr)
        return 2
    print(f"run dir: {orch.run_dir}", flush=True)
    summary = asyncio.run(orch.run())
    summary["cards_result"] = card_results(orch)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
