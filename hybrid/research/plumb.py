"""Operator plumbing check: run S4, S5 and S6 (or another run of stages from S3 on) on a card that the gates
stopped.

No card has passed S3 live yet, so S4 (build), S5 (bench in Podman) and S6 (analyst) have only run against
fakes. `python -m hybrid.research plumb` runs the real stage functions on one card taken from another
run, in a NEW run directory, so those stages can be seen working end to end.

This is an operator override, not a card passing the gates. Three things keep it from being mistaken for
one:
- the first ledger event of the new run is `operator_override`: reason, source run, card id, the card's
  S0 to S3 verdicts in the source run, and a statement that gates were bypassed;
- the card is copied with its source history intact (the S2 verdict stays what it was) and the source card
  file and verdict summary are stored verbatim as `plumb_source` artifacts;
- `inspect` shows the override on every page of the report when the ledger holds the event.

The source run is only read, with plain file reads and no ledger lock, so it may still be running.

`--stages` picks the stages, a consecutive run from S3 to S6 (default S4,S5,S6). `--stages S3` replays S3 alone
on the copied card, for example to see the executed obligation checks on a card whose judges split; the card's
latest `s1_residual` artifact is copied too, so the formalizer and disprover get the same S1 open issues as in
the source run. The line stops after the last stage named, whatever that stage's verdict.

S5 includes the card falsifier tasks (s5.falsifiers, `--falsifiers/--no-falsifiers`): the falsifier_designer
writes them from the copied card, the executor validates them, the bench runs them, and S6 decides the claims
their rules name. The summary reports the validated and dropped tasks and the claims ledger counts.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hybrid.executor.job import Caps
from hybrid.research import graveyard
from hybrid.research.ledger import Ledger
from hybrid.research.stages import STAGE_RUNNERS
from hybrid.research.stages.common import Budget, CodeRunner, RunBudgetExhausted, StageContext, StageError
from hybrid.research.store import CARD_ID, Store

OPERATOR = "operator"
ORCHESTRATOR = "orchestrator"
STATEMENT = "gates bypassed for a plumbing check; results are not evidence that the card passed S2/S3"
SOURCE_STAGES = ("S0", "S1", "S2", "S3")
PLUMB_STAGES = ("S4", "S5", "S6")
RUNNABLE_STAGES = ("S3", "S4", "S5", "S6")
# The math stages M and C (s3.mode = "math") in place of S3. A run of them starts at M: C needs M's frozen statements.
MATH_STAGES = ("M", "C", "S4", "S5", "S6")
# Under the override, C runs after M whatever M's verdict, so a card that M leaves unproven still gets its
# challenge results recorded (calibration in plan section 12 needs both).
C_AFTER_M = "C runs after M whatever M's verdict (operator override); a normal run stops a card at M unless M passes"
# Source artifacts copied next to the card: the S2 counts in `inspect` and the S3 inputs stay what they were.
COPIED_KINDS = ("reduction_critic", "s1_residual")


class PlumbError(Exception):
    """The request cannot be run (bad source run, card or run directory). Raised before anything is created."""


@dataclass
class Deps:
    """The external dependencies of the stage functions."""
    llm: Any
    executor: Any
    bench: Any
    image: str
    # hybrid.math.runner.MathRunner for stages M and C
    math: Any = None


@dataclass
class SourceCard:
    run_id: str
    version: int
    path: str            # relative to the source run
    raw: bytes           # the card file exactly as the source run wrote it
    card: dict
    # kind -> (artifact name, bytes) for the latest source artifact of each COPIED_KINDS kind the card has
    copied: dict


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_stages(text: str) -> tuple[str, ...]:
    """`--stages` value: comma-separated stage names forming a consecutive run within S3 to S6, or within M, C, S4,
    S5, S6 starting at M."""
    stages = tuple(part.strip().upper() for part in text.split(",") if part.strip())
    order = MATH_STAGES if any(stage in ("M", "C") for stage in stages) else RUNNABLE_STAGES
    if not stages or any(stage not in order for stage in stages):
        raise PlumbError(f"--stages takes stages from {', '.join(RUNNABLE_STAGES)} or {', '.join(MATH_STAGES)}, "
                         f"got {text!r}")
    start = order.index(stages[0])
    if stages != order[start:start + len(stages)]:
        raise PlumbError(f"--stages must be consecutive and in order, got {text!r}")
    if stages[0] == "C":
        raise PlumbError("--stages C needs M first: C reads M's frozen statements")
    return stages


def load_source(source_run: Path, card_id: str) -> SourceCard:
    """Read the latest version of a card, and its latest artifact of each COPIED_KINDS kind, from another run.

    Plain file reads only. Files are found through the source ledger (the store appends an event only
    after the file is written, and a last line that is still being written is skipped), and the
    artifact is checked against the sha256 the ledger recorded.
    """
    source_run = Path(source_run)
    if not CARD_ID.match(card_id):
        raise PlumbError(f"malformed card id: {card_id!r}")
    if not (source_run / "ledger.jsonl").is_file():
        raise PlumbError(f"source run has no ledger.jsonl: {source_run}")
    latest: dict | None = None
    artifacts: dict[str, dict] = {}
    for event in graveyard.read_events(source_run):
        data = event["data"]
        if event["type"] == "card_version" and data["card_id"] == card_id:
            latest = data
        elif event["type"] == "artifact" and data["kind"] in COPIED_KINDS and data.get("card_id") == card_id:
            artifacts[data["kind"]] = data
    if latest is None:
        raise PlumbError(f"card {card_id} is not in {source_run}")
    raw = (source_run / latest["path"]).read_bytes()
    if _sha256(raw) != latest["sha256"]:
        raise PlumbError(f"{latest['path']} does not match the sha256 in the source ledger")
    source = SourceCard(source_run.resolve().name, latest["version"], latest["path"], raw, json.loads(raw), {})
    for kind, data in artifacts.items():
        artifact_raw = (source_run / data["path"]).read_bytes()
        if _sha256(artifact_raw) != data["sha256"]:
            raise PlumbError(f"{data['path']} does not match the sha256 in the source ledger")
        source.copied[kind] = (data["name"], artifact_raw)
    return source


def source_verdicts(card: dict) -> dict:
    """S0 to S3 verdicts of the source card, None for a stage that gave none (an S3 that never ran)."""
    verdicts = card["status"]["verdicts"]
    return {stage: ({"verdict": verdicts[stage]["verdict"], "reasons": verdicts[stage]["reasons"],
                     "by_calls": verdicts[stage]["by_calls"]} if stage in verdicts else None)
            for stage in SOURCE_STAGES}


def override_data(source: SourceCard, source_run: Path, card_id: str, reason: str, routing: dict | None,
                  s5_mode: str, falsifiers: bool, stages: tuple[str, ...] = PLUMB_STAGES) -> dict:
    return {
        "statement": STATEMENT,
        "reason": reason,
        "source_run": str(Path(source_run).resolve()),
        "source_run_id": source.run_id,
        "card_id": card_id,
        "source_card": {"version": source.version, "path": source.path, "sha256": _sha256(source.raw)},
        "source_verdicts": source_verdicts(source.card),
        "source_parent_ids": source.card["parent_ids"],
        "stages_run": list(stages),
        **({"math_note": C_AFTER_M} if "M" in stages and "C" in stages else {}),
        "s5_mode": s5_mode,
        "falsifiers": falsifiers,
        "routing": routing,
    }


def reset_status(card: dict, first_stage: str) -> dict | None:
    """When the copied card is already past the first stage to run, its status goes back to the stage before it,
    so the store accepts the new verdicts. For M (which replaces S3) the status keeps S0 to S2. Returns what was
    removed (None when nothing was), for the override record; the full source status stays in plumb_source."""
    from hybrid.research.store import STAGES

    status = card["status"]
    if first_stage == "M":
        keep = ("S0", "S1", "S2")
    else:
        if STAGES.index(status["stage"]) <= STAGES.index(first_stage):
            return None
        keep = STAGES[:STAGES.index(first_stage)]
    removed = {stage: entry["verdict"] for stage, entry in status["verdicts"].items() if stage not in keep}
    if not removed:
        return None
    kept = {stage: entry for stage, entry in status["verdicts"].items() if stage in keep}
    card["status"] = {"stage": max(kept, key=STAGES.index) if kept else "S0", "verdicts": kept}
    return {"from_stage": status["stage"], "to_stage": card["status"]["stage"], "removed_verdicts": removed}


def _copy_provenance(store: Store, source: SourceCard, card_id: str, verdicts: dict,
                     first_stage: str | None = None) -> dict | None:
    """Write the copied card (as version 1, no parents: the parents are not in this run) and its provenance
    artifacts: the source card file verbatim, the verdict summary, the S2 reduction critic and the S1 residual
    findings. A card already past ``first_stage`` gets its status reset (reset_status); the reset is returned."""
    store.put_artifact("plumb_source", f"{card_id}-source-v{source.version}.json", source.raw, card_id=card_id)
    summary = {"source_run": source.run_id, "card_id": card_id, "source_version": source.version,
               "source_verdicts": verdicts, "source_parent_ids": source.card["parent_ids"]}
    store.put_artifact("plumb_source", f"{card_id}-source-verdicts.json",
                       (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"), card_id=card_id)
    for kind, (name, raw) in source.copied.items():
        store.put_artifact(kind, name, raw, card_id=card_id)
    card = copy.deepcopy(source.card)
    card["version"] = 1
    card["parent_ids"] = []
    reset = reset_status(card, first_stage) if first_stage else None
    store.put_card(card)
    return reset


async def plumb(*, source_run: Path, card_id: str, run_dir: Path, reason: str, cfg: dict,
                deps_factory: Callable[[Ledger], Deps], routing: dict | None = None,
                config_bytes: bytes | None = None, stages: tuple[str, ...] = PLUMB_STAGES) -> dict:
    """Create `run_dir`, record the override first, copy the card, then run `stages` (default S4, S5, S6) on it.

    `deps_factory(ledger)` builds the model client, executor and bench once the run directory and its
    ledger exist; the real one is `real_deps_factory`. Returns a summary dict. Errors that mean the request
    cannot be run raise PlumbError before anything is created.
    """
    reason = reason.strip()
    if not reason:
        raise PlumbError("a plumbing check needs a non-empty --reason")
    source_run, run_dir = Path(source_run).resolve(), Path(run_dir).resolve()
    if run_dir == source_run or source_run in run_dir.parents:
        raise PlumbError(f"the new run directory must not be the source run or inside it: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PlumbError(f"{run_dir} is not empty")
    stages = parse_stages(",".join(stages))
    source = load_source(source_run, card_id)

    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    ledger.append("operator_override", OPERATOR,
                  override_data(source, source_run, card_id, reason, routing, cfg["s5"]["mode"],
                                cfg["s5"]["falsifiers"], stages))
    b = cfg["budgets"]
    budget = Budget(b["max_cards"], b["max_calls"], b["max_tokens"], b["max_wall_s"])
    ledger.append("run_start", ORCHESTRATOR, {"run_id": run_dir.name, "plumb": True, "resume": False,
                                              "budget": budget.snapshot(), "s5_mode": cfg["s5"]["mode"],
                                              "routing": routing, "stages": list(stages)})
    if config_bytes is not None:
        store.put_artifact("config", f"plumb-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.toml",
                           config_bytes)
    reset = _copy_provenance(store, source, card_id, source_verdicts(source.card), stages[0])
    if reset is not None:
        ledger.append("plumb_status_reset", OPERATOR, {"card_id": card_id, **reset})

    deps = deps_factory(ledger)
    if "M" in stages and deps.math is None:
        raise RuntimeError("stages M and C need a math runner (Deps.math)")
    ex = cfg["executor"]
    caps = Caps(wall_s=ex["wall_s"], memory_mb=ex["memory_mb"], pids=ex["pids"], tmp_mb=ex["tmp_mb"])
    ctx = StageContext(run_id=run_dir.name, run_dir=run_dir, ledger=ledger, store=store, llm=deps.llm,
                       runner=CodeRunner(deps.executor, run_dir, store, deps.image, caps), bench=deps.bench,
                       cfg=cfg, budget=budget, lenses={}, math=deps.math)

    stage: str | None = stages[0]
    stopped: str | None = None
    try:
        while stage in stages:
            ctx.emit("stage_start", {"card_id": card_id, "stage": stage})
            current = stage
            try:
                stage = await STAGE_RUNNERS[stage](ctx, card_id)
                if current == "M" and "C" in stages:
                    stage = "C"
            except StageError as exc:
                ctx.emit("stage_error", {"card_id": card_id, "stage": stage, "error": str(exc)[:4000]})
                stopped = f"stage_error at {stage}"
                break
            except RunBudgetExhausted as exc:
                ctx.emit("run_stop", {"reason": str(exc), **budget.snapshot()})
                stopped = f"budget: {exc}"
                break
    except Exception as exc:  # internal bug: record it, then fail fast
        ctx.emit("orchestrator_error", {"error": f"{type(exc).__name__}: {exc}"})
        raise

    final = store.get_card(card_id)
    # Only verdicts recorded in this run: a copied card keeps its source verdicts in its status.
    recorded = {e["data"]["stage"] for e in ledger.iter_events("verdict") if e["data"]["card_id"] == card_id}
    verdicts = {s: final["status"]["verdicts"][s]["verdict"] for s in stages if s in recorded}
    completed = (stopped is None and all(s in verdicts for s in stages)
                 and (stages[-1] != "S6" or verdicts["S6"] == "pass"))
    reason_text = "plumb_complete" if completed else (stopped or f"ended at {final['status']['stage']}")
    summary = {"run_id": run_dir.name, "run_dir": str(run_dir), "card_id": card_id, "reason": reason_text,
               "verdicts": verdicts, "stage": final["status"]["stage"], "completed": completed,
               "budget": budget.snapshot(), "falsifiers": falsifier_outcome(ctx, card_id)}
    if "S3" in stages:
        summary["s3"] = s3_outcome(ctx, card_id)
    if "M" in stages:
        summary["math"] = math_outcome(ctx, card_id)
    ctx.emit("run_end", summary)
    return summary


def falsifier_outcome(ctx: StageContext, card_id: str) -> dict:
    """Validated and dropped falsifier tasks and the claims ledger counts, from the run's artifacts."""
    tasks = ctx.latest_json_artifact(card_id, "falsifier_tasks")
    ledger = ctx.latest_json_artifact(card_id, "claims_ledger")
    return {"validated": [t["task_id"] for t in tasks["validated"]] if tasks else [],
            "dropped": [t["task_id"] for t in tasks["dropped"]] if tasks else [],
            "claims": ledger["counts"] if ledger else None,
            "decided_by_rule": (sum(1 for c in ledger["claims"] if c["decided_by"] == "rule") if ledger else 0)}


def math_outcome(ctx: StageContext, card_id: str) -> dict:
    """Per stage M and C: verdict, kill codes, claim statuses and tiers, premise statuses, definitions, the
    measurements deferred to test design, models; plus the role calls of the run."""
    out: dict = {"stages": {}, "role_calls": {}}
    for data in ctx.card_events(card_id, "artifact"):
        if data["kind"] != "math_verdict":
            continue
        record = json.loads((ctx.run_dir / data["path"]).read_text(encoding="utf-8"))
        ledger = json.loads((ctx.run_dir / "artifacts" / "assumption_ledger" / record["assumption_ledger"])
                            .read_text(encoding="utf-8"))
        out["stages"][record["stage"]] = {
            "verdict": record["verdict"], "kill_codes": record["kill_codes"], "models": record["models"],
            "claims": {c["claim_id"]: {"status": c["status"], "tier": c["label"]} for c in record["claims"]},
            "premises": {p["id"]: p["status"] for p in ledger["premises"]},
            "definitions": [d["id"] for d in ledger["definitions"]],
            "scheduled_measurements": [m["premise_id"] for m in record["scheduled_measurements"]],
            **({"target_coupled": {"status": record["target_coupled"]["status"],
                                   "applied": record["target_coupled"]["applied"],
                                   "slots": {slot: entry["scale"] for slot, entry
                                             in (record["target_coupled"].get("slots") or {}).items()}}}
               if record.get("target_coupled") else {})}
    for event in ctx.card_events(card_id, "math_role_call"):
        out["role_calls"][event["role"]] = out["role_calls"].get(event["role"], 0) + 1
    return out


def s3_outcome(ctx: StageContext, card_id: str) -> dict | None:
    """The S3 gate rule, votes and obligation check statuses, from the run's artifacts."""
    gate = ctx.latest_json_artifact(card_id, "s3_gate")
    if gate is None:
        return None
    checks = [data for data in ctx.card_events(card_id, "artifact") if data["kind"] == "obligation_check"]
    records = [json.loads((ctx.run_dir / data["path"]).read_text(encoding="utf-8")) for data in checks]
    return {"rule": gate["rule"], "verdict": gate["verdict"], "round": gate["round"], "votes": gate.get("votes"),
            "effective_votes": gate.get("effective_votes"), "contested": gate.get("contested"),
            "all_checkable_discharged": gate["all_checkable_discharged"],
            "checks": {r["obligation"]: {"kind": r["kind"], "status": r["status"], "replayed": r["replayed"],
                                         "optional": r["optional"]} for r in records}}


# ----- real dependencies and the command --------------------------------------------------------


def real_deps_factory(cfg: dict, routing: str | None, stages: tuple[str, ...] = PLUMB_STAGES
                      ) -> Callable[[Ledger], Deps]:
    """The dependencies the orchestrator uses: LLMClient over models.toml, the Podman Executor, the trusted
    bench with the s5 config, and the pinned executor image; plus a MathRunner on the pinned tools image when
    ``stages`` include M."""
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.llm import LLMClient
    from hybrid.research.orchestrator import ROOT
    from hybrid.research.stages.s5_evaluate import BenchEvaluator
    from hybrid.slots import SLOTS_DIR, job_slots_from_config

    slots, _ = job_slots_from_config(cfg)

    def build(ledger: Ledger) -> Deps:
        from hybrid.math.runner import MathRunner

        run_dir = ledger.run_dir.resolve()
        llm = LLMClient(str(ROOT / cfg["run"]["models_config"]), str(run_dir), ledger, routing=routing,
                        lanes_dir=SLOTS_DIR)
        executor = Executor(run_dir.name, run_dir / "jobs" / "receipts", slots=slots)
        bench = BenchEvaluator(cfg["s5"]["mode"], ROOT / cfg["s5"]["bench_config"], run_dir.name, run_dir)
        math = MathRunner(executor, read_lock("tools")["reference"], run_dir) if "M" in stages else None
        return Deps(llm, executor, bench, read_lock()["reference"], math)

    return build


def plumb_command(args) -> int:
    import sys

    from hybrid.llm import ConfigError, GatewayUnavailable, check_gateways, load_config
    from hybrid.research.orchestrator import ROOT, load_run_config

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = Path("runs") / ("plumb-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    overrides = {"s5.mode": args.s5_mode} if args.s5_mode is not None else {}
    if args.falsifiers is not None:
        overrides["s5.falsifiers"] = args.falsifiers
    try:
        cfg = load_run_config(args.config, overrides)
        models = load_config(ROOT / cfg["run"]["models_config"], args.routing)
        gateways = check_gateways(models)
    except (GatewayUnavailable, ConfigError) as exc:
        print(f"cannot start the plumbing check: {exc}", file=sys.stderr)
        return 2
    try:
        stages = parse_stages(args.stages)
    except PlumbError as exc:
        print(f"cannot start the plumbing check: {exc}", file=sys.stderr)
        return 2
    routing = {"table": models.routing, "source": models.routing_source, "gateways": gateways}
    print("PLUMBING CHECK (operator override): " + STATEMENT, flush=True)
    print(f"stages: {', '.join(stages)}", flush=True)
    print(f"run dir: {Path(run_dir).resolve()}", flush=True)
    try:
        summary = asyncio.run(plumb(source_run=args.source_run, card_id=args.card, run_dir=run_dir,
                                    reason=args.reason, cfg=cfg, routing=routing,
                                    deps_factory=real_deps_factory(cfg, args.routing, stages),
                                    config_bytes=Path(args.config).read_bytes(), stages=stages))
    except PlumbError as exc:
        print(f"cannot start the plumbing check: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if summary["completed"] else 1
