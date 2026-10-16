"""Command-line entry point: ``python -m hybrid.research <command> ...``."""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from hybrid import diskguard, localmodels
from hybrid.research.inspect import inspect_run

OVERRIDES = {
    "n_explorers": "run.n_explorers",
    "max_cards": "budgets.max_cards",
    "max_calls": "budgets.max_calls",
    "max_wall_s": "budgets.max_wall_s",
    "s5_mode": "s5.mode",
    "draft_only": "generation.draft_only",
    "arm_quota": "generation.quota",
    "generation_mode": "generation.mode",
}


def run_overrides(args) -> dict:
    """Command-line values over run.toml. An --arm-quota without --max-cards sets max_cards to the quota's sum."""
    overrides = {dotted: getattr(args, name) for name, dotted in OVERRIDES.items() if getattr(args, name) is not None}
    if args.arm_quota and args.max_cards is None:
        overrides["budgets.max_cards"] = sum(args.arm_quota.values())
    return overrides


def parse_quota(text: str) -> dict[str, int]:
    """`A1=15,A2=15` -> {"A1": 15, "A2": 15}."""
    quota = {}
    for part in text.split(","):
        arm, _, count = part.partition("=")
        if not arm.strip() or not count.strip().isdigit():
            raise argparse.ArgumentTypeError(f"expected ARM=COUNT[,ARM=COUNT...], got {text!r}")
        quota[arm.strip()] = int(count)
    return quota


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hybrid.research")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="verify the ledger, print a summary, build report/index.html")
    inspect.add_argument("run_dir", type=Path)

    run = commands.add_parser("run", help="run the research loop")
    run.add_argument("--config", type=Path, default=Path("hybrid/config/run.toml"))
    run.add_argument("--run-dir", type=Path, default=None,
                     help="default runs/run-<UTC yyyymmddThhmmssZ>; must be new unless --resume")
    run.add_argument("--resume", action="store_true", help="continue an existing run from its ledger and cards")
    run.add_argument("--n-explorers", type=int)
    run.add_argument("--max-cards", type=int)
    run.add_argument("--max-calls", type=int)
    run.add_argument("--max-wall-s", type=int)
    run.add_argument("--s5-mode", choices=("podman", "local"))
    run.add_argument("--routing", help="routing table in models.toml (llxprt or gateway); overrides "
                                       "BIGBRANE_ROUTING and the file. gateway is checked before the run starts")
    run.add_argument("--draft-only", action="store_true", default=None,
                     help="stop every card at S0: draft, check and classify only (generation.draft_only)")
    run.add_argument("--arm-quota", type=parse_quota, default=None,
                     help="drafts per generation arm, e.g. A1=15,A2=15,A3=10,A4=10 (generation.quota; needs "
                          "generation.mode = arms). With no --max-cards, max_cards becomes their sum")
    run.add_argument("--generation-mode", choices=("lenses", "arms"), help="generation.mode")

    plumb = commands.add_parser(
        "plumb", help="OPERATOR OVERRIDE: run S4 to S6 (or --stages) live on a card the gates stopped, in a new run directory. "
                      "Not a card passing the gates; the run is labeled as an override everywhere")
    plumb.add_argument("--source-run", type=Path, required=True,
                       help="run directory holding the card (only read; it may still be running)")
    plumb.add_argument("--card", required=True, help="card id in the source run; its latest version is copied")
    plumb.add_argument("--run-dir", type=Path, default=None,
                       help="default runs/plumb-<UTC yyyymmddThhmmssZ>; must be new or empty")
    plumb.add_argument("--reason", required=True, help="why the gates are bypassed; recorded in the ledger")
    plumb.add_argument("--config", type=Path, default=Path("hybrid/config/run.toml"))
    plumb.add_argument("--s5-mode", choices=("podman", "local"))
    plumb.add_argument("--routing", help="routing table in models.toml; overrides BIGBRANE_ROUTING and the file")
    plumb.add_argument("--falsifiers", action=argparse.BooleanOptionalAction, default=None,
                       help="design, validate and run card falsifier tasks in S5 (default: s5.falsifiers in the config)")
    plumb.add_argument("--stages", default="S4,S5,S6",
                       help="comma-separated consecutive stages from S3 to S6, or from M, C, S4, S5, S6 starting at M, "
                            "to run (default S4,S5,S6); S3 alone replays the feasibility stage, M,C runs the math and "
                            "challenge stages (C runs whatever M's verdict)")
    calibrate = commands.add_parser(
        "calibrate", help="OPERATOR OVERRIDE: run the anonymized known-model cards (hybrid/fixtures/calibration) through "
                          "stages M and C N times and compare with the expected table (plan section 12, Phase 4)")
    calibrate.add_argument("--runs", type=int, default=3, help="runs per card (acceptance needs 3)")
    calibrate.add_argument("--cards", default="k1,k2,k3,k4", help="comma-separated card keys (default k1,k2,k3,k4)")
    calibrate.add_argument("--reason", required=True, help="why the gates are bypassed; recorded in every ledger")
    calibrate.add_argument("--config", type=Path, default=Path("hybrid/config/run.toml"))
    calibrate.add_argument("--routing", help="routing table in models.toml; overrides BIGBRANE_ROUTING and the file")
    calibrate.add_argument("--out", type=Path, default=None,
                           help="results directory (default results/calibration-math-<UTC>); must be new or empty")
    calibrate.add_argument("--run-root", type=Path, default=None,
                           help="parent of the per-run directories (default runs/calibration-math-<UTC>)")
    calibrate.add_argument("--parallel", type=int, default=1,
                           help="card-runs at once, each in its own run directory (default 1); executor jobs are "
                                "bounded host-wide by [math] max_parallel_jobs")
    calibrate.add_argument("--rerun-invalid", type=Path, default=None, metavar="OUT_DIR",
                           help="FIX-22: rerun every invalid_environment card-run of the calibration in OUT_DIR, each "
                                "in a fresh run directory <card>-run<N>-rerun<M>, and rescore with the reruns in place "
                                "of the invalid ones (the invalid run directories are kept)")
    planted = commands.add_parser(
        "planted", help="the planted-card test (plan section 13, Phase 9): run a sealed set of planted cards through the "
                        "live pipeline N times and score it against the sealed expectations; `seal` seals a written "
                        "set, `check` runs every sealing check on a written set without sealing it, `verify` checks "
                        "a sealed set without running it")
    planted.add_argument("action", nargs="?", default="run", choices=("run", "seal", "check", "verify"))
    planted.add_argument("--set", default="main", help="set name under hybrid/fixtures/planted (main or held_back)")
    planted.add_argument("--cards-dir", type=Path, default=None,
                         help="set directory (default hybrid/fixtures/planted/<set>); holds cards/ and SEAL.json")
    planted.add_argument("--sealed-dir", type=Path, default=None,
                         help="directory of the sealed expectations (default $BIGBRANE_PLANTED_DIR or "
                              "~/.bigbrane/planted); must be outside the workspace")
    planted.add_argument("--expectations", type=Path, default=None,
                         help="seal and check: the expectations file (seal moves it into the sealed directory)")
    planted.add_argument("--runs", type=int, default=2, help="whole-set runs (repeatability needs 2)")
    planted.add_argument("--add-types", default=None,
                         help="also run the main-set cards of these types, e.g. T3,T4 (the re-run after a fix: the "
                              "held-back set plus the failing type's original cards)")
    planted.add_argument("--reason", default=None, help="why this planted run is made; recorded in every ledger")
    planted.add_argument("--config", type=Path, default=Path("hybrid/config/run.toml"))
    planted.add_argument("--routing", help="routing table in models.toml; overrides BIGBRANE_ROUTING and the file")
    planted.add_argument("--out", type=Path, default=None,
                         help="results directory (default results/planted-<set>-<UTC>); must be new or empty")
    planted.add_argument("--run-root", type=Path, default=None,
                         help="parent of the run directories (default runs/planted-<set>-<UTC>)")
    planted.add_argument("--parallel", type=int, default=1,
                         help="whole-set runs and T5 override runs at once, each in its own run directory (default 1); "
                              "executor jobs are bounded host-wide by [math] max_parallel_jobs")
    planted.add_argument("--store", type=Path, default=Path("results/outcomes"),
                         help="outcome store each run copies (default results/outcomes); never written")
    planted.add_argument("--max-wall-s", type=int, default=None, help="wall budget per run (default 7 days)")
    outcomes = commands.add_parser("outcomes", help="outcome store (results/outcomes): backfill, verify, rebuild, query")
    outcomes.add_argument("action", choices=("backfill", "verify", "rebuild", "query"))
    outcomes.add_argument("--store", type=Path, default=Path("results/outcomes"),
                          help="store directory (default results/outcomes)")
    outcomes.add_argument("--runs", type=Path, default=Path("runs"), help="backfill: runs root, read only")
    outcomes.add_argument("--report", type=Path, default=None, help="backfill: write the counts as JSON here")
    outcomes.add_argument("--card", help="query: a card id; prints its records and its own-history matches")
    return parser


def run_command(args) -> int:
    from hybrid.literature.keys import ApiKeyError
    from hybrid.llm import ConfigError, GatewayUnavailable
    from hybrid.research.orchestrator import build_orchestrator
    from hybrid.slots import SlotLimitError, VMUnavailable

    run_dir = args.run_dir
    if run_dir is None:
        if args.resume:
            raise SystemExit("--resume needs --run-dir")
        run_dir = Path("runs") / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    overrides = run_overrides(args)
    try:
        orchestrator = build_orchestrator(args.config, run_dir, resume=args.resume, overrides=overrides,
                                          routing=args.routing)
    except (GatewayUnavailable, ConfigError, ApiKeyError, SlotLimitError, VMUnavailable) as exc:
        print(f"cannot start the run: {exc}", file=sys.stderr)
        return 2
    print(f"run dir: {orchestrator.run_dir}", flush=True)
    summary = asyncio.run(orchestrator.run())
    print(json.dumps(summary, indent=2))
    return 0


def local_model_check(args) -> bool:
    """Local-model preflight (hybrid/localmodels.py) for `run` and `plumb`: every LM Studio model the command's stages
    route to must be loaded with the context models.toml requires. A config, routing or --stages error is left to the
    command, which reports it with its own message."""
    from hybrid.llm import ConfigError, load_config
    from hybrid.research.orchestrator import ROOT, load_run_config
    from hybrid.research.plumb import PlumbError, parse_stages
    from hybrid.research.stages.common import pipeline_order

    if args.command == "run":
        overrides = run_overrides(args)
    elif args.command in ("calibrate", "planted"):
        overrides = {}
    else:
        overrides = {"s5.mode": args.s5_mode} if args.s5_mode is not None else {}
    cfg = load_run_config(args.config, overrides)
    try:
        models = load_config(ROOT / cfg["run"]["models_config"], args.routing)
        if args.command == "calibrate":
            stages = ("M", "C")
        elif args.command == "planted":
            from hybrid.research.planted import planted_config

            cfg = planted_config(cfg, store_path=Path(args.store), n_cards=1)
            stages = pipeline_order(cfg)
        else:
            stages = pipeline_order(cfg) if args.command == "run" else parse_stages(args.stages)
        if args.command == "run" and cfg["generation"]["draft_only"]:
            stages = stages[:1]
    except (ConfigError, PlumbError):
        return True
    arms = args.command in ("run", "planted") and cfg["generation"]["mode"] == "arms"
    return localmodels.guard_cli(models, stages, f"the research {args.command}", arms=arms)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return inspect_run(args.run_dir)
    if args.command == "outcomes":
        from hybrid.research.outcome_cli import outcomes_command

        return outcomes_command(args)
    if args.command == "planted" and args.action != "run":
        from hybrid.research.planted import planted_command

        return planted_command(args)
    # Free-space precheck (hybrid/diskguard.py): refuse to start a run or plumbing check on a nearly full volume.
    if not diskguard.guard_cli(diskguard.require_run_space, f"the research {args.command}"):
        return 2
    # Local-model preflight: refuse when a routed LM Studio model is not loaded with the context the config needs.
    if not local_model_check(args):
        return 2
    if args.command == "run":
        return run_command(args)
    if args.command == "plumb":
        from hybrid.research.plumb import plumb_command

        return plumb_command(args)
    if args.command == "calibrate":
        from hybrid.research.calibration import calibrate_command

        return calibrate_command(args)
    if args.command == "planted":
        from hybrid.research.planted import planted_command

        return planted_command(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
