"""Command line for the local arXiv index.

    python -m hybrid.literature harvest [--data-dir DIR] [--log-dir DIR] [--sets SET ...] [--from DATE]
                                        [--defer-after N]
    python -m hybrid.literature info [--data-dir DIR]
    python -m hybrid.literature keys [--live]

`keys` reports whether OpenAlex and Semantic Scholar API keys are configured and where they come from (the last
four characters only). With --live it makes one search request per found key and prints the HTTP status; exit 1
if any is not 200, exit 2 if a key file is readable by group or others.

`harvest` resumes whatever the index's saved state says is unfinished. A set whose requests fail N times in a
row is deferred, the others run, and deferred sets get one more pass at the end; if any is still deferred the
run exits 1. SIGINT or SIGTERM stops it after the page in flight is stored; a second signal stops immediately.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from hybrid.literature.arxiv_index import ArxivIndexReader, ArxivIndexWriter
from hybrid.literature.harvest import (
    BASE_URL,
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_PLANS,
    MAX_ATTEMPTS_PER_REQUEST,
    Harvester,
    new_client,
)
from hybrid.literature.key_report import report_keys
from hybrid.literature.keys import ApiKeyError, load_api_keys
from hybrid.literature.sources.arxiv_local import DEFAULT_INDEX_PATH, REPO_ROOT

INDEX_FILE = Path(DEFAULT_INDEX_PATH).name
MANIFEST_FILE = "manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hybrid.literature", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    default_data = REPO_ROOT / Path(DEFAULT_INDEX_PATH).parent
    harvest = commands.add_parser("harvest", help="harvest arXiv metadata over OAI-PMH into the local index")
    harvest.add_argument("--data-dir", type=Path, default=default_data, help="holds arxiv.sqlite and manifest.json")
    harvest.add_argument("--log-dir", type=Path, default=None,
                         help="harvest.log and a manifest copy go here (default results/literature-<UTC date>)")
    harvest.add_argument("--sets", nargs="+", metavar="SET", default=None,
                         help=f"subset of {[p.set_spec for p in DEFAULT_PLANS]} (default: all)")
    harvest.add_argument("--from", dest="from_date", default=None, metavar="YYYY-MM-DD",
                         help="only records modified on or after this date (applies to sets not yet started)")
    harvest.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_S,
                         help="seconds between requests (default %(default)s)")
    harvest.add_argument("--max-pages", type=int, default=None, help="stop after this many pages (for trial runs)")
    harvest.add_argument("--base-url", default=BASE_URL)
    harvest.add_argument("--defer-after", type=int, default=MAX_ATTEMPTS_PER_REQUEST, metavar="N",
                         help="failed attempts on one request before its set is deferred (default %(default)s)")

    info = commands.add_parser("info", help="print the index snapshot and per-set progress")
    info.add_argument("--data-dir", type=Path, default=default_data)

    keys = commands.add_parser("keys", help="show which OpenAlex and Semantic Scholar API keys are configured")
    keys.add_argument("--live", action="store_true",
                      help="also make one request per found key and report the HTTP status")
    return parser


def _keys(args: argparse.Namespace) -> int:
    try:
        found = load_api_keys()
    except ApiKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lines = report_keys(found, live=args.live)
    for line in lines:
        print(line.text)
    return 0 if all(line.ok for line in lines) else 1


def _harvest(args: argparse.Namespace) -> int:
    plans = DEFAULT_PLANS
    if args.sets:
        known = {plan.set_spec: plan for plan in DEFAULT_PLANS}
        unknown = [spec for spec in args.sets if spec not in known]
        if unknown:
            raise SystemExit(f"unknown set(s) {unknown}; choose from {sorted(known)}")
        plans = tuple(known[spec] for spec in args.sets)
    log_dir = args.log_dir or REPO_ROOT / "results" / f"literature-{datetime.now(timezone.utc):%Y%m%d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(log_dir / "harvest.log")

    stop = threading.Event()

    def request_stop(signum: int, _frame: object) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        logging.getLogger("hybrid.literature.harvest").warning(
            "signal %s received: finishing the page in flight, then stopping (signal again to stop now)", signum
        )
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    index_path = args.data_dir / INDEX_FILE
    with ArxivIndexWriter(index_path) as writer, new_client() as http:
        harvester = Harvester(
            writer,
            http,
            plans,
            stop=stop,
            base_url=args.base_url,
            min_interval_s=args.min_interval,
            from_date=args.from_date,
            max_pages=args.max_pages,
            manifest_paths=(args.data_dir / MANIFEST_FILE, log_dir / MANIFEST_FILE),
            max_attempts=args.defer_after,
        )
        logging.getLogger("hybrid.literature.harvest").info(
            "harvest start: index %s, sets %s", index_path, [p.set_spec for p in plans]
        )
        outcome = harvester.run()
        deferred = harvester.deferred_sets()
    logging.getLogger("hybrid.literature.harvest").info("harvest finished with status %s", outcome)
    if outcome == "deferred":
        logging.getLogger("hybrid.literature.harvest").error(
            "harvest incomplete: %d set(s) still deferred: %s; run the harvest again to resume them",
            len(deferred), ", ".join(deferred),
        )
        return 1
    return 0


def _info(args: argparse.Namespace) -> int:
    reader = ArxivIndexReader(args.data_dir / INDEX_FILE)
    if not reader.exists():
        print(f"no index at {reader.path}", file=sys.stderr)
        return 1
    manifest = args.data_dir / MANIFEST_FILE
    info = reader.info()
    print(f"{info.path}: {info.record_count} records, snapshot {info.snapshot_ts}, complete {info.complete}")
    deferred = reader.deferred_sets()
    if deferred:
        print(f"deferred: {', '.join(deferred)} (resumption tokens kept; the next harvest resumes them)")
    if manifest.is_file():
        print(manifest.read_text(encoding="utf-8"))
    return 0


def _configure_logging(log_file: Path) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return {"harvest": _harvest, "info": _info, "keys": _keys}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
