"""`python -m hybrid.research outcomes <action>`: the outcome store from the command line.

- backfill: import every card under --runs (read-only) into --store; prints the counts (and writes them to --report).
- verify: check the JSONL hash chain and that the SQLite index holds the same rows.
- rebuild: write the SQLite index again from the JSONL.
- query --card <id>: the card's records, and its own-history matches against the rest of the store (every record
  except the card's own record in the run of its latest record, so its deaths in other runs are listed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from hybrid.research.outcome_records import backfill
from hybrid.research.orchestrator import OUTCOMES_DEFAULTS as DEFAULTS
from hybrid.research.outcomes import OutcomeStore, own_history


def outcomes_command(args) -> int:
    store = OutcomeStore(args.store)
    if args.action == "backfill":
        started = time.monotonic()
        report = backfill(args.runs, store).as_dict()
        report["seconds"] = round(time.monotonic() - started, 2)
        text = json.dumps(report, indent=2)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if args.action == "verify":
        ok, bad = store.verify()
        same = store.jsonl_rows() == store.sqlite_rows()
        print(f"chain: {'intact' if ok else f'broken at seq {bad}'}; records: {len(store.lines())}; "
              f"sqlite matches jsonl: {same}")
        return 0 if ok and same else 1
    if args.action == "rebuild":
        print(f"rebuilt {store.sqlite} with {store.rebuild_sqlite()} rows")
        return 0
    if args.action == "query":
        if not args.card:
            print("query needs --card", file=sys.stderr)
            return 2
        mine = [s for s in store.current() if s.record["card_id"] == args.card]
        if not mine:
            print(f"no record for {args.card} in {args.store}", file=sys.stderr)
            return 1
        latest = mine[-1].record
        # As S2 queries it in the latest record's run: that run's own record is left out, other runs' are not.
        matches = own_history(store, {"id": args.card, "core_idea": latest.get("core_idea") or "",
                                      "summary": latest["summary"]},
                              latest["mechanism_signature"], current_run=latest["run"],
                              limit=DEFAULTS["history_max"], jaccard_min=DEFAULTS["jaccard_min"],
                              tfidf_min=DEFAULTS["tfidf_min"])
        print(json.dumps({"records": [{"record_id": s.record_id, "run": s.record["run"], "outcome": s.record["outcome"],
                                       "terminal_stage": s.record["terminal_stage"],
                                       "kill_codes": s.record["kill_codes"],
                                       "signature": s.record["mechanism_signature"]["signature"]} for s in mine],
                          "own_history": [m.as_dict() for m in matches]}, indent=2))
        return 0
    raise AssertionError(f"unhandled outcomes action {args.action!r}")
