"""Verify a run's ledger, print a text summary, and build a static HTML report under <run>/report/.

The report reads only the ledger, the card files and the artifact files. Transcript and artifact
links are relative to the run directory, so the whole run folder can be zipped and opened from disk.
"""
import difflib
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from hybrid.research.ledger import Ledger
from hybrid.research.schemas import validate
from hybrid.research.store import Store

TEMPLATE_DIR = Path(__file__).resolve().parent / "report"
REPORT_DIR = "report"
PAGES = {"index.html": "overview.html", "cards.html": "cards.html", "lineage.html": "lineage.html",
         "models.html": "models.html"}


def inspect_run(run_dir) -> int:
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "ledger.jsonl").is_file():
        print(f"No ledger.jsonl in {run_dir}")
        return 2
    ledger = Ledger(run_dir)
    ledger_ok, first_bad_seq = ledger.verify()
    if not ledger_ok:
        print(f"Ledger: BROKEN at seq {first_bad_seq}. Report not built.")
        return 1
    store = Store(run_dir, ledger)
    summary = summarize(run_dir, ledger, store)
    print(render_text(summary))
    report_index = build_report(run_dir, store, summary)
    print(f"Report: {report_index}")
    return 1 if summary["file_problems"] else 0


# Ids in a verdict's by_calls that name executor jobs, not model calls: stage jobs, and the Lean and math-library
# jobs of stages M and C (hybrid/math/runner.py).
EXECUTOR_JOB_PREFIXES = ("job-", "lean-", "mathlib-")
# A call on which every backend entry failed is recorded with backend and model null.
NO_BACKEND = "(all failed)"


def _usage_rows(llm_events: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for event in llm_events:
        data = event["data"]
        backend = data["backend"] if data["backend"] is not None else NO_BACKEND
        model = data["model"] if data["model"] is not None else NO_BACKEND
        row = groups.setdefault((backend, model), {
            "backend": backend, "model": model, "calls": 0, "errors": 0,
            "input": 0, "output": 0, "cached": 0, "calls_without_usage": 0,
            "latency_total": 0.0, "roles": Counter(),
        })
        usage = data["usage"]
        row["calls"] += 1
        row["roles"][data["role"]] += 1
        row["latency_total"] += data["latency_s"]
        if data.get("error"):
            row["errors"] += 1
        if any(usage[key] is None for key in ("input", "output", "cached")):
            row["calls_without_usage"] += 1
        for key in ("input", "output", "cached"):
            if usage[key] is not None:
                row[key] += usage[key]
    rows = sorted(groups.values(), key=lambda row: (row["backend"], row["model"]))
    for row in rows:
        row["latency_mean"] = row["latency_total"] / row["calls"]
        row["roles"] = ", ".join(f"{role} {count}" for role, count in sorted(row["roles"].items()))
    return rows


def _wall_seconds(events: list[dict]) -> float:
    if not events:
        return 0.0
    start = datetime.fromisoformat(events[0]["ts"])
    end = datetime.fromisoformat(events[-1]["ts"])
    return (end - start).total_seconds()


def _hms(seconds: float) -> str:
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _recorded_json(run_dir: Path, data: dict) -> dict | None:
    """An artifact's JSON content, or None when the file is missing or no longer matches its sha256
    (file_problems reports both)."""
    path = run_dir / data["path"]
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != data["sha256"]:
        return None
    return json.loads(raw)


def creativity_metrics(run_dir: Path, events: list[dict], store: Store) -> dict:
    """Textbook transplants versus modified ideas (from the latest S2 reduction critic per card), the lenses
    of the cards that received a verdict at each stage, and the size of each card's core."""
    created = {e["data"]["card_id"]: e["data"] for e in events if e["type"] == "card_created"}
    critic_artifacts: dict[str, dict] = {}
    for event in events:
        data = event["data"]
        if event["type"] == "artifact" and data["kind"] == "reduction_critic" and data.get("card_id"):
            critic_artifacts[data["card_id"]] = data

    textbook = {"transplant": [], "modified": [], "not_assessed": []}
    by_model: dict[str, Counter] = defaultdict(Counter)
    lens_stage: dict[str, Counter] = defaultdict(Counter)
    state_sizes, update_sizes = [], []
    for card_id in store.list_cards():
        card = store.get_card(card_id)
        for stage in card["status"]["verdicts"]:
            lens_stage[stage][card["lens"]] += 1
        state_sizes.append(len(card["mechanism"]["state"]))
        update_sizes.append(len(card["mechanism"]["update"]))
        record = critic_artifacts.get(card_id)
        critic = _recorded_json(run_dir, record) if record else None
        if critic is None or "textbook_transplant" not in critic:
            textbook["not_assessed"].append(card_id)
            continue
        bucket = "transplant" if critic["textbook_transplant"] else "modified"
        textbook[bucket].append(card_id)
        by_model[created.get(card_id, {}).get("model", "unknown")][bucket] += 1

    def spread(values: list[int]) -> dict:
        return {"mean": round(sum(values) / len(values), 1), "max": max(values)} if values else {"mean": 0, "max": 0}

    return {
        "textbook": {key: sorted(ids) for key, ids in textbook.items()},
        "textbook_by_model": {model: dict(counts) for model, counts in sorted(by_model.items())},
        "lens_stage": {stage: dict(sorted(lens_stage[stage].items())) for stage in sorted(lens_stage)},
        "core_size": {"state_objects": spread(state_sizes), "update_equations": spread(update_sizes)},
    }


def escape_metrics(run_dir: Path, events: list[dict], store: Store) -> dict:
    """The known families S2 reduced cards to (latest reduction critic of each reduces_to_known card), and
    the escape lineage: each escape card with its parent, the parent's family and where the child is now."""
    critic_artifacts: dict[str, dict] = {}
    for event in events:
        data = event["data"]
        if event["type"] == "artifact" and data["kind"] == "reduction_critic" and data.get("card_id"):
            critic_artifacts[data["card_id"]] = data

    def family(card_id: str) -> str | None:
        """reduces_to of a card reduced at S2; None when S2 did not reduce it."""
        card = store.get_card(card_id)
        if card["status"]["verdicts"].get("S2", {}).get("verdict") != "reduces_to_known":
            return None
        record = critic_artifacts.get(card_id)
        critic = _recorded_json(run_dir, record) if record else None
        return critic["reduces_to"] if critic is not None else "(critic artifact missing or changed)"

    families: Counter = Counter()
    for card_id in store.list_cards():
        name = family(card_id)
        if name is not None:
            families[name] += 1
    lineage = []
    for event in events:
        data = event["data"]
        if event["type"] == "card_created" and data["op"] == "escape":
            parent, child = data["parents"][0], data["card_id"]
            child_card = store.get_card(child)
            stage = child_card["status"]["stage"]
            lineage.append({
                "parent": parent, "parent_title": store.get_card(parent)["title"], "parent_family": family(parent),
                "child": child, "child_title": child_card["title"], "child_stage": stage,
                "child_verdict": child_card["status"]["verdicts"].get(stage, {}).get("verdict"),
                "child_family": family(child),
            })
    return {
        "reduces_to": dict(sorted(families.items(), key=lambda item: (-item[1], item[0]))),
        "requested": sum(1 for e in events if e["type"] == "escape_requested"),
        "skipped": sum(1 for e in events if e["type"] == "escape_skipped"),
        "lineage": lineage,
    }


def _ledger_view(ledger: dict | None) -> tuple[list[dict], list[dict]]:
    """(definitions, premises) of an ``assumption_ledger`` artifact. Ledgers written before definitions were
    separated from premises (records of 2026-10-12 and earlier, such as the first live run on card-46dbb349) hold
    one ``assumptions`` list; each entry shows as a premise whose testability was not recorded."""
    if ledger is None:
        return [], []
    if "premises" in ledger:
        return ledger["definitions"], ledger["premises"]
    return [], [dict(a, testable_by="not recorded", why_needed="", measurement_plan=None, citation=None)
                for a in ledger["assumptions"]]


def _claim_view(claim: dict) -> dict:
    """A claim tier record; old records name their open premises ``open_assumptions`` and have no circular flag."""
    if "open_premises" in claim:
        return claim
    return dict(claim, open_premises=claim["open_assumptions"], circular_premises=[])


def _math_verdicts(run_dir: Path, events: list[dict], card_id: str | None = None) -> list[tuple[dict, dict]]:
    """Every math_verdict artifact (of one card when ``card_id`` is given), in ledger order, with its event data and
    its premise ledger view: (event data, {record, definitions, premises})."""
    names = {e["data"]["name"]: e["data"] for e in events
             if e["type"] == "artifact" and e["data"]["kind"] == "assumption_ledger"}
    out = []
    for event in events:
        data = event["data"]
        if event["type"] != "artifact" or data["kind"] != "math_verdict" or not data.get("card_id"):
            continue
        if card_id is not None and data["card_id"] != card_id:
            continue
        record = _recorded_json(run_dir, data)
        if record is None:
            continue
        ledger_data = names.get(record["assumption_ledger"])
        definitions, premises = _ledger_view(_recorded_json(run_dir, ledger_data) if ledger_data else None)
        out.append((data, {"record": dict(record, claims=[_claim_view(c) for c in record["claims"]]),
                           "definitions": definitions, "premises": premises}))
    return out


def math_metrics(run_dir: Path, events: list[dict]) -> dict:
    """Stages M and C (s3.mode = "math", or plumb --stages M,C): the latest math_verdict of each card per stage,
    with its kill codes, the tier record of each claim (conditional claims with their open premises, circular
    flags), the definitions and the premise ledger; the kill-code counts, and the role calls per role."""
    latest: dict[tuple[str, str], dict] = {}
    for data, view in _math_verdicts(run_dir, events):
        latest[(data["card_id"], view["record"]["stage"])] = view
    rows, kill_codes = [], Counter()
    for (card_id, stage), view in sorted(latest.items()):
        record = view["record"]
        kill_codes.update(record["kill_codes"])
        rows.append({"card_id": card_id, "stage": stage, "verdict": record["verdict"],
                     "kill_codes": record["kill_codes"],
                     "claims": [(c["claim_id"], c["status"], c["label"], c["open_premises"], c["circular_premises"])
                                for c in record["claims"]],
                     "definitions": [(d["id"], d["kind"]) for d in view["definitions"]],
                     "premises": [(p["id"], p["status"], p["kind"], p["testable_by"]) for p in view["premises"]],
                     "scheduled": [m["premise_id"] for m in record.get("scheduled_measurements", [])],
                     "target_coupled": _coupled_view(record.get("target_coupled"))})
    role_calls = Counter(e["data"]["role"] for e in events if e["type"] == "math_role_call")
    return {"verdicts": rows, "kill_codes": dict(sorted(kill_codes.items())), "role_calls": dict(sorted(
        role_calls.items())), "frozen": sum(1 for e in events if e["type"] == "math_statements_frozen"),
            "checks": sum(1 for e in events if e["type"] == "math_check")}


def _coupled_view(block: dict | None) -> dict | None:
    """FIX-17: the skill scale at target a stage M verdict records (absent in verdicts written before FIX-17)."""
    if not block:
        return None
    return {"status": block["status"], "applied": block.get("applied", False), "reasons": block.get("reasons", []),
            "slots": {slot: entry["scale"] for slot, entry in (block.get("slots") or {}).items()}}


def render_math(metrics: dict) -> list[str]:
    if not metrics["verdicts"] and not metrics["frozen"]:
        return []
    lines = ["", f"Math stages M and C: {metrics['frozen']} statement sets frozen, {metrics['checks']} checks, "
                 "role calls " + (", ".join(f"{r} {n}" for r, n in metrics["role_calls"].items()) or "none")]
    for row in metrics["verdicts"]:
        codes = f" [{', '.join(row['kill_codes'])}]" if row["kill_codes"] else ""
        lines.append(f"  {row['card_id']} {row['stage']} {row['verdict']}{codes}")
        for cid, status, label, open_premises, circular in row["claims"]:
            extra = (f"; conditional on {', '.join(open_premises)}" if status == "conditional" else "") + (
                f"; circular premises {', '.join(circular)}" if circular else "")
            lines.append(f"    claim {cid}: {status} ({label}){extra}")
        lines.append("    definitions: " + (", ".join(f"{did} ({kind})" for did, kind in row["definitions"])
                                            or "none"))
        lines.extend(f"    premise {pid}: {status} (kind {kind}, testable by {testable})"
                     + (" CIRCULAR" if status == "circular" else "")
                     for pid, status, kind, testable in row["premises"])
        if row["scheduled"]:
            lines.append("    measurements deferred to test design: " + ", ".join(row["scheduled"]))
        coupled = row.get("target_coupled")
        if coupled:
            lines.append(f"    skill scale at target: {coupled['status']}"
                         + (" (gate applied)" if coupled["applied"] else " (recorded only)")
                         + (": " + ", ".join(f"{slot} {scale}" for slot, scale in coupled["slots"].items())
                            if coupled["slots"] else "; " + "; ".join(coupled["reasons"])))
    if metrics["kill_codes"]:
        lines.append("  kill codes: " + ", ".join(f"{code} {n}" for code, n in metrics["kill_codes"].items()))
    return lines


def _statements_view(record: dict, href: str | None) -> dict:
    """The frozen statements for the dossier. Records from before definitions were separated have no definitions
    and name an obligation's premises ``assumptions``."""
    definitions = {d["id"]: d for d in record.get("definitions", [])}
    obligations = []
    for o in record["obligations"]:
        premises = o["premises"] if "premises" in o else o["assumptions"]
        obligations.append({
            "id": o["id"], "claim_id": o["claim_id"], "premises": premises, "discharges": o["discharges"],
            "definitions": o.get("definitions", []),
            "statements": [{"id": s["id"], "form": s["form"], "sha256": s["sha256"][:12], "bound": s["bound"],
                            "text": s["spec"] if isinstance(s["spec"], str) else json.dumps(
                                s["spec"], indent=1, sort_keys=True)} for s in o["statements"]]})
    return {"href": href, "obligations": obligations, "definitions": list(definitions.values()),
            "dropped": len(record["dropped"]), "not_formalized": record["not_formalized"]}


def _math_dossier(run_dir: Path, events: list[dict], card_id: str, run_root: str) -> dict:
    """The card's frozen statements and definitions, every M and C verdict with its claim tiers, conditional claims,
    circular flags and premise ledger, and every math check, for the dossier."""
    mine = [e for e in events if e["type"] == "artifact" and e["data"].get("card_id") == card_id]
    statements, checks = None, []
    for event in mine:
        data = event["data"]
        href, _ = _run_link(data["path"], run_dir, run_root)
        if data["kind"] == "math_statements":
            record = _recorded_json(run_dir, data)
            if record is not None:
                statements = _statements_view(record, href)
        elif data["kind"] == "math_check":
            record = _recorded_json(run_dir, data)
            if record is not None:
                checks.append({"stage": record["stage"], "statement_id": record["statement_id"],
                               "purpose": record["purpose"], "form": record["form"], "status": record["status"],
                               "decided": record["decided"], "evidence": str(record.get("evidence") or "")[:600],
                               "jobs": record["job_ids"], "href": href, "ts": event["ts"]})
    verdicts = []
    for data, view in _math_verdicts(run_dir, events, card_id):
        record = view["record"]
        href, _ = _run_link(data["path"], run_dir, run_root)
        verdicts.append({"stage": record["stage"], "verdict": record["verdict"], "kill_codes": record["kill_codes"],
                         "claims": record["claims"], "href": href, "models": record["models"],
                         "definitions": view["definitions"], "premises": view["premises"],
                         "conditional": [c for c in record["claims"] if c["status"] == "conditional"],
                         "circular": [p for p in view["premises"] if p["status"] == "circular"],
                         "scheduled_measurements": record.get("scheduled_measurements", []),
                         "target_coupled": _coupled_view(record.get("target_coupled"))})
    return {"statements": statements, "verdicts": verdicts, "checks": checks}


BUDGET_EVENTS = ("lineage_assigned", "budget_charge", "loop_stop", "repair_rejected")


def budget_metrics(events: list[dict]) -> dict:
    """Loop budgets and stalls (Phase 8, hybrid.research.lineage): the caps the run started with, each lineage's
    ideas and model calls (counts include earlier runs' records when the outcome store seeded them), each card's
    idea key, revisions per loop and model calls, and every loop stop with its kill code. Empty for a run with no
    budget events (every run before Phase 8)."""
    if not any(e["type"] in BUDGET_EVENTS for e in events):
        return {}
    caps = None
    for event in events:
        if event["type"] == "run_start" and "loop_budgets" in event["data"]:
            caps = event["data"]["loop_budgets"]
    cards: dict[str, dict] = {}
    lineages: dict[str, dict] = {}
    idea_calls: dict[str, int] = {}
    stops: list[dict] = []
    repairs: list[dict] = []
    for event in events:
        data, kind = event["data"], event["type"]
        if kind == "lineage_assigned":
            cards[data["card_id"]] = {"card_id": data["card_id"], "lineage_id": data["lineage_id"],
                                      "idea_key": data["idea_key"], "joined": data["joined"], "via": data["via"],
                                      "admitted": data["admitted"], "loops": Counter(), "calls": 0, "stops": []}
            row = lineages.setdefault(data["lineage_id"], {"lineage_id": data["lineage_id"], "ideas": 0,
                                                           "ideas_cap": data["ideas_cap"], "calls": 0, "cards": [],
                                                           "refused": 0})
            row["ideas"] = data["lineage_ideas"]
            row["calls"] = max(row["calls"], data["lineage_calls"])
            row["cards"].append(data["card_id"])
            row["refused"] += 0 if data["admitted"] else 1
            idea_calls.setdefault(data["idea_key"], data["idea_calls"])
        elif kind == "budget_charge":
            card = cards[data["card_id"]]
            if data["loop"] == "model_calls":
                card["calls"] += 1
                idea_calls[data["idea_key"]] = data["used"]
                lineages[data["lineage_id"]]["calls"] = data["lineage_used"]
            else:
                card["loops"][data["loop"]] += 1
                if "lineage_ideas" in data:   # a send-back round takes an idea slot of the lineage
                    lineages[data["lineage_id"]]["ideas"] = data["lineage_ideas"]
        elif kind == "repair_rejected":
            repairs.append({key: data[key] for key in ("card_id", "stage", "loop", "round_id", "outside")})
        elif kind == "loop_stop":
            stop = {key: data[key] for key in ("card_id", "stage", "loop", "reason", "kill_code", "detail",
                                               "terminal")}
            stops.append(stop)
            if data["card_id"] in cards:
                cards[data["card_id"]]["stops"].append(stop)
    for card in cards.values():
        card["loops"] = dict(sorted(card["loops"].items()))
        card["idea_calls"] = idea_calls.get(card["idea_key"], 0)
    return {"caps": caps, "lineages": sorted(lineages.values(), key=lambda row: row["cards"][0]),
            "cards": cards, "stops": stops, "repairs_rejected": repairs,
            "kill_codes": dict(sorted(Counter(stop["kill_code"] for stop in stops).items())),
            "stalls": sum(1 for stop in stops if stop["reason"] == "stall"),
            "budget_stops": sum(1 for stop in stops if stop["reason"] == "budget"),
            "skipped": {kind: sum(1 for e in events if e["type"] == kind and
                                  str(e["data"].get("reason", "")).startswith(("budget:", "stall:")))
                        for kind in ("escape_skipped", "mutation_skipped")}}


def render_budgets(metrics: dict) -> list[str]:
    if not metrics:
        return []
    caps = metrics["caps"]
    lines = [""]
    if caps:
        loops = ", ".join(f"{loop} {cap}" for loop, cap in caps["loops"].items())
        lines.append(f"Loop budgets: per idea {caps['idea']['model_calls']} model calls, loops {loops}; per lineage "
                     f"{caps['lineage']['ideas']} ideas and {caps['lineage']['model_calls']} model calls")
        if caps.get("prior", {}).get("records"):
            lines.append(f"  counts include {caps['prior']['records']} outcome-store record(s) of earlier runs "
                         f"({caps['prior']['lineages']} lineages)")
    else:
        lines.append("Loop budgets:")
    cap_calls = caps["idea"]["model_calls"] if caps else "?"
    lineage_calls = caps["lineage"]["model_calls"] if caps else "?"
    for row in metrics["lineages"]:
        refused = f", {row['refused']} card(s) refused" if row["refused"] else ""
        lines.append(f"  lineage {row['lineage_id']}: {row['ideas']} of {row['ideas_cap']} ideas, {row['calls']} of "
                     f"{lineage_calls} model calls{refused}")
        for card_id in row["cards"]:
            card = metrics["cards"][card_id]
            loops = ", ".join(f"{loop} {n}" for loop, n in card["loops"].items()) or "no revisions"
            how = card["joined"] + (f" {card['via']}" if card["via"] else "")
            state = "" if card["admitted"] else " REFUSED"
            lines.append(f"    {card_id} ({how}{state}) idea {card['idea_key']}: {loops}; {card['calls']} model "
                         f"call(s) on this card, idea total {card['idea_calls']} of {cap_calls}")
    lines.append(f"Loop stops: {metrics['stalls']} stall(s), {metrics['budget_stops']} budget stop(s); escapes "
                 f"skipped {metrics['skipped']['escape_skipped']}, mutations skipped "
                 f"{metrics['skipped']['mutation_skipped']}")
    for stop in metrics["stops"]:
        how = "ended the card" if stop["terminal"] else "the gate decided after it"
        lines.append(f"  {stop['card_id']} {stop['stage']} {stop['kill_code']} ({how}): {stop['detail'][:200]}")
    if metrics["kill_codes"]:
        lines.append("  codes: " + ", ".join(f"{code} {n}" for code, n in metrics["kill_codes"].items()))
    if metrics["repairs_rejected"]:
        lines.append(f"Repairs rejected without running checks: {len(metrics['repairs_rejected'])}")
        for repair in metrics["repairs_rejected"]:
            lines.append(f"  {repair['card_id']} {repair['stage']} {repair['loop']} {repair['round_id']}: touched "
                         f"{', '.join(repair['outside'])}, which its failures did not name")
    return lines


def override_record(events: list[dict]) -> dict | None:
    """The operator_override event of a plumbing check (hybrid.research.plumb), with its seq and time; None
    for a normal run. The report and the text summary show it prominently."""
    for event in events:
        if event["type"] == "operator_override":
            return dict(event["data"], seq=event["seq"], ts=event["ts"])
    return None


def summarize(run_dir: Path, ledger: Ledger, store: Store) -> dict:
    events = list(ledger.iter_events())
    llm_events = [event for event in events if event["type"] == "llm_call"]
    verdict_events = [event for event in events if event["type"] == "verdict"]
    file_problems = store.file_problems()

    cards = []
    stage_counts: Counter = Counter()
    for card_id in store.list_cards():
        card = store.get_card(card_id)
        stage = card["status"]["stage"]
        stage_counts[stage] += 1
        verdicts = card["status"]["verdicts"]
        cards.append({
            "id": card_id,
            "title": card["title"],
            "lens": card["lens"],
            "version": card["version"],
            "stage": stage,
            "reached_verdict": verdicts[stage]["verdict"] if stage in verdicts else None,
            "reached_reasons": verdicts[stage]["reasons"] if stage in verdicts else [],
            "verdicts": [(s, verdicts[s]["verdict"]) for s in sorted(verdicts)],
        })

    usage = _usage_rows(llm_events)
    return {
        "run_id": run_dir.name,
        "ledger_events": len(events),
        "ledger_head": events[-1]["hash"] if events else None,
        "started": events[0]["ts"] if events else None,
        "finished": events[-1]["ts"] if events else None,
        "wall_seconds": _wall_seconds(events),
        "cards": cards,
        "stage_counts": dict(sorted(stage_counts.items())),
        "verdict_counts": sorted(Counter((e["data"]["stage"], e["data"]["verdict"]) for e in verdict_events).items()),
        "usage": usage,
        "calls_total": sum(row["calls"] for row in usage),
        "tokens_total": sum(row["input"] + row["output"] for row in usage),
        "file_problems": file_problems,
        "lineage": store.lineage(),
        "creativity": creativity_metrics(run_dir, events, store),
        "escapes": escape_metrics(run_dir, events, store),
        "math": math_metrics(run_dir, events),
        "budgets": budget_metrics(events),
        "arms": arms_summary(ledger, store, events),
        "phase7": phase7_metrics(events, store),
        "ops": {e["data"]["card_id"]: e["data"]["op"] for e in events if e["type"] == "card_created"},
        "override": override_record(events),
    }


def render_override(override: dict) -> list[str]:
    verdicts = ", ".join(f"{stage} {record['verdict'] if record else 'none (not run)'}"
                         for stage, record in override["source_verdicts"].items())
    return [
        "*** OPERATOR OVERRIDE: PLUMBING CHECK. THIS RUN IS NOT A CARD PASSING THE GATES. ***",
        f"  {override['statement']}",
        f"  reason     {override['reason']}",
        f"  source     run {override['source_run_id']}, card {override['card_id']} "
        f"(v{override['source_card']['version']})",
        f"  source verdicts  {verdicts}",
        f"  stages run here  {', '.join(override['stages_run'])}",
        f"  recorded as ledger event {override['seq']} at {override['ts']}",
        "",
    ]


def render_text(summary: dict) -> str:
    lines = render_override(summary["override"]) if summary.get("override") else []
    lines += [
        f"Run {summary['run_id']}",
        f"  started    {summary['started'] or '-'}",
        f"  finished   {summary['finished'] or '-'}",
        f"  wall time  {_hms(summary['wall_seconds'])}",
        f"Ledger: verified, {summary['ledger_events']} events, head {(summary['ledger_head'] or '-')[:12]}",
    ]
    if summary["file_problems"]:
        lines.append(f"Files: {len(summary['file_problems'])} problem(s)")
        lines.extend(f"  {problem}" for problem in summary["file_problems"])
    else:
        lines.append("Files: every card and artifact file matches its recorded sha256")

    lines.append("")
    lines.append(f"Cards: {len(summary['cards'])}")
    for stage, count in summary["stage_counts"].items():
        lines.append(f"  stage {stage}: {count}")

    lines.append("Verdicts:")
    if not summary["verdict_counts"]:
        lines.append("  none recorded")
    for (stage, verdict), count in summary["verdict_counts"]:
        lines.append(f"  {stage} {verdict}: {count}")

    lines.append("")
    lines.append("Calls per model:")
    lines.append(f"  {'backend':<18}{'model':<22}{'calls':>6}{'errors':>8}{'input':>10}{'output':>10}"
                 f"{'cached':>10}{'mean s':>9}")
    for row in summary["usage"]:
        lines.append(f"  {row['backend']:<18}{row['model']:<22}{row['calls']:>6}{row['errors']:>8}"
                     f"{row['input']:>10}{row['output']:>10}{row['cached']:>10}{row['latency_mean']:>9.2f}")
    if not summary["usage"]:
        lines.append("  no llm_call events")
    lines.append(f"Totals: {summary['calls_total']} calls, {summary['tokens_total']} input+output tokens")
    lines.extend(render_creativity(summary["creativity"]))
    lines.extend(render_escapes(summary["escapes"]))
    lines.extend(render_math(summary["math"]))
    lines.extend(render_budgets(summary["budgets"]))
    lines.extend(render_arms(summary.get("arms")))
    lines.extend(render_phase7(summary.get("phase7")))
    return "\n".join(lines)


def phase7_metrics(events: list[dict], store: Store) -> dict | None:
    """Refinement dialogue (stage R) and S2 difference statement counts (Phase 7), or None for a run without them."""
    rounds, answers, judged, ends = Counter(), Counter(), Counter(), Counter()
    overlaps = 0
    for event in events:
        data = event["data"]
        if event["type"] == "refine_round":
            rounds[data["card_id"]] += 1
            overlaps += 1 if data["family_overlap"] else 0
        elif event["type"] == "refine_answer":
            answers["accepted" if data["accepted"] else "rejected (growth limit)"] += 1
        elif event["type"] == "difference_judged":
            judged[f"attempt {data['attempt']} {'accepted' if data['accepted'] else 'rejected'}"] += 1
    if not rounds and not judged:
        return None
    gates = Counter()
    for card_id in store.list_cards():
        verdicts = store.get_card(card_id)["status"]["verdicts"]
        r = verdicts.get("R")
        if r is not None and r["reasons"][0].startswith("dialogue ended: "):
            ends[r["reasons"][0].split(" ")[2]] += 1
        s2 = verdicts.get("S2")
        gate = next((reason for reason in (s2 or {}).get("reasons", []) if reason.startswith("gate: ")), None)
        for code in ("excluded_family", "no_difference", "not_radical:R3"):
            if gate is not None and gate.startswith(f"gate: {code}"):
                gates[gate.split(": ")[1] if code == "excluded_family" else code] += 1
    return {"dialogue_cards": len(rounds), "challenger_rounds": sum(rounds.values()), "answers": dict(answers),
            "dialogue_ends": dict(ends), "family_overlap_rounds": overlaps, "difference": dict(judged),
            "s2_kills": dict(gates)}


def render_phase7(metrics: dict | None) -> list[str]:
    if not metrics:
        return []
    lines = ["", "Refinement dialogue and difference statement (Phase 7):",
             f"  dialogue: {metrics['dialogue_cards']} card(s), {metrics['challenger_rounds']} challenger round(s), "
             f"answers {metrics['answers'] or 'none'}, ends {metrics['dialogue_ends'] or 'none'}"]
    if metrics["family_overlap_rounds"]:
        lines.append(f"  challenger in the proposer's family (fallback) in {metrics['family_overlap_rounds']} round(s)")
    lines.append(f"  difference statements: {metrics['difference'] or 'none'}")
    lines.append(f"  S2 kills: {metrics['s2_kills'] or 'none'}")
    return lines


def arms_summary(ledger: Ledger, store: Store, events: list[dict]) -> dict | None:
    """Per-arm metrics of generation mode "arms" (Phase 6), or None for a run without arm drafts."""
    if not any(event["type"] == "arm_draft" for event in events):
        return None
    from hybrid.research.stages.g_generate import arm_metrics

    starts = [event["data"] for event in events if event["type"] == "run_start"]
    order = tuple(starts[0].get("stages") or ("S0", "S1", "S2", "S3", "S4", "S5", "S6"))
    coverage = [event["data"]["coverage"] for event in events if event["type"] == "arm_metrics"]
    return {"arms": arm_metrics(ledger, store, order), "order": order, "coverage": coverage[-1] if coverage else None,
            "generation": (starts[-1].get("generation") or {})}


def render_arms(metrics: dict | None) -> list[str]:
    if not metrics:
        return []
    lines = ["", "Generation arms (drafts, schema failures = JSON schema + card invalid + arm structure):"]
    lines.append(f"  {'arm':<12}{'drafts':>7}{'admit':>6}{'schema':>7}{'rate':>6}{'excl':>5}{'closed':>7}"
                 f"{'nodiff':>7}{'dup':>4}{'other':>6}{'newcell':>8}{'reward':>7}{'calls':>6}{'tok out':>9}  survivors")
    for key, m in metrics["arms"].items():
        other = m["transport_failure"] + m["classifier_failure"] + m["budget"]
        rate = "-" if m["schema_failure_rate"] is None else f"{m['schema_failure_rate']:.2f}"
        survivors = " ".join(f"{stage}:{m['survivors'].get(stage, 0)}" for stage in metrics["order"])
        lines.append(f"  {key:<12}{m['drafts']:>7}{m['admitted']:>6}{m['schema_failures']:>7}{rate:>6}"
                     f"{m['excluded_family']:>5}{m['closed_cell']:>7}{m['no_difference']:>7}{m['duplicate']:>4}"
                     f"{other:>6}{m['new_cells']:>8}{m['rewards']:>7}{m['calls']:>6}{m['tokens_out']:>9}  {survivors}")
    base = (metrics["generation"].get("base_arm") or {})
    if base:
        lines.append(f"  base-model arm A5: {'available' if base.get('available') else 'not available'} "
                     f"({base.get('reason')}); weights {metrics['generation'].get('weights')}")
    coverage = metrics.get("coverage")
    if coverage:
        axes = ", ".join(f"{axis} {v['filled']}/{v['values']}" for axis, v in coverage["per_axis"].items())
        lines.append(f"  archive: {coverage['cells_filled']} cells filled; per axis {axes}")
    return lines


def render_escapes(metrics: dict) -> list[str]:
    lines = ["", "Reductions at S2 (reduces_to_known cards by known family):"]
    if not metrics["reduces_to"]:
        lines.append("  none")
    else:
        lines.append("  " + ", ".join(f"{name} {count}" for name, count in metrics["reduces_to"].items()))
    lines.append(f"Escapes: {metrics['requested']} requested, {metrics['skipped']} skipped, "
                 f"{len(metrics['lineage'])} created")
    for row in metrics["lineage"]:
        verdict = row["child_verdict"] or "no verdict"
        after = f", reduced to {row['child_family']}" if row["child_family"] else ""
        lines.append(f"  {row['parent']} ({row['parent_family']}) -> {row['child']} "
                     f"[{row['child_stage']} {verdict}{after}] {row['child_title']}")
    return lines


def render_creativity(metrics: dict) -> list[str]:
    textbook = metrics["textbook"]
    assessed = len(textbook["transplant"]) + len(textbook["modified"])
    lines = ["", "Creativity (S2 reduction critic: core idea is a textbook construction applied unchanged?):",
             f"  textbook transplants {len(textbook['transplant'])}, modified {len(textbook['modified'])} "
             f"(of {assessed} assessed); not assessed {len(textbook['not_assessed'])}"]
    for model, counts in metrics["textbook_by_model"].items():
        lines.append(f"  {model}: transplant {counts.get('transplant', 0)}, modified {counts.get('modified', 0)}")
    lines.append("Lenses per stage (cards with a verdict at that stage):")
    if not metrics["lens_stage"]:
        lines.append("  none")
    for stage, lenses in metrics["lens_stage"].items():
        lines.append(f"  {stage}: " + ", ".join(f"{lens} {count}" for lens, count in lenses.items()))
    size = metrics["core_size"]
    lines.append(f"Core size (latest versions): state objects mean {size['state_objects']['mean']}, "
                 f"max {size['state_objects']['max']}; update equations mean {size['update_equations']['mean']}, "
                 f"max {size['update_equations']['max']}")
    return lines


def _run_link(path: str, run_dir: Path, run_root: str) -> tuple[str | None, str | None]:
    """Return (relative href from a report page, None) or (None, reason the target cannot be linked)."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(run_dir)
    except ValueError:
        return None, "outside the run directory"
    if not candidate.is_file():
        return None, "file missing"
    return run_root + relative.as_posix(), None


def _unified_diff(previous: dict, current: dict, previous_version: int, version: int) -> str:
    before = json.dumps(previous, indent=2, sort_keys=True).splitlines()
    after = json.dumps(current, indent=2, sort_keys=True).splitlines()
    return "\n".join(difflib.unified_diff(before, after, f"v{previous_version}", f"v{version}", lineterm=""))


def _dossier(card_id: str, run_dir: Path, store: Store, events: list[dict], run_root: str,
             budgets: dict | None = None) -> dict:
    latest = store.latest_version(card_id)
    card = store.get_card(card_id, latest)
    warnings: list[str] = []

    versions = []
    previous = None
    for version in store.versions(card_id):
        current = store.get_card(card_id, version)
        href = f"{run_root}cards/{card_id}/v{version}.json"
        diff = "" if previous is None else _unified_diff(previous, current, version - 1, version)
        versions.append({"version": version, "href": href, "stage": current["status"]["stage"],
                         "diff": diff or ("first version" if previous is None else "no change from previous version")})
        previous = current

    calls = {event["data"]["call_id"]: event["data"] for event in events if event["type"] == "llm_call"}

    def call_entry(call_id: str) -> dict:
        if call_id.startswith(EXECUTOR_JOB_PREFIXES):
            return {"id": call_id, "label": "executor job", "href": None, "note": None}
        data = calls.get(call_id)
        if data is None:
            warnings.append(f"{call_id} is not in the ledger")
            return {"id": call_id, "label": "unknown call", "href": None, "note": "not in the ledger"}
        href, note = _run_link(data["transcript_path"], run_dir, run_root)
        if note is not None:
            warnings.append(f"transcript for {call_id}: {note}")
        return {"id": call_id, "label": f"{data['role']}: {data['backend']} {data['model']}", "href": href, "note": note}

    verdicts = []
    for event in events:
        if event["type"] == "verdict" and event["data"]["card_id"] == card_id:
            record = event["data"]
            verdicts.append({
                "stage": record["stage"], "verdict": record["verdict"], "card_version": record["card_version"],
                "reasons": record["reasons"], "ts": event["ts"],
                "calls": [call_entry(call_id) for call_id in record["by_calls"]],
            })

    literature = []
    artifacts = []
    bench = []
    obligation_checks = []
    s3_gates = []
    for event in events:
        data = event["data"]
        if event["type"] == "literature_receipt" and data.get("card_id") == card_id:
            validate("literature_receipt", data)
            literature.append(data)
        if event["type"] == "artifact" and data.get("card_id") == card_id:
            href, note = _run_link(data["path"], run_dir, run_root)
            artifacts.append({"kind": data["kind"], "name": data["name"], "sha256": data["sha256"],
                              "bytes": data["bytes"], "href": href, "note": note})
            if data["kind"] in ("obligation_check", "s3_gate"):
                path = run_dir / data["path"]
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != data["sha256"]:
                    warnings.append(f"{data['kind']} {data['name']} not shown: file missing or changed since recorded")
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                if data["kind"] == "s3_gate":
                    s3_gates.append(dict(record, href=href, ts=event["ts"]))
                else:
                    obligation_checks.append(_obligation_check_row(record, href, event["ts"]))
            if data["kind"] == "bench_result":
                bench_path = run_dir / data["path"]
                if not bench_path.is_file() or hashlib.sha256(bench_path.read_bytes()).hexdigest() != data["sha256"]:
                    warnings.append(f"bench result {data['name']} not shown: file missing or changed since recorded")
                    continue
                summary = json.loads(bench_path.read_text(encoding="utf-8"))
                validate("bench_result", summary)
                bench.append({
                    "name": data["name"],
                    "href": href,
                    "per_task": json.dumps(summary["per_task"], indent=2, sort_keys=True),
                    "baselines": json.dumps(summary["baselines"], indent=2, sort_keys=True),
                    "scaling": sorted(summary["scaling_slopes"].items()),
                })

    reached = card["status"]["stage"]
    return {
        "card": card,
        "root": "../",
        "run_root": run_root,
        "versions": versions,
        "verdicts": verdicts,
        "literature": literature,
        "artifacts": artifacts,
        "bench": bench,
        "obligation_checks": obligation_checks,
        "s3_gates": s3_gates,
        "math": _math_dossier(run_dir, events, card_id, run_root),
        "budget": (budget_metrics(events) if budgets is None else budgets).get("cards", {}).get(card_id),
        "warnings": warnings,
        "reached": reached,
        "toy_python": card["toy_example"]["python"],
        "expected": json.dumps(card["toy_example"]["expected"], indent=2, sort_keys=True),
    }


def _obligation_check_row(record: dict, href: str | None, ts: str) -> dict:
    """One row of the dossier's obligation check table (artifact written by stages/obligation_checks.py)."""
    counterexample = record["counterexample"]
    return {"obligation": record["obligation"], "kind": record["kind"] or "unclassified",
            "optional": record["optional"], "status": record["status"], "replayed": record["replayed"],
            "evidence": record["evidence"], "why": record["why"], "problems": record["problems"],
            "counterexample": None if counterexample is None else json.dumps(counterexample, indent=2, sort_keys=True,
                                                                             default=str),
            "jobs": [run["job_id"] for run in record["runs"]], "program": record["program"], "href": href, "ts": ts}


def _forest(lineage: dict, ops: dict[str, str]) -> list[dict]:
    nodes = lineage["nodes"]

    def build(card_id: str) -> dict:
        node = nodes[card_id]
        return {"id": card_id, "title": node["title"], "stage": node["stage"], "op": ops.get(card_id),
                "children": [build(child) for child in node["child_ids"]]}

    return [build(root) for root in lineage["roots"]]


def _ledger_line(summary: dict) -> str:
    head = (summary["ledger_head"] or "-")[:12]
    files = ("all recorded files match their sha256" if not summary["file_problems"]
             else f"{len(summary['file_problems'])} file problem(s)")
    return f"Ledger verified: {summary['ledger_events']} events, head {head}, {files}."


def build_report(run_dir: Path, store: Store, summary: dict) -> Path:
    events = list(store.ledger.iter_events())
    report_dir = run_dir / REPORT_DIR
    dossier_dir = report_dir / "dossiers"
    dossier_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    base_context = {
        "run_id": summary["run_id"],
        "root": "",
        "run_root": "../",
        "ledger_line": _ledger_line(summary),
        "file_problems": summary["file_problems"],
        "override": summary["override"],
    }

    for page, template_name in PAGES.items():
        context = dict(base_context, summary=summary, page_title=template_name.removesuffix(".html"))
        if template_name == "lineage.html":
            context["forest"] = _forest(summary["lineage"], summary["ops"])
        html = env.get_template(template_name).render(context)
        (report_dir / page).write_text(html, encoding="utf-8")

    for card in summary["cards"]:
        dossier = _dossier(card["id"], run_dir, store, events, run_root="../../", budgets=summary["budgets"])
        context = dict(base_context, root="../", run_root="../../", page_title=card["id"],
                       dossier=dossier, card_summary=card)
        html = env.get_template("dossier.html").render(context)
        (dossier_dir / f"{card['id']}.html").write_text(html, encoding="utf-8")

    return report_dir / "index.html"
