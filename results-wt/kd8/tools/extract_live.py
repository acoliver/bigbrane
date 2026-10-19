"""Extract saved PA2 evidence only. No network calls; no prompts or raw model output are exported.

Requests are attributed by the first card JSON object in each saved user prompt. Literature spans
are paired within a card, never by global event position. HTTP attempt counts are not in these
ledgers: source search receipts are reported separately, not relabeled as HTTP requests.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from hybrid.literature.paper import normalize_arxiv_id, normalize_title

ROOT = Path(__file__).resolve().parents[1]
REFEREE = {
    "card-3485a13f": {
        "Hwang": ("1110.5667", "Inducing Probabilistic Programs by Bayesian Program Merging"),
        "Srinivas": ("2204.09837", "Memory Bounds for the Experts Problem"),
        "Silver": ("1904.06317", "Few-Shot Bayesian Imitation Learning with Logical Program Policies"),
    },
    "card-f5ee7edb": {
        "Rendle": (None, "Factorization Machines"),
        "Rossi": ("1906.05059", "Higher-Order Ranking and Link Prediction"),
    },
    "card-a4f684fb": {
        "SciPy QAP 2-opt": (None, "Quadratic assignment problem (2-opt)"),
        "Rehn": (None, "Fundamental Permutation Group Algorithms for Symmetry Computation"),
    },
    "card-89248ee5": {
        "Jiang": ("1409.0334", "Storing Sequences in Binary Tournament-Based Neural Networks"),
        "Zhou": ("0901.0553", "Predicting Missing Links via Local Information"),
        "Sacramento": ("1207.7196", "Binary Willshaw Learning Yields High Synaptic Capacity for Long-Term Familiarity Memory"),
        "Stoklosa": ("2411.14480", "Associative Knowledge Graphs for Efficient Sequence Storage and Retrieval"),
    },
    "card-8154d3b4": {
        "Biamonte": ("1405.7375", "Tensor Network Contractions for #SAT"),
        "Stoudenmire": ("1605.05775", "Supervised Learning with Quantum-Inspired Tensor Networks"),
        "Yu": ("1711.00073", "Long-Term Forecasting using Higher-Order Tensor RNNs"),
    },
}
INCREMENTAL = tuple(cid for cid in REFEREE if cid != "card-8154d3b4")
PROBES = ("card-17100001", "card-17100003", "card-17100006", "card-17100008")


def ts(value):
    return datetime.fromisoformat(value)


def prompt_card(user):
    """Only an explicit primary card id counts; references to other cards in history do not."""
    offset = user.find("{")
    if offset < 0:
        return None
    try:
        card, _ = json.JSONDecoder().raw_decode(user[offset:])
    except json.JSONDecodeError:
        return None
    cid = card.get("id") if isinstance(card, dict) else None
    return cid if isinstance(cid, str) and re.fullmatch(r"card-[0-9a-f]+", cid) else None


def exact_match(candidate, work):
    arxiv_id, title = work
    ids = []
    if candidate.get("source") == "arxiv":
        ids.append(candidate.get("source_id", ""))
    ids.extend([candidate.get("arxiv_id", ""), *candidate.get("aliases", [])])
    url = candidate.get("url", "")
    if url.startswith(("https://arxiv.org/abs/", "http://arxiv.org/abs/")):
        ids.append(url.split("/abs/", 1)[1])
    id_match = arxiv_id and arxiv_id in {normalize_arxiv_id(value or "") for value in ids}
    title_match = normalize_title(candidate["title"]) == normalize_title(title)
    return "arxiv_id" if id_match else "full_title" if title_match else None


def exposure(call):
    """Read actual bounded listing positions from the saved prompt, not today's selection code."""
    user = call["user"]
    if "SEARCH RESULTS (" in user:
        section = user.split("SEARCH RESULTS (", 1)[1]
        kind = "classifier" if call["role"] == "prior_art" else "critic"
    elif "PRIOR-ART SEARCH (" in user:
        section = user.split("PRIOR-ART SEARCH (", 1)[1].split("OWN HISTORY", 1)[0]
        kind = "difference_checker" if "DIFFERENCE STATEMENT:" in user else "difference_writer"
    else:
        return None
    keys = re.findall(r"^\s*- \[?((?:arxiv|s2|openalex):[^\s\]]+)\]? \(", section, re.M)
    return {"kind": kind, "call_id": call["call_id"], "keys": keys}


def card_spans(events, cid):
    spans, start = [], None
    for event in events:
        if event["data"].get("card_id") != cid:
            continue
        if event["type"] == "prior_art_queries":
            if start is not None:
                spans.append((start, None))
            start = event
        elif event["type"] == "prior_art_result":
            spans.append((start, event))
            start = None
    if start is not None:
        spans.append((start, None))
    return spans


def registered_guard(verdict):
    if verdict["verdict"] == "reduces_to_known":
        return True
    if verdict["verdict"] != "fail":
        return False
    return any(reason.startswith(("gate: not radical enough", "gate: not_radical"))
               or reason.startswith("gate: prior_art_tested and no untested element")
               for reason in verdict["reasons"])


def extract(root=ROOT):
    out = {"runs": {}, "cards": defaultdict(list), "missing_data": []}
    for run in sorted((root / "runs").iterdir()):
        ledger = run / "ledger.jsonl"
        if not ledger.exists():
            out["missing_data"].append(f"{run.name}: missing ledger")
            continue
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        calls, unattributed = [], []
        for file in sorted((run / "llm/calls").glob("*.json")):
            call = json.loads(file.read_text())
            cid = prompt_card(call["user"])
            if cid is None:
                unattributed.append(call["call_id"])
            calls.append((cid, call))
        ends = [e for e in events if e["type"] == "run_end"]
        errors = [{"type": e["type"], "seq": e["seq"], "card_id": e["data"].get("card_id")}
                  for e in events if "error" in e["type"]]
        out["runs"][run.name] = {
            "complete": bool(ends and ends[-1]["data"].get("reason") == "complete"
                             and not ends[-1]["data"].get("queued_unfinished")),
            "end_reason": ends[-1]["data"].get("reason") if ends else None,
            "errors": errors, "unattributed_call_ids": unattributed,
            "saved_calls": len(calls), "recorded_calls": sum(e["type"] == "llm_call" for e in events),
            "elapsed_s": ends[-1]["data"].get("budget", {}).get("elapsed_s") if ends else None,
            "failed_model_calls": sum(bool(c.get("error")) for _, c in calls),
        }
        cards = sorted({e["data"]["card_id"] for e in events if e["data"].get("card_id")})
        for cid in cards:
            own_calls = [c for owner, c in calls if owner == cid]
            own = [e for e in events if e["data"].get("card_id") == cid]
            attempts = []
            previous_end = None
            spans = card_spans(events, cid)
            for attempt, (start, end) in enumerate(spans, 1):
                next_start = spans[attempt][0] if len(spans) > attempt else None
                first = start["seq"] if start else 0
                last = end["seq"] if end else next_start["seq"] - 1 if next_start else events[-1]["seq"]
                span = [e for e in own if first <= e["seq"] <= last]
                literature_calls = [c for c in own_calls if c["role"] == "prior_art"
                                    and (previous_end is None or ts(c["started_at"]) > previous_end)
                                    and (end is None or ts(c["finished_at"]) <= ts(end["ts"]))
                                    and (next_start is None or ts(c["finished_at"]) < ts(next_start["ts"]))]
                searches = [e["data"] for e in span if e["type"] == "literature_search"]
                related = [e["data"] for e in span if e["type"] == "literature_related"]
                result = end["data"] if end else {}
                candidates = result.get("candidates", [])
                decision_calls = [c for c in own_calls if end and ts(c["finished_at"]) >= ts(end["ts"])]
                if next_start:
                    decision_calls = [c for c in decision_calls if ts(c["started_at"]) < ts(next_start["ts"])]
                listings = [e for c in literature_calls + decision_calls if (e := exposure(c)) is not None]
                matches = {}
                for label, work in REFEREE.get(cid, {}).items():
                    hits = []
                    for rank, candidate in enumerate(candidates, 1):
                        basis = exact_match(candidate, work)
                        if basis:
                            key = candidate["key"]
                            hits.append({"key": key, "title": candidate["title"], "rank": rank, "match_basis": basis,
                                         "exposed": [{"kind": listing["kind"], "call_id": listing["call_id"],
                                                      "position": listing["keys"].index(key) + 1}
                                                     for listing in listings if key in listing["keys"]]})
                    matches[label] = {"retrieved": bool(hits), "hits": hits,
                                      "required_arxiv_id": work[0], "required_full_title": work[1]}
                failures = Counter(source for search in searches for source in search["errors"])
                related_failures = Counter(source for receipt in related for source in receipt["errors"])
                starts_at = min((ts(c["started_at"]) for c in literature_calls), default=ts(start["ts"]) if start else None)
                cost = {"searches": len(searches), "related_lookups": len(related),
                        "source_search_receipts": sum(e["type"] == "literature_receipt" for e in span),
                        "http_requests": None, "http_requests_note": "No per-HTTP-attempt ledger; search receipts include local/cache/skipped sources.",
                        "model_calls": len(literature_calls),
                        "model_request_attempts": sum(a.get("kind") == "request" for c in literature_calls for a in c["attempts"]),
                        "proposal_to_result_wall_s": round((ts(end["ts"]) - starts_at).total_seconds(), 3) if end and starts_at else None,
                        "queries_to_result_wall_s": round((ts(end["ts"]) - ts(start["ts"])).total_seconds(), 3) if start and end else None}
                query_event = start["data"] if start else {}
                attempts.append({"attempt": attempt, "complete": start is not None and end is not None,
                                 "start_seq": first if start else None, "end_seq": last if end else None,
                                 "label": result.get("label"), "candidates": len(candidates),
                                 "queries": query_event.get("queries"), "fields": query_event.get("fields"),
                                 "referee": matches, "cost": cost,
                                 "source_errors": dict(failures), "related_errors": dict(related_failures),
                                 "search_failures": [{"query": s["query"], "errors": s["errors"]} for s in searches if s["errors"]],
                                 "related_failures": [{"paper": r["paper"], "errors": r["errors"]} for r in related if r["errors"]],
                                 "listings": [{"kind": listing["kind"], "call_id": listing["call_id"], "shown_count": len(listing["keys"])} for listing in listings]})
                if end:
                    previous_end = ts(end["ts"])
            artifacts = sorted((run / "artifacts/prior_art").glob(f"{cid}-*.json"))
            missing = {file.name: [k for k in ("query_fields", "round2_queries", "coverage", "listing")
                                   if k not in json.loads(file.read_text())] for file in artifacts}
            verdicts = [e["data"] for e in own if e["type"] == "verdict" and e["data"].get("stage") == "S2"]
            out["cards"][cid].append({"run": run.name, "attempts": attempts, "artifact_missing_fields": missing,
                                       "total_model_calls": len(own_calls),
                                       "total_model_request_attempts": sum(a.get("kind") == "request" for c in own_calls for a in c["attempts"]),
                                       "s2": [{"verdict": v["verdict"], "reasons": v["reasons"]} for v in verdicts],
                                       "outcome": [e["data"] for e in own if e["type"] == "outcome_recorded"],
                                       "stopped_alive_at_s2": any(e["type"] == "live_check_stopped" for e in own)})
    retrieved = [cid for cid in INCREMENTAL if any(work["retrieved"] for card in out["cards"].get(cid, [])
                                                 for attempt in card["attempts"] for work in attempt["referee"].values())]
    guard = [cid for cid in PROBES if any(card["s2"] and registered_guard(card["s2"][-1])
                                        for card in out["cards"].get(cid, []))]
    out["criteria"] = {"retrieval": {"status": "PASS" if len(retrieved) >= 3 else "FAIL", "found_cards": retrieved,
                                     "score": len(retrieved), "required": 3, "total": 4,
                                     "interpretation": "Generous any registered work per card, union across saved attempts; no fuzzy title matches."},
                       "published_guard": {"status": "PASS" if len(guard) == 4 else "FAIL", "score": len(guard), "required": 4}}
    out["missing_data"].append("Historical run snapshots differ from the repaired tree; final-tree live experiment and combined full suite NOT TESTED.")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    out = extract(args.root)
    (args.root / "live.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["criteria"], indent=2))


if __name__ == "__main__":
    main()
