"""Summarise the PA2 live check: per card, searches and wall time of the literature step, whether the referee's closest
works are among the stored candidates, and the S2 outcome. Reads results-wt/kd8/runs/*; writes results-wt/kd8/live.json."""
import glob, json, re, sys
from datetime import datetime
from pathlib import Path

ROOT = Path("results-wt/kd8")
REFEREE = {  # card -> {label: regex over candidate titles}
    "card-3485a13f": {"Hwang 1110.5667 Bayesian program merging": r"program merging|inducing probabilistic programs",
                      "Srinivas 2204.09837 memory bounds experts": r"memory bounds for the experts",
                      "Silver 1904.06317 logical program policies": r"logical program policies"},
    "card-f5ee7edb": {"Rendle factorization machines": r"factorization machines",
                      "Rossi 1906.05059 higher-order ranking": r"higher-order ranking and link prediction"},
    "card-a4f684fb": {"SciPy/2-opt graph matching (QAP 2-opt)": r"2-opt|two-opt|quadratic assignment.*(swap|local)|graph matching.*swap",
                      "Rehn permutation group algorithms/base change": r"permutation group algorithms|base change"},
    "card-89248ee5": {"Jiang 1409.0334 tournament sequences": r"binary tournament|storing sequences",
                      "Zhou 0901.0553 local information link prediction": r"predicting missing links via local information",
                      "Sacramento 1207.7196 Willshaw": r"willshaw",
                      "Stoklosa 2411.14480 associative knowledge graphs": r"associative knowledge graphs"},
    "card-8154d3b4": {"Biamonte 1405.7375 tensor network #SAT": r"tensor network contractions for #?sat",
                      "Stoudenmire 1605.05775": r"quantum-inspired tensor networks",
                      "Yu 1711.00073 HOT-RNN": r"higher-order tensor rnn"},
}
def ts(s): return datetime.fromisoformat(s)
out = {}
for run in sorted((ROOT / "runs").iterdir()):
    events = [json.loads(l) for l in open(run / "ledger.jsonl")]
    cards = sorted({e["data"].get("card_id") for e in events if e["data"].get("card_id")})
    for pa in sorted(glob.glob(str(run / "artifacts/prior_art/*.json"))):
        d = json.load(open(pa))
        cid = re.search(r"(card-[0-9a-f]+)", pa).group(1)
        titles = [c["title"] for c in d["candidates"]]
        rank = {c["title"]: i + 1 for i, c in enumerate(d["candidates"])}
        found = {}
        for label, rx in REFEREE.get(cid, {}).items():
            hit = [t for t in titles if re.search(rx, t, re.I)]
            found[label] = {"retrieved": bool(hit), "rank": rank[hit[0]] if hit else None, "title": hit[0] if hit else None}
        rec = out.setdefault(cid, {})
        rec.update({"run": run.name, "label": d["label"], "queries": len(d["queries"]), "fields": sorted(set(d.get("query_fields", []))),
                    "round2_queries": d.get("round2_queries", []), "candidates": len(d["candidates"]), "referee": found,
                    "evidence_titles": [e["title"] for e in d["evidence"]][:6]})
    for cid in cards:
        rec = out.setdefault(cid, {})
        # literature events of this run carry no card id; every run here holds the cards one after another, so the
        # per-card cost is read from the span between its prior_art_queries and prior_art_result events.
        starts = [i for i, e in enumerate(events) if e["type"] == "prior_art_queries"]
        results = [i for i, e in enumerate(events) if e["type"] == "prior_art_result"]
        costs = []
        for i, j in zip(starts, results):
            span = events[i:j + 1]
            costs.append({"searches": sum(e["type"] == "literature_search" for e in span),
                          "related_lookups": sum(e["type"] == "literature_related" for e in span),
                          "source_requests_receipts": sum(e["type"] == "literature_receipt" for e in span),
                          "wall_s": round((ts(events[j]["ts"]) - ts(events[i]["ts"])).total_seconds())})
        rec["literature_cost"] = costs
        rec["outcome"] = [{"kill_codes": e["data"].get("kill_codes"), "outcome": e["data"].get("outcome")} for e in events
                          if e["type"] == "outcome_recorded" and e["data"].get("card_id") == cid]
        rec["stopped_alive_at_s2"] = any(e["type"] == "live_check_stopped" and e["data"].get("card_id") == cid for e in events)
json.dump(out, open(ROOT / "live.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
