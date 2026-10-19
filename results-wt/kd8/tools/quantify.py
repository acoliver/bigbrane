"""KD8 census: every counted comparator in families rnn, lstm_gru and other_known across recorded reduction_critic
artifacts, listed with its update/parameter_mapping text so a person can judge slot-filling. Output: JSONL."""
import glob, json, sys
roots = ["/Volumes/XS1000/acoliver/projects/bigbrane/runs", "/Volumes/XS1000/acoliver/projects/bigbrane/results",
         "/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/runs", "/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results",
         "/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt"]
seen = set(); out = []
for r in roots:
    for p in glob.glob(r + "/**/artifacts/reduction_critic/*.json", recursive=True):
        rp = p
        try:
            d = json.load(open(p))
        except Exception:
            continue
        rc = d.get("reduction_check") or {}
        for c in rc.get("comparisons", []):
            if c.get("counted") and c.get("family") in ("rnn", "lstm_gru", "other_known"):
                decl = next((x for x in d.get("comparators", []) if x.get("id") == c.get("declared_id")), {})
                out.append({"path": p.split("/projects/")[1], "outcome": d.get("reduction_outcome"),
                            "family": c["family"], "id": c.get("declared_id"), "comparator": c.get("comparator","")[:160],
                            "base_form": decl.get("base_form", "")[:200], "parameter_mapping": decl.get("parameter_mapping", "")[:400],
                            "update": decl.get("update", "")[:300]})
with open("results-wt/kd8/census.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
print(len(out))
