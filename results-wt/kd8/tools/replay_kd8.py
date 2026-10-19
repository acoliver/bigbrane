"""KD8 replay: take recorded reduction_critic artifacts (no live calls), annotate each counted comparator with the
`components` list the new schema asks for, and run the host rule (judge_comparisons) on the recorded numbers.

The three KD8 records and the P08 Rule 110 record are annotated by hand from the recorded text (what is the card's
own function is read off the recorded update and parameter_mapping). Every other record is annotated mechanically:
one published transition component named by the recorded base form and cited by the recorded reference, which is how
a critic that follows the new prompt declares a named mechanism. Output: results-wt/kd8/replay.json and the fixture
hybrid/tests/research/fixtures/kd8/replay_records.json."""
import glob, json, os, re, sys
from hybrid.research.stages.s2_prior_art import judge_comparisons, component_problem

MAIN = "/Volumes/XS1000/acoliver/projects/bigbrane/results"
HAND = {  # (card id, comparator id) -> components
    ("card-8f2e71cf", "fixed_quadratic_rnn"): [
        {"role": "activation", "name": "H^2/tr(H^2) matrix square, normalized", "origin": "card", "cite": ""},
        {"role": "readout", "name": "d^d det(G)", "origin": "card", "cite": ""},
        {"role": "input_map", "name": "one-hot embedding table", "origin": "card", "cite": ""}],
    ("card-136a1e9b", "elman_rnn_identity_w"): [
        {"role": "activation", "name": "u - sigma E_kappa(u) pulse-equilibrate-withdraw map", "origin": "card", "cite": ""},
        {"role": "readout", "name": "sigma E_kappa(u)", "origin": "card", "cite": ""}],
    ("card-136a1e9b", "rnn_diff_sum_basis"): [
        {"role": "activation", "name": "E_kappa(S,D) closed form in rotated basis", "origin": "card", "cite": ""}],
    ("card-383bd914", "block_sequential_ca_moore"): [
        {"role": "transition", "name": "block-sequential sweep of the rule c + l*r mod 3", "origin": "card", "cite": ""},
        {"role": "readout", "name": "output map z_1", "origin": "card", "cite": ""}],
    ("card-ea50345f", "jeffrey_ipf_step"): [
        {"role": "update", "name": "Jeffrey's rule as one IPF scaling step", "origin": "published",
         "cite": "Jeffrey 1965; Kullback 1959; Csiszar 1975"},
        {"role": "readout", "name": "conditional label marginal given the observed cell", "origin": "published",
         "cite": "Bayesian conditioning"}],
    ("card-17100008", "conv_rnn_lookup_activation"): [
        {"role": "activation", "name": "Rule 110 lookup table", "origin": "published", "cite": "Wolfram 1983; Cook 2004"},
        {"role": "update", "name": "tridiagonal (4,2,1) recurrent matrix", "origin": "published", "cite": "Gilpin 2019"}],
}
SOURCES = {
    "card-8f2e71cf": f"{MAIN}/s2-probe-20261018/runs/s2-P/artifacts/reduction_critic/card-8f2e71cf-v7-43bcd2.json",
    "card-136a1e9b": f"{MAIN}/s2-probe-20261018/runs/s2-P/artifacts/reduction_critic/card-136a1e9b-v7-ca3ca3.json",
    "card-383bd914": f"{MAIN}/s2-probe-20261018/runs/s2-P/artifacts/reduction_critic/card-383bd914-v7-ed1a97.json",
    "card-ea50345f": f"{MAIN}/s2-probe-20261018/runs/s2-P/artifacts/reduction_critic/card-ea50345f-v7-867406.json",
}
for n in range(1, 11):
    cid = f"card-1710000{n:x}" if n > 9 else f"card-1710000{n}"
    cid = "card-17100001"[:-1] + format(n, "x") if n < 10 else "card-1710000a"
    g = glob.glob(f"{MAIN}/novelty-probe-20261017/later/s2-{cid}/artifacts/reduction_critic/*.json")
    if g:
        SOURCES[cid] = g[0]

def annotate(card, comp):
    if (card, comp["id"]) in HAND:
        return HAND[(card, comp["id"])]
    head = re.split(r"[(,;:]| whose | with ", comp["base_form"], maxsplit=1)[0].strip()
    return [{"role": "transition", "name": head, "origin": "published", "cite": comp["base_form"][:200]}]

records = []
for card, path in SOURCES.items():
    d = json.load(open(path))
    comps = []
    for c in d["comparators"]:
        comps.append(dict(c, components=annotate(card, c)))
    recorded = d["reduction_check"]["comparisons"]
    declared_ids = {c["id"] for c in comps}
    comparisons = [{k: r[k] for k in ("declared_id", "comparator", "family", "inputs_tested", "max_abs_diff", "limit_only",
                                      "transformations", "reference")} for r in recorded if r.get("declared_id") in declared_ids]
    old_counted = sorted(r["declared_id"] for r in recorded if r.get("counted") and r.get("within_tolerance"))
    outcome, lines, judged = judge_comparisons(comparisons, comps, d["reduces_to"])
    new_counted = sorted(c["declared_id"] for c in judged if c["counted"] and c["within_tolerance"])
    records.append({"card": card, "source": path.split("/projects/")[1], "reduces_to": d["reduces_to"],
                    "recorded_outcome": d["reduction_outcome"], "recorded_counted": old_counted,
                    "new_outcome": outcome, "new_counted": new_counted,
                    "restated": {c["declared_id"]: component_problem(next(x for x in comps if x["id"] == c["declared_id"]))
                                 for c in judged if c["eligibility"] == "restatement"},
                    "comparators": comps, "comparisons": comparisons})
os.makedirs("hybrid/tests/research/fixtures/kd8", exist_ok=True)
json.dump(records, open("hybrid/tests/research/fixtures/kd8/replay_records.json", "w"), indent=1)
summary = [{k: r[k] for k in ("card", "reduces_to", "recorded_outcome", "recorded_counted", "new_outcome", "new_counted", "restated")} for r in records]
json.dump(summary, open("results-wt/kd8/replay.json", "w"), indent=1)
for s in summary: print(json.dumps(s))
