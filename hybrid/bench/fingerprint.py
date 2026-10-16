"""Mechanism fingerprint: what an entrant behaves like, measured by trusted code. NumPy only.

Runs in the evaluator job, after every entrant's metrics exist. Nothing here reads an entrant's
description of itself except describe()["state_floats_per_sequence"], which the recall capacity
comparison needs and which is reported next to the measured curve. Every other number comes from
the entrant's prediction arrays, the oracle, and the scaling probe the runner wrote.

Per entrant (entrants[name]["fingerprint"]):
- recall_capacity: accuracy by K at the largest train size, the K at which accuracy falls to half
  its value at the smallest K (log-K interpolation), and that capacity per reported state float.
- scaling: time exponent from the probe, and whether peak memory grows with context length.
- length_generalization: KL and accuracy on hmm, vom and compose sequences 2x and 4x longer than
  the training sequences, and the change from the training-length test set.
- fewshot: accuracy by number of in-context examples and the slope per doubling.
- similarity: to every baseline (other than itself), per task, mean Jensen-Shannon divergence of the
  predictive distributions on the same test prefixes, mean KL, and an oracle-normalized distance.
- nearest_baseline and flags.

Why the normalized distance: two models that are both close to the oracle are close to each other
without sharing a mechanism. Writing d(a, b) = sqrt(mean JS(a, b)), which satisfies the triangle
inequality, ratio = d(a, b) / (d(a, oracle) + d(b, oracle)) lies in [0, 1]. It is near 0 when the two
models make the same errors and near 1 when their errors point in unrelated directions. A task counts
as "same behavior" only when both the raw JS and this ratio are under their thresholds.
"""
from __future__ import annotations

import math

import numpy as np

from . import metrics

EPS = 1e-9


def oracle_matrix(kind, oracle, vocab):
    """Oracle as a (N, V) distribution: itself for distribution tasks, one-hot for answer tasks."""
    return oracle if kind == "dist" else np.eye(vocab)[oracle]


def _kl_to_mixture(a, m):
    safe_a = np.where(a > 0, a, 1.0)
    return np.where(a > 0, a * (np.log(safe_a) - np.log(m)), 0.0).sum(1)


def js_rows(p, q):
    """Jensen-Shannon divergence in nats per row (between 0 and ln 2)."""
    m = 0.5 * (p + q)
    m = np.where(m > 0, m, 1.0)   # a zero mixture entry means both are zero there, and contributes nothing
    return 0.5 * _kl_to_mixture(p, m) + 0.5 * _kl_to_mixture(q, m)


def half_capacity_k(ks, accs):
    """Smallest K at which accuracy has fallen to half its value at ks[0], by interpolation in log K
    between the bracketing points; None if it never does within the measured range."""
    half = accs[0] / 2
    for i in range(1, len(ks)):
        if accs[i] <= half:
            above, below = accs[i - 1], accs[i]
            frac = (above - half) / (above - below)
            return math.exp(math.log(ks[i - 1]) + frac * (math.log(ks[i]) - math.log(ks[i - 1])))
    return None


def _largest_metrics(task, block):
    return block["sizes"][str(task["train_sizes"][-1])]


def _state_floats(entry, task_name, n):
    try:
        value = entry["describe"][task_name][str(n)]["state_floats_per_sequence"]
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def recall_capacity(spec, entry, cfg):
    task = spec["tasks"].get("recall")
    if task is None or "tasks" not in entry:
        return {"status": "no_recall_task"}
    n = task["train_sizes"][-1]
    m = _largest_metrics(task, entry["tasks"]["recall"])
    if not m["valid"]:
        return {"status": "invalid_predictions"}
    ks = [int(k) for k in task["group_labels"]]
    accs = [m["groups"][str(k)]["acc"] for k in ks]
    out = {"train_size": n, "k": ks, "acc": accs, "chance": task["chance_acc"], "n_per_k": m["groups"][str(ks[0])]["n"]}
    state = _state_floats(entry, "recall", n)
    if state is None:
        return {**out, "status": "no_state_floats"}
    out["state_floats"] = state
    learned = accs[0] >= cfg["learned_ratio"] * task["chance_acc"]
    k_half = half_capacity_k(ks, accs) if learned else None
    if not learned:
        pairs, out["status"] = 0.0, "not_learned"
    elif k_half is None:
        pairs, out["status"] = float(ks[-1]), "no_knee_in_range"   # capacity is at least the largest K measured
    else:
        pairs, out["status"] = k_half, "knee"
    out.update(k_half=k_half, pairs=pairs, pairs_is_lower_bound=out["status"] == "no_knee_in_range")
    out["capacity_per_state_float"] = pairs / max(state, 1)
    out["state_floats_per_pair"] = None if pairs == 0 else max(state, 1) / pairs
    out["value_bits_per_state_float"] = pairs * task["value_bits"] / max(state, 1)
    return out


def scaling_block(entry, cfg):
    s = entry.get("scaling") or {}
    if "error" in s or not s:
        return {"status": "unavailable"}
    tail, full, mem = s.get("time_fit_tail"), s.get("time_fit_all"), s.get("memory_fit_tail")
    peaks = [p for p in s["peak_rss_delta_bytes"] if p is not None]
    peak_mb = peaks[-1] / 2 ** 20 if peaks else None
    out = {"status": "ok", "lengths": [s["lengths"][0], s["lengths"][-1]], "stopped": s.get("stopped"),
           "time_slope_tail": None if tail is None else tail["slope"],
           "time_slope_all": None if full is None else full["slope"],
           "time_r2_tail": None if tail is None else tail["r2"],
           "claimed_exponent": s.get("claimed_exponent"), "verdict": s["verdict"],
           "memory_slope_tail": None if mem is None else mem["slope"], "peak_rss_delta_mb_at_max_n": peak_mb}
    if mem is None or peak_mb is None:
        out["memory_growth"] = "unknown"
    else:
        flat = mem["slope"] < cfg["memory_flat_slope"] or peak_mb < cfg["memory_floor_mb"]
        out["memory_growth"] = "flat" if flat else "growing"
    return out


def length_generalization(spec, entry):
    out = {}
    for name, task in spec["tasks"].items():
        if not task.get("length_probes") or "tasks" not in entry:
            continue
        block = entry["tasks"][name]
        base = _largest_metrics(task, block)
        if not base["valid"]:
            out[name] = {"status": "invalid_predictions"}
            continue
        row = {"status": "ok", "base": {"kl": base["kl"], "acc": base["acc"]}, "factors": {}}
        for factor, info in task["length_probes"].items():
            probe = block.get("length_probes", {}).get(factor)
            if not probe or not probe["valid"]:
                row["factors"][factor] = {"valid": False}
                row["status"] = "incomplete"
                continue
            row["factors"][factor] = {
                "valid": True, "length": info["length"], "kl": probe["kl"], "acc": probe["acc"],
                "kl_delta": probe["kl"] - base["kl"], "acc_delta": probe["acc"] - base["acc"],
                "kl_ratio": probe["kl"] / base["kl"] if base["kl"] > 0 else None}
        out[name] = row
    return out


def fewshot_block(spec, entry):
    task = spec["tasks"].get("fewshot")
    if task is None or "tasks" not in entry:
        return {"status": "no_fewshot_task"}
    m = _largest_metrics(task, entry["tasks"]["fewshot"])
    if not m["valid"]:
        return {"status": "invalid_predictions"}
    shots = [int(s) for s in task["group_labels"]]
    accs = [m["groups"][str(s)]["acc"] for s in shots]
    x = np.log2(np.asarray(shots, dtype=np.float64))
    y = np.asarray(accs)
    slope = float(np.polyfit(x, y, 1)[0])
    # Standard error from the binomial noise of each accuracy: slope = sum w_i y_i with w_i = (x_i - mean x) / Sxx.
    counts = np.asarray([m["groups"][str(s)]["n"] for s in shots], dtype=np.float64)
    weights = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    se = float(math.sqrt((weights ** 2 * y * (1 - y) / counts).sum()))
    return {"status": "ok", "train_size": task["train_sizes"][-1], "shots": shots, "acc": accs,
            "chance": task["chance_acc"], "slope_per_doubling": slope, "slope_se": se}


def pair_similarity(dist_to_oracle, a, b, d_b, cfg):
    """a, b: (N, V) predictions; dist_to_oracle: sqrt(mean JS(a, oracle)); d_b the same for b."""
    js = js_rows(a, b)
    mean_js = float(js.mean())
    denom = dist_to_oracle + d_b
    ratio = math.sqrt(mean_js) / denom if denom > EPS else None
    close = ratio is not None and mean_js <= cfg["js_close"] and ratio <= cfg["ratio_close"]
    return {"js": mean_js, "kl": float(metrics.kl_rows(a, b).mean()), "ratio": ratio, "close": bool(close)}


def similarity_block(spec, name, preds, oracles, others, cfg):
    """preds: {entrant: {task: (N, V) array}}; others: names to compare with."""
    out = {}
    for other in others:
        per_task = {}
        for task_name, task in spec["tasks"].items():
            a, b = preds[name].get(task_name), preds[other].get(task_name)
            if a is None or b is None:
                continue
            o = oracle_matrix(task["oracle_kind"], oracles[task_name]["oracle"], task["vocab_size"])
            d_a, d_b = math.sqrt(float(js_rows(a, o).mean())), math.sqrt(float(js_rows(b, o).mean()))
            per_task[task_name] = pair_similarity(d_a, a, b, d_b, cfg)
        if not per_task:
            continue
        ratios = [t["ratio"] for t in per_task.values() if t["ratio"] is not None]
        out[other] = {"per_task": per_task, "mean_js": float(np.mean([t["js"] for t in per_task.values()])),
                      "mean_ratio": float(np.mean(ratios)) if ratios else None,
                      "close_tasks": sum(t["close"] for t in per_task.values()), "compared_tasks": len(per_task)}
    return out


def nearest_baseline(similarity):
    if not similarity:
        return None
    with_ratio = {b: s["mean_ratio"] for b, s in similarity.items() if s["mean_ratio"] is not None}
    by_js = min(similarity, key=lambda b: similarity[b]["mean_js"])
    name = min(with_ratio, key=with_ratio.get) if with_ratio else by_js
    tasks = {t: min(similarity, key=lambda b: similarity[b]["per_task"][t]["js"])
             for t in next(iter(similarity.values()))["per_task"]
             if all(t in s["per_task"] for s in similarity.values())}
    return {"name": name, "by": "mean_ratio" if with_ratio else "mean_js",
            "mean_ratio": similarity[name]["mean_ratio"], "mean_js": similarity[name]["mean_js"],
            "by_mean_js": by_js, "per_task_by_js": tasks}


def flags_for(name, fp, spec, all_fp, cfg):
    """Returns (flags, details). all_fp: every entrant's fingerprint so far (capacity needed for the baselines)."""
    flags, details = [], {}
    for baseline, sim in fp["similarity"].items():
        if sim["compared_tasks"] and sim["close_tasks"] / sim["compared_tasks"] > cfg["close_task_fraction"]:
            flags.append("behaves_like:%s" % baseline)
            details["behaves_like:%s" % baseline] = {"close_tasks": sim["close_tasks"],
                                                     "compared_tasks": sim["compared_tasks"],
                                                     "mean_js": sim["mean_js"], "mean_ratio": sim["mean_ratio"]}
    cap = fp["recall_capacity"]
    if cap.get("status") in ("knee", "no_knee_in_range", "not_learned"):
        fixed = {b: all_fp[b]["recall_capacity"]["capacity_per_state_float"] for b, e in spec["entrants"].items()
                 if b != name and e["role"] == "baseline" and e.get("state") == "fixed" and b in all_fp
                 and all_fp[b]["recall_capacity"].get("status") in ("knee", "no_knee_in_range", "not_learned")}
        if fixed:
            best = max(fixed, key=fixed.get)
            details["capacity"] = {"capacity_per_state_float": cap["capacity_per_state_float"], "status": cap["status"],
                                   "best_fixed_baseline": best, "best_fixed_capacity_per_state_float": fixed[best],
                                   "ratio_threshold": cfg["capacity_advantage_ratio"]}
            if cap["capacity_per_state_float"] <= cfg["capacity_advantage_ratio"] * fixed[best]:
                flags.append("no_capacity_advantage")
    slope = fp["scaling"].get("time_slope_tail")
    if slope is not None and slope > cfg["quadratic_slope"]:
        flags.append("quadratic_or_worse")
        details["quadratic_or_worse"] = {"time_slope_tail": slope, "threshold": cfg["quadratic_slope"]}
    gain = fp["fewshot"].get("slope_per_doubling")
    if gain is not None and gain < cfg["fewshot_slope_min"]:
        flags.append("no_fewshot_gain")
        details["no_fewshot_gain"] = {"slope_per_doubling": gain, "threshold": cfg["fewshot_slope_min"]}
    return flags, details


def attach(spec, entrants, preds, oracles):
    """Adds entrants[name]["fingerprint"] for every entrant with metrics. Returns the result's top-level block."""
    cfg = spec["fingerprint"]
    names = [n for n, e in entrants.items() if e["status"] == "ok" and "tasks" in e]
    baselines = [n for n in names if spec["entrants"][n]["role"] == "baseline"]
    all_fp = {}
    for name in names:
        entry = entrants[name]
        all_fp[name] = {"recall_capacity": recall_capacity(spec, entry, cfg), "scaling": scaling_block(entry, cfg),
                        "length_generalization": length_generalization(spec, entry),
                        "fewshot": fewshot_block(spec, entry),
                        "similarity": similarity_block(spec, name, preds, oracles, [b for b in baselines if b != name],
                                                       cfg)}
        all_fp[name]["nearest_baseline"] = nearest_baseline(all_fp[name]["similarity"])
    for name in names:
        flags, details = flags_for(name, all_fp[name], spec, all_fp, cfg)
        all_fp[name]["flags"], all_fp[name]["flag_details"] = flags, details
        entrants[name]["fingerprint"] = all_fp[name]
    block = {"config": cfg}
    for name in names:
        if spec["entrants"][name]["role"] == "candidate":
            block["candidate"] = {"nearest_baseline": all_fp[name]["nearest_baseline"], "flags": all_fp[name]["flags"]}
    return block


__all__ = ["attach", "js_rows", "half_capacity_k", "oracle_matrix", "recall_capacity", "scaling_block",
           "length_generalization", "fewshot_block", "similarity_block", "nearest_baseline", "flags_for"]
