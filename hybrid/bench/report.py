"""Markdown tables from a bench_result dict."""
from __future__ import annotations


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return ("%%.%df" % digits) % value
    return str(value)


def _size_metric(entry, task, n, key):
    sizes = entry.get("tasks", {}).get(task, {}).get("sizes", {})
    m = sizes.get(str(n))
    if not m or not m.get("valid"):
        return None
    return m.get(key)


def _group(entry, task, n, label, key):
    sizes = entry.get("tasks", {}).get(task, {}).get("sizes", {})
    m = sizes.get(str(n))
    if not m or not m.get("valid"):
        return None
    return m["groups"][str(label)][key]


def render(result):
    tasks, entrants = result["tasks"], result["entrants"]
    names = list(entrants)
    lines = []
    lines.append("### Largest train size: KL to oracle (nats) / top-1 accuracy\n")
    header = ["entrant"] + ["%s (n=%d)" % (t, tasks[t]["train_sizes"][-1]) for t in tasks]
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for name in names:
        e, row = entrants[name], [name]
        for t in tasks:
            n = tasks[t]["train_sizes"][-1]
            row.append("%s / %s" % (_fmt(_size_metric(e, t, n, "kl")), _fmt(_size_metric(e, t, n, "acc"))))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for t, task in tasks.items():
        if task["group_name"] == "all":
            continue
        n = task["train_sizes"][-1]
        lines.append("### %s accuracy by %s (n=%d)\n" % (t, task["group_name"], n))
        header = ["entrant"] + ["%s=%s" % (task["group_name"], g) for g in task["group_labels"]]
        lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
        for name in names:
            lines.append("| " + " | ".join([name] + [_fmt(_group(entrants[name], t, n, g, "acc"))
                                                     for g in task["group_labels"]]) + " |")
        lines.append("")
    if "compose" in tasks:
        n = tasks["compose"]["train_sizes"][-1]
        lines.append("### compose close-bracket accuracy (n=%d)\n" % n)
        lines += ["| entrant | close_acc |", "| --- | --- |"]
        for name in names:
            lines.append("| %s | %s |" % (name, _fmt(_size_metric(entrants[name], "compose", n, "close_acc"))))
        lines.append("")
    lines.append("### Sample efficiency: KL (acc) at each train size\n")
    for t, task in tasks.items():
        header = ["%s entrant" % t] + ["n=%d (%d tok)" % (n, task["train_tokens"][str(n)]) for n in task["train_sizes"]]
        target = task.get("target")
        if target:
            header.append("tokens to %s %s %s" % (target["metric"], target["op"], target["value"]))
        lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
        for name in names:
            e = entrants[name]
            row = [name] + ["%s (%s)" % (_fmt(_size_metric(e, t, n, "kl")), _fmt(_size_metric(e, t, n, "acc")))
                            for n in task["train_sizes"]]
            if target:
                ttt = e.get("tasks", {}).get(t, {}).get("tokens_to_target")
                row.append("not reached" if ttt and not ttt["reached"] else
                           _fmt(ttt["interp_tokens"], 0) if ttt else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.append("### Scaling of predict_proba with context length\n")
    lines += ["| entrant | params | tail slope (time) | full slope | claimed | verdict | median s at max n | peak RSS delta MB at max n | stopped |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        e = entrants[name]
        s = e.get("scaling") or {}
        if "error" in s or not s:
            lines.append("| %s | | error: %s | | | | | | |" % (name, s.get("error", "no data")))
            continue
        params = (e.get("describe") or {})
        first = next(iter(next(iter(params.values()), {}).values()), {}) if params else {}
        peak = s["peak_rss_delta_bytes"][-1] if s["peak_rss_delta_bytes"] else None
        lines.append("| %s | %s | %s | %s | %s | %s | %s (n=%s) | %s | %s |" % (
            name, first.get("params", "n/a"), _fmt((s.get("time_fit_tail") or {}).get("slope"), 2),
            _fmt((s.get("time_fit_all") or {}).get("slope"), 2), _fmt(s.get("claimed_exponent"), 1), s["verdict"],
            _fmt(s["median_s"][-1], 4), s["lengths"][-1], _fmt(None if peak is None else peak / 2 ** 20, 1),
            s.get("stopped") or "no"))
    lines.append("")
    lines += render_fingerprint(result)
    return "\n".join(lines)


def _fp(entry):
    return entry.get("fingerprint")


def render_fingerprint(result):
    """Tables for the mechanism fingerprint (empty when no entrant has one)."""
    entrants = {n: e for n, e in result["entrants"].items() if _fp(e)}
    if not entrants:
        return []
    lines = ["### Mechanism fingerprint: capacity, scaling, few-shot slope, nearest baseline, flags\n",
             "| entrant | recall status | K at half accuracy | state floats | floats per pair | time slope (tail) | "
             "memory growth | few-shot slope per doubling | nearest baseline (mean ratio / mean JS) | flags |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, e in entrants.items():
        fp = _fp(e)
        cap, sc, few, near = fp["recall_capacity"], fp["scaling"], fp["fewshot"], fp["nearest_baseline"]
        k_half = cap.get("k_half")
        k_text = "n/a" if cap.get("status") == "not_learned" else (
            (">= %s" % cap["pairs"]) if cap.get("pairs_is_lower_bound") else _fmt(k_half, 1))
        near_text = "n/a" if not near else "%s (%s / %s)" % (near["name"], _fmt(near["mean_ratio"], 2),
                                                           _fmt(near["mean_js"], 4))
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, cap.get("status"), k_text, _fmt(cap.get("state_floats")), _fmt(cap.get("state_floats_per_pair"), 1),
            _fmt(sc.get("time_slope_tail"), 2), sc.get("memory_growth", "n/a"), _fmt(few.get("slope_per_doubling"), 3),
            near_text, ", ".join(fp["flags"]) or "none"))
    lines.append("")
    lines += ["#### Length generalization: KL (change from the training-length test set) at 2x and 4x the training length\n"]
    tasks = [t for t, spec in result["tasks"].items() if spec.get("length_probes")]
    factors = sorted({f for t in tasks for f in result["tasks"][t]["length_probes"]}, key=int)
    header = ["entrant"] + ["%s x%s" % (t, f) for t in tasks for f in factors]
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for name, e in entrants.items():
        row = [name]
        for t in tasks:
            block = _fp(e)["length_generalization"].get(t, {})
            for f in factors:
                cell = block.get("factors", {}).get(f)
                row.append("n/a" if not cell or not cell.get("valid") else
                           "%s (%+.3f)" % (_fmt(cell["kl"]), cell["kl_delta"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    baselines = [n for n, e in result["entrants"].items() if e["role"] == "baseline" and _fp(e)]
    lines += ["#### Behavioral similarity: mean ratio / mean JS (nats) / tasks close, to each baseline at the largest train size\n"]
    header = ["entrant"] + baselines
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for name, e in entrants.items():
        sim = _fp(e)["similarity"]
        lines.append("| " + " | ".join([name] + [
            "self" if b == name else ("n/a" if b not in sim else "%s / %s / %d of %d" % (
                _fmt(sim[b]["mean_ratio"], 2), _fmt(sim[b]["mean_js"], 4), sim[b]["close_tasks"], sim[b]["compared_tasks"]))
            for b in baselines]) + " |")
    lines.append("")
    return lines
