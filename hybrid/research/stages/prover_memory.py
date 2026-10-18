"""What stage M's prover keeps of its own learnability attempts across rounds (FIX-5).

In the 2026-10-13 smokes card k1's P4.composition learner passed on all three seeds with one recipe
(runs/calibration-smoke-20261013T170308Z, attempt 3), and the attempts of the next smoke changed initialization,
learning rate, decay, batch size and epochs together and never came near it again; each round showed the prover only
its latest source, cut to 16000 characters, and a compact history line per attempt
(results/phase4-20261013T073748Z/p4comp-diag/REPORT.md). ``BestAttempts`` therefore keeps, per learnability statement,
the best measured attempt of this run: its full source, its recipe, and per seed the accuracy, both control
accuracies, the wall time of each training and the learner's own training curve. The prover reads it in every later
round (``prompt_block``).

A recipe that passed on every seed in an earlier round of the same run, while the statement still does not hold (the
replay disagreed, the architecture survey failed, or a later stage reopened it), is kept unless the prover says why it
must change: a new source whose RECIPE differs from the passing one and whose reply gives no
``recipe_change_reason`` is refused before any job (``discarded_without_reason``). Nothing here changes what a check
decides: the threshold, m, the seeds and both controls are the frozen statement's.

P2 constructions (FIX-6). In the 2026-10-14 calibration (card k1, runs/calibration-math-20261014T053341Z/k1-run1)
the prover's four composition constructions measured 0.27, 0, 0.27 and 0.53 at T=4, and each round showed it only the
latest source; it argued that three rounds were needed and ended on a most-frequent-output witness, while a two-round
construction within the same frozen contract had passed in an earlier run
(results/phase4-20261013T073748Z/k1-calib-diag/REPORT.md, section 2). ``BestAttempts`` also keeps, per construction
statement, the best measured attempt of this run (ranked by holding, then the lowest accuracy over the listed sizes,
then the mean) with its per-size accuracy and state and its full source; ``frozen_contract`` gives the resources the
check allows at each size. Only this run's own attempts are kept: nothing is carried from another run.

Ranking against a trivial baseline (FIX-7). That ranking chose k1-run1's attempt 4, a most-frequent-output witness
(0.52625, 0.19344, 0.10984 at T = 4, 16, 32), because its lowest accuracy was the highest of the four. A witness that
ignores the query can score that much. ``query_independent_baseline`` therefore computes, per listed size and on the
very instances the check runs (the statement's count, seed and params), the accuracy of the best output that does not
depend on the query: for each instance, the one answer that is right most often among its queries (the instance's
modal label), scored with the task's metric. No query-independent witness can exceed it, so an attempt that does not
beat it at some size is a shortcut, however high its accuracy. Construction attempts are ranked by holding, then by
the number of sizes where they beat the baseline, then by the lowest and the mean excess over it. An attempt that
beats the baseline at no size is never presented as ``best_attempt``; the prover sees ``attempts_vs_baseline``
instead: the top two attempts with per-size accuracy, the baseline and the excess, each marked with whether it beat
the baseline anywhere. On k1-run1's composition attempts the baseline is 0.64, 0.27 and 0.16, every attempt is at or
below it, and no best attempt is shown.
"""
from __future__ import annotations

import ast
import json
from collections import Counter

from hybrid.math import architecture, constructions, learnability, spec_checks

# Curve points per training shown to the prover for the best attempt (the check record keeps up to CURVE_SHOWN).
PROMPT_CURVE_POINTS = 12


def recipe_node(source: str | None) -> str | None:
    """The module-level RECIPE dict of a learner source as a normalized AST dump (formatting and comments ignored,
    0.002 and 2e-3 equal), or None when the source has none or does not parse."""
    if not source:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == learnability.RECIPE_NAME for t in targets):
            return ast.dump(node.value, annotate_fields=False) if node.value is not None else None
    return None


def rows_of(check: dict) -> list[dict]:
    return list((check.get("details") or {}).get("results") or [])


def passed_every_seed(check: dict, seeds: int) -> bool:
    """Every listed seed was measured and passed: accuracy at the threshold and a control below it."""
    rows = rows_of(check)
    return len(rows) == seeds and all(r.get("passed") for r in rows)


def score(check: dict, seeds: int) -> tuple | None:
    """How good a learnability attempt was, for choosing the best: passed on every seed first, then the lowest
    seed accuracy (an unmeasured seed counts 0), then the mean. None for an attempt that measured no seed."""
    rows = rows_of(check)
    if not rows:
        return None
    accuracies = [float(r.get("accuracy") or 0.0) for r in rows] + [0.0] * max(0, seeds - len(rows))
    return passed_every_seed(check, seeds), min(accuracies), sum(accuracies) / len(accuracies)


def _curve(curve: dict | None) -> dict | None:
    if not curve:
        return None
    points = curve.get("points") or []
    if len(points) > PROMPT_CURVE_POINTS:
        step = (len(points) - 1) / (PROMPT_CURVE_POINTS - 1)
        points = [points[round(i * step)] for i in range(PROMPT_CURVE_POINTS)]
    shown = {"points": points, "count": curve.get("count", len(points))}
    if curve.get("problem"):
        shown["problem"] = curve["problem"]
    return shown


def training_times(check: dict, key: str = "train_s") -> dict:
    """Per seed, the time of each training of a learnability check (kept apart from its results rows): ``train_s``
    the wall seconds, ``train_cpu_s`` the process CPU seconds the budget counts (decision CT; absent in checks made
    before it)."""
    clock = (check.get("details") or {}).get(learnability.WALL_CLOCK_KEY) or {}
    return {t.get("seed"): {k: v for k, v in t.items() if k != "seed"} for t in clock.get(key) or []}


def seed_line(row: dict, train_s: dict | None = None, train_cpu_s: dict | None = None) -> dict:
    """One seed of an attempt as the prover reads it, with that seed's training times (CPU, which the budget counts,
    and wall)."""
    curves = row.get("curves") or {}
    return {"seed": row.get("seed"), "accuracy": row.get("accuracy"), "passed": row.get("passed"),
            "empty_control_accuracy": row.get("empty_control_accuracy"),
            "wrong_control_accuracy": row.get("wrong_control_accuracy"), "reason": row.get("reason"),
            "train_cpu_s": train_cpu_s, "train_s": train_s,
            "curve_true_data": _curve(curves.get("true")),
            "curve_empty_control": _curve(curves.get("empty")),
            "curve_wrong_control": _curve(curves.get("wrong"))}


def query_independent_baseline(statement: dict) -> dict[int, float]:
    """Per listed size, the best accuracy an output that ignores the query reaches on the instances the check runs
    (module docstring, Ranking against a trivial baseline): per instance, the answer right most often among its
    queries. Per-query tasks score right answers over all queries; per-instance tasks score instances whose answers
    are all one label."""
    spec = statement["spec"]
    task = spec["task"]
    count = spec.get("count") or constructions.DEFAULT_COUNT
    baseline = {}
    for size in spec["sizes"]:
        instances = constructions.generate(task, size, count, spec.get("seed") or 0, spec.get("params"))
        if constructions.METRIC[task] == "instance":
            baseline[size] = sum(len({answer for _, answer in inst.queries}) == 1 for inst in instances) / count
        else:
            right = sum(Counter(answer for _, answer in inst.queries).most_common(1)[0][1] for inst in instances)
            baseline[size] = right / sum(len(inst.queries) for inst in instances)
    return baseline


def construction_score(check: dict, baseline: dict[int, float]) -> tuple | None:
    """How good a construction attempt was: holding first, then the number of listed sizes where it beat the
    query-independent baseline, then the lowest and the mean excess over the baseline (an unmeasured size counts 0
    accuracy). None for an attempt that measured no size."""
    rows = {r["size"]: float(r.get("accuracy") or 0.0) for r in rows_of(check)}
    if not rows:
        return None
    excess = [rows.get(size, 0.0) - base for size, base in baseline.items()]
    held = bool(check.get("decided")) and check.get("status") in ("pass", "discharged")
    return held, sum(e > 0 for e in excess), min(excess), sum(excess) / len(excess)


def beats_baseline(score: tuple) -> bool:
    """A construction attempt that held, or beat the query-independent baseline at some listed size."""
    return score[0] or score[1] > 0


def size_line(row: dict) -> dict:
    """One size of a construction attempt as the prover reads it."""
    line = {key: row.get(key) for key in ("size", "accuracy", "passed", "state_peak_words", "declared_state_words")}
    # FIX-8: why a size failed although its accuracy met the threshold, and how many carried configurations it used.
    line.update({key: row[key] for key in ("carried_states", "resource_violations") if row.get(key)})
    return line


def frozen_contract(statement: dict, shared: dict | None) -> dict:
    """The resources a construction's check allows at each listed size, evaluated from the frozen statement and the
    profile's architecture contract (FIX-4): rounds, d, the feed-forward width H, positions (the declared formula, or
    the longest token stream the generator makes there), the vocabulary, the precision and the declared state words.
    The check applies exactly these; the prover reads them as the authority (prompt section construction_contract)."""
    spec = statement["spec"]
    task = spec["task"]
    name = constructions.SIZE_NAME[task]
    widths = spec.get("widths")
    facts = {"task": task, "threshold": spec.get("threshold", constructions.DEFAULT_THRESHOLD),
             "widths": widths, "state_words": spec.get("state_words"), "token_store": spec.get("token_store"),
             "architecture": shared, "sizes": []}
    for size in spec["sizes"]:
        point = {name: size}
        try:
            lay = constructions.layout(task, size, spec.get("params"))
            positions = constructions.stream_lengths(task, size, spec.get("count") or constructions.DEFAULT_COUNT,
                                                     spec.get("seed") or 0, spec.get("params"))[1]
            point.update(vocab=int(lay.get("vocab") or 0), longest_stream=positions)
            if widths is not None and shared is not None:
                values = {name: size, **{k: v for k, v in (spec.get("params") or {}).items()
                                         if isinstance(v, (int, float)) and not isinstance(v, bool)}}
                dims = architecture.dimensions(shared, widths, values, positions=positions, vocab=point["vocab"])
                point.update({key: dims[key] for key in ("rounds", "d", "hidden", "positions")},
                             largest_axis=dims["bound"], precision_bits=shared["precision_bits"])
                bits = architecture.state_bits(shared, {"d": dims["d"]})
                if bits is not None:
                    # FIX-8: the carried-state budget, separate from the float precision.
                    point.update(state_bits=bits, carried_states_allowed=architecture.latent_states(bits))
            if spec.get("state_words") is not None:
                point["state_words"] = spec_checks.state_words_bound(spec, size)
        except ValueError as exc:
            # A statement frozen before the host checked its layout; its check refuses it the same way.
            point["problem"] = str(exc)
        facts["sizes"].append(point)
    return facts


class BestAttempts:
    """Per learnability or construction statement id, the best measured attempt of this run so far, and the
    learnability attempts that passed on every seed."""

    def __init__(self) -> None:
        self.best: dict[str, dict] = {}
        self.passing: dict[str, dict] = {}
        # Construction statements (FIX-7): every measured attempt and the query-independent baseline per size.
        self.constructions: dict[str, list[dict]] = {}
        self.baselines: dict[str, dict[int, float]] = {}

    def note(self, statement: dict, attempt: int, check: dict, source: str | None) -> None:
        """Record a checked attempt. Learnability attempts that measured at least one seed and construction attempts
        that measured at least one size count."""
        if statement["form"] not in ("learnability", "construction") or not source:
            return
        if statement["form"] == "construction":
            sid = statement["id"]
            if not rows_of(check):
                return
            if sid not in self.baselines:
                self.baselines[sid] = query_independent_baseline(statement)
            found = construction_score(check, self.baselines[sid])
            if found is None:
                return
            entry = {"attempt": attempt, "check": check, "source": source, "score": found, "form": "construction"}
            self.constructions.setdefault(sid, []).append(entry)
            current = self.best.get(sid)
            if beats_baseline(found) and (current is None or found > current["score"]):
                self.best[sid] = entry
            return
        seeds = len(statement["spec"]["seeds"])
        found = score(check, seeds)
        if found is None:
            return
        entry = {"attempt": attempt, "check": check, "source": source, "score": found}
        current = self.best.get(statement["id"])
        if current is None or found > current["score"]:
            self.best[statement["id"]] = entry
        if found[0] and statement["id"] not in self.passing:
            self.passing[statement["id"]] = entry

    def prompt_block(self, statement_id: str) -> dict | None:
        """The best attempt as the prover reads it, with its full source; None before any measured attempt."""
        entry = self.best.get(statement_id)
        if entry is None:
            return None
        check = entry["check"]
        if entry.get("form") == "construction":
            violations = ((check.get("details") or {}).get("architecture") or {}).get("violations") or []
            return {"attempt": entry["attempt"], "status": check.get("status"), "decided": check.get("decided"),
                    "held": entry["score"][0], "sizes": [size_line(r) for r in rows_of(check)],
                    **({"architecture_violations": violations[:4]} if violations else {}),
                    "evidence": (check.get("evidence") or "")[:800], "source": entry["source"]}
        times, cpu_times = training_times(check), training_times(check, "train_cpu_s")
        return {"attempt": entry["attempt"], "status": check.get("status"), "decided": check.get("decided"),
                "passed_every_seed": entry["score"][0],
                "recipe": (check.get("details") or {}).get("recipe"),
                "seeds": [seed_line(r, times.get(r.get("seed")), cpu_times.get(r.get("seed")))
                          for r in rows_of(check)],
                "evidence": (check.get("evidence") or "")[:800],
                "source": entry["source"]}

    def ranking_block(self, statement_id: str, top: int = 2) -> dict | None:
        """The top construction attempts of this run against the query-independent baseline, without sources: per
        size the accuracy, the baseline and the excess, and whether the attempt beat the baseline anywhere. None
        before any measured construction attempt."""
        entries = self.constructions.get(statement_id)
        if not entries:
            return None
        baseline = self.baselines[statement_id]
        ranked = sorted(entries, key=lambda e: e["score"], reverse=True)[:top]
        attempts = []
        for entry in ranked:
            rows = {r["size"]: r.get("accuracy") for r in rows_of(entry["check"])}
            attempts.append({"attempt": entry["attempt"], "held": entry["score"][0],
                             "beats_baseline": beats_baseline(entry["score"]),
                             "sizes": [{"size": size, "accuracy": rows.get(size), "baseline": round(base, 6),
                                        "excess": round((rows.get(size) or 0.0) - base, 6)}
                                       for size, base in baseline.items()]})
        return {"baseline": "the best output that ignores the query, per instance, on the instances the check runs",
                "attempts": attempts,
                "best_attempt_shown": statement_id in self.best}

    def latest_is_best(self, statement_id: str, attempt: int | None) -> bool:
        entry = self.best.get(statement_id)
        return entry is not None and entry["attempt"] == attempt

    def discarded_without_reason(self, statement: dict, source: str | None, reason: str | None) -> str | None:
        """Why a new learner is refused before any job: an earlier attempt of this run passed on every seed and this
        source changes its RECIPE without a ``recipe_change_reason``. None when it may run."""
        entry = self.passing.get(statement["id"])
        if entry is None or (reason or "").strip():
            return None
        if recipe_node(source) == recipe_node(entry["source"]):
            return None
        return (f"attempt {entry['attempt']} of this run passed on every seed with RECIPE "
                f"{json.dumps((entry['check'].get('details') or {}).get('recipe'), sort_keys=True)}, and this source "
                "changes that RECIPE without a recipe_change_reason. Send the passing recipe again (its full source "
                "is under best_attempt), changing only what the reported failure requires, or give "
                "recipe_change_reason saying why the recipe itself must change.")
