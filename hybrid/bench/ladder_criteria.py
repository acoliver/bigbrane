"""Calibration acceptance for the test ladder, rung 1 (IMPROVEMENT-PLAN.md B.5), computed from stored run results.

Every criterion returns {"status", "detail", "numbers"}. Status values:
- "pass" / "fail": decided on the three acceptance seeds the protocol requires;
- "incomplete": some required runs (cells or seeds) are missing; numbers so far are reported, nothing is decided;
- "unresolved": every run exists but a model never learned the low-K control, so the property the criterion tests
  was not measured (an optimization result, never a pass and never evidence against the property);
- "info": a part the bench cannot test at these sizes, reported with its numbers.

The criteria read the results at each cell's chosen learning rate (`ladder.chosen_points`). The grid points they
use are fixed here and match configs/ladder.toml; a renamed model or task makes the criterion "incomplete".

Decision B10 (2026-10-16) revised criteria 2, 4, 6 and 9 and fixed the execution policy, both recorded in `policy()`
and stored in every report:
- Acceptance runs are the pinned one-thread Podman CPU container ("trusted": results whose execution mode is
  container on device cpu) on seeds 1, 2 and 3 (seed 0 only picks the learning rate). Host MPS runs are
  diagnostics: `build_report` labels a report that holds any of them "diagnostic" and `gate` refuses it.
- 2: a fixed-state model has to learn the low-K control before its K drop counts; a model that never learns is
  "unresolved", not a pass or a fail on noise. Attention has to learn the same control.
- 4: the window limit is the end-to-end receptive field, layers x window, not one window.
- 6: the empirical copy separation and the finite-state information bound are separate parts. B10t adds the long
  copy task (LONG_COPY_TASK) whose length makes the bound testable.
- 9: meaningful order and same-seed repeatability are separate parts; ties are predeclared.
"""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import numpy as np

from . import ladder
from .baselines import bb_recall

SEEDS = ladder.ACCEPTANCE_SEEDS
FIXED_FAMILIES = ("linear_attention", "deltanet", "gated_deltanet", "mamba2")
BITS_PER_FLOAT = 32
SCHEMA = "bigbrane.ladder_calibration/2"
# Criteria the gate requires as "pass". Criterion 5 is an implementation review, not a stored-result check.
REQUIRED_CRITERIA = ("1", "2", "3", "4", "6", "7", "8", "9")

# Thresholds fixed before the acceptance sweep (decision B10). They are recorded in every report by `policy()`.
# Criterion 2: the low-K control is K8 accuracy of at least 0.25, twice the 0.125 a model reaches by guessing one
# of the K stored values; the capacity drop from K8 to K64 is unchanged from B.5.
LOW_K, HIGH_K = 8, 64
LEARNED_MIN_ACC = 0.25
FIXED_STATE_MIN_DROP = 0.2
ATTENTION_MAX_DROP = 0.05
# Criterion 4: the window model is near chance beyond its receptive field; a rival counts as equal-state when its
# counted state floats (convolution buffers included) are at most this multiple of the window's.
WINDOW_NEAR_CHANCE = 0.1
STATE_MATCH_MAX_RATIO = 1.25
# Criterion 6: the information bound is tested at a copy length only when it caps whole-string accuracy at this
# value or lower (payload at least ~4.3 bits above the state bits); below that margin it is reported. The long
# copy lengths live in their own task (configs/ladder.toml [tasks.copy_long]): the same training as `copy`, run
# for the fixed-state models only, evaluated at lengths no attention model could process.
BOUND_TESTABLE_ACC = 0.05
LONG_COPY_TASK = "copy_long"
# Criterion 9: pairs are eligible when the mean accuracy gap exceeds GAP_ELIGIBLE. On one seed, a difference of at
# most TIE_MARGIN (evaluation resolution), or both models at or above CEILING (the level criterion 1 calls learned,
# where saturation hides the order), is a tie that neither agrees with nor reverses the order.
# A cell has unstable onset when its seeds' accuracies on some set spread by more than UNSTABLE_SPREAD; such a
# cell needs a same-seed repeat attempt whose accuracy on every set is within REPEAT_MAX_DIFF of the original.
GAP_ELIGIBLE = 0.1
TIE_MARGIN = 0.01
CEILING = 0.95
UNSTABLE_SPREAD = 0.5
REPEAT_MAX_DIFF = 0.1


def policy():
    """Execution policy and predeclared thresholds, stored in every report."""
    return {"seeds": list(SEEDS), "sweep_seed": ladder.SWEEP_SEED,
            "acceptance_execution": "container on cpu (trusted: true); host_mps results are diagnostics",
            "criterion_2": {"low_k": LOW_K, "high_k": HIGH_K, "learned_min_acc": LEARNED_MIN_ACC,
                            "fixed_state_min_drop": FIXED_STATE_MIN_DROP, "attention_max_drop": ATTENTION_MAX_DROP},
            "criterion_4": {"near_chance": WINDOW_NEAR_CHANCE, "state_match_max_ratio": STATE_MATCH_MAX_RATIO},
            "criterion_6": {"bound_testable_acc": BOUND_TESTABLE_ACC, "long_copy_task": LONG_COPY_TASK},
            "criterion_9": {"gap_eligible": GAP_ELIGIBLE, "tie_margin": TIE_MARGIN, "ceiling": CEILING,
                            "unstable_spread": UNSTABLE_SPREAD, "repeat_max_diff": REPEAT_MAX_DIFF}}


def result_device(result):
    """Device a stored run trained on. Results written before the host_mps mode existed carry no device field;
    every one of them ran on CPU in the executor container."""
    return result.get("device", "cpu")


def result_trusted(result):
    """True for a run that executed in the pinned Podman container on CPU, the only acceptance evidence. Results
    from before the host_mps mode carry no execution record and were container runs."""
    return result.get("execution", {}).get("mode", ladder.CONTAINER) == ladder.CONTAINER \
        and result_device(result) == "cpu"


def _combine(*statuses):
    """Status of a criterion made of parts: any fail fails it, otherwise any part that is not decided keeps it
    undecided, and "info" parts do not count."""
    for status in ("fail", "incomplete", "unresolved"):
        if status in statuses:
            return status
    return "pass"


def _repeat_group(spec):
    """Models that differ only by their `attempt` label share a group: same family, width, depth and hyper."""
    return json.dumps([spec["family"], spec["d"], spec.get("layers", 2), spec.get("hyper", {})], sort_keys=True)


def _is_repeat(cfg, model):
    return "attempt" in cfg["models"][model]


def _seed_values(points, cell, set_name, metric="acc"):
    """{seed: value} for a cell and evaluation set, or {} when the cell is missing."""
    runs = points.get(cell, {})
    return {seed: runs[seed]["sets"][set_name][metric] for seed in runs if set_name in runs[seed]["sets"]}


def _complete(values):
    return all(seed in values for seed in SEEDS)


def _round(values):
    return {str(k): round(float(v), 4) for k, v in values.items()}


def _bucket_acc(result, set_names, lo_exclusive, hi_inclusive=None):
    """Accuracy over MQAR queries whose key-to-query distance lies in (lo, hi], pooled over sets."""
    correct = total = 0.0
    for name in set_names:
        for label, entry in result["sets"][name].get("distance", {}).items():
            lo, hi = (int(x) for x in label.split("-"))
            lo -= 1
            if lo >= lo_exclusive and (hi_inclusive is None or hi <= hi_inclusive):
                correct += entry["acc"] * entry["n"]
                total += entry["n"]
    return (correct / total if total else float("nan")), int(total)


def criterion_1(cfg, points):
    cell = ("mqar256", "tf-d64")
    ks = cfg["tasks"]["mqar256"]["eval_k"]
    per_k = {"K%d" % k: _seed_values(points, cell, "K%d" % k) for k in ks if k <= 64}
    numbers = {name: _round(v) for name, v in per_k.items()}
    if not all(_complete(v) for v in per_k.values()):
        return {"status": "incomplete", "detail": "transformer d64 at N 256 lacks seeds", "numbers": numbers}
    worst = min(min(v.values()) for v in per_k.values())
    ok = worst >= 0.95
    return {"status": "pass" if ok else "fail", "numbers": numbers,
            "detail": "full attention d64, N 256: lowest accuracy over K <= 64 and seeds %.3f (needs >= 0.95)" % worst}


def _capacity_outcome(low, high, attention):
    """Per-seed outcome of the criterion 2 capacity check: "unlearned" when the low-K control is missed (the
    drop then measures nothing), else "pass" or "fail" on the predeclared drop."""
    if low < LEARNED_MIN_ACC:
        return "unlearned"
    drop = low - high
    return "pass" if (drop <= ATTENTION_MAX_DROP if attention else drop >= FIXED_STATE_MIN_DROP) else "fail"


def criterion_2(cfg, points):
    lo, hi = "K%d" % LOW_K, "K%d" % HIGH_K
    models = [(m, False) for m in ("la-d32", "gdn-d32", "m2-s16-d32")] + [("tf-d32", True)]
    numbers, verdicts, missing = {}, {}, []
    for model, attention in models:
        low, high = _seed_values(points, ("mqar512", model), lo), _seed_values(points, ("mqar512", model), hi)
        if not (_complete(low) and _complete(high)):
            missing.append(model)
            numbers[model] = {lo: _round(low), hi: _round(high)}
            continue
        outcomes = {s: _capacity_outcome(low[s], high[s], attention) for s in SEEDS}
        verdicts[model] = ("fail" if "fail" in outcomes.values() else
                           "unresolved" if "unlearned" in outcomes.values() else "pass")
        numbers[model] = {lo: _round(low), hi: _round(high), "drop": _round({s: low[s] - high[s] for s in SEEDS}),
                          "outcome": {str(s): o for s, o in outcomes.items()}, "verdict": verdicts[model]}
    rule = ("N 512, d 32: a model counts only on seeds where it learns the low-K control (%s accuracy >= %g); "
            "each fixed-state model must then lose >= %g from %s to %s, full attention at most %g. A model that "
            "never learns the control is unresolved, not a pass"
            % (lo, LEARNED_MIN_ACC, FIXED_STATE_MIN_DROP, lo, hi, ATTENTION_MAX_DROP))
    if missing:
        return {"status": "incomplete", "numbers": numbers, "detail": "missing seeds for %s; %s" % (missing, rule)}
    status = _combine(*verdicts.values())
    unlearned = {m: [s for s in SEEDS if numbers[m]["outcome"][str(s)] == "unlearned"] for m in verdicts}
    unlearned = {m: seeds for m, seeds in unlearned.items() if seeds}
    note = "" if not unlearned else "; seeds that never learned the control: %s" % unlearned
    return {"status": status, "numbers": numbers, "detail": rule + note}


def criterion_3(cfg, points):
    by_family = {}
    for (task, model), runs in points.items():
        if task != "mqar512" or not all(s in runs for s in SEEDS) or _is_repeat(cfg, model):
            continue
        spec = cfg["models"][model]
        if spec["family"] not in FIXED_FAMILIES:
            continue
        state = bb_recall.state_formula(spec["family"], spec["d"], 512, hyper=spec.get("hyper", {}))
        vals = [runs[s]["sets"]["K64"]["acc"] for s in SEEDS]
        by_family.setdefault(spec["family"], []).append((state, model, float(np.mean(vals)), float(np.std(vals))))
    numbers, verdicts = {}, []
    for family, rows in by_family.items():
        rows.sort()
        numbers[family] = [{"model": m, "state_floats": s, "mean_K64": round(a, 4), "std": round(sd, 4)}
                           for s, m, a, sd in rows]
        if len(rows) < 2:
            continue
        inversions = 0
        for (s0, m0, a0, d0), (s1, m1, a1, d1) in zip(rows, rows[1:]):
            if a1 < a0:
                inversions += 1 if a0 - a1 <= 2 * max(d0, d1, 0.01) else 2
        verdicts.append(inversions <= 1)
    if not verdicts:
        return {"status": "incomplete", "numbers": numbers,
                "detail": "needs at least two state sizes of one fixed-state family at N 512 (slice second)"}
    return {"status": "pass" if all(verdicts) else "fail", "numbers": numbers,
            "detail": "R(N 512, K 64) non-decreasing in state within each family, one inversion within seed noise "
                      "allowed; families with one size are listed but not judged"}


def criterion_4(cfg, points):
    window_model = "swa-w32-d64"
    rivals = ["gdn-d64", "m2-s32-d64"]
    window_spec = cfg["models"][window_model]
    window, layers = window_spec["hyper"]["window"], window_spec.get("layers", 2)
    field = layers * window       # each layer sees w tokens back, so a dependency can travel layers x w
    sets = ["K%d" % k for k in cfg["tasks"]["mqar256"]["eval_k"]]
    numbers = {"receptive_field": {"window": window, "layers": layers, "tokens": field}}
    for model in [window_model] + rivals:
        runs = points.get(("mqar256", model), {})
        numbers[model] = {}
        for s in runs:
            beyond, n_beyond = _bucket_acc(runs[s], sets, field)
            numbers[model][str(s)] = {"beyond_field": round(beyond, 4), "n_beyond": n_beyond,
                                      "within_field": round(_bucket_acc(runs[s], sets, 0, field)[0], 4)}
        spec = cfg["models"][model]
        numbers[model]["state_floats_total"] = bb_recall.state_formula(spec["family"], spec["d"], 256,
                                                                       hyper=spec.get("hyper", {}))
        numbers[model]["state_floats_mixer"] = bb_recall.state_formula(spec["family"], spec["d"], 256,
                                                                       hyper=spec.get("hyper", {}),
                                                                       include_conv=False)
    models = [window_model] + rivals
    if not all(all(str(s) in numbers[m] for s in SEEDS) for m in models):
        return {"status": "incomplete", "numbers": numbers, "detail": "missing seeds"}
    if any(numbers[m][str(s)]["n_beyond"] == 0 for m in models for s in SEEDS):
        return {"status": "incomplete", "numbers": numbers,
                "detail": "no scored query lies wholly beyond the %d-token receptive field" % field}
    window_state = numbers[window_model]["state_floats_total"]
    ratios = {m: round(numbers[m]["state_floats_total"] / window_state, 4) for m in rivals}
    numbers["state_ratio_to_window"] = ratios
    state_matched = all(r <= STATE_MATCH_MAX_RATIO for r in ratios.values())
    beyond = {m: [numbers[m][str(s)]["beyond_field"] for s in SEEDS] for m in models}
    near_chance = all(b <= WINDOW_NEAR_CHANCE for b in beyond[window_model])
    rival_wins = all(max(beyond[m][i] for m in rivals) > beyond[window_model][i] for i in range(len(SEEDS)))
    return {"status": "pass" if near_chance and rival_wins and state_matched else "fail", "numbers": numbers,
            "detail": "window w %d, %d layers, receptive field %d tokens: accuracy on queries whose key is more than "
                      "%d tokens back <= %g on every seed (%s); the better of Gated DeltaNet d64 and Mamba-2 s32 d64 "
                      "above the window there on every seed (%s); rival counted state (convolution buffers "
                      "included) within %gx of the window's %d floats (%s, ratios %s)"
                      % (window, layers, field, field, WINDOW_NEAR_CHANCE, near_chance, rival_wins,
                         STATE_MATCH_MAX_RATIO, window_state, state_matched, ratios)}


def bound_testable_length(state_bits, alphabet):
    """Shortest copy length whose payload L log2(D) exceeds `state_bits` by enough that the information bound
    2^(state bits - payload bits) is at most BOUND_TESTABLE_ACC."""
    return math.ceil((state_bits + math.log2(1 / BOUND_TESTABLE_ACC)) / math.log2(alphabet))


def counted_state_bits(cfg, model):
    """FP32 bits of a fixed-state model's counted state (independent of the sequence length)."""
    spec = cfg["models"][model]
    return BITS_PER_FLOAT * bb_recall.state_formula(spec["family"], spec["d"], 0, layers=spec.get("layers", 2),
                                                    hyper=spec.get("hyper", {}))


def copy_length_tasks(cfg):
    """{evaluation set name: copy task that holds it} for the copy task and, when configured, the long-copy task."""
    owners = {}
    for name in ("copy", LONG_COPY_TASK):
        if name in cfg["tasks"]:
            for length in cfg["tasks"][name]["eval_lengths"]:
                if "L%d" % length in owners:
                    raise ValueError("copy length %d is evaluated by both %s and %s"
                                     % (length, owners["L%d" % length], name))
                owners["L%d" % length] = name
    return owners


def _copy_bound(cfg, models):
    """Theorem 2.7 premise per fixed-state model: counted state bits against payload bits L log2(D) at each tested
    copy length, and the whole-string accuracy 2^(state bits - payload bits) that a deterministic model with that
    state cannot exceed once the payload is larger (1 while the state still holds the payload)."""
    bits_per_symbol = math.log2(cfg["tasks"]["copy"]["alphabet"])
    lengths = sorted(int(name[1:]) for name in copy_length_tasks(cfg))
    rows = {}
    for m in models:
        state_bits = counted_state_bits(cfg, m)
        rows[m] = {"state_bits_fp32": state_bits, "copy_length_at_bound": state_bits / bits_per_symbol,
                   "lengths": {"L%d" % n: {"payload_bits": n * bits_per_symbol,
                                           "max_string_acc": 2.0 ** min(0.0, state_bits - n * bits_per_symbol)}
                               for n in lengths}}
    return rows


def _bound_part(cfg, points, models, bound):
    """Finite-state information bound: at lengths where it caps whole-string accuracy at BOUND_TESTABLE_ACC or
    lower, the measured accuracy of each fixed-state model must stay under the cap plus binomial slack. Where
    no tested length reaches that margin the part is "info": a short-copy failure is not a test of the bound."""
    owners = copy_length_tasks(cfg)
    testable = [(m, name) for m in models for name, row in bound[m]["lengths"].items()
                if row["max_string_acc"] <= BOUND_TESTABLE_ACC]
    if not testable:
        longest = max(int(name[1:]) for name in owners)
        smallest = min(bound[m]["state_bits_fp32"] for m in models)
        return {"status": "info", "numbers": {"checked": []},
                "detail": "payload at the longest tested length (L %d) is %.0f bits against at least %d counted state "
                          "bits, so the bound does not cap accuracy at any tested length; reported, not tested"
                          % (longest, bound[models[0]]["lengths"]["L%d" % longest]["payload_bits"], smallest)}
    checked, violations = [], []
    for m, name in testable:
        measured = _seed_values(points, (owners[name], m), name, "seq_acc")
        if not _complete(measured):
            return {"status": "incomplete", "numbers": {"checked": checked},
                    "detail": "missing seeds for %s on %s (task %s)" % (m, name, owners[name])}
        n_eval = cfg["tasks"][owners[name]]["n_eval"]
        cap = bound[m]["lengths"][name]["max_string_acc"]
        allowed = cap + 3 * math.sqrt(cap * (1 - cap) / n_eval) + 1 / n_eval
        row = {"model": m, "set": name, "max_string_acc": cap, "allowed": allowed, "measured": _round(measured)}
        checked.append(row)
        if any(v > allowed for v in measured.values()):
            violations.append(row)
    return {"status": "fail" if violations else "pass", "numbers": {"checked": checked, "violations": violations},
            "detail": "%d model and length cells where the state bound caps whole-string accuracy at %g or lower; "
                      "%d measured above the cap plus binomial slack" % (len(checked), BOUND_TESTABLE_ACC,
                                                                          len(violations))}


def criterion_6(cfg, points):
    transformers = ["tf-d64", "tf-halibi-d64"]
    fixed = ["la-d64", "gdn-d64", "m2-s32-d64"]
    copy_cfg, lookup_cfg = cfg["tasks"]["copy"], cfg["tasks"]["lookup_prefix"]
    copy_set, train_set = "L%d" % (2 * copy_cfg["train_max_len"]), "L%d" % copy_cfg["train_max_len"]
    lookup_set = "L%d" % lookup_cfg["train_max_len"]
    copy = {m: _seed_values(points, ("copy", m), copy_set, "seq_acc") for m in transformers + fixed}
    lookup = {m: _seed_values(points, ("lookup_prefix", m), lookup_set, "seq_acc") for m in fixed}
    numbers = {"copy_string_acc_%s" % copy_set: {m: _round(v) for m, v in copy.items()},
               "copy_string_acc_%s" % train_set: {m: _round(_seed_values(points, ("copy", m), train_set, "seq_acc"))
                                                  for m in transformers + fixed},
               "prefix_lookup_answer_acc_%s" % lookup_set: {m: _round(v) for m, v in lookup.items()}}
    longest = "L%d" % max(lookup_cfg["eval_lengths"])
    numbers["prefix_lookup_answer_acc_%s" % longest] = {
        m: _round(_seed_values(points, ("lookup_prefix", m), longest, "seq_acc")) for m in fixed}
    bound = _copy_bound(cfg, fixed)
    numbers["theorem_2_7_bound"] = bound
    # Empirical part: copy separation at 2 x L_train and the prefix-key positive control.
    if all(_complete(v) for v in copy.values()) and all(_complete(v) for v in lookup.values()):
        best_tf = max(transformers, key=lambda m: np.mean(list(copy[m].values())))
        copy_ok = all(copy[best_tf][s] > copy[m][s] for m in fixed for s in SEEDS)
        lookup_ok = all(lookup[m][s] >= 0.9 for m in fixed for s in SEEDS)
        numbers["best_transformer"] = best_tf
        empirical = {"status": "pass" if copy_ok and lookup_ok else "fail",
                     "detail": "copy at 2 x L_train (%s): %s string accuracy above every fixed-state model on every "
                               "seed (%s); prefix-key lookup at %s >= 0.9 for every fixed-state model on every seed "
                               "(%s)" % (copy_set, best_tf, copy_ok, lookup_set, lookup_ok)}
    else:
        empirical = {"status": "incomplete", "detail": "missing seeds"}
    information = _bound_part(cfg, points, fixed, bound)
    numbers["bound_part"] = information["numbers"]
    parts = {"empirical": empirical, "bound": {"status": information["status"], "detail": information["detail"]}}
    return {"status": _combine(empirical["status"], information["status"]), "numbers": numbers, "parts": parts,
            "detail": "empirical part (%s): %s. Finite-state bound part (%s): %s"
                      % (empirical["status"], empirical["detail"], information["status"], information["detail"])}


def _slope(rows):
    xs = np.log([r["length"] for r in rows])
    ys = np.log([r["median_s"] for r in rows])
    return float(np.polyfit(xs, ys, 1)[0])


def criterion_7(cfg, out_dir):
    numbers = {"state_floats": {}, "flops_per_token_4096": {}}
    families = {"tf-d64": None, "swa-w32-d64": None, "la-d64": None, "gdn-d64": None, "m2-s32-d64": None}
    flat = True
    for model in families:
        spec = cfg["models"][model]
        states = {n: bb_recall.state_formula(spec["family"], spec["d"], n, hyper=spec.get("hyper", {}))
                  for n in (256, 4096)}
        numbers["state_floats"][model] = states
        numbers["flops_per_token_4096"][model] = bb_recall.forward_flops_per_token(
            spec["family"], spec["d"], 4096, 512, hyper=spec.get("hyper", {}))
        growing = bb_recall.STATE_CLASS[spec["family"]] == "growing"
        flat &= (states[4096] > states[256]) if growing else (states[4096] == states[256])
    flops_ok = all(numbers["flops_per_token_4096"]["tf-d64"] > v
                   for m, v in numbers["flops_per_token_4096"].items() if m != "tf-d64")
    slopes = {}
    for path in sorted(Path(out_dir, "microbench").glob("*/microbench.json")):
        for row in json.loads(path.read_text())["rows"]:
            if row["threads"] == 1 and row["infer"] and row["model"] in families and row["model"] not in slopes:
                slopes[row["model"]] = round(_slope([r for r in row["infer"] if r["length"] >= 1024]), 3)
    numbers["measured_time_slope_1024_4096"] = slopes
    if "tf-d64" not in slopes or len(slopes) < 2:
        return {"status": "incomplete", "numbers": numbers, "detail": "no microbenchmark inference timings"}
    # The window model runs as dense masked attention on CPU (O(N^2) by implementation, C.3), so its measured
    # slope is not a property of the mechanism; it is reported and left out of the comparison.
    time_ok = all(slopes["tf-d64"] > v for m, v in slopes.items() if m not in ("tf-d64", "swa-w32-d64"))
    return {"status": "pass" if flat and flops_ok and time_ok else "fail", "numbers": numbers,
            "detail": "state flat in N for fixed-state and window models and growing for attention (%s); analytic "
                      "FLOPs per token at N 4096 highest for attention (%s); measured log-log forward-time slope "
                      "over 1024 to 4096 tokens steeper for attention than for the fixed-state families (%s)" % (flat, flops_ok, time_ok)}


def criterion_8(cfg, points):
    pairs = [("mqar256", "K64", "acc"), ("mqar256", "K32", "acc"), ("copy", "L64", "seq_acc")]
    numbers, verdicts = {}, []
    for task, set_name, metric in pairs:
        la = _seed_values(points, (task, "la-d64"), set_name, metric)
        tf = _seed_values(points, (task, "tf-d64"), set_name, metric)
        numbers["%s/%s" % (task, set_name)] = {"la-d64": _round(la), "tf-d64": _round(tf)}
        if not (_complete(la) and _complete(tf)):
            return {"status": "incomplete", "numbers": numbers, "detail": "missing seeds"}
        gap = np.mean([tf[s] for s in SEEDS]) - np.mean([la[s] for s in SEEDS])
        same = all(np.sign(tf[s] - la[s]) == np.sign(gap) for s in SEEDS)
        verdicts.append(abs(gap) > 0.1 and same)
    separated_tasks = sum(verdicts)
    return {"status": "pass" if verdicts[0] and verdicts[2] else "fail", "numbers": numbers,
            "detail": "transformer d64 and linear attention d64 differ by more than 0.1 with the same order on every "
                      "seed on MQAR N 256 K 64 and on copy at 2 x L_train (%d of %d sets separated)"
                      % (separated_tasks, len(verdicts))}


def _seed_order(a, b):
    """Order of two accuracies on one seed: 1 or -1, or 0 for a tie (a difference within TIE_MARGIN, or both
    models at or above CEILING), which neither agrees with nor reverses the order."""
    if abs(a - b) <= TIE_MARGIN or min(a, b) >= CEILING:
        return 0
    return 1 if a > b else -1


def _order_part(cfg, points):
    """Meaningful order: every pair whose mean accuracy differs by more than GAP_ELIGIBLE at a grid point keeps
    its order on the seeds where the two models are not tied. A reversal beyond the tie margin is a violation
    whatever its size. Repeat attempts of one model are not separate models."""
    by_set = {}
    for (task, model), runs in points.items():
        if not all(s in runs for s in SEEDS) or _is_repeat(cfg, model):
            continue
        for set_name in runs[SEEDS[0]]["sets"]:
            by_set.setdefault((task, set_name), {})[model] = {s: runs[s]["sets"][set_name]["acc"] for s in SEEDS}
    checked, tied_seeds, violations = 0, 0, []
    for (task, set_name), models in by_set.items():
        names = sorted(models)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ma, mb = np.mean(list(models[a].values())), np.mean(list(models[b].values()))
                if abs(ma - mb) <= GAP_ELIGIBLE:
                    continue
                checked += 1
                sign = 1 if ma > mb else -1
                orders = [_seed_order(models[a][s], models[b][s]) for s in SEEDS]
                tied_seeds += orders.count(0)
                if -sign in orders:
                    violations.append({"task": task, "set": set_name, "a": a, "b": b,
                                       "a_values": _round(models[a]), "b_values": _round(models[b])})
    numbers = {"pairs_checked": checked, "violations": violations, "tied_seed_orders": tied_seeds,
               "sets": len(by_set)}
    if not checked:
        return {"status": "incomplete", "numbers": numbers,
                "detail": "no grid point with three seeds for two models"}
    return {"status": "fail" if violations else "pass", "numbers": numbers,
            "detail": "every pair of models whose mean accuracy differs by more than %g at a grid point keeps its "
                      "order on every seed where the two are not tied (a difference of at most %g, or both at "
                      "least %g, is a tie) (%d pairs over %d grid points, %d violations, %d tied seed orders)"
                      % (GAP_ELIGIBLE, TIE_MARGIN, CEILING, checked, len(by_set), len(violations), tied_seeds)}


def _repeat_part(cfg, points):
    """Repeatability: a cell whose seeds spread by more than UNSTABLE_SPREAD on some set has unstable onset, so
    three seed labels do not show the outcome reproduces. It needs a same-seed repeat attempt (a model that
    differs from the cell's model only by `attempt`, run at the same learning rate), and the repeat has to agree
    with the original within REPEAT_MAX_DIFF on every set."""
    groups = {}
    for name, spec in cfg["models"].items():
        groups.setdefault(_repeat_group(spec), []).append(name)
    unstable, checks, unchecked = [], [], []
    for (task, model), runs in sorted(points.items()):
        if _is_repeat(cfg, model) or not all(s in runs for s in SEEDS):
            continue
        spread = max(max(runs[s]["sets"][n]["acc"] for s in SEEDS) - min(runs[s]["sets"][n]["acc"] for s in SEEDS)
                     for n in runs[SEEDS[0]]["sets"])
        if spread <= UNSTABLE_SPREAD:
            continue
        unstable.append("%s/%s" % (task, model))
        found = False
        for other in groups[_repeat_group(cfg["models"][model])]:
            if not _is_repeat(cfg, other):
                continue
            for s, repeat in points.get((task, other), {}).items():
                original = runs.get(s)
                if original is None or abs(original["key"]["lr"] - repeat["key"]["lr"]) > 1e-15:
                    continue
                diff = max(abs(original["sets"][n]["acc"] - repeat["sets"][n]["acc"]) for n in original["sets"])
                checks.append({"cell": "%s/%s" % (task, model), "repeat": other, "seed": s,
                               "max_abs_diff": round(diff, 4), "agree": diff <= REPEAT_MAX_DIFF})
                found = True
        if not found:
            unchecked.append("%s/%s" % (task, model))
    disagree = [c for c in checks if not c["agree"]]
    numbers = {"unstable_cells": unstable, "checks": checks, "without_repeat": unchecked}
    detail = ("%d cells with unstable onset (seed spread > %g on some set), %d same-seed repeat checks, %d "
              "disagreeing by more than %g, %d cells without a repeat"
              % (len(unstable), UNSTABLE_SPREAD, len(checks), len(disagree), REPEAT_MAX_DIFF, len(unchecked)))
    status = "fail" if disagree else "incomplete" if unchecked else "pass"
    return {"status": status, "numbers": numbers, "detail": detail}


def criterion_9(cfg, points):
    order, repeat = _order_part(cfg, points), _repeat_part(cfg, points)
    numbers = dict(order["numbers"], repeatability=repeat["numbers"])
    return {"status": _combine(order["status"], repeat["status"]), "numbers": numbers,
            "parts": {"order": {"status": order["status"], "detail": order["detail"]},
                      "repeatability": {"status": repeat["status"], "detail": repeat["detail"]}},
            "detail": "order (%s): %s. Repeatability (%s): %s"
                      % (order["status"], order["detail"], repeat["status"], repeat["detail"])}


def table(cfg, points):
    rows = []
    for (task, model), runs in sorted(points.items()):
        for set_name in sorted(next(iter(runs.values()))["sets"], key=lambda s: int(s[1:])):
            accs = {s: runs[s]["sets"][set_name]["acc"] for s in sorted(runs)}
            seq = {s: runs[s]["sets"][set_name]["seq_acc"] for s in sorted(runs)}
            first = next(iter(runs.values()))
            rows.append({"task": task, "model": model, "set": set_name, "lr": first["key"]["lr"],
                         "acc": _round(accs), "seq_acc": _round(seq),
                         "mean_acc": round(float(np.mean(list(accs.values()))), 4),
                         "steps_run": {str(s): runs[s]["train"]["steps_run"] for s in sorted(runs)},
                         "params": first["train"]["params"],
                         "state_floats_N": first["state_floats"]})
    return rows


def build_report(cfg, out_dir):
    """Criteria over the results in out_dir. The report's code hash is the one hash every result carries; results
    from different code cannot be judged together, so a mixed directory is an error."""
    out_dir = Path(out_dir)
    results = ladder.load_results(out_dir)
    hashes = sorted({r["code_hash"] for r in results})
    if len(hashes) != 1:
        raise ValueError("%s holds results from %d code hashes %s; a report needs exactly one"
                         % (out_dir, len(hashes), [h[:12] for h in hashes]))
    points = ladder.chosen_points(results, out_dir)
    criteria = {"1": criterion_1(cfg, points), "2": criterion_2(cfg, points), "3": criterion_3(cfg, points),
                "4": criterion_4(cfg, points), "6": criterion_6(cfg, points), "7": criterion_7(cfg, out_dir),
                "8": criterion_8(cfg, points), "9": criterion_9(cfg, points)}
    status = {}
    for result in results:
        status[result.get("status")] = status.get(result.get("status"), 0) + 1
    trusted = all(result_trusted(r) for r in results)
    return {"schema": SCHEMA,
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "code_hash": hashes[0], "devices": sorted({result_device(r) for r in results}),
            "trusted": trusted, "scope": "acceptance" if trusted else "diagnostic", "policy": policy(),
            "runs": {"total": len(results), "by_status": status},
            "criteria": criteria, "table": table(cfg, points)}


def gate(report, current_code_hash):
    """Rung 1 calibration verdict from a stored report: calibrated only when the report has the expected schema,
    holds only trusted results (the Podman CPU container; decision B10), was judged on the acceptance seeds,
    belongs to the current ladder code (a change to the runner, the baselines or the tasks needs a new
    calibration) and records every required criterion as "pass". Any other status (fail, incomplete, unresolved,
    info) or a missing criterion keeps rung 1 uncalibrated. Returns {"calibrated", "reasons", "code_hash",
    "devices"}."""
    reasons = []
    if report.get("schema") != SCHEMA:
        reasons.append("report schema %r, expected %r" % (report.get("schema"), SCHEMA))
    if report.get("trusted") is not True:
        reasons.append("report is a %s report over devices %s: acceptance needs trusted results from the pinned "
                       "Podman CPU container, and host MPS results are diagnostics"
                       % (report.get("scope", "diagnostic"), report.get("devices")))
    if report.get("policy", {}).get("seeds") != list(SEEDS):
        reasons.append("report judged seeds %s, acceptance seeds are %s"
                       % (report.get("policy", {}).get("seeds"), list(SEEDS)))
    if report.get("code_hash") != current_code_hash:
        reasons.append("report belongs to code %s, current ladder code is %s; rerun the rung 1 grid and store a "
                       "new report" % (str(report.get("code_hash"))[:12], current_code_hash[:12]))
    criteria = report.get("criteria", {})
    for name in REQUIRED_CRITERIA:
        crit = criteria.get(name)
        if crit is None:
            reasons.append("criterion %s missing from the report" % name)
        elif crit["status"] != "pass":
            reasons.append("criterion %s is %s: %s" % (name, crit["status"], crit.get("detail", "")))
    return {"calibrated": not reasons, "reasons": reasons, "code_hash": str(report.get("code_hash")),
            "devices": list(report.get("devices", ["cpu"]))}


def render_markdown(report):
    lines = ["scope: %s (trusted: %s, devices: %s, seeds: %s)" % (
        report["scope"], str(report["trusted"]).lower(), ", ".join(report["devices"]), report["policy"]["seeds"]), "",
             "| criterion | status | detail |", "| --- | --- | --- |"]
    for key, crit in report["criteria"].items():
        lines.append("| %s | %s | %s |" % (key, crit["status"], crit["detail"]))
    lines += ["", "| task | model | set | lr | acc per seed | mean | string acc per seed |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["table"]:
        lines.append("| %s | %s | %s | %g | %s | %.3f | %s |" % (
            row["task"], row["model"], row["set"], row["lr"],
            " / ".join("%.3f" % v for v in row["acc"].values()), row["mean_acc"],
            " / ".join("%.3f" % v for v in row["seq_acc"].values())))
    return "\n".join(lines)
