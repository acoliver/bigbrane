"""Metrics computed by the evaluator from raw predictions and oracle data. NumPy only.

Conventions (applied identically to every entrant):
- Predictions must be finite, non-negative and have rows summing to 1 within SUM_TOL;
  otherwise the whole file is invalid and gets no metrics.
- Before logs, predicted probabilities are floored at FLOOR (no renormalization). A perfect
  prediction is never floored, so it scores KL exactly 0.
- DIST tasks: kl = KL(oracle || prediction) in nats. ANSWER tasks: kl = cross-entropy
  -log q(answer), which is the KL to the one-hot oracle.
- acc gives tie-aware top-1 credit: if the prediction's maximum is shared by m tokens, the
  credit is (number of those tokens that are correct) / m. Correct means the exact answer,
  or any token in the oracle's own argmax set for DIST tasks. A uniform prediction therefore
  scores |argmax set| / V, and never 1 by accident.
"""
from __future__ import annotations

import math

import numpy as np

FLOOR = 1e-12
SUM_TOL = 1e-3
TIE = 1e-12


def validate(pred, n, vocab):
    """Return a list of problems (empty when valid)."""
    problems = []
    if pred.dtype.kind != "f":
        problems.append("dtype %s is not floating point" % pred.dtype)
        return problems
    if pred.shape != (n, vocab):
        problems.append("shape %s, expected %s" % (pred.shape, (n, vocab)))
        return problems
    if not np.isfinite(pred).all():
        problems.append("non-finite values")
        return problems
    if (pred < 0).any():
        problems.append("negative probabilities")
    worst = float(np.max(np.abs(pred.sum(1) - 1))) if n else 0.0
    if worst > SUM_TOL:
        problems.append("row sums deviate from 1 by up to %.3g" % worst)
    return problems


def kl_rows(p, q):
    q = np.maximum(q, FLOOR)
    support = p > 0
    terms = np.zeros_like(p)
    terms[support] = p[support] * (np.log(p[support]) - np.log(q[support]))
    return terms.sum(1)


def ce_rows(answers, q):
    return -np.log(np.maximum(q[np.arange(len(answers)), answers], FLOOR))


def top1_credit(q, correct):
    """correct: boolean (N, V) mask of tokens counted as right."""
    tied = q >= q.max(1, keepdims=True) - TIE
    return (tied & correct).sum(1) / tied.sum(1)


def correct_mask(kind, oracle, vocab):
    if kind == "dist":
        return oracle >= oracle.max(1, keepdims=True) - TIE
    mask = np.zeros((len(oracle), vocab), dtype=bool)
    mask[np.arange(len(oracle)), oracle] = True
    return mask


def _summary(kl, acc):
    return {"kl": float(kl.mean()), "acc": float(acc.mean()), "n": int(len(kl))}


def task_metrics(kind, oracle, pred, group, group_labels, extras=None, meta=None):
    vocab = pred.shape[1]
    kl = kl_rows(oracle, pred) if kind == "dist" else ce_rows(oracle, pred)
    acc = top1_credit(pred, correct_mask(kind, oracle, vocab))
    out = _summary(kl, acc)
    if len(group_labels) > 1:
        out["groups"] = {str(label): _summary(kl[group == g], acc[group == g]) for g, label in enumerate(group_labels)}
    if extras and "close_token" in extras:
        lo, hi = int(meta["close_lo"]), int(meta["close_hi"])
        rows = extras["close_token"] >= 0
        sub = pred[rows, lo:hi]
        target = np.zeros_like(sub, dtype=bool)
        target[np.arange(rows.sum()), extras["close_token"][rows] - lo] = True
        out["close_acc"] = float(top1_credit(sub, target).mean()) if rows.any() else None
        out["close_n"] = int(rows.sum())
    return out


def metric_value(metrics, name, group=None):
    if group is None:
        return metrics[name]
    return metrics["groups"][str(group)][name]


def tokens_to_target(tokens, values, op, target):
    """Smallest train-token count reaching the target, plus a log-linear interpolation.

    tokens: increasing train token counts; values: metric per size (None for missing).
    op: "<=" (lower is better, e.g. kl) or ">=" (higher is better, e.g. acc)."""
    reached = [v is not None and (v <= target if op == "<=" else v >= target) for v in values]
    if not any(reached):
        return {"reached": False, "first_tokens": None, "interp_tokens": None}
    i = reached.index(True)
    interp = float(tokens[i])
    if i > 0 and values[i - 1] is not None and values[i - 1] != values[i]:
        frac = (target - values[i - 1]) / (values[i] - values[i - 1])
        interp = math.exp(math.log(tokens[i - 1]) + frac * (math.log(tokens[i]) - math.log(tokens[i - 1])))
    return {"reached": True, "first_tokens": int(tokens[i]), "interp_tokens": interp}
