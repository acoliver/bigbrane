"""Log-log slope fit for the scaling probe. NumPy only; runs in the evaluator job.

The probe (runner.py) records, for each context length n, several wall times of one
predict_proba call on a fixed batch, plus the peak resident memory of that measurement.
Here we take the median time per length and fit log(time) = slope * log(n) + c by least
squares. Fixed per-call overhead flattens the curve at short contexts, so the verdict uses
the tail fit over the longest `tail_points` lengths; the full-range fit is reported too.
"""
from __future__ import annotations

import math

import numpy as np


def fit_slope(lengths, values):
    x = np.log(np.asarray(lengths, dtype=np.float64))
    y = np.log(np.asarray(values, dtype=np.float64))
    if len(x) < 2:
        raise ValueError("need at least two points to fit a slope")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / total if total > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "points": int(len(x))}


def summarize(raw, claimed_exponent=None, tolerance=0.3, tail_points=4):
    """raw: scaling.json from the candidate job. Returns the evaluator's scaling block."""
    rows = [r for r in raw["lengths"] if r.get("times_s")]
    lengths = [r["n"] for r in rows]
    medians = [float(np.median(r["times_s"])) for r in rows]
    peaks = [r.get("peak_rss_delta_bytes") for r in rows]
    out = {"lengths": lengths, "median_s": medians, "peak_rss_delta_bytes": peaks,
           "batch": raw.get("batch"), "repeats": raw.get("repeats"), "rss_method": raw.get("rss_method"),
           "stopped": raw.get("stopped"), "errors": [r for r in raw["lengths"] if r.get("error")],
           "claimed_exponent": claimed_exponent, "tolerance": tolerance}
    if any(m <= 0 for m in medians):
        raise ValueError("non-positive median time")
    out["time_fit_all"] = fit_slope(lengths, medians) if len(rows) >= 2 else None
    tail = rows[-tail_points:]
    out["time_fit_tail"] = fit_slope([r["n"] for r in tail], medians[-len(tail):]) if len(tail) >= 2 else None
    mem = [(n, p) for n, p in zip(lengths, peaks) if p is not None and p > 0]
    mem_tail = mem[-tail_points:]
    out["memory_fit_tail"] = fit_slope(*zip(*mem_tail)) if len(mem_tail) >= 2 else None
    fit = out["time_fit_tail"]
    if claimed_exponent is None or fit is None:
        out["verdict"] = "no_claim" if claimed_exponent is None else "insufficient_data"
    else:
        out["verdict"] = "match" if abs(fit["slope"] - claimed_exponent) <= tolerance else "mismatch"
        out["slope_minus_claim"] = fit["slope"] - claimed_exponent
    if fit is not None and not math.isfinite(fit["slope"]):
        raise ValueError("non-finite slope")
    return out
