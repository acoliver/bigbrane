"""Scaling probe: the measured log-log slope of synthetic O(n) and O(n^2) predict functions."""
from __future__ import annotations

import time

import numpy as np
import pytest

from hybrid.bench import runner, scaling


class Synthetic:
    """predict_proba busy-waits c * n^p seconds for a batch of length-n prefixes (sleep is coarse on macOS)."""

    def __init__(self, power, scale):
        self.power, self.scale = power, scale

    def predict_proba(self, prefixes):
        end = time.perf_counter() + self.scale * len(prefixes[0]) ** self.power
        while time.perf_counter() < end:
            pass
        return np.full((len(prefixes), 4), 0.25)


def test_fit_slope_exact_on_power_laws():
    n = [64, 128, 256, 512, 1024]
    for p in (0.0, 1.0, 1.5, 2.0):
        fit = scaling.fit_slope(n, [3e-5 * x ** p for x in n])
        assert fit["slope"] == pytest.approx(p, abs=1e-9) and fit["r2"] == pytest.approx(1.0)


@pytest.mark.parametrize("power,scale", [(1.0, 2e-5), (2.0, 1e-8)])
def test_probe_recovers_synthetic_exponent(power, scale):
    lengths = [2 ** k for k in range(6, 13)]
    raw = runner.scaling_probe(Synthetic(power, scale), 4, lengths, batch=2, repeats=5, warmup=0, seed=0,
                               stop_after_s=10.0)
    assert [r["n"] for r in raw["lengths"]] == lengths and raw["stopped"] is None
    out = scaling.summarize(raw, claimed_exponent=power, tolerance=0.15, tail_points=4)
    assert out["time_fit_tail"]["slope"] == pytest.approx(power, abs=0.15)
    assert out["verdict"] == "match"
    wrong = scaling.summarize(raw, claimed_exponent=power + 1.0, tolerance=0.15, tail_points=4)
    assert wrong["verdict"] == "mismatch"


def test_probe_stops_when_a_length_is_too_slow():
    raw = runner.scaling_probe(Synthetic(2.0, 1e-6), 4, [64, 128, 256, 512], batch=1, repeats=3, warmup=0, seed=0,
                               stop_after_s=0.04)
    assert raw["stopped"].startswith("median") and [r["n"] for r in raw["lengths"]] == [64, 128, 256]


def test_summary_without_claim_and_with_too_few_points():
    raw = {"lengths": [{"n": 64, "times_s": [0.1, 0.2, 0.3]}]}
    out = scaling.summarize(raw, None)
    assert out["verdict"] == "no_claim" and out["time_fit_tail"] is None
    assert scaling.summarize(raw, 1.0)["verdict"] == "insufficient_data"
