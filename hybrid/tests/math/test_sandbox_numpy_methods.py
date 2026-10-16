"""NumPy array methods inside the checkers' restricted namespaces (calibration 2026-10-13, card k3).

NumPy's C methods import helper modules on first use through PyImport_Import, which reads ``__import__`` from the
builtins of the calling frame. The checkers ran candidate code with SAFE_BUILTINS only, so ``a.max()``, ``a.mean()``
and, in the tools image, ``a.sum(axis=1, keepdims=True)`` raised KeyError('__import__'): the k3 learner failed that
way on attempts 1 and 3 and its learnability statements stayed undecided. The live learner is a fixture in
hybrid/tests/research/fixtures/calibration_live_20261013/k3_learner.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid.math import cost, learnability

LIVE = Path(__file__).resolve().parents[1] / "research" / "fixtures" / "calibration_live_20261013"

# The four functions a learner must define; their bodies do not matter here.
STUBS = ("RECIPE = {\"epochs\": 0}\n"  # FIX-3: every learner declares its training recipe
         "def learn(layout, examples, seed):\n    return {}\n"
         "def init(layout, params):\n    return {}\n"
         "def update(state, token):\n    return state\n"
         "def predict(state):\n    return 0\n")


RECORDED_RECIPE = 'RECIPE = {"recorded": "2026-10-13 calibration; schedule as written in learn"}\n'


def test_array_methods_run_in_the_learner_namespace():
    namespace = learnability.load_learner("def f():\n    a = np.ones((2, 3))\n"
                                          "    return float(a.max()) + float(a.mean()) + float(a.std()) "
                                          "+ float(a.sum(axis=1, keepdims=True).sum())\n" + STUBS)
    assert namespace["f"]() == pytest.approx(1.0 + 1.0 + 0.0 + 6.0)


def test_candidate_code_still_cannot_name_import():
    with pytest.raises(cost.CandidateError, match="__import__"):
        cost.load_functions("def init(s, o):\n    return __import__('os')\n", ("init",))
    assert "imports are not allowed" in "; ".join(cost.audit_source("import os\n"))


def test_the_live_k3_learner_now_gets_a_decided_result():
    live = json.loads((LIVE / "k3_learner.json").read_text())
    assert "KeyError: '__import__'" in live["evidence_then"]
    spec = dict(live["spec"], test_count=100)
    # The recorded learner predates FIX-3's RECIPE requirement; its declared recipe is added, its code is unchanged.
    result = learnability.run_learnability(spec, RECORDED_RECIPE + live["learner"])
    assert not result.get("prover_fault"), result["evidence"]
    assert result["status"] in ("pass", "violated") and "seed 0 accuracy" in result["evidence"]
