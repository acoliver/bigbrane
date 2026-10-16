"""Causality of the baselines and the template: batch independence through the public API, and
for the neural baselines, logits at position t unchanged when tokens after t change."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from hybrid.bench import api
from hybrid.bench.baselines import HERE as BASELINE_DIR
from hybrid.bench.tasks import build_task

from .conftest import TEMPLATE

sys.path.insert(0, str(BASELINE_DIR))
import bb_neural  # noqa: E402
import bb_ngram  # noqa: E402

BUDGET = {"max_steps": 20, "max_epochs": 5, "batch_size": 8, "lr": 3e-3}
CLASSES = {"ngram": bb_ngram.Candidate, "gru": bb_neural.GRUCandidate,
           "transformer": bb_neural.TransformerCandidate, "ssm": bb_neural.SSMCandidate,
           "linear_attention": bb_neural.LinearAttentionCandidate}


def load_template():
    namespace = {}
    exec(compile(TEMPLATE.read_text(), str(TEMPLATE), "exec"), namespace)
    return namespace["Candidate"]


@pytest.fixture(scope="module")
def task():
    return build_task("compose", 0, 32, 24, {"n_types": 2, "n_fillers": 3, "max_depth": 3, "length": 40})


@pytest.mark.parametrize("name", [*CLASSES, "template"])
def test_batch_independence(name, task):
    cls = load_template() if name == "template" else CLASSES[name]
    model = cls(task.vocab_size, 0, BUDGET)
    model.fit(task.train)
    report = api.check_causality(model, task.prefixes)
    assert report["ok"], report
    probs = model.predict_proba(task.prefixes)
    assert probs.shape == (len(task.prefixes), task.vocab_size)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-9)
    assert not api.describe_problems(model.describe())


@pytest.mark.parametrize("name", ["gru", "transformer", "ssm", "linear_attention"])
def test_future_tokens_do_not_change_past_logits(name, task):
    model = CLASSES[name](task.vocab_size, 0, BUDGET)
    model.fit(task.train)
    rng = np.random.default_rng(0)
    seq = rng.integers(0, task.vocab_size, 64).tolist()
    base = model.logits_all([seq])[0]
    for t in (0, 10, 31, 62):
        changed = seq[:t + 1] + rng.integers(0, task.vocab_size, 63 - t).tolist()
        other = model.logits_all([changed])[0]
        torch.testing.assert_close(other[:t + 1], base[:t + 1], atol=1e-5, rtol=1e-5)
        assert not torch.allclose(other[t + 1:], base[t + 1:])


def test_check_causality_catches_a_batch_dependent_model():
    class Leaky:
        def predict_proba(self, prefixes):
            v = np.full((len(prefixes), 3), 1.0)
            v[:, 0] += len(prefixes)
            return v / v.sum(1, keepdims=True)
    assert not api.check_causality(Leaky(), [[1], [2, 0], [1, 1, 1]])["ok"]
