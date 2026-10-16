"""Candidate API (RESEARCH-SYSTEM-SPEC.md section 4) and host-side checks for it.

A candidate is a directory containing `candidate.py` that defines:

    class Candidate:
        def __init__(self, vocab_size: int, seed: int, budget: dict): ...
        def fit(self, train: list[list[int]]) -> None: ...
        def predict_proba(self, prefixes: list[list[int]]) -> np.ndarray:  # (B, vocab), causal
        def describe(self) -> dict:
            # {"params": int, "state_floats_per_sequence": int, "notes": str,
            #  "hyper": {name: int, ...}, "train_max_len": int}

Rules the bench relies on:
- The directory may hold other .py helper files; it is put first on sys.path, so
  `candidate.py` can import them by module name. Only NumPy, CPU torch and the standard
  library are available in the executor image. No network.
- predict_proba(prefixes)[i] is the next-token distribution after prefixes[i]. Every prefix
  is non-empty. Row i must depend only on prefixes[i] and the fitted parameters, never on
  other rows in the batch, call order or earlier calls (check_causality tests this).
- Rows must be finite, non-negative and sum to 1 (within 1e-3). Probabilities are floored
  at 1e-12 before logs, so a zero on a correct token costs about 27.6 nats, not infinity.
- budget holds max_steps, max_epochs, batch_size and lr. The baselines train for
  min(max_steps, max_epochs * ceil(len(train) / batch_size)) optimizer steps. The wall and
  memory caps of the candidate job are the hard limits.
- hyper holds the model's integer hyperparameters by name (the state width d, the number of heads or
  slots M, and so on). Card falsifier tasks compare them with the values their protocol needs.
  train_max_len is the length of the longest training sequence fit in full: fit() must use every token of
  every training sequence up to that length (report 2048 if it never truncates). Falsifier tasks whose
  training sequences are longer are not decided.
- A fresh Candidate is built for every (task, train size); vocab sizes differ per task.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

DESCRIBE_KEYS = {"params": int, "state_floats_per_sequence": int, "notes": str}


class Candidate(Protocol):
    def __init__(self, vocab_size: int, seed: int, budget: dict) -> None: ...

    def fit(self, train: list[list[int]]) -> None: ...

    def predict_proba(self, prefixes: list[list[int]]) -> np.ndarray: ...

    def describe(self) -> dict: ...


def contract_problems(describe):
    """The describe() fields that card falsifier tasks rely on: hyper (non-empty dict of str to int) and
    train_max_len (positive int). Checked on candidates only; the baselines do not report them."""
    if not isinstance(describe, dict):
        return ["describe() returned %s, not dict" % type(describe).__name__]
    problems = []
    hyper = describe.get("hyper")
    if (not isinstance(hyper, dict) or not hyper
            or not all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
                       for k, v in hyper.items())):
        problems.append("describe()['hyper'] must be a non-empty dict of integer hyperparameters by name, got %r"
                        % (hyper,))
    limit = describe.get("train_max_len")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        problems.append("describe()['train_max_len'] must be a positive int (the longest training sequence fit in "
                        "full), got %r" % (limit,))
    return problems


def describe_problems(describe):
    problems = []
    if not isinstance(describe, dict):
        return ["describe() returned %s, not dict" % type(describe).__name__]
    for key, kind in DESCRIBE_KEYS.items():
        value = describe.get(key)
        if not isinstance(value, kind) or isinstance(value, bool):
            problems.append("describe()[%r] must be %s, got %r" % (key, kind.__name__, value))
    return problems


def check_causality(model, prefixes, atol=1e-5):
    """Batch independence checks on a fitted candidate. Returns a dict with max deviations and ok.

    1. Each prefix predicted alone equals its row in the full batch.
    2. A reversed batch gives the same rows.
    3. A repeated call gives the same rows (no state carried between calls).
    4. Appending an extra, longer prefix to the batch does not change the other rows."""
    base = np.asarray(model.predict_proba([list(p) for p in prefixes]), dtype=np.float64)
    alone = np.stack([np.asarray(model.predict_proba([list(p)]), dtype=np.float64)[0] for p in prefixes])
    reverse = np.asarray(model.predict_proba([list(p) for p in prefixes[::-1]]), dtype=np.float64)[::-1]
    again = np.asarray(model.predict_proba([list(p) for p in prefixes]), dtype=np.float64)
    longest = max(prefixes, key=len)
    extra = list(longest) + list(longest)
    padded = np.asarray(model.predict_proba([list(p) for p in prefixes] + [extra]), dtype=np.float64)[:-1]
    dev = {"alone": float(np.abs(base - alone).max()), "reversed": float(np.abs(base - reverse).max()),
           "repeat": float(np.abs(base - again).max()), "with_longer": float(np.abs(base - padded).max())}
    return {"max_dev": dev, "ok": all(v <= atol for v in dev.values()), "atol": atol}
