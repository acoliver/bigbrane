"""Reference-mechanism capacity control for card falsifier tasks. NumPy only.

Runs inside the falsifier validation job (staged next to falsifier_eval.py). A task whose rules name capacity
claims declares

    "reference": {"kind": "pair_memory", "d": <the card's d>, "min_token": <int>}

and the harness scores the d x d pair memory below on the task's test prefixes. The memory stores the adjacent
token pairs (x[t-1], x[t]) of the prefix without its last token, keeping only pairs where both tokens are at
least min_token (the card's ideal gate: separators and markers are not stored), reads out with the last prefix
token and decodes with the tied lift, argmax over the whole vocabulary:

    lift     phi: V x d, Gaussian entries with variance 1/d, drawn from a fixed seed per lift;
    "rls"    the card's least-squares operator (lam I + U^T U)^-1 U^T Y with lam = LAMBDA;
    "la"     plain linear attention, the sum of outer products U^T Y.

Both are averaged over LIFT_SEEDS lifts, so a prefix's reference accuracy is the fraction of lifts that decode
its answer. The gates that compare rules with these scores live in falsifier_harness.reference_problems.
"""
from __future__ import annotations

import math

import numpy as np

LAMBDA = 1e-3
LIFT_SEEDS = 5
SEED_BASE = 100
KINDS = ("pair_memory",)
VARIANTS = ("rls", "la")


def pair_memory_guesses(prefixes, vocab, d, min_token):
    """{variant: int array (LIFT_SEEDS, n)} of decoded tokens; -1 when a prefix holds no pair to store."""
    n = len(prefixes)
    guesses = {v: np.full((LIFT_SEEDS, n), -1, dtype=np.int64) for v in VARIANTS}
    stored = []
    for p in prefixes:
        context = np.asarray(p[:-1], dtype=np.int64)
        keep = (context[:-1] >= min_token) & (context[1:] >= min_token)
        stored.append((context[:-1][keep], context[1:][keep]))
    for s in range(LIFT_SEEDS):
        phi = np.random.default_rng(SEED_BASE + s).standard_normal((vocab, d)) / math.sqrt(d)
        for i, (keys, values) in enumerate(stored):
            if not len(keys):
                continue
            u, y = phi[keys], phi[values]
            outer = u.T @ y
            operator = {"rls": np.linalg.solve(LAMBDA * np.eye(d) + u.T @ u, outer), "la": outer}
            for variant, memory in operator.items():
                guesses[variant][s, i] = int(np.argmax(phi @ (memory.T @ phi[prefixes[i][-1]])))
    return guesses
