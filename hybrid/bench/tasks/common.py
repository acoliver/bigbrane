"""Shared task types. Every task produces a train pool, test prefixes and an exact oracle.

Test prefixes are independent: each test sequence contributes exactly one prefix and the
tokens after the scored position are never generated into any file. A candidate that reads
every input it is given therefore learns nothing about the scored next tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DIST = "dist"      # oracle is the exact next-token distribution, shape (N, V)
ANSWER = "answer"  # oracle is one exact token, shape (N,)

# Fixed per-task stream ids so seeds never collide between tasks.
TASK_STREAM = {"hmm": 1, "vom": 2, "recall": 3, "compose": 4, "fewshot": 5}
PARAMS, TRAIN, TEST = 0, 1, 2


def length_part(factor):
    """Stream part for the length-generalization test set at `factor` times the training length, so it shares
    no draws with the base test set."""
    return 100 + int(factor)


def rng(seed_tuple):
    """Copied from bigbrane/core.py (rng): PCG64 seeded by a SeedSequence over a tuple."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_tuple)))


def stream(seed, task, part):
    return rng((int(seed), TASK_STREAM[task], part))


@dataclass
class TaskData:
    name: str
    vocab_size: int
    train: list            # pool of training sequences; train size n uses train[:n]
    prefixes: list         # one scored prefix per independent test sequence
    oracle_kind: str       # DIST or ANSWER
    oracle: np.ndarray     # (N, V) float64 distributions or (N,) int64 answers
    group: np.ndarray      # (N,) int64 group index per prefix
    group_name: str        # e.g. "K" for recall, "shots" for fewshot, "all" otherwise
    group_labels: list     # label of each group index
    extras: dict = field(default_factory=dict)  # extra oracle arrays (e.g. compose close tokens)
    meta: dict = field(default_factory=dict)    # generator parameters (oracle side only)

    def __post_init__(self):
        n = len(self.prefixes)
        if not all(len(p) >= 1 for p in self.prefixes):
            raise ValueError("%s: empty test prefix" % self.name)
        if self.oracle_kind == DIST:
            if self.oracle.shape != (n, self.vocab_size):
                raise ValueError("%s: oracle shape %s" % (self.name, self.oracle.shape))
            if np.max(np.abs(self.oracle.sum(1) - 1)) > 1e-9 or (self.oracle < 0).any():
                raise ValueError("%s: oracle rows are not distributions" % self.name)
        elif self.oracle_kind == ANSWER:
            if self.oracle.shape != (n,) or self.oracle.min() < 0 or self.oracle.max() >= self.vocab_size:
                raise ValueError("%s: bad answers" % self.name)
        else:
            raise ValueError("unknown oracle kind %r" % self.oracle_kind)
        if self.group.shape != (n,) or self.group.min() < 0 or self.group.max() >= len(self.group_labels):
            raise ValueError("%s: bad group array" % self.name)
        for seq in self.train:
            if min(seq) < 0 or max(seq) >= self.vocab_size:
                raise ValueError("%s: train token out of range" % self.name)
        for seq in self.prefixes:
            if min(seq) < 0 or max(seq) >= self.vocab_size:
                raise ValueError("%s: prefix token out of range" % self.name)

    def train_tokens(self, n):
        if n > len(self.train):
            raise ValueError("%s: train size %d exceeds pool %d" % (self.name, n, len(self.train)))
        return int(sum(len(s) for s in self.train[:n]))
