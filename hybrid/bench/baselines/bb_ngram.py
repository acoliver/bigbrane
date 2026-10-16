"""n-gram baseline with Witten-Bell interpolated backoff. Standalone (NumPy only).

P(w | h) = (c(h, w) + T(h) * P(w | h')) / (c(h) + T(h)), where h' drops the oldest token of
h and T(h) is the number of distinct tokens seen after h. The base case is an add-one
unigram. Prediction reads only the last ORDER - 1 tokens, so predict cost is O(1) in context
length per prefix.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

ORDER = 4


class Candidate:
    def __init__(self, vocab_size, seed, budget):
        self.V = int(vocab_size)
        self.seed = seed
        self.counts = defaultdict(lambda: np.zeros(self.V))  # context tuple -> next-token counts
        self.unigram = np.ones(self.V)

    def fit(self, train):
        for seq in train:
            for t, tok in enumerate(seq):
                self.unigram[tok] += 1
                for k in range(1, ORDER):
                    if t - k < 0:
                        break
                    self.counts[tuple(seq[t - k:t])][tok] += 1
        self.counts = dict(self.counts)

    def _dist(self, prefix):
        p = self.unigram / self.unigram.sum()
        for k in range(1, ORDER):
            if len(prefix) < k:
                break
            c = self.counts.get(tuple(prefix[-k:]))
            if c is None:
                break
            total, types = c.sum(), np.count_nonzero(c)
            p = (c + types * p) / (total + types)
        return p

    def predict_proba(self, prefixes):
        return np.stack([self._dist(p) for p in prefixes]).astype(np.float64)

    def describe(self):
        return {"params": int(self.V + sum(np.count_nonzero(c) for c in self.counts.values())),
                "state_floats_per_sequence": ORDER - 1,
                "notes": "Witten-Bell interpolated %d-gram; params counts non-zero n-gram entries" % ORDER}
