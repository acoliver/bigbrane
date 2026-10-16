"""Hierarchical composition: typed nested brackets with type-dependent filler content.

Tokens: open_k = k, close_k = T + k (k < T bracket types), filler f = 2T + f.
The process keeps a stack of open types. At each step it chooses, with unnormalized weights,
open (w_open, only below max_depth), close (w_close, only when the stack is non-empty) or
filler (w_fill). An open picks its type uniformly; a close must close the innermost type; a
filler is drawn from a distribution that depends on the innermost open type (or the root).
The next-token distribution is a function of the parsed stack alone, so the oracle is exact.
The extra metric close_acc asks whether, at depth >= 1, the model's most likely closing token
is the one that matches the innermost open bracket.
"""
from __future__ import annotations

import numpy as np

from .common import DIST, PARAMS, TEST, TRAIN, TaskData, stream


class BracketSource:
    def __init__(self, n_types, n_fillers, max_depth, w_open, w_close, w_fill, filler_dists):
        self.T, self.F, self.D = n_types, n_fillers, max_depth
        self.w = (w_open, w_close, w_fill)
        self.filler = filler_dists     # (T + 1, F); row 0 is the root, row 1 + k is inside type k
        self.vocab = 2 * n_types + n_fillers

    def parse(self, tokens):
        stack = []
        for tok in tokens:
            if tok < self.T:
                if len(stack) >= self.D:
                    raise ValueError("open beyond max depth")
                stack.append(tok)
            elif tok < 2 * self.T:
                if not stack or stack[-1] != tok - self.T:
                    raise ValueError("ill-formed close")
                stack.pop()
            elif tok >= self.vocab:
                raise ValueError("token out of range")
        return stack

    def next_dist(self, stack):
        w_open = self.w[0] if len(stack) < self.D else 0.0
        w_close = self.w[1] if stack else 0.0
        total = w_open + w_close + self.w[2]
        dist = np.zeros(self.vocab)
        dist[:self.T] = w_open / total / self.T
        if stack:
            dist[self.T + stack[-1]] = w_close / total
        row = stack[-1] + 1 if stack else 0
        dist[2 * self.T:] = self.w[2] / total * self.filler[row]
        return dist

    def sample(self, count, length, r):
        out = np.empty((count, length), dtype=np.int64)
        for a in range(count):
            stack = []
            for t in range(length):
                tok = int(r.choice(self.vocab, p=self.next_dist(stack)))
                if tok < self.T:
                    stack.append(tok)
                elif tok < 2 * self.T:
                    stack.pop()
                out[a, t] = tok
        return out


def make_source(seed, n_types, n_fillers, max_depth, w_open, w_close, w_fill, filler_alpha):
    r = stream(seed, "compose", PARAMS)
    filler = r.dirichlet(np.full(n_fillers, filler_alpha), size=n_types + 1)
    return BracketSource(n_types, n_fillers, max_depth, w_open, w_close, w_fill, filler)


def build(seed, train_pool, n_test, n_types=3, n_fillers=6, max_depth=6, w_open=0.3, w_close=0.3, w_fill=0.4,
          filler_alpha=0.5, length=96, min_prefix=1, test_part=TEST):
    src = make_source(seed, n_types, n_fillers, max_depth, w_open, w_close, w_fill, filler_alpha)
    train = src.sample(train_pool, length, stream(seed, "compose", TRAIN)).tolist()
    r = stream(seed, "compose", test_part)
    seqs = src.sample(n_test, length, r)
    cut = r.integers(min_prefix, length, size=n_test)
    prefixes = [seqs[i, :cut[i]].tolist() for i in range(n_test)]
    stacks = [src.parse(p) for p in prefixes]
    dists = np.stack([src.next_dist(s) for s in stacks])
    close = np.array([n_types + s[-1] if s else -1 for s in stacks], dtype=np.int64)
    depth = np.array([len(s) for s in stacks], dtype=np.int64)
    return TaskData(
        name="compose", vocab_size=src.vocab, train=train, prefixes=prefixes, oracle_kind=DIST, oracle=dists,
        group=np.zeros(n_test, dtype=np.int64), group_name="all", group_labels=["all"],
        extras={"close_token": close, "depth": depth},
        meta={"n_types": n_types, "n_fillers": n_fillers, "max_depth": max_depth, "weights": [w_open, w_close, w_fill],
              "close_lo": n_types, "close_hi": 2 * n_types, "length": length,
              "filler": src.filler.tolist()})
