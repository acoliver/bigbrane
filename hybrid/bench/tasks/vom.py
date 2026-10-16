"""Variable-order Markov source given by a random context tree. Oracle: exact.

A context is a tuple of the most recent tokens, oldest first. The node set is suffix-closed
and every node, internal or leaf, carries a next-token distribution. The next-token
distribution of a history is that of the deepest node reached by walking backwards through
the history; when the history is shorter than the tree, the walk stops early and that
internal node's distribution applies. Sampling uses the same rule, so it is exact.
"""
from __future__ import annotations

import numpy as np

from .common import DIST, PARAMS, TEST, TRAIN, TaskData, stream


class ContextTree:
    def __init__(self, dists, split):
        self.dists = dists    # context tuple -> (V,) distribution
        self.split = split    # set of contexts that have children

    @classmethod
    def random(cls, r, vocab, split_probs, alpha):
        dists, split = {}, set()
        frontier = [()]
        while frontier:
            context = frontier.pop()
            dists[context] = r.dirichlet(np.full(vocab, alpha))
            depth = len(context)
            if depth < len(split_probs) and r.random() < split_probs[depth]:
                split.add(context)
                frontier.extend((v,) + context for v in range(vocab - 1, -1, -1))
        return cls(dists, split)

    def node(self, history):
        context = ()
        while context in self.split and len(context) < len(history):
            context = (history[-len(context) - 1],) + context
        return context

    def next_dist(self, history):
        return self.dists[self.node(history)]

    def sample(self, count, length, r):
        vocab = len(self.dists[()])
        out = np.empty((count, length), dtype=np.int64)
        for a in range(count):
            history = []
            for t in range(length):
                history.append(int(r.choice(vocab, p=self.next_dist(history))))
            out[a] = history
        return out

    def max_depth(self):
        return max(len(c) for c in self.dists)


def build(seed, train_pool, n_test, vocab=6, split_probs=(1.0, 0.6, 0.45, 0.35, 0.25), alpha=0.4, length=128,
          min_prefix=1, test_part=TEST):
    tree = ContextTree.random(stream(seed, "vom", PARAMS), vocab, tuple(split_probs), alpha)
    train = tree.sample(train_pool, length, stream(seed, "vom", TRAIN)).tolist()
    r = stream(seed, "vom", test_part)
    seqs = tree.sample(n_test, length, r)
    cut = r.integers(min_prefix, length, size=n_test)
    prefixes = [seqs[i, :cut[i]].tolist() for i in range(n_test)]
    dists = np.stack([tree.next_dist(p) for p in prefixes])
    depth = np.array([len(tree.node(p)) for p in prefixes], dtype=np.int64)
    return TaskData(
        name="vom", vocab_size=vocab, train=train, prefixes=prefixes, oracle_kind=DIST, oracle=dists,
        group=np.zeros(n_test, dtype=np.int64), group_name="all", group_labels=["all"],
        extras={"context_depth": depth},
        meta={"n_contexts": len(tree.dists), "max_depth": tree.max_depth(), "alpha": alpha,
              "split_probs": list(split_probs), "length": length})
