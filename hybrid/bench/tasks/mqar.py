"""Multi-query associative recall (MQAR), as defined by Zoology (Arora et al., arXiv 2312.04927).

Self-contained (NumPy only, no package imports) so the ladder runner can import it inside the executor job.

Format, following `multiquery_ar` in HazyResearch/zoology `zoology/data/associative_recall.py`
(commit d90de385ad9febeae8b53d98f105af089ac2a477), checked against the code on 2026-10-12:

- K key-value pairs come first: k1 v1 k2 v2 ... kK vK (context of 2K tokens). Keys are drawn without repeats
  from 1 .. V/2 - 1, values without repeats from V/2 .. V - 1, so keys and values use disjoint halves of the
  vocabulary and token 0 is never a key.
- The rest of the sequence (N - 2K tokens) is a query region with (N - 2K) / 2 even slots. Every key is
  queried exactly once, at slot gap_i, chosen without repeats with probability proportional to
  power_a * i^(power_a - 1) for i = 1 .. space (power_a 0.01 by default: short gaps are more likely).
- The label for the query of key i sits at the query position itself: reading the query key, the model must
  predict that key's value. The token after a query is not the value. Every other position is unlabeled.
- With random_non_queries (the Zoology default) every position that is still empty (token 0) is filled with
  a uniform token from the whole vocabulary [0, V).

Differences from the Zoology code, all in how randomness is drawn and none in the distribution: draws come
from a PCG64 generator passed in by the caller instead of the global NumPy seed, and one row is drawn at a
time so several K values can share a batch.

Oracle: `oracle(prefix, k)` reads the answer from the visible prefix alone: the last token is a query key,
the keys sit at even positions of the first 2K tokens, and the answer is the token after the matching key.
"""
from __future__ import annotations

import numpy as np

IGNORE = -100
NAME = "mqar"


def vocab_size(vocab=512, **_):
    return int(vocab)


def check_params(seq_len, k, vocab):
    if seq_len % 2:
        raise ValueError("mqar: seq_len must be even, got %d" % seq_len)
    if vocab <= seq_len:
        raise ValueError("mqar: Zoology requires vocab > seq_len (got vocab %d, seq_len %d)" % (vocab, seq_len))
    if 4 * k > seq_len:
        raise ValueError("mqar: Zoology requires 4K <= seq_len (got K %d, seq_len %d)" % (k, seq_len))
    if k > vocab // 2 - 1:
        raise ValueError("mqar: K %d exceeds the %d available keys" % (k, vocab // 2 - 1))


def gap_probabilities(space, power_a):
    p = power_a * np.arange(1, space + 1, dtype=np.float64) ** (power_a - 1)
    return p / p.sum()


def make_row(rng, seq_len, k, vocab, power_a=0.01, random_non_queries=True):
    """One MQAR sequence. Returns (tokens (N,), targets (N,), distance (N,)); targets is IGNORE except at
    query positions, and distance is (query position - key position) at query positions, else -1."""
    check_params(seq_len, k, vocab)
    half = vocab // 2
    keys = rng.choice(np.arange(1, half), size=k, replace=False)
    values = rng.choice(np.arange(half, vocab), size=k, replace=False)
    context = 2 * k
    space = (seq_len - context) // 2
    gaps = rng.choice(space, size=k, replace=False, p=gap_probabilities(space, power_a))
    tokens = np.zeros(seq_len, dtype=np.int64)
    tokens[0:context:2] = keys
    tokens[1:context:2] = values
    query_pos = context + 2 * gaps
    tokens[query_pos] = keys
    targets = np.full(seq_len, IGNORE, dtype=np.int64)
    targets[query_pos] = values
    distance = np.full(seq_len, -1, dtype=np.int64)
    distance[query_pos] = query_pos - 2 * np.arange(k)
    if random_non_queries:
        empty = tokens == 0
        tokens[empty] = rng.integers(0, vocab, size=int(empty.sum()))
    return tokens, targets, distance


def generate(rng, n, seq_len, k_values, vocab=512, power_a=0.01, random_non_queries=True):
    """n sequences of length seq_len; each row's K is drawn uniformly from k_values.

    Returns a dict: tokens (n, N) int64, targets (n, N) int64 (IGNORE where unscored), k (n,) int64,
    distance (n, N) int64 (-1 where unscored)."""
    k_values = np.asarray(k_values, dtype=np.int64)
    tokens = np.empty((n, seq_len), dtype=np.int64)
    targets = np.empty((n, seq_len), dtype=np.int64)
    distance = np.empty((n, seq_len), dtype=np.int64)
    ks = k_values[rng.integers(0, len(k_values), size=n)]
    for row in range(n):
        tokens[row], targets[row], distance[row] = make_row(rng, seq_len, int(ks[row]), vocab, power_a,
                                                            random_non_queries)
    return {"tokens": tokens, "targets": targets, "k": ks, "distance": distance}


def oracle(prefix, k):
    """The answer after `prefix`, which must end with a query key, computed from the prefix alone."""
    prefix = list(prefix)
    if len(prefix) <= 2 * k:
        raise ValueError("mqar oracle: prefix of %d tokens ends inside the key-value context" % len(prefix))
    query = prefix[-1]
    for i in range(k):
        if prefix[2 * i] == query:
            return int(prefix[2 * i + 1])
    raise ValueError("mqar oracle: last token %d is not a stored key" % query)
