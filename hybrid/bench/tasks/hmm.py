"""Small HMM. Oracle: exact forward filtering P(x_t | x_<t).

The generator, sampler and filter are copied from bigbrane/core.py (generator, sample,
hmm_filter, stationary, validate_hmm) with the regime constants turned into parameters.
bigbrane/ is not imported.
"""
from __future__ import annotations

import numpy as np

from .common import DIST, PARAMS, TEST, TRAIN, TaskData, stream


def validate_hmm(A, B, pi):
    k = len(pi)
    if A.shape != (k, k) or B.ndim != 2 or B.shape[0] != k:
        raise ValueError("incompatible HMM shapes")
    for p in (A, B, pi):
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError("HMM probabilities must be finite and positive")
        if np.max(np.abs(p.sum(axis=-1) - 1)) > 1e-10:
            raise ValueError("HMM rows must sum to one")


def stationary(A):
    mat = A.T - np.eye(len(A))
    mat[-1] = 1.
    rhs = np.zeros(len(A))
    rhs[-1] = 1.
    pi = np.linalg.solve(mat, rhs)
    if (pi < 0).any() or abs(pi.sum() - 1) > 1e-10 or np.max(np.abs(pi @ A - pi)) > 1e-10:
        raise ArithmeticError("invalid stationary solution")
    return pi


def make_hmm(seed, n_states, vocab, rho, eta):
    """bigbrane/core.py generator(): sticky transitions, two-token emission support per state."""
    if vocab < 2 * n_states:
        raise ValueError("vocab must be at least 2 * n_states")
    r = stream(seed, "hmm", PARAMS)
    A = rho * np.eye(n_states) + (1 - rho) * r.dirichlet(np.ones(n_states), size=n_states)
    perm = r.permutation(vocab)
    S = np.zeros((n_states, vocab))
    for i in range(n_states):
        S[i, perm[2 * i:2 * i + 2]] = .5
    B = eta * S + (1 - eta) / vocab
    pi = stationary(A)
    validate_hmm(A, B, pi)
    return A, B, pi


def sample(A, B, pi, count, length, r):
    """bigbrane/core.py sample() without the stress branch."""
    k, v = B.shape
    xs = np.empty((count, length), dtype=np.int64)
    for a in range(count):
        current = r.choice(k, p=pi)
        for t in range(length):
            xs[a, t] = r.choice(v, p=B[current])
            if t + 1 < length:
                current = r.choice(k, p=A[current])
    return xs


def hmm_filter(tokens, A, B, pi):
    """bigbrane/core.py hmm_filter(): out[:, t] = P(x_t | x_<t) for a (N, L) batch."""
    validate_hmm(A, B, pi)
    data = np.asarray(tokens, dtype=np.int64)
    q = np.broadcast_to(pi, (len(data), len(pi))).copy()
    out = np.empty((*data.shape, B.shape[1]), dtype=np.float64)
    for t in range(data.shape[1]):
        out[:, t] = q @ B
        post = q * B[:, data[:, t]].T / out[np.arange(len(data)), t, data[:, t], None]
        q = post @ A
    return out


def build(seed, train_pool, n_test, n_states=4, vocab=8, rho=0.9, eta=0.6, length=64, min_prefix=1, test_part=TEST):
    A, B, pi = make_hmm(seed, n_states, vocab, rho, eta)
    train = sample(A, B, pi, train_pool, length, stream(seed, "hmm", TRAIN)).tolist()
    r = stream(seed, "hmm", test_part)
    seqs = sample(A, B, pi, n_test, length, r)
    cut = r.integers(min_prefix, length, size=n_test)  # prefix length t, scored token index t
    dists = hmm_filter(seqs, A, B, pi)[np.arange(n_test), cut]
    prefixes = [seqs[i, :cut[i]].tolist() for i in range(n_test)]
    return TaskData(
        name="hmm", vocab_size=vocab, train=train, prefixes=prefixes, oracle_kind=DIST, oracle=dists,
        group=np.zeros(n_test, dtype=np.int64), group_name="all", group_labels=["all"],
        meta={"n_states": n_states, "rho": rho, "eta": eta, "length": length,
              "A": A.tolist(), "B": B.tolist(), "pi": pi.tolist()})
