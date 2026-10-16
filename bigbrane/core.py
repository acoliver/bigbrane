"""Observed-token generators, exact causal oracles, and scaled MAP EM."""
from __future__ import annotations
import numpy as np

K, V, L = 4, 8, 64
REGIMES = ((.60, .80), (.95, .40))
SIZES = (8, 32, 128, 256)


def rng(seed_tuple):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_tuple)))


def validate_hmm(A, B, pi):
    k = len(pi)
    if A.shape != (k, k) or B.ndim != 2 or B.shape[0] != k:
        raise ValueError('incompatible HMM shapes')
    for p in (A, B, pi):
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError('HMM probabilities must be finite and positive')
        if np.max(np.abs(p.sum(axis=-1)-1)) > 1e-10:
            raise ValueError('HMM rows must sum to one')
    return max(float(np.max(np.abs(p.sum(axis=-1)-1))) for p in (A, B, pi))


def stationary(A):
    mat = A.T - np.eye(len(A))
    mat[-1] = 1.
    rhs = np.zeros(len(A)); rhs[-1] = 1.
    pi = np.linalg.solve(mat, rhs)
    if (pi < 0).any() or abs(pi.sum()-1) > 1e-10 or np.max(np.abs(pi @ A-pi)) > 1e-10:
        raise ArithmeticError('invalid stationary solution')
    return pi


def generator(root, regime, g):
    rho, eta = REGIMES[regime]
    r = rng((root, 0, regime, g))
    A = rho*np.eye(K)+(1-rho)*r.dirichlet(np.ones(K), size=K)
    perm = r.permutation(V)
    S = np.zeros((K, V))
    for i in range(K):
        S[i, perm[2*i:2*i+2]] = .5
    B = eta*S+(1-eta)/V
    pi = stationary(A)
    validate_hmm(A, B, pi)
    return A, B, pi, S, perm


def sample(A, B, pi, count, length, seed, stress=False, lam=.5):
    k, v = B.shape
    r = rng(seed)
    xs = np.empty((count, length), dtype=np.int64); zs = np.empty_like(xs)
    for a in range(count):
        previous = r.choice(k, p=pi)
        current = r.choice(k, p=A[previous]) if stress else previous
        for t in range(length):
            zs[a, t] = current
            xs[a, t] = r.choice(v, p=B[current])
            if t+1 < length:
                prob = ((1-lam)*A[current]+lam*(.9*(np.arange(k) == previous)+.1/k)) if stress else A[current]
                previous, current = current, r.choice(k, p=prob)
    return xs, zs


def hmm_filter(tokens, A, B, pi):
    validate_hmm(A, B, pi)
    tokens = np.asarray(tokens, dtype=np.int64)
    single = tokens.ndim == 1
    data = tokens[None, :] if single else tokens
    q = np.broadcast_to(pi, (len(data), len(pi))).copy()
    out = np.empty((*data.shape, B.shape[1]), dtype=np.float64)
    for t in range(data.shape[1]):
        out[:, t] = q @ B
        post = q*B[:, data[:, t]].T/out[np.arange(len(data)), t, data[:, t], None]
        q = post @ A
    return out[0] if single else out


def stress_filter(tokens, A, B, pi, lam=.5):
    validate_hmm(A, B, pi)
    tokens = np.asarray(tokens, dtype=np.int64)
    single = tokens.ndim == 1
    data = tokens[None, :] if single else tokens
    k = len(pi)
    q = np.broadcast_to(pi[:, None]*A, (len(data), k, k)).copy()
    transition = (1-lam)*A[None, :, :]+lam*(.9*np.eye(k)[:, None, :]+.1/k)
    out = np.empty((*data.shape, B.shape[1]), dtype=np.float64)
    for t in range(data.shape[1]):
        out[:, t] = q.sum(axis=1) @ B
        post = q*B[:, data[:, t]].T[:, None, :]/out[np.arange(len(data)), t, data[:, t], None, None]
        q = np.einsum('aij,ijk->ajk', post, transition)
    return out[0] if single else out


def batch_forward_backward(data, A, B, pi):
    validate_hmm(A, B, pi)
    data = np.asarray(data)
    count, length = data.shape; k = len(pi)
    emissions = B[:, data].transpose(1, 2, 0)
    alpha = np.empty((count, length, k)); scales = np.empty((count, length))
    alpha[:, 0] = pi*emissions[:, 0]
    scales[:, 0] = alpha[:, 0].sum(1); alpha[:, 0] /= scales[:, 0, None]
    for t in range(1, length):
        alpha[:, t] = (alpha[:, t-1] @ A)*emissions[:, t]
        scales[:, t] = alpha[:, t].sum(1); alpha[:, t] /= scales[:, t, None]
    beta = np.ones_like(alpha)
    for t in range(length-2, -1, -1):
        beta[:, t] = (emissions[:, t+1]*beta[:, t+1]) @ A.T/scales[:, t+1, None]
    gamma = alpha*beta
    gamma /= gamma.sum(2, keepdims=True)
    xi = alpha[:, :-1, :, None]*A*(emissions[:, 1:]*beta[:, 1:])[:, :, None, :]
    xi /= xi.sum((2, 3), keepdims=True)
    return np.log(scales).sum(1), gamma, xi


def forward_backward(seq, A, B, pi):
    ll, g, x = batch_forward_backward(np.asarray(seq)[None, :], A, B, pi)
    return float(ll[0]), g[0], x[0]


def map_objective(data, A, B, pi, eps=.001):
    probs = hmm_filter(data, A, B, pi)
    ll = np.take_along_axis(np.log(probs), np.asarray(data)[..., None], -1).sum()
    return float(ll+eps*(np.log(pi).sum()+np.log(A).sum()+np.log(B).sum()))


def em_step(data, A, B, pi, eps=.001):
    _, g, x = batch_forward_backward(data, A, B, pi)
    k, v = B.shape
    cp = eps+g[:, 0].sum(0)
    ca = eps+x.sum((0, 1))
    cb = np.full((k, v), eps)
    for symbol in range(v):
        cb[:, symbol] += g[np.asarray(data) == symbol].sum(0)
    pp = cp/(len(data)+k*eps)
    aa = ca/(g[:, :-1].sum((0, 1))[:, None]+k*eps)
    bb = cb/(g.sum((0, 1))[:, None]+v*eps)
    validate_hmm(aa, bb, pp)
    return pp, aa, bb


def nll(probs, tokens):
    return float(-np.take_along_axis(np.log(probs), np.asarray(tokens)[..., None], -1).mean())


def kl(p, q):
    return float(np.sum(p*np.log(p/q), axis=-1).mean())
