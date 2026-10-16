import itertools
import unittest
import numpy as np
from bigbrane.core import (
    em_step, forward_backward, generator, hmm_filter, kl, map_objective,
    rng, sample, stationary, stress_filter, validate_hmm,
)


A = np.array([[.8, .2], [.3, .7]])
B = np.array([[.9, .1], [.2, .8]])
PI = np.array([.65, .35])


def enumerate_hmm(tokens, A, B, pi):
    weights = []
    paths = list(itertools.product(range(len(pi)), repeat=len(tokens)))
    for z in paths:
        w = pi[z[0]] * B[z[0], tokens[0]]
        for t in range(1, len(tokens)):
            w *= A[z[t-1], z[t]] * B[z[t], tokens[t]]
        weights.append(w)
    weights = np.array(weights)
    g = np.zeros((len(tokens), len(pi)))
    xi = np.zeros((len(tokens)-1, len(pi), len(pi)))
    for z, w in zip(paths, weights / weights.sum()):
        for t, state in enumerate(z):
            g[t, state] += w
        for t in range(len(tokens)-1):
            xi[t, z[t], z[t+1]] += w
    return weights.sum(), g, xi


def second_order_joint(tokens, A, B, pi, lam):
    total = 0.
    k = len(pi)
    for z in itertools.product(range(k), repeat=len(tokens)+1):
        w = pi[z[0]] * A[z[0], z[1]] * B[z[1], tokens[0]]
        for t in range(1, len(tokens)):
            transition = (1-lam)*A[z[t], z[t+1]] + lam*(.9*(z[t+1] == z[t-1]) + .1/k)
            w *= transition * B[z[t+1], tokens[t]]
        total += w
    return total


class CoreTests(unittest.TestCase):
    def test_hmm_filter_exhaustive_and_initial(self):
        for length in range(1, 6):
            for symbols in itertools.product(range(2), repeat=length):
                x = np.array(symbols)
                p = hmm_filter(x, A, B, PI)
                for t in range(length):
                    prefix = 1. if t == 0 else enumerate_hmm(x[:t], A, B, PI)[0]
                    exact = [enumerate_hmm(np.append(x[:t], v), A, B, PI)[0]/prefix for v in range(2)]
                    np.testing.assert_allclose(p[t], exact, atol=1e-10, rtol=0)
                ll, g, xi = forward_backward(x, A, B, PI)
                joint, eg, ex = enumerate_hmm(x, A, B, PI)
                self.assertAlmostEqual(ll, np.log(joint), places=10)
                np.testing.assert_allclose(g, eg, atol=1e-10, rtol=0)
                np.testing.assert_allclose(xi, ex, atol=1e-10, rtol=0)
                self.assertAlmostEqual(ll, np.log(p[np.arange(length), x]).sum(), places=10)
        np.testing.assert_allclose(hmm_filter([0], A, B, PI)[0], PI @ B, atol=1e-10)

    def test_stress_exhaustive_and_lambda_zero(self):
        for lam in (0., .5, 1.):
            for length in range(1, 6):
                for symbols in itertools.product(range(2), repeat=length):
                    x = np.array(symbols)
                    p = stress_filter(x, A, B, PI, lam)
                    for t in range(length):
                        prefix = 1. if t == 0 else second_order_joint(x[:t], A, B, PI, lam)
                        exact = [second_order_joint(np.append(x[:t], v), A, B, PI, lam)/prefix for v in range(2)]
                        np.testing.assert_allclose(p[t], exact, atol=1e-10, rtol=0)
                    if lam == 0:
                        np.testing.assert_allclose(p, hmm_filter(x, A, B, PI @ A), atol=1e-10, rtol=0)
        pi = stationary(A)
        np.testing.assert_allclose(stress_filter([0, 1, 0], A, B, pi, 0), hmm_filter([0, 1, 0], A, B, pi), atol=1e-10, rtol=0)

    def test_fb_counts_and_map_updates(self):
        d = np.array([[0, 1, 1], [1, 0, 1]])
        cp = np.full(2, .001); ca = np.full((2, 2), .001); cb = np.full((2, 2), .001)
        occupancy = np.zeros(2); transitions = np.zeros(2)
        for seq in d:
            _, g, x = forward_backward(seq, A, B, PI)
            np.testing.assert_allclose(g.sum(1), 1, atol=1e-10)
            np.testing.assert_allclose(x.sum(2), g[:-1], atol=1e-10)
            np.testing.assert_allclose(x.sum(1), g[1:], atol=1e-10)
            cp += g[0]; ca += x.sum(0)
            occupancy += g.sum(0); transitions += g[:-1].sum(0)
            for t, v in enumerate(seq):
                cb[:, v] += g[t]
        p, a, b = em_step(d, A, B, PI)
        np.testing.assert_allclose(p, cp/(len(d)+.002), atol=1e-10)
        np.testing.assert_allclose(a, ca/(transitions[:, None]+.002), atol=1e-10)
        np.testing.assert_allclose(b, cb/(occupancy[:, None]+.002), atol=1e-10)
        old = map_objective(d, A, B, PI)
        for _ in range(20):
            p, a, b = em_step(d, A if _ == 0 else a, B if _ == 0 else b, PI if _ == 0 else p)
            obj = map_objective(d, a, b, p)
            self.assertGreaterEqual(obj, old - 1e-8*max(1, abs(old)))
            old = obj

    def test_rng_data_repeat_and_stress_initialization(self):
        a, b, pi, s, perm = generator(20260931, 0, 0)
        x, z = sample(a, b, pi, 3, 8, (1, 2, 3))
        x2, z2 = sample(a, b, pi, 3, 8, (1, 2, 3))
        np.testing.assert_array_equal(x, x2); np.testing.assert_array_equal(z, z2)
        # Deterministic emissions expose whether the sampler incorrectly emits z0.
        emission = np.eye(2)
        sx, sz = sample(A, emission, PI, 4, 3, (8, 9), stress=True)
        r = rng((8, 9)); expected = []
        for _ in range(4):
            prev = r.choice(2, p=PI); current = r.choice(2, p=A[prev]); row = []
            for t in range(3):
                row.append(r.choice(2, p=emission[current]))
                if t < 2:
                    prob = .5*A[current]+.5*(.9*(np.arange(2) == prev)+.1/2)
                    prev, current = current, r.choice(2, p=prob)
            expected.append(row)
        np.testing.assert_array_equal(sx, expected); np.testing.assert_array_equal(sx, sz)
        validate_hmm(a, b, pi)
        np.testing.assert_allclose(pi @ a, pi, atol=1e-10)
        np.testing.assert_allclose(s.sum(1), 1)
        self.assertEqual(len(set(perm)), 8)

    def test_probability_integrity_and_causality(self):
        with self.assertRaises(ValueError):
            validate_hmm(A*2, B, PI)
        x = [0, 1, 0, 1]; y = [0, 1, 1, 0]
        for scorer in (hmm_filter, stress_filter):
            p = scorer(x, A, B, PI); q = scorer(y, A, B, PI)
            np.testing.assert_allclose(p[:3], q[:3], atol=1e-10, rtol=0)
            self.assertAlmostEqual(kl(p, p), 0., places=12)
        batch = hmm_filter(np.array([x, y]), A, B, PI)
        np.testing.assert_allclose(batch[0], hmm_filter(x, A, B, PI), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
