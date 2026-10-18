import json
import numpy as np

# ---------- Card: u1..u6 exactly as written (ledger C, R; registers M) ----------
def card_run(T, pi0, B0, xs):
    K, V = B0.shape
    C = np.zeros((K, V))
    R = B0.sum(axis=1).copy()
    M = np.zeros((3, K))
    ps, hs, ss = [], [], []
    for t, x in enumerate(xs, start=1):
        Phi = (B0 + C) / R[:, None]                      # u1
        q = pi0.copy() if t == 1 else T.T @ M[0]         # u2
        M[1] = q
        p = q @ Phi                                      # u3
        if t >= 2:                                       # u4 (stale h, stale Phi)
            s = M[0] * (T @ Phi[:, x]) / p[x]
            M[2] = s
        M[0] = q * Phi[:, x] / p[x]                      # u5
        if t >= 2:                                       # u6
            C[:, xs[t - 2]] += M[2]
            R += M[2]
            ss.append(M[2].copy())
        ps.append(p.copy()); hs.append(M[0].copy())
    Phi = (B0 + C) / R[:, None]
    p_next = (T.T @ M[0]) @ Phi
    return ps, hs, ss, p_next

# ---------- Known: online quasi-Bayes EM for HMM emissions, fixed T, fixed-lag-1 smoothing ----------
# Dirichlet count table N (prior = B0), plug-in emission mean b = N / rowsum(N) recomputed
# from scratch each step (M-step); E-step uses the two-slice joint xi(z_{t-1}, z_t | x_{1:t})
# under the current estimate; gamma = marginal of xi on z_{t-1}; alpha_t = marginal on z_t.
def online_em_run(Tm, init, prior, xs):
    N = prior.copy()
    alpha = None
    ps, hs, gs = [], [], []
    for t, x in enumerate(xs, start=1):
        b = N / N.sum(axis=1, keepdims=True)
        if t == 1:
            pred_state = init
            pred = pred_state @ b
            a = pred_state * b[:, x]
            alpha = a / a.sum()
        else:
            pred_state = Tm.T @ alpha
            pred = pred_state @ b
            xi = alpha[:, None] * Tm * b[None, :, x]
            xi = xi / xi.sum()
            gamma = xi.sum(axis=1)
            alpha = xi.sum(axis=0)
            N[:, xs[t - 2]] += gamma
            gs.append(gamma)
        ps.append(pred); hs.append(alpha.copy())
    b = N / N.sum(axis=1, keepdims=True)
    p_next = (Tm.T @ alpha) @ b
    return ps, hs, gs, p_next

def softmax_rows(A):
    A = A - A.max(axis=1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(axis=1, keepdims=True)

rng = np.random.default_rng(12345)
maxdiff = 0.0
scale = 0.0
trials = 0
for trial in range(200):
    K = int(rng.integers(2, 6)); V = int(rng.integers(2, 8)); n = 12
    T = softmax_rows(rng.normal(size=(K, K)) * 2.0)
    if trial % 4 == 0:  # sparse/deterministic-ish structure
        T = T * (rng.random((K, K)) < 0.5); T[np.arange(K), rng.integers(0, K, K)] += 1.0
        T = T / T.sum(axis=1, keepdims=True)
    pi0 = rng.dirichlet(np.ones(K))
    B0 = np.exp(rng.normal(size=(K, V)) - 2.0)
    xs = [int(v) for v in rng.integers(0, V, n)]
    a = card_run(T, pi0, B0, xs)
    b = online_em_run(T.copy(), pi0.copy(), B0.copy(), xs)
    for A, B in zip(a[0] + a[1] + a[2] + [a[3]], b[0] + b[1] + b[2] + [b[3]]):
        maxdiff = max(maxdiff, float(np.abs(A - B).max()))
        scale = max(scale, float(np.abs(A).max()))
    trials += 1

# Toy sanity: the card's own example (a c b a) through the known recursion
Tt = np.array([[0.0, .5, .5], [1.0, 0, 0], [0, 1.0, 0]])
pt = np.array([1.0, 0, 0]); Bt = 0.05 * np.ones((3, 3))
_, _, gt, pn = online_em_run(Tt, pt, Bt, [0, 2, 1, 0])
_, _, _, pc = card_run(Tt, pt, Bt, [0, 2, 1, 0])
maxdiff = max(maxdiff, float(np.abs(pn - pc).max()))

rel = maxdiff / max(scale, 1e-12)
eq = bool(rel < 1e-6)
notes = ('%d random (T,pi0,B0,x) trials; compared p_t, h_t, s_t each step and final readout. '
         'Online-EM toy p_next=%s credit_c=%s' % (trials, [round(float(v), 4) for v in pn],
                                                  [round(float(v), 3) for v in gt[1]]))
print(json.dumps({"equivalent": eq, "max_abs_diff": maxdiff, "notes": notes}))
