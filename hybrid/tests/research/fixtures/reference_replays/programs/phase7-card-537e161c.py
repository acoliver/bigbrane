import json
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def card_model(seq, Phi, W_o, w_beta, A, lam, M):
    # Card: RLS update of P, K, then power readout, at every position t = 1..n
    d = Phi.shape[1]
    P = np.eye(d) / lam
    K = np.zeros((d, d))
    outs = []
    for t in range(1, len(seq)):
        u = Phi[seq[t - 1]]
        v = Phi[seq[t]]
        beta = sigmoid(w_beta @ np.concatenate([u, v]))
        Pu = P @ u
        P = P - beta * np.outer(Pu, Pu) / (1.0 + beta * (u @ Pu))
        K = K + beta * np.outer(P @ u, v - u @ K)
        alpha = softmax(A[seq[t - 1]])
        h = Phi[seq[t]]
        y = alpha[0] * h
        for m in range(1, M + 1):
            h = K.T @ h
            y = y + alpha[m] * h
        outs.append(W_o @ y)
    return np.array(outs)

def known_model(seq, Phi, W_o, w_beta, A, lam, M):
    # Linear-attention / MesaNet layer computed in parallel with cumulative sums:
    # keys k_t = Phi[x_{t-1}] (token shift), values v_t = Phi[x_t], gate beta_t,
    # KV state C_t = sum beta k v^T, Gram G_t = sum beta k k^T,
    # read r(q) = C_t^T (G_t + lam I)^{-1} q, iterated as a multi-hop fast-weight read
    # with hop outputs mixed by alpha = softmax(A[x_{t-1}]). LM head W_o.
    d = Phi.shape[1]
    x = np.array(seq)
    keys = Phi[x[:-1]]
    vals = Phi[x[1:]]
    betas = sigmoid(np.concatenate([keys, vals], axis=1) @ w_beta)
    Gs = np.cumsum(betas[:, None, None] * keys[:, :, None] * keys[:, None, :], axis=0)
    Cs = np.cumsum(betas[:, None, None] * keys[:, :, None] * vals[:, None, :], axis=0)
    outs = []
    for i in range(len(keys)):
        Greg = Gs[i] + lam * np.eye(d)
        C = Cs[i]
        q = vals[i]
        alpha = softmax(A[x[i]])
        h = q
        y = alpha[0] * h
        for m in range(1, M + 1):
            h = C.T @ np.linalg.solve(Greg, h)
            y = y + alpha[m] * h
        outs.append(W_o @ y)
    return np.array(outs)

rng = np.random.default_rng(0)
V, d, M, n = 10, 6, 4, 12
max_diff = 0.0
max_rel = 0.0
for trial in range(30):
    Phi = rng.normal(size=(V, d)) / np.sqrt(d)
    W_o = rng.normal(size=(V, d))
    w_beta = rng.normal(size=2 * d)
    A = rng.normal(size=(V, M + 1)) * 2.0
    lam = 1e-3 if trial % 2 == 0 else float(rng.uniform(0.01, 1.0))
    BOS = V - 1
    seq = [BOS] + list(rng.integers(0, V - 1, size=n))
    a = card_model(seq, Phi, W_o, w_beta, A, lam, M)
    b = known_model(seq, Phi, W_o, w_beta, A, lam, M)
    diff = float(np.max(np.abs(a - b)))
    scale = float(max(1.0, np.max(np.abs(a))))
    max_diff = max(max_diff, diff)
    max_rel = max(max_rel, diff / scale)

equiv = bool(max_rel < 1e-6)
print(json.dumps({"equivalent": equiv, "max_abs_diff": max_diff, "notes": "card RLS recurrence + power readout vs parallel linear-attention KV/Gram state with MesaNet solve read iterated M hops (FWM-style), alpha-mixed; max relative diff %.2e over 30 trials, %d steps each" % (max_rel, n)}))
