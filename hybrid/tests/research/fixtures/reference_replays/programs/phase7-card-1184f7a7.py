import numpy as np, json

def card_model(x, Theta, u, beta):
    # Card update and readout, written as the card states them (0-indexed positions).
    V, d = Theta.shape
    E = np.exp(1j * Theta)
    R = np.zeros(d, dtype=complex)
    logits_all, probs_all = [], []
    for t in range(len(x)):
        if t >= 1:
            R = R + np.conj(E[x[t-1]]) * E[x[t]]
        Rh = R / (np.abs(R) + 1e-9)
        s = np.real((Rh * E[x[t]]) @ np.conj(E).T) / d
        lg = u + beta * s
        p = np.exp(lg - lg.max()); p /= p.sum()
        logits_all.append(lg); probs_all.append(p)
    return np.array(logits_all), np.array(probs_all)

def shift_ssm(X, kernel):
    # Causal convolution along time, as in the H3 shift SSM.
    n = X.shape[0]
    Y = np.zeros_like(X)
    for t in range(n):
        for j, kj in enumerate(kernel):
            if t - j >= 0:
                Y[t] = Y[t] + kj * X[t - j]
    return Y

def diag_ssm(X, A, B, C):
    # Diagonal SSM: h_t = A*h_{t-1} + B*x_t, y_t = C*h_t.
    n, d = X.shape
    h = np.zeros(d, dtype=complex)
    Y = np.zeros_like(X)
    for t in range(n):
        h = A * h + B * X[t]
        Y[t] = C * h
    return Y

def h3_model(x, Wq, Wk, Wv, Wout, u, beta, d, g):
    V = Wq.shape[0]
    oh = np.eye(V)[x].astype(complex)
    Q = oh @ Wq
    K = oh @ Wk
    Vv = oh @ Wv
    Ks = shift_ssm(K, kernel=[0.0, 1.0])
    S = diag_ssm(Ks * Vv, A=np.ones(d), B=np.ones(d), C=np.ones(d))
    O = Q * g(S)
    logits = u[None, :] + (beta / d) * np.real(O @ np.conj(Wout).T)
    P = np.exp(logits - logits.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return logits, P

rng = np.random.default_rng(0)
max_diff = 0.0; max_scale = 1.0; plain_diff = 0.0
for trial in range(6):
    if trial < 4:
        V, d, n = 10, 7, 12
        Theta = rng.uniform(0, 2*np.pi, size=(V, d))
        u = rng.normal(size=V); beta = float(rng.uniform(1, 10))
    else:
        V = d = 16; n = 12
        Theta = 2*np.pi*np.outer(np.arange(V), np.arange(d))/d
        u = np.zeros(V); beta = 8.0
    x = rng.integers(0, V, size=n)
    lc, pc = card_model(x, Theta, u, beta)
    # Mapping: tied phasor embedding E from the card's Theta; K = conj(E), Q = V = E; tied unembedding E.
    E = np.exp(1j * Theta)
    gnorm = lambda S: S / (np.abs(S) + 1e-9)
    lk, pk = h3_model(x, E, np.conj(E), E, E, u, beta, d, gnorm)
    max_diff = max(max_diff, np.abs(lc - lk).max(), np.abs(pc - pk).max())
    max_scale = max(max_scale, np.abs(lc).max())
    # Same architecture without the phase-normalization nonlinearity (plain diagonal linear attention).
    lp, pp = h3_model(x, E, np.conj(E), E, E, u, beta, d, lambda S: S)
    plain_diff = max(plain_diff, np.abs(pc - pp).max())
rel = max_diff / max_scale
print(json.dumps({"equivalent": bool(rel < 1e-6), "max_abs_diff": float(max_diff), "notes": "H3-style shift-SSM x V, A=I diag SSM, Q*g(S) with g=phase norm, tied unembed; exact match incl. g. Without g (plain linear attn) prob diff=%.3g, so the nonlinearity is essential." % plain_diff}))
