import json
import numpy as np

rng = np.random.default_rng(0)
V, r, Pmax, J, kappa = 16, 3, 6, 3, 1e-3
sp = lambda z: np.log1p(np.exp(z))
E_raw = rng.normal(size=(V, r)); F_raw = rng.normal(size=(V, r))
E_raw[12:] -= 12.0; F_raw[12:] -= 12.0   # filler tokens 12..15 have near-zero affinity -> gated out
E = sp(E_raw); F = sp(F_raw)
beta = float(sp(rng.normal() + 1.0))
l = rng.normal(size=V); u = np.exp(l - l.max()); u /= u.sum()
# sequence with repeated pairs and fillers so that counts > 1, gating and eviction all occur
pairs = [(int(rng.integers(0, 6)), int(rng.integers(6, 12))) for _ in range(6)]
seq = []
while len(seq) < 60:
    if rng.random() < 0.25:
        seq.append(int(rng.integers(12, 16)))
    else:
        pr = pairs[int(rng.integers(0, 6))]; seq.extend(pr)
seq = seq[:60]

# ---------------- card model, as written ----------------
def card_run(seq):
    table = {}
    a = np.zeros(V); x = np.zeros(V); y = np.zeros(V)
    outs = []; evictions = 0; gated_out = 0
    for t, w in enumerate(seq, 1):
        if t >= 2:
            i = seq[t - 2]; k0 = float(E[i] @ F[w])
            if k0 > kappa:
                if (i, w) in table:
                    table[(i, w)][0] += 1; table[(i, w)][1] = t
                else:
                    table[(i, w)] = [1, t, k0]
                if len(table) > Pmax:
                    kk = min(table, key=lambda k: (table[k][0], table[k][1]))
                    del table[kk]; evictions += 1
            else:
                gated_out += 1
        if a[w] == 0:
            a[w] = 1.0; x[w] = 1.0; y[w] = 1.0
        for _ in range(J):
            KTx = F @ (E.T @ x)
            for (i, j), (c, _, k0) in table.items():
                KTx[j] += k0 * (np.exp(beta * c) - 1.0) * x[i]
            y = a / (1.0 + KTx)
            Ky = E @ (F.T @ y)
            for (i, j), (c, _, k0) in table.items():
                Ky[i] += k0 * (np.exp(beta * c) - 1.0) * y[j]
            x = a / (1.0 + Ky)
        q = w
        Kq = E[q] @ F.T
        for (i, j), (c, _, k0) in table.items():
            if i == q:
                Kq[j] += k0 * (np.exp(beta * c) - 1.0)
        outs.append(Kq * x[q] * y + x[q] * u)
    return np.array(outs), evictions, gated_out

# ------------- known model: one-hot linear-attention count state (dense, LFU cap)
# ------------- + log-domain unbalanced Sinkhorn with dustbin, pointer-sentinel readout
def lse(M, axis):
    m = np.max(M, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(M - m), axis=axis))

def known_run(seq, iters=J, return_state=False):
    K0d = E @ F.T
    S = np.zeros((V, V)); Lst = np.zeros((V, V))
    a = np.zeros(V)
    f = np.full(V + 1, -np.inf); g = np.full(V + 1, -np.inf); f[V] = 0.0; g[V] = 0.0
    outs = []
    for t, w in enumerate(seq, 1):
        if t >= 2:
            i = seq[t - 2]
            if K0d[i, w] > kappa:
                S[i, w] += 1.0; Lst[i, w] = t
                nz = np.argwhere(S > 0)
                if len(nz) > Pmax:
                    order = np.lexsort((Lst[nz[:, 0], nz[:, 1]], S[nz[:, 0], nz[:, 1]]))
                    ii, jj = nz[order[0]]; S[ii, jj] = 0.0; Lst[ii, jj] = 0.0
        if a[w] == 0:
            a[w] = 1.0; f[w] = 0.0; g[w] = 0.0
        A = np.zeros((V + 1, V + 1))
        A[:V, :V] = np.log(K0d) + beta * S   # attention logits; dustbin kernel entries = 1
        loga = np.where(a > 0, 0.0, -np.inf)
        for _ in range(iters):
            g[:V] = loga - lse(A[:, :V] + f[:, None], axis=0)
            f[:V] = loga - lse(A[:V, :] + g[None, :], axis=1)
        q = w
        plan_row = np.exp(f[q] + A[q, :V] + g[:V])
        dust = np.exp(f[q] + A[q, V] + g[V])
        outs.append(plan_row / a[q] + dust * u)
    return np.array(outs)

with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
    pc, ev, go = card_run(seq)
    pk = known_run(seq)
    pconv = known_run(seq, iters=2000)

diff = float(np.max(np.abs(pc - pk)))
scale = float(max(1.0, np.max(np.abs(pc))))
tv_conv = float(np.max(0.5 * np.sum(np.abs(pc - pconv), axis=1)))
mass_err = float(np.max(np.abs(pc.sum(1) - 1.0)))
equiv = bool(np.isfinite(diff) and diff < 1e-6 * scale)
print(json.dumps({"equivalent": equiv, "max_abs_diff": diff,
  "notes": f"n={len(seq)} steps compared; evictions={ev}, gated-out bigrams={go}; known=dense LFU-capped one-hot outer-product state + log-domain dustbin Sinkhorn (J warm-started iters) + sentinel mix; card mass err={mass_err:.1e}; max TV of J=3 vs converged={tv_conv:.3e}"}))
