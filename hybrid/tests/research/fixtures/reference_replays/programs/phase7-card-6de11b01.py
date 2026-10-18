import json, itertools
import numpy as np

rng = np.random.default_rng(12345)
MOD = 2 ** 61 - 1

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def make_params(V, d, N, w):
    return dict(E=rng.normal(0, 1 / np.sqrt(d), (V, d)),
                U=rng.normal(0, 1 / np.sqrt(d), (w, d, d)),
                Wo=rng.normal(0, 1 / np.sqrt(d), (d, d)),
                P=rng.normal(0, 1, (N, d)))

# ---------------- card, implemented as written ----------------
def card_run(seq, prm, k, B, tau, beta, r):
    E, U, Wo, P = prm['E'], prm['U'], prm['Wo'], prm['P']
    w, d = U.shape[0], E.shape[1]
    n = len(seq)
    M = np.zeros((B, d)); T = [None] * B
    def h1(kappa):
        return sum((i + 1) * r ** (j + 1) for j, i in enumerate(kappa)) % MOD % B
    hs, ps, gs = [], [], []
    for t in range(n):
        acc = np.zeros(d)
        for j in range(w):
            if t - j >= 0:
                acc = acc + U[j] @ E[seq[t - j]]
        u = gelu(acc)
        z = Wo @ u
        kappa = tuple(sorted(np.argsort(-(P @ u))[:k].tolist()))
        a = h1(kappa)
        hit = (T[a] == kappa)
        h = z + (M[a] if hit else 0.0)
        logits = -beta * ((E - h) ** 2).sum(axis=1)
        p = np.exp(logits - logits.max()); p /= p.sum()
        hs.append(h.copy()); ps.append(p)
        if t + 1 < n:
            tgt = E[seq[t + 1]]
            g = np.linalg.norm(tgt - h) > tau
            if g:
                M[a] = tgt - z
                T[a] = kappa
            gs.append(int(g))
    return np.array(hs), np.array(ps), gs

# ------------- known model: short conv + gated delta-rule linear attention -------------
# key/query feature map phi(u) = one-hot over B slots (fixed hash of TopK code);
# tag check = second delta-rule memory with same key storing one-hot code id (eviction on collision)
def delta_run(seq, prm, k, B, tau, beta, r, N, use_tag=True):
    E, U, Wo, P = prm['E'], prm['U'], prm['Wo'], prm['P']
    w, d = U.shape[0], E.shape[1]
    V = E.shape[0]
    codes = {c: i for i, c in enumerate(itertools.combinations(range(N), k))}
    C = len(codes)
    Ucat = np.concatenate([U[j] for j in range(w)], axis=1)  # d x (w d) conv kernel
    out_bias = -beta * (E ** 2).sum(axis=1)                   # tied embedding + norm bias
    S = np.zeros((B, d)); Stag = np.zeros((B, C))
    n = len(seq)
    hs, ps, gs = [], [], []
    for t in range(n):
        win = np.concatenate([E[seq[t - j]] if t - j >= 0 else np.zeros(d) for j in range(w)])
        u = gelu(Ucat @ win)
        z = Wo @ u
        top = np.sort(np.argsort(-(P @ u))[:k])
        a = sum((int(i) + 1) * r ** (j + 1) for j, i in enumerate(top)) % MOD % B
        key = np.zeros(B); key[a] = 1.0
        cvec = np.zeros(C); cvec[codes[tuple(int(i) for i in top)]] = 1.0
        y = S.T @ key
        gate = float((Stag.T @ key) @ cvec) if use_tag else 1.0
        h = z + gate * y
        logits = 2 * beta * (E @ h) + out_bias
        p = np.exp(logits - logits.max()); p /= p.sum()
        hs.append(h.copy()); ps.append(p)
        if t + 1 < n:
            v = E[seq[t + 1]] - z
            g = float(np.linalg.norm(E[seq[t + 1]] - h) > tau)
            S = S - g * np.outer(key, key @ S) + g * np.outer(key, v)
            Stag = Stag - g * np.outer(key, key @ Stag) + g * np.outer(key, cvec)
            gs.append(int(g))
    return np.array(hs), np.array(ps), gs

maxdiff, maxdiff_notag, scale = 0.0, 0.0, 1.0
gate_mismatch, total_hits_possible = 0, 0
trials = 0
for (w, B) in [(1, 4096), (1, 8), (2, 8), (2, 4), (1, 3), (2, 64)]:
    for rep in range(6):
        V, d, N, k, n = 10, 8, 16, 3, 12
        prm = make_params(V, d, N, w)
        tau = 0.5; beta = float(rng.uniform(0.5, 3.0))
        r = int(rng.integers(2, 2 ** 61 - 2))
        seq = rng.integers(0, 6, size=n).tolist()  # small alphabet -> repeated contexts -> hits
        hc, pc, gc = card_run(seq, prm, k, B, tau, beta, r)
        hk, pk, gk = delta_run(seq, prm, k, B, tau, beta, r, N, True)
        hn, pn, gn = delta_run(seq, prm, k, B, tau, beta, r, N, False)
        scale = max(scale, float(np.abs(hc).max()))
        maxdiff = max(maxdiff, float(np.abs(hc - hk).max()), float(np.abs(pc - pk).max()))
        maxdiff_notag = max(maxdiff_notag, float(np.abs(hc - hn).max()), float(np.abs(pc - pn).max()))
        gate_mismatch += sum(int(x != y) for x, y in zip(gc, gk))
        trials += 1

equiv = bool(maxdiff / scale < 1e-6 and gate_mismatch == 0)
notes = ('%d trials incl. small B to force collisions; gated delta rule with one-hot hashed key + tag memory gate matches card h_t and p_t at every step (gate mismatches=%d). '
         'Without tag gate (plain one-hot delta rule) max diff=%.3g, so the tag/eviction check is required.') % (trials, gate_mismatch, maxdiff_notag)
print(json.dumps({'equivalent': equiv, 'max_abs_diff': maxdiff, 'notes': notes}))
