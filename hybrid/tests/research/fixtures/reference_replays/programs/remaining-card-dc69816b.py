import json, math
import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

def ident(x):
    return x

def softmax(l):
    l = l - l.max()
    p = np.exp(l)
    return p / p.sum()

# ---------------- card's own equations (update steps 1-6, readout) ----------------
def card_run(pr, seq):
    E, U, Wo, P, phi = pr['E'], pr['U'], pr['Wo'], pr['P'], pr['phi']
    k, B, m, tau, beta = pr['k'], pr['B'], pr['m'], pr['tau'], pr['beta']
    w, d = U.shape[0], E.shape[1]
    H = []
    hs, ps, ev = [], [], []
    n = len(seq)
    for t in range(n):
        acc = np.zeros(d)
        for j in range(w):
            if t - j >= 0:
                acc = acc + U[j] @ E[seq[t - j]]
        u = phi(acc)
        z = Wo @ u
        kappa = frozenset(np.argsort(-(P @ u), kind='stable')[:k].tolist())
        J = [e for e in H if e['K'] <= kappa]
        win = max(J, key=lambda e: (len(e['K']), e['s'])) if J else None
        h = z + (win['rho'] if win is not None else 0.0)
        if win is not None:
            win['s'] = t
        hs.append(np.array(h, dtype=float))
        ps.append(softmax(-beta * ((E - h) ** 2).sum(axis=1)))
        if t + 1 < n:
            tgt = E[seq[t + 1]]
            if np.linalg.norm(tgt - h) > tau:
                rho = tgt - z
                if win is not None and win['K'] == kappa:
                    win['rho'] = rho
                    ev.append('a')
                else:
                    C = [e for e in H if np.linalg.norm(e['rho'] - rho) <= tau and len(e['K'] & kappa) >= m]
                    if C:
                        c = max(C, key=lambda e: (len(e['K'] & kappa), e['s']))
                        c['K'] = c['K'] & kappa
                        c['s'] = t
                        ev.append('b')
                    else:
                        if len(H) >= B:
                            H.pop(min(range(len(H)), key=lambda i: H[i]['s']))
                        H.append({'K': kappa, 'rho': rho, 's': t})
                        ev.append('c')
            else:
                ev.append('-')
    return np.array(hs), np.array(ps), ev

# ---------------- known form: incremental NGE / Find-S rule learner ----------------
# Rules are boolean condition vectors over N binary features. Each rule has a class (a vector)
# and a recency stamp. Classification fires the most specific matching rule (a Holland default
# hierarchy) and falls back to the default predictor. Training happens only on a misclassified
# example: relabel an exact duplicate, else LGG(=AND) of the nearest same-class rule, else store
# a new exemplar with LRU eviction.
def base_predictor(pr, seq, t):
    E, U, Wo, P, phi = pr['E'], pr['U'], pr['Wo'], pr['P'], pr['phi']
    x = np.zeros(E.shape[1])
    for j in range(U.shape[0]):
        if t - j >= 0:
            x = x + U[j] @ E[seq[t - j]]
    u = phi(x)
    return Wo @ u, P @ u

def nge_run(pr, seq):
    E = pr['E']
    k, B, m, tau, beta = pr['k'], pr['B'], pr['m'], pr['tau'], pr['beta']
    N, d = pr['P'].shape[0], E.shape[1]
    conds = np.zeros((0, N), dtype=bool)
    cls = np.zeros((0, d))
    stamp = np.zeros(0, dtype=np.int64)
    hs, ps, ev = [], [], []
    n = len(seq)
    for t in range(n):
        z, score = base_predictor(pr, seq, t)
        feat = np.zeros(N, dtype=bool)
        feat[np.argsort(-score, kind='stable')[:k]] = True
        match = np.all((~conds) | feat[None, :], axis=1)
        spec = conds.sum(axis=1)
        fired, best = -1, None
        for i in np.nonzero(match)[0]:
            key = (int(spec[i]), int(stamp[i]))
            if best is None or key > best:
                best, fired = key, int(i)
        h = z + (cls[fired] if fired >= 0 else 0.0)
        if fired >= 0:
            stamp[fired] = t
        hs.append(np.array(h, dtype=float))
        ps.append(softmax(-beta * ((E - h) ** 2).sum(axis=1)))
        if t + 1 < n:
            tgt = E[seq[t + 1]]
            if np.linalg.norm(tgt - h) > tau:
                lab = tgt - z
                if fired >= 0 and np.array_equal(conds[fired], feat):
                    cls[fired] = lab
                    ev.append('a')
                else:
                    same = np.linalg.norm(cls - lab[None, :], axis=1) <= tau
                    ov = (conds & feat[None, :]).sum(axis=1)
                    cand = np.nonzero(same & (ov >= m))[0]
                    if len(cand) > 0:
                        c, bk = -1, None
                        for i in cand:
                            key = (int(ov[i]), int(stamp[i]))
                            if bk is None or key > bk:
                                bk, c = key, int(i)
                        conds[c] = conds[c] & feat
                        stamp[c] = t
                        ev.append('b')
                    else:
                        if conds.shape[0] >= B:
                            i = int(np.argmin(stamp))
                            conds = np.delete(conds, i, axis=0)
                            cls = np.delete(cls, i, axis=0)
                            stamp = np.delete(stamp, i)
                        conds = np.vstack([conds, feat[None, :]])
                        cls = np.vstack([cls, lab[None, :]])
                        stamp = np.append(stamp, t)
                        ev.append('c')
            else:
                ev.append('-')
    return np.array(hs), np.array(ps), ev

# ---------------- strongest delta-rule repair (diagnostic only) ----------------
# Rows are lifted to all subsets of {0..N-1} and stored lazily. The write key is one-hot e_kappa
# and the gated delta rule uses beta = g. The query map marks every subset of kappa, so the read is
# the linear functional sum_{K subset kappa} S[K].
def lifted_delta_run(pr, seq):
    E, tau = pr['E'], pr['tau']
    k, d = pr['k'], E.shape[1]
    S = {}
    hs = []
    n = len(seq)
    for t in range(n):
        z, score = base_predictor(pr, seq, t)
        kappa = frozenset(np.argsort(-score, kind='stable')[:k].tolist())
        y = np.zeros(d)
        for K, v in S.items():
            if K <= kappa:
                y = y + v
        h = z + y
        hs.append(h)
        if t + 1 < n:
            tgt = E[seq[t + 1]]
            if np.linalg.norm(tgt - h) > tau:
                rho = tgt - z
                old = S.get(kappa, np.zeros(d))
                S[kappa] = old - old + rho
    return np.array(hs)

# ---------------- scenarios ----------------
rng = np.random.default_rng(0)
scen = []
# toy instance from the card
E_t = np.array([[i, -1.0] for i in range(6)] + [[0.0, 1.0]])
U_t = np.stack([np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 0.0]])])
Wo_t = np.array([[1.0, 0.0], [0.0, -1.0]])
P_t = np.array([[j, -j * j / 2.0] for j in range(6)] + [[0.0, 100.0]])
toy = dict(E=E_t, U=U_t, Wo=Wo_t, P=P_t, phi=ident, k=2, B=4, m=1, tau=0.5, beta=1.0)
scen.append((toy, [3, 6, 4, 1, 6, 2, 4, 6]))
# toy parameters with random rule/exception sequences
for _ in range(200):
    seq = []
    for _ in range(4):
        a = int(rng.integers(0, 5))
        b = a + 1 if rng.random() < 0.75 else int(rng.integers(0, 6))
        seq += [a, 6, b]
    scen.append((toy, seq))
# fully random parameters
for _ in range(400):
    d, V, N = 4, 6, 10
    w = int(rng.integers(1, 3))
    pr = dict(E=rng.normal(0, 1, (V, d)), U=rng.normal(0, 1 / math.sqrt(d), (w, d, d)),
              Wo=rng.normal(0, 1 / math.sqrt(d), (d, d)), P=rng.normal(0, 1, (N, d)),
              phi=gelu if rng.random() < 0.5 else ident, k=int(rng.integers(2, 5)),
              B=int(rng.integers(2, 7)), m=int(rng.integers(1, 3)),
              tau=float(rng.uniform(0.2, 2.5)), beta=1.0)
    seq = rng.integers(0, int(rng.integers(3, V + 1)), 24).tolist()
    scen.append((pr, seq))

maxdiff, scale, evmis, cons, writes = 0.0, 1.0, 0, 0, 0
dmax, dbad = 0.0, 0
toy_preds_card = toy_preds_known = None
for idx, (pr, seq) in enumerate(scen):
    hc, pc, evc = card_run(pr, seq)
    hk, pk, evk = nge_run(pr, seq)
    hd = lifted_delta_run(pr, seq)
    maxdiff = max(maxdiff, float(np.abs(hc - hk).max()), float(np.abs(pc - pk).max()))
    scale = max(scale, float(np.abs(hc).max()))
    if evc != evk:
        evmis += 1
    cons += evc.count('b')
    writes += sum(1 for e in evc if e != '-')
    dd = float(np.abs(hc - hd).max())
    dmax = max(dmax, dd)
    if dd > 1e-6 * max(1.0, float(np.abs(hc).max())):
        dbad += 1
    if idx == 0:
        toy_preds_card = [int(np.argmax(pc[t])) for t in range(len(seq)) if seq[t] == 6]
        toy_preds_known = [int(np.argmax(pk[t])) for t in range(len(seq)) if seq[t] == 6]

equiv = bool(maxdiff / scale < 1e-6 and evmis == 0)
notes = ('card vs incremental NGE/Find-S+default-hierarchy over %d scenarios; writes=%d consolidations=%d; '
         'event-sequence mismatches=%d; toy preds card=%s known=%s; '
         'lifted 2^N delta-rule repair (subset-sum read) max diff=%.3g, differs in %d/%d scenarios') % (
    len(scen), writes, cons, evmis, toy_preds_card, toy_preds_known, dmax, dbad, len(scen))
print(json.dumps({'equivalent': equiv, 'max_abs_diff': maxdiff, 'notes': notes}))
