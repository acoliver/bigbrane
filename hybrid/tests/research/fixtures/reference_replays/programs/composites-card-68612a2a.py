import json
import numpy as np

rng = np.random.default_rng(0)
m, d, s, U, V = 8, 12, 2, 16, 64
eps, tau = 1e-6, 1e-9
beta = float(rng.uniform(0.5, 2.0))
C = rng.standard_normal((U, m))
C /= np.linalg.norm(C, axis=1, keepdims=True)
E = np.arange(V, dtype=float)  # p = 1, E[y] = y
vals = rng.integers(0, 8, U)

# ---- stream: atoms, consistent / inconsistent pairs, triples, queries ----
stream = []
for a in range(6):
    stream.append(('DEMO', [a], int(vals[a])))
for _ in range(90):
    t = rng.integers(0, 5)
    if t == 0:
        a = int(rng.integers(0, U)); stream.append(('DEMO', [a], int(vals[a])))
    elif t == 1:
        X = list(rng.choice(U, 2, replace=False)); stream.append(('DEMO', X, int(vals[X].sum())))
    elif t == 2:
        X = list(rng.choice(U, 2, replace=False)); stream.append(('DEMO', X, int(vals[X].sum()) + 1))
    elif t == 3:
        X = list(rng.choice(U, 3, replace=False)); stream.append(('DEMO', X, int(vals[X].sum())))
    else:
        X = list(rng.choice(U, int(rng.integers(1, 4)), replace=False)); stream.append(('QUERY', X, None))
# explicit explaining-away episode
stream += [('DEMO', [14, 15], int(vals[14] + vals[15])), ('DEMO', [14], int(vals[14])), ('QUERY', [15], None)]

def softmax_logp(yhat):
    lg = -beta * (E - yhat) ** 2
    lg = lg - lg.max()
    return lg - np.log(np.exp(lg).sum())

# ---- CARD: u1-u4 as written (lists, lstsq refit) ----
class Card:
    def __init__(self):
        self.K, self.Y, self.N = [], [], []
    def pursuit(self, k):
        S, alpha, r = [], np.zeros(0), k.copy()
        for _ in range(s):
            best, bi = tau, -1
            for i, Ki in enumerate(self.K):
                if i in S:
                    continue
                c = abs(float(Ki @ r)) / self.N[i]
                if c > best:
                    best, bi = c, i
            if bi < 0:
                break
            S.append(bi)
            KS = np.stack([self.K[i] for i in S], axis=1)
            alpha = np.linalg.lstsq(KS, k, rcond=None)[0]
            r = k - KS @ alpha
        return S, alpha, r
    def step(self, z, X, y):
        k = C[X].sum(0)
        S, alpha, r = self.pursuit(k)
        yhat = float(sum(alpha[j] * self.Y[i] for j, i in enumerate(S))) if S else 0.0
        logp = softmax_logp(yhat)
        wrote = False
        if z == 'DEMO' and (np.linalg.norm(r) > eps or abs(E[y] - yhat) > eps):
            wrote = True
            self.K.append(k); self.Y.append(float(E[y])); self.N.append(float(np.linalg.norm(k)))
            if len(self.K) > d:
                self.K.pop(0); self.Y.pop(0); self.N.pop(0)
        return yhat, logp, wrote, len(self.K)

# ---- KNOWN: coupled-dictionary sparse coding, textbook OMP (normalized atoms,
#      incremental Gram-Schmidt QR), code transferred to D_h; RAN/ALD-style
#      allocation on sparse residual; budgeted FIFO dictionary ----
class Coupled:
    def __init__(self):
        self.Dl = np.zeros((m, 0)); self.Dh = np.zeros(0)
    def omp(self, x):
        Nat = self.Dl.shape[1]
        if Nat == 0:
            return [], np.zeros(0), x.copy()
        norms = np.linalg.norm(self.Dl, axis=0)
        Dn = self.Dl / norms
        r = x.copy(); S = []; Q = np.zeros((m, 0))
        for _ in range(s):
            c = np.abs(Dn.T @ r)
            c[S] = -np.inf
            i = int(np.argmax(c))
            if not c[i] > tau:
                break
            S.append(i)
            u = Dn[:, i] - Q @ (Q.T @ Dn[:, i])
            Q = np.hstack([Q, (u / np.linalg.norm(u))[:, None]])
            r = x - Q @ (Q.T @ x)
        if not S:
            return [], np.zeros(0), r
        R = Q.T @ Dn[:, S]
        a_n = np.linalg.solve(R, Q.T @ x)
        return S, a_n / norms[S], r
    def step(self, z, X, y):
        x = np.zeros(m)
        for c in X:
            x = x + C[c]
        S, a, r = self.omp(x)
        yhat = float(self.Dh[S] @ a) if S else 0.0
        logp = softmax_logp(yhat)
        wrote = False
        if z == 'DEMO':
            err = abs(E[y] - yhat)
            if np.linalg.norm(r) > eps or err > eps:
                wrote = True
                self.Dl = np.hstack([self.Dl, x[:, None]]); self.Dh = np.append(self.Dh, E[y])
                if self.Dl.shape[1] > d:
                    self.Dl = self.Dl[:, 1:]; self.Dh = self.Dh[1:]
        return yhat, logp, wrote, self.Dl.shape[1]

card, known = Card(), Coupled()
maxdiff, scale, mism, nw = 0.0, 1.0, 0, 0
for z, X, y in stream:
    y1, l1, w1, n1 = card.step(z, X, y)
    y2, l2, w2, n2 = known.step(z, X, y)
    maxdiff = max(maxdiff, abs(y1 - y2), float(np.max(np.abs(np.exp(l1) - np.exp(l2)))))
    scale = max(scale, abs(y1))
    if w1 != w2 or n1 != n2:
        mism += 1
    nw += int(w1)
equiv = bool(mism == 0 and maxdiff < 1e-6 * scale)
print(json.dumps({"equivalent": equiv, "max_abs_diff": float(maxdiff),
                  "notes": f"steps={len(stream)} writes={nw} gate/size mismatches={mism}; card OMP-store vs coupled-dictionary OMP + RAN/ALD-style sparse-residual allocation, FIFO budget d={d}>m={m}"}))
