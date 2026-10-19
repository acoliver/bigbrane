import json
import numpy as np

lam, s2, eta, R = 1e-3, 0.01, 0.5, 4

def gate(w, u, v):
    return 1.0 / (1.0 + np.exp(-(w @ np.concatenate([u, v]))))

# ---------------- card: equations exactly as written ----------------
def card_run(seq, Phi, w):
    d = Phi.shape[1]
    P = np.zeros((R, d, d)); K = np.zeros((R, d, d)); N = np.zeros(R)
    r, z = 0, 0
    logits, assign, gaps = [], [], []
    for t in range(1, len(seq)):
        u, v = Phi[seq[t - 1]], Phi[seq[t]]
        beta = gate(w, u, v)
        if beta >= 0.5:
            S = []
            for j in range(r):
                c = 1.0 + u @ P[j] @ u
                e = v - K[j].T @ u
                S.append(np.log(N[j]) - e @ e / (2 * s2 * c))
            if r < R:
                S.append(np.log(eta) - v @ v / (2 * s2 * (1.0 + u @ u / lam)))
            S = np.array(S)
            z = int(np.argmax(S))
            if len(S) > 1:
                ss = np.sort(S); gaps.append(float(ss[-1] - ss[-2]))
            if z == r:
                P[z] = np.eye(d) / lam; K[z] = 0.0; N[z] = 0.0; r += 1
            Pu = P[z] @ u
            P[z] = P[z] - beta * np.outer(Pu, Pu) / (1.0 + beta * (u @ Pu))
            K[z] = K[z] + beta * np.outer(P[z] @ u, v - u @ K[z])
            N[z] += 1
            assign.append(z)
        else:
            assign.append(-1)
        y = K[z].T @ Phi[seq[t]]
        logits.append(Phi @ y)
    return np.array(logits), assign, gaps

# ---------------- known form: SUGS greedy CRP mixture of conjugate Bayesian linear regressions ----------------
# Each cluster stores its raw assigned data; posterior is recomputed in batch (no RLS recursion).
# Prior per output column: N(0, (s2/lam) I); noise variance s2/beta (beta = observation weight).
# Score = log CRP count + log posterior predictive (constants dropped; log-det dropped unless logdet=True).
def sugs_run(seq, Phi, w, logdet=False):
    d = Phi.shape[1]
    clusters = []
    z = 0
    logits, assign = [], []
    def post(cl):
        A = lam * np.eye(d); C = np.zeros((d, d))
        for (uu, vv, bb) in cl:
            A += bb * np.outer(uu, uu); C += bb * np.outer(uu, vv)
        return A, np.linalg.solve(A, C)
    for t in range(1, len(seq)):
        u, v = Phi[seq[t - 1]], Phi[seq[t]]
        b = gate(w, u, v)
        if b >= 0.5:
            S = []
            for cl in clusters:
                A, M = post(cl)
                c = 1.0 + u @ np.linalg.solve(A, u)
                e = v - M.T @ u
                s = np.log(len(cl)) - e @ e / (2 * s2 * c)
                if logdet:
                    s -= 0.5 * d * np.log(c)
                S.append(s)
            if len(clusters) < R:
                c0 = 1.0 + u @ u / lam
                s = np.log(eta) - v @ v / (2 * s2 * c0)
                if logdet:
                    s -= 0.5 * d * np.log(c0)
                S.append(s)
            z = int(np.argmax(np.array(S)))
            if z == len(clusters):
                clusters.append([])
            clusters[z].append((u, v, b))
            assign.append(z)
        else:
            assign.append(-1)
        if z < len(clusters) and len(clusters[z]) > 0:
            M = post(clusters[z])[1]
        else:
            M = np.zeros((d, d))
        logits.append(Phi @ (M.T @ Phi[seq[t]]))
    return np.array(logits), assign

# ---------------- inputs ----------------
rng = np.random.default_rng(12345)

def fourier_phi(Vc, d):
    m = np.arange(1, d // 2 + 1)
    ang = 2 * np.pi * np.outer(np.arange(Vc), m) / Vc
    Phi = np.zeros((Vc + 2, d))
    Phi[:Vc] = np.concatenate([np.cos(ang), np.sin(ang)], axis=1) / np.sqrt(d // 2)
    Phi[Vc:] = rng.normal(size=(2, d)) * 0.5
    return Phi

def switched_seq(Vc):
    BOS, SEP = Vc, Vc + 1
    shifts = rng.choice(np.arange(1, Vc), size=5, replace=False)
    nblocks = int(rng.integers(2, 7))
    seq = [BOS]
    for _ in range(nblocks):
        s = int(shifts[rng.integers(0, 5)])
        for _ in range(int(rng.integers(1, 7))):
            a = int(rng.integers(0, Vc))
            seq += [a, (a + s) % Vc, SEP]
    seq.append(int(rng.integers(0, Vc)))
    return seq

trials = []
for _ in range(60):
    d = 4; Vc = 16
    Phi = fourier_phi(Vc, d)
    w = rng.normal(size=2 * d) * float(rng.choice([0.0, 0.5, 3.0]))
    trials.append((switched_seq(Vc), Phi, w))
for _ in range(60):
    d = 4; V = 10
    Phi = rng.normal(size=(V, d)) / np.sqrt(d)
    w = rng.normal(size=2 * d) * float(rng.choice([0.0, 0.5, 3.0]))
    seq = [int(x) for x in rng.integers(0, V, size=12)]
    trials.append((seq, Phi, w))

max_diff = 0.0; max_rel = 0.0; mism = 0; mism_ld = 0; nsteps = 0; min_gap = np.inf; nbirth_multi = 0
for seq, Phi, w in trials:
    Lc, Ac, gaps = card_run(seq, Phi, w)
    Lk, Ak = sugs_run(seq, Phi, w, logdet=False)
    _, Ak2 = sugs_run(seq, Phi, w, logdet=True)
    diff = float(np.max(np.abs(Lc - Lk)))
    scale = max(1.0, float(np.max(np.abs(Lc))))
    max_diff = max(max_diff, diff); max_rel = max(max_rel, diff / scale)
    mism += sum(1 for a, b in zip(Ac, Ak) if a != b)
    mism_ld += sum(1 for a, b in zip(Ac, Ak2) if a != b)
    nsteps += len(Ac)
    if gaps:
        min_gap = min(min_gap, min(gaps))
    if max(Ac) >= 1:
        nbirth_multi += 1

equiv = bool(max_rel < 1e-6 and mism == 0)
notes = ('card vs SUGS (log-det dropped, R-truncated, last-cluster readout) over %d trials/%d steps; routing mismatches=%d; max rel diff=%.2e; trials with >=2 modes=%d; min top-2 score gap=%.3g; with full predictive log-det SUGS would differ in %d routing steps' % (len(trials), nsteps, mism, max_rel, nbirth_multi, min_gap, mism_ld))
print(json.dumps({'equivalent': equiv, 'max_abs_diff': max_diff, 'notes': notes}))
