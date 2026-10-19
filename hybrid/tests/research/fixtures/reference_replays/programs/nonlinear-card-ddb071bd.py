import numpy as np, json

rng = np.random.default_rng(12345)

# ---------------- card equations, as written ----------------
def card_run(Theta, seq, K, tau, beta, u):
    E = np.exp(1j * Theta)
    V, d = E.shape
    P, m, alpha = [], [], None
    outs = []
    stats = {'merge': 0, 'evict': 0}
    for t in range(len(seq)):
        if t >= 1:
            w = np.conj(E[seq[t - 1]]) * E[seq[t]]
            c = [float(np.real(np.vdot(Pj, w))) / d for Pj in P]
            j = int(np.argmax(c)) if c else -1
            if j >= 0 and c[j] >= tau:
                z = m[j] * P[j] + w
                az = np.abs(z)
                P[j] = np.where(az > 0, z / np.where(az > 0, az, 1.0), P[j])
                m[j] += 1
                alpha = j
                stats['merge'] += 1
            else:
                if len(P) == K:
                    jev = min([i for i in range(len(P)) if i != alpha], key=lambda i: (m[i], i))
                    del P[jev]
                    del m[jev]
                    if alpha is not None and alpha > jev:
                        alpha -= 1
                    stats['evict'] += 1
                P.append(w.copy())
                m.append(1)
        if alpha is None:
            s = np.zeros(V)
        else:
            s = np.real((P[alpha] * E[seq[t]]) @ np.conj(E).T) / d
        lg = u + beta * s
        p = np.exp(lg - lg.max())
        p /= p.sum()
        outs.append(p)
    return np.array(outs), stats

# ------- known form: ART vigilance categorization / leader clustering with a bounded
# category pool under LFU-then-FIFO replacement (pinning the last resonating node),
# fed HRR relation spectra, with an HRR bind + clean-up readout -------
class ARTPool:
    def __init__(self, d, capacity, vigilance):
        self.d = d
        self.cap = capacity
        self.rho = vigilance
        self.cats = {}        # id -> dict(proto, n, birth)
        self.clock = 0
        self.last = None      # id of the most recently resonating category

    def choice(self, I):
        # choice function T_j = (1/d) Re <proto_j, I>, scanned oldest-first, strict > (ties -> oldest)
        best, bestT = None, None
        for cid in sorted(self.cats, key=lambda k: self.cats[k]['birth']):
            pr = self.cats[cid]['proto']
            T = float(np.sum(pr.real * I.real + pr.imag * I.imag)) / self.d
            if bestT is None or T > bestT:
                best, bestT = cid, T
        return best, bestT

    def present(self, I):
        win, T = self.choice(I)
        if win is not None and T >= self.rho:
            # resonance: count-weighted prototype update projected to the unit torus
            c = self.cats[win]
            z = c['n'] * c['proto'] + I
            r = np.hypot(z.real, z.imag)
            safe = np.where(r > 0, r, 1.0)
            c['proto'] = np.where(r > 0, z / safe, c['proto'])
            c['n'] += 1
            self.last = win
        else:
            # mismatch reset: recruit uncommitted node, replacing LFU (FIFO tie-break), last pinned
            if len(self.cats) == self.cap:
                cand = [k for k in self.cats if k != self.last]
                victim = min(cand, key=lambda k: (self.cats[k]['n'], self.cats[k]['birth']))
                del self.cats[victim]
            self.cats[self.clock] = {'proto': I.copy(), 'n': 1, 'birth': self.clock}
            self.clock += 1

def art_run(Theta, seq, K, tau, beta, u):
    E = np.cos(Theta) + 1j * np.sin(Theta)
    V, d = E.shape
    pool = ARTPool(d, K, tau)
    outs = []
    for t in range(len(seq)):
        if t >= 1:
            I = E[seq[t]] / E[seq[t - 1]]   # HRR unbind = circular correlation (unit modulus)
            pool.present(I)
        if pool.last is None:
            s = np.zeros(V)
        else:
            bound = pool.cats[pool.last]['proto'] * E[seq[t]]
            s = np.array([np.real(np.sum(bound * np.conj(E[y]))) for y in range(V)]) / d
        lg = u + beta * s
        lg = lg - np.max(lg)
        p = np.exp(lg) / np.sum(np.exp(lg))
        outs.append(p)
    return np.array(outs)

# ------- parent's linear-attention (H3, phase-normalized) form, for contrast only -------
def parent_run(Theta, seq, beta, u):
    E = np.exp(1j * Theta)
    V, d = E.shape
    R = np.zeros(d, complex)
    outs = []
    for t in range(len(seq)):
        if t >= 1:
            R = R + np.conj(E[seq[t - 1]]) * E[seq[t]]
        g = R / (np.abs(R) + 1e-12)
        s = np.real((g * E[seq[t]]) @ np.conj(E).T) / d
        lg = u + beta * s
        p = np.exp(lg - lg.max())
        p /= p.sum()
        outs.append(p)
    return np.array(outs)

# ---------------- inputs ----------------
def lattice_theta(rng):
    Th = np.zeros((65, 64))
    v = np.arange(64)[:, None]
    w = np.arange(64)[None, :]
    Th[:64] = 2 * np.pi * v * w / 64
    Th[64] = rng.uniform(0, 2 * np.pi, 64)
    return Th

def capacity_seq(rng, R):
    shifts = rng.choice(np.arange(1, 64), R, replace=False)
    a_s = rng.choice(64, 2 * R + 1, replace=False)
    seq, k = [], 0
    for r in range(R):
        for _ in range(2):
            a = int(a_s[k]); k += 1
            seq += [a, (a + int(shifts[r])) % 64, 64]
    seq.append(int(a_s[k]))
    return seq

def reinstate_seq(rng, L, control=False):
    sh = rng.choice(np.arange(1, 64), 3, replace=False)
    dA, dB, dC = (int(x) for x in sh)
    a_s = rng.choice(64, 2 + L + 2, replace=False)
    deltas = [dA, dA] + [dB] * L + [dC if control else dA]
    seq = []
    for i, dl in enumerate(deltas):
        a = int(a_s[i])
        seq += [a, (a + dl) % 64, 64]
    seq.append(int(a_s[-1]))
    return seq

cases = []
for _ in range(15):
    Th = lattice_theta(rng)
    R = int(rng.integers(2, 9))
    cases.append((Th, capacity_seq(rng, R), 8, 0.5, 8.0, np.zeros(65)))
for _ in range(10):
    Th = lattice_theta(rng)
    L = int(rng.choice([6, 20]))
    cases.append((Th, reinstate_seq(rng, L, bool(rng.integers(2))), 16, 0.5, 8.0, np.zeros(65)))
for _ in range(60):
    V = int(rng.integers(3, 7)); d = int(rng.integers(4, 12))
    Th = rng.uniform(0, 2 * np.pi, (V, d))
    seq = [int(x) for x in rng.integers(0, V, 12)]
    K = int(rng.integers(2, 6)); tau = float(rng.uniform(0.2, 0.8))
    beta = float(rng.uniform(1, 10)); u = rng.normal(0, 1, V)
    cases.append((Th, seq, K, tau, beta, u))
for _ in range(30):
    V = 12; d = 16
    Th = rng.uniform(0, 2 * np.pi, (V, d))
    seq = [int(x) for x in rng.integers(0, V, 12)]
    cases.append((Th, seq, int(rng.integers(2, 6)), float(rng.uniform(0.1, 0.6)), float(rng.uniform(1, 10)), rng.normal(0, 1, V)))

maxdiff, scale, merges, evicts, pdiff = 0.0, 0.0, 0, 0, 0.0
for Th, seq, K, tau, beta, u in cases:
    pc, st = card_run(Th, seq, K, tau, beta, u)
    pa = art_run(Th, seq, K, tau, beta, u)
    pp = parent_run(Th, seq, beta, u)
    maxdiff = max(maxdiff, float(np.max(np.abs(pc - pa))))
    scale = max(scale, float(np.max(np.abs(pc))))
    pdiff = max(pdiff, float(np.max(np.abs(pc - pp))))
    merges += st['merge']; evicts += st['evict']

eq = bool(maxdiff < 1e-6 * max(scale, 1e-12))
print(json.dumps({'equivalent': eq, 'max_abs_diff': maxdiff,
  'notes': 'ART/leader-clustering+LFU-FIFO pool+HRR bind vs card on %d seqs, every step; merges=%d evictions=%d; parent H3 phase-norm form differs by %.3f (repair fails)' % (len(cases), merges, evicts, pdiff)}))
