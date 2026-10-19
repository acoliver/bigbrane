import json
import numpy as np

def run(p, episodes, rng):
    V = p + 1
    def point(z):
        return (1, 0) if z == p else (z % p, 1)
    def bracket(h, k):
        return (h[0] * k[1] - h[1] * k[0]) % p
    def decode(h):
        h0, h1 = h[0] % p, h[1] % p
        return p if h1 == 0 else (h0 * pow(h1, -1, p)) % p
    def frame(u1, u2, u3):
        e1, e2, e3 = point(u1), point(u2), point(u3)
        ca = bracket(e3, e1)
        cb = bracket(e3, e2)
        return [[ca * e2[1] % p, (-ca * e2[0]) % p], [cb * e1[1] % p, (-cb * e1[0]) % p]]
    def inv(M):
        a, b = M[0]
        c, d = M[1]
        s = pow((a * d - b * c) % p, -1, p)
        return [[d * s % p, (-b * s) % p], [(-c * s) % p, a * s % p]]
    def mul(X, Y):
        return [[sum(X[i][k] * Y[k][j] for k in range(2)) % p for j in range(2)] for i in range(2)]
    def act(M, z):
        e = point(z)
        return decode((M[0][0] * e[0] + M[0][1] * e[1], M[1][0] * e[0] + M[1][1] * e[1]))

    class Card:
        def __init__(self, mode):
            self.A = [None] * 3
            self.r = 0
            self.phi = 0
            self.s = 0
            self.mode = mode
        def T(self, z):
            al = [self.A[i][0] for i in range(3)]
            be = [self.A[i][1] for i in range(3)]
            return act(mul(inv(frame(*be)), frame(*al)), z)
        def astar(self):
            if 0 <= self.r < 3:
                used = {self.A[i][0] for i in range(self.r)}
                return min(set(range(4)) - used)
            return 0
        def dist(self):
            P = np.zeros(V)
            if self.phi == 0:
                if self.mode == 'active':
                    P[self.astar()] = 1.0
                else:
                    P[:] = 1.0 / V
                return P
            if self.r == -1:
                P[:] = 1.0 / V
                return P
            if self.r == 3:
                P[self.T(self.s)] = 1.0
                return P
            for i in range(self.r):
                if self.A[i][0] == self.s:
                    P[self.A[i][1]] = 1.0
                    return P
            bs = {self.A[i][1] for i in range(self.r)}
            for v in range(V):
                if v not in bs:
                    P[v] = 1.0 / (V - self.r)
            return P
        def U(self, x, y):
            r = self.r
            if r == -1:
                return
            if r < 3:
                for i in range(r):
                    if self.A[i][0] == x:
                        if self.A[i][1] != y:
                            self.r = -1
                        return
                if y in {self.A[i][1] for i in range(r)}:
                    self.r = -1
                    return
                self.A[r] = (x, y)
                self.r = r + 1
                return
            if self.T(x) != y:
                self.r = -1
        def consume(self, w):
            if self.phi == 0:
                self.phi, self.s = 1, w
            else:
                self.U(self.s, w)
                self.phi, self.s = 0, 0

    # Known model: explicit version space over enumerated PGL(2,p), Bayes posterior predictive,
    # max-information-gain membership query (smallest-label tie-break).
    Gs = set()
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    if (a * d - b * c) % p:
                        Gs.add(tuple(act([[a, b], [c, d]], z) for z in range(V)))
    G = list(Gs)
    group_ok = (len(G) == V * (V - 1) * (V - 2))

    class VS:
        def __init__(self, mode):
            self.H = list(G)
            self.phi = 0
            self.s = 0
            self.mode = mode
        def dist(self):
            P = np.zeros(V)
            if self.phi == 0:
                if self.mode == 'passive':
                    P[:] = 1.0 / V
                    return P
                if not self.H:
                    P[0] = 1.0
                    return P
                gains = []
                for x in range(V):
                    c = np.bincount([g[x] for g in self.H], minlength=V) / len(self.H)
                    nz = c[c > 0]
                    gains.append(float(-(nz * np.log(nz)).sum()))
                m = max(gains)
                x = min(i for i in range(V) if gains[i] >= m - 1e-9)
                P[x] = 1.0
                return P
            if not self.H:
                P[:] = 1.0 / V
                return P
            c = np.bincount([g[self.s] for g in self.H], minlength=V).astype(float)
            return c / len(self.H)
        def consume(self, w):
            if self.phi == 0:
                self.phi, self.s = 1, w
            else:
                self.H = [g for g in self.H if g[self.s] == w]
                self.phi, self.s = 0, 0

    expected_sizes = {0: V * (V - 1) * (V - 2), 1: (V - 1) * (V - 2), 2: V - 2, 3: 1}
    maxdiff = 0.0
    mism = 0
    steps = 0
    for ep in range(episodes):
        mode = 'active' if rng.random() < 0.5 else 'passive'
        while True:
            Mh = rng.integers(0, p, (2, 2))
            if (int(Mh[0, 0]) * int(Mh[1, 1]) - int(Mh[0, 1]) * int(Mh[1, 0])) % p:
                break
        hidden = [[int(Mh[0, 0]), int(Mh[0, 1])], [int(Mh[1, 0]), int(Mh[1, 1])]]
        noise = float(rng.choice([0.0, 0.0, 0.15, 1.0]))
        card = Card(mode)
        vs = VS(mode)
        n = int(rng.choice([12, 16]))
        past = []
        q = 0
        for t in range(n):
            Pc = card.dist()
            Pk = vs.dist()
            maxdiff = max(maxdiff, float(np.abs(Pc - Pk).max()))
            steps += 1
            if t % 2 == 0:
                if mode == 'active':
                    w = int(np.argmax(Pc))
                elif past and rng.random() < 0.3:
                    w = int(past[int(rng.integers(0, len(past)))])
                else:
                    w = int(rng.integers(0, V))
                q = w
                past.append(w)
            else:
                w = act(hidden, q) if rng.random() >= noise else int(rng.integers(0, V))
            card.consume(w)
            vs.consume(w)
            if t % 2 == 1:
                if (card.r == -1) != (len(vs.H) == 0):
                    mism += 1
                if card.r >= 0 and len(vs.H) != expected_sizes[card.r]:
                    mism += 1
        Pc = card.dist()
        Pk = vs.dist()
        maxdiff = max(maxdiff, float(np.abs(Pc - Pk).max()))
    return maxdiff, mism, group_ok, steps

rng = np.random.default_rng(12345)
tot_diff = 0.0
tot_mism = 0
all_group = True
tot_steps = 0
for p, eps in ((5, 300), (7, 300), (11, 120)):
    d, m, g, s = run(p, eps, rng)
    tot_diff = max(tot_diff, d)
    tot_mism += m
    all_group = all_group and g
    tot_steps += s
equiv = bool(tot_diff < 1e-6 and tot_mism == 0 and all_group)
notes = 'card FSM vs enumerated PGL(2,p) version space (p=5,7,11), passive+active, noisy/clean streams; %d steps compared; rejection/size mismatches=%d; group_size_ok=%s' % (tot_steps, tot_mism, all_group)
print(json.dumps({'equivalent': equiv, 'max_abs_diff': tot_diff, 'notes': notes}))
