import json
import numpy as np

# ---------- token protocol (arithmetic encoding, V = 3*d*d + 2) ----------
def enc_demo(a, b, s, d):
    return (a * d + b) * 2 + s

def enc_ask(a, b, d):
    return 2 * d * d + a * d + b

def enc_ans(bit, d):
    return 3 * d * d + bit

def decode(tok, d):
    if tok < 2 * d * d:
        s = tok % 2
        ab = tok // 2
        return ('DEMO', ab // d, ab % d, s)
    if tok < 3 * d * d:
        k = tok - 2 * d * d
        return ('ASK', k // d, k % d, None)
    return ('ANS', tok - 3 * d * d, None, None)

# ---------- card model, written as the card writes it ----------
class Card:
    def __init__(self, d):
        self.d = d
        self.P = list(range(d))
        self.H = [0] * d
        self.M = [1] * d
        self.C = False
    def find(self, i):
        h = 0
        while self.P[i] != i:
            h ^= self.H[i]
            i = self.P[i]
        return i, h
    def step(self, tok):
        d = self.d
        V = 3 * d * d + 2
        kind, a, b, s = decode(tok, d)
        p = np.zeros(V)
        q = 0.5
        if kind == 'DEMO':
            ra, ha = self.find(a)
            rb, hb = self.find(b)
            if ra != rb:
                if self.M[ra] > self.M[rb] or (self.M[ra] == self.M[rb] and ra < rb):
                    u, v = ra, rb
                else:
                    u, v = rb, ra
                self.P[v] = u
                self.H[v] = ha ^ hb ^ s
                self.M[u] = self.M[u] + self.M[v]
            else:
                self.C = self.C or ((ha ^ hb) != s)
        elif kind == 'ASK':
            ra, ha = self.find(a)
            rb, hb = self.find(b)
            z = ha ^ hb
            if ra == rb and not self.C:
                q = 1.0 if z == 1 else 0.0
        p[3 * d * d + 1] = q
        p[3 * d * d] = 1.0 - q
        return p

# ---------- known architecture: textbook Z2-labeled union-find ----------
# Independent implementation: path compression plus union by size.
# Answers depend only on partition and relative parity, so they must agree.
class LabeledUF:
    def __init__(self, d):
        self.parent = list(range(d))
        self.label = [0] * d
        self.size = [1] * d
        self.unsat = False
    def find(self, i):
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        root = i
        acc = 0
        for node in reversed(path):
            acc ^= self.label[node]
            self.label[node] = acc
            self.parent[node] = root
        if path:
            return root, self.label[path[0]]
        return root, 0
    def union(self, a, b, s):
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            if (pa ^ pb) != s:
                self.unsat = True
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.label[rb] = pa ^ pb ^ s
        self.size[ra] += self.size[rb]
    def query(self, a, b):
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if self.unsat or ra != rb:
            return None
        return pa ^ pb

def known_step(m, tok, d):
    V = 3 * d * d + 2
    kind, a, b, s = decode(tok, d)
    q = 0.5
    if kind == 'DEMO':
        m.union(a, b, s)
    elif kind == 'ASK':
        r = m.query(a, b)
        if r is not None:
            q = float(r)
    p = np.zeros(V)
    p[3 * d * d + 1] = q
    p[3 * d * d] = 1.0 - q
    return p

# ---------- independent oracle: BFS over a spanning forest of constraints ----------
class BFSOracle:
    def __init__(self, d):
        self.adj = [[] for _ in range(d)]
        self.bad = False
    def reach(self, a):
        par = {a: 0}
        stack = [a]
        while stack:
            x = stack.pop()
            for (y, s) in self.adj[x]:
                if y not in par:
                    par[y] = par[x] ^ s
                    stack.append(y)
        return par
    def step(self, tok, d):
        V = 3 * d * d + 2
        kind, a, b, s = decode(tok, d)
        q = 0.5
        if kind == 'DEMO':
            par = self.reach(a)
            if b in par:
                if par[b] != s:
                    self.bad = True
            else:
                self.adj[a].append((b, s))
                self.adj[b].append((a, s))
        elif kind == 'ASK':
            par = self.reach(a)
            if b in par and not self.bad:
                q = float(par[b])
        p = np.zeros(V)
        p[3 * d * d + 1] = q
        p[3 * d * d] = 1.0 - q
        return p

rng = np.random.default_rng(12345)
max_diff_known = 0.0
max_diff_oracle = 0.0
steps = 0
determined = 0
contradicted_trials = 0
for trial in range(300):
    d = int(rng.choice([2, 4, 6, 8]))
    n = int(rng.choice([12, 40, 80]))
    noise = 0.0 if trial % 2 == 0 else 0.04
    x = rng.integers(0, 2, size=d)
    card = Card(d)
    known = LabeledUF(d)
    oracle = BFSOracle(d)
    for t in range(n):
        u = rng.random()
        a = int(rng.integers(0, d))
        b = int(rng.integers(0, d))
        if u < 0.45:
            s = int(x[a] ^ x[b])
            if rng.random() < noise:
                s ^= 1
            tok = enc_demo(a, b, s, d)
        elif u < 0.9:
            tok = enc_ask(a, b, d)
        else:
            tok = enc_ans(int(rng.integers(0, 2)), d)
        pc = card.step(tok)
        pk = known_step(known, tok, d)
        po = oracle.step(tok, d)
        max_diff_known = max(max_diff_known, float(np.max(np.abs(pc - pk))))
        max_diff_oracle = max(max_diff_oracle, float(np.max(np.abs(pc - po))))
        if decode(tok, d)[0] == 'ASK' and pc[3 * d * d + 1] != 0.5:
            determined += 1
        steps += 1
    if card.C:
        contradicted_trials += 1

max_abs_diff = max(max_diff_known, max_diff_oracle)
equivalent = bool(max_abs_diff < 1e-6)
notes = 'card vs textbook Z2 labeled union-find (path compression) diff=%g; vs BFS oracle diff=%g; %d steps, %d determined asks, %d trials with C set' % (max_diff_known, max_diff_oracle, steps, determined, contradicted_trials)
print(json.dumps({'equivalent': equivalent, 'max_abs_diff': max_abs_diff, 'notes': notes}))
