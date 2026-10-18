import json, math
import numpy as np
from collections import deque, defaultdict

LABELS = [0, 1, 2, 3, 4]

def unif():
    return [1.0 / len(LABELS)] * len(LABELS)

def onehot(y):
    return [1.0 if l == y else 0.0 for l in LABELS]

# ---------------- Card model, written as the card states it ----------------
class Card:
    def __init__(s, d, B):
        s.d = d; s.B = B; s.E = []; s.c = 0
        s.g = 0; s.P = {}; s.v = True
    def rho(s, x):
        return x if s.g == 0 else x % s.g
    def compile(s):
        anchors = {}; g = 0
        for x, y in s.E:  # oldest to newest
            if y not in anchors:
                anchors[y] = x
            g = math.gcd(g, abs(x - anchors[y]))
        s.g = g
        P = {}; v = True
        for y, a in anchors.items():
            r = s.rho(a)
            if r in P and P[r] != y:
                v = False
            P[r] = y
        s.v = v; s.P = P if v else {}
    def step(s, tok):
        if tok[0] == 'DEMO':
            s.E.append((tok[1], tok[2]))
            if len(s.E) > s.d:
                s.E.pop(0)
            s.c = (s.c + 1) % s.B
            if s.c == 0:
                s.compile()
        elif tok[0] == 'COMMIT':
            s.c = 0
            s.compile()
    def read(s, x):
        J = None
        for a, y in reversed(s.E):
            if a == x:
                J = y; break
        if J is not None:
            return onehot(J)
        if s.v:
            r = s.rho(x)
            if r in s.P:
                return onehot(s.P[r])
        return unif()

# ---------------- Known model: hard exact-match attention over a sliding window
# plus the HSW most-specific-lattice learner (1-D closure of pairwise differences)
# with a coset-to-label table. Built independently: pairwise closure, newest
# representative, pairwise coset-disjointness test, and argmax attention.
class Known:
    def __init__(s, d, B):
        s.win = deque(maxlen=d); s.B = B; s.k = 0
        s.L = 0; s.cos = {}; s.ok = True
    def fit(s):
        groups = defaultdict(list)
        for x, y in s.win:
            groups[y].append(x)
        L = 0
        for xs in groups.values():
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    L = math.gcd(L, abs(xs[i] - xs[j]))
        reps = {y: xs[-1] for y, xs in groups.items()}
        ys = list(reps); ok = True
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                diff = reps[ys[i]] - reps[ys[j]]
                same = (diff == 0) if L == 0 else (diff % L == 0)
                if same:
                    ok = False
        s.L = L; s.ok = ok
        s.cos = {((r % L) if L else r): y for y, r in reps.items()} if ok else {}
    def step(s, tok):
        if tok[0] == 'DEMO':
            s.win.append((tok[1], tok[2]))
            s.k += 1
            if s.k % s.B == 0:
                s.fit()
        elif tok[0] == 'COMMIT':
            s.k = 0
            s.fit()
    def read(s, x):
        if len(s.win) > 0:
            keys = np.array([r[0] for r in s.win])
            n = len(keys)
            score = (keys == x).astype(float) + np.arange(n) / (2.0 * n)
            i = int(np.argmax(score))
            if score[i] >= 1.0:
                return onehot(s.win[i][1])
        if s.ok:
            key = (x % s.L) if s.L else x
            if key in s.cos:
                return onehot(s.cos[key])
        return unif()

rng = np.random.default_rng(12345)
max_diff = 0.0
comparisons = 0
stats = {'valid_periodic': 0, 'invalid': 0, 'evictions': 0}

def run(stream, d, B, extra_probes):
    global max_diff, comparisons
    A = Card(d, B); K = Known(d, B)
    for tok in stream:
        A.step(tok); K.step(tok)
        if len(A.E) == d and tok[0] == 'DEMO':
            stats['evictions'] += 1
        probes = set(extra_probes) | set(x for x, _ in A.E)
        if tok[0] in ('DEMO', 'QUERY'):
            probes.add(tok[1])
        for x in sorted(probes):
            a = np.array(A.read(x)); b = np.array(K.read(x))
            max_diff = max(max_diff, float(np.max(np.abs(a - b))))
            comparisons += 1
        if A.g > 0 and A.v:
            stats['valid_periodic'] += 1
        if not A.v:
            stats['invalid'] += 1

# toy example from the card
toy = [('DEMO', 0, 0), ('DEMO', 6, 0), ('DEMO', 1, 1), ('QUERY', 3), ('DEMO', 10, 1),
       ('QUERY', 7), ('DEMO', 2, 2), ('COMMIT',), ('QUERY', 5), ('DEMO', 3, 1), ('COMMIT',), ('QUERY', 11)]
run(toy, 8, 4, list(range(0, 15)))

for trial in range(400):
    d = int(rng.integers(3, 12)); B = int(rng.integers(1, 6))
    mode = trial % 3
    span = int(rng.choice([30, 60, 1000]))
    p = int(rng.integers(1, 6))
    perm = [int(v) for v in rng.permutation(5)[:p]]
    stream = []
    for t in range(12):
        u = rng.random()
        x = int(rng.integers(0, span + 1))
        if u < 0.5:
            if mode == 0 or (mode == 1 and rng.random() < 0.85):
                y = perm[x % p]
            else:
                y = int(rng.integers(0, 5))
            stream.append(('DEMO', x, y))
        elif u < 0.85:
            stream.append(('QUERY', x))
        else:
            stream.append(('COMMIT',))
    stream.append(('COMMIT',))
    probes = [int(v) for v in rng.integers(0, span + 1, size=10)]
    run(stream, d, B, probes)

equiv = max_diff < 1e-6
notes = 'compared %d readouts over 401 streams; states valid&g>0=%d, invalid=%d, full-window demos=%d' % (
    comparisons, stats['valid_periodic'], stats['invalid'], stats['evictions'])
print(json.dumps({'equivalent': bool(equiv), 'max_abs_diff': max_diff, 'notes': notes}))
