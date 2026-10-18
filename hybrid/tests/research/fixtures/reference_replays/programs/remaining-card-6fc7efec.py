import json
import numpy as np

rng = np.random.default_rng(12345)
p = 7
V = p + 5
ROW0, ROW1, COL0, MUL, ASK = p, p + 1, p + 2, p + 3, p + 4
U = np.full(V, 1.0 / V)

def make_isotope():
    ah, bh, gh = rng.permutation(p), rng.permutation(p), rng.permutation(p)
    ginv = np.empty(p, dtype=int)
    ginv[gh] = np.arange(p)
    return lambda a, b: int(ginv[(ah[a] + bh[b]) % p])

# ---------------- Card implementation (as written) ----------------
class Card:
    def __init__(self):
        self.D = {0: [None] * p, 1: [None] * p, 2: [None] * p}
        self.Z = None
        self.valid = False
        self.mode = 'expr'
        self.r = 0
        self.S = []
        self.rejected = False
        self.compiles = 0

    def reject(self):
        self.valid = False
        self.Z = None
        self.D = {0: [None] * p, 1: [None] * p, 2: [None] * p}
        self.S = []
        self.rejected = True

    def compile(self):
        D0, D1, Dc = self.D[0], self.D[1], self.D[2]
        if sorted(D0) != list(range(p)) or sorted(D1) != list(range(p)):
            return self.reject()
        P = [0] * p
        for b in range(p):
            P[D0[b]] = D1[b]
        gamma = [-1] * p
        ginv = [-1] * p
        y = D0[0]
        for j in range(p):
            if gamma[y] != -1:
                return self.reject()
            gamma[y] = j
            ginv[j] = y
            y = P[y]
        if y != D0[0]:
            return self.reject()
        alpha = [gamma[Dc[a]] for a in range(p)]
        beta = [gamma[D0[b]] for b in range(p)]
        if sorted(alpha) != list(range(p)) or sorted(beta) != list(range(p)):
            return self.reject()
        for b in range(p):
            if ginv[(alpha[0] + beta[b]) % p] != D0[b]:
                return self.reject()
            if ginv[(alpha[1] + beta[b]) % p] != D1[b]:
                return self.reject()
        for a in range(p):
            if ginv[(alpha[a] + beta[0]) % p] != Dc[a]:
                return self.reject()
        self.Z = (P, gamma, ginv, alpha, beta)
        self.valid = True
        self.compiles += 1

    def contract(self, u, v):
        _, _, ginv, alpha, beta = self.Z
        m = np.zeros(p)
        for a in range(p):
            for b in range(p):
                m[ginv[(alpha[a] + beta[b]) % p]] += u[a] * v[b]
        return m

    def step(self, x):
        if x in (ROW0, ROW1, COL0):
            if self.mode in ('row0', 'row1', 'col0') and self.r < p:
                self.reject()
                return U.copy()
            if self.valid:
                self.valid = False
                self.Z = None
                self.D = {0: [None] * p, 1: [None] * p, 2: [None] * p}
            self.mode = {ROW0: 'row0', ROW1: 'row1', COL0: 'col0'}[x]
            self.r = 0
            return U.copy()
        if self.mode in ('row0', 'row1', 'col0'):
            if x >= p:
                self.reject()
                return U.copy()
            idx = {'row0': 0, 'row1': 1, 'col0': 2}[self.mode]
            self.D[idx][self.r] = x
            self.r += 1
            if self.r == p:
                self.mode = 'expr'
                D = self.D
                if all(v is not None for k in D for v in D[k]):
                    if D[0][0] != D[2][0] or D[1][0] != D[2][1]:
                        self.reject()
                    else:
                        self.compile()
            return U.copy()
        if x < p:
            e = np.zeros(p)
            e[x] = 1.0
            self.S.append(e)
            return U.copy()
        if x == MUL:
            if len(self.S) < 2 or not self.valid:
                self.reject()
                return U.copy()
            v = self.S.pop()
            u = self.S.pop()
            self.S.append(self.contract(u, v))
            return U.copy()
        if x == ASK:
            if self.valid and len(self.S) == 1:
                q = np.zeros(V)
                q[:p] = self.S[-1]
                self.S = []
                return q
            return U.copy()
        return U.copy()

# ------------- Known architecture: shift-reduce RNTN (bilinear only) -------------
class ShiftReduceRNTN:
    def __init__(self):
        self.E = np.eye(p)  # identity leaf embedding
        self.T = None
        self.stack = []

    def set_params_from_mapping(self, Z):
        # mapping: T[c,a,b] = I[c = gamma_inv((alpha[a]+beta[b]) mod p)]
        _, _, ginv, alpha, beta = Z
        T = np.zeros((p, p, p))
        for a in range(p):
            for b in range(p):
                T[ginv[(alpha[a] + beta[b]) % p], a, b] = 1.0
        self.T = T

    def reduce(self, hl, hr):
        return np.einsum('cab,a,b->c', self.T, hl, hr)

    def step(self, x):
        if x < p:
            self.stack.append(self.E[:, x].copy())  # SHIFT
            return U.copy()
        if x == MUL:
            hr = self.stack.pop()
            hl = self.stack.pop()
            self.stack.append(self.reduce(hl, hr))  # REDUCE
            return U.copy()
        if x == ASK:
            q = np.zeros(V)
            q[:p] = self.stack[-1]
            self.stack = []
            return q
        return U.copy()

def rand_postfix(nleaves):
    if nleaves == 1:
        return [int(rng.integers(p))]
    k = int(rng.integers(1, nleaves))
    return rand_postfix(k) + rand_postfix(nleaves - k) + [MUL]

def calib_tokens(op):
    t = [ROW0] + [op(0, b) for b in range(p)]
    t += [ROW1] + [op(1, b) for b in range(p)]
    t += [COL0] + [op(a, 0) for a in range(p)]
    return t

card = Card()
known = ShiftReduceRNTN()
maxdiff = 0.0
steps = 0
fail_notes = []

for block in range(3):
    op = make_isotope()
    for x in calib_tokens(op):
        qc = card.step(x)
        qk = U.copy()  # known model during calibration emits uniform (protocol)
        maxdiff = max(maxdiff, float(np.max(np.abs(qc - qk))))
    if not card.valid:
        fail_notes.append('calibration rejected')
        continue
    known.set_params_from_mapping(card.Z)
    known.stack = []
    for e in range(6):
        nl = int(rng.integers(1, 13))
        expr = rand_postfix(nl) + [ASK]
        for x in expr:
            qc = card.step(x)
            qk = known.step(x)
            maxdiff = max(maxdiff, float(np.max(np.abs(qc - qk))))
            if x != ASK:
                if len(card.S) != len(known.stack):
                    fail_notes.append('stack size mismatch')
                    maxdiff = max(maxdiff, 1.0)
                else:
                    for a_, b_ in zip(card.S, known.stack):
                        maxdiff = max(maxdiff, float(np.max(np.abs(a_ - b_))))
            steps += 1
    # soft (non-one-hot) messages: contraction vs bilinear reduce
    for _ in range(20):
        u = rng.random(p); u /= u.sum()
        v = rng.random(p); v /= v.sum()
        maxdiff = max(maxdiff, float(np.max(np.abs(card.contract(u, v) - known.reduce(u, v)))))

scale = 1.0
equiv = (maxdiff < 1e-6 * scale) and not fail_notes and card.compiles == 3
notes = 'card shift/contract vs RNTN tensor T from mapping; 3 calibration sets, %d expr steps, plus soft messages' % steps
if fail_notes:
    notes += '; ' + ','.join(sorted(set(fail_notes)))
print(json.dumps({'equivalent': bool(equiv), 'max_abs_diff': maxdiff, 'notes': notes}))
