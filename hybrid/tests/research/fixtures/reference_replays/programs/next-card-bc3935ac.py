import json, random
import numpy as np

RES = ['<', '?', 'yes', 'no']
NSYM = 20
SYMS = ['s%d' % i for i in range(NSYM)]
VOCAB = RES + SYMS
V = len(VOCAB)
IDX = {v: i for i, v in enumerate(VOCAB)}
EPS = 1e-3

def mul(P, Q):
    return [[P[0][0]*Q[0][0]+P[0][1]*Q[1][0], P[0][0]*Q[0][1]+P[0][1]*Q[1][1]],
            [P[1][0]*Q[0][0]+P[1][1]*Q[1][0], P[1][0]*Q[0][1]+P[1][1]*Q[1][1]]]

# ---------------- card: exactly as written ----------------
def card_C(j):
    return [[1+j, j], [1, 1]]

def card_adj(M):
    return [[M[1][1], -M[0][1]], [-M[1][0], M[0][0]]]

def run_card(stream):
    M = {'RHO': [[1, 0], [0, 1]]}
    K = {'RHO': 0}
    outs = []
    for t in range(len(stream)):
        p = np.full(V, 1.0/V)
        if t >= 2:
            X, o, Y = stream[t-2], stream[t-1], stream[t]
            if X not in RES and Y not in RES:
                if o == '<':
                    if not (X in M or X == Y):
                        if Y not in M:
                            M[Y] = mul(M['RHO'], card_C(K['RHO'])); K['RHO'] += 1; K[Y] = 0
                        j = K[Y]; M[X] = mul(M[Y], card_C(j)); K[Y] = j+1; K[X] = 0
                elif o == '?':
                    p = np.full(V, EPS/V)
                    if X == Y:
                        p[IDX['yes']] += 1-EPS
                    elif X in M and Y in M:
                        R = mul(card_adj(M[Y]), M[X])
                        s = 1 if min(R[0][0], R[0][1], R[1][0], R[1][1]) >= 0 else 0
                        p[IDX['yes' if s else 'no']] += 1-EPS
                    else:
                        p[IDX['yes']] += (1-EPS)/2; p[IDX['no']] += (1-EPS)/2
        outs.append(p)
    return outs, M

# ---------------- known: online Dewey labeling + prefix test, and nested-interval test via mapping ----------------
U = [[1, 1], [0, 1]]
L = [[1, 0], [1, 1]]

def gen(j):  # mapping j -> U^j L built from generators, not from the card's C
    A = [[1, 0], [0, 1]]
    for _ in range(j):
        A = mul(A, U)
    return mul(A, L)

def dewey_to_matrix(path):
    A = [[1, 0], [0, 1]]
    for j in path:
        A = mul(A, gen(j))
    return A

def interval_contains(MY, MX):
    # interval (b/d, a/c); X's interval inside Y's
    ay, by, cy, dy = MY[0][0], MY[0][1], MY[1][0], MY[1][1]
    ax, bx, cx, dx = MX[0][0], MX[0][1], MX[1][0], MX[1][1]
    hi_ok = ax*cy <= ay*cx
    lo_ok = by*dx <= bx*dy
    return hi_ok and lo_ok

def run_known(stream):
    dew = {'RHO': ()}
    cnt = {'RHO': 0}
    outs = []; mism_iv = 0
    for t in range(len(stream)):
        p = np.full(V, 1.0/V)
        if t >= 2:
            X, o, Y = stream[t-2], stream[t-1], stream[t]
            if X not in RES and Y not in RES:
                if o == '<':
                    if X not in dew and X != Y:  # insert as new leaf only
                        if Y not in dew:
                            dew[Y] = (cnt['RHO'],); cnt['RHO'] += 1; cnt[Y] = 0
                        dew[X] = dew[Y] + (cnt[Y],); cnt[Y] += 1; cnt[X] = 0
                elif o == '?':
                    p = np.full(V, EPS/V)
                    if X == Y:
                        p[IDX['yes']] += 1-EPS
                    elif X in dew and Y in dew:
                        dy = dew[Y]; dx = dew[X]
                        s = dx[:len(dy)] == dy
                        s2 = interval_contains(dewey_to_matrix(dy), dewey_to_matrix(dx))
                        if s != s2:
                            mism_iv += 1
                        p[IDX['yes' if s else 'no']] += 1-EPS
                    else:
                        p[IDX['yes']] += (1-EPS)/2; p[IDX['no']] += (1-EPS)/2
        outs.append(p)
    return outs, dew, mism_iv

def make_stream(rng):
    nodes = rng.sample(SYMS, rng.randint(10, 16))
    ntop = rng.randint(1, 3)
    tops = nodes[:ntop]
    par = {}
    for i in range(ntop, len(nodes)):
        par[nodes[i]] = rng.choice(nodes[:i])
    placed = set(tops); pending = [c for c in par]
    stream = []
    while pending:
        avail = [c for c in pending if par[c] in placed]
        c = rng.choice(avail)
        if par[c] in tops and par[c] not in placed:
            pass
        stream += [c, '<', par[c]]; placed.add(c); pending.remove(c)
        r = rng.random()
        if r < 0.5:
            a, b = rng.choice(SYMS), rng.choice(SYMS)
            stream += [a, '?', b]
            if rng.random() < 0.3:
                stream.append(rng.choice(['yes', 'no']))
        elif r < 0.6:
            a = rng.choice(list(placed)); b = rng.choice(SYMS)
            stream += [a, '<', b]  # likely rejected
        elif r < 0.65:
            stream.append(rng.choice(VOCAB))
    for _ in range(12):
        a, b = rng.choice(nodes), rng.choice(nodes)
        stream += [a, '?', b]
    return stream

rng = random.Random(12345)
maxd = 0.0; label_diff = 0; iv_mism = 0; nq = 0; scale = 0.0
for trial in range(200):
    st = make_stream(rng)
    pc, Mc = run_card(st)
    pk, dew, mm = run_known(st)
    iv_mism += mm
    for a, b in zip(pc, pk):
        maxd = max(maxd, float(np.max(np.abs(a-b))))
        scale = max(scale, float(np.max(np.abs(a))))
    if set(Mc.keys()) != set(dew.keys()):
        label_diff = max(label_diff, 10**9)
    else:
        for k in Mc:
            A = Mc[k]; B = dewey_to_matrix(dew[k])
            for i in range(2):
                for j in range(2):
                    label_diff = max(label_diff, abs(A[i][j]-B[i][j]))
    nq += sum(1 for t in range(2, len(st)) if st[t-1] == '?')
tot = max(maxd, float(label_diff), float(iv_mism))
eq = tot < 1e-6*max(scale, 1.0)
print(json.dumps({"equivalent": bool(eq), "max_abs_diff": tot,
  "notes": "200 streams; dist diff %.3g, label diff %d, interval-vs-prefix mismatches %d, ~%d query windows" % (maxd, label_diff, iv_mism, nq)}))
