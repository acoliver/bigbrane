import json, random
from collections import deque
import numpy as np

CTRL = ['D', 'Q', 'L', 'R', '?']

def vidx(z, d):
    return z if isinstance(z, int) else d + CTRL.index(z)

# ---------------- Card implementation (as written) ----------------
def card_run(T, U, tokens, d):
    V = d + 5
    F = [None] * d
    c = False
    b = 'idle'
    r = 0
    outs = []
    for z in tokens:
        e = None
        num = isinstance(z, int) and 0 <= z < d
        if b == 'idle':
            b = 'demo_source' if z == 'D' else ('query_source' if z == 'Q' else 'idle')
        elif b == 'demo_source':
            if num:
                r = z; b = 'demo_target'
            else:
                b = 'idle'
        elif b == 'demo_target':
            if num:
                e = (r, z); b = 'idle'
            else:
                b = 'idle'
        elif b == 'query_source':
            if num:
                r = z; b = 'word'
            else:
                b = 'idle'
        elif b == 'word':
            if z == 'L':
                r = T[0][r]
            elif z == 'R':
                r = T[1][r]
            elif z == '?':
                b = 'answer'
            else:
                b = 'idle'
        elif b == 'answer':
            b = 'idle'
        if e is not None and not c:
            W = deque([e])
            while W and not c:
                x, y = W.popleft()
                if F[x] is None:
                    F[x] = y
                    W.append((T[0][x], U[0][y]))
                    W.append((T[1][x], U[1][y]))
                elif F[x] == y:
                    pass
                else:
                    c = True
                    W.clear()
        p = np.zeros(V)
        if b == 'answer':
            if (not c) and F[r] is not None:
                p[F[r]] = 1.0
            else:
                p[:d] = 1.0 / d
        else:
            p[:] = 1.0 / V
        outs.append(p)
    return np.array(outs)

# ------- Known model: automaton-homomorphism extension via product-automaton reachability -------
# Pairs relation R subset S x S' closed under (x,y)->(delta1(x,a),delta2(y,a)); h is a homomorphism
# iff R is functional. DFS stack, visited-pair set (different traversal order and data structure).
def known_run(T, U, tokens, d):
    V = d + 5
    delta1 = {(i, a): T[k][i] for k, a in enumerate('LR') for i in range(d)}
    delta2 = {(j, a): U[k][j] for k, a in enumerate('LR') for j in range(d)}
    R = set()
    img = {}
    bad = False
    # tokenizer DFA as an explicit transition table: (state, class) -> (state, action)
    def cls(z):
        if isinstance(z, int): return 'num'
        return z
    table = {
        ('idle', 'D'): ('ds', None), ('idle', 'Q'): ('qs', None),
        ('ds', 'num'): ('dt', 'store'), ('dt', 'num'): ('idle', 'seed'),
        ('qs', 'num'): ('w', 'store'), ('w', 'L'): ('w', 'L'), ('w', 'R'): ('w', 'R'),
        ('w', '?'): ('ans', None),
    }
    st = 'idle'; reg = 0
    outs = []
    for z in tokens:
        st2, act = table.get((st, cls(z)), ('idle', None))
        if act == 'store':
            reg = z
        elif act in ('L', 'R'):
            reg = delta1[(reg, act)]
        elif act == 'seed':
            stack = [(reg, z)]
            while stack:
                pr = stack.pop()
                if pr in R:
                    continue
                R.add(pr)
                x, y = pr
                img.setdefault(x, set()).add(y)
                if len(img[x]) > 1:
                    bad = True
                for a in 'LR':
                    stack.append((delta1[(x, a)], delta2[(y, a)]))
        st = st2
        p = np.zeros(V)
        if st == 'ans':
            if (not bad) and reg in img:
                p[next(iter(img[reg]))] = 1.0
            else:
                p[:d] = 1.0 / d
        else:
            p[:] = 1.0 / V
        outs.append(p)
    return np.array(outs)

# ---------------- random inputs ----------------
rng = random.Random(12345)
maxdiff = 0.0
n_tok = 0; n_point = 0; n_unif_ans = 0
for trial in range(200):
    d = rng.choice([4, 6, 8, 12])
    T = [[rng.randrange(d) for _ in range(d)] for _ in range(2)]
    pi = list(range(d)); rng.shuffle(pi)
    inv = [0] * d
    for i, v in enumerate(pi): inv[v] = i
    if rng.random() < 0.6:
        U = [[pi[T[a][inv[j]]] for j in range(d)] for a in range(2)]
    else:
        U = [[rng.randrange(d) for _ in range(d)] for _ in range(2)]
    toks = []
    pwrong = rng.choice([0.0, 0.0, 0.1, 0.4])
    while len(toks) < 150:
        u = rng.random()
        if u < 0.3:
            i = rng.randrange(d)
            j = pi[i] if rng.random() > pwrong else rng.randrange(d)
            toks += ['D', i, j]
        elif u < 0.9:
            toks += ['Q', rng.randrange(d)] + [rng.choice('LR') for _ in range(rng.randrange(0, 8))] + ['?', rng.randrange(d)]
        else:
            toks.append(rng.choice(CTRL + [rng.randrange(d)]))
    a = card_run(T, U, toks, d)
    b = known_run(T, U, toks, d)
    maxdiff = max(maxdiff, float(np.max(np.abs(a - b))))
    n_tok += len(toks)
    n_point += int(np.sum(np.max(a, axis=1) == 1.0))
    n_unif_ans += int(np.sum(np.isclose(a[:, 0], 1.0 / d) & np.isclose(a[:, -1], 0.0)))

scale = 1.0
print(json.dumps({"equivalent": bool(maxdiff < 1e-6 * scale), "max_abs_diff": maxdiff,
                  "notes": "200 trials, %d tokens; card FIFO closure vs product-automaton pair reachability + functionality test; point-mass answers=%d, uniform-answer outputs=%d" % (n_tok, n_point, n_unif_ans)}))
