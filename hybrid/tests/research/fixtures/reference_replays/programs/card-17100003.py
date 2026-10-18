import json
import numpy as np

rng = np.random.default_rng(0)
N = 12  # maximum input length

# ---------------- card: delta and readout exactly as written ----------------
def card_step(q, c1, c2, x):
    if q == 'dead':
        return q, c1, c2
    if x == 'a' and q == 'a':
        return 'a', c1 + 1, c2
    if x == 'b' and q in ('a', 'b'):
        if c1 > 0:
            return 'b', c1 - 1, c2 + 1
        return 'dead', c1, c2
    if x == 'c' and q in ('b', 'c'):
        if c1 == 0 and c2 > 0:
            return 'c', c1, c2 - 1
        return 'dead', c1, c2
    return 'dead', c1, c2

def card_read(q, c1, c2):
    return 1.0 if (q == 'c' and c1 == 0 and c2 == 0) else 0.0

def card_run(s):
    st = ('a', 0, 0)
    out = []
    for x in s:
        st = card_step(*st, x)
        out.append(card_read(*st))
    return out

# ------- comparator 1: Minsky two-counter machine as an instruction program -------
SEQ_B = [('JZ', 0, 'dead'), ('DEC', 0), ('INC', 1), ('GOTO', 'b')]
SEQ_C = [('JNZ', 0, 'dead'), ('JZ', 1, 'dead'), ('DEC', 1), ('GOTO', 'c')]
PROG = {('a', 'a'): [('INC', 0), ('GOTO', 'a')],
        ('a', 'b'): SEQ_B, ('b', 'b'): SEQ_B,
        ('b', 'c'): SEQ_C, ('c', 'c'): SEQ_C}

def minsky_run(s):
    L = 'a'
    R = [0, 0]
    out = []
    for x in s:
        if L != 'dead':
            instrs = PROG.get((L, x), [('GOTO', 'dead')])
            for ins in instrs:
                op = ins[0]
                if op == 'INC':
                    R[ins[1]] += 1
                elif op == 'DEC':
                    R[ins[1]] -= 1
                elif op == 'JZ':
                    if R[ins[1]] == 0:
                        L = ins[2]
                        break
                elif op == 'JNZ':
                    if R[ins[1]] != 0:
                        L = ins[2]
                        break
                elif op == 'GOTO':
                    L = ins[1]
                    break
        acc = 1.0 if (L == 'c' and R[0] == 0 and R[1] == 0) else 0.0
        out.append(acc)
    return out

# ------- comparator 2: one-hot lifted input-dependent linear recurrence -------
configs = [(q, i, j) for q in ('a', 'b', 'c') for i in range(N + 1) for j in range(N + 1)]
configs.append(('dead',))
idx = {c: k for k, c in enumerate(configs)}
D = len(configs)
DEAD = idx[('dead',)]

def cfg_index(q, c1, c2):
    if q == 'dead':
        return DEAD
    assert 0 <= c1 <= N and 0 <= c2 <= N
    return idx[(q, c1, c2)]

M = {}
for x in 'abc':
    A = np.zeros((D, D))
    for c in configs:
        if c[0] == 'dead':
            A[DEAD, DEAD] = 1.0
            continue
        q, c1, c2 = c
        nq, n1, n2 = card_step(q, c1, c2, x)
        if nq != 'dead' and (n1 > N or n2 > N):
            # unreachable within length <= N from (a,0,0); route to dead
            A[DEAD, idx[c]] = 1.0
        else:
            A[cfg_index(nq, n1, n2), idx[c]] = 1.0
    M[x] = A
C = np.zeros(D)
C[idx[('c', 0, 0)]] = 1.0
h0 = np.zeros(D)
h0[idx[('a', 0, 0)]] = 1.0

def lifted_run(s):
    h = h0.copy()
    out = []
    for x in s:
        h = M[x] @ h
        out.append(float(C @ h))
    return out

# ------- comparator 3: blind input-conditioned affine counter -------
W0, W1, W2 = 1.0, -1.0, -1.0  # fixed a priori, not fitted to card outputs

def blind_run(s):
    c1 = 0.0
    c2 = 0.0
    out = []
    for x in s:
        c1 = c1 + (x == 'a') - (x == 'b')
        c2 = c2 + (x == 'b') - (x == 'c')
        out.append(W0 + W1 * c1 + W2 * c2)
    return out

# ---------------- inputs ----------------
inputs = ['aabbcc', 'aabbbcc', 'abc', 'aaabbbccc', 'aaaabbbbcccc']
for _ in range(100):
    L = int(rng.integers(1, N + 1))
    inputs.append(''.join(rng.choice(list('abc'), size=L)))
structured = []
for _ in range(100):
    i = int(rng.integers(0, 5))
    j = int(rng.integers(0, 5))
    k = int(rng.integers(0, 5))
    s = 'a' * i + 'b' * j + 'c' * k
    if len(s) == 0:
        s = 'a'
    structured.append(s)
inputs += structured
for s in structured:
    if len(s) >= 2:
        l = list(s)
        p = int(rng.integers(0, len(l) - 1))
        l[p], l[p + 1] = l[p + 1], l[p]
        inputs.append(''.join(l))
    else:
        inputs.append(s + 'c')
assert all(1 <= len(s) <= N for s in inputs)

def maxdiff(fn):
    m = 0.0
    for s in inputs:
        a = np.array(card_run(s))
        b = np.array(fn(s))
        m = max(m, float(np.max(np.abs(a - b))))
    return m

d_minsky = maxdiff(minsky_run)
d_lift = maxdiff(lifted_run)
d_blind = maxdiff(blind_run)

# sanity: toy answer key (final acceptance) vs card
toy_expected = {'aabbcc': 1.0, 'aabbbcc': 0.0}
d_toy = max(abs(card_run(s)[-1] - v) for s, v in toy_expected.items())

# sanity: zero-test witness, best affine fit on counter values 0,1,2 to 1,0,0
X = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
yz = np.array([1.0, 0.0, 0.0])
coef, *_ = np.linalg.lstsq(X, yz, rcond=None)
d_zt = float(np.max(np.abs(X @ coef - yz)))

report = {
    'comparisons': [
        {'declared_id': 'minsky_two_counter',
         'comparator': 'Minsky two-counter machine program (INC/DEC/JZ/JNZ) as one-way acceptor',
         'family': 'other_known', 'transformations': [], 'limit_only': False,
         'reference': 'Minsky (1967), Computation: Finite and Infinite Machines, ch. 14',
         'inputs_tested': len(inputs), 'max_abs_diff': d_minsky},
        {'declared_id': 'onehot_lifted_input_dep_ssm',
         'comparator': 'one-hot lifted input-dependent linear recurrence h_t = M(x_t) h_{t-1}, y_t = C h_t',
         'family': 'ssm', 'transformations': ['lifted_state'], 'limit_only': False,
         'reference': 'input-dependent linear recurrence (SSM form); one-hot configuration lift of a counter automaton',
         'inputs_tested': len(inputs), 'max_abs_diff': d_lift},
        {'declared_id': 'blind_affine_counter',
         'comparator': 'blind input-conditioned affine counters with affine read w=(1,-1,-1)',
         'family': 'ssm', 'transformations': [], 'limit_only': False,
         'reference': 'card closest comparator: affine update without state-tested branch',
         'inputs_tested': len(inputs), 'max_abs_diff': d_blind},
        {'declared_id': None,
         'comparator': 'sanity: card final acceptance vs toy answer key (aabbcc, aabbbcc)',
         'family': 'other_known', 'transformations': [], 'limit_only': False,
         'reference': 'card toy_example', 'inputs_tested': 2, 'max_abs_diff': float(d_toy)},
        {'declared_id': None,
         'comparator': 'sanity: best affine fit to zero-test outputs 1,0,0 on counter 0,1,2',
         'family': 'ssm', 'transformations': [], 'limit_only': False,
         'reference': 'card capability witness', 'inputs_tested': 1, 'max_abs_diff': d_zt}
    ],
    'notes': ('Outputs are 0/1 acceptance indicators per prefix, no rescaling. Lifted state uses dimension 3(N+1)^2+1 = %d '
              '({a,b,c}x[0,N]^2 plus dead, as in the declared update; the mapping text says 4(N+1)^2+1), N=%d. '
              'M(x) is built from the card delta. Blind affine read weights are fixed a priori, not fitted.') % (D, N)
}
print(json.dumps(report))
