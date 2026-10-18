import json
import numpy as np

rng = np.random.default_rng(0)
OP = ['(', '[']
CL = [')', ']']
SYM = OP + CL
MATCH = {')': '(', ']': '['}

# ---------- card: update and readout as written ----------
def card_run(seq):
    q = 'live'; z = []; ys = []
    for x in seq:
        if q == 'live':
            if x in OP:
                z = z + [x]
            else:
                if len(z) > 0 and z[-1] == MATCH[x]:
                    z = z[:-1]
                else:
                    q = 'dead'  # mismatch or underflow; stack unchanged
        # dead: stack unchanged, stays dead
        ys.append(1.0 if (q == 'live' and len(z) == 0) else 0.0)
    return ys

# ---------- comparator 1: textbook DPDA, declared update clauses in order ----------
def dpda_run(seq):
    q = 'live'; z = []; ys = []
    for x in seq:
        if q == 'live' and x in OP:
            z = z + [x]; q = 'live'
        elif q == 'live' and x in CL and len(z) > 0 and z[-1] == MATCH[x]:
            z = z[:-1]; q = 'live'
        elif q == 'live' and (len(z) == 0 or z[-1] != MATCH[x]):
            q = 'dead'
        else:  # q dead
            q = 'dead'
        ys.append(1.0 if (q == 'live' and len(z) == 0) else 0.0)
    return ys

# ---------- comparator 2: stack-augmented RNN, hard (argmax) actions ----------
# Features: [q one-hot(2); x one-hot(4); top one-hot(3: '(', '[', empty); x (x) top outer (12)]
# The outer-product block is needed because no linear map of [x; top] separates match from mismatch.
TOPS = ['(', '[', None]
F = 2 + 4 + 3 + 12
A_PUSH, A_POP, A_KILL = 0, 1, 2
beta = 5.0
W = np.zeros((3, F))
for i, x in enumerate(SYM):
    for j, t in enumerate(TOPS):
        col = 9 + i * 3 + j
        if x in OP:
            W[A_PUSH, col] = beta
        elif t is not None and MATCH[x] == t:
            W[A_POP, col] = beta
        else:
            W[A_KILL, col] = beta
W[A_KILL, 1] = 3 * beta  # dead control -> kill (no stack change)

def feats(q, x, top):
    f = np.zeros(F)
    f[0 if q == 'live' else 1] = 1.0
    i = SYM.index(x); f[2 + i] = 1.0
    j = TOPS.index(top); f[6 + j] = 1.0
    f[9 + i * 3 + j] = 1.0
    return f

def hard_stack_run(seq):
    q = 'live'; z = []; ys = []
    for x in seq:
        top = z[-1] if len(z) > 0 else None
        s = W @ feats(q, x, top)
        p = np.exp(s - s.max()); p = p / p.sum()
        a = int(np.argmax(p))
        if a == A_PUSH:
            z = z + [x]
        elif a == A_POP:
            z = z[:-1]
        q = 'dead' if (a == A_KILL or q == 'dead') else 'live'
        ys.append(1.0 if (q == 'live' and len(z) == 0) else 0.0)
    return ys

# ---------- sanity: card toy python on full strings (not declared) ----------
def toy_accepts(s):
    stack = []
    for c in s:
        if c in '([': stack.append(c)
        elif not stack or stack.pop() != {')': '(', ']': '['}[c]: return False
    return not stack

# ---------- inputs ----------
def rand_dyck(n):
    st = []; out = []
    for k in range(n):
        rem = n - k
        if len(st) == rem or (len(st) > 0 and rng.random() < 0.5):
            o = st.pop(); out.append(')' if o == '(' else ']')
        else:
            o = OP[rng.integers(2)]; st.append(o); out.append(o)
    return out

seqs = []
for _ in range(400):
    seqs.append([SYM[k] for k in rng.integers(0, 4, size=12)])
for _ in range(400):
    seqs.append(rand_dyck(12))
for _ in range(400):
    s = rand_dyck(12); p = int(rng.integers(12)); s[p] = SYM[rng.integers(4)]; seqs.append(s)
for L in [1, 2, 3, 5, 8, 16, 24]:
    for _ in range(30):
        seqs.append([SYM[k] for k in rng.integers(0, 4, size=L)])
        if L % 2 == 0:
            seqs.append(rand_dyck(L))
seqs.append(list('(()[])')); seqs.append(list('([)]'))

d1 = 0.0; d2 = 0.0; d3 = 0.0; n_acc = 0
for s in seqs:
    yc = np.array(card_run(s))
    y1 = np.array(dpda_run(s))
    y2 = np.array(hard_stack_run(s))
    d1 = max(d1, float(np.max(np.abs(yc - y1))))
    d2 = max(d2, float(np.max(np.abs(yc - y2))))
    d3 = max(d3, abs(yc[-1] - (1.0 if toy_accepts(''.join(s)) else 0.0)))
    n_acc += int(yc[-1])

res = {"comparisons": [
  {"declared_id": "dpda_dyck2_textbook", "comparator": "Textbook DPDA for Dyck-2, accept by final state with empty stack", "family": "other_known", "transformations": [], "limit_only": False, "reference": "Ginsburg and Greibach 1966, Information and Control 9:620-648; Hopcroft and Ullman ch. 10", "inputs_tested": len(seqs), "max_abs_diff": d1},
  {"declared_id": "hard_stack_rnn", "comparator": "Stack-augmented RNN with one-hot (argmax) push/pop/kill actions", "family": "other_known", "transformations": [], "limit_only": True, "reference": "Joulin and Mikolov 2015; Grefenstette et al. 2015", "inputs_tested": len(seqs), "max_abs_diff": d2},
  {"declared_id": None, "comparator": "Card toy_example python accepts() on full strings vs card final output (sanity)", "family": "other_known", "transformations": [], "limit_only": False, "reference": "card toy_example", "inputs_tested": len(seqs), "max_abs_diff": float(d3)}
 ],
 "notes": "Outputs are 0/1 acceptance bits per step, so no rescaling is applied. The inputs are random strings, random Dyck-2 strings and single-symbol mutations, with lengths 1-24; %d sequences are accepted at the end. In the hard-stack controller the softmax scores come from one-hot features [q; x; top; x outer top]. The outer-product block is needed because a linear map on [x; top] cannot separate match from mismatch. Actions use argmax, as declared." % n_acc}
print(json.dumps(res))
