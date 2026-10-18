import json
import numpy as np
from collections import deque

rng = np.random.default_rng(12345)

# ---------- card: update and readout as written ----------
def card_outputs(seq):
    q = 'store'
    Q = deque()
    ys = []
    for x in seq:
        if q == 'store':
            if x == '#':
                q = 'compare'
            else:
                Q.append(x)
        elif q == 'compare':
            if x == '#' or len(Q) == 0 or Q[0] != x:
                q = 'dead'
            else:
                Q.popleft()
        # dead is absorbing
        ys.append(1.0 if (q == 'compare' and len(Q) == 0) else 0.0)
    return ys

# ---------- comparator 1: real-time one-queue automaton (immutable configs) ----------
def qa_delta(config, x):
    ctrl, word = config
    if ctrl == 'S':
        if x in ('a', 'b'):
            return ('S', word + x)
        return ('C', word)
    if ctrl == 'C':
        if x in ('a', 'b') and len(word) > 0 and word[0] == x:
            return ('C', word[1:])
        return ('D', word)
    return ('D', word)

def queue_automaton_outputs(seq):
    config = ('S', '')
    ys = []
    for x in seq:
        config = qa_delta(config, x)
        ys.append(1.0 if (config[0] == 'C' and config[1] == '') else 0.0)
    return ys

# ---------- comparator 2: positional exact-match cache with hard top-1 read ----------
SYM = {'a': 0, 'b': 1, '#': 2}

def cache_outputs(seq):
    keys = []   # feature_map: rank among a/b tokens before '#'
    vals = []   # lifted_state: one entry per stored token
    n = 0
    p = 0
    j = 0
    d = 0
    ys = []
    for x in seq:
        xv = SYM[x]
        if p == 0:
            if x == '#':
                p = 1
                j = 0
            else:
                n = n + 1
                keys.append(n)
                vals.append(xv)
        else:
            j = j + 1
            # hard top-1 exact-match read (sparsity)
            if len(keys) > 0:
                K = np.array(keys, dtype=float)
                scores = (K == float(j)).astype(float)
                i = int(np.argmax(scores))
                hit = scores[i] == 1.0
                r = vals[i] if hit else None
            else:
                r = None
            mismatch = (r is None) or (r != xv)
            d = int(d or (j > n) or mismatch or (x == '#'))
        ys.append(1.0 if (p == 1 and not d and j == n) else 0.0)
    return ys

# ---------- inputs ----------
def rand_word(L):
    return ''.join(rng.choice(['a', 'b'], size=L)) if L > 0 else ''

inputs = ['ab#ab', 'ab#ba', '#', '', 'a#', '#a', 'ab#a', 'ab#abb', 'ab#ab#', '##']
for _ in range(300):
    L = 12
    probs = [0.4, 0.4, 0.2]
    inputs.append(''.join(rng.choice(['a', 'b', '#'], size=L, p=probs)))
for _ in range(300):
    m = int(rng.integers(0, 6))
    w = rand_word(m)
    s = w + '#' + w
    kind = int(rng.integers(0, 5))
    if kind == 1 and len(s) > 0:
        k = int(rng.integers(0, len(s)))
        s = s[:k] + rng.choice(['a', 'b', '#']) + s[k + 1:]
    elif kind == 2:
        s = s[:int(rng.integers(0, len(s) + 1))]
    elif kind == 3:
        s = s + rand_word(int(rng.integers(1, 3)))
    elif kind == 4:
        s = w + '#' + w[::-1]
    inputs.append(s)

max_q = 0.0
max_c = 0.0
accept_count = 0
for s in inputs:
    yc = np.array(card_outputs(s), dtype=float)
    yq = np.array(queue_automaton_outputs(s), dtype=float)
    yk = np.array(cache_outputs(s), dtype=float)
    if len(yc) > 0:
        max_q = max(max_q, float(np.max(np.abs(yc - yq))))
        max_c = max(max_c, float(np.max(np.abs(yc - yk))))
        accept_count += int(yc[-1] == 1.0)

toy_ok = (card_outputs('ab#ab')[-1] == 1.0) and (card_outputs('ab#ba')[-1] == 0.0)

report = {
    'comparisons': [
        {
            'declared_id': 'queue_automaton',
            'comparator': 'Real-time deterministic one-queue automaton for the copy language w#w',
            'family': 'other_known',
            'transformations': [],
            'limit_only': False,
            'reference': 'Baeten and Luttik (2025), The Queue Automaton Revisited, arXiv:2502.08345; Post machine',
            'inputs_tested': len(inputs),
            'max_abs_diff': max_q
        },
        {
            'declared_id': 'positional_exact_match_cache',
            'comparator': 'Hard top-1 attention over a growing KV cache with exact-match positional keys plus phase bit and absorbing mismatch flag',
            'family': 'transformer_variant',
            'transformations': ['sparsity', 'feature_map', 'lifted_state', 'other'],
            'limit_only': False,
            'reference': 'Hard-attention transformer constructions, e.g. Perez et al. 2019',
            'inputs_tested': len(inputs),
            'max_abs_diff': max_c
        }
    ],
    'notes': 'Outputs are 0/1 acceptance per step (order one, no rescaling). Compared at every step on %d sequences (random length-12 strings over {a,b,#} and w#w variants); toy example reproduced: %s; final accepts: %d.' % (len(inputs), str(toy_ok), accept_count)
}
print(json.dumps(report))
