import json
import numpy as np

rng = np.random.default_rng(12345)
V = 3
T = 2
n = 12
S = 0
NSEQ = 6

def draw_grammar():
    raw_bin = rng.random((V, V, V)) + 0.05
    raw_lex = rng.random((V, T)) + 0.05
    tot = raw_bin.reshape(V, -1).sum(1) + raw_lex.sum(1)
    return raw_bin / tot[:, None, None], raw_lex / tot[:, None]

# Card: incremental chart. Append x_(t-1), set the lexical entries,
# then fill the spans ending at j=t in order of increasing length.
def card_run(x, P, E, start=0):
    m = len(x)
    Vl = E.shape[0]
    I = np.zeros((Vl, m + 1, m + 1))
    ys = []
    for t in range(1, m + 1):
        I[:, t - 1, t] = E[:, x[t - 1]]
        for w in range(2, t + 1):
            i = t - w
            j = t
            acc = np.zeros(Vl)
            for k in range(i + 1, j):
                acc += np.einsum('abc,b,c->a', P, I[:, i, k], I[:, k, j])
            I[:, i, j] = acc
        ys.append(I[start, 0, t])
    return ys

# Comparator: textbook batch Inside algorithm (Baker 1979; Lari and Young 1990).
# Explicit rule list, run from scratch on each prefix.
def rule_list(P):
    return [(a, b, c, float(P[a, b, c])) for a in range(V) for b in range(V) for c in range(V)]

def inside_batch(x, rules, lex, Vl, start=0, use_max=False):
    m = len(x)
    beta = {}
    for i in range(m):
        for A in range(Vl):
            beta[(A, i, i + 1)] = float(lex[A][x[i]])
    for w in range(2, m + 1):
        for i in range(m - w + 1):
            j = i + w
            for A in range(Vl):
                beta[(A, i, j)] = 0.0
            for (A, B, C, p) in rules:
                vals = [beta[(B, i, k)] * beta[(C, k, j)] for k in range(i + 1, j)]
                if use_max:
                    beta[(A, i, j)] = max(beta[(A, i, j)], p * max(vals))
                else:
                    beta[(A, i, j)] += p * sum(vals)
    return beta[(start, 0, m)]

max_norm = 0.0
max_rel = 0.0
max_vit = 0.0
for s in range(NSEQ):
    P, E = draw_grammar()
    rules = rule_list(P)
    lex = E.tolist()
    x = [int(v) for v in rng.integers(0, T, size=n)]
    yc = np.array(card_run(x, P, E, S))
    yi = np.array([inside_batch(x[:t], rules, lex, V, S) for t in range(1, n + 1)])
    yv = np.array([inside_batch(x[:t], rules, lex, V, S, use_max=True) for t in range(1, n + 1)])
    scale = np.max(np.abs(yc))
    max_norm = max(max_norm, float(np.max(np.abs(yc - yi)) / scale))
    max_rel = max(max_rel, float(np.max(np.abs(yc - yi) / np.abs(yc))))
    max_vit = max(max_vit, float(np.max(np.abs(yc - yv)) / scale))

# Toy sanity check against the card's expected values.
Pt = np.array([[[0.4]]])
Et = np.array([[0.6]])
yt = card_run([0, 0, 0], Pt, Et, 0)
toy_diff = max(abs(yt[1] - 0.144), abs(yt[2] - 0.06912))
vt = inside_batch([0, 0, 0], [(0, 0, 0, 0.4)], [[0.6]], 1, 0, use_max=True)
toy_vit_diff = abs(yt[2] - vt)

out = {'comparisons': [
    {'declared_id': 'inside_algorithm_pcfg',
     'comparator': 'Batch Inside algorithm (sum-product CKY) recomputed from scratch on each prefix, explicit rule list',
     'family': 'other_known',
     'transformations': [],
     'limit_only': False,
     'reference': 'Inside algorithm for CNF PCFGs, Baker 1979; Lari and Young 1990, Computer Speech and Language 4:35-56; Goodman 1999 semiring parsing',
     'inputs_tested': NSEQ,
     'max_abs_diff': max_norm},
    {'declared_id': None,
     'comparator': 'Same inside comparison, per-step relative difference (sanity, stricter)',
     'family': 'other_known', 'transformations': [], 'limit_only': False,
     'reference': 'Lari and Young 1990', 'inputs_tested': NSEQ, 'max_abs_diff': max_rel},
    {'declared_id': None,
     'comparator': 'Viterbi max-product chart (separating control, expected to differ)',
     'family': 'other_known', 'transformations': [], 'limit_only': False,
     'reference': 'Viterbi CKY, max-product semiring (Goodman 1999)', 'inputs_tested': NSEQ, 'max_abs_diff': max_vit},
    {'declared_id': None,
     'comparator': 'Toy grammar S->SS .4, S->a .6: card vs expected aa=.144, aaa=.06912',
     'family': 'other_known', 'transformations': [], 'limit_only': False,
     'reference': 'card toy_example', 'inputs_tested': 1, 'max_abs_diff': float(toy_diff)},
    {'declared_id': None,
     'comparator': 'Toy grammar: card sum-product vs max-product on aaa (expected .03456)',
     'family': 'other_known', 'transformations': [], 'limit_only': False,
     'reference': 'card capability witness', 'inputs_tested': 1, 'max_abs_diff': float(toy_vit_diff)}],
 'notes': 'Likelihoods shrink with length, so for the random-grammar comparisons both outputs are divided by the largest absolute card output of each sequence before differencing. The relative-difference row is extra and uncounted. Random CNF grammars: V=3, 2 terminals, all 27 binary rules, rule probabilities normalized per LHS, n=12.'}
print(json.dumps(out))
