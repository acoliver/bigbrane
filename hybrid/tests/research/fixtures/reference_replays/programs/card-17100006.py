import json
import numpy as np

rng = np.random.default_rng(12345)
D, V, n, S = 4, 3, 12, 25
T_TEMP = 0.25  # free temperature of the tempered comparator (no card counterpart)

def card_run(pi, A, e, xs):
    # Card equations, plain probability domain, explicit loops
    ys = []
    v = None
    for t, x in enumerate(xs):
        if t == 0:
            v = [pi[j] * e[j, x] for j in range(D)]
        else:
            v = [e[j, x] * max(v[i] * A[i, j] for i in range(D)) for j in range(D)]
        ys.append(max(v))
    return np.array(ys)

def viterbi_published(pi, A, e, xs):
    # Textbook Viterbi delta recursion, log-domain implementation, converted back
    lA = np.log(A)
    ld = np.log(pi) + np.log(e[:, xs[0]])
    ys = [np.exp(ld.max())]
    for x in xs[1:]:
        ld = np.log(e[:, x]) + np.max(ld[:, None] + lA, axis=0)
        ys.append(np.exp(ld.max()))
    return np.array(ys)

def hmm_forward(pi, A, e, xs):
    a = pi * e[:, xs[0]]
    ys = [a.sum()]
    for x in xs[1:]:
        a = e[:, x] * (A.T @ a)
        ys.append(a.sum())
    return np.array(ys)

def tempered_forward(pi, A, e, xs, T):
    p = 1.0 / T
    v = pi * e[:, xs[0]]
    ys = [np.sum(v ** p) ** T]
    for x in xs[1:]:
        v = e[:, x] * np.sum((v[:, None] * A) ** p, axis=0) ** T
        ys.append(np.sum(v ** p) ** T)
    return np.array(ys)

mx = {'vit': 0.0, 'fwd': 0.0, 'tmp': 0.0}
rel = {'fwd': 0.0, 'tmp': 0.0}
for s in range(S):
    pi = rng.dirichlet(np.ones(D))
    A = rng.dirichlet(np.ones(D), size=D)
    e = rng.dirichlet(np.ones(V), size=D)
    xs = rng.integers(0, V, size=n)
    c = card_run(pi, A, e, xs)
    scale = np.max(np.abs(c))
    outs = {'vit': viterbi_published(pi, A, e, xs),
            'fwd': hmm_forward(pi, A, e, xs),
            'tmp': tempered_forward(pi, A, e, xs, T_TEMP)}
    for k, o in outs.items():
        mx[k] = max(mx[k], float(np.max(np.abs(o - c)) / scale))
    for k in ('fwd', 'tmp'):
        rel[k] = max(rel[k], float(np.max(np.abs(outs[k] - c) / np.abs(c))))

# Card toy witness: D=2, pi=(.5,.5), A=.5, e(a)=.5, sequence aa
pi2 = np.array([.5, .5]); A2 = np.full((2, 2), .5); e2 = np.full((2, 2), .5)
w_card = card_run_small = None
def card2(xs):
    v = [pi2[j] * e2[j, xs[0]] for j in range(2)]
    for x in xs[1:]:
        v = [e2[j, x] * max(v[i] * A2[i, j] for i in range(2)) for j in range(2)]
    return max(v)
w_card = card2([0, 0])
w_toy_diff = abs(w_card - 0.0625)

res = {"comparisons": [
    {"declared_id": "viterbi_published", "comparator": "Viterbi max-product score recursion (log-domain implementation), running max readout",
     "family": "other_known", "transformations": [], "limit_only": False,
     "reference": "Viterbi 1967 (doi:10.1109/TIT.1967.1054010); Rabiner 1989 delta recursion",
     "inputs_tested": S, "max_abs_diff": mx['vit']},
    {"declared_id": "hmm_forward_sum_product", "comparator": "HMM forward sum-product recursion, likelihood readout",
     "family": "ssm", "transformations": [], "limit_only": False,
     "reference": "Rabiner 1989 forward algorithm alpha_t(j)=b_j(o_t) sum_i alpha_{t-1}(i) a_ij",
     "inputs_tested": S, "max_abs_diff": mx['fwd']},
    {"declared_id": "tempered_forward_powermean", "comparator": "Forward recursion with temperature-T power sum (T=0.25) replacing the sum",
     "family": "ssm", "transformations": ["other"], "limit_only": True,
     "reference": "Tempered/log-sum-exp forward recursion, tends to Viterbi as T->0",
     "inputs_tested": S, "max_abs_diff": mx['tmp']},
    {"declared_id": None, "comparator": "sanity: forward vs card, per-step relative difference",
     "family": "ssm", "transformations": [], "limit_only": False,
     "reference": "Rabiner 1989 forward algorithm", "inputs_tested": S, "max_abs_diff": rel['fwd']},
    {"declared_id": None, "comparator": "sanity: tempered forward vs card, per-step relative difference",
     "family": "ssm", "transformations": ["other"], "limit_only": True,
     "reference": "tempered forward recursion", "inputs_tested": S, "max_abs_diff": rel['tmp']},
    {"declared_id": None, "comparator": "sanity: card toy witness viterbi_aa vs expected 0.0625",
     "family": "other_known", "transformations": [], "limit_only": False,
     "reference": "card toy_example expected", "inputs_tested": 1, "max_abs_diff": float(w_toy_diff)}],
  "notes": "Card outputs are path probabilities that shrink with t, so for each sequence the comparator and card outputs were both divided by the largest absolute card output of that sequence before differencing. D=4, V=3, n=12, 25 random Dirichlet (pi, A, e) and token sequences, seed 12345. Null entries give per-step relative differences and the toy witness."}
print(json.dumps(res))
