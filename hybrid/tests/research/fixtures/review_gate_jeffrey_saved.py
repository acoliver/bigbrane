import json
import numpy as np

rng = np.random.default_rng(12345)
d, V, n, n_inputs = 8, 6, 12, 50

def make_params(rng):
    M = np.zeros((V, d), dtype=float)
    for v in range(V):
        while True:
            row = (rng.random(d) < 0.4).astype(float)
            if 0 < row.sum() < d:
                break
        M[v] = row
    mu = rng.uniform(0.05, 0.95, size=V)
    lab = rng.integers(0, V, size=d)
    return M, mu, lab

# Card: update and readout as written (vectorized, separate masked sums)
def card_run(M, mu, lab, xs):
    p = np.full(d, 1.0 / d)
    ys = []
    for x in xs:
        m = M[x].astype(bool)
        s = float(p[m].sum())
        c = float(p[~m].sum())
        q = p.copy()
        q[m] = q[m] * (mu[x] / s)
        q[~m] = q[~m] * ((1.0 - mu[x]) / c)
        p = q
        y = np.zeros(V)
        np.add.at(y, lab[m], p[m])
        y = y / float(p[m].sum())
        ys.append(y)
    return np.array(ys)

# Comparator jeffrey_ipf_step: Jeffrey / IPF scaling step, written per atom with loops
def ipf_run(M, mu, lab, xs):
    q = [1.0 / d] * d
    ys = []
    for x in xs:
        s = 0.0
        c = 0.0
        for i in range(d):
            if M[x, i] == 1.0:
                s += q[i]
            else:
                c += q[i]
        q = [q[i] * (M[x, i] * mu[x] / s + (1.0 - M[x, i]) * (1.0 - mu[x]) / c) for i in range(d)]
        num = [0.0] * V
        den = 0.0
        for i in range(d):
            if M[x, i] == 1.0:
                num[int(lab[i])] += q[i]
                den += q[i]
        ys.append([a / den for a in num])
    return np.array(ys)

# Comparator diag_ssm_block_norm: per-group L1 normalizer, then input-only diagonal gate; masked read with read normalizer
def ssm_block_run(M, mu, lab, xs):
    h = np.full(d, 1.0 / d)
    ys = []
    for x in xs:
        mrow = M[x]
        s = float(mrow @ h)
        c = float((1.0 - mrow) @ h)
        g = h / (mrow * s + (1.0 - mrow) * c)
        A = mu[x] * mrow + (1.0 - mu[x]) * (1.0 - mrow)
        h = A * g
        C = np.zeros((V, d))
        for i in range(d):
            C[int(lab[i]), i] = mrow[i]
        ys.append((C @ h) / float(mrow @ h))
    return np.array(ys)

# Comparator hedge_global_norm: diagonal multiplicative gate with global renormalization
def hedge_run(M, mu, lab, xs):
    h = np.full(d, 1.0 / d)
    ys = []
    for x in xs:
        mrow = M[x]
        D = mu[x] * mrow + (1.0 - mu[x]) * (1.0 - mrow)
        z = D * h
        h = z / float(z.sum())
        C = np.zeros((V, d))
        for i in range(d):
            C[int(lab[i]), i] = mrow[i]
        ys.append((C @ h) / float(mrow @ h))
    return np.array(ys)

# Undeclared sanity: natural-parameter (log-space) rewrite of the card's update
def logspace_run(M, mu, lab, xs):
    theta = np.full(d, np.log(1.0 / d))
    ys = []
    for x in xs:
        p = np.exp(theta - theta.max()); p = p / p.sum()
        mrow = M[x]
        s = float(mrow @ p)
        c = float((1.0 - mrow) @ p)
        theta = np.log(p) + mrow * (np.log(mu[x] / (1.0 - mu[x])) - np.log(s / c))
        p = np.exp(theta - theta.max()); p = p / p.sum()
        theta = np.log(p)
        y = np.zeros(V)
        m = mrow.astype(bool)
        np.add.at(y, lab[m], p[m])
        ys.append(y / float(p[m].sum()))
    return np.array(ys)

diffs = {'ipf': 0.0, 'ssm': 0.0, 'hedge': 0.0, 'log': 0.0}
for _ in range(n_inputs):
    M, mu, lab = make_params(rng)
    xs = rng.integers(0, V, size=n)
    yc = card_run(M, mu, lab, xs)
    diffs['ipf'] = max(diffs['ipf'], float(np.max(np.abs(yc - ipf_run(M, mu, lab, xs)))))
    diffs['ssm'] = max(diffs['ssm'], float(np.max(np.abs(yc - ssm_block_run(M, mu, lab, xs)))))
    diffs['hedge'] = max(diffs['hedge'], float(np.max(np.abs(yc - hedge_run(M, mu, lab, xs)))))
    diffs['log'] = max(diffs['log'], float(np.max(np.abs(yc - logspace_run(M, mu, lab, xs)))))

out = {"comparisons": [
    {"declared_id": "jeffrey_ipf_step", "comparator": "Jeffrey's rule / I-projection / one IPF scaling step, token-selected constraint, conditional label readout", "family": "other_known", "transformations": [], "limit_only": False, "reference": "Jeffrey 1965; Kullback 1959; Csiszar 1975 (I-divergence geometry, IPF)", "inputs_tested": n_inputs, "max_abs_diff": diffs['ipf']},
    {"declared_id": "diag_ssm_block_norm", "comparator": "input-only diagonal recurrence with per-group L1 normalizer over input-masked groups, masked normalized read", "family": "ssm", "transformations": ["normalizer", "sparsity", "gate", "normalizer"], "limit_only": False, "reference": "selective diagonal SSM h_t = A_{x_t} h_{t-1} (Mamba, Gu and Dao 2023) with declared repairs", "inputs_tested": n_inputs, "max_abs_diff": diffs['ssm']},
    {"declared_id": "hedge_global_norm", "comparator": "diagonal multiplicative recurrence with global renormalization (Hedge / Bayes filter), masked normalized read", "family": "ssm", "transformations": ["normalizer", "normalizer"], "limit_only": False, "reference": "Freund and Schapire 1997 Hedge; Freund, Schapire, Singer, Warmuth 1997 specialists; Bayes filtering", "inputs_tested": n_inputs, "max_abs_diff": diffs['hedge']},
    {"declared_id": None, "comparator": "card update rewritten in natural parameters: theta += M(logit mu - logit s), then normalize", "family": "ssm", "transformations": [], "limit_only": False, "reference": "card known_relatives (logit-space form)", "inputs_tested": n_inputs, "max_abs_diff": diffs['log']}
],
 "notes": "d=8, V=6, n=12, 50 random (M, mu, lab, stream) draws, seed 12345. Outputs y_t are probabilities, so they are compared unscaled at every step. The jeffrey_ipf_step comparator is the card's update in IPF form, written with independent per-atom loops. hedge_global_norm is the plain commuting form and is expected to differ."}
print(json.dumps(out))
