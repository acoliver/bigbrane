import json
import numpy as np

rng = np.random.default_rng(1710)
n, dk, dv = 12, 4, 3

def card_run(K, V, Q, B):
    S = np.zeros((dv, dk))
    ys = []
    for t in range(len(K)):
        k, v, q, b = K[t], V[t], Q[t], B[t]
        S = S + b * np.outer(v - S @ k, k)
        ys.append(S @ q)
    return np.array(ys)

def delta_comparator(K, V, Q, B):
    # independent scalar-loop implementation of W_t = W_{t-1} + beta (v - W k) k^T, y = W q
    W = [[0.0] * dk for _ in range(dv)]
    ys = []
    for t in range(len(K)):
        k = [float(x) for x in K[t]]; v = [float(x) for x in V[t]]; q = [float(x) for x in Q[t]]; b = float(B[t])
        pred = [sum(W[i][j] * k[j] for j in range(dk)) for i in range(dv)]
        res = [v[i] - pred[i] for i in range(dv)]
        W = [[W[i][j] + b * res[i] * k[j] for j in range(dk)] for i in range(dv)]
        ys.append([sum(W[i][j] * q[j] for j in range(dk)) for i in range(dv)])
    return np.array(ys)

def additive_comparator(K, V, Q, B):
    S = np.zeros((dv, dk))
    ys = []
    for t in range(len(K)):
        S = S + B[t] * np.outer(V[t], K[t])
        ys.append(S @ Q[t])
    return np.array(ys)

def product_form(K, V, Q, B):
    # sanity: rearranged card equation S(I - beta k k^T) + beta v k^T
    S = np.zeros((dv, dk))
    I = np.eye(dk)
    ys = []
    for t in range(len(K)):
        k = K[t]
        S = S @ (I - B[t] * np.outer(k, k)) + B[t] * np.outer(V[t], k)
        ys.append(S @ Q[t])
    return np.array(ys)

beta_modes = ['0.2', '0.7', '1.0', 'random']
diff = {'delta': 0.0, 'add': 0.0, 'prod': 0.0}
count = 0
for rep in range(4):
    for mode in beta_modes:
        base = rng.standard_normal(dk)
        K = 0.5 * base + 0.5 * rng.standard_normal((n, dk))  # shared component -> nonorthogonal keys
        K = K / np.maximum(1.0, np.linalg.norm(K, axis=1, keepdims=True))  # ||k|| <= 1 for stability
        V = rng.standard_normal((n, dv))
        Q = rng.standard_normal((n, dk))
        B = rng.uniform(0, 1, n) if mode == 'random' else np.full(n, float(mode))
        yc = card_run(K, V, Q, B)
        scale = max(np.max(np.abs(yc)), 1e-12)
        diff['delta'] = max(diff['delta'], float(np.max(np.abs(delta_comparator(K, V, Q, B) - yc)) / scale))
        diff['add'] = max(diff['add'], float(np.max(np.abs(additive_comparator(K, V, Q, B) - yc)) / scale))
        diff['prod'] = max(diff['prod'], float(np.max(np.abs(product_form(K, V, Q, B) - yc)) / scale))
        count += 1

# toy check from the card (scalar case)
s = 0.0
for k, v, b in [(1., 2., .5), (1., 3., .5)]:
    s += b * (v - s * k) * k
toy_diff = abs(s * 0.25 - 0.5)

out = {"comparisons": [
  {"declared_id": "deltanet_delta_rule", "comparator": "Delta-rule fast weight programmer y_t = W_t q_t, W_t = W_{t-1} + beta_t (v_t - W_{t-1} k_t) k_t^T", "family": "linear_attention", "transformations": [], "limit_only": False, "reference": "Schlag, Irie and Schmidhuber 2021 arXiv:2102.11174; Yang et al. 2024 arXiv:2406.06484", "inputs_tested": count, "max_abs_diff": diff['delta']},
  {"declared_id": "additive_linear_attention", "comparator": "Unnormalized additive linear attention with write gate beta_t", "family": "linear_attention", "transformations": ["gate"], "limit_only": False, "reference": "Katharopoulos et al. 2020 arXiv:2006.16236", "inputs_tested": count, "max_abs_diff": diff['add']},
  {"declared_id": None, "comparator": "Sanity: card update rearranged as S(I - beta k k^T) + beta v k^T", "family": "linear_attention", "transformations": [], "limit_only": False, "reference": "algebraic rearrangement of the card", "inputs_tested": count, "max_abs_diff": diff['prod']},
  {"declared_id": None, "comparator": "Card toy example read vs expected 0.5", "family": "linear_attention", "transformations": [], "limit_only": False, "reference": "card toy_example", "inputs_tested": 1, "max_abs_diff": float(toy_diff)}
 ],
 "notes": "n=12, dk=4, dv=3. Keys are nonorthogonal (shared component) and have norm <= 1. Beta is in {0.2, 0.7, 1.0, uniform}. For each sequence, the differences are divided by the largest absolute card output."}
print(json.dumps(out))
