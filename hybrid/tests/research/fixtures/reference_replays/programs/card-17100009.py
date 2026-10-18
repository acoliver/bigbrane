import json
import numpy as np

rng = np.random.default_rng(17100009)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

def card(X, Wf, bf, Wu):
    d = Wf.shape[0]
    h = np.zeros(d)
    ys = []
    for x in X:
        f = sigmoid(Wf @ x + bf)
        u = Wu @ x
        h = f * h + (1.0 - f) * u
        ys.append(h.copy())
    return np.array(ys)

def selective_diag_ssm(X, Wf, bf, Wu):
    # Generic S_t = A_t S_{t-1} + B_t x_t, y_t = C_t S_t with full matrices
    d = Wf.shape[0]
    S = np.zeros(d)
    C = np.eye(d)
    ys = []
    for x in X:
        g = sigmoid(Wf @ x + bf)
        A = np.diag(g)
        B = np.diag(1.0 - g) @ Wu
        S = A @ S + B @ x
        ys.append(C @ S)
    return np.array(ys)

def mamba_zoh_n1(X, Wf, bf, Wu):
    # Mamba ZOH discretization, N = 1 per channel, A = -1, B = 1, C = 1
    d = Wf.shape[0]
    A = -np.ones(d)
    Bc = np.ones(d)
    Cc = np.ones(d)
    Wdelta = -Wf
    bdelta = -bf
    h = np.zeros(d)
    ys = []
    for x in X:
        u = Wu @ x  # in_proj
        Delta = softplus(Wdelta @ x + bdelta)
        DA = Delta * A
        Abar = np.exp(DA)
        Bbar = (1.0 / DA) * (np.exp(DA) - 1.0) * Delta * Bc
        h = Abar * h + Bbar * u
        ys.append(Cc * h)
    return np.array(ys)

p = 5
max_ssm = 0.0
max_mamba = 0.0
count = 0
for d in (2, 4, 8):
    for n in (1, 8, 32):
        for rep in range(5):
            Wf = rng.normal(size=(d, p))
            bf = rng.normal(size=d)
            Wu = rng.normal(size=(d, p))
            X = rng.normal(size=(n, p))
            yc = card(X, Wf, bf, Wu)
            scale = max(np.max(np.abs(yc)), 1e-12)
            y1 = selective_diag_ssm(X, Wf, bf, Wu)
            y2 = mamba_zoh_n1(X, Wf, bf, Wu)
            max_ssm = max(max_ssm, float(np.max(np.abs(y1 - yc)) / scale))
            max_mamba = max(max_mamba, float(np.max(np.abs(y2 - yc)) / scale))
            count += 1

out = {"comparisons": [
    {"declared_id": "selective_diag_ssm",
     "comparator": "input-dependent diagonal SSM with A_t=diag(sigma(Wf x+bf)), B_t=diag(1-f_t)Wu, C_t=I",
     "family": "ssm", "transformations": [], "limit_only": False,
     "reference": "Selective SSM, Gu and Dao 2023, arXiv:2312.00752",
     "inputs_tested": count, "max_abs_diff": max_ssm},
    {"declared_id": "mamba_zoh_n1",
     "comparator": "Mamba ZOH selective scan, N=1 per channel, A=-1, B=1, C=1, Delta=softplus(-(Wf x+bf)), in_proj=Wu",
     "family": "ssm", "transformations": [], "limit_only": False,
     "reference": "Mamba ZOH discretization, Gu and Dao 2023, arXiv:2312.00752",
     "inputs_tested": count, "max_abs_diff": max_mamba}],
 "notes": "Outputs are divided by the largest absolute card output per sequence before differencing. Configurations: d in {2,4,8}, n in {1,8,32}, p=5, 5 random parameter and input draws each, float64."}
print(json.dumps(out))
