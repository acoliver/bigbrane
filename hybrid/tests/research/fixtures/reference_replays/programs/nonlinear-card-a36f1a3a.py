import json, math
import numpy as np

def softplus(x):
    return float(np.log1p(np.exp(x)))

# ---------- Card, exactly as written ----------
def card_run(tokens, d, alpha, beta):
    E = np.zeros((d, d))
    M = 1.0 - np.eye(d)
    K = int(math.ceil(math.log2(d))) + 2
    out = []
    for tid in tokens:
        if tid < 2 * d * d:
            a = tid // (2 * d); b = (tid // 2) % d; s = tid % 2
            if a != b:
                sig = 1 - 2 * s
                E[a, b] += sig; E[b, a] += sig
            out.append((0.5, 0.5))
        elif tid < 3 * d * d:
            j = tid - 2 * d * d; a = j // d; b = j % d
            if a == b:
                h = 1.0
            else:
                H = np.zeros((d, d))
                for _ in range(K):
                    G = M * (H @ H)
                    H = M * np.tanh(alpha * E + beta * G)
                h = H[a, b]
            out.append(((1 + h) / 2, (1 - h) / 2))
        else:
            out.append((0.5, 0.5))
    return np.array(out)

# ---------- Known form: linear-attention count table + tied single-channel PPGN ----------
def build_tables(d):
    V = 3 * d * d + 2
    keys = np.zeros((V, d * d)); vals = np.zeros(V)
    ask = {}
    for a in range(d):
        for b in range(d):
            for s in (0, 1):
                t = 2 * d * a + 2 * b + s
                if a != b:
                    keys[t, a * d + b] += 1.0; keys[t, b * d + a] += 1.0
                    vals[t] = 1.0 - 2.0 * s
            ask[2 * d * d + a * d + b] = (a, b)
    return keys, vals, ask

def ppgn_readout(Emat, d, L, W1, W2, W4):
    X0 = Emat[..., None]
    Z = np.zeros((d, d, 1))
    mask = (1.0 - np.eye(d))[..., None]
    for _ in range(L):
        A = Z @ W1; B = Z @ W2
        P = np.einsum('ikc,kjc->ijc', A, B)
        Z = np.tanh(np.concatenate([X0, P], axis=-1) @ W4) * mask
    return Z[..., 0]

def known_run(tokens, d, alpha, beta):
    keys, vals, ask = build_tables(d)
    S = np.zeros(d * d)
    L = int(math.ceil(math.log2(d))) + 2
    W1 = np.eye(1); W2 = np.eye(1); W4 = np.array([[alpha], [beta]])
    out = []
    for tid in tokens:
        S = S + vals[tid] * keys[tid]
        if tid in ask:
            a, b = ask[tid]
            if a == b:
                h = 1.0
            else:
                Z = ppgn_readout(S.reshape(d, d), d, L, W1, W2, W4)
                h = Z[a, b]
            out.append(((1 + h) / 2, (1 - h) / 2))
        else:
            out.append((0.5, 0.5))
    return np.array(out)

rng = np.random.default_rng(12345)
maxdiff = 0.0; nask = 0; nonhalf = 0; trials = 0
for d in [3, 4, 5, 6, 8, 9]:
    for rep in range(6):
        alpha = softplus(rng.normal() * 1.5); beta = softplus(rng.normal() * 1.5)
        n = 40
        toks = []
        for _ in range(n):
            u = rng.random()
            if u < 0.55:
                toks.append(int(rng.integers(0, 2 * d * d)))
            elif u < 0.9:
                toks.append(int(rng.integers(2 * d * d, 3 * d * d)))
            else:
                toks.append(3 * d * d + int(rng.integers(0, 2)))
        pc = card_run(toks, d, alpha, beta)
        pk = known_run(toks, d, alpha, beta)
        maxdiff = max(maxdiff, float(np.max(np.abs(pc - pk))))
        nask += sum(1 for t in toks if 2 * d * d <= t < 3 * d * d)
        nonhalf += int(np.sum(np.abs(pc[:, 0] - 0.5) > 1e-3))
        trials += 1

equiv = maxdiff < 1e-6
print(json.dumps({"equivalent": bool(equiv), "max_abs_diff": maxdiff,
  "notes": f"{trials} random token streams, {nask} ASK steps, {nonhalf} non-0.5 outputs; card update/readout vs one-hot-key linear-attention count table + tied 1-channel PPGN (m1=m2=id, m4=tanh([alpha,beta]), diag mask), output scale 1"}))
