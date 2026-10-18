import numpy as np, json, itertools
rng = np.random.default_rng(0)

def make_tokens(V, d):
    E = rng.choice([-1.0, 0.0, 1.0], size=(V, d), p=[0.4, 0.2, 0.4])
    E[0] = 0.0  # SEP token, E[SEP] = 0
    return E

# ---------- card model, implemented exactly as the card writes it ----------
def card_run(E, seq, W, beta, gamma, K):
    d = E.shape[1]
    h = np.zeros(d); s = np.zeros(d); prev = np.zeros(d)  # E[BOS] = 0
    outs = []
    for x in seq:
        h = h + prev * E[x]
        u = s.copy()
        for _ in range(K):
            u = np.tanh(gamma * (np.einsum('abc,a,b->c', W, u, u) + beta * h))
        s = u
        z = E @ (s * E[x])
        p = np.exp(z - z.max()); p = p / p.sum()
        outs.append((h.copy(), s.copy(), p))
        prev = E[x]
    return outs

# ---------- known model: cumsum SSM (A=I) -> polynomial (x^3) DAM with external field -> gated bilinear readout ----------
def known_run(E, seq, A, beta, gamma, K):
    d = E.shape[1]
    feats = np.stack([E[x] for x in seq])
    prevs = np.vstack([np.zeros((1, d)), feats[:-1]])
    phi = prevs * feats              # fixed bilinear pair feature (SSM input)
    H = np.zeros_like(phi); acc = np.zeros(d)
    for t in range(len(seq)):        # SSM scan with A = I
        acc = 1.0 * acc + phi[t]
        H[t] = acc
    s = np.zeros(d); outs = []
    for t, x in enumerate(seq):
        u = s.copy()
        for _ in range(K):           # Krotov-Hopfield DAM, F(x)=x^3, soft-sign mean-field update
            u = np.tanh(gamma * (A.T @ ((A @ u) ** 2) + beta * H[t]))
        s = u
        G = E * E[x][None, :]        # multiplicative gate
        z = G @ s
        p = np.exp(z - z.max()); p = p / p.sum()
        outs.append((H[t].copy(), s.copy(), p))
    return outs

def compare(o1, o2):
    diff = 0.0; scale = 1.0
    for (h1, s1, p1), (h2, s2, p2) in zip(o1, o2):
        diff = max(diff, np.abs(h1 - h2).max(), np.abs(s1 - s2).max(), np.abs(p1 - p2).max())
        scale = max(scale, np.abs(h1).max(), np.abs(s1).max(), np.abs(p1).max())
    return diff, scale

n = 12; K = 3
results = []

# Case 1: general learned W = Sym(Theta). Patterns come from the polarization identity
# Sym(e_a x e_b x e_c) = (1/48) sum_eps eps1 eps2 eps3 (eps1 e_a + eps2 e_b + eps3 e_c)^{x3}, signs absorbed by cube root
d = 6; V = 15
E = make_tokens(V, d)
Theta = 0.3 * rng.standard_normal((d, d, d))
W = sum(np.transpose(Theta, p) for p in itertools.permutations(range(3))) / 6.0
beta = rng.uniform(0.1, 1.0); gamma = rng.uniform(0.5, 2.0)
rows = []
I = np.eye(d)
for a in range(d):
    for b in range(d):
        for c in range(d):
            for e1, e2, e3 in itertools.product([-1.0, 1.0], repeat=3):
                coef = e1 * e2 * e3 * Theta[a, b, c] / 48.0
                v = e1 * I[a] + e2 * I[b] + e3 * I[c]
                rows.append(np.cbrt(coef) * v)
A = np.array(rows)
for trial in range(3):
    seq = list(rng.integers(0, V, size=n))
    results.append(compare(card_run(E, seq, W, beta, gamma, K), known_run(E, seq, A, beta, gamma, K)))

# Case 2: Hebbian W = d^-2 sum_mu xi^{x3}; known patterns a_mu = xi_mu / d^{2/3}
d = 10; V = 20; M = 6
E = make_tokens(V, d)
Xi = rng.choice([-1.0, 1.0], size=(M, d))
W2 = np.einsum('mi,mj,mk->ijk', Xi, Xi, Xi) / d ** 2
A2 = Xi / d ** (2.0 / 3.0)
for trial in range(3):
    beta = rng.uniform(0.1, 1.0); gamma = rng.uniform(0.5, 5.0)
    seq = list(rng.integers(0, V, size=n))
    results.append(compare(card_run(E, seq, W2, beta, gamma, K), known_run(E, seq, A2, beta, gamma, K)))

maxdiff = max(r[0] for r in results)
scale = max(r[1] for r in results)
eq = bool(maxdiff / scale < 1e-6)
print(json.dumps({'equivalent': eq, 'max_abs_diff': float(maxdiff), 'notes': 'h, sigma and p compared at every step; general symmetric W mapped to sum_r (a_r.u)^2 a_r via polarization (R=8d^3), Hebbian W via a=xi/d^(2/3); h via identity-transition SSM scan'}))
