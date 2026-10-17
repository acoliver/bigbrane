RECIPE = {'width_d': 64, 'initialization': 'Gaussian: E and P std 0.7; W_a, W_c, W_p, V std 1/sqrt(d); W_o, U std 0.5/sqrt(d); O std 1/sqrt(d)', 'optimizer': 'AdamW', 'learning_rate': 0.002, 'betas': [0.9, 0.98], 'eps': 1e-08, 'weight_decay': 0.05, 'warmup_steps': 300, 'schedule': 'linear warmup then cosine decay to 0.05 of peak', 'grad_clip_norm': 1.0, 'epochs': 400, 'batch_size': 200, 'loss': 'next-symbol cross-entropy at the query positions, back-propagated through both rounds', 'dtype': 'float64'}


def make_params(V, L, d, rng):
    s = 1.0 / np.sqrt(d)
    p = {}
    p['E'] = (rng.standard_normal((V, d)) * 0.7).astype(np.float64)
    p['P'] = (rng.standard_normal((L, d)) * 0.7).astype(np.float64)
    for l in ['1', '2']:
        p['Wa' + l] = (rng.standard_normal((d, d)) * s).astype(np.float64)
        p['Wc' + l] = (rng.standard_normal((d, d)) * s).astype(np.float64)
        p['Wp' + l] = (rng.standard_normal((d, d)) * s).astype(np.float64)
        p['Wo' + l] = (rng.standard_normal((d, d)) * (0.5 * s)).astype(np.float64)
        p['V' + l] = (rng.standard_normal((d, d)) * s).astype(np.float64)
        p['U' + l] = (rng.standard_normal((d, d)) * (0.5 * s)).astype(np.float64)
    p['O'] = (rng.standard_normal((d, V)) * s).astype(np.float64)
    return p


def mm(x, w):
    sh = x.shape
    return (x.reshape(-1, sh[-1]) @ w).reshape(sh[:-1] + (w.shape[1],))


def gw(x, g):
    return x.reshape(-1, x.shape[-1]).T @ g.reshape(-1, g.shape[-1])


def grads(p, X, Y, qs, mask1, mask2, d, V):
    Bn, L = X.shape
    sc = np.float64(1.0 / np.sqrt(d))
    h0 = p['E'][X] + p['P'][:L][None]
    A1 = mm(h0, p['Wa1'])
    C1 = mm(h0, p['Wc1'])
    P1 = mm(h0, p['Wp1'])
    S1 = (P1 @ A1.transpose(0, 2, 1)) * sc + mask1
    S1 = S1 - S1.max(-1, keepdims=True)
    W1 = np.exp(S1)
    W1 = W1 / W1.sum(-1, keepdims=True)
    R1 = W1 @ C1
    G1 = h0 + mm(R1, p['Wo1'])
    Z1 = mm(G1, p['V1'])
    M1 = np.maximum(Z1, 0)
    h1 = G1 + mm(M1, p['U1'])
    A2 = mm(h1, p['Wa2'])
    C2 = mm(h1, p['Wc2'])
    hq = h1[:, qs]
    P2 = mm(hq, p['Wp2'])
    S2 = (P2 @ A2.transpose(0, 2, 1)) * sc + mask2
    S2 = S2 - S2.max(-1, keepdims=True)
    W2 = np.exp(S2)
    W2 = W2 / W2.sum(-1, keepdims=True)
    R2 = W2 @ C2
    G2 = hq + mm(R2, p['Wo2'])
    Z2 = mm(G2, p['V2'])
    M2 = np.maximum(Z2, 0)
    h2 = G2 + mm(M2, p['U2'])
    lg = mm(h2, p['O'])
    lg = lg - lg.max(-1, keepdims=True)
    pr = np.exp(lg)
    pr = pr / pr.sum(-1, keepdims=True)
    valid = (Y >= 0)
    nv = max(1, int(valid.sum()))
    Yc = np.where(valid, Y, 0)
    oh = (Yc[..., None] == np.arange(V)[None, None, :]).astype(np.float64)
    dlog = (pr - oh) * (valid[..., None].astype(np.float64) / np.float64(nv))
    g = {}
    g['O'] = gw(h2, dlog)
    dh2 = mm(dlog, p['O'].T)
    g['U2'] = gw(M2, dh2)
    dZ2 = mm(dh2, p['U2'].T) * (Z2 > 0)
    g['V2'] = gw(G2, dZ2)
    dG2 = dh2 + mm(dZ2, p['V2'].T)
    g['Wo2'] = gw(R2, dG2)
    dR2 = mm(dG2, p['Wo2'].T)
    dhq = dG2
    dW2 = dR2 @ C2.transpose(0, 2, 1)
    dC2 = W2.transpose(0, 2, 1) @ dR2
    dS2 = W2 * (dW2 - (dW2 * W2).sum(-1, keepdims=True)) * sc
    dP2 = dS2 @ A2
    dA2 = dS2.transpose(0, 2, 1) @ P2
    g['Wp2'] = gw(hq, dP2)
    dhq = dhq + mm(dP2, p['Wp2'].T)
    g['Wa2'] = gw(h1, dA2)
    g['Wc2'] = gw(h1, dC2)
    dh1 = mm(dA2, p['Wa2'].T) + mm(dC2, p['Wc2'].T)
    dh1[:, qs] += dhq
    g['U1'] = gw(M1, dh1)
    dZ1 = mm(dh1, p['U1'].T) * (Z1 > 0)
    g['V1'] = gw(G1, dZ1)
    dG1 = dh1 + mm(dZ1, p['V1'].T)
    g['Wo1'] = gw(R1, dG1)
    dR1 = mm(dG1, p['Wo1'].T)
    dW1 = dR1 @ C1.transpose(0, 2, 1)
    dC1 = W1.transpose(0, 2, 1) @ dR1
    dS1 = W1 * (dW1 - (dW1 * W1).sum(-1, keepdims=True)) * sc
    dP1 = dS1 @ A1
    dA1 = dS1.transpose(0, 2, 1) @ P1
    g['Wp1'] = gw(h0, dP1)
    g['Wa1'] = gw(h0, dA1)
    g['Wc1'] = gw(h0, dC1)
    dh0 = dG1 + mm(dP1, p['Wp1'].T) + mm(dA1, p['Wa1'].T) + mm(dC1, p['Wc1'].T)
    ohx = (X.reshape(-1)[:, None] == np.arange(V)[None, :]).astype(np.float64)
    g['E'] = ohx.T @ dh0.reshape(-1, d)
    gP = np.zeros_like(p['P'])
    gP[:L] = dh0.sum(0)
    g['P'] = gP
    return g


def learn(layout, examples, seed):
    V = int(layout['vocab'])
    d = int(RECIPE['width_d'])
    rng = np.random.default_rng(seed)
    n = len(examples)
    lens = [len(ex['tokens']) for ex in examples]
    Lp = max(lens + [5 * int(layout.get('size', 0)) + 3]) + 8
    p = make_params(V, Lp, d, rng)
    if n == 0:
        return p
    L = max(lens)
    X = np.zeros((n, L), dtype=np.int64)
    qd = {}
    for i in range(n):
        tk = examples[i]['tokens']
        X[i, :len(tk)] = np.array(tk, dtype=np.int64)
        for q in examples[i]['queries']:
            qd[int(q[0])] = 1
    keys = sorted(qd.keys())
    if len(keys) == 0:
        return p
    qs = np.array(keys, dtype=np.int64)
    K = len(keys)
    kidx = {}
    for k in range(K):
        kidx[keys[k]] = k
    Y = -np.ones((n, K), dtype=np.int64)
    for i in range(n):
        for q in examples[i]['queries']:
            a = int(q[1])
            if a >= 0 and a < V:
                Y[i, kidx[int(q[0])]] = a
    mask1 = np.triu(np.ones((L, L), dtype=np.float64), 1) * np.float64(-1e9)
    mask2 = (np.arange(L)[None, :] > qs[:, None]).astype(np.float64) * np.float64(-1e9)
    names = list(p.keys())
    m1 = {}
    m2 = {}
    for k in names:
        m1[k] = np.zeros_like(p[k])
        m2[k] = np.zeros_like(p[k])
    B = int(RECIPE['batch_size'])
    E = int(RECIPE['epochs'])
    peak = float(RECIPE['learning_rate'])
    beta1 = float(RECIPE['betas'][0])
    beta2 = float(RECIPE['betas'][1])
    eps = float(RECIPE['eps'])
    wd = float(RECIPE['weight_decay'])
    warm = float(RECIPE['warmup_steps'])
    clip = float(RECIPE['grad_clip_norm'])
    spe = (n + B - 1) // B
    total = float(E * spe)
    step = 0
    for ep in range(E):
        perm = rng.permutation(n)
        for b0 in range(0, n, B):
            idx = perm[b0:b0 + B]
            g = grads(p, X[idx], Y[idx], qs, mask1, mask2, d, V)
            step += 1
            prog = min(1.0, step / total)
            lr = float(peak * min(1.0, step / warm) * (0.05 + 0.95 * 0.5 * (1.0 + float(np.cos(np.pi * prog)))))
            gn = float(np.sqrt(sum([float(np.sum(g[k] * g[k])) for k in names])))
            cf = float(min(1.0, clip / (gn + 1e-6)))
            b1c = float(1.0 - beta1 ** step)
            b2c = float(1.0 - beta2 ** step)
            for k in names:
                gk = g[k] * np.float64(cf)
                m1[k] = beta1 * m1[k] + (1.0 - beta1) * gk
                m2[k] = beta2 * m2[k] + (1.0 - beta2) * (gk * gk)
                upd = (m1[k] / b1c) / (np.sqrt(m2[k] / b2c) + eps)
                p[k] = (p[k] * np.float64(1.0 - lr * wd) - np.float64(lr) * upd).astype(np.float64)
    return p


def init(layout, params):
    d = params['E'].shape[1]
    return {'p': params, 'd': d, 't': 0, 'A1': [], 'C1': [], 'A2': [], 'C2': [], 'h': None}


def update(state, token):
    p = state['p']
    d = state['d']
    t = state['t']
    prow = p['P'][min(t, p['P'].shape[0] - 1)]
    h = p['E'][int(token)] + prow
    sc = 1.0 / np.sqrt(d)
    for l in ['1', '2']:
        a = h @ p['Wa' + l]
        c = h @ p['Wc' + l]
        q = h @ p['Wp' + l]
        state['A' + l].append(a)
        state['C' + l].append(c)
        A = np.array(state['A' + l])
        C = np.array(state['C' + l])
        s = (A @ q) * sc
        s = s - s.max()
        w = np.exp(s)
        w = w / w.sum()
        r = w @ C
        gv = h + r @ p['Wo' + l]
        h = gv + np.maximum(gv @ p['V' + l], 0) @ p['U' + l]
    state['h'] = h
    state['t'] = t + 1
    return state


def predict(state):
    lg = state['h'] @ state['p']['O']
    return int(np.argmax(lg))
