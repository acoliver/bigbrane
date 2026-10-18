RECIPE = {'init': 'gaussian: E,P std 0.5; Wa,Wc,Wp,Wo,O std 1/sqrt(d); U std 0.5/sqrt(d); V std sqrt(2/d)', 'optimizer': 'adamw', 'lr': 0.003, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'weight_decay': 0.01, 'warmup_steps': 300, 'schedule': 'cosine', 'min_lr_frac': 0.05, 'epochs': 300, 'batch_size': 100, 'clip_norm': 1.0, 'loss': 'next-symbol cross-entropy at query positions, through both rounds', 'd': 64, 'rounds': 2}
CURVE = []
KEYS = [['Wa0', 'Wc0', 'Wp0', 'Wo0', 'U0', 'V0'], ['Wa1', 'Wc1', 'Wp1', 'Wo1', 'U1', 'V1']]
AK = ['A0', 'A1']
CK = ['C0', 'C1']
DMODEL = 64


def positions_for(layout, examples):
    L = int(layout['size'])
    if 'SEP' in layout:
        P = 2 * L
    elif 'QSEP' in layout:
        P = 3 * L + 1
    elif 'OPEN' in layout:
        P = 4 * L
    else:
        P = 1
    for ex in examples:
        if len(ex['tokens']) > P:
            P = len(ex['tokens'])
    return P


def init_params(nv, npos, d, rng):
    s = 1.0 / np.sqrt(d)
    p = {'E': rng.normal(0.0, 0.5, (nv, d)), 'P': rng.normal(0.0, 0.5, (npos, d)), 'O': rng.normal(0.0, s, (d, nv))}
    for l in range(2):
        k = KEYS[l]
        p[k[0]] = rng.normal(0.0, s, (d, d))
        p[k[1]] = rng.normal(0.0, s, (d, d))
        p[k[2]] = rng.normal(0.0, s, (d, d))
        p[k[3]] = rng.normal(0.0, s, (d, d))
        p[k[4]] = rng.normal(0.0, 0.5 * s, (d, d))
        p[k[5]] = rng.normal(0.0, np.sqrt(2.0) * s, (d, d))
    return p


def forward(p, X, d):
    B, T = X.shape
    caus = np.tril(np.ones((T, T), dtype=bool))
    neg = np.where(caus, 0.0, -1e30)
    h = p['E'][X] + p['P'][None, :T, :]
    sc = 1.0 / np.sqrt(d)
    caches = []
    for l in range(2):
        k = KEYS[l]
        W3 = np.concatenate([p[k[0]], p[k[1]], p[k[2]]], axis=1)
        hf = h.reshape(B * T, d)
        acp = hf @ W3
        A = acp[:, :d].reshape(B, T, d)
        C = acp[:, d:2 * d].reshape(B, T, d)
        Q = acp[:, 2 * d:].reshape(B, T, d)
        S = np.matmul(Q, A.transpose(0, 2, 1)) * sc + neg
        S = S - S.max(axis=2, keepdims=True)
        Wt = np.exp(S)
        Wt = Wt / Wt.sum(axis=2, keepdims=True)
        R = np.matmul(Wt, C)
        Rf = R.reshape(B * T, d)
        g = hf + Rf @ p[k[3]]
        z = g @ p[k[5]]
        m = np.maximum(z, 0.0)
        hn = g + m @ p[k[4]]
        caches.append((hf, A, C, Q, Wt, Rf, g, z, m, W3))
        h = hn.reshape(B, T, d)
    return h, caches


def backward(p, X, caches, dh, d):
    B, T = X.shape
    grads = {}
    sc = 1.0 / np.sqrt(d)
    for l in [1, 0]:
        hf, A, C, Q, Wt, Rf, g, z, m, W3 = caches[l]
        k = KEYS[l]
        dhf = dh.reshape(B * T, d)
        grads[k[4]] = m.T @ dhf
        dz = (dhf @ p[k[4]].T) * (z > 0)
        grads[k[5]] = g.T @ dz
        dg = dhf + dz @ p[k[5]].T
        grads[k[3]] = Rf.T @ dg
        dR = (dg @ p[k[3]].T).reshape(B, T, d)
        dWt = np.matmul(dR, C.transpose(0, 2, 1))
        dC = np.matmul(Wt.transpose(0, 2, 1), dR)
        dS = Wt * (dWt - (dWt * Wt).sum(axis=2, keepdims=True)) * sc
        dQ = np.matmul(dS, A)
        dA = np.matmul(dS.transpose(0, 2, 1), Q)
        dacp = np.concatenate([dA.reshape(B * T, d), dC.reshape(B * T, d), dQ.reshape(B * T, d)], axis=1)
        G3 = hf.T @ dacp
        grads[k[0]] = G3[:, :d]
        grads[k[1]] = G3[:, d:2 * d]
        grads[k[2]] = G3[:, 2 * d:]
        dh = (dg + dacp @ W3.T).reshape(B, T, d)
    nv = p['E'].shape[0]
    oh = (X.reshape(-1)[:, None] == np.arange(nv)[None, :]).astype(np.float64)
    grads['E'] = oh.T @ dh.reshape(B * T, d)
    gP = np.zeros_like(p['P'])
    gP[:T] = dh.sum(axis=0)
    grads['P'] = gP
    return grads


def loss_and_grads(p, Xb, bi, ti, ans, d):
    hT, caches = forward(p, Xb, d)
    hq = hT[bi, ti]
    lg = hq @ p['O']
    lg = lg - lg.max(axis=1, keepdims=True)
    e = np.exp(lg)
    pr = e / e.sum(axis=1, keepdims=True)
    nq = len(bi)
    ar = np.arange(nq)
    loss = float(-np.log(pr[ar, ans] + 1e-300).mean())
    corr = float((lg.argmax(axis=1) == ans).sum())
    dl = pr.copy()
    dl[ar, ans] -= 1.0
    dl = dl / nq
    gO = hq.T @ dl
    dh = np.zeros_like(hT)
    dh[bi, ti] = dl @ p['O'].T
    grads = backward(p, Xb, caches, dh, d)
    grads['O'] = gO
    return loss, corr, grads


def learn(layout, examples, seed):
    d = DMODEL
    nv = int(layout['vocab'])
    npos = positions_for(layout, examples)
    rng = np.random.default_rng(int(seed) + 12345)
    p = init_params(nv, npos, d, rng)
    n = len(examples)
    if n == 0:
        return p
    T = max([len(ex['tokens']) for ex in examples])
    X = np.zeros((n, T), dtype=np.int64)
    M = np.zeros((n, T), dtype=bool)
    Y = np.zeros((n, T), dtype=np.int64)
    for i, ex in enumerate(examples):
        tk = ex['tokens']
        X[i, :len(tk)] = np.array(tk, dtype=np.int64)
        for q in ex['queries']:
            M[i, int(q[0])] = True
            Y[i, int(q[0])] = min(max(int(q[1]), 0), nv - 1)
    bs = int(RECIPE['batch_size'])
    epochs = int(RECIPE['epochs'])
    peak = float(RECIPE['lr'])
    warm = int(RECIPE['warmup_steps'])
    minf = float(RECIPE['min_lr_frac'])
    b1 = float(RECIPE['beta1'])
    b2 = float(RECIPE['beta2'])
    eps = float(RECIPE['eps'])
    wd = float(RECIPE['weight_decay'])
    clip = float(RECIPE['clip_norm'])
    spe = (n + bs - 1) // bs
    total = epochs * spe
    keys = list(p.keys())
    mom = {}
    vel = {}
    for k in keys:
        mom[k] = np.zeros_like(p[k])
        vel[k] = np.zeros_like(p[k])
    step = 0
    lr = 0.0
    every = max(1, epochs // 6)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tl = 0.0
        tc = 0.0
        tn = 0.0
        for s0 in range(0, n, bs):
            idx = perm[s0:s0 + bs]
            Xb = X[idx]
            Mb = M[idx]
            Yb = Y[idx]
            bi, ti = np.nonzero(Mb)
            if len(bi) == 0:
                continue
            loss, corr, grads = loss_and_grads(p, Xb, bi, ti, Yb[bi, ti], d)
            gn = np.sqrt(sum([float((grads[k] * grads[k]).sum()) for k in keys]))
            c = 1.0
            if gn > clip:
                c = clip / gn
            step += 1
            if step <= warm:
                lr = peak * step / warm
            else:
                frac = (step - warm) / max(1, total - warm)
                lr = peak * (minf + (1.0 - minf) * 0.5 * (1.0 + np.cos(np.pi * min(frac, 1.0))))
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step
            for k in keys:
                gk = grads[k] * c
                mom[k] = b1 * mom[k] + (1.0 - b1) * gk
                vel[k] = b2 * vel[k] + (1.0 - b2) * gk * gk
                p[k] = p[k] - lr * ((mom[k] / bc1) / (np.sqrt(vel[k] / bc2) + eps) + wd * p[k])
            tl += loss * len(bi)
            tc += corr
            tn += len(bi)
        if (ep + 1) % every == 0 or ep == epochs - 1:
            CURVE.append({'epoch': ep + 1, 'loss': float(tl / max(tn, 1.0)), 'train_accuracy': float(tc / max(tn, 1.0)), 'lr': float(lr)})
    return p


def init(layout, params):
    d = params['E'].shape[1]
    st = {'p': params, 't': 0, 'd': d, 'h': np.zeros(d)}
    for l in range(2):
        st[AK[l]] = np.zeros((0, d))
        st[CK[l]] = np.zeros((0, d))
    return st


def update(state, token):
    p = state['p']
    d = state['d']
    t = min(state['t'], p['P'].shape[0] - 1)
    h = p['E'][int(token)] + p['P'][t]
    for l in range(2):
        k = KEYS[l]
        a = h @ p[k[0]]
        c = h @ p[k[1]]
        q = h @ p[k[2]]
        A = np.concatenate([state[AK[l]], a[None, :]], axis=0)
        C = np.concatenate([state[CK[l]], c[None, :]], axis=0)
        s = (A @ q) / np.sqrt(d)
        s = s - s.max()
        w = np.exp(s)
        w = w / w.sum()
        r = w @ C
        g = h + r @ p[k[3]]
        h = g + np.maximum(g @ p[k[5]], 0.0) @ p[k[4]]
        state[AK[l]] = A
        state[CK[l]] = C
    state['h'] = h
    state['t'] = state['t'] + 1
    return state


def predict(state):
    lg = state['h'] @ state['p']['O']
    return int(np.argmax(lg))
