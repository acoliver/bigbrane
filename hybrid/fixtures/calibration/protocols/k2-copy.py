RECIPE = {
    'model': 'card store: sigmoid retain/admit/release switches and tanh candidate on [y_prev; e_t], readout y O, d=64, s and y start at zero',
    'init': 'gate matrices normal(0, 1/sqrt(2d)), E normal(0,1), O normal(0,1/sqrt(d)), offsets 0 except retain offset 1.0',
    'init_scale': 1.0,
    'retain_bias': 1.0,
    'd': 64,
    'optimizer': 'adam',
    'lr': 0.005,
    'lr_final': 0.0003,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-08,
    'clip_norm': 1.0,
    'epochs': 300,
    'batch_size': 50,
    'loss': 'next-symbol cross-entropy at the scored query positions, full backpropagation through time',
    'augment': 'none',
    'stop_patience': 5,
    'stop_loss': 0.005,
}
CURVE = []

def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

def prep(examples):
    n = len(examples)
    T = 1
    for ex in examples:
        for q in ex['queries']:
            if int(q[0]) + 1 > T:
                T = int(q[0]) + 1
    X = np.zeros((n, T), dtype=np.int64)
    Y = np.zeros((n, T), dtype=np.int64)
    M = np.zeros((n, T))
    for a in range(n):
        tk = list(examples[a]['tokens'])
        k = min(len(tk), T)
        X[a, :k] = np.asarray(tk[:k], dtype=np.int64)
        for q in examples[a]['queries']:
            Y[a, int(q[0])] = int(q[1])
            M[a, int(q[0])] = 1.0
    return X, Y, M

def init_params(V, d, rng):
    sc = float(RECIPE['init_scale']) / np.sqrt(2.0 * d)
    W = rng.normal(0.0, sc, (2 * d, 4 * d))
    b = np.zeros(4 * d)
    b[:d] = float(RECIPE['retain_bias'])
    E = rng.normal(0.0, 1.0, (V, d))
    O = rng.normal(0.0, 1.0 / np.sqrt(d), (d, V))
    return {'W': W, 'b': b, 'E': E, 'O': O}

def fb(P, X, Y, M):
    W = P['W']
    b = P['b']
    E = P['E']
    O = P['O']
    B = X.shape[0]
    T = X.shape[1]
    V = E.shape[0]
    d = E.shape[1]
    Z = np.zeros((T, B, 2 * d))
    G = np.zeros((T, B, 4 * d))
    SP = np.zeros((T, B, d))
    TS = np.zeros((T, B, d))
    YS = np.zeros((T, B, d))
    y = np.zeros((B, d))
    s = np.zeros((B, d))
    for t in range(T):
        Z[t, :, :d] = y
        Z[t, :, d:] = E[X[:, t]]
        a = Z[t] @ W + b
        g = G[t]
        g[:, :3 * d] = sigm(a[:, :3 * d])
        g[:, 3 * d:] = np.tanh(a[:, 3 * d:])
        SP[t] = s
        s = g[:, :d] * s + g[:, d:2 * d] * g[:, 3 * d:]
        ts = np.tanh(s)
        TS[t] = ts
        y = g[:, 2 * d:3 * d] * ts
        YS[t] = y
    N = T * B
    Yf = YS.reshape(N, d)
    lg = Yf @ O
    lg = lg - lg.max(axis=1, keepdims=True)
    p = np.exp(lg)
    p = p / p.sum(axis=1, keepdims=True)
    Mt = M.T.reshape(N)
    Yt = Y.T.reshape(N)
    Xt = X.T.reshape(N)
    cnt = max(float(Mt.sum()), 1.0)
    idx = np.arange(N)
    loss = float(-(np.log(p[idx, Yt] + 1e-12) * Mt).sum() / cnt)
    correct = float(((np.argmax(p, axis=1) == Yt) * Mt).sum())
    dl = p.copy()
    dl[idx, Yt] -= 1.0
    dl = dl * (Mt / cnt)[:, None]
    dO = Yf.T @ dl
    DY = (dl @ O.T).reshape(T, B, d)
    DA = np.zeros((T, B, 4 * d))
    Wy = W[:d]
    dyn = np.zeros((B, d))
    dsn = np.zeros((B, d))
    for t in reversed(range(T)):
        g = G[t]
        r = g[:, :d]
        i = g[:, d:2 * d]
        o = g[:, 2 * d:3 * d]
        u = g[:, 3 * d:]
        ts = TS[t]
        dy = DY[t] + dyn
        ds = dy * o * (1.0 - ts * ts) + dsn
        da = DA[t]
        da[:, :d] = ds * SP[t] * r * (1.0 - r)
        da[:, d:2 * d] = ds * u * i * (1.0 - i)
        da[:, 2 * d:3 * d] = dy * ts * o * (1.0 - o)
        da[:, 3 * d:] = ds * i * (1.0 - u * u)
        dsn = ds * r
        dyn = da @ Wy.T
    DAf = DA.reshape(N, 4 * d)
    dW = Z.reshape(N, 2 * d).T @ DAf
    db = DAf.sum(axis=0)
    DE = DAf @ W[d:].T
    oh = np.zeros((N, V))
    oh[idx, Xt] = 1.0
    dE = oh.T @ DE
    return loss, correct, {'W': dW, 'b': db, 'E': dE, 'O': dO}

def export(P):
    d = P['E'].shape[1]
    W = P['W']
    b = P['b']
    out = {}
    nm = ['r', 'i', 'o', 'u']
    for k in range(4):
        out['W' + nm[k] + 'y'] = np.ascontiguousarray(W[:d, k * d:(k + 1) * d]).astype(np.float32)
        out['W' + nm[k] + 'e'] = np.ascontiguousarray(W[d:, k * d:(k + 1) * d]).astype(np.float32)
        out['b' + nm[k]] = np.ascontiguousarray(b[k * d:(k + 1) * d]).astype(np.float32)
    out['E'] = P['E'].astype(np.float32)
    out['O'] = P['O'].astype(np.float32)
    return out

def learn(layout, examples, seed):
    V = int(layout['vocab'])
    d = int(RECIPE['d'])
    rng = np.random.default_rng(7919 + int(seed))
    P = init_params(V, d, rng)
    if len(examples) == 0:
        CURVE.append({'epoch': 0, 'loss': 0.0, 'train_accuracy': 0.0})
        return export(P)
    X, Y, M = prep(examples)
    n = X.shape[0]
    bs = min(int(RECIPE['batch_size']), n)
    nb = (n + bs - 1) // bs
    epochs = int(RECIPE['epochs'])
    total = float(epochs * nb)
    aug = RECIPE['augment'] == 'relabel'
    kb = 0
    nk = 0
    vb = 0
    nv = 0
    if aug:
        kb = int(layout['key_base'])
        nk = int(layout['n_keys'])
        vb = int(layout['value_base'])
        nv = int(layout['V'])
    mom = {}
    vel = {}
    for k in P:
        mom[k] = np.zeros_like(P[k])
        vel[k] = np.zeros_like(P[k])
    b1 = float(RECIPE['beta1'])
    b2 = float(RECIPE['beta2'])
    eps = float(RECIPE['eps'])
    lr0 = float(RECIPE['lr'])
    lrf = float(RECIPE['lr_final'])
    clip = float(RECIPE['clip_norm'])
    step = 0
    streak = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        tl = 0.0
        tc = 0.0
        tq = 0.0
        for j in range(nb):
            ids = order[j * bs:(j + 1) * bs]
            xb = X[ids]
            yb = Y[ids]
            mb = M[ids]
            if aug:
                B = xb.shape[0]
                mp = np.tile(np.arange(V), (B, 1))
                mp[:, kb:kb + nk] = np.argsort(rng.random((B, nk)), axis=1) + kb
                mp[:, vb:vb + nv] = np.argsort(rng.random((B, nv)), axis=1) + vb
                rows = np.arange(B)[:, None]
                xb = mp[rows, xb]
                yb = mp[rows, yb]
            loss, corr, gr = fb(P, xb, yb, mb)
            gn = 0.0
            for k in gr:
                gn += float((gr[k] * gr[k]).sum())
            gn = float(np.sqrt(gn))
            sc = 1.0
            if gn > clip:
                sc = clip / gn
            lr = lr0 * (lrf / lr0) ** (step / total)
            step += 1
            for k in P:
                gk = gr[k] * sc
                mom[k] = b1 * mom[k] + (1.0 - b1) * gk
                vel[k] = b2 * vel[k] + (1.0 - b2) * gk * gk
                mh = mom[k] / (1.0 - b1 ** step)
                vh = vel[k] / (1.0 - b2 ** step)
                P[k] = P[k] - lr * mh / (np.sqrt(vh) + eps)
            q = float(mb.sum())
            tl += loss * q
            tc += corr
            tq += q
        acc = tc / max(tq, 1.0)
        ml = tl / max(tq, 1.0)
        if ep % 10 == 0:
            CURVE.append({'epoch': ep, 'loss': float(ml), 'train_accuracy': float(acc)})
        if acc >= 1.0 and ml < float(RECIPE['stop_loss']):
            streak += 1
        else:
            streak = 0
        if streak >= int(RECIPE['stop_patience']):
            CURVE.append({'epoch': ep, 'loss': float(ml), 'train_accuracy': float(acc)})
            break
    return export(P)

def init(layout, params):
    d = params['E'].shape[1]
    return {'p': params, 's': np.zeros(d, dtype=np.float32), 'y': np.zeros(d, dtype=np.float32)}

def update(state, token):
    P = state['p']
    y = state['y']
    e = P['E'][int(token)]
    r = sigm(y @ P['Wry'] + e @ P['Wre'] + P['br'])
    i = sigm(y @ P['Wiy'] + e @ P['Wie'] + P['bi'])
    o = sigm(y @ P['Woy'] + e @ P['Woe'] + P['bo'])
    u = np.tanh(y @ P['Wuy'] + e @ P['Wue'] + P['bu'])
    s = (r * state['s'] + i * u).astype(np.float32)
    yn = (o * np.tanh(s)).astype(np.float32)
    return {'p': P, 's': s, 'y': yn}

def predict(state):
    lg = state['y'] @ state['p']['O']
    return int(np.argmax(lg))
