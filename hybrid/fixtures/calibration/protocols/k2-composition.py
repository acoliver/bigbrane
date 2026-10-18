RECIPE = {'d': 64, 'init': 'E~N(0,0.5^2); W~N(0,1/(2d)); O~N(0,1/d); biases 0 except retain bias 1', 'optimizer': 'adam(0.9,0.999,1e-8), global-norm clip 1.0', 'lr': 0.004, 'lr_schedule': 'cosine to 0', 'epochs': 300, 'batch_size': 128, 'loss': 'next-symbol cross-entropy at query positions, full BPTT', 'emb_scale': 0.5, 'retain_bias': 1.0, 'clip': 1.0, 'augment': 'each epoch, re-shuffle the order of the g pairs, the f pairs and the queries (with their answers) inside every training instance'}
CURVE = []

def sigm(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def pack(E, W, b, O, d):
    P = {'E': np.ascontiguousarray(E).astype(np.float32), 'O': np.ascontiguousarray(O).astype(np.float32)}
    names = ['r', 'i', 'o', 'u']
    for k in range(4):
        nm = names[k]
        P['Wy_' + nm] = np.ascontiguousarray(W[:d, k * d:(k + 1) * d]).astype(np.float32)
        P['We_' + nm] = np.ascontiguousarray(W[d:, k * d:(k + 1) * d]).astype(np.float32)
        P['b_' + nm] = np.ascontiguousarray(b[k * d:(k + 1) * d]).astype(np.float32)
    return P

def perm_pairs(X, start, K, rng):
    n = X.shape[0]
    pp = np.argsort(rng.rand(n, K), axis=1)
    P = X[:, start:start + 2 * K].reshape(n, K, 2)
    idx = np.repeat(pp[:, :, None], 2, axis=2)
    P2 = np.take_along_axis(P, idx, axis=1)
    X[:, start:start + 2 * K] = P2.reshape(n, 2 * K)

def perm_queries(X, A, M, start, K, rng):
    n = X.shape[0]
    qp = np.argsort(rng.rand(n, K), axis=1)
    X[:, start:start + K] = np.take_along_axis(X[:, start:start + K], qp, axis=1)
    A[:, start:start + K] = np.take_along_axis(A[:, start:start + K], qp, axis=1)
    M[:, start:start + K] = np.take_along_axis(M[:, start:start + K], qp, axis=1)

def augment(X, A, M, layout, ok, rng):
    X2 = X.copy()
    A2 = A.copy()
    M2 = M.copy()
    if not ok:
        return X2, A2, M2
    K = int(layout['size'])
    if 'GSEP' in layout:
        perm_pairs(X2, 1, K, rng)
        perm_pairs(X2, 2 * K + 2, K, rng)
        perm_queries(X2, A2, M2, 4 * K + 3, K, rng)
    else:
        perm_pairs(X2, 0, K, rng)
        perm_queries(X2, A2, M2, 2 * K + 1, K, rng)
    return X2, A2, M2

def fb(E, W, b, O, xb, ab, mb, d, V):
    B = xb.shape[0]
    T = xb.shape[1]
    # W stacks the recurrent block Wy (rows 0..d) over the input block We (rows d..2d). The input half of every
    # gate preactivation takes only V distinct embeddings, so it is a table (E @ We) gathered by token.
    Wy = W[:d]
    We = W[d:]
    Pe = (E @ We)[xb] + b
    Yp = np.zeros((B, T, d), np.float32)
    R = np.zeros((B, T, d), np.float32)
    I = np.zeros((B, T, d), np.float32)
    Og = np.zeros((B, T, d), np.float32)
    U = np.zeros((B, T, d), np.float32)
    SP = np.zeros((B, T, d), np.float32)
    TS = np.zeros((B, T, d), np.float32)
    Y = np.zeros((B, T, d), np.float32)
    y = np.zeros((B, d), np.float32)
    s = np.zeros((B, d), np.float32)
    for t in range(T):
        Yp[:, t] = y
        pre = y @ Wy + Pe[:, t]
        r = sigm(pre[:, :d])
        ig = sigm(pre[:, d:2 * d])
        o = sigm(pre[:, 2 * d:3 * d])
        u = np.tanh(pre[:, 3 * d:])
        SP[:, t] = s
        s = r * s + ig * u
        ts = np.tanh(s)
        y = o * ts
        R[:, t] = r
        I[:, t] = ig
        Og[:, t] = o
        U[:, t] = u
        TS[:, t] = ts
        Y[:, t] = y
    N = B * T
    Yf = Y.reshape(N, d)
    lg = Yf @ O
    lg = lg - lg.max(1, keepdims=True)
    ex = np.exp(lg)
    p = ex / ex.sum(1, keepdims=True)
    af = ab.reshape(-1)
    mf = mb.reshape(-1)
    nq = max(1.0, float(mf.sum()))
    rows = np.arange(N)
    loss = -float((np.log(p[rows, af] + 1e-12) * mf).sum()) / nq
    corr = float(((np.argmax(lg, 1) == af) * mf).sum())
    dl = p.copy()
    dl[rows, af] -= 1.0
    dl = dl * (mf / nq)[:, None]
    dl = dl.astype(np.float32)
    dO = Yf.T @ dl
    dY = (dl @ O.T).reshape(B, T, d)
    DP = np.zeros((B, T, 4 * d), np.float32)
    dyr = np.zeros((B, d), np.float32)
    dsr = np.zeros((B, d), np.float32)
    WyT = np.ascontiguousarray(Wy.T)
    for t in reversed(range(T)):
        r = R[:, t]
        ig = I[:, t]
        o = Og[:, t]
        u = U[:, t]
        ts = TS[:, t]
        dy = dY[:, t] + dyr
        do = dy * ts
        ds = dsr + dy * o * (1.0 - ts * ts)
        DP[:, t, :d] = ds * SP[:, t] * r * (1.0 - r)
        DP[:, t, d:2 * d] = ds * u * ig * (1.0 - ig)
        DP[:, t, 2 * d:3 * d] = do * o * (1.0 - o)
        DP[:, t, 3 * d:] = ds * ig * (1.0 - u * u)
        dsr = ds * r
        dyr = DP[:, t] @ WyT
    DPf = DP.reshape(N, 4 * d)
    OH = (xb.reshape(-1)[:, None] == np.arange(V)[None, :]).astype(np.float32)
    G = OH.T @ DPf
    dW = np.concatenate([Yp.reshape(N, d).T @ DPf, E.T @ G], axis=0)
    db = DPf.sum(0)
    dE = G @ We.T
    return [dE, dW, db, dO], loss, corr, float(mf.sum())

def learn(layout, examples, seed):
    d = int(RECIPE['d'])
    V = int(layout['vocab'])
    rng = np.random.RandomState(int(seed) + 12345)
    E = (rng.randn(V, d) * RECIPE['emb_scale']).astype(np.float32)
    W = (rng.randn(2 * d, 4 * d) / np.sqrt(2.0 * d)).astype(np.float32)
    b = np.zeros(4 * d, np.float32)
    b[0:d] = RECIPE['retain_bias']
    O = (rng.randn(d, V) / np.sqrt(1.0 * d)).astype(np.float32)
    n = len(examples)
    if n == 0:
        return pack(E, W, b, O, d)
    lens = [len(ex['tokens']) for ex in examples]
    Tm = max(lens)
    K = int(layout['size'])
    if 'GSEP' in layout:
        expect = 5 * K + 3
    else:
        expect = 3 * K + 1
    ok = (min(lens) == Tm) and (Tm == expect)
    X = np.zeros((n, Tm), np.int64)
    A = np.zeros((n, Tm), np.int64)
    M = np.zeros((n, Tm), np.float32)
    for k, ex in enumerate(examples):
        tk = ex['tokens']
        X[k, :len(tk)] = np.array(tk, np.int64)
        for q in ex['queries']:
            A[k, int(q[0])] = int(q[1])
            M[k, int(q[0])] = 1.0
    params = [E, W, b, O]
    mom = [np.zeros_like(q) for q in params]
    vel = [np.zeros_like(q) for q in params]
    B = int(RECIPE['batch_size'])
    epochs = int(RECIPE['epochs'])
    nb = (n + B - 1) // B
    total = float(epochs * nb)
    step = 0
    base = float(RECIPE['lr'])
    for ep in range(epochs):
        Xe, Ae, Me = augment(X, A, M, layout, ok, rng)
        perm = rng.permutation(n)
        tl = 0.0
        tc = 0.0
        tq = 0.0
        for bi in range(nb):
            idx = perm[bi * B:(bi + 1) * B]
            grads, loss, corr, nq = fb(params[0], params[1], params[2], params[3], Xe[idx], Ae[idx], Me[idx], d, V)
            tl += loss
            tc += corr
            tq += nq
            gn = float(np.sqrt(sum([float((g * g).sum()) for g in grads])))
            sc = 1.0
            if gn > RECIPE['clip']:
                sc = RECIPE['clip'] / (gn + 1e-12)
            lr = base * 0.5 * (1.0 + float(np.cos(np.pi * step / total)))
            step += 1
            c1 = 1.0 - 0.9 ** step
            c2 = 1.0 - 0.999 ** step
            for k in range(4):
                g = (grads[k] * sc).astype(np.float32)
                mom[k] *= 0.9
                mom[k] += 0.1 * g
                vel[k] *= 0.999
                vel[k] += 0.001 * g * g
                params[k] -= ((lr / c1) * mom[k] / (np.sqrt(vel[k] / c2) + 1e-8)).astype(np.float32)
        if ep % 25 == 0 or ep == epochs - 1:
            CURVE.append({'epoch': ep, 'loss': float(tl / nb), 'train_accuracy': float(tc / max(1.0, tq))})
    return pack(params[0], params[1], params[2], params[3], d)

def init(layout, params):
    d = params['E'].shape[1]
    return {'P': params, 's': np.zeros(d, np.float32), 'y': np.zeros(d, np.float32)}

def update(state, token):
    P = state['P']
    y = state['y']
    s = state['s']
    e = P['E'][int(token)]
    r = sigm(y @ P['Wy_r'] + e @ P['We_r'] + P['b_r'])
    ig = sigm(y @ P['Wy_i'] + e @ P['We_i'] + P['b_i'])
    o = sigm(y @ P['Wy_o'] + e @ P['We_o'] + P['b_o'])
    u = np.tanh(y @ P['Wy_u'] + e @ P['We_u'] + P['b_u'])
    s2 = (r * s + ig * u).astype(np.float32)
    y2 = (o * np.tanh(s2)).astype(np.float32)
    return {'P': P, 's': s2, 'y': y2}

def predict(state):
    lg = state['y'] @ state['P']['O']
    return int(np.argmax(lg))
