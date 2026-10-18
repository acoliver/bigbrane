RECIPE = {"init": "weights uniform(-1/sqrt(d),1/sqrt(d)); E normal(0,0.5); retain bias 1, other biases 0, no output bias; s,y start at 0", "optimizer": "AdamW b1=0.9 b2=0.999 eps=1e-8, decoupled weight decay on matrices, global gradient-norm clip", "learning_rate": 0.004, "schedule": "cosine decay to 0.1x over all steps", "weight_decay": 0.0005, "epochs": 300, "batch_size": 128, "loss": "next-symbol cross-entropy at the scored positions, full backpropagation through time", "d": 128, "emb_scale": 0.5, "retain_bias": 1.0, "clip": 1.0, "augmentation": "per example and per batch, a fresh random relabeling of token ids within the key-id class and within the value-id class, applied consistently to the input tokens and the answers"}
CURVE = []

def sigm(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def relabel(rng, tb, gb, groups, V):
    B = tb.shape[0]
    mp = np.tile(np.arange(V, dtype=np.int64), (B, 1))
    for g in groups:
        base = g[0]
        cnt = g[1]
        mp[:, base:base + cnt] = np.argsort(rng.random((B, cnt)), axis=1) + base
    nt = np.take_along_axis(mp, tb, 1)
    ng = np.where(gb >= 0, np.take_along_axis(mp, np.maximum(gb, 0), 1), gb)
    return nt, ng

def grad(P, tok, tgt, msk, eye, d):
    E, We, Wy, b, O = P
    f32 = np.float32
    B, T = tok.shape
    # The input projection E[tok] @ We takes only V distinct rows, so it is a table (E @ We) gathered by token.
    OH = eye[tok.reshape(-1)]
    Ze = (E @ We)[tok] + b
    Yall = np.zeros((B, T + 1, d), f32)
    Sall = np.zeros((B, T + 1, d), f32)
    Gs = np.empty((B, T, 4 * d), f32)
    TS = np.empty((B, T, d), f32)
    for t in range(T):
        z = Ze[:, t] + Yall[:, t] @ Wy
        g = Gs[:, t]
        g[:, :3 * d] = sigm(z[:, :3 * d])
        g[:, 3 * d:] = np.tanh(z[:, 3 * d:])
        s = g[:, :d] * Sall[:, t] + g[:, d:2 * d] * g[:, 3 * d:]
        ts = np.tanh(s)
        Sall[:, t + 1] = s
        TS[:, t] = ts
        Yall[:, t + 1] = g[:, 2 * d:3 * d] * ts
    Y = Yall[:, 1:].reshape(B * T, d)
    logits = Y @ O
    logits = logits - logits.max(1, keepdims=True)
    ex = np.exp(logits)
    pr = ex / ex.sum(1, keepdims=True)
    mflat = msk.reshape(-1)
    nq = max(float(mflat.sum()), 1.0)
    tflat = tgt.reshape(-1)
    rows = np.arange(B * T)
    loss = -float((np.log(pr[rows, tflat] + 1e-12) * mflat).sum()) / nq
    correct = float(((np.argmax(pr, 1) == tflat) * mflat).sum())
    dlog = ((pr - eye[tflat]) * (mflat / nq)[:, None]).astype(f32)
    dO = Y.T @ dlog
    dY = (dlog @ O.T).reshape(B, T, d)
    dZ = np.empty((B, T, 4 * d), f32)
    dyn = np.zeros((B, d), f32)
    dsn = np.zeros((B, d), f32)
    WyT = Wy.T
    for t in range(T - 1, -1, -1):
        g = Gs[:, t]
        r = g[:, :d]
        i = g[:, d:2 * d]
        o = g[:, 2 * d:3 * d]
        u = g[:, 3 * d:]
        ts = TS[:, t]
        dy = dY[:, t] + dyn
        ds = dsn + dy * o * (1.0 - ts * ts)
        dz = dZ[:, t]
        dz[:, :d] = ds * Sall[:, t] * r * (1.0 - r)
        dz[:, d:2 * d] = ds * u * i * (1.0 - i)
        dz[:, 2 * d:3 * d] = dy * ts * o * (1.0 - o)
        dz[:, 3 * d:] = ds * i * (1.0 - u * u)
        dsn = ds * r
        dyn = dz @ WyT
    dZf = dZ.reshape(B * T, 4 * d)
    dWy = Yall[:, :T].reshape(B * T, d).T @ dZf
    G = OH.T @ dZf
    dWe = E.T @ G
    db = dZf.sum(0)
    dE = G @ We.T
    return loss, correct, [dE, dWe, dWy, db, dO]

def learn(layout, examples, seed):
    f32 = np.float32
    d = int(RECIPE['d'])
    V = int(layout['vocab'])
    groups = [(int(layout['key_base']), int(layout['n_keys'])), (int(layout['value_base']), int(layout['V']))]
    rng = np.random.default_rng(int(seed))
    k = 1.0 / np.sqrt(d)
    E = (rng.standard_normal((V, d)) * RECIPE['emb_scale']).astype(f32)
    We = rng.uniform(-k, k, (d, 4 * d)).astype(f32)
    Wy = rng.uniform(-k, k, (d, 4 * d)).astype(f32)
    b = np.zeros(4 * d, f32)
    b[:d] = RECIPE['retain_bias']
    O = rng.uniform(-k, k, (d, V)).astype(f32)
    P = [E, We, Wy, b, O]
    N = len(examples)
    if N > 0:
        T = max(len(ex['tokens']) for ex in examples)
        tok = np.zeros((N, T), np.int64)
        tgt = np.zeros((N, T), np.int64)
        msk = np.zeros((N, T), f32)
        for n in range(N):
            tk = examples[n]['tokens']
            tok[n, :len(tk)] = np.array(tk, np.int64)
            for q in examples[n]['queries']:
                p = int(q[0])
                tgt[n, p] = int(q[1])
                msk[n, p] = 1.0
        eye = np.eye(V, dtype=f32)
        Bs = int(RECIPE['batch_size'])
        epochs = int(RECIPE['epochs'])
        spe = (N + Bs - 1) // Bs
        total = epochs * spe
        ms = [np.zeros_like(x) for x in P]
        vs = [np.zeros_like(x) for x in P]
        base = RECIPE['learning_rate']
        wd = RECIPE['weight_decay']
        clip = RECIPE['clip']
        step = 0
        every = max(1, epochs // 8)
        for ep in range(epochs):
            perm = rng.permutation(N)
            tl = 0.0
            tc = 0.0
            tq = 0.0
            for st in range(0, N, Bs):
                idx = perm[st:st + Bs]
                tb, gb = relabel(rng, tok[idx], tgt[idx], groups, V)
                loss, corr, G = grad(P, tb, gb, msk[idx], eye, d)
                tl += loss
                tc += corr
                tq += float(msk[idx].sum())
                gn = float(np.sqrt(sum(float((g * g).sum()) for g in G)))
                sc = 1.0
                if gn > clip:
                    sc = clip / gn
                step += 1
                lr = base * (0.1 + 0.45 * (1.0 + np.cos(np.pi * step / total)))
                bc1 = 1.0 - 0.9 ** step
                bc2 = 1.0 - 0.999 ** step
                for j in range(5):
                    g = G[j] * f32(sc)
                    ms[j] *= f32(0.9)
                    ms[j] += f32(0.1) * g
                    vs[j] *= f32(0.999)
                    vs[j] += f32(0.001) * g * g
                    upd = (ms[j] / f32(bc1)) / (np.sqrt(vs[j] / f32(bc2)) + f32(1e-8))
                    if j != 3:
                        P[j] -= f32(lr * wd) * P[j]
                    P[j] -= f32(lr) * upd
            if ep % every == 0 or ep == epochs - 1:
                CURVE.append({"epoch": ep, "loss": float(tl / spe), "train_accuracy": float(tc / max(tq, 1.0))})
    E, We, Wy, b, O = P
    out = {'E': np.ascontiguousarray(E, f32), 'O': np.ascontiguousarray(O, f32)}
    names = ['r', 'i', 'o', 'u']
    for j in range(4):
        nm = names[j]
        out['W' + nm + '_y'] = np.ascontiguousarray(Wy[:, j * d:(j + 1) * d].T, f32)
        out['W' + nm + '_e'] = np.ascontiguousarray(We[:, j * d:(j + 1) * d].T, f32)
        out['b' + nm] = np.ascontiguousarray(b[j * d:(j + 1) * d], f32)
    return out

def init(layout, params):
    d = params['E'].shape[1]
    return {'p': params, 's': np.zeros(d, np.float32), 'y': np.zeros(d, np.float32)}

def update(state, token):
    p = state['p']
    e = p['E'][int(token)]
    y = state['y']
    r = sigm(p['Wr_y'] @ y + p['Wr_e'] @ e + p['br'])
    i = sigm(p['Wi_y'] @ y + p['Wi_e'] @ e + p['bi'])
    o = sigm(p['Wo_y'] @ y + p['Wo_e'] @ e + p['bo'])
    u = np.tanh(p['Wu_y'] @ y + p['Wu_e'] @ e + p['bu'])
    s = (r * state['s'] + i * u).astype(np.float32)
    yn = (o * np.tanh(s)).astype(np.float32)
    return {'p': p, 's': s, 'y': yn}

def predict(state):
    return int(np.argmax(state['y'] @ state['p']['O']))
