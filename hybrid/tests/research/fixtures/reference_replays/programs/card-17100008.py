import json
import numpy as np

rng = np.random.default_rng(0)
T = 12

# ---- Card: Rule 110 synchronous step on the growing zero-padded interval ----
def card_run(init, T):
    h = list(init)
    outs = []
    for t in range(1, T + 1):
        prev = h
        L = len(prev)
        def at(j, prev=prev, L=L):
            return prev[j] if 0 <= j < L else 0
        # new interval is the previous one expanded by one cell at each end
        h = [(110 // (2 ** (4 * at(j - 1) + 2 * at(j) + at(j + 1)))) % 2 for j in range(-1, L + 1)]
        outs.append((-t, list(h)))  # readout: interval with origin -t
    return outs

# ---- Comparator: Elman RNN h_t = f(W_rec h_{t-1} + U x_t), lifted (growing) state ----
def f_lookup(z):
    zi = np.rint(z).astype(np.int64)
    return ((110 // (2 ** zi)) % 2).astype(float)

Z = np.arange(8, dtype=float)
VALS = np.array([(110 >> k) & 1 for k in range(8)], dtype=float)

def f_poly(z):
    # degree-7 Lagrange interpolant through (k, bit k of 110), k = 0..7
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    for k in range(8):
        term = np.full_like(z, VALS[k])
        for m in range(8):
            if m != k:
                term = term * (z - Z[m]) / (Z[k] - Z[m])
        out = out + term
    return out

D_IN = 3

def rnn_run(init, T, f):
    h = np.array(init, dtype=float)
    outs = []
    for t in range(1, T + 1):
        n_prev = len(h)
        n = n_prev + 2
        Wrec = np.zeros((n, n_prev))
        for i in range(n):
            p = i - 1  # previous-state index of the same cell
            for off, w in ((-1, 4.0), (0, 2.0), (1, 1.0)):
                q = p + off
                if 0 <= q < n_prev:
                    Wrec[i, q] = w
        U = np.zeros((n, D_IN))  # constant tick carries no information
        x = np.ones(D_IN)
        z = Wrec @ h + U @ x
        h = f(z)
        outs.append((-t, h.copy()))  # identity readout, origin -t
    return outs

def diff(card_outs, cmp_outs):
    md = 0.0
    for (oc, yc), (om, ym) in zip(card_outs, cmp_outs):
        yc = np.array(yc, float)
        ym = np.array(ym, float)
        if yc.shape != ym.shape:
            return 1e9
        md = max(md, float(np.max(np.abs(yc - ym))) if yc.size else 0.0, abs(oc - om))
    return md

inputs = [[(k >> (7 - b)) & 1 for b in range(8)] for k in range(256)]
for _ in range(20):
    w = int(rng.integers(3, 16))
    inputs.append(list(rng.integers(0, 2, size=w)))

md_lookup = 0.0
md_poly = 0.0
X = []
Y = []
for init in inputs:
    c = card_run(init, T)
    md_lookup = max(md_lookup, diff(c, rnn_run(init, T, f_lookup)))
    md_poly = max(md_poly, diff(c, rnn_run(init, T, f_poly)))
    prev = list(init)
    for (_, y) in c:
        L = len(prev)
        at = lambda j: prev[j] if 0 <= j < L else 0
        for idx, j in enumerate(range(-1, L + 1)):
            X.append([at(j - 1), at(j), at(j + 1), 1.0])
            Y.append(y[idx])
        prev = y

X = np.array(X, float)
Y = np.array(Y, float)
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
md_affine = float(np.max(np.abs(X @ coef - Y)))

toy = card_run([int(ch) for ch in "00010000"], 1)[0]
toy_ok = ("".join(map(str, toy[1])) == "0001100000") and toy[0] == -1

report = {
    "comparisons": [
        {"declared_id": "conv_rnn_lookup_activation",
         "comparator": "Elman RNN with tridiagonal (4,2,1) recurrent matrix, Rule-110 lookup activation, U=0, identity readout, growing zero-padded state",
         "family": "rnn",
         "transformations": ["lifted_state"],
         "limit_only": False,
         "reference": "Elman (1990) h_t=f(W h_{t-1}+U x_t); CA-as-convolutional-recurrence (Gilpin 2019; Mordvintsev et al. 2020)",
         "inputs_tested": len(inputs),
         "max_abs_diff": md_lookup},
        {"declared_id": None,
         "comparator": "same RNN with degree-7 interpolating polynomial activation (sanity check)",
         "family": "rnn",
         "transformations": ["lifted_state"],
         "limit_only": False,
         "reference": "Elman (1990) with polynomial activation",
         "inputs_tested": len(inputs),
         "max_abs_diff": md_poly},
        {"declared_id": None,
         "comparator": "best least-squares affine same-coordinate update on raw cells (negative control)",
         "family": "ssm",
         "transformations": [],
         "limit_only": False,
         "reference": "affine SSM on raw cell coordinates (card's closest comparator)",
         "inputs_tested": len(inputs),
         "max_abs_diff": md_affine}
    ],
    "notes": "All 256 width-8 initial conditions plus 20 random widths 3..15, %d ticks each; outputs are binary (order one, no rescaling); origin mismatch also counted. Toy witness 00010000 -> 0001100000 origin -1 reproduced: %s." % (T, toy_ok)
}
print(json.dumps(report))
