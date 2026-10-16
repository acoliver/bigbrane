"""Reference candidates for the cost verifier, with hand-derived cost formulas (counting rules in cost.py).

Each entry is (source, claim fields). The first four are correct; the planted ones are wrong on purpose and the
verifier must reject them. Self-contained (strings only) so it runs in the tools image as part of ``bbmath``.

Derivations, per token, with d the width, V the vocabulary and k the HMM states:

* ``elman_rnn``: update h = tanh(W @ h + E[tok]) is d^2 mac + d arith + d transcendental; predict
  argmax(h @ O) is d*V mac + (V - 1) compare. Total n*(d^2 + 2d + dV + V - 1); state h, d words.
* ``softmax_attention`` at step t (t keys stored): update makes q, k, v with 3d^2 mac; predict does scores K @ q
  (t*d mac), max (t - 1 compare), subtract (t arith), exp (t transcendental), sum (t - 1 arith), divide (t arith),
  weights @ V (t*d mac), output projection (d*V mac) and argmax (V - 1 compare). Per step
  3d^2 + dV + V - 1 + 2td + 5t - 2; summed over t = 1..n:
  n*(3d^2 + dV + V - 3) + (2d + 5) * n(n + 1)/2. State: K and V hold t*d words each, plus q: 2nd + d at the end.
* ``decayed_unigram``: update c = 0.5*c (V arith) then c[tok] = c[tok] + 1 (1 arith); predict argmax (V - 1
  compare). Total 2nV; state V words.
* ``hmm_forward`` with k states: update alpha = (alpha @ A) * B[:, tok] (k^2 mac + k arith), then normalize
  (k - 1 arith for the sum, k arith for the divide); predict argmax((alpha @ A) @ B) (k^2 + kV mac, V - 1
  compare). Total n*(2k^2 + kV + 3k + V - 2); state k words.
* ``planted_quadratic`` (planted exponent error): claims n*(2d - 1), linear, but update re-applies a decay t times
  at step t, each d arith, so the true count is d*n(n + 1)/2 + n(d - 1).
* ``planted_uncounted_loop``: the counted ops match its linear claim exactly, but update runs a Python loop of
  t iterations of uncounted scalar work. Only the executed-line slope (about 2) shows it.
"""
from __future__ import annotations

ELMAN_RNN = '''
def init(sizes, ops):
    d, V = sizes["d"], sizes["V"]
    params = {"W": ops.np.normal((d, d), 1, 0.3), "E": ops.np.normal((V, d), 2), "O": ops.np.normal((d, V), 3)}
    return params, {"h": ops.np.zeros(d)}

def update(params, state, token, ops):
    return {"h": ops.np.tanh(params["W"] @ state["h"] + params["E"][token])}

def predict(params, state, ops):
    return int((state["h"] @ params["O"]).argmax())
'''

SOFTMAX_ATTENTION = '''
def init(sizes, ops):
    d, V = sizes["d"], sizes["V"]
    params = {"Wq": ops.np.normal((d, d), 1, 0.3), "Wk": ops.np.normal((d, d), 2, 0.3),
              "Wv": ops.np.normal((d, d), 3, 0.3), "E": ops.np.normal((V, d), 4), "O": ops.np.normal((d, V), 5)}
    return params, {"K": ops.np.zeros((0, d)), "V": ops.np.zeros((0, d)), "q": ops.np.zeros(d)}

def update(params, state, token, ops):
    x = params["E"][token]
    k, v, q = x @ params["Wk"], x @ params["Wv"], x @ params["Wq"]
    return {"K": ops.np.concatenate([state["K"], k.reshape(1, -1)]),
            "V": ops.np.concatenate([state["V"], v.reshape(1, -1)]), "q": q}

def predict(params, state, ops):
    scores = state["K"] @ state["q"]
    e = ops.np.exp(scores - scores.max())
    w = e / e.sum()
    return int(((w @ state["V"]) @ params["O"]).argmax())
'''

DECAYED_UNIGRAM = '''
def init(sizes, ops):
    return {}, {"c": ops.np.zeros(sizes["V"])}

def update(params, state, token, ops):
    c = 0.5 * state["c"]
    c[token] = c[token] + 1
    return {"c": c}

def predict(params, state, ops):
    return int(state["c"].argmax())
'''

HMM_FORWARD = '''
def init(sizes, ops):
    k, V = sizes["k"], sizes["V"]
    A = ops.np.exp(ops.np.normal((k, k), 1))
    B = ops.np.exp(ops.np.normal((k, V), 2))
    A = A / A.sum(axis=1).reshape(-1, 1)
    B = B / B.sum(axis=1).reshape(-1, 1)
    return {"A": A, "B": B}, {"alpha": ops.np.full(k, 1.0 / k)}

def update(params, state, token, ops):
    alpha = (state["alpha"] @ params["A"]) * params["B"][:, token]
    return {"alpha": alpha / alpha.sum()}

def predict(params, state, ops):
    return int(((state["alpha"] @ params["A"]) @ params["B"]).argmax())
'''

PLANTED_QUADRATIC = '''
def init(sizes, ops):
    d, V = sizes["d"], sizes["V"]
    return {"E": ops.np.normal((V, d), 1)}, {"h": ops.np.zeros(d), "t": 0}

def update(params, state, token, ops):
    t = state["t"] + 1
    h = params["E"][token]
    for _ in range(t):
        h = h * 0.5
    return {"h": h, "t": t}

def predict(params, state, ops):
    return float(state["h"].sum())
'''

PLANTED_UNCOUNTED_LOOP = '''
def init(sizes, ops):
    d, V = sizes["d"], sizes["V"]
    return {"E": ops.np.normal((V, d), 1)}, {"h": ops.np.zeros(d), "t": 0}

def update(params, state, token, ops):
    t = state["t"] + 1
    acc = 0
    for j in range(t):
        acc = acc + j
    return {"h": state["h"] * 0.5 + params["E"][token], "t": t}

def predict(params, state, ops):
    return float(state["h"].sum())
'''

# name -> (source, claim kwargs for cost.CostClaim, expected verdict)
REFERENCES = {
    "elman_rnn": (ELMAN_RNN, {"ops": "n*(d**2 + 2*d + d*V + V - 1)", "state": "d",
                              "sizes": {"d": [8, 16], "V": [16, 32]}, "exponent": 1}, "discharged"),
    "softmax_attention": (SOFTMAX_ATTENTION, {"ops": "n*(3*d**2 + d*V + V - 3) + (2*d + 5)*n*(n + 1)/2",
                                              "state": "2*n*d + d", "sizes": {"d": [8, 16], "V": [16, 32]},
                                              "exponent": 2}, "discharged"),
    "decayed_unigram": (DECAYED_UNIGRAM, {"ops": "2*n*V", "state": "V", "sizes": {"V": [16, 32]}, "exponent": 1},
                        "discharged"),
    "hmm_forward": (HMM_FORWARD, {"ops": "n*(2*k**2 + k*V + 3*k + V - 2)", "state": "k",
                                  "sizes": {"k": [4, 8], "V": [16, 32]}, "exponent": 1}, "discharged"),
    "planted_quadratic": (PLANTED_QUADRATIC, {"ops": "n*(2*d - 1)", "state": "d + 1",
                                              "sizes": {"d": [8, 16], "V": [16, 32]}, "exponent": 1}, "violated"),
    "planted_uncounted_loop": (PLANTED_UNCOUNTED_LOOP, {"ops": "n*(3*d - 1)", "state": "d + 1",
                                                        "sizes": {"d": [8, 16], "V": [16, 32]}, "exponent": 1},
                               "violated"),
}


# ----- hand-set constructions for the P2 tasks (constructions.py contract) -----------------------------------------
# CAPACITY = None means unbounded; a number caps the memory, so the construction must fail above that size.

COPY_BUFFER = '''
CAPACITY = {capacity}

def init(layout):
    return [[], 0, layout["SEP"]]          # buffer, copy index (0 before the separator), separator id

def update(state, token):
    buf, index, sep = state
    if index == 0 and token != sep:
        if CAPACITY is None or len(buf) < CAPACITY:
            buf = buf + [token]
        return [buf, 0, sep]
    return [buf, index + 1, sep]

def predict(state):
    buf, index, sep = state
    return buf[index - 1] if index - 1 < len(buf) else 0
'''

RECALL_TABLE = '''
CAPACITY = {capacity}

def init(layout):
    return {"table": {}, "key": -1, "queries": 0, "qsep": layout["QSEP"], "vbase": layout["value_base"]}

def update(state, token):
    if token == state["qsep"]:
        state["queries"] = 1
    elif state["queries"] or token < state["vbase"]:
        state["key"] = token
    elif CAPACITY is None or len(state["table"]) < CAPACITY:
        state["table"][state["key"]] = token
    return state

def predict(state):
    return state["table"].get(state["key"], state["vbase"])
'''

DYCK_STACK = '''
CAPACITY = {capacity}

def init(layout):
    return {"stack": [], "dropped": 0}

def update(state, token):
    stack = state["stack"]
    if token in (0, 1):
        stack.append(token)
        if CAPACITY is not None and len(stack) > CAPACITY:
            stack.pop(0)
            state["dropped"] += 1
    elif stack:
        stack.pop()
    else:
        state["dropped"] -= 1
    return state

def predict(state):
    return 2 + state["stack"][-1] if state["stack"] else 2
'''

COMPOSITION_TABLES = '''
def init(layout):
    return {"g": {}, "f": {}, "phase": 0, "key": -1, "seps": [layout["GSEP"], layout["FSEP"], layout["QSEP"]]}

def update(state, token):
    if token in state["seps"]:
        state["phase"] = state["seps"].index(token) + 1
        state["key"] = -1
    elif state["phase"] == 3:
        state["key"] = token
    elif state["key"] < 0:
        state["key"] = token
    else:
        state["g" if state["phase"] == 1 else "f"][state["key"]] = token
        state["key"] = -1
    return state

def predict(state):
    return state["f"][state["g"][state["key"]]]
'''

BINDING_ENV = '''
def init(layout):
    return {"env": {}, "key": -1, "queries": 0, "qsep": layout["QSEP"], "vbase": layout["value_base"]}

def update(state, token):
    if token == state["qsep"]:
        state["queries"] = 1
    elif state["queries"] or state["key"] < 0:
        state["key"] = token
    else:
        state["env"][state["key"]] = token if token >= state["vbase"] else state["env"].get(token, state["vbase"])
        state["key"] = -1
    return state

def predict(state):
    return state["env"].get(state["key"], state["vbase"])
'''


def construction_source(template: str, capacity: int | None = None) -> str:
    return template.replace("{capacity}", repr(capacity))


# name -> (source, task, sizes, params, expected status)
CONSTRUCTIONS = {
    "copy_buffer": (construction_source(COPY_BUFFER), "copy", [4, 16, 64], {"D": 4}, "discharged"),
    "copy_buffer_cap8": (construction_source(COPY_BUFFER, 8), "copy", [4, 8, 16], {"D": 4}, "violated"),
    "recall_table": (construction_source(RECALL_TABLE), "recall", [4, 16, 64], {"V": 16}, "discharged"),
    "recall_table_cap8": (construction_source(RECALL_TABLE, 8), "recall", [4, 8, 16], {"V": 16}, "violated"),
    "dyck_stack": (construction_source(DYCK_STACK), "dyck", [2, 8, 32], {}, "discharged"),
    "dyck_stack_cap4": (construction_source(DYCK_STACK, 4), "dyck", [2, 4, 8], {}, "violated"),
    "composition_tables": (COMPOSITION_TABLES, "composition", [4, 16, 64], {}, "discharged"),
    "binding_env": (BINDING_ENV, "binding", [1, 4, 16], {}, "discharged"),
}


# ----- magic-X examples (magic_x.py contract) ----------------------------------------------------------------------

# A perfect retrieval index with no counted cost (plan section 13.1, T3): returns the value bound to the queried key.
RETRIEVAL_ORACLE = '''
def x(prefix, layout, ops):
    qsep = layout["QSEP"]
    if qsep not in prefix or prefix[-1] == qsep:
        return -1
    split = prefix.index(qsep)
    key = prefix[-1]
    for i in range(0, split, 2):
        if prefix[i] == key:
            return prefix[i + 1]
    return -1
'''

# An assumption that carries no information about the answer: the parity of the position.
POSITION_PARITY = '''
def x(prefix, layout, ops):
    return len(prefix) % 2
'''

# The card claims 0.95 per-query recall at K=8 over V=16 values from 64 training instances.
MAGIC_X_CLAIM = {"task": "recall", "size": 8, "value": 0.95, "tolerance": 0.02, "examples": 64, "params": {"V": 16}}
