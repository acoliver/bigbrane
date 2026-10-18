"""Host-owned published equations and explicitly bounded family repairs. No model code is executed here."""
from __future__ import annotations

import numpy as np

from hybrid.research.stages.sequitur_reference import sequitur_outputs

# References are host-owned equations, not a second implementation supplied by the critic.
# Parameters and card traces remain external evidence; they are retained in the artifact for reproduction.
REFERENCE_OPERATIONS = {
    "elman_tanh": ("rnn", ("W", "U", "b", "C", "h0"), ()),             # Elman 1990
    "elman_identity": ("rnn", ("W", "U", "b", "C", "h0"), ()),
    "linear_ssm_zero_lift": ("ssm", ("A", "B", "C", "h0"), ("lifted_state",)),
    "linear_ssm": ("ssm", ("A", "B", "C", "h0"), ()),                # linear state-space equations
    "input_ssm": ("ssm", ("A", "B", "C", "h0", "mask"), ("sparsity",)),
    "linear_attention": ("linear_attention", ("S0",), ()),             # Katharopoulos et al. 2020
    "delta_rule": ("linear_attention", ("S0",), ()),                   # Schlag et al. 2021
    "jeffrey_ipf": ("other_known", ("p0", "masks", "targets", "labels", "label_count"), ()),  # Jeffrey 1965; Csiszar 1975
    "softmax_memory": ("hopfield_or_kernel_memory", ("keys", "values", "beta"), ()),  # modern Hopfield read
    "dyck2": ("other_known", (), ()),                                  # fixed two-bracket PDA
    "rule110": ("other_known", ("h0",), ()),                           # Wolfram rule 110, periodic boundary
    "viterbi": ("other_known", ("initial", "transition", "emission"), ()),  # Viterbi 1967
}


def reference_policy_text() -> str:
    lines = ["INDEPENDENT REFERENCE POLICY: declarations and two model-written programs are not reference proof. "
             "For a supported comparator declare reference_spec with operation and parameters; execute the card "
             "and print reference_samples (inputs and card_outputs at every step) under its comparison id. "
             "The host executes its own published operation and ignores your max_abs_diff for verification. "
             "Use the card's free parameters through the mapping, not frozen witnesses. Do not normalize or "
             "rescale the reference samples. Unknown operations/repairs stay unverified, not refuted."]
    lines += [f"- {op}: family {family}; parameters {', '.join(names) or '(none)'}; "
              f"repair types {', '.join(repairs) or '(none)'}"
              for op, (family, names, repairs) in REFERENCE_OPERATIONS.items()]
    return "\n".join(lines)


def reference_outputs(operation: str, parameters: dict, inputs: list) -> np.ndarray:
    """Execute the specified published operation. No code, state-dependent feature map or repair prose is read."""
    def array(name):
        value = np.asarray(parameters[name], dtype=float)
        if not value.size or not np.isfinite(value).all():
            raise ValueError(f"{name} must be nonempty and finite")
        return value

    def token(x, count):
        if not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < count:
            raise ValueError("input must index the published operation's fixed table")
        return x

    outputs = []
    if operation.startswith("elman_") or operation in ("linear_ssm", "linear_ssm_zero_lift", "input_ssm"):
        h = array("h0")
        C = array("C")
        if operation.startswith("elman_"):
            W, U, b = array("W"), array("U"), array("b")
        else:
            A, B = array("A"), array("B")
        if operation == "linear_ssm_zero_lift":
            # Direct sum with one zero channel leaves the original readout unchanged.
            A = np.pad(A, ((0, 1), (0, 1)))
            B = np.pad(B, ((0, 1), (0, 0)))
            C = np.pad(C, ((0, 0), (0, 1)))
            h = np.pad(h, (0, 1))
        if operation == "input_ssm":
            mask = array("mask")
            if mask.shape != (len(A), len(h)) or not np.isin(mask, [0, 1]).all():
                raise ValueError("mask must be a fixed binary table indexed only by the input token")
        for x in inputs:
            if operation.startswith("elman_"):
                h = W @ h + U @ np.asarray(x, dtype=float) + b
                if operation == "elman_tanh":
                    h = np.tanh(h)
                y = C @ h
            elif operation in ("linear_ssm", "linear_ssm_zero_lift"):
                h = A @ h + B @ np.asarray(x, dtype=float)
                y = C @ h
            else:
                i = token(x, len(A))
                h = (A[i] @ h + B[i]) * mask[i]
                y = C[i] @ h
            outputs.append(y.copy())
    elif operation in ("linear_attention", "delta_rule"):
        S = array("S0").copy()
        for x in inputs:
            expected = {"key", "value", "query"} | ({"beta"} if operation == "delta_rule" else set())
            if set(x) != expected:
                raise ValueError("attention inputs must specify exactly key, value, query and delta beta")
            k, v, q = (np.asarray(x[name], dtype=float) for name in ("key", "value", "query"))
            write = v if operation == "linear_attention" else float(x["beta"]) * (v - S @ k)
            S = S + np.outer(write, k)
            outputs.append(S @ q)
    elif operation == "jeffrey_ipf":
        p, masks, targets, labels = array("p0"), array("masks"), array("targets"), array("labels")
        if (p.ndim != 1 or (p <= 0).any() or not np.isclose(p.sum(), 1) or
                masks.ndim != 2 or masks.shape[1] != len(p) or not np.isin(masks, [0, 1]).all() or
                targets.shape != (len(masks),) or ((targets <= 0) | (targets >= 1)).any() or
                labels.shape != p.shape or (labels < 0).any() or not (labels == labels.astype(int)).all()):
            raise ValueError("invalid probability partition or label parameters")
        label_count = parameters["label_count"]
        if not isinstance(label_count, int) or isinstance(label_count, bool) or label_count <= labels.max():
            raise ValueError("label_count must be an integer above every label index")
        for x in inputs:
            i = token(x, len(masks))
            mask, target = masks[i], targets[i]
            inside, outside = p @ mask, p @ (1 - mask)
            if inside <= 0 or outside <= 0:
                raise ValueError("Jeffrey partition must have positive mass on both cells")
            p = p * (mask * target / inside + (1 - mask) * (1 - target) / outside)
            outputs.append(np.bincount(labels.astype(int), weights=p * mask, minlength=label_count) / target)
    elif operation == "softmax_memory":
        keys, values = array("keys"), array("values")
        beta = float(parameters["beta"])
        for x in inputs:
            scores = beta * (keys @ np.asarray(x, dtype=float))
            weights = np.exp(scores - scores.max())
            outputs.append((weights / weights.sum()) @ values)
    elif operation == "dyck2":
        stack, valid = [], True
        for x in inputs:  # 0,1 open; 2,3 close their respective brackets
            i = token(x, 4)
            if i < 2:
                stack.append(i)
            elif not stack or stack.pop() != i - 2:
                valid = False
            outputs.append([int(valid and not stack)])
    elif operation == "rule110":
        h = array("h0")
        if h.ndim != 1 or not np.isin(h, [0, 1]).all():
            raise ValueError("Rule 110 state must be a binary ring")
        h = h.astype(int)
        for _ in inputs:  # one synchronous evolution per clock tick
            neighborhood = 4 * np.roll(h, 1) + 2 * h + np.roll(h, -1)
            h = (110 >> neighborhood) & 1
            outputs.append(h.copy())
    elif operation == "viterbi":
        scores, transition, emission = array("initial"), array("transition"), array("emission")
        if (scores < 0).any() or (transition < 0).any() or (emission < 0).any():
            raise ValueError("Viterbi requires nonnegative probabilities")
        for t, x in enumerate(inputs):
            i = token(x, emission.shape[1])
            scores = emission[:, i] * (scores if t == 0 else (scores[:, None] * transition).max(axis=0))
            outputs.append(scores.copy())
    else:
        return extended_outputs(operation, parameters, inputs)
    return np.asarray(outputs, dtype=float)


REFERENCE_OPERATIONS.update({
    "sequitur": ("other_known", (), ()),
    "queue_copy": ("other_known", (), ()),  # deterministic FIFO automaton for w#w
    "two_counter_abc": ("other_known", (), ()),  # Minsky two-counter a^n b^n c^n recognizer
    "inside_pcfg": ("other_known", ("binary", "lexical", "start"), ()),  # Baker sum-product Inside
    "earley": ("other_known", ("grammar", "start"), ()),  # Earley 1970
    "viterbi_max": ("other_known", ("initial", "transition", "emission"), ()),
    "selective_diagonal": ("ssm", ("Wf", "bf", "Wu"), ()),  # Gu and Dao selective diagonal recurrence
    "mamba_zoh": ("ssm", ("Wf", "bf", "Wu"), ()),  # N=1, A=-1 ZOH specialization
    "gated_additive_attention": ("linear_attention", ("S0",), ("gate",)),
    "diagonal_attention_count": ("linear_attention", ("theta", "bias", "beta", "tau"), ()),
    "diagonal_attention_epsilon_logits_probs": ("linear_attention", ("theta", "bias", "beta", "epsilon"), ("normalizer",)),
    "mesa_mixture_logits": ("linear_attention", ("phi", "output", "gate", "mixture", "ridge"), ("multi_read",)),
    "diagonal_attention_phase": ("linear_attention", ("theta", "bias", "beta", "tau"), ("normalizer",)),
    "mesa_single": ("linear_attention", ("phi", "output", "gate", "mixture", "ridge"), ()),
    "mesa_mixture": ("linear_attention", ("phi", "output", "gate", "mixture", "ridge"), ("multi_read",)),
    "bimap_hard_l1_read": ("other_known", ("rho", "prior", "eta", "threshold", "capacity"), ("normalizer",)),
    "bimap_hard": ("other_known", ("rho", "prior", "eta", "threshold", "capacity"), ()),
    "bimap_weighted": ("other_known", ("rho", "prior", "eta", "threshold", "capacity"), ("gate",)),
})


def _softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()


def extended_outputs(operation: str, p: dict, inputs: list) -> np.ndarray:
    """Specified operations only; malformed external data fails at the comparison boundary."""
    out = []
    if operation == "sequitur":
        return sequitur_outputs(inputs)
    if operation == "queue_copy":
        stored, phase, dead = [], False, False
        for x in inputs:
            if x not in ("a", "b", "#"):
                raise ValueError("queue alphabet is a,b,#")
            if not dead:
                if not phase:
                    if x == "#":
                        phase = True
                    else:
                        stored.append(x)
                elif x == "#" or not stored or stored.pop(0) != x:
                    dead = True
            out.append(float(phase and not dead and not stored))
    elif operation == "two_counter_abc":
        a = b = 0
        phase, dead = "a", False
        for x in inputs:
            if x not in ("a", "b", "c"):
                raise ValueError("counter alphabet is a,b,c")
            if not dead:
                if phase == "a" and x == "a":
                    a += 1
                elif phase in ("a", "b") and x == "b" and a > 0:
                    a -= 1
                    b += 1
                    phase = "b"
                elif phase in ("b", "c") and x == "c" and a == 0 and b > 0:
                    b -= 1
                    phase = "c"
                else:
                    dead = True
            out.append(float(not dead and phase == "c" and a == b == 0))
    elif operation == "inside_pcfg":
        rules, lex = np.asarray(p["binary"]), np.asarray(p["lexical"])
        n, d = len(inputs), len(lex)
        chart = np.zeros((n + 1, n + 1, d))
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length
                if length == 1:
                    chart[i, j] = lex[:, inputs[i]]
                else:
                    for A in range(d):
                        chart[i, j, A] = sum(rules[A, B, C] * chart[i, k, B] * chart[k, j, C]
                            for k in range(i + 1, j) for B in range(d) for C in range(d))
        out = [chart[0, j, p["start"]] for j in range(1, n + 1)]
    elif operation == "earley":
        grammar = {A: [tuple(rhs) for rhs in rhss] for A, rhss in p["grammar"].items()}
        start = ("<host-start>", (p["start"],), 0, 0)
        charts = [{start}]
        for j in range(len(inputs) + 1):
            current = charts[j]
            while True:
                additions = set()
                for A, rhs, dot, origin in current:
                    if dot < len(rhs) and rhs[dot] in grammar:
                        additions.update((rhs[dot], r, 0, j) for r in grammar[rhs[dot]])
                    elif dot == len(rhs):
                        additions.update((B, body, d + 1, k) for B, body, d, k in charts[origin]
                                         if d < len(body) and body[d] == A)
                if additions <= current:
                    break
                current |= additions
            out.append(float((start[0], start[1], 1, 0) in current))
            if j < len(inputs):
                charts.append({(A, rhs, dot + 1, origin) for A, rhs, dot, origin in current
                               if dot < len(rhs) and rhs[dot] == inputs[j]})
    elif operation == "viterbi_max":
        return reference_outputs("viterbi", p, inputs).max(axis=1)
    elif operation in ("selective_diagonal", "mamba_zoh"):
        W, b, U = (np.asarray(p[k]) for k in ("Wf", "bf", "Wu"))
        h = np.zeros(len(b))
        for x in inputs:
            x = np.asarray(x)
            z = W @ x + b
            if operation == "mamba_zoh":
                step = np.logaddexp(0, -z)
                decay = np.exp(-step)
                write = -np.expm1(-step)
            else:
                decay = 1 / (1 + np.exp(-z))
                write = 1 - decay
            h = decay * h + write * (U @ x)
            out.append(h.copy())
    elif operation == "gated_additive_attention":
        S = np.asarray(p["S0"], dtype=float).copy()
        for x in inputs:
            S += x["beta"] * np.outer(x["value"], x["key"])
            out.append(S @ x["query"])
    elif operation in ("diagonal_attention_count", "diagonal_attention_phase", "diagonal_attention_epsilon_logits_probs"):
        E = np.exp(1j * np.asarray(p["theta"]))
        state = np.zeros(E.shape[1], dtype=complex)
        for t, x in enumerate(inputs):
            if t:
                state += np.conj(E[inputs[t - 1]]) * E[x]
            if operation == "diagonal_attention_epsilon_logits_probs":
                read = state / (abs(state) + p["epsilon"])
            elif operation.endswith("phase"):
                mag = abs(state)
                read = np.divide(state, mag, out=np.zeros_like(state), where=mag > p["tau"])
            else:
                read = state / t if t else state
            scores = np.real(np.conj(E) @ (read * E[x])) / E.shape[1]
            logits = np.asarray(p["bias"]) + p["beta"] * scores
            probabilities = _softmax(logits)
            out.append(np.concatenate([logits, probabilities]) if operation.endswith("logits_probs") else probabilities)
    elif operation in ("mesa_single", "mesa_mixture", "mesa_mixture_logits"):
        phi, output, gate, mix = (np.asarray(p[k]) for k in ("phi", "output", "gate", "mixture"))
        G = p["ridge"] * np.eye(phi.shape[1])
        C = np.zeros_like(G)
        for previous, current in inputs:
            k, v = phi[previous], phi[current]
            beta = 1 / (1 + np.exp(-gate @ np.concatenate([k, v])))
            G += beta * np.outer(k, k)
            C += beta * np.outer(v, k)
            S = C @ np.linalg.inv(G)
            if operation == "mesa_single":
                read = S @ v
            else:
                alpha = _softmax(mix[previous])
                read = sum(alpha[m] * np.linalg.matrix_power(S, m) @ v for m in range(len(alpha)))
            logits = output @ read
            out.append(logits if operation.endswith("logits") else _softmax(logits))
    elif operation in ("bimap_hard", "bimap_weighted", "bimap_hard_l1_read"):
        rho, prior = np.asarray(p["rho"]), np.asarray(p["prior"])
        seen = np.zeros(len(prior))
        entries = {}  # key -> (value, weight, insertion time)
        previous = None
        for t, x in enumerate(inputs):
            seen[x] = 1
            if previous is not None and rho[previous, x] > p["threshold"]:
                owner = next((k for k, e in entries.items() if e[0] == x), None)
                if previous not in entries or owner is None:
                    entries = {k: e for k, e in entries.items() if k != previous and e[0] != x}
                    entries[previous] = (x, float(rho[previous, x]) if operation.endswith("weighted") else 1., t)
                    while len(entries) > p["capacity"]:
                        del entries[min(entries, key=lambda k: entries[k][1:])]
            free, answer = seen.copy(), np.zeros(len(prior))
            for value, weight, _ in entries.values():
                free[value] -= weight
            remaining = seen[x]
            if x in entries:
                value, weight, _ = entries[x]
                answer[value] = weight
                remaining -= weight
            fallback = prior * (free + p["eta"])
            read = answer + remaining * fallback / fallback.sum()
            if operation == "bimap_hard_l1_read":
                read = read / np.linalg.norm(read, ord=1)
            out.append(read)
            previous = x
    else:
        raise ValueError(f"unsupported host operation {operation}")
    return np.asarray(out, dtype=float)
