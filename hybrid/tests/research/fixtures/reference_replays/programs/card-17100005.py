import json, random
import numpy as np

Z_START = ("Z", ("S",), 0, 0)
Z_DONE = ("Z", ("S",), 1, 0)

# Card: update and readout exactly as the card writes them (toy code).
# Saturate prediction and completion to a fixed point, then scan.
def card_charts(G, s):
    C = [set() for _ in range(len(s) + 1)]
    C[0].add(Z_START)
    for j in range(len(C)):
        changed = True
        while changed:
            before = len(C[j])
            for A, r, d, i in list(C[j]):
                if d < len(r) and r[d] in G:
                    for rhs in G[r[d]]:
                        C[j].add((r[d], rhs, 0, j))
                elif d == len(r):
                    for B, u, e, k in list(C[i]):
                        if e < len(u) and u[e] == A:
                            C[j].add((B, u, e + 1, k))
            changed = len(C[j]) != before
        if j < len(s):
            for A, r, d, i in C[j]:
                if d < len(r) and r[d] == s[j]:
                    C[j + 1].add((A, r, d + 1, i))
    return C

# Comparator earley_1970: Earley's worklist recognizer. Each state set is an
# ordered list processed once per state. Predictor, Scanner and Completer act
# while the list is walked; a state already in the set is not added again.
def earley_charts(G, s):
    n = len(s)
    S = [[] for _ in range(n + 1)]
    seen = [set() for _ in range(n + 1)]
    def add(j, st):
        if st not in seen[j]:
            seen[j].add(st)
            S[j].append(st)
    add(0, Z_START)
    for j in range(n + 1):
        k = 0
        while k < len(S[j]):
            A, r, d, i = S[j][k]
            k += 1
            if d < len(r):
                X = r[d]
                if X in G:
                    for g in G[X]:
                        add(j, (X, g, 0, j))
                elif j < n and s[j] == X:
                    add(j + 1, (A, r, d + 1, i))
            else:
                for (B, u, e, h) in list(S[i]):
                    if e < len(u) and u[e] == A:
                        add(j, (B, u, e + 1, h))
    return [set(x) for x in S]

def readout(C):
    return np.array([1.0 if Z_DONE in C[t] else 0.0 for t in range(len(C))])

rng = random.Random(12345)
TERMS = ["a", "b"]
NTS = ["S", "A", "B"]

def random_grammar():
    G = {"Z": [("S",)]}
    for X in NTS:
        prods = {(rng.choice(TERMS),)}
        for _ in range(rng.randint(1, 3)):
            L = rng.randint(1, 3)
            prods.add(tuple(rng.choice(NTS + TERMS) for _ in range(L)))
        G[X] = sorted(prods)
    return G

G0 = {"Z": [("S",)], "S": [("S", "S"), ("a",)]}
cases = [(G0, "aaa"), (G0, "aab")]
for _ in range(200):
    n = rng.randint(1, 12)
    alph = ["a"] * 4 + ["b"]
    cases.append((G0, "".join(rng.choice(alph) for _ in range(n))))
for _ in range(50):
    G = random_grammar()
    for _ in range(20):
        n = rng.randint(1, 12)
        cases.append((G, "".join(rng.choice(TERMS) for _ in range(n))))

max_y = 0.0
max_chart = 0.0
accepts = 0
steps = 0
toy_ok = True
for G, s in cases:
    Cc = card_charts(G, s)
    Ce = earley_charts(G, s)
    yc = readout(Cc)
    ye = readout(Ce)
    max_y = max(max_y, float(np.max(np.abs(yc - ye))))
    max_chart = max(max_chart, float(max(len(a ^ b) for a, b in zip(Cc, Ce))))
    accepts += int(yc.sum())
    steps += len(yc)

if bool(readout(card_charts(G0, "aaa"))[-1]) is not True:
    toy_ok = False
if bool(readout(card_charts(G0, "aab"))[-1]) is not False:
    toy_ok = False

out = {"comparisons": [
    {"declared_id": "earley_1970",
     "comparator": "Earley worklist recognizer (Predictor/Scanner/Completer over state sets, dummy start state), per-prefix acceptance bit",
     "family": "other_known",
     "transformations": [],
     "limit_only": False,
     "reference": "Earley (1970), An efficient context-free parsing algorithm, CACM 13(2), doi:10.1145/362007.362035",
     "inputs_tested": len(cases),
     "max_abs_diff": max_y},
    {"declared_id": None,
     "comparator": "sanity: max symmetric-difference size between card chart C[j] and Earley state set S_j",
     "family": "other_known",
     "transformations": [],
     "limit_only": False,
     "reference": "same as above; chart-level sanity check",
     "inputs_tested": len(cases),
     "max_abs_diff": max_chart}],
    "notes": "Readout is a 0/1 acceptance bit per prefix t=0..n, so no rescaling. Inputs are grammar S->SS|a (202 strings including aaa, aab) plus 50 random epsilon-free grammars over S,A,B with terminals a,b, 20 strings each, lengths 1-12. Accepting prefixes: %d of %d steps. Card toy expectations (aaa true, aab false) reproduced: %s." % (accepts, steps, toy_ok)}
print(json.dumps(out))
