"""Construction harness (DISCOVERY-PIPELINE-PLAN.md section 4.4, P2): runs a hand-set configuration of a
mechanism, with no learning, on generated toy inputs and reports exact accuracy against an oracle.

This is the ``executed`` tier: a pass means the construction was observed to work on the listed sizes, never that
it was proven.

Tasks (size parameter in brackets):

* ``copy`` [L]: a uniform string of length L over D symbols, a separator, then the string again; after the
  separator and after each copied symbol the mechanism predicts the next symbol. The instance stores the true string
  after the separator (it is training data for P4), but ``run_instance`` feeds the mechanism its own previous
  prediction there (self-fed continuation, ``SELF_FED``), so no unrevealed symbol reaches it. Metric: per string
  (every symbol right).
* ``recall`` [K]: multi-query associative recall in the Zoology MQAR format (arXiv 2312.04927): K distinct
  keys each followed by its value (uniform over V), a query separator, then every key once in random order; after
  each query key the mechanism predicts its value. Metric: per query.
* ``dyck`` [depth]: a Dyck-2 word that opens D brackets of random types, wanders between depth 1 and D, then
  closes everything; after every token that leaves the stack non-empty the mechanism predicts the legal closer
  (the one matching the top). Metric: per sequence.
* ``composition`` [T]: a table g from T symbols to T symbols, a table f from those to T symbols, both given in
  context as pairs, then every x once as a query; the answer is f(g(x)). Metric: per query.
* ``binding`` [hops]: variable binding, the second composition variant. Two interleaved chains of assignments
  v1 = c, v2 = v1, ..., each of ``hops`` steps; querying the last variable of each chain asks for its constant.
  Metric: per query.

Each task has a generator, deterministic in (task, size, seed, params), and a solver that recomputes the answers
from the tokens alone. ``oracle_self_score`` checks that the two agree on every instance (it must be 1.0).

Mechanism contract (Python source, NumPy available as ``np``): ``init(layout) -> state``,
``update(state, token) -> state``, ``predict(state) -> int``. ``layout`` holds the task, the size, the params and
the token ids. predict is called only at query positions, after the token at that position was read.

Carried state and module data (FIX-8). Everything a mechanism carries from one token to the next is the state
``update`` returns; module-level data is fixed once ``init`` has returned. ``run_construction`` makes every array
reachable from module data (and from the source's function defaults, closures and classes) read-only after each
``init`` and compares the rest of that data after every token: a write is hidden storage, and the size fails with
a resource violation. With a declared carried-state budget (``state_bits`` at a size, from the architecture
contract's ``state_bits``), ``CarriedStateAudit`` checks the state at initialization and after every token: integers
only, one index z with 0 <= z < 2**state_bits (or several integers packing into state_bits bits), and at most
2**state_bits distinct whole configurations at the size. A size with a resource violation never passes, whatever
its accuracy.

Self-contained (stdlib, NumPy and the sibling check modules) so it runs in the tools image as part of ``bbmath``.
"""
from __future__ import annotations

import builtins
import math
import random
from dataclasses import asdict, dataclass, field
from types import FunctionType, ModuleType

import numpy as np

from . import tiers
from .architecture import latent_states
from .cost import StateDomainError, configuration, packed_bits, words

DEFAULT_THRESHOLD = 0.99
DEFAULT_COUNT = 100
DEFAULTS = {
    "copy": {"D": 4},
    "recall": {"V": 16},
    "dyck": {},
    "composition": {},
    "binding": {"values": 8},
}
# Tasks whose output side is fed back: after the first prediction, each token the harness would read at a query
# position's successor is the mechanism's own previous prediction, not the stored true symbol. Card k2's frozen
# copy definition promised this in the 2026-10-13 smoke while the harness fed the true string (teacher forcing).
SELF_FED = frozenset({"copy"})
METRIC = {"copy": "instance", "recall": "query", "dyck": "instance", "composition": "query", "binding": "query"}
SIZE_NAME = {"copy": "L", "recall": "K", "dyck": "depth", "composition": "T", "binding": "hops"}


@dataclass(frozen=True)
class Instance:
    tokens: tuple
    queries: tuple   # ((position, answer), ...): answer expected after reading tokens[: position + 1]


def params_for(task: str, params: dict | None) -> dict:
    if task not in DEFAULTS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(DEFAULTS)}")
    merged = dict(DEFAULTS[task], **(params or {}))
    unknown = set(merged) - set(DEFAULTS[task]) - ({"n_keys"} if task == "recall" else set())
    if unknown:
        raise ValueError(f"unknown params {sorted(unknown)} for task {task}")
    return merged


def layout(task: str, size: int, params: dict | None = None) -> dict:
    """Token ids of a task at a size. Mechanisms read their constants from here."""
    p = params_for(task, params)
    if size < 1:
        raise ValueError("size must be at least 1")
    if task == "copy":
        return {"task": task, "size": size, "params": p, "D": p["D"], "SEP": p["D"], "vocab": p["D"] + 1}
    if task == "recall":
        n_keys = p.get("n_keys", max(2 * size, 16))
        if n_keys < size:
            raise ValueError("n_keys must be at least K")
        return {"task": task, "size": size, "params": p, "n_keys": n_keys, "key_base": 0, "value_base": n_keys,
                "V": p["V"], "QSEP": n_keys + p["V"], "vocab": n_keys + p["V"] + 1}
    if task == "dyck":
        return {"task": task, "size": size, "params": p, "OPEN": (0, 1), "CLOSE": (2, 3), "vocab": 4}
    if task == "composition":
        T = size
        return {"task": task, "size": size, "params": p, "T": T, "x_base": 0, "y_base": T, "z_base": 2 * T,
                "GSEP": 3 * T, "FSEP": 3 * T + 1, "QSEP": 3 * T + 2, "vocab": 3 * T + 3}
    n_vars = 2 * size + 2
    return {"task": task, "size": size, "params": p, "n_vars": n_vars, "var_base": 0, "value_base": n_vars,
            "values": p["values"], "QSEP": n_vars + p["values"], "vocab": n_vars + p["values"] + 1}


def _rng(task: str, size: int, seed: int, index: int) -> random.Random:
    return random.Random(f"{task}:{size}:{seed}:{index}")


def generate(task: str, size: int, count: int, seed: int, params: dict | None = None) -> list[Instance]:
    lay = layout(task, size, params)
    return [_GENERATORS[task](lay, _rng(task, size, seed, i)) for i in range(count)]


def _gen_copy(lay, rng) -> Instance:
    L, sep = lay["size"], lay["SEP"]
    x = [rng.randrange(lay["D"]) for _ in range(L)]
    tokens = x + [sep] + x[:-1]
    return Instance(tuple(tokens), tuple((L + i, x[i]) for i in range(L)))


def _gen_recall(lay, rng) -> Instance:
    K = lay["size"]
    keys = rng.sample(range(lay["n_keys"]), K)
    values = {k: lay["value_base"] + rng.randrange(lay["V"]) for k in keys}
    tokens = [t for k in keys for t in (lay["key_base"] + k, values[k])] + [lay["QSEP"]]
    queries = []
    for k in rng.sample(keys, K):
        tokens.append(lay["key_base"] + k)
        queries.append((len(tokens) - 1, values[k]))
    return Instance(tuple(tokens), tuple(queries))


def _gen_dyck(lay, rng) -> Instance:
    D = lay["size"]
    stack, tokens = [], []

    def push():
        kind = rng.randrange(2)
        stack.append(kind)
        tokens.append(lay["OPEN"][kind])

    def pop():
        tokens.append(lay["CLOSE"][stack.pop()])

    for _ in range(D):
        push()
    for _ in range(2 * D):
        if len(stack) < D and (len(stack) <= 1 or rng.random() < 0.5):
            push()
        else:
            pop()
    while stack:
        pop()
    return Instance(tuple(tokens), tuple(_dyck_answers(lay, tokens)))


def _dyck_answers(lay, tokens) -> list:
    stack, answers = [], []
    for position, token in enumerate(tokens):
        if token in lay["OPEN"]:
            stack.append(lay["OPEN"].index(token))
        elif token in lay["CLOSE"]:
            if not stack or lay["CLOSE"].index(token) != stack[-1]:
                raise ValueError(f"not a Dyck-2 word at position {position}")
            stack.pop()
        else:
            raise ValueError(f"token {token} is not a bracket")
        if stack:
            answers.append((position, lay["CLOSE"][stack[-1]]))
    return answers


def _gen_composition(lay, rng) -> Instance:
    T = lay["T"]
    g = {x: rng.randrange(T) for x in range(T)}
    f = {y: rng.randrange(T) for y in range(T)}
    tokens = [lay["GSEP"]]
    for x in rng.sample(range(T), T):
        tokens += [lay["x_base"] + x, lay["y_base"] + g[x]]
    tokens.append(lay["FSEP"])
    for y in rng.sample(range(T), T):
        tokens += [lay["y_base"] + y, lay["z_base"] + f[y]]
    tokens.append(lay["QSEP"])
    queries = []
    for x in rng.sample(range(T), T):
        tokens.append(lay["x_base"] + x)
        queries.append((len(tokens) - 1, lay["z_base"] + f[g[x]]))
    return Instance(tuple(tokens), tuple(queries))


def _gen_binding(lay, rng) -> Instance:
    hops = lay["size"]
    variables = rng.sample(range(lay["n_vars"]), 2 * hops + 2)
    chains = [variables[: hops + 1], variables[hops + 1:]]
    constants = [lay["value_base"] + rng.randrange(lay["values"]) for _ in chains]
    statements = [[(c[0], constants[i])] + [(c[j], c[j - 1]) for j in range(1, hops + 1)]
                  for i, c in enumerate(chains)]
    order = [0] * len(statements[0]) + [1] * len(statements[1])
    rng.shuffle(order)
    cursor, tokens = [0, 0], []
    for which in order:
        lhs, rhs = statements[which][cursor[which]]
        cursor[which] += 1
        tokens += [lay["var_base"] + lhs, rhs if rhs >= lay["value_base"] else lay["var_base"] + rhs]
    tokens.append(lay["QSEP"])
    queries = []
    for which in rng.sample([0, 1], 2):
        tokens.append(lay["var_base"] + chains[which][-1])
        queries.append((len(tokens) - 1, constants[which]))
    return Instance(tuple(tokens), tuple(queries))


_GENERATORS = {"copy": _gen_copy, "recall": _gen_recall, "dyck": _gen_dyck, "composition": _gen_composition,
               "binding": _gen_binding}


# ----- solvers: answers recomputed from the tokens alone -----------------------------------------------------------

def solve(task: str, size: int, tokens, params: dict | None = None) -> list:
    lay = layout(task, size, params)
    tokens = list(tokens)
    if task == "copy":
        start = tokens.index(lay["SEP"])
        source = tokens[:start]
        return [(start + i, source[i]) for i in range(len(source))]
    if task == "dyck":
        return _dyck_answers(lay, tokens)
    if task == "recall":
        split = tokens.index(lay["QSEP"])
        table = dict(zip(tokens[0:split:2], tokens[1:split:2]))
        return [(split + 1 + i, table[k]) for i, k in enumerate(tokens[split + 1:])]
    if task == "composition":
        fsep, qsep = tokens.index(lay["FSEP"]), tokens.index(lay["QSEP"])
        g = dict(zip(tokens[1:fsep:2], tokens[2:fsep:2]))
        f = dict(zip(tokens[fsep + 1:qsep:2], tokens[fsep + 2:qsep:2]))
        return [(qsep + 1 + i, f[g[x]]) for i, x in enumerate(tokens[qsep + 1:])]
    split = tokens.index(lay["QSEP"])
    env = {}
    for lhs, rhs in zip(tokens[0:split:2], tokens[1:split:2]):
        env[lhs] = rhs if rhs >= lay["value_base"] else env[rhs]
    return [(split + 1 + i, env[v]) for i, v in enumerate(tokens[split + 1:])]


# ----- task contracts: what the generators emit, measured from them --------------------------------------------------
#
# The statement fixer and the prover read these instead of guessing the layout. In the 2026-10-13 smoke (card k1) the
# fixer froze capacities of 4T+3 tokens for composition and 3hops+3 for binding while the generators emit 5T+3 and
# 4hops+7, and the prover assumed an ordered g table and one chain, read a nonexistent layout['hops'] and confused the
# recall pair count K with the key universe n_keys (results/phase4-20261013T073748Z/k1-smoke-diag/REPORT.md). Every
# number and every property below is computed by running the generators, so the text follows the code.

CONTRACT_SAMPLE = 200
LENGTH_PROBE_SIZES = tuple(range(1, 9))
EXAMPLE_SIZE = {"copy": 3, "recall": 2, "dyck": 2, "composition": 2, "binding": 1}


def stream_lengths(task: str, size: int, count: int, seed: int, params: dict | None = None) -> tuple[int, int]:
    """Shortest and longest token stream among the ``count`` instances the generator makes at (size, seed)."""
    lengths = [len(inst.tokens) for inst in generate(task, size, count, seed, params)]
    return min(lengths), max(lengths)


def length_formula(task: str, params: dict | None = None) -> str | None:
    """The stream length as ``a*S+b`` in the size S, when every probe size gives one length for all its instances and
    the lengths are exactly linear; None otherwise (dyck lengths vary per instance)."""
    points = []
    for size in LENGTH_PROBE_SIZES:
        try:
            low, high = stream_lengths(task, size, 50, 0, params)
        except ValueError:
            continue
        if low != high:
            return None
        points.append((size, low))
    if len(points) < 3:
        return None
    (s0, l0), (s1, l1) = points[0], points[1]
    if (l1 - l0) % (s1 - s0):
        return None
    slope = (l1 - l0) // (s1 - s0)
    offset = l0 - slope * s0
    if any(length != slope * size + offset for size, length in points):
        return None
    head = f"{slope}*{SIZE_NAME[task]}"
    return head if offset == 0 else f"{head}{'+' if offset > 0 else '-'}{abs(offset)}"


def _sorted_share(pairs_by_instance) -> int:
    return sum(keys == sorted(keys) for keys in pairs_by_instance)


def task_facts(task: str, size: int, params: dict | None = None, *, count: int = CONTRACT_SAMPLE,
               seed: int = 0) -> dict:
    """Properties of the generated instances at one size, measured on ``count`` of them."""
    lay = layout(task, size, params)
    instances = generate(task, size, count, seed, params)
    lengths = [len(inst.tokens) for inst in instances]
    queries = [len(inst.queries) for inst in instances]
    facts = {"count": count, "length": (min(lengths), max(lengths)), "queries": (min(queries), max(queries)),
             "layout": {k: v for k, v in lay.items() if k not in ("task", "params")}}
    sep = lay.get("QSEP", lay.get("SEP"))
    if sep is not None:
        facts["queries_after_separator"] = all(
            all(position >= list(inst.tokens).index(sep) for position, _ in inst.queries) for inst in instances)
    if task == "copy":
        L = size
        facts["teacher_forced"] = all(list(inst.tokens[L + 1:]) == list(inst.tokens[:L - 1]) for inst in instances)
        facts["first_query_at_separator"] = all(inst.queries[0][0] == L for inst in instances)
    elif task == "recall":
        pair_keys = [list(inst.tokens[0:2 * size:2]) for inst in instances]
        query_keys = [list(inst.tokens[2 * size + 1:]) for inst in instances]
        facts["pairs_sorted"] = _sorted_share(pair_keys)
        facts["queries_in_pair_order"] = sum(q == p for q, p in zip(query_keys, pair_keys))
        facts["keys_distinct"] = all(len(set(keys)) == size for keys in pair_keys)
        facts["largest_key_seen"] = max(max(keys) for keys in pair_keys)
    elif task == "composition":
        T = size
        g_keys = [list(inst.tokens[1:2 * T + 1:2]) for inst in instances]
        f_keys = [list(inst.tokens[2 * T + 2:4 * T + 2:2]) for inst in instances]
        facts["g_sorted"], facts["f_sorted"] = _sorted_share(g_keys), _sorted_share(f_keys)
        g_values = [list(inst.tokens[2:2 * T + 1:2]) for inst in instances]
        facts["g_permutations"] = sum(len(set(values)) == T for values in g_values)
        facts["every_x_queried_once"] = all(sorted(inst.tokens[4 * T + 3:]) == list(range(lay["x_base"],
                                                                                          lay["x_base"] + T))
                                            for inst in instances)
    elif task == "binding":
        chains, steps, interleaved, used = set(), set(), 0, set()
        for inst in instances:
            split = list(inst.tokens).index(lay["QSEP"])
            pairs = list(zip(inst.tokens[0:split:2], inst.tokens[1:split:2]))
            constants = [lhs for lhs, rhs in pairs if rhs >= lay["value_base"]]
            chains.add(len(constants))
            used.add(len({lhs for lhs, _ in pairs}))
            rhs_of = dict(pairs)
            owner = []
            for lhs, _ in pairs:
                root, hops_seen = lhs, 0
                while rhs_of[root] < lay["value_base"]:
                    root, hops_seen = rhs_of[root], hops_seen + 1
                owner.append(root)
            interleaved += sum(owner[i] != owner[i + 1] for i in range(len(owner) - 1)) > 1
            for variable in inst.tokens[split + 1:]:
                walk, hops_seen = variable, 0
                while rhs_of[walk] < lay["value_base"]:
                    walk, hops_seen = rhs_of[walk], hops_seen + 1
                steps.add(hops_seen)
        facts.update(chains=sorted(chains), query_steps=sorted(steps), interleaved=interleaved,
                     variables_used=sorted(used))
    return facts


def _share(k: int, n: int, what: str, *, yes: str, no: str) -> str:
    if k == 0:
        return no
    if k == n:
        return yes
    return f"{no} ({what} in only {k} of {n} sampled instances)"


TASK_ORDER = {
    "copy": "L data symbols (ids 0..D-1), then SEP, then the first L-1 data symbols again",
    "recall": "K pairs (key token key_base+k, value token value_base+v), then QSEP, then the K keys again as queries",
    "dyck": "a two-type balanced bracket word: OPEN = (0, 1), CLOSE = (2, 3); it opens `depth` brackets, wanders "
            "between depth 1 and `depth`, then closes everything",
    "composition": "GSEP, then T pairs (x_base+x, y_base+g(x)), then FSEP, then T pairs (y_base+y, z_base+f(y)), then "
                   "QSEP, then T query tokens x_base+x",
    "binding": "assignments as token pairs (lhs variable var_base+v, rhs) where rhs is a value token value_base+c or a "
               "variable token, then QSEP, then query variables",
}


def task_contract(task: str, sizes, params: dict | None = None, *, count: int | None = None,
                  seed: int = 0) -> str:
    """The executed task's contract at ``sizes``, measured from the generator: layout fields, token order, whether
    the tables are shuffled, the chain structure, query timing, size and vocabulary semantics, and the exact token
    stream length at each size (over the ``count`` instances at ``seed`` a check runs, when given)."""
    name = SIZE_NAME[task]
    sizes = list(sizes)
    sample = count or CONTRACT_SAMPLE
    facts = {size: task_facts(task, size, params, count=sample, seed=seed) for size in sizes}
    first = facts[sizes[0]]
    n = first["count"]
    lines = [f"TASK CONTRACT {task} (measured from hybrid/math/constructions.py; the generator, not this text, "
             f"decides). Size {name} = layout['size']; params {params_for(task, params)}."]
    lines.append("Layout (init receives this dict): " + "; ".join(
        f"{name}={size}: {facts[size]['layout']}" for size in sizes) + ".")
    lines.append(f"Token order: {TASK_ORDER[task]}.")
    if task == "copy":
        lines.append(("The stored tokens after SEP are the true source symbols (training examples carry them)"
                      if first["teacher_forced"] else "The stored tokens after SEP are not the source string")
                     + ", but when a construction or a trained learner is run the harness feeds, after SEP, the "
                     "mechanism's own previous prediction instead (self-fed continuation, no teacher forcing): the "
                     "token read after the i-th prediction is that prediction, so an error is fed back. predict is "
                     "called " + ("first right after SEP is read, then after each fed symbol"
                                  if first["first_query_at_separator"] else "at the listed positions")
                     + "; the i-th answer is data symbol i of the source.")
    elif task == "recall":
        top = facts[sizes[-1]]
        lines.append(f"{name} is the number of pairs; n_keys is the key universe the K distinct keys are drawn from "
                     f"(" + ", ".join(f"{name}={s}: n_keys={facts[s]['layout']['n_keys']}" for s in sizes)
                     + f"); key ids seen go up to {top['largest_key_seen']} at {name}={sizes[-1]}. Keys are "
                     + ("distinct" if first["keys_distinct"] else "not always distinct") + "; pairs are "
                     + _share(first["pairs_sorted"], n, "sorted by key", yes="sorted by key", no="in random key order")
                     + "; queries ask every key once, "
                     + _share(first["queries_in_pair_order"], n, "in pair order", yes="in pair order",
                              no="in an order independent of the pairs")
                     + ". predict is called right after each query key is read; the answer is that key's value token.")
    elif task == "dyck":
        lines.append("predict is called after every token that leaves the stack non-empty; the answer is the closer "
                     "matching the top of the stack. Stream lengths and query counts vary per instance.")
    elif task == "composition":
        lines.append("Pairs in the g table are "
                     + _share(first["g_sorted"], n, "sorted by x", yes="sorted by x", no="shuffled")
                     + "; pairs in the f table are "
                     + _share(first["f_sorted"], n, "sorted by y", yes="sorted by y", no="shuffled")
                     + ". g and f are arbitrary functions, not permutations (g was a permutation in "
                     f"{first['g_permutations']} of {n} sampled instances at {name}={sizes[0]}). The queries "
                     + ("ask every x once, in random order" if first["every_x_queried_once"] else "are a sample of x")
                     + "; predict is called right after each query token is read and the answer is z_base+f(g(x)). "
                     "A lookup must find a pair by its key token, not by its position.")
    else:
        lines.append(f"{name} is the number of variable-to-variable steps per chain. Each instance has "
                     + " or ".join(map(str, first["chains"])) + " chains; each chain is one constant assignment "
                     f"v = c followed by {name} assignments v' = v, and the chains' assignments are interleaved "
                     f"(both chains mixed in {facts[sizes[-1]]['interleaved']} of {n} sampled instances at "
                     f"{name}={sizes[-1]}) while each chain keeps its order. The queries are the last variable of each "
                     "chain, in random order, after QSEP; the answer is that chain's constant, reached in "
                     + ", ".join(f"{name}={s}: {facts[s]['query_steps']}" for s in sizes)
                     + " variable steps. n_vars variables are all used ("
                     + ", ".join(f"{name}={s}: {facts[s]['variables_used']}" for s in sizes)
                     + "). The layout has no 'hops' field: read layout['size'].")
    formula = length_formula(task, params)
    lines.append("Token stream length (every token the harness feeds, separators and queries included): "
                 + "; ".join(f"{name}={s}: " + (f"{facts[s]['length'][0]}" if facts[s]["length"][0] ==
                                                 facts[s]["length"][1]
                                                 else f"{facts[s]['length'][0]} to {facts[s]['length'][1]}")
                             + f" tokens, {facts[s]['queries'][0]}"
                             + ("" if facts[s]["queries"][0] == facts[s]["queries"][1]
                                else f" to {facts[s]['queries'][1]}") + f" predict calls, vocab "
                             + str(facts[s]["layout"]["vocab"]) for s in sizes)
                 + (f" (exactly {formula} at every size)" if formula else "")
                 + f"; measured over {n} instances at seed {seed}"
                 + (" (the instances the check runs)" if count else "") + ".")
    example_size = EXAMPLE_SIZE[task]
    try:
        example = generate(task, example_size, 1, seed, params)[0]
        lines.append(f"Example at {name}={example_size}: tokens {list(example.tokens)}; (position, answer) queries "
                     f"{[list(q) for q in example.queries]}.")
    except ValueError:
        pass
    return "\n".join(lines)


def oracle_self_score(task: str, size: int, count: int, seed: int, params: dict | None = None) -> float:
    """Share of generated instances on which the solver reproduces the generator's answers (must be 1.0)."""
    instances = generate(task, size, count, seed, params)
    agree = sum(list(inst.queries) == solve(task, size, inst.tokens, params) for inst in instances)
    return agree / len(instances)


# ----- running a construction --------------------------------------------------------------------------------------

def load_mechanism(source: str):
    namespace = {"__builtins__": builtins, "np": np}
    exec(compile(source, "<construction>", "exec"), namespace)
    missing = [name for name in ("init", "update", "predict") if not callable(namespace.get(name))]
    if missing:
        raise ValueError(f"construction does not define {missing}")
    return namespace


@dataclass
class SizeResult:
    size: int
    instances: int
    queries: int
    query_accuracy: float
    instance_accuracy: float
    metric: str
    accuracy: float
    passed: bool
    state_peak_words: int
    first_error: dict | None
    # FIX-8: the declared carried-state budget at this size, the distinct whole configurations the state reached (both
    # None without a budget), and why the size broke its resource contract. A size with a resource violation has
    # passed False whatever its accuracy, so it never counts as a P2 success.
    state_bits: float | None = None
    carried_states: int | None = None
    resource_violations: list = field(default_factory=list)


# Most input tokens a counterexample carries (composition at T=32 has 163).
COUNTEREXAMPLE_TOKENS = 200
# How deep the module-data guard walks nested containers.
GUARD_DEPTH = 8
SOURCE_FILE = "<construction>"


class MechanismCrashed(Exception):
    """The mechanism raised on an instance; carries the size and instance index for the prover's feedback."""

    def __init__(self, size: int, instance: int, exc: Exception):
        super().__init__(f"at {exc!r}")
        self.size, self.instance, self.exc = size, instance, exc


class HiddenStorage(Exception):
    """The mechanism changed module-level data after init: storage outside the carried state."""


class NonFiniteState(Exception):
    """The carried state holds a NaN or an infinity (FIX-13). A witness whose normalized weights overflow carries NaN
    from then on, and argmax over NaN logits still returns a token (token 0 in NumPy), so accuracy alone can pass a
    witness that no longer computes anything. Every number the state carries must be finite at the declared
    precision."""


def nonfinite(value, path: str = "state", depth: int = 0) -> str | None:
    """Where the first NaN or infinity in a carried state is, or None when every number is finite."""
    if depth > GUARD_DEPTH:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc":
            bad = ~np.isfinite(value)
            if bad.any():
                first = value[bad].flat[0]
                return f"{path} holds {first} ({int(bad.sum())} of {value.size} entries not finite)"
            return None
        if value.dtype.kind == "O":
            for index, item in enumerate(value.flat):
                found = nonfinite(item, f"{path}.flat[{index}]", depth + 1)
                if found:
                    return found
        return None
    if isinstance(value, (complex, np.complexfloating)):
        return None if math.isfinite(value.real) and math.isfinite(value.imag) else f"{path} is {value}"
    if isinstance(value, (float, np.floating)):
        return None if math.isfinite(value) else f"{path} is {value}"
    if isinstance(value, dict):
        items = ((f"{path}[{key!r}]", item) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        items = ((f"{path}[{index}]", item) for index, item in enumerate(value))
    else:
        return None
    for where, item in items:
        found = nonfinite(item, where, depth + 1)
        if found:
            return found
    return None


class ModuleGuard:
    """Module-level data is fixed once init has returned (FIX-8). ``after_init`` makes every array reachable from the
    mechanism's globals, its functions' defaults and closures and its classes read-only and records the rest of that
    data (scalars, containers, which array sits where); ``check`` compares it after every token; ``release`` makes
    the arrays writable again before the next init, which may rebuild them. Arrays are compared through the
    read-only flag rather than by content, so the per-token cost does not grow with table sizes."""

    def __init__(self, mechanism: dict):
        self.mechanism = mechanism
        self.frozen: list[np.ndarray] = []
        self.baseline = None

    def _roots(self) -> dict:
        roots = {}
        for name, value in self.mechanism.items():
            if name.startswith("__") or name == "np" or isinstance(value, ModuleType):
                continue
            if isinstance(value, FunctionType):
                if value.__code__.co_filename == SOURCE_FILE:
                    roots[name] = ("function", value.__defaults__, value.__kwdefaults__,
                                   tuple(c.cell_contents for c in value.__closure__ or () if _filled(c)))
            elif isinstance(value, type):
                if any(isinstance(v, FunctionType) and v.__code__.co_filename == SOURCE_FILE
                       for v in vars(value).values()):
                    roots[name] = ("class", {k: v for k, v in vars(value).items() if not k.startswith("__")
                                             and not isinstance(v, FunctionType)})
            elif not callable(value):
                roots[name] = value
        return roots

    def _shape(self, value, arrays: list | None, depth: int = 0):
        if type(value) in _SCALARS:
            # The value itself: a float that did not change is the same object, so NaN compares equal to itself.
            return type(value), value
        if depth > GUARD_DEPTH:
            return "deep"
        if isinstance(value, np.ndarray):
            if arrays is not None:
                arrays.append(value)
            return "array", id(value), value.shape, value.dtype.str, bool(value.flags.writeable)
        if isinstance(value, (bool, int, float, complex, str, bytes, np.generic)):
            return type(value), repr(value)
        if isinstance(value, (list, tuple)):
            return type(value).__name__, tuple(self._shape(v, arrays, depth + 1) for v in value)
        if isinstance(value, dict):
            return "dict", tuple((repr(k), self._shape(v, arrays, depth + 1)) for k, v in value.items())
        if isinstance(value, (set, frozenset)):
            return type(value).__name__, tuple(sorted(repr(v) for v in value))
        if isinstance(value, (bytearray, memoryview)):
            return type(value).__name__, bytes(value)
        if isinstance(value, (FunctionType, ModuleType, type)) or callable(value) and not hasattr(value, "__dict__"):
            return "callable", id(value)
        if hasattr(value, "__dict__"):
            return "object", type(value).__name__, self._shape(vars(value), arrays, depth + 1)
        return "other", type(value).__name__, id(value)

    def after_init(self) -> None:
        arrays: list[np.ndarray] = []
        self._shape(self._roots(), arrays)
        for array in arrays:
            chain = array
            while isinstance(chain, np.ndarray):
                if chain.flags.writeable:
                    chain.flags.writeable = False
                    self.frozen.append(chain)
                chain = chain.base
        self.baseline = self._shape(self._roots(), None)

    def check(self, where: str) -> None:
        if self._shape(self._roots(), None) != self.baseline:
            changed = sorted(name for name, value in self._roots().items()
                             if self._shape(value, None) != dict(self.baseline[1]).get(repr(name)))
            raise HiddenStorage(f"module-level data {changed} changed {where}: everything carried between tokens "
                                "must be in the state update returns")

    def release(self) -> None:
        for array in reversed(self.frozen):
            array.flags.writeable = True
        self.frozen.clear()
        self.baseline = None


_SCALARS = frozenset({type(None), bool, int, float, complex, str, bytes})


def _filled(cell) -> bool:
    try:
        cell.cell_contents
    except ValueError:
        return False
    return True


class CarriedStateAudit:
    """The finite carried-state contract at one size (FIX-8): the state init returns and every state update returns
    is made of integers, packs into the declared bits (one index z: 0 <= z < 2**state_bits), and the size reaches at
    most 2**state_bits distinct whole configurations. Card k3's honest witness carries one index in 0..15; one Python
    integer that is one word can otherwise hold any number of configurations."""

    def __init__(self, bits: float):
        self.bits = bits
        self.limit = latent_states(bits)
        self.seen: set = set()
        self.violation: str | None = None

    def check(self, state, where: str) -> None:
        try:
            shape, fields = configuration(state)
        except StateDomainError as exc:
            self._flag(f"{where}: {exc}")
            return
        self.seen.add((shape, fields))
        if shape == ("int",):
            if not 0 <= fields[0] < self.limit:
                self._flag(f"{where}: the carried index is {fields[0]}, outside the declared finite domain "
                           f"0 <= z < {self.limit} (state_bits {self.bits:g})")
        elif packed_bits(fields) > self.bits + 1e-9:
            self._flag(f"{where}: the carried state {_short(state)} packs into {packed_bits(fields)} bits, above the "
                       f"declared state_bits {self.bits:g} (every carried field counts, jointly)")

    def _flag(self, message: str) -> None:
        if self.violation is None:
            self.violation = message

    def violations(self, name: str, size: int) -> list[str]:
        found = [self.violation] if self.violation else []
        if len(self.seen) > self.limit:
            found.append(f"{len(self.seen)} distinct carried configurations at {name}={size}, above the {self.limit} "
                         f"that state_bits {self.bits:g} allows")
        return found


def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def self_fed_from(lay, instance: Instance) -> int | None:
    """First position whose token is the mechanism's own previous prediction, or None when every token is read as
    stored: for copy, every position after the first query (the separator)."""
    if lay["task"] not in SELF_FED or not instance.queries:
        return None
    return instance.queries[0][0] + 1


def run_instance(mechanism, lay, instance: Instance, *, guard: ModuleGuard | None = None,
                 audit: CarriedStateAudit | None = None, finite: bool = False):
    """One instance: (right answers, peak state words, first error). With ``guard`` module-level data must stay as
    init left it after every token; with ``audit`` the carried state is checked at initialization and after every
    token; with ``finite`` every number of the state must be finite after init and after every update, before
    predict reads it (NonFiniteState otherwise, FIX-13)."""
    if guard is not None:
        guard.release()
    state = mechanism["init"](dict(lay))
    if guard is not None:
        guard.after_init()
    if finite:
        _require_finite(state, "after init")
    if audit is not None:
        audit.check(state, "after init")
    peak = words(state)
    wanted = dict(instance.queries)
    right, error = 0, None
    fed_from, last = self_fed_from(lay, instance), None
    for position, token in enumerate(instance.tokens):
        if fed_from is not None and position >= fed_from:
            token = last
        try:
            state = mechanism["update"](state, token)
            if finite:
                _require_finite(state, f"after token {position}")
            got = int(mechanism["predict"](state)) if position in wanted else None
        except ValueError as exc:
            if guard is not None and "read-only" in str(exc):
                raise HiddenStorage(f"the mechanism wrote into a module-level array at token {position} ({exc}): "
                                    "module data is fixed after init") from exc
            raise
        if guard is not None:
            guard.check(f"at token {position}")
        if audit is not None:
            audit.check(state, f"after token {position}")
        peak = max(peak, words(state))
        if got is not None:
            last = got
            if got == wanted[position]:
                right += 1
            elif error is None:
                error = {"position": position, "expected": wanted[position], "predicted": got,
                         "tokens": list(instance.tokens[:COUNTEREXAMPLE_TOKENS]),
                         "tokens_total": len(instance.tokens)}
    return right, peak, error


def _require_finite(state, where: str) -> None:
    found = nonfinite(state)
    if found:
        raise NonFiniteState(f"{where}: {found}")


def run_construction(source: str, task: str, sizes, *, count: int = DEFAULT_COUNT, seed: int = 0,
                     params: dict | None = None,
                     threshold: float = DEFAULT_THRESHOLD, inspect=None, state_bits: dict | None = None) -> dict:
    """Run a hand-set mechanism on ``count`` generated instances at every size. Pass needs the task's metric at
    or above ``threshold`` at every size, with no resource violation there. Returns the obligation-check contract
    plus per-size results. ``inspect(size, lay, instances, mechanism)``, when given, is called after each size has run
    (FIX-4: the architecture survey reads the arrays the mechanism holds at that size). ``state_bits`` ({size: bits})
    is the declared carried-state budget at each size (FIX-8); module-level data is guarded at every size, and every
    number of the carried state must stay finite (FIX-13: a NaN or infinity is a resource violation at that size)."""
    mechanism = load_mechanism(source)
    name = SIZE_NAME[task]
    results = []
    for size in sorted(sizes) if inspect is not None else sizes:
        lay = layout(task, size, params)
        instances = generate(task, size, count, seed, params)
        bits = (state_bits or {}).get(size)
        audit = CarriedStateAudit(bits) if bits is not None else None
        guard = ModuleGuard(mechanism)
        total_q = sum(len(instance.queries) for instance in instances)
        right_q = right_i = peak = 0
        first_error, violations = None, []
        try:
            for index, instance in enumerate(instances):
                try:
                    right, used, error = run_instance(mechanism, lay, instance, guard=guard, audit=audit,
                                                      finite=True)
                except HiddenStorage as breach:
                    violations.append(f"hidden storage at {name}={size}, instance {index}: {breach}")
                    break
                except NonFiniteState as breach:
                    violations.append(f"non-finite carried state at {name}={size}, instance {index}, {breach}: "
                                      "the state's numbers must stay finite at the declared precision")
                    break
                except Exception as exc:  # the mechanism's own fault; the caller reports it with the size
                    raise MechanismCrashed(size, index, exc) from exc
                right_q += right
                right_i += right == len(instance.queries)
                peak = max(peak, used)
                if error is not None and first_error is None:
                    first_error = dict(error, instance=index, size=size)
        finally:
            guard.release()
        if audit is not None:
            violations = audit.violations(name, size) + violations
        if inspect is not None:
            inspect(size, lay, instances, mechanism)
        q_acc, i_acc = right_q / total_q, right_i / len(instances)
        accuracy = i_acc if METRIC[task] == "instance" else q_acc
        results.append(SizeResult(size, len(instances), total_q, q_acc, i_acc, METRIC[task], accuracy,
                                  accuracy >= threshold and not violations, peak, first_error,
                                  bits, len(audit.seen) if audit is not None else None, violations))
    failed = [r for r in results if not r.passed]
    listed = [f"{name}={s}" for s in sizes]
    evidence = (f"{task}: " + ", ".join(f"{name}={r.size} {r.metric} accuracy {r.accuracy:.4f}"
                                        + (f" ({r.carried_states} carried configurations)"
                                           if r.carried_states is not None else "")
                                        for r in results)
                + f"; threshold {threshold}; {tiers.verdict_phrase(tiers.EXECUTED, sizes=listed)}")
    broken = [r for r in results if r.resource_violations]
    if broken:
        evidence += "; resource contract broken: " + "; ".join(v for r in broken for v in r.resource_violations)
    counterexample = None
    if failed:
        worst = failed[0]
        counterexample = {"task": task, "size": worst.size, "accuracy": worst.accuracy, "threshold": threshold,
                          "seed": seed, "first_error": worst.first_error}
        if worst.resource_violations:
            counterexample.update(check="carried_state", resource_violations=worst.resource_violations)
    return {"status": "violated" if failed else "discharged", "evidence": evidence, "counterexample": counterexample,
            "tier": tiers.EXECUTED, "task": task, "threshold": threshold, "seed": seed, "count": count,
            "results": [asdict(r) for r in results]}
