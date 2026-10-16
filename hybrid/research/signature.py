"""Normalized mechanism signature (DISCOVERY-PIPELINE-PLAN.md section 9).

A signature names what a mechanism does, not what the card calls it. It is the sha256 of a normal form built from
a skeleton:

- `write`, `read`, `learn`: operators from a fixed vocabulary (WRITE_OPS for the update and readout equations,
  LEARN_OPS for the learning rule), sorted and deduplicated;
- `state`: the rank of each state object (scalar, vector, matrix, higher), with the dimension names dropped;
- `commitments`: the five archive axes when a classifier has assigned them (Phase 6); left out otherwise.

Where the skeleton comes from. The plan has the statement fixer write it, but the fixer runs at stage M, after S2
needs the signature, so every skeleton today comes from `extract_skeleton`, a deterministic reading of the card's
equations. It never reads the title, summary, lens, equation names or state meanings, and it masks every identifier
that is not an operator word before matching: declared state names, other variables and latex symbols all become
numbered placeholders in order of first appearance within an equation. Renaming variables, roles or titles
therefore leaves the signature unchanged, unless a name is renamed into an operator word (a variable renamed
`softmax`). `normalize` also accepts a skeleton written by a model; it rejects operators outside the vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

SIGNATURE_VERSION = 1
AXES = ("substrate", "learning_rule", "unit", "objective", "structure_source")

# Operators of the update and readout equations. Each entry: words (each a regex that must match one whole token
# of the masked text, or a phrase of such words) and symbols (regexes searched in the masked text).
WRITE_OPS: dict[str, dict[str, tuple[str, ...]]] = {
    "outer_product_write": {"words": ("outer", "transpose", "rank-one", "rank one"),
                            "symbols": (r"\\otimes", "⊗", r"\^\s*\{?\s*\\top",
                                        r"einsum\(\s*['\"]\w\s*,\s*\w\s*->\s*\w\w['\"]")},
    "bilinear_product": {"words": ("dot", "matmul", "inner product", "einsum"),
                         "symbols": ("@", r"\\cdot", "·", r"\\langle")},
    "softmax_read": {"words": ("softmax", "logsumexp", "log-sum-exp"), "symbols": ()},
    "exponential_weighting": {"words": ("exp",), "symbols": (r"\\exp",)},
    "argmax_select": {"words": ("argmax", "argmin", "argsort", "arg max", "arg min"),
                      "symbols": (r"\\arg\s*\\?(max|min)", r"\\operatorname\*?\{\s*arg\s*(max|min)")},
    "max_min": {"words": ("max", "min", "maximum", "minimum"), "symbols": (r"(?<!arg)\\(max|min)(?![a-z])",)},
    "sort_or_topk": {"words": ("sort", "sorted", "sorting", "top-k", "topk", "top_k", "descending", "ascending",
                               "ranked", "ranking"), "symbols": ()},
    "normalize": {"words": ("normalize", "normalise", "normalized", "normalised", "normalization",
                            "normalisation", "renormalize", "renormalise", "renormalized", "norm", "unit length"),
                  "symbols": (r"/\s*\(?\s*(\\sum|sum\b|abs\b|\\lvert|\|)", r"\.sum\(\s*\)", r"\\\|")},
    "elementwise_product": {"words": ("hadamard", "elementwise", "element-wise"), "symbols": (r"\\odot", "⊙")},
    "additive_accumulate": {"words": ("cumsum", "accumulate", "accumulates", "accumulated", "accumulation",
                                      "cumulative sum", "prefix sum", "running sum"),
                            "symbols": (r"\+=", r"(§\d+)\s*=\s*\1(?!\d)\s*\+",
                                        r"(§\d+)\[\s*t\s*\]\s*=\s*\1(?!\d)\[\s*t\s*-\s*1\s*\]\s*\+")},
    "gated_update": {"words": ("sigmoid", "gate", "gated", "gates", "gating", "decay", "decays", "decayed", "forget",
                               "leak", "leaky"), "symbols": (r"\(\s*1\s*-\s*§\d+\s*\)\s*(\*|\\cdot)",)},
    "error_correction": {"words": ("prediction error", "residual", "innovation", "delta rule", "error signal",
                                   "td error"),
                         "symbols": (r"\(\s*§\d+(\[[^\]]*\])?\s*-\s*§\d+\s*(\[[^\]]*\])?\s*(@|\\cdot|\*)",)},
    "pointwise_nonlinearity": {"words": ("tanh", "relu", "gelu", "softplus", "arctanh", "sign", "heaviside"),
                               "symbols": (r"\\tanh",)},
    "phase_rotation": {"words": ("sin", "cos", "phase", "phases", "phasor", "phasors", "rotation", "rotate",
                                 "rotates"), "symbols": (r"\\sin", r"\\cos")},
    "complex_conjugate": {"words": ("conj", "conjugate"), "symbols": (r"\\overline", r"\\bar\s*\{")},
    "linear_solve": {"words": ("inv", "pinv", "solve", "solves", "lstsq", "least squares", "least-squares",
                               "inverse", "sherman", "woodbury"),
                     "symbols": (r"\^\s*\{\s*-\s*1\s*\}", r"\^\s*\(?\s*-\s*1(?![0-9])")},
    "matrix_decomposition": {"words": ("qr", "svd", "singular value", "singular values", "eigenvalue",
                                       "eigenvalues", "eigenvector", "eigenvectors", "eigh", "eig", "cholesky",
                                       "orthogonalize", "orthogonalise", "gram-schmidt"), "symbols": ()},
    "modular_or_bitwise": {"words": ("mod", "modulo", "xor", "bitwise", "popcount", "lowbit"),
                           "symbols": (r"\\bmod", r"\\pmod", r"\\oplus", "<<", ">>", "%")},
    "hashing": {"words": ("hash", "hashed", "hashing", "hashes"), "symbols": ()},
    "count_increment": {"words": ("count", "counts", "counter", "counters", "increment", "increments",
                                  "incremented", "tally", "frequency", "frequencies"),
                        "symbols": (r"\+=\s*1(?![0-9.])",)},
    "index_lookup": {"words": ("lookup", "look up", "table", "tables"), "symbols": ()},
    "merge_equivalence_classes": {"words": ("union", "merge", "merges", "merged", "equivalence class",
                                            "equivalence classes", "disjoint-set", "quotient"),
                                  "symbols": (r"(?<![a-z0-9_§])find\s*\(",)},
    "stack_push_pop": {"words": ("push", "pushes", "pushed", "pop", "pops", "popped", "stack"), "symbols": ()},
    "pointer_structure": {"words": ("pointer", "pointers", "linked list", "rewire", "rewires", "tree", "trees",
                                    "node", "nodes", "daughter", "daughters"), "symbols": ()},
    "graph_or_chart": {"words": ("graph", "graphs", "edge", "edges", "chart", "parse", "parser", "cky", "grammar"),
                       "symbols": ()},
    "sampling_step": {"words": ("sample", "samples", "sampled", "sampling", "multinomial", "gumbel", "random"),
                      "symbols": ()},
    "posterior_update": {"words": ("posterior", "bayes", "bayesian", "likelihood", "prior", "evidence",
                                   "dirichlet", "marginal"), "symbols": ()},
    "hypothesis_enumeration": {"words": ("enumerate", "enumerates", "enumeration", "hypothesis", "hypotheses",
                                         "candidate", "candidates", "program", "programs"), "symbols": ()},
    "append_slot": {"words": ("append", "appends", "appended", "allocate", "allocates", "allocated", "insert",
                              "inserts", "store", "stores", "new slot", "new row", "new entry"), "symbols": ()},
    "evict_or_prune": {"words": ("evict", "evicts", "evicted", "delete", "deletes", "remove", "removes", "retire",
                                 "retires", "discard", "discards", "prune", "prunes", "overwrite", "overwrites"),
                       "symbols": ()},
    "threshold_branch": {"words": ("if", "threshold", "thresholds", "otherwise", "vigilance"),
                         "symbols": (">=", "<=", r"\\geq", r"\\leq", r"\\ge(?![a-z])", r"\\le(?![a-z])")},
    "sum_reduction": {"words": ("sum", "prod", "product over"), "symbols": (r"\\sum", r"\\prod", r"\.sum\(")},
    "logarithm": {"words": ("log",), "symbols": (r"\\log",)},
    "inner_optimization": {"words": ("minimize", "minimise", "minimizes", "minimises", "maximize", "maximise",
                                     "maximizes", "maximises", "optimize", "optimise", "linear program",
                                     "fixed point", "iterate until", "converge", "converges"), "symbols": ()},
    "gradient_step": {"words": ("gradient", "gradients", "grad", "backprop", "backpropagation"),
                      "symbols": (r"\\nabla", r"\\partial")},
    "distance_kernel": {"words": ("distance", "gaussian", "rbf", "kernel", "hamming"),
                        "symbols": (r"\|\|[^|]+\|\|\s*\^\s*2",)},
    "convolution_or_transform": {"words": ("convolution", "convolve", "fft", "fourier", "circular"), "symbols": ()},
    "automaton_transition": {"words": ("automaton", "automata", "state machine", "dfa", "fsm", "transition",
                                       "transitions"), "symbols": ()},
    "algebraic_structure": {"words": ("polynomial", "polynomials", "group", "lattice", "ring", "field",
                                      "permutation", "monoid", "semiring"), "symbols": ()},
    "energy_minimization": {"words": ("energy", "hopfield", "attractor"), "symbols": ()},
    "clipping": {"words": ("clip", "clipped", "clamp", "clamped", "saturate", "saturates"), "symbols": ()},
}

# Operators of the learning rule (learning.description and learning.equations).
LEARN_OPS: dict[str, dict[str, tuple[str, ...]]] = {
    "gradient_training": {"words": ("gradient", "gradients", "backprop", "backpropagation", "adam", "sgd",
                                    "cross-entropy", "meta-training", "meta-trained"),
                          "symbols": (r"\\nabla",)},
    "hebbian_local": {"words": ("hebbian", "hebb", "local rule", "local learning"), "symbols": ()},
    "closed_form": {"words": ("closed form", "closed-form", "least squares", "least-squares", "lstsq",
                              "pseudo-inverse", "pseudoinverse"), "symbols": ()},
    "counting": {"words": ("count", "counts", "counting", "frequency", "frequencies", "tally"), "symbols": ()},
    "bayesian_update": {"words": ("posterior", "bayes", "bayesian", "likelihood", "dirichlet", "prior"),
                        "symbols": ()},
    "search_or_induction": {"words": ("search", "enumerate", "enumeration", "induction", "hypothesis",
                                      "hypotheses", "program", "programs"), "symbols": ()},
    "compression_mdl": {"words": ("description length", "mdl", "compression", "compress"), "symbols": ()},
    "evolutionary": {"words": ("evolutionary", "evolution", "mutation", "genetic"), "symbols": ()},
    "no_training": {"words": ("no training", "no learning", "not trained", "not learned", "fixed by hand",
                              "hand-set", "hand set", "hand-coded", "parameter-free", "no gradient"),
                    "symbols": ()},
    "in_context_write": {"words": ("in-context", "in context", "online", "test time", "test-time",
                                   "at inference"), "symbols": ()},
}

# Latex commands kept as written; every other latex command (Greek letters and other symbols used as names) is
# masked like an identifier.
LATEX_KEEP = frozenset(r"""\otimes \odot \oplus \sum \prod \exp \log \max \min \arg \top \overline \bar \nabla
\partial \cdot \langle \rangle \sin \cos \tanh \mathrm \operatorname \frac \pmod \bmod \geq \leq \ge \le \lvert
\rvert \left \right \mathbb \text \quad \qquad \in \to \leftarrow \gets""".split())

IDENT = re.compile(r"\\[A-Za-z]+|[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z][A-Za-z0-9_]*)*")
# Identifiers kept exactly as written: the time index (so x_{t-1} stays readable) and union-find's find().
KEEP_EXACT = frozenset({"t", "find"})
PRE_PASS = (
    # Subscripts as indices, so a name keeps one token: S_{t-1} and S_t become S[t-1] and S[t].
    (re.compile(r"([A-Za-z][A-Za-z0-9]*)_\{([^{}]*)\}"), r"\1[\2]"),
    (re.compile(r"([A-Za-z][A-Za-z0-9]*)_t(?![A-Za-z0-9_])"), r"\1[t]"),
    (re.compile(r"(?<![A-Za-z0-9_\\])e\s*\^\s*(?=[{(])"), "exp"),
    (re.compile(r"\^\s*\{?\s*T\s*\}?(?![A-Za-z0-9_])"), r"^\\top "),
    (re.compile(r"\.T(?![A-Za-z0-9_])"), r"^\\top "),
)
SKELETON_KEYS = {"write", "read", "learn", "state", "commitments"}
RANKS = ("rank0", "rank1", "rank2", "rank3+")


def _word_regex(word: str) -> str:
    return r"(?<![a-z0-9_§\\])" + r"\s+".join(re.escape(part) for part in word.split()) + r"(?![a-z0-9_§])"


@lru_cache(maxsize=None)
def _compiled(table_name: str) -> tuple[tuple[str, tuple[re.Pattern, ...]], ...]:
    """Each pattern is compiled on its own: symbol patterns use back-references (`\1`) to their own group."""
    table = WRITE_OPS if table_name == "write" else LEARN_OPS
    out = []
    for op, spec in table.items():
        words = [re.compile("|".join(f"(?:{_word_regex(word)})" for word in spec["words"]))] if spec["words"] else []
        out.append((op, tuple(words + [re.compile(symbol) for symbol in spec["symbols"]])))
    return tuple(out)


@lru_cache(maxsize=None)
def _kept_words() -> frozenset[str]:
    """Every single word an operator pattern uses. An identifier is kept (lowercased) only when it is one of these."""
    words = set()
    for table in (WRITE_OPS, LEARN_OPS):
        for spec in table.values():
            for phrase in spec["words"]:
                words.update(phrase.split())
    return frozenset(words)


def mask(text: str, state_names: frozenset[str] = frozenset()) -> str:
    """Text in which every identifier that is not an operator word, and every declared state name, is a numbered
    placeholder `§n` in order of first appearance; operator words are lowercased and latex commands in LATEX_KEEP
    stay as written. Notation that would otherwise be masked is rewritten first: subscripts `x_{t-1}` and `x_t` as
    `x[t-1]` and `x[t]`, `e^{...}` as `exp{...}`, a transpose written `^T`, `^{T}` or `.T` as `^\\top`."""
    for pattern, replacement in PRE_PASS:
        text = pattern.sub(replacement, text)
    numbers: dict[str, int] = {}
    kept = _kept_words()

    def placeholder(token: str) -> str:
        return f"§{numbers.setdefault(token, len(numbers) + 1)}"

    def one(token: str) -> str:
        if token.startswith("\\"):
            return token.lower() if token.lower() in LATEX_KEEP else placeholder(token)
        if token not in state_names and (token.lower() in kept or token in KEEP_EXACT):
            return token.lower()
        return placeholder(token)

    def replace(match: re.Match) -> str:
        token = match.group(0)
        if "-" in token and token.lower() not in kept:
            # `v-Sk` is a subtraction, not a word: mask each side.
            return "-".join(one(part) for part in token.split("-"))
        return one(token)

    return IDENT.sub(replace, text)


def detect(text: str, table: str) -> set[str]:
    return {op for op, patterns in _compiled(table) if any(pattern.search(text) for pattern in patterns)}


def state_rank(shape: list) -> str:
    """Rank of a state object with the dimension names dropped; a dimension of size 1 does not count."""
    rank = sum(1 for dim in shape if dim != 1)
    return RANKS[min(rank, 3)]


def _equation_text(equation: dict) -> str:
    return f"{equation['plain']}\n{equation['latex']}"


def extract_skeleton(card: dict) -> dict:
    """The skeleton of a theory card, read from its update, readout and learning equations only."""
    mech = card["mechanism"]
    names = frozenset(item["name"] for item in mech["state"])
    write: set[str] = set()
    for equation in mech["update"]:
        write |= detect(mask(_equation_text(equation), names), "write")
    read: set[str] = set()
    for equation in mech["readout"]:
        read |= detect(mask(_equation_text(equation), names), "write")
    learning = mech["learning"]
    learn = detect(mask(learning["description"], names), "learn")
    for equation in learning.get("equations", []):
        learn |= detect(mask(_equation_text(equation), names), "learn")
    skeleton = {"write": sorted(write), "read": sorted(read), "learn": sorted(learn),
                "state": [list(item["shape"]) for item in mech["state"]]}
    if card.get("commitments"):
        skeleton["commitments"] = dict(card["commitments"])
    return skeleton


def normalize(skeleton: dict) -> dict:
    """The normal form that is hashed: operators checked against the vocabulary, sorted and deduplicated; state
    shapes reduced to their set of ranks; commitments kept only when all five axes are given."""
    unknown_keys = set(skeleton) - SKELETON_KEYS
    if unknown_keys:
        raise ValueError(f"skeleton has unknown keys {sorted(unknown_keys)}")
    form: dict = {"version": SIGNATURE_VERSION}
    for key, table in (("write", WRITE_OPS), ("read", WRITE_OPS), ("learn", LEARN_OPS)):
        ops = skeleton.get(key, [])
        bad = sorted(set(ops) - set(table))
        if bad:
            raise ValueError(f"skeleton.{key} has operators outside the vocabulary: {bad}")
        form[key] = sorted(set(ops))
    ranks = set()
    for entry in skeleton.get("state", []):
        ranks.add(entry if entry in RANKS else state_rank(entry))
    form["state"] = sorted(ranks)
    commitments = skeleton.get("commitments")
    if commitments:
        missing = [axis for axis in AXES if not commitments.get(axis)]
        extra = sorted(set(commitments) - set(AXES))
        if missing or extra:
            raise ValueError(f"commitments need exactly the axes {AXES}; missing {missing}, unknown {extra}")
        form["commitments"] = {axis: str(commitments[axis]) for axis in AXES}
    return form


def signature(skeleton: dict) -> str:
    form = normalize(skeleton)
    text = json.dumps(form, sort_keys=True, separators=(",", ":"))
    return "sig-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def tokens(skeleton: dict) -> list[str]:
    """Signature tokens for nearest-neighbour search (Jaccard): w:, r:, l: operators, s: ranks, c: commitments."""
    form = normalize(skeleton)
    out = [f"w:{op}" for op in form["write"]] + [f"r:{op}" for op in form["read"]]
    out += [f"l:{op}" for op in form["learn"]] + [f"s:{rank}" for rank in form["state"]]
    out += [f"c:{axis}={value}" for axis, value in form.get("commitments", {}).items()]
    return out


def card_signature(card: dict) -> dict:
    """{"signature", "tokens", "skeleton", "source"} for a card, from the deterministic extractor."""
    skeleton = extract_skeleton(card)
    return {"signature": signature(skeleton), "tokens": tokens(skeleton), "skeleton": skeleton,
            "source": "extracted", "version": SIGNATURE_VERSION}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
