"""Conservation-law checks: lower bounds a card's numbers must respect (DISCOVERY-PIPELINE-PLAN.md section 5.2).

Each law is a pure function of numbers, with its assumptions and source recorded in ``LAWS``. A check compares a
card's claim with the bound and returns a ``Check``: ``discharged`` when the claim is within the bound, or
``violated`` with a counterexample naming the numbers that break it and the smallest size at which the claim
already fails. No model judgment enters; the inputs are the card's numbers from P1, P2 and P4.

``check_claims(claims)`` takes the card's claims as one dict (schema in its docstring) and runs every law that
applies. ``report(checks)`` turns the result into the obligation-check contract (status, evidence,
counterexample). ``find_state_collision`` gives an executed pigeonhole witness when the card's state code exists.

Self-contained (stdlib only) so it runs in the tools image as part of ``bbmath``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

DISCHARGED = "discharged"
VIOLATED = "violated"
# Slack for comparisons of bounds computed in floating point (log2 of integers, binary entropy).
EPS = 1e-9


@dataclass(frozen=True)
class Law:
    name: str
    function: str
    statement: str
    assumptions: tuple[str, ...]
    source: str


LAWS = {
    "storage": Law(
        "storage", "storage_bound",
        "A state of S bits takes at most 2^S values, so it cannot keep apart more than 2^S contents. Holding K "
        "independent facts of b bits each needs S >= K*b.",
        ("the K facts are independent and each is uniform over 2^b values",
         "every fact can be queried after the state is written, so the state must determine all of them",
         "answers must be exact (error 0); use the copy, recall or index laws for an error rate"),
        "counting (pigeonhole) argument, plan section 5.2"),
    "copy": Law(
        "copy", "copy_error_lower_bound",
        "A model that reads the string into a state of S bits and then writes the copy from that state has "
        "per-string error above 1 - 2^S / D^L on uniform strings of length L over D tokens.",
        ("the string is uniform over D^L values",
         "the output depends on the input only through the state after the separator (a fixed-state or "
         "generalized state space model)",
         "accuracy is per string (every token of the copy right)"),
        "Jelassi, Brandfonbrener, Kakade, Malach, 'Repeat After Me', ICML 2024, Theorem 2.7 "
        "(arXiv 2402.01032); IMPROVEMENT-PLAN.md B.5"),
    "dyck": Law(
        "dyck", "dyck_error_lower_bound",
        "After D unmatched openers of two bracket types the stack is one of 2^D values, and the closing sequence "
        "is a function of the stack, so a one-pass model with S bits of state predicts every closer right on at "
        "most a 2^S / 2^D share of uniform stacks. Exact prediction needs S >= D.",
        ("the D openers are uniform over the two types",
         "the closers fed back during closing are the correct ones, so they carry no new information",
         "accuracy is per sequence (every closer right)"),
        "counting argument, plan section 4.4 P5 (D bits for Dyck-2 depth D); same form as Jelassi Theorem 2.7 "
        "with D=2 tokens and L=depth"),
    "index": Law(
        "index", "streaming_index_bound",
        "INDEX: a stream holds n bits x, then one query i asks for x_i. A single left-to-right pass with S bits "
        "of state answers with error at most e on every query only if S >= n * (1 - h2(e)); S >= n for e = 0.",
        ("x is uniform over {0,1}^n and the query comes after all of x",
         "one pass, no access to x after the state is written (one-way communication from the stream prefix to "
         "the query)",
         "e < 1/2 is the error on each query; h2 is the binary entropy"),
        "Kremer, Nisan, Ron 1999, one-way randomized communication complexity of INDEX is Omega(n) "
        "(doi 10.1007/s000370050018); the constant (1 - h2(e)) is the standard information argument (Fano per "
        "coordinate), derived here, since the plan marks the paper's exact statement UNVERIFIED"),
    "associative_recall": Law(
        "associative_recall", "associative_recall_required_bits",
        "Associative recall with K distinct keys, each bound to a value uniform over V symbols, all pairs before "
        "the queries, reduces to INDEX over K symbols of log2 V bits. Average query error e needs "
        "S >= K * (log2 V - h2(e) - e * log2(V - 1)); S >= K * log2 V for e = 0.",
        ("keys distinct, values independent and uniform over V symbols",
         "all key-value pairs precede the queries (MQAR, Zoology format) and are read in one pass",
         "e is the average error over queries, at most 1 - 1/V"),
        "reduction to INDEX (Kremer, Nisan, Ron 1999) with Fano's inequality per key; plan section 5.2"),
    "sample": Law(
        "sample", "sample_bound",
        "Identifying one of |H| hypotheses needs log2|H| bits. With m examples of at most c bits each and P bits "
        "supplied by the prior or built-in structure, exact identification needs m >= (log2|H| - P) / c; with "
        "error e Fano gives m >= ((1 - e) * log2|H| - 1 - P) / c.",
        ("the target is uniform over H (worst case over the class is at least this)",
         "each example carries at most c bits about the target (for a label from k classes, c <= log2 k)",
         "built-in structure, pretrained parts and hand-set primitives count as prior bits P"),
        "counting and Fano argument, plan sections 4.4 P5 and 5.2; prior-bit accounting from "
        "reports/2026-10-lineages/research-pre2017.md section 18"),
    "read": Law(
        "read", "read_bound",
        "A model whose output depends on every one of n tokens does at least n operations, one read per token.",
        ("every token can change the output",),
        "trivial (plan section 5.2)"),
}


@dataclass(frozen=True)
class Check:
    law: str
    status: str
    claim: dict
    bound: dict
    evidence: str
    counterexample: dict | None

    @property
    def kill_code(self) -> str | None:
        return f"conservation:{self.law}" if self.status == VIOLATED else None

    def to_dict(self) -> dict:
        return asdict(self)


# ----- the bounds ------------------------------------------------------------------------------------------------

def h2(p: float) -> float:
    """Binary entropy in bits."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def facts_bits(facts: int, bits_per_fact: float) -> float:
    """Bits needed to hold ``facts`` independent facts of ``bits_per_fact`` bits each."""
    _nonneg(facts=facts, bits_per_fact=bits_per_fact)
    return facts * bits_per_fact


def storage_bound(state_bits: float, required_bits: float) -> Check:
    """Pigeonhole: a state of ``state_bits`` bits holds at most ``state_bits`` bits of independent content."""
    _nonneg(state_bits=state_bits, required_bits=required_bits)
    claim, bound = {"state_bits": state_bits}, {"required_bits": required_bits}
    if state_bits + EPS >= required_bits:
        return Check("storage", DISCHARGED, claim, bound,
                     f"state of {state_bits} bits holds the {required_bits} bits required", None)
    return Check("storage", VIOLATED, claim, bound,
                 f"state of {state_bits} bits is below the {required_bits} bits required",
                 {"state_bits": state_bits, "required_bits": required_bits,
                  "shortfall_bits": required_bits - state_bits,
                  "witness": f"2^{required_bits:g} equally likely contents map into at most 2^{state_bits:g} states, "
                             "so two contents share a state and a query that tells them apart is answered wrongly "
                             "for one of them"})


def storage_check(state_bits: float, facts: int, bits_per_fact: float) -> Check:
    """Storage law for K facts of b bits; on violation also names the smallest K that already fails."""
    check = storage_bound(state_bits, facts_bits(facts, bits_per_fact))
    claim = dict(check.claim, facts=facts, bits_per_fact=bits_per_fact)
    if check.status == DISCHARGED:
        return Check(check.law, check.status, claim, check.bound, check.evidence, None)
    smallest = math.floor((state_bits + EPS) / bits_per_fact) + 1
    return Check(check.law, check.status, claim, check.bound, check.evidence,
                 dict(check.counterexample, facts=facts, smallest_failing_facts=smallest))


def distinguish_error_lower_bound(state_bits: float, log2_contents: float) -> float:
    """Error lower bound 1 - 2^S / N for recovering one of N = 2^log2_contents uniform contents from S bits."""
    _nonneg(state_bits=state_bits, log2_contents=log2_contents)
    exponent = state_bits - log2_contents
    return 0.0 if exponent >= 0 else 1.0 - 2.0 ** exponent


def copy_error_lower_bound(state_bits: float, D: int, L: int) -> float:
    """Jelassi et al. Theorem 2.7: per-string copy error of a fixed-state model is above 1 - 2^S / D^L."""
    _positive(D=D)
    _nonneg(L=L)
    return distinguish_error_lower_bound(state_bits, L * math.log2(D))


def dyck_error_lower_bound(state_bits: float, depth: int) -> float:
    """Per-sequence error of predicting every closer of a uniform Dyck-2 stack of the given depth."""
    _nonneg(depth=depth)
    return distinguish_error_lower_bound(state_bits, float(depth))


def copy_check(state_bits: float, D: int, L: int, accuracy: float) -> Check:
    _unit(accuracy=accuracy)
    bound = copy_error_lower_bound(state_bits, D, L)
    claim = {"state_bits": state_bits, "D": D, "L": L, "accuracy": accuracy}
    return _accuracy_check("copy", claim, accuracy, 1.0 - bound, {"error_lower_bound": bound},
                           lambda acc: _smallest_failing_length(state_bits, math.log2(D), acc), "L")


def dyck_check(state_bits: float, depth: int, accuracy: float) -> Check:
    _unit(accuracy=accuracy)
    bound = dyck_error_lower_bound(state_bits, depth)
    claim = {"state_bits": state_bits, "depth": depth, "accuracy": accuracy}
    return _accuracy_check("dyck", claim, accuracy, 1.0 - bound, {"error_lower_bound": bound},
                           lambda acc: _smallest_failing_length(state_bits, 1.0, acc), "depth")


def streaming_index_required_bits(n: int, error: float = 0.0) -> float:
    """State bits a one-pass model needs to answer INDEX on n bits with per-query error ``error``."""
    _nonneg(n=n)
    if not 0.0 <= error < 0.5:
        raise ValueError(f"INDEX error must be in [0, 0.5), got {error}")
    return n * (1.0 - h2(error))


def streaming_index_bound(n: int, state_bits: float, error: float = 0.0) -> Check:
    required = streaming_index_required_bits(n, error)
    claim = {"n": n, "state_bits": state_bits, "error": error}
    if state_bits + EPS >= required:
        return Check("index", DISCHARGED, claim, {"required_bits": required},
                     f"{state_bits} bits of state meet the {required:.6g} bits INDEX needs on n={n}", None)
    per_item = 1.0 - h2(error)
    return Check("index", VIOLATED, claim, {"required_bits": required},
                 f"{state_bits} bits of state are below the {required:.6g} bits INDEX needs on n={n} at error {error}",
                 {"n": n, "state_bits": state_bits, "required_bits": required,
                  "smallest_failing_n": math.floor((state_bits + EPS) / per_item) + 1,
                  "witness": "two streams x != x' reach the same state; a query at an index where they differ "
                             "gets one answer, wrong for one of them" if error == 0 else
                             "the state carries less than the n * (1 - h2(e)) bits Fano requires"})


def associative_recall_required_bits(K: int, V: int, error: float = 0.0) -> float:
    """State bits for associative recall of K values over V symbols at average query error ``error``."""
    _nonneg(K=K)
    if V < 2:
        raise ValueError(f"V must be at least 2, got {V}")
    if not 0.0 <= error <= 1.0 - 1.0 / V:
        raise ValueError(f"recall error must be in [0, 1 - 1/V], got {error}")
    per_key = math.log2(V) - h2(error) - error * math.log2(V - 1)
    return K * max(per_key, 0.0)


def associative_recall_check(state_bits: float, K: int, V: int, accuracy: float) -> Check:
    _unit(accuracy=accuracy)
    error = min(1.0 - accuracy, 1.0 - 1.0 / V)
    required = associative_recall_required_bits(K, V, error)
    claim = {"state_bits": state_bits, "K": K, "V": V, "accuracy": accuracy}
    if state_bits + EPS >= required:
        return Check("associative_recall", DISCHARGED, claim, {"required_bits": required},
                     f"{state_bits} bits of state meet the {required:.6g} bits recall of K={K} over V={V} needs "
                     f"at accuracy {accuracy}", None)
    per_key = required / K
    return Check("associative_recall", VIOLATED, claim, {"required_bits": required},
                 f"{state_bits} bits of state are below the {required:.6g} bits recall of K={K} over V={V} "
                 f"needs at accuracy {accuracy}",
                 {"K": K, "V": V, "accuracy": accuracy, "state_bits": state_bits, "required_bits": required,
                  "smallest_failing_K": math.floor((state_bits + EPS) / per_key) + 1})


# ----- information ceilings (FIX-11) -------------------------------------------------------------------------------
# The largest accuracy each law allows a state of S bits at one size: the number the matching check discharges at
# its edge. The laws above are unchanged; these only invert them.

def distinguish_ceiling(state_bits: float, log2_contents: float) -> float:
    """1 - distinguish_error_lower_bound, written as min(1, 2^(S - log2 N)) so a small ceiling such as 2^-60 is not
    lost to cancellation in 1 - (1 - 2^-60)."""
    _nonneg(state_bits=state_bits, log2_contents=log2_contents)
    exponent = state_bits - log2_contents
    return 1.0 if exponent >= 0 else 2.0 ** exponent


def copy_ceiling(state_bits: float, D: int, L: int) -> float:
    """Largest per-string copy accuracy the copy law allows: min(1, 2^S / D^L)."""
    _positive(D=D)
    _nonneg(L=L)
    return distinguish_ceiling(state_bits, L * math.log2(D))


def dyck_ceiling(state_bits: float, depth: int) -> float:
    """Largest per-sequence Dyck-2 accuracy the dyck law allows: min(1, 2^S / 2^depth)."""
    _nonneg(depth=depth)
    return distinguish_ceiling(state_bits, float(depth))


def recall_ceiling(state_bits: float, K: int, V: int) -> float:
    """Largest average per-query recall accuracy the associative-recall law allows a state of S bits: 1 when S
    holds K log2 V bits, otherwise the accuracy at which the required bits fall to S (found by bisection on the
    law's own required-bits function, which falls as the error grows). Never below chance, 1/V, where the law
    requires no bits. The result is on the allowed side, so associative_recall_check discharges it."""
    _nonneg(state_bits=state_bits)
    if state_bits + EPS >= associative_recall_required_bits(K, V, 0.0):
        return 1.0
    low, high = 1.0 / V, 1.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if middle in (low, high):
            break
        if state_bits + EPS >= associative_recall_required_bits(K, V, 1.0 - middle):
            low = middle
        else:
            high = middle
    return low


CEILINGS = {"copy": lambda bits, entry: copy_ceiling(bits, entry["D"], entry["L"]),
            "dyck": lambda bits, entry: dyck_ceiling(bits, entry["depth"]),
            "recall": lambda bits, entry: recall_ceiling(bits, entry["K"], entry["V"])}


def sample_bound(log2_H: float, prior_bits: float, bits_per_example: float, error: float = 0.0) -> int:
    """Smallest number of examples that can identify one of 2^log2_H hypotheses (with error ``error``)."""
    _nonneg(log2_H=log2_H, prior_bits=prior_bits)
    _positive(bits_per_example=bits_per_example)
    if not 0.0 <= error < 1.0:
        raise ValueError(f"error must be in [0, 1), got {error}")
    needed = log2_H - prior_bits if error == 0 else (1.0 - error) * log2_H - 1.0 - prior_bits
    if needed <= EPS:
        return 0
    return math.ceil(needed / bits_per_example - EPS)


def sample_check(log2_H: float, prior_bits: float, bits_per_example: float, examples: int,
                 error: float = 0.0) -> Check:
    required = sample_bound(log2_H, prior_bits, bits_per_example, error)
    claim = {"log2_H": log2_H, "prior_bits": prior_bits, "bits_per_example": bits_per_example,
             "examples": examples, "error": error}
    if examples >= required:
        return Check("sample", DISCHARGED, claim, {"required_examples": required},
                     f"{examples} examples meet the {required} the hypothesis count requires", None)
    return Check("sample", VIOLATED, claim, {"required_examples": required},
                 f"{examples} examples carry at most {examples * bits_per_example:.6g} bits; identifying the target "
                 f"needs {required} examples",
                 {"examples": examples, "required_examples": required,
                  "information_bits": examples * bits_per_example + prior_bits,
                  "unresolved_log2_hypotheses": log2_H - prior_bits - examples * bits_per_example,
                  "witness": "the examples' answer patterns number at most 2^(m*c); with the prior's 2^P they "
                             "cannot single out one of 2^log2_H hypotheses, so two hypotheses give the same data"})


def read_bound(n: int) -> int:
    _nonneg(n=n)
    return n


def read_check(n: int, ops: float) -> Check:
    required = read_bound(n)
    claim = {"n": n, "ops": ops}
    if ops >= required:
        return Check("read", DISCHARGED, claim, {"required_ops": required},
                     f"{ops} operations cover the {n} reads", None)
    return Check("read", VIOLATED, claim, {"required_ops": required},
                 f"{ops} operations cannot read {n} tokens",
                 {"n": n, "ops": ops, "witness": "some token is never read; changing it cannot change the output"})


# ----- card claims -----------------------------------------------------------------------------------------------

CLAIM_KEYS = {
    "storage": ("facts", "bits_per_fact"),
    "copy": ("D", "L", "accuracy"),
    "dyck": ("depth", "accuracy"),
    "index": ("n", "accuracy"),
    "recall": ("K", "V", "accuracy"),
    "sample": ("log2_H", "prior_bits", "bits_per_example", "examples"),
    "reads": ("n", "ops"),
}


def check_claims(claims: dict) -> list[Check]:
    """Run every law that applies to a card's numerical claims.

    ``claims`` holds an optional default ``state_bits`` and lists of entries, each of which may override
    ``state_bits``:

    * ``storage``: {facts, bits_per_fact}
    * ``copy``: {D, L, accuracy} (per-string accuracy)
    * ``dyck``: {depth, accuracy} (per-sequence accuracy on Dyck-2)
    * ``index``: {n, accuracy} (per-query accuracy on INDEX)
    * ``recall``: {K, V, accuracy} (average per-query accuracy)
    * ``sample``: {log2_H, prior_bits, bits_per_example, examples, error (optional)}
    * ``reads``: {n, ops}

    Unknown sections or missing fields raise ValueError: a claim the library cannot read is not discharged.
    """
    unknown = set(claims) - set(CLAIM_KEYS) - {"state_bits"}
    if unknown:
        raise ValueError(f"unknown claim sections {sorted(unknown)}; known: {sorted(CLAIM_KEYS)} and state_bits")
    default_bits = claims.get("state_bits")
    checks = []
    for section, required in CLAIM_KEYS.items():
        for entry in claims.get(section, []):
            missing = [key for key in required if key not in entry]
            if missing:
                raise ValueError(f"{section} entry {entry} lacks {missing}")
            if section in ("sample", "reads"):
                bits = None
            else:
                bits = entry.get("state_bits", default_bits)
                if bits is None:
                    raise ValueError(f"{section} entry {entry} has no state_bits and the claims give no default")
            checks.append(_one(section, entry, bits))
    return checks


def report(checks: list[Check]) -> dict:
    """Obligation-check contract for a list of checks: violated when any law is violated."""
    if not checks:
        raise ValueError("no checks: a card with no numerical claims has nothing to discharge")
    violated = [c for c in checks if c.status == VIOLATED]
    lines = [f"{c.law}: {c.status}: {c.evidence}" for c in checks]
    if not violated:
        return {"status": DISCHARGED, "evidence": "; ".join(lines), "counterexample": None,
                "checks": [c.to_dict() for c in checks]}
    first = violated[0]
    return {"status": VIOLATED, "evidence": "; ".join(lines),
            "counterexample": dict(first.counterexample, law=first.law, kill_code=first.kill_code,
                                   claim=first.claim),
            "kill_codes": sorted({c.kill_code for c in violated}),
            "checks": [c.to_dict() for c in checks]}


def find_state_collision(encode, inputs, answers) -> dict | None:
    """Executed pigeonhole witness: two inputs with different required answers whose encoded states are equal.

    ``encode(input)`` returns the card's state after reading the input, as something with a stable ``repr``
    (bytes, a tuple, a NumPy array's ``tobytes()``). ``answers(input)`` returns what any query must recover.
    Returns {"first", "second", "state"} for the first collision, or None if the inputs map to states that keep
    every pair of different answers apart.
    """
    seen = {}
    for item in inputs:
        key = repr(encode(item))
        if key in seen and answers(seen[key]) != answers(item):
            return {"first": seen[key], "second": item, "state": key[:200]}
        seen.setdefault(key, item)
    return None


def _one(section: str, entry: dict, bits) -> Check:
    if section == "storage":
        return storage_check(bits, entry["facts"], entry["bits_per_fact"])
    if section == "copy":
        return copy_check(bits, entry["D"], entry["L"], entry["accuracy"])
    if section == "dyck":
        return dyck_check(bits, entry["depth"], entry["accuracy"])
    if section == "index":
        _unit(accuracy=entry["accuracy"])
        return streaming_index_bound(entry["n"], bits, 1.0 - entry["accuracy"])
    if section == "recall":
        return associative_recall_check(bits, entry["K"], entry["V"], entry["accuracy"])
    if section == "sample":
        return sample_check(entry["log2_H"], entry["prior_bits"], entry["bits_per_example"], entry["examples"],
                            entry.get("error", 0.0))
    return read_check(entry["n"], entry["ops"])


def _accuracy_check(law, claim, accuracy, max_accuracy, bound, smallest_failing, size_name) -> Check:
    bound = dict(bound, max_accuracy=max_accuracy)
    if accuracy <= max_accuracy + EPS:
        return Check(law, DISCHARGED, claim, bound,
                     f"claimed accuracy {accuracy} is within the bound {max_accuracy:.6g}", None)
    return Check(law, VIOLATED, claim, bound,
                 f"claimed accuracy {accuracy} exceeds the bound {max_accuracy:.6g}",
                 dict(claim, max_accuracy=max_accuracy, **{f"smallest_failing_{size_name}": smallest_failing(accuracy)}))


def _smallest_failing_length(state_bits: float, bits_per_symbol: float, accuracy: float) -> int:
    """Smallest length at which 2^S / 2^(length * bits_per_symbol) drops below ``accuracy``."""
    # accuracy > 2^(S - l*b)  <=>  l > (S - log2 accuracy) / b
    limit = (state_bits - math.log2(accuracy)) / bits_per_symbol
    return math.floor(limit + EPS) + 1


def _nonneg(**values) -> None:
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")


def _positive(**values) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _unit(**values) -> None:
    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
