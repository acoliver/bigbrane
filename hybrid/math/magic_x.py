"""Magic-X harness (DISCOVERY-PIPELINE-PLAN.md section 5.1).

For an assumption X a card relies on, written as runnable code, the harness gives X to a plain baseline and
measures whether baseline + X alone reaches the card's claimed advantage on the card's own task. If it does, the
advantage is attributed to X and the card fails the magic-X test, unless X is ``proven`` or ``measured`` to be
obtainable from the data at a cost that is included in P1 (``cost_in_p1``).

The assumption's code is ``def x(prefix, layout, ops)``: it sees the tokens read so far (never the answers or
later tokens) and returns a feature for the current position (a number or a tuple of numbers). It runs under the
cost verifier's counter and line tracer, so its cost is measured, not declared. Code that fails to load or run is
reported ``unknown`` (the assumption is ``open``, which blocks the card under section 4.3).

Baseline. The plan names an n-gram, an LSTM or a small transformer, chosen from the assumption's kind. The tools
image has NumPy and no torch, so this harness implements the n-gram only: a count table keyed on the last
``order`` tokens and X's feature, with backoff to the feature alone, then the context alone, then the most common
answer. ``BASELINE_FOR_KIND`` maps every kind to it; an LSTM or transformer baseline is future work and asking for
one raises.

Tasks and accuracy come from constructions.py (per-query accuracy on train and test instances drawn with
different seeds). Self-contained (stdlib, NumPy, SymPy) so it runs in the tools image as part of ``bbmath``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from . import constructions
from .cost import CandidateError, LineTracer, OpCounter, load_functions

ASSUMPTION_KINDS = ("oracle", "store", "prior", "other")
BASELINE_FOR_KIND = {kind: "ngram" for kind in ASSUMPTION_KINDS}
ASSUMPTION_STATUSES = ("proven", "cited", "measured", "open")
EXEMPT_STATUSES = ("proven", "measured")
DEFAULT_ORDER = 2
DEFAULT_TEST_COUNT = 200


@dataclass(frozen=True)
class Assumption:
    id: str
    kind: str
    source: str
    status: str = "open"
    cost_in_p1: bool = False

    def __post_init__(self):
        if self.kind not in ASSUMPTION_KINDS:
            raise ValueError(f"kind must be one of {ASSUMPTION_KINDS}, got {self.kind!r}")
        if self.status not in ASSUMPTION_STATUSES:
            raise ValueError(f"status must be one of {ASSUMPTION_STATUSES}, got {self.status!r}")


@dataclass(frozen=True)
class AdvantageClaim:
    """The card's claim: per-query accuracy ``value`` on ``task`` at ``size`` after ``examples`` training
    instances, within ``tolerance``."""
    task: str
    size: int
    value: float
    tolerance: float
    examples: int
    params: dict | None = None


class NgramBaseline:
    """Counts answers by (last ``order`` tokens, feature), backing off to the feature alone, the context alone and
    the most common answer."""

    def __init__(self, order: int = DEFAULT_ORDER):
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.tables = {name: defaultdict(Counter) for name in ("both", "feature", "context")}
        self.answers = Counter()

    def _keys(self, prefix, feature):
        context = tuple(prefix[-self.order:])
        return {"both": (context, feature), "feature": feature, "context": context}

    def fit(self, rows):
        for prefix, feature, answer in rows:
            for name, key in self._keys(prefix, feature).items():
                if name == "feature" and feature is None:
                    continue
                self.tables[name][key][answer] += 1
            self.answers[answer] += 1

    def predict(self, prefix, feature):
        keys = self._keys(prefix, feature)
        for name in ("both", "feature", "context"):
            if name == "feature" and feature is None:
                continue
            counts = self.tables[name].get(keys[name])
            if counts:
                return _top(counts)
        return _top(self.answers)


def _top(counts: Counter):
    best = max(counts.values())
    return min(answer for answer, count in counts.items() if count == best)


def _feature(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(_feature(v) for v in value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (bool, int, float)):
        return value
    raise CandidateError(f"assumption feature of type {type(value).__name__}; return numbers or tuples")


class _XRunner:
    def __init__(self, assumption: Assumption | None):
        self.fn = load_functions(assumption.source, ("x",)).x if assumption else None
        self.counter = OpCounter()
        self.calls = 0
        self.lines = 0
        self.max_ops = 0
        self.max_lines = 0

    def __call__(self, prefix, lay):
        if self.fn is None:
            return None
        before = self.counter.total
        with LineTracer() as tracer:
            value = self.fn(list(prefix), dict(lay), self.counter)
        self.calls += 1
        self.lines += tracer.lines
        self.max_ops = max(self.max_ops, self.counter.total - before)
        self.max_lines = max(self.max_lines, tracer.lines)
        return _feature(value)

    def cost(self) -> dict:
        return {"calls": self.calls, "ops_total": self.counter.total, "ops_max_per_call": self.max_ops,
                "lines_total": self.lines, "lines_max_per_call": self.max_lines}


def _rows(instances, lay, x):
    for inst in instances:
        for position, answer in inst.queries:
            prefix = inst.tokens[: position + 1]
            yield prefix, x(prefix, lay), answer


def baseline_accuracy(claim: AdvantageClaim, assumption: Assumption | None, *, order: int = DEFAULT_ORDER,
                      test_count: int = DEFAULT_TEST_COUNT, seed: int = 0) -> tuple[float, dict]:
    """Per-query test accuracy of the n-gram baseline, with X's feature when ``assumption`` is given."""
    lay = constructions.layout(claim.task, claim.size, claim.params)
    train = constructions.generate(claim.task, claim.size, claim.examples, seed, claim.params)
    test = constructions.generate(claim.task, claim.size, test_count, seed + 1_000_003, claim.params)
    x = _XRunner(assumption)
    model = NgramBaseline(order)
    model.fit(_rows(train, lay, x))
    right = total = 0
    for prefix, feature, answer in _rows(test, lay, x):
        right += model.predict(prefix, feature) == answer
        total += 1
    return right / total, x.cost()


def magic_x_test(claim: AdvantageClaim, assumption: Assumption, *, order: int = DEFAULT_ORDER,
                 test_count: int = DEFAULT_TEST_COUNT, seed: int = 0) -> dict:
    """Run baseline alone and baseline + X on the card's task. Returns the obligation-check contract: violated
    (the card fails magic-X) when baseline + X reaches the claim within tolerance and X is not exempt."""
    baseline = BASELINE_FOR_KIND[assumption.kind]
    if baseline != "ngram":
        raise NotImplementedError(f"baseline {baseline!r} is not implemented")
    if claim.examples < 1:
        raise ValueError("the claim must state at least one training example")
    try:
        with_x, x_cost = baseline_accuracy(claim, assumption, order=order, test_count=test_count, seed=seed)
    except CandidateError as exc:
        return {"status": "unknown", "evidence": f"assumption {assumption.id} cannot run: {exc}",
                "counterexample": None, "assumption_status": "open"}
    alone, _ = baseline_accuracy(claim, None, order=order, test_count=test_count, seed=seed)
    reaches = with_x >= claim.value - claim.tolerance
    exempt = assumption.status in EXEMPT_STATUSES and assumption.cost_in_p1
    details = {"assumption": assumption.id, "kind": assumption.kind, "baseline": f"ngram(order={order})",
               "accuracy_baseline_alone": round(alone, 6), "accuracy_baseline_plus_x": round(with_x, 6),
               "claimed": claim.value, "tolerance": claim.tolerance, "examples": claim.examples,
               "task": claim.task, "size": claim.size, "x_cost": x_cost, "test_count": test_count, "seed": seed}
    evidence = (f"{claim.task} size {claim.size}, {claim.examples} training instances: baseline alone "
                f"{alone:.4f}, baseline + {assumption.id} {with_x:.4f}, claim {claim.value} +- {claim.tolerance}")
    if reaches and not exempt:
        return {"status": "violated", "evidence": evidence + f"; the advantage is attributable to {assumption.id}",
                "counterexample": details, "kill_code": "magic_x", "claim": asdict(claim)}
    note = (f"; baseline + {assumption.id} reaches the claim, but {assumption.id} is {assumption.status} with its "
            "cost in P1") if reaches else f"; baseline + {assumption.id} stays below the claim"
    return {"status": "discharged", "evidence": evidence + note, "counterexample": None, "details": details,
            "claim": asdict(claim)}
