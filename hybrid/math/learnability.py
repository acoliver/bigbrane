"""Toy learnability check, profile obligation P4 (DISCOVERY-PIPELINE-PLAN.md section 4.4).

From its initial state, using its own learning rule, the mechanism must learn a P2 construction from data in a
solvable case: m training instances of a construction task (constructions.py), then exact accuracy on fresh test
instances, on every listed seed. The card states m before the run; it is part of the frozen statement.

Statement data (frozen): ``task``, ``size``, optional ``params``, ``examples`` (m), ``seeds`` (one entry per training
run), optional ``threshold`` (default 0.99) and ``test_count`` (default 100).

Prover source (audited like every executed candidate: no imports, no private attributes, no eval or getattr; NumPy
as ``np``):

* ``RECIPE``: a module-level dict of plain values, the training recipe the prover chose (initialization, optimizer,
  learning rate, epochs or passes, batch size, and anything else the training uses that the frozen statement leaves
  open). The frozen statement fixes the model class, m, the seeds, the threshold, both controls and the budget; the
  recipe is the prover's (decision FIX-3, 2026-10-13: in the smoke the statement fixer froze an untested schedule,
  width 16, learning rate 0.01 and 50 epochs, that moved the loss from 3.000009 to 2.987527, and no prover could change
  it). The harness records the recipe with the result and applies the same one to the true data and to both controls:
  ``learn`` is one function of one namespace for all three trainings, and a recipe that changes during any training
  stops the check (prover fault);
* ``learn(layout, examples, seed) -> params``: ``examples`` is a list of {"tokens": [...], "queries": [[position,
  answer], ...]} as plain data; ``seed`` is the run's seed; the training follows ``RECIPE``;
* ``init(layout, params) -> state``, ``update(state, token) -> state``, ``predict(state) -> int``: the trained
  mechanism, run on the test instances exactly as the construction harness runs a hand-set one.
* ``CURVE`` (optional, FIX-5): before every training the harness binds the module-level name ``CURVE`` to a fresh
  empty list; ``learn`` may append points such as {"epoch": 10, "loss": 0.41, "train_accuracy": 0.83} to it. The
  harness records each training's points (true data and both controls, at most CURVE_SHOWN of them, evenly spaced)
  with the seed's result. The curve is the learner's own report, it decides nothing, and it is not module data of the
  trained model (the architecture survey skips it). The wall time of each training is recorded apart from the
  results, under ``wall_clock`` (see Result). It exists so
  the next prover round sees how its best recipe trained (results/phase4-20261013T073748Z/p4comp-diag/REPORT.md:
  the passing and the failing recipes differed in ways that only training curves showed).

The learner never sees a test instance: training and test instances come from disjoint generator seeds.

Data controls. A learner that ignores its data (a hand-set construction behind a ``learn`` that returns nothing) would
pass the accuracy test without learning anything. Each seed therefore trains twice more: on no examples at all, and
on the same token sequences with every answer replaced by a random token from the task's answer range, with the same
``learn`` and the same RECIPE as the true training. When both
controls still reach the threshold on the true test answers, the rule did not come from the data and the check fails
with ``data_independent``. Requiring both keeps a real learner from being flagged by luck: a learner that searches a
small rule class can pick the right rule from random answers by chance, or default to it with no data, but not both
unless its answer does not depend on the data. A control that raises counts as not reaching the threshold.

Result: the obligation-check contract, tier ``executed`` (observed on the listed seeds and size, never proven), with
per-seed accuracy, the control accuracy and the peak state words. Everything in the result except ``wall_clock`` is
a function of the statement and the source, so two runs of the same check give equal results. ``wall_clock`` holds
the measurements: ``{"train_s": [{"seed": s, "true": ..., "empty": ..., "wrong": ...}, ...]}``, the wall time in
seconds of each training of each measured seed (the wrong-answer one includes making its data). It decides nothing
and the replay does not compare it.

Timing probe (``timing_probe``, request kind ``learnability_probe``). The whole check, every seed with both controls,
runs in one job under the runner's wall cap. In the 2026-10-13 calibration (card k1) one such job ran out of its
1800 s after 1800 s with no output: 1000 full-batch epochs on m=3000, six substantive trainings in all
(results/phase4-20261013T073748Z/p4-timing/REPORT.md). Before a full job the runner therefore runs this probe: the
first listed seed only, on min(m, PROBE_EXAMPLES) examples, with the learner's own full schedule and both controls,
each phase timed. ``projected_seconds`` scales the phases that grow with m (generating the training data, the true
and the wrong-answer training) by m over the probe's example count, keeps the others (test data, the empty-data
training, the nine test passes) and multiplies by the number of seeds. Linear scaling in m is an assumption: it
holds for full-batch and one-pass training, and a learner whose cost grows faster than m is underestimated. The probe
prints status ``unknown`` and decides nothing; its accuracy on the smaller sample is diagnostic only.
"""
from __future__ import annotations

import ast
import json
import random
import time

import numpy as np

from . import constructions, tiers
from .cost import SAFE_BUILTINS, audit_source, crash_line, describe_exception

DEFAULT_THRESHOLD = 0.99
DEFAULT_TEST_COUNT = 100
MAX_EXAMPLES = 5000
MAX_TEST_COUNT = 500
MAX_SEEDS = 5
TRAIN_SEED_BASE = 10_000
TEST_SEED_BASE = 20_000
CONTROL_SEED_BASE = 30_000
NAMES = ("learn", "init", "update", "predict")
# The one result field that holds wall-clock measurements; nothing else in a result depends on the clock.
WALL_CLOCK_KEY = "wall_clock"
# Training phases of one seed, by the name its wall time is reported under.
TRAINING_PHASES = {"true": "train", "empty": "train_empty", "wrong": "train_wrong"}
# Timing probe: examples it trains on, and the share of the wall cap a projected full check may use (the rest is the
# margin for the projection's error and for the tools VM).
PROBE_EXAMPLES = 300
PROBE_FIT = 0.8
# Phases of one seed, in the order they run; the first three grow with the number of training examples.
SCALED_PHASES = ("generate_train", "train", "train_wrong")
FIXED_PHASES = ("generate_test", "test", "train_empty", "test_empty", "test_wrong")
# Trainings on m examples per seed (true and wrong answers); the empty-data control trains on none.
TRAININGS_ON_M = 2


class LearnerError(ValueError):
    """The prover's learner was refused or could not be loaded."""


def spec_problems(spec: dict) -> list[str]:
    """Why ``spec`` is not a learnability statement this module can run."""
    problems = []
    if not isinstance(spec, dict):
        return ["learnability: statement must be an object"]
    task = spec.get("task")
    if task not in constructions.SIZE_NAME:
        problems.append(f"learnability: task must be one of {sorted(constructions.SIZE_NAME)}")
    size = spec.get("size")
    if not _int(size) or size < 1:
        problems.append("learnability: size must be a positive integer")
    examples = spec.get("examples")
    if not _int(examples) or not 1 <= examples <= MAX_EXAMPLES:
        problems.append(f"learnability: examples (m) must be an integer from 1 to {MAX_EXAMPLES}")
    seeds = spec.get("seeds")
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= MAX_SEEDS or not all(_int(s) and s >= 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        problems.append(f"learnability: seeds must be 1 to {MAX_SEEDS} distinct non-negative integers")
    threshold = spec.get("threshold", DEFAULT_THRESHOLD)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        problems.append("learnability: threshold must be a number in [0, 1]")
    test_count = spec.get("test_count", DEFAULT_TEST_COUNT)
    if not _int(test_count) or not 1 <= test_count <= MAX_TEST_COUNT:
        problems.append(f"learnability: test_count must be an integer from 1 to {MAX_TEST_COUNT}")
    params = spec.get("params")
    if params is not None and not isinstance(params, dict):
        problems.append("learnability: params must be an object or null")
    if spec.get("widths") is not None and task in constructions.SIZE_NAME:
        from . import architecture
        problems += [f"learnability: {p}" for p in architecture.widths_problems(
            spec["widths"], {constructions.SIZE_NAME[task]})]
    if not problems:
        try:
            constructions.layout(task, size, params)
        except ValueError as exc:
            problems.append(f"learnability: {exc}")
    return problems


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


CURVE_NAME = "CURVE"
# Points kept from one training's CURVE (evenly spaced, the last always kept), and most numeric fields per point.
CURVE_SHOWN = 24
MAX_CURVE_FIELDS = 6
RECIPE_NAME = "RECIPE"
MAX_RECIPE_KEYS = 24
RECIPE_HINT = ("RECIPE = {...} at module level: the training recipe you chose (for example init, optimizer, "
               "learning_rate, epochs, batch_size), which the harness applies identically to the true data and both "
               "controls")


def recipe_problems(source: str) -> list[str]:
    """Without running the source: it must assign ``RECIPE`` to a dict literal at module level."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == RECIPE_NAME for t in targets):
            if isinstance(node.value, ast.Dict) and node.value.keys:
                return []
            return [f"line {node.lineno}: {RECIPE_NAME} must be a non-empty dict literal"]
    return [f"the learner source must set {RECIPE_HINT}"]


def _plain_value(value) -> bool:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return True
    return isinstance(value, (list, tuple)) and len(value) <= 16 and all(
        isinstance(v, (str, bool, int, float)) for v in value)


def recipe_of(namespace: dict) -> dict:
    """The learner's RECIPE as plain data, or LearnerError when it is missing or not plain."""
    recipe = namespace.get(RECIPE_NAME)
    if not isinstance(recipe, dict) or not recipe:
        raise LearnerError(f"the learner source must set {RECIPE_HINT}")
    if len(recipe) > MAX_RECIPE_KEYS or not all(isinstance(k, str) and _plain_value(v) for k, v in recipe.items()):
        raise LearnerError(f"{RECIPE_NAME} must map at most {MAX_RECIPE_KEYS} names to numbers, strings, booleans or "
                           "short lists of them")
    return json.loads(json.dumps(recipe))


def load_learner(source: str) -> dict:
    problems = audit_source(source) if isinstance(source, str) and source.strip() else ["code is missing"]
    if problems:
        raise LearnerError("; ".join(problems))
    namespace = {"__builtins__": dict(SAFE_BUILTINS), "np": np}
    exec(compile(source, "<learner>", "exec"), namespace)
    missing = [name for name in NAMES if not callable(namespace.get(name))]
    if missing:
        raise LearnerError(f"learner does not define {missing}")
    namespace["__recipe__"] = recipe_of(namespace)
    namespace["__source__"] = source
    return namespace


def _same_recipe(learner: dict, phase: str) -> None:
    """The recipe recorded at load must still be the one in the namespace after every training."""
    try:
        now = recipe_of(learner)
    except LearnerError as exc:
        raise RecipeChanged(phase, str(exc)) from exc
    if now != learner["__recipe__"]:
        raise RecipeChanged(phase, f"{RECIPE_NAME} was {learner['__recipe__']} at load and {now} after {phase}")


class RecipeChanged(Exception):
    """The learner changed its recipe during a training, so the controls would not have used the same one."""

    def __init__(self, phase: str, detail: str):
        super().__init__(f"the training recipe changed during {phase}: {detail}"[:1000])


def _number(value) -> float | None:
    """A finite plain number from a curve field (NumPy scalars and 0-d arrays included), or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def curve_record(learner: dict) -> dict:
    """The points the last training appended to ``CURVE``, as plain numbers: {"points": [...], "count": n}, at most
    CURVE_SHOWN points evenly spaced with the last kept. Points that are not objects of numbers are skipped and
    counted as ``skipped``; a CURVE that is not a list gives {"points": [], "count": 0, "problem": ...}."""
    found = learner.get(CURVE_NAME)
    if not isinstance(found, list):
        return {"points": [], "count": 0, "problem": f"{CURVE_NAME} is not a list"}
    points, skipped = [], 0
    for item in found:
        if not isinstance(item, dict):
            skipped += 1
            continue
        point = {str(k): _number(v) for k, v in list(item.items())[:MAX_CURVE_FIELDS]}
        point = {k: round(v, 6) for k, v in point.items() if v is not None}
        if point:
            points.append(point)
        else:
            skipped += 1
    if len(points) > CURVE_SHOWN:
        step = (len(points) - 1) / (CURVE_SHOWN - 1)
        points = [points[round(i * step)] for i in range(CURVE_SHOWN)]
    record = {"points": points, "count": len(found)}
    if skipped:
        record["skipped"] = skipped
    return record


def _fresh_curve(learner: dict) -> None:
    learner[CURVE_NAME] = []


def as_examples(instances) -> list[dict]:
    return [{"tokens": list(inst.tokens), "queries": [list(q) for q in inst.queries]} for inst in instances]


def answer_range(lay: dict) -> list[int]:
    """Token ids an answer of the task can take; the control draws its wrong answers from here."""
    task = lay["task"]
    if task == "copy":
        return list(range(lay["D"]))
    if task == "recall":
        return list(range(lay["value_base"], lay["value_base"] + lay["V"]))
    if task == "dyck":
        return list(lay["CLOSE"])
    if task == "composition":
        return list(range(lay["z_base"], lay["z_base"] + lay["T"]))
    return list(range(lay["value_base"], lay["value_base"] + lay["values"]))


def corrupted(examples: list[dict], choices: list[int], seed: int) -> list[dict]:
    rng = random.Random(f"control:{seed}")
    return [{"tokens": list(ex["tokens"]), "queries": [[pos, rng.choice(choices)] for pos, _ in ex["queries"]]}
            for ex in examples]


def _accuracy(learner: dict, lay: dict, params, instances, task: str) -> tuple[float, int, dict | None]:
    mechanism = {"init": lambda layout: learner["init"](layout, params), "update": learner["update"],
                 "predict": learner["predict"]}
    right_q = total_q = right_i = peak = 0
    first_error = None
    for index, instance in enumerate(instances):
        right, used, error = constructions.run_instance(mechanism, lay, instance)
        total_q += len(instance.queries)
        right_q += right
        right_i += right == len(instance.queries)
        peak = max(peak, used)
        if error is not None and first_error is None:
            first_error = dict(error, instance=index)
    accuracy = (right_i / len(instances) if constructions.METRIC[task] == "instance" else right_q / total_q)
    return accuracy, peak, first_error


def _timed(timings: dict, phase: str, fn, *args):
    """``fn(*args)``, adding its wall time to ``timings[phase]`` whether or not it raises."""
    start = time.perf_counter()
    try:
        return fn(*args)
    finally:
        timings[phase] = timings.get(phase, 0.0) + time.perf_counter() - start


def _control(learner: dict, lay: dict, examples: list[dict], seed: int, test, task: str, timings: dict,
             name: str, curves: dict | None = None) -> float:
    """Test accuracy after training on control data; a learner that raises on it scores 0. The training's CURVE is
    put in ``curves[name]``."""
    _fresh_curve(learner)
    try:
        params = _timed(timings, "train_" + name, learner["learn"], dict(lay), examples, seed)
    except Exception:  # a learner may refuse empty or contradictory data; that is dependence on the data
        params = _REFUSED
    if curves is not None:
        curves[name] = curve_record(learner)
    _same_recipe(learner, f"the {name}-data control training")
    if params is _REFUSED:
        return 0.0
    try:
        return _timed(timings, "test_" + name, _accuracy, learner, lay, params, test, task)[0]
    except Exception:  # a model trained on control data may not run; it reached nothing
        return 0.0


_REFUSED = object()


def _plain(value):
    return json.loads(json.dumps(value))


class LearnerFailed(Exception):
    """The learner raised while training on the true examples or while being tested, or changed its recipe."""

    def __init__(self, seed: int, exc: Exception, phase: str = "", source: str | None = None):
        where = crash_line(exc, source, "<learner>")
        self.raised_at = where
        self.phase = phase
        text = str(exc) if isinstance(exc, RecipeChanged) else describe_exception(exc)
        super().__init__((f"learner failed at seed {seed}" + (f" during {phase}" if phase else "") + f": {text}"
                          + (f"; raised at {where}" if where else ""))[:2000])


def _one_seed(learner: dict, lay: dict, spec: dict, seed: int, m: int, timings: dict, inspect=None) -> dict:
    """Train on m examples and on both controls for one seed, test each, and time every phase into ``timings``.
    The full check and the timing probe run exactly this. ``inspect(learned, train)``, when given, is called on the
    model trained on the true examples (FIX-4: the architecture survey)."""
    task, size, params = spec["task"], spec["size"], spec.get("params")
    threshold = spec.get("threshold", DEFAULT_THRESHOLD)
    test_count = spec.get("test_count", DEFAULT_TEST_COUNT)
    train = _timed(timings, "generate_train", lambda: as_examples(
        constructions.generate(task, size, m, TRAIN_SEED_BASE + seed, params)))
    plain = _timed(timings, "generate_train", _plain, train)
    test = _timed(timings, "generate_test", constructions.generate, task, size, test_count, TEST_SEED_BASE + seed,
                  params)
    source = learner.get("__source__")
    phase = "training on the true examples"
    curves: dict = {}
    try:
        _fresh_curve(learner)
        learned = _timed(timings, "train", learner["learn"], dict(lay), plain, seed)
        curves["true"] = curve_record(learner)
        _same_recipe(learner, phase)
        if inspect is not None:
            inspect(learned, train)
        phase = "the test of the trained model"
        accuracy, peak, error = _timed(timings, "test", _accuracy, learner, lay, learned, test, task)
    except Exception as exc:  # the learner crashed; another attempt may fix it
        raise LearnerFailed(seed, exc, phase, source) from exc
    try:
        empty = _control(learner, lay, [], seed, test, task, timings, "empty", curves)
        wrong_data = _timed(timings, "train_wrong", corrupted, train, answer_range(lay), CONTROL_SEED_BASE + seed)
        wrong = _control(learner, lay, wrong_data, seed, test, task, timings, "wrong", curves)
    except RecipeChanged as exc:
        raise LearnerFailed(seed, exc, "a control training", source) from exc
    control = min(empty, wrong)
    passed = accuracy >= threshold and control < threshold
    reason = None
    if accuracy < threshold:
        reason = "below_threshold"
    elif control >= threshold:
        reason = "data_independent"
    return {"seed": seed, "recipe": learner["__recipe__"], "accuracy": round(accuracy, 6),
            "control_accuracy": round(control, 6),
            "empty_control_accuracy": round(empty, 6), "wrong_control_accuracy": round(wrong, 6),
            "passed": passed, "state_peak_words": peak, "first_error": error, "reason": reason,
            # Diagnostics only (FIX-5): the learner's own CURVE points. The wall times go to WALL_CLOCK_KEY.
            "curves": curves}


def training_seconds(seed: int, timings: dict) -> dict:
    """The wall time of each training of one seed, from that seed's own ``timings``."""
    return {"seed": seed, **{name: round(timings.get(phase, 0.0), 3) for name, phase in TRAINING_PHASES.items()}}


def _prepare(spec: dict, source: str) -> tuple[dict | None, dict | None]:
    """The loaded learner, or the unknown result that stops the check before any training."""
    problems = spec_problems(spec)
    if problems:
        return None, {"status": "unknown", "tier": tiers.EXECUTED, "counterexample": None,
                      "evidence": "statement cannot be read: " + "; ".join(problems)}
    try:
        return load_learner(source), None
    except Exception as exc:  # the prover's code is refused or fails to load; another attempt may fix it
        return None, {"status": "unknown", "tier": tiers.EXECUTED, "counterexample": None, "prover_fault": True,
                      "evidence": f"learner refused: {type(exc).__name__}: {exc}"[:2000]}


def _survey_hook(shared: dict, spec: dict, lay: dict, learner: dict, points: list[dict]):
    """The architecture survey of a trained learner (FIX-4, architecture.py): the arrays of the trained model and
    the learner's module data against the statement's widths and the shared hidden width. Training precision is
    recorded, not enforced: the precision of the contract is the inference precision of P2."""
    from . import architecture

    def inspect(learned, train):
        found = architecture.survey(model=learned, module=architecture.namespace_values(
            {k: v for k, v in learner.items() if not k.startswith("__") and k != CURVE_NAME}))
        values = {constructions.SIZE_NAME[spec["task"]]: spec["size"]}
        points.append(architecture.check_point(shared, spec["widths"], values, found,
                                               positions=max((len(e["tokens"]) for e in train), default=1),
                                               vocab=int(lay.get("vocab") or 0), precision=False))
    return inspect


def run_learnability(spec: dict, source: str, architecture_contract: dict | None = None) -> dict:
    """Train and test the prover's learner on every seed of a frozen learnability statement. With the profile's
    architecture contract the trained model must stay within the statement's declared widths (FIX-4)."""
    learner, stop = _prepare(spec, source)
    if stop is not None:
        return stop
    task, size = spec["task"], spec["size"]
    m, threshold = spec["examples"], spec.get("threshold", DEFAULT_THRESHOLD)
    lay = constructions.layout(task, size, spec.get("params"))
    results, failed = [], None
    wall_clock: dict = {"train_s": []}
    points: list[dict] = []
    inspect = (_survey_hook(architecture_contract, spec, lay, learner, points)
               if architecture_contract is not None and spec.get("widths") else None)
    for seed in spec["seeds"]:
        timings: dict = {}
        try:
            results.append(_one_seed(learner, lay, spec, seed, m, timings, inspect))
        except LearnerFailed as exc:
            return {"status": "unknown", "tier": tiers.EXECUTED, "counterexample": None, "prover_fault": True,
                    "evidence": str(exc), "crash": {"seed": seed, "phase": exc.phase, "raised_at": exc.raised_at},
                    "results": results, WALL_CLOCK_KEY: wall_clock}
        wall_clock["train_s"].append(training_seconds(seed, timings))
        if not results[-1]["passed"] and failed is None:
            failed = results[-1]
    listed = [f"{task} {constructions.SIZE_NAME[task]}={size} m={m} seed={r['seed']}" for r in results]
    evidence = (f"{task} at {constructions.SIZE_NAME[task]}={size} from m={m} examples: "
                + ", ".join(f"seed {r['seed']} accuracy {r['accuracy']:.4f} (control {r['control_accuracy']:.4f})"
                            for r in results)
                + f"; threshold {threshold}; recipe {json.dumps(learner['__recipe__'], sort_keys=True)} applied to "
                "the true data and both controls; " + tiers.verdict_phrase(tiers.EXECUTED, sizes=listed))
    base = {"tier": tiers.EXECUTED, "evidence": evidence, "results": results, "task": task, "size": size,
            "examples": m, "threshold": threshold, "recipe": learner["__recipe__"], WALL_CLOCK_KEY: wall_clock}
    if inspect is not None:
        from . import architecture
        base["architecture"] = architecture.summary(points)
        # The controls decide first: a data-independent or failed recipe is reported as such.
        if base["architecture"]["violations"] and failed is None:
            return dict(base, status="violated",
                        evidence=evidence + "; architecture contract broken: "
                        + "; ".join(base["architecture"]["violations"]),
                        counterexample={"task": task, "size": size, "examples": m, "reason": "architecture",
                                        "violations": base["architecture"]["violations"]})
    if failed is not None:
        return dict(base, status="violated", counterexample={"task": task, "size": size, "examples": m,
                                                              "seed": failed["seed"], "reason": failed["reason"],
                                                              "accuracy": failed["accuracy"],
                                                              "control_accuracy": failed["control_accuracy"],
                                                              "first_error": failed["first_error"]})
    return dict(base, status="discharged", counterexample=None)


# ----- timing probe ------------------------------------------------------------------------------------------------

def probe_examples(spec: dict, examples: int = PROBE_EXAMPLES) -> int:
    return min(examples, spec["examples"])


def timing_probe(spec: dict, source: str, examples: int = PROBE_EXAMPLES) -> dict:
    """Time one seed of the check on min(m, ``examples``) examples, both controls included. Never decides: the
    status is always unknown, and a learner that is refused or crashes is reported as in the full check."""
    learner, stop = _prepare(spec, source)
    if stop is not None:
        return dict(stop, probe=True, evidence="timing probe: " + stop["evidence"])
    n, seed = probe_examples(spec, examples), spec["seeds"][0]
    lay = constructions.layout(spec["task"], spec["size"], spec.get("params"))
    timings: dict = {}
    start = time.perf_counter()
    try:
        row = _one_seed(learner, lay, spec, seed, n, timings)
    except LearnerFailed as exc:
        return {"status": "unknown", "tier": tiers.EXECUTED, "counterexample": None, "prover_fault": True,
                "probe": True, "evidence": f"timing probe on {n} examples: {exc}"[:2000],
                "crash": {"seed": seed, "phase": exc.phase, "raised_at": exc.raised_at}}
    timing = {"seed": seed, "examples": n, "total_s": round(time.perf_counter() - start, 6),
              "phases": {phase: round(timings.get(phase, 0.0), 6) for phase in SCALED_PHASES + FIXED_PHASES},
              "probe_accuracy": row["accuracy"]}
    projected = projected_seconds(spec, timing)
    return {"status": "unknown", "tier": tiers.EXECUTED, "counterexample": None, "probe": True, "timing": timing,
            "projected_s": round(projected, 3),
            "evidence": (f"timing probe only, decides nothing: seed {seed} on {n} of m={spec['examples']} examples "
                         f"took {timing['total_s']:.2f} s; projected full check {projected:.1f} s for "
                         f"{len(spec['seeds'])} seeds ({timing_line(timing)}; accuracy {row['accuracy']:.4f} at "
                         f"{n} examples is diagnostic only)")}


def projected_seconds(spec: dict, timing: dict, overhead_s: float = 0.0) -> float:
    """Projected wall time of the full check from a probe's phases: the phases that grow with the training data
    scaled by m over the probe's examples, the others as measured, per seed, plus the job's fixed overhead."""
    phases, scale = timing["phases"], spec["examples"] / timing["examples"]
    per_seed = (sum(phases.get(p, 0.0) for p in SCALED_PHASES) * scale
                + sum(phases.get(p, 0.0) for p in FIXED_PHASES))
    return overhead_s + len(spec["seeds"]) * per_seed


def timing_line(timing: dict) -> str:
    phases = timing["phases"]
    tests = sum(phases.get(p, 0.0) for p in ("test", "test_empty", "test_wrong"))
    return (f"training on true answers {phases.get('train', 0.0):.2f} s, wrong-answer control "
            f"{phases.get('train_wrong', 0.0):.2f} s, empty control {phases.get('train_empty', 0.0):.4f} s, "
            f"three test passes {tests:.2f} s, data "
            f"{phases.get('generate_train', 0.0) + phases.get('generate_test', 0.0):.2f} s")


def task_contract(spec: dict) -> str:
    """The learnability statement's task as the learner will see it: the construction contract measured on the first
    seed's training instances, then where the training and test data come from, the answer ids, and the recipe
    rule."""
    task, size, params = spec["task"], spec["size"], spec.get("params")
    seed = spec["seeds"][0]
    lay = constructions.layout(task, size, params)
    answers = answer_range(lay)
    text = constructions.task_contract(task, [size], params, count=spec["examples"], seed=TRAIN_SEED_BASE + seed)
    return (text + "\nLEARNABILITY DATA: learn(layout, examples, seed) receives the m=" + str(spec["examples"])
            + " training instances of the generator seed " + str(TRAIN_SEED_BASE) + "+seed (the measurement above is "
            "the first seed's), each as {\"tokens\": [...], \"queries\": [[position, answer], ...]}; the test uses "
            + str(spec.get("test_count", DEFAULT_TEST_COUNT)) + " fresh instances of the seed " + str(TEST_SEED_BASE)
            + "+seed, never seen in training. Answers are token ids " + (f"{answers[0]}..{answers[-1]}"
                                                                        if answers == list(range(answers[0],
                                                                                                 answers[-1] + 1))
                                                                        else str(answers))
            + f"; the vocabulary is layout['vocab'] = {lay['vocab']} token ids (0..{lay['vocab'] - 1}); size it from "
            "the layout, not from the largest id in the data. The controls train the same learn with the same RECIPE "
            "on no examples and on these token sequences with answers replaced by random ids from that range.")


def budget_text(wall_s: float, memory_mb: int, cpus: float, seeds: int = 3) -> str:
    """What one learnability check may spend, for the statement fixer and the prover; the caps are the runner's."""
    fit = PROBE_FIT * wall_s
    per_training = fit / (TRAININGS_ON_M * seeds)
    return (f"LEARNABILITY BUDGET (the runner's caps). A learnability check is one job of at most {wall_s:.0f} s wall "
            f"clock, {memory_mb} MB of memory and {cpus:g} CPU, with NumPy and BLAS pinned to one thread; a deciding "
            "run is replayed in a second job under the same cap. Inside that one job, for every seed, the harness "
            "trains the learner three times and tests each result on the test instances: on the m true examples, "
            "on no examples (control) and on the same m token sequences with random answers (control). With "
            f"{seeds} seeds that is {TRAININGS_ON_M * seeds} trainings on m examples, {seeds} on none and "
            f"{3 * seeds} test passes. Before the full job a timing probe, which decides nothing and never "
            f"discharges a statement, runs one seed on min(m, {PROBE_EXAMPLES}) examples with the full training "
            "schedule and both controls, and projects the full check assuming training cost linear in m. A "
            f"projection above {fit:.0f} s ({PROBE_FIT:.0%} of the cap) is refused without running the full job, "
            "and a full job that hits the cap is stopped; either leaves the statement unknown as a resource failure, "
            f"not a pass and not a counterexample. One training on m examples therefore has about {per_training:.0f} "
            "s on one core. Scale for comparison, measured in the 2026-10-13 calibration: a two-layer width-64 "
            "attention model trained full-batch on m=3000 length-8 sequences cost about 0.29 s per epoch with "
            "unoptimized np.einsum gradients and about 0.12 s with optimized contractions, so 1000 epochs took about "
            "290 s per training and the job ran out of its cap.")
