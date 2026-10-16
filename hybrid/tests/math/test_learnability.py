"""Toy learnability, profile obligation P4 (hybrid/math/learnability.py)."""
import pytest

from hybrid.math import learnability, references
from hybrid.tests.research import calibration_fakes as cf

SPEC = {"task": "recall", "size": 3, "examples": 40, "seeds": [0, 1, 2], "test_count": 40}

# A hand-set recall table behind a learn() that ignores its examples. Every learner declares the RECIPE that FIX-3
# requires (the harness applies it to the true data and both controls).
HAND_SET = references.construction_source(references.RECALL_TABLE).replace("def init(layout):", "def init_base(layout):") + '''
RECIPE = {"training": "none"}

def learn(layout, examples, seed):
    return {}

def init(layout, params):
    return init_base(layout)
'''

# A learner that only memorizes training answers by position: it cannot generalize to new tables.
MEMORIZER = '''
RECIPE = {"training": "memorize answers by position"}

def learn(layout, examples, seed):
    table = {}
    for ex in examples:
        for pos, ans in ex["queries"]:
            table[pos] = ans
    return table

def init(layout, params):
    return {"n": -1, "table": params}

def update(state, token):
    state["n"] = state["n"] + 1
    return state

def predict(state):
    return state["table"].get(state["n"], 0)
'''


@pytest.mark.parametrize("task,size", [("copy", 3), ("recall", 3), ("dyck", 2), ("composition", 3)])
def test_the_reference_learners_learn_every_task_from_data(task, size):
    spec = dict(SPEC, task=task, size=size)
    result = learnability.run_learnability(spec, cf.LEARNERS[task])
    assert result["status"] == "discharged", result["evidence"]
    assert result["tier"] == "executed" and "observed on the listed sizes" in result["evidence"]
    assert [r["seed"] for r in result["results"]] == [0, 1, 2]
    assert all(r["accuracy"] >= 0.99 and r["control_accuracy"] < 0.99 for r in result["results"])


def test_a_hand_set_mechanism_behind_learn_is_caught_by_the_data_controls():
    result = learnability.run_learnability(SPEC, HAND_SET)
    assert result["status"] == "violated"
    assert result["counterexample"]["reason"] == "data_independent"
    first = result["results"][0]
    assert first["accuracy"] >= 0.99 and first["empty_control_accuracy"] >= 0.99
    assert first["wrong_control_accuracy"] >= 0.99


def test_a_learner_that_does_not_generalize_fails_below_threshold():
    result = learnability.run_learnability(SPEC, MEMORIZER)
    assert result["status"] == "violated" and result["counterexample"]["reason"] == "below_threshold"


def test_training_and_test_instances_are_disjoint_and_deterministic():
    first = learnability.run_learnability(SPEC, cf.LEARNERS["recall"])
    second = learnability.run_learnability(SPEC, cf.LEARNERS["recall"])
    # The training times are the one wall-clock field of a result, recorded apart from everything that decides it.
    for run in (first, second):
        times = run.pop(learnability.WALL_CLOCK_KEY)
        assert [t["seed"] for t in times["train_s"]] == SPEC["seeds"]
    assert first == second
    lay_train = learnability.as_examples(learnability.constructions.generate("recall", 3, 5, 10_000, None))
    lay_test = learnability.as_examples(learnability.constructions.generate("recall", 3, 5, 20_000, None))
    assert lay_train != lay_test


def test_refused_and_crashing_learners_are_prover_faults():
    refused = learnability.run_learnability(SPEC, "import os\n" + cf.LEARNERS["recall"])
    assert refused["status"] == "unknown" and refused["prover_fault"]
    crashing = learnability.run_learnability(SPEC, cf.LEARNERS["recall"].replace('return {"shift": best}',
                                                                                  'return 1 / 0'))
    assert crashing["status"] == "unknown" and crashing["prover_fault"]


@pytest.mark.parametrize("bad,message", [
    ({"task": "sorting"}, "task must be"),
    ({"seeds": [1, 1, 2]}, "distinct"),
    ({"examples": 0}, "examples"),
    ({"threshold": 1.5}, "threshold"),
    ({"test_count": 0}, "test_count"),
])
def test_spec_problems(bad, message):
    problems = learnability.spec_problems(dict(SPEC, **bad))
    assert any(message in p for p in problems)
    assert learnability.run_learnability(dict(SPEC, **bad), cf.LEARNERS["recall"])["status"] == "unknown"
