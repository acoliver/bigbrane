"""FIX-5: the learner's own CURVE, recorded with every seed, and the wall time of each training, recorded apart from
the result rows; neither decides anything (results/phase4-20261013T073748Z/p4comp-diag/REPORT.md asked for per-seed
curves of the best recipe)."""
import numpy as np

from hybrid.math import architecture, learnability
from hybrid.tests.research import calibration_fakes as cf

SPEC = {"task": "copy", "size": 3, "examples": 40, "seeds": [0, 1, 2], "test_count": 40}

CURVED = cf.COPY_LEARNER.replace(
    "    if not examples:\n",
    "    for step in range(50):\n"
    "        CURVE.append({\"step\": step, \"loss\": np.float64(1.0) / (step + 1), \"note\": \"text\"})\n"
    "    if not examples:\n")


def test_each_training_records_its_curve_and_time_and_the_verdict_is_unchanged():
    plain = learnability.run_learnability(SPEC, cf.COPY_LEARNER)
    curved = learnability.run_learnability(SPEC, CURVED)
    assert curved["status"] == plain["status"] == "discharged"
    assert [r["accuracy"] for r in curved["results"]] == [r["accuracy"] for r in plain["results"]]
    for row in curved["results"]:
        assert set(row["curves"]) == {"true", "empty", "wrong"}
        true = row["curves"]["true"]
        assert true["count"] == 50 and len(true["points"]) == learnability.CURVE_SHOWN
        assert true["points"][0] == {"step": 0.0, "loss": 1.0}          # text fields dropped, numbers kept
        assert true["points"][-1]["step"] == 49.0                       # the last point is always kept
        assert row["curves"]["empty"]["count"] == 50                     # each training starts a fresh list
    for row in plain["results"]:
        assert row["curves"]["true"] == {"points": [], "count": 0}


def test_training_times_sit_in_the_wall_clock_record_and_not_in_the_results():
    result = learnability.run_learnability(SPEC, cf.COPY_LEARNER)
    times = result[learnability.WALL_CLOCK_KEY]["train_s"]
    assert [t["seed"] for t in times] == SPEC["seeds"]
    for t in times:
        assert set(t) == {"seed", "true", "empty", "wrong"} and all(t[k] >= 0 for k in ("true", "empty", "wrong"))
    assert all("train_s" not in row for row in result["results"])
    # Everything except the wall-clock record is the same on a second run.
    again = learnability.run_learnability(SPEC, cf.COPY_LEARNER)
    result.pop(learnability.WALL_CLOCK_KEY), again.pop(learnability.WALL_CLOCK_KEY)
    assert result == again


def test_a_crash_keeps_the_training_times_of_the_seeds_measured_before_it():
    crashing = cf.COPY_LEARNER.replace(
        "def learn(layout, examples, seed):\n",
        "def learn(layout, examples, seed):\n    if seed == 1:\n        raise ValueError(\"no\")\n")
    assert crashing != cf.COPY_LEARNER
    result = learnability.run_learnability(SPEC, crashing)
    assert result["status"] == "unknown" and result["crash"]["seed"] == 1
    assert [r["seed"] for r in result["results"]] == [0]
    assert [t["seed"] for t in result[learnability.WALL_CLOCK_KEY]["train_s"]] == [0]


def test_a_curve_of_non_objects_is_skipped_and_counted():
    learner = {"CURVE": [1, {"loss": float("nan")}, {"loss": 0.5}]}
    assert learnability.curve_record(learner) == {"points": [{"loss": 0.5}], "count": 3, "skipped": 2}
    assert learnability.curve_record({"CURVE": "x"})["problem"] == "CURVE is not a list"


def test_the_architecture_survey_does_not_count_the_curve_as_model_data():
    contract = {"hidden": "d", "precision_bits": 32}
    spec = dict(SPEC, widths={"d": "64", "rounds": "1"})
    wide_curve = CURVED.replace("np.float64(1.0) / (step + 1)", "np.zeros(200)")
    result = learnability.run_learnability(spec, wide_curve, contract)
    assert result["status"] == "discharged", result["evidence"]
    assert not result["architecture"]["violations"]
    # Without the exclusion the same arrays are found as module data and break the contract.
    found = architecture.survey(module=architecture.namespace_values({"CURVE": [{"loss": np.zeros(200)}]}))
    assert any(max(a["shape"]) == 200 for a in found["arrays"])
