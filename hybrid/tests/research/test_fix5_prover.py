"""FIX-5 regressions for stage M (results/phase4-20261013T073748Z/FIX-5.md).

1. Output exhaustion. In runs/calibration-smoke-20261013T222017Z k2-run1 the prover call for the single construction
   statement O7.a came back empty at the 64000-token cap from Claude Opus and from Claude Sonnet (adaptive thinking
   spent the whole cap; fixtures/calibration_smoke_20261014/k2_prover_length_empty.json) and the stage error ended the
   card-run. Now a call that held several statements is split once, and a statement whose call is still exhausted has
   a failed attempt; the rounds go on.
2. Recipe carry-over (p4comp-diag/REPORT.md). Each learnability statement's best measured attempt of the run is shown
   in later rounds with its full source, per-seed accuracies, both controls, training times and the learner's CURVE,
   and a recipe that passed on every seed earlier in the run is not changed without a recipe_change_reason.
"""
from __future__ import annotations

import json
from pathlib import Path

from hybrid.llm import OutputExhausted
from hybrid.math import learnability
from hybrid.research.calibration import load_cards
from hybrid.research.stages import m_math
from hybrid.research.stages import prover_memory as pm
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research.test_profile_stage import Harness

RECORDED = json.loads((Path(__file__).parent / "fixtures" / "calibration_smoke_20261014"
                       / "k2_prover_length_empty.json").read_text(encoding="utf-8"))


def recorded_exhaustion() -> OutputExhausted:
    return OutputExhausted(RECORDED["error"], usage={"input": 8252, "output": 64000, "cached": 2154})


def shown_ids(user: str) -> list[str]:
    return [item["statement_id"] for item in cf._shown(user)]


def events(h: Harness, kind: str) -> list[dict]:
    return [e["data"] for e in h.ledger.iter_events(kind)]


def attempts_of(h: Harness) -> list[tuple]:
    return [(e["statement_id"], e["attempt"], e["status"]) for e in events(h, "proof_attempt")]


def prover_calls(h: Harness) -> list[dict]:
    return [c for c in h.llm.calls if c["prompt"] == "prover"]


def k1_script() -> dict:
    return cf.script("k1", load_cards(("k1",))["k1"])


def statement_of_slot(h: Harness, slot: str) -> list[str]:
    record = h.frozen()
    return [s["id"] for o in record["obligations"] if o["claim_id"] == slot for s in o["statements"]]


# ----- 1. output exhaustion is a failed attempt, not a stage error ----------------------------------------------------

def test_the_recorded_exhaustion_of_one_statement_costs_one_attempt_and_the_rounds_go_on(tmp_path):
    script = k1_script()
    scripted = script["prover"]
    raised = []

    def prover(index, user):
        items = cf._shown(user)
        if (len(items) == 1 and items[0]["form"] == "construction" and items[0]["frozen_statement"]["task"] == "copy"
                and not raised):
            raised.append(items[0]["statement_id"])
            raise recorded_exhaustion()
        return scripted(index, user)

    script["prover"] = prover
    h = Harness(tmp_path, "k1", script=script)
    h.m()                                                      # no StageError
    sid = raised[0]
    assert (sid, 1, "output_exhausted") in attempts_of(h)
    assert any(a[0] == sid and a[1] == 2 and a[2] not in ("output_exhausted",) for a in attempts_of(h))
    exhausted = [e for e in events(h, "math_role_call") if e.get("output_exhausted")]
    assert len(exhausted) == 1 and "finish_reason='length'" in exhausted[0]["output_exhausted"]
    assert not events(h, "stage_error")
    retry = next(c["user"] for c in prover_calls(h) if shown_ids(c["user"]) == [sid] and "previous_attempt_failed"
                 in c["user"])
    item = cf._shown(retry)[0]
    assert "whole output limit on reasoning" in item["previous_attempt_failed"]
    assert item["attempt_history"][0].startswith("attempt 1: the reply used the whole output limit")
    assert h.verdict("M")["stage"] == "M"


def test_an_exhausted_call_of_several_statements_is_split_once_in_the_same_round(tmp_path):
    script = k1_script()
    scripted = script["prover"]
    seen, exhausted = [], []

    def prover(index, user):
        ids = shown_ids(user)
        seen.append(ids)
        if len(ids) > 1 and not any(set(ids) <= done for done in exhausted):
            exhausted.append(set(ids))                 # every first-round multi-statement call; its halves answer
            raise recorded_exhaustion()
        return scripted(index, user)

    script["prover"] = prover
    h = Harness(tmp_path, "k1", script=script)
    h.m()
    splits = events(h, "prover_split")
    assert splits, "no multi-statement call was exhausted"
    for split in splits:
        halves = split["into"]
        assert [sid for half in halves for sid in half] == split["statements"] and len(halves) == 2
        for half in halves:
            assert half in seen                                   # each half was asked
        for sid in split["statements"]:
            # The split call charged no attempt: attempt 1 of each statement is the half's checked reply.
            assert (sid, 1, "output_exhausted") not in attempts_of(h)
            assert any(a[0] == sid and a[1] == 1 for a in attempts_of(h))
    marked = [e for e in events(h, "math_role_call") if e.get("split")]
    assert len(marked) == len(splits)


def test_a_half_that_is_still_exhausted_is_a_failed_attempt_with_the_split_named(tmp_path):
    script = k1_script()
    scripted = script["prover"]
    target = []

    def prover(index, user):
        ids = shown_ids(user)
        if not target and len(ids) > 1:
            target.extend(ids)
        if target and set(ids) <= set(target) and len(target) > 1 and "previous_attempt_failed" not in user:
            raise recorded_exhaustion()
        return scripted(index, user)

    script["prover"] = prover
    h = Harness(tmp_path, "k1", script=script)
    h.m()
    for sid in target:
        assert (sid, 1, "output_exhausted") in attempts_of(h)
    retry = next(c["user"] for c in prover_calls(h) if target[0] in shown_ids(c["user"])
                 and "previous_attempt_failed" in c["user"])
    item = next(i for i in cf._shown(retry) if i["statement_id"] == target[0])
    assert "also when asked for fewer statements per call" in item["previous_attempt_failed"]


def test_another_transport_failure_still_stops_the_stage(tmp_path):
    from hybrid.llm import TransportError
    from hybrid.research.stages.common import StageError
    import pytest

    script = k1_script()

    def prover(index, user):
        raise TransportError("all backend entries failed for role 'prover': gateway: HTTP 503")

    script["prover"] = prover
    h = Harness(tmp_path, "k1", script=script)
    with pytest.raises(StageError, match="HTTP 503"):
        h.m()


# ----- 2. the best attempt with its full source and curves ------------------------------------------------------------

CURVE_LEARNER = cf.COPY_LEARNER.replace(
    'RECIPE = {"method": "exhaustive search over a small rule class", "init": "none"}',
    'RECIPE = {"method": "fixed offset", "offset": 1}').replace(
    "    if not examples:\n        return {\"offset\": 0}\n",
    "    for step in range(30):\n        CURVE.append({\"step\": step, \"loss\": 1.0 / (step + 1)})\n"
    "    if not examples:\n        return {\"offset\": 0}\n    return {\"offset\": 1}\n")


def test_the_next_round_shows_the_best_attempt_with_full_source_seeds_controls_and_curves(tmp_path):
    script = k1_script()
    scripted = script["prover"]

    def prover(index, user):
        answer = scripted(index, user)
        for item, attempt in zip(cf._shown(user), answer["attempts"]):
            if item["form"] == "learnability" and item["frozen_statement"]["task"] == "copy":
                attempt["source"] = CURVE_LEARNER
        return answer

    script["prover"] = prover
    h = Harness(tmp_path, "k1", script=script, protocols=False)
    h.m()
    second = [c["user"] for c in prover_calls(h) if '"best_attempt"' in c["user"]]
    item = next(i for u in second for i in cf._shown(u) if i["form"] == "learnability"
                and i["frozen_statement"]["task"] == "copy")
    best = item["best_attempt"]
    assert best["attempt"] == 1 and best["passed_every_seed"] is False
    assert best["source"] == CURVE_LEARNER                       # the whole source, not a prefix
    assert "previous_source" not in item                         # the latest is the best: shown once
    assert best["recipe"] == {"method": "fixed offset", "offset": 1}
    assert [s["seed"] for s in best["seeds"]] == [0, 1, 2]
    seed = best["seeds"][0]
    assert {"accuracy", "empty_control_accuracy", "wrong_control_accuracy", "train_s", "train_cpu_s"} <= set(seed)
    assert set(seed["train_s"]) == {"true", "empty", "wrong"}       # read from the check's wall-clock record
    assert set(seed["train_cpu_s"]) == {"true", "empty", "wrong"}   # decision CT: the CPU time the budget counts
    curve = seed["curve_true_data"]
    assert curve["count"] == 30 and len(curve["points"]) == pm.PROMPT_CURVE_POINTS
    assert curve["points"][0] == {"step": 0.0, "loss": 1.0} and curve["points"][-1]["step"] == 29.0
    assert seed["curve_empty_control"]["count"] == 30
    user = next(u for u in second if item["statement_id"] in shown_ids(u))
    assert "LEARNER HISTORY" in user and "CURVE.append" in user


# ----- 2. a passing recipe is not discarded without a reason ----------------------------------------------------------

WIDE_PASSING = cf.COPY_LEARNER.replace('    return {"offset": best}\n',
                                       '    return {"offset": best, "pad": np.zeros(100)}\n')
OTHER_RECIPE = cf.COPY_LEARNER.replace('"init": "none"}', '"init": "random"}')


def wide_then(second_source: str, reason: str | None):
    script = k1_script()
    scripted = script["prover"]

    def prover(index, user):
        answer = scripted(index, user)
        for item, attempt in zip(cf._shown(user), answer["attempts"]):
            if item["form"] == "learnability" and item["frozen_statement"]["task"] == "copy":
                if "best_attempt" in item:
                    attempt["source"] = second_source
                    if reason is not None:
                        attempt["recipe_change_reason"] = reason
                else:
                    attempt["source"] = WIDE_PASSING
        return answer

    script["prover"] = prover
    return script


def copy_learner_attempts(h: Harness) -> list[dict]:
    sid = statement_of_slot(h, "P4.copy")[0]
    return [e for e in events(h, "proof_attempt") if e["statement_id"] == sid]


def test_a_changed_recipe_after_every_seed_passed_is_refused_without_a_job(tmp_path):
    h = Harness(tmp_path, "k1", script=wide_then(OTHER_RECIPE, None), protocols=False)
    h.m()
    first, second = copy_learner_attempts(h)
    assert first["status"] == "violated"                       # every seed passed; the architecture survey did not
    assert second["status"] == "unknown" and second["refused"] == "passing_recipe_changed"
    check = json.loads((h.run_dir / "artifacts" / "math_check" / second["artifact"]).read_text())
    assert check["job_ids"] == [] and check["prover_fault"] is True
    assert "passed on every seed" in check["evidence"] and "recipe_change_reason" in check["evidence"]
    sources = [json.loads((Path(j.input_dir) / "request.json").read_text()).get("source") for j in h.executor.jobs]
    assert OTHER_RECIPE not in sources


def test_the_same_recipe_with_a_fixed_model_runs(tmp_path):
    h = Harness(tmp_path, "k1", script=wide_then(cf.COPY_LEARNER, None), protocols=False)
    h.m()
    _, second = copy_learner_attempts(h)
    assert "refused" not in second and second["status"] == "discharged"


def test_a_changed_recipe_with_a_reason_runs_and_the_reason_is_recorded(tmp_path):
    h = Harness(tmp_path, "k1", script=wide_then(OTHER_RECIPE, "the survey needs a narrower model"), protocols=False)
    h.m()
    _, second = copy_learner_attempts(h)
    assert "refused" not in second and second["recipe_change_reason"] == "the survey needs a narrower model"


# ----- units --------------------------------------------------------------------------------------------------------

def row(seed, accuracy, passed):
    return {"seed": seed, "accuracy": accuracy, "passed": passed, "control_accuracy": 0.2}


def check(rows, status="violated"):
    return {"status": status, "decided": True, "details": {"results": rows, "recipe": {"lr": 1}}, "evidence": "e"}


def statement(seeds=(0, 1, 2)):
    return {"id": "O13.a", "form": "learnability", "spec": {"seeds": list(seeds)}}


def test_the_best_attempt_ranks_passing_first_then_the_lowest_seed():
    best = pm.BestAttempts()
    s = statement()
    best.note(s, 1, check([row(0, 0.95, False), row(1, 0.99, True), row(2, 0.99, True)]), "A")
    best.note(s, 2, check([row(0, 0.60, False), row(1, 0.61, False), row(2, 0.62, False)]), "B")
    assert best.prompt_block("O13.a")["source"] == "A"
    best.note(s, 3, check([row(0, 1.0, True)]), "C")             # crashed after one seed: missing seeds count 0
    assert best.prompt_block("O13.a")["source"] == "A"
    best.note(s, 4, check([row(0, 0.991, True), row(1, 0.992, True), row(2, 0.993, True)]), "D")
    assert best.prompt_block("O13.a")["source"] == "D" and best.passing["O13.a"]["attempt"] == 4
    best.note(s, 5, {"status": "unknown", "details": {}}, "E")   # measured nothing: never the best
    assert best.prompt_block("O13.a")["attempt"] == 4


def test_recipe_comparison_ignores_formatting_and_numeric_spelling():
    a = 'RECIPE = {"lr": 0.002, "epochs": 280}\ndef learn(l, e, s):\n    return 1\n'
    b = 'RECIPE = {\n    "lr": 2e-3,  # same\n    "epochs": 280,\n}\ndef learn(l, e, s):\n    return 2\n'
    c = 'RECIPE = {"lr": 0.003, "epochs": 280}\n'
    assert pm.recipe_node(a) == pm.recipe_node(b) != pm.recipe_node(c)
    assert pm.recipe_node("def learn():\n    pass\n") is None and pm.recipe_node("RECIPE = {") is None


def test_the_guard_applies_only_after_a_pass_on_every_seed():
    best = pm.BestAttempts()
    s = statement()
    old = 'RECIPE = {"lr": 0.002}\n'
    best.note(s, 1, check([row(0, 0.99, True), row(1, 0.99, True), row(2, 0.5, False)]), old)
    assert best.discarded_without_reason(s, 'RECIPE = {"lr": 0.003}\n', None) is None
    best.note(s, 2, check([row(0, 0.99, True), row(1, 0.99, True), row(2, 0.99, True)]), old)
    assert "attempt 2 of this run passed on every seed" in best.discarded_without_reason(
        s, 'RECIPE = {"lr": 0.003}\n', "")
    assert best.discarded_without_reason(s, 'RECIPE = {"lr": 0.003}\n', "replay disagreed at seed 1") is None
    assert best.discarded_without_reason(s, old + "def learn(l, e, s):\n    return 0\n", None) is None


def test_the_prover_schema_accepts_an_optional_recipe_change_reason():
    from hybrid.research.stages import math_contracts as mc
    import jsonschema
    schema = mc.prover_schema(["O1.a"])
    base = {"statement_id": "O1.a", "proof": None, "source": "x", "notes": ""}
    jsonschema.validate({"attempts": [base]}, schema)
    jsonschema.validate({"attempts": [dict(base, recipe_change_reason="why")]}, schema)
    jsonschema.validate({"attempts": [dict(base, recipe_change_reason=None)]}, schema)


def test_the_check_keeps_the_threshold_and_both_controls():
    # Nothing in FIX-5 changes what decides a learnability check.
    assert learnability.DEFAULT_THRESHOLD == 0.99 and learnability.TRAININGS_ON_M == 2
    assert m_math.holds({"decided": True, "status": "discharged"})
    assert not m_math.holds({"decided": False, "status": "discharged"})
