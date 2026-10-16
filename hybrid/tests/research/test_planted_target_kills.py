"""DECISIONS T7c in the planted scoring: a valid card (T5 or T7) killed only by target-gate codes is reported as a
target kill, separately from false kills, per type and overall. Any other kill of a valid card at M or C is still a
false kill. Under the operator override the same rule applies to the override's recorded kill codes."""
import pytest

from hybrid.research import planted_metrics as pm
from hybrid.tests.research.test_planted_metrics import CAP, MAIN_COUNTS, checks, keys_of, perfect, result

FIXED = "obligation_failed:target"
COUPLED = "obligation_failed:target_coupled"


def override(m="pass", c="pass", codes=(), killed_at=None):
    return {"M": m, "C": c, "killed_at": killed_at, "stopped": None, "kill_codes": list(codes)}


def test_the_target_codes_are_the_fixed_size_and_the_coupled_gate():
    assert set(pm.TARGET_CODES) == {FIXED, COUPLED}
    assert pm.target_only([FIXED]) and pm.target_only([FIXED, COUPLED])
    assert not pm.target_only([]) and not pm.target_only([FIXED, "obligation_failed:P2.recall"])
    assert not pm.target_only(["obligation_failed:target_other"])


@pytest.mark.parametrize("stop", ["M", "C"])
@pytest.mark.parametrize("codes", [[FIXED], [COUPLED], [FIXED, COUPLED]])
def test_a_t7_killed_only_by_target_codes_is_a_target_kill_not_a_false_kill(stop, codes):
    judged = pm.judge(result("T7", stop=stop, codes=codes), "T7")
    assert judged["target_kill"] and not judged["false_kill"] and not judged["correct"]


@pytest.mark.parametrize("codes", [["obligation_failed:P2.recall"], [FIXED, "obligation_failed:P2.recall"],
                                   ["magic_x:A1"], [COUPLED, "statement_mismatch"], []])
def test_a_valid_card_killed_with_any_other_code_is_still_a_false_kill(codes):
    judged = pm.judge(result("T7", stop="M", codes=codes), "T7")
    assert judged["false_kill"] and not judged["target_kill"]


def test_a_target_code_at_another_stop_is_neither():
    judged = pm.judge(result("T7", stop="P", codes=[FIXED]), "T7")
    assert not judged["false_kill"] and not judged["target_kill"]
    assert not pm.judge(result("T3", stop="M", codes=[FIXED]), "T3")["target_kill"]


def test_a_t4_killed_by_the_target_gate_is_the_expected_kill_not_a_target_kill():
    judged = pm.judge(result("T4", stop="M", codes=[FIXED]), "T4")
    assert not judged["target_kill"] and not judged["false_kill"]


def test_override_m_is_passed_when_its_only_kill_codes_are_target_codes():
    killed = override(m="fail", codes=[COUPLED], killed_at="M")
    assert pm.override_passed(result("T5", override=killed))
    judged = pm.judge(result("T5", override=killed), "T5")
    assert judged["target_kill"] and not judged["false_kill"] and judged["correct"]


@pytest.mark.parametrize("failed", [
    override(m="fail", codes=[FIXED, "obligation_failed:P2.copy"], killed_at="M"),
    override(m="fail", codes=["obligation_failed:P2.copy"], killed_at="M"),
    override(m="fail", codes=[], killed_at="M"),
    override(m="pass", c="fail", codes=["magic_x:A1"], killed_at="C"),
    override(m="fail", c="fail", codes=[FIXED, "magic_x:A1"], killed_at="M"),
])
def test_an_override_failure_with_any_non_target_code_is_a_false_kill(failed):
    assert not pm.override_passed(result("T5", override=failed))
    judged = pm.judge(result("T5", override=failed), "T5")
    assert judged["false_kill"] and not judged["target_kill"] and not judged["correct"]


def test_a_stopped_override_is_not_passed_even_with_target_codes():
    stopped = dict(override(m="fail", c=None, codes=[FIXED], killed_at="M"), stopped="run budget")
    assert not pm.override_passed(result("T5", override=stopped))


def test_target_kills_are_counted_per_type_and_overall_apart_from_false_kills():
    run, types = perfect(MAIN_COUNTS)
    t5, t7 = keys_of(types, "T5"), keys_of(types, "T7")
    run[t5[0]] = result("T5", override=override(m="fail", codes=[FIXED], killed_at="M"))
    run[t7[0]] = result("T7", stop="M", codes=[COUPLED])
    run[t7[1]] = result("T7", stop="C", codes=[FIXED])
    run[t7[2]] = result("T7", stop="M", codes=["obligation_failed:P2.recall"])
    scored = pm.score_run(run, types, idea_calls_cap=CAP)
    assert scored["per_type"]["T5"]["target_kills"] == 1 and scored["per_type"]["T5"]["false_kills"] == 0
    assert scored["per_type"]["T7"]["target_kills"] == 2 and scored["per_type"]["T7"]["false_kills"] == 1
    assert scored["per_type"]["T1"]["target_kills"] == 0
    assert scored["target_kills"] == {"total": 3, "per_type": {"T5": 1, "T7": 2}}
    false_kills = checks(scored)["false_kills"]
    assert false_kills["value"] == 1 and false_kills["ok"]
    assert "3 more killed only by the target gate" in false_kills["detail"]


def test_target_kills_do_not_fail_the_false_kill_check_but_the_other_checks_still_apply():
    run, types = perfect(MAIN_COUNTS)
    for key in keys_of(types, "T7")[:2]:
        run[key] = result("T7", stop="M", codes=[COUPLED])
    scored = pm.score_run(run, types, idea_calls_cap=CAP)
    found = checks(scored)
    assert found["false_kills"]["value"] == 0 and found["false_kills"]["ok"]
    assert found["t7_survival"]["value"] == 2 and not found["t7_survival"]["ok"]


def test_a_card_with_a_false_kill_in_one_line_and_a_target_kill_in_the_other_is_a_false_kill():
    both = result("T5", override=override(m="fail", c="fail", codes=[FIXED, "magic_x:A1"], killed_at="M"))
    judged = pm.judge(both, "T5")
    assert judged["false_kill"] and not judged["target_kill"]


def test_the_report_names_target_kills_in_the_table_and_beside_the_card():
    run, types = perfect({"T5": 3, "T7": 2, "T1": 3})
    t7 = keys_of(types, "T7")[0]
    run[t7] = result("T7", stop="M", codes=[COUPLED])
    report = pm.score([run, run], types, idea_calls_cap=CAP, audit={"differences": []})
    meta = {"set": "held_back", "cards": 8, "runs": 2, "started": "s", "finished": "f"}
    text = pm.render_markdown(report, [run, run], {k: {"type": t} for k, t in types.items()}, meta)
    assert "| Target kills |" in text
    assert "| T7 | 2 | 1 | 0 | 0 | 0 | 1 | 1 | 2 | 1 |" in text
    assert "target kill (obligation_failed:target_coupled)" in text
