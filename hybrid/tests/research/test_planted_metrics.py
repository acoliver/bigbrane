"""Scoring of the planted-card test (DISCOVERY-PIPELINE-PLAN.md section 13.2) on scripted per-card results: the
thresholds on the 49-card set and their rounded-down form on the 21 held-back cards, stop point, reason, false kills
(main line and override), T7 survival, loop termination, cost, two-run agreement, the check-hash audit, and the
report files."""
import json

import pytest

from hybrid.research import planted_metrics as pm

CAP = 80
MAIN_COUNTS = {"T1": 10, "T2": 6, "T3": 8, "T4": 9, "T5": 6, "T6": 6, "T7": 4}
HELD_COUNTS = {t: 3 for t in MAIN_COUNTS}
# A correct result per type: where a perfect pipeline stops each type and with which code.
GOOD = {"T1": ("P", ["reduces_to_known:ngram"]), "T2": ("P", ["no_difference"]), "T3": ("C", ["magic_x:A1"]),
        "T4": ("M", ["obligation_failed:P1"]), "T5": ("P", ["reduces_to_known:lstm"]),
        "T6": ("G", ["duplicate:card-0000abcd"]), "T7": ("T", [])}


def result(kind, *, stop=None, codes=None, outcome=None, calls=10, over=(), override=None):
    good_stop, good_codes = GOOD[kind]
    stop = good_stop if stop is None else stop
    codes = list(good_codes if codes is None else codes)
    outcome = outcome or ("reached" if stop == "T" else "open" if stop == "open" else "dead")
    if kind == "T5" and override is None:
        override = {"M": "pass", "C": "pass", "killed_at": None, "stopped": None, "kill_codes": []}
    return {"key": None, "card_id": "card-00000001", "stop": stop, "stage": None, "outcome": outcome,
            "kill_codes": codes, "model_calls": calls, "tokens": calls * 150, "executor_s": 1.5,
            "over_budget": list(over), "notes": [], "override": override}


def perfect(counts):
    results, types = {}, {}
    for kind, n in counts.items():
        for i in range(n):
            key = f"pc-{kind.lower()}{i:06d}"
            results[key], types[key] = result(kind), kind
    return results, types


def keys_of(types, kind):
    return sorted(k for k, t in types.items() if t == kind)


def checks(scored):
    return {c["metric"]: c for c in scored["checks"]}


def test_a_perfect_run_meets_every_threshold_and_two_agreeing_runs_are_accepted():
    run, types = perfect(MAIN_COUNTS)
    report = pm.score([run, json.loads(json.dumps(run))], types, idea_calls_cap=CAP,
                      audit={"files": 3, "math_checks": 5, "differences": []})
    assert report["accepted"] and report["agreement"]["rate"] == 1.0
    found = checks(report["runs"][0])
    assert {m for m, c in found.items() if not c["ok"]} == set()
    assert found["catch T1"]["needed"] == 9 and found["catch T2"]["needed"] == 5
    assert found["catch T3"]["needed"] == 7 and found["catch T4"]["needed"] == 8 and found["catch T6"]["needed"] == 5
    assert found["false_kills"]["needed"] == 1 and found["t7_survival"]["needed"] == 3
    assert found["termination"]["value"] == 49


def test_held_back_thresholds_are_the_same_rates_rounded_down():
    run, types = perfect(HELD_COUNTS)
    found = checks(pm.score_run(run, types, idea_calls_cap=CAP))
    assert all(found[f"catch {t}"]["needed"] == 2 for t in pm.CATCH_TYPES)
    assert found["false_kills"]["needed"] == 0 and found["t7_survival"]["needed"] == 2


def test_catch_rate_counts_any_kill_and_fails_below_the_plan_count():
    run, types = perfect(MAIN_COUNTS)
    for key in keys_of(types, "T1")[:2]:
        run[key] = result("T1", stop="T", codes=[], outcome="reached")
    found = checks(pm.score_run(run, types, idea_calls_cap=CAP))
    assert found["catch T1"]["value"] == 8 and not found["catch T1"]["ok"]
    run, types = perfect(MAIN_COUNTS)
    run[keys_of(types, "T1")[0]] = result("T1", stop="T", codes=[], outcome="reached")
    # A kill at the wrong stage with the wrong code is still a catch.
    run[keys_of(types, "T3")[0]] = result("T3", stop="M", codes=["obligation_failed:P2.copy"])
    found = checks(pm.score_run(run, types, idea_calls_cap=CAP))
    assert found["catch T1"]["ok"] and found["catch T3"]["value"] == 8


@pytest.mark.parametrize("kind,stop,codes", [
    ("T1", "G", ["excluded_family:linear_attention"]), ("T1", "G", ["not_radical:R1"]), ("T1", "G", ["no_difference"]),
    ("T2", "G", ["no_difference"]), ("T2", "P", ["not_radical:R3"]), ("T3", "C", ["unproven:A2"]),
    ("T3", "C", ["magic_x_missing:A1"]), ("T3", "M", ["unproven:A3"]), ("T4", "C", ["conservation:storage"]), ("T4", "M", ["obligation_failed:O7"]),
    ("T5", "G", ["not_radical:R1"]), ("T6", "P", ["no_difference"]), ("T6", "P", ["not_radical:R3"]),
])
def test_expected_stops_and_codes_per_type_including_g_for_the_p_types(kind, stop, codes):
    judged = pm.judge(result(kind, stop=stop, codes=codes), kind)
    assert judged["stop_ok"] and judged["reason_ok"] and judged["correct"]


@pytest.mark.parametrize("kind,stop,codes,stop_ok,reason_ok", [
    ("T1", "M", ["obligation_failed:P1"], False, False),
    ("T2", "P", ["reduces_to_known:ngram"], True, False),
    ("T3", "M", ["obligation_failed:P2.recall"], False, False),
    ("T3", "M", ["magic_x:A1"], False, True),
    ("T4", "P", ["reduces_to_known:ngram"], False, False),
    ("T6", "C", ["no_difference"], False, True),
    ("T4", "M", ["stall:prover"], True, False),
])
def test_wrong_stop_or_reason_is_judged_wrong(kind, stop, codes, stop_ok, reason_ok):
    judged = pm.judge(result(kind, stop=stop, codes=codes), kind)
    assert (judged["stop_ok"], judged["reason_ok"], judged["correct"]) == (stop_ok, reason_ok, False)


def test_stop_and_reason_accuracy_are_shares_of_caught_cards():
    run, types = perfect(MAIN_COUNTS)
    # 45 killed cards of T1 to T6 (T5 is killed at P as known). 7 at the wrong stage: 38/45 = 0.844 < 0.85; 6 pass.
    for key in keys_of(types, "T4")[:7]:
        run[key] = result("T4", stop="P", codes=["obligation_failed:P1"])
    found = checks(pm.score_run(run, types, idea_calls_cap=CAP))
    assert found["stop_accuracy"]["value"] == pytest.approx(38 / 45) and not found["stop_accuracy"]["ok"]
    assert found["reason_accuracy"]["ok"]
    run[keys_of(types, "T4")[0]] = result("T4")
    assert checks(pm.score_run(run, types, idea_calls_cap=CAP))["stop_accuracy"]["ok"]
    run, types = perfect(MAIN_COUNTS)
    # 10 of 45 with a code outside the type's family: 35/45 = 0.78 < 0.80; 9 (36/45 = 0.80) pass.
    for key in keys_of(types, "T1"):
        run[key] = result("T1", codes=["statement_mismatch"])
    assert not checks(pm.score_run(run, types, idea_calls_cap=CAP))["reason_accuracy"]["ok"]
    run[keys_of(types, "T1")[0]] = result("T1")
    assert checks(pm.score_run(run, types, idea_calls_cap=CAP))["reason_accuracy"]["ok"]


def test_false_kills_count_valid_cards_killed_at_m_or_c_in_the_line_or_under_the_override():
    run, types = perfect(MAIN_COUNTS)
    t7, t5 = keys_of(types, "T7"), keys_of(types, "T5")
    run[t7[0]] = result("T7", stop="M", codes=["obligation_failed:P2.recall"])
    found = checks(pm.score_run(run, types, idea_calls_cap=CAP))
    assert found["false_kills"]["value"] == 1 and found["false_kills"]["ok"]
    run[t5[0]] = result("T5", override={"M": "pass", "C": "fail", "killed_at": "C", "stopped": None,
                                        "kill_codes": ["magic_x:A1"]})
    scored = pm.score_run(run, types, idea_calls_cap=CAP)
    assert checks(scored)["false_kills"]["value"] == 2 and not checks(scored)["false_kills"]["ok"]
    assert scored["judged"][t5[0]]["false_kill"] and not scored["judged"][t5[0]]["correct"]
    # A valid card killed at P (flagged known) is not a false kill; a T7 killed at G is wrong but not a false kill.
    assert not pm.judge(result("T5"), "T5")["false_kill"]
    wrong = pm.judge(result("T7", stop="G", codes=["no_difference"]), "T7")
    assert not wrong["false_kill"] and not wrong["correct"]


def test_t7_survival_needs_three_of_four_reaching_t():
    run, types = perfect(MAIN_COUNTS)
    t7 = keys_of(types, "T7")
    run[t7[0]] = result("T7", stop="P", codes=["no_difference"])
    assert checks(pm.score_run(run, types, idea_calls_cap=CAP))["t7_survival"]["ok"]
    run[t7[1]] = result("T7", stop="open", codes=[])
    found = checks(pm.score_run(run, types, idea_calls_cap=CAP))
    assert found["t7_survival"]["value"] == 2 and not found["t7_survival"]["ok"]


def test_termination_fails_on_an_open_card_an_overrun_or_an_unfinished_override():
    run, types = perfect(MAIN_COUNTS)
    run[keys_of(types, "T3")[0]] = result("T3", stop="open", codes=[])
    found = checks(pm.score_run(run, types, idea_calls_cap=CAP))
    assert found["termination"]["value"] == 48 and not found["termination"]["ok"]
    run, types = perfect(MAIN_COUNTS)
    key = keys_of(types, "T4")[0]
    run[key] = result("T4", over=["math_revision rounds 4 above the cap 3"])
    found = checks(pm.score_run(run, types, idea_calls_cap=CAP))
    assert not found["termination"]["ok"] and key in found["termination"]["detail"]
    run, types = perfect(MAIN_COUNTS)
    run[keys_of(types, "T5")[0]] = result("T5", override={"M": "pass", "C": None, "killed_at": None,
                                                          "stopped": "run budget", "kill_codes": []})
    scored = pm.score_run(run, types, idea_calls_cap=CAP)
    assert not checks(scored)["termination"]["ok"] and not scored["passed"]


def test_cost_reports_median_and_maximum_and_checks_the_per_idea_cap():
    run, types = perfect(MAIN_COUNTS)
    run[keys_of(types, "T7")[0]] = result("T7", calls=81)
    scored = pm.score_run(run, types, idea_calls_cap=CAP)
    assert scored["costs"]["model_calls"] == {"median": 10.0, "max": 81.0}
    assert scored["costs"]["tokens"]["max"] == 81 * 150 and scored["costs"]["executor_s"]["median"] == 1.5
    assert not checks(scored)["cost_cap"]["ok"]


def test_agreement_compares_stop_kill_families_and_override_verdicts():
    run, types = perfect(MAIN_COUNTS)
    other = json.loads(json.dumps(run))
    # Same family, different detail: agrees.
    other[keys_of(types, "T4")[0]]["kill_codes"] = ["obligation_failed:P2.nesting"]
    assert pm.agreement([run, other])["rate"] == 1.0
    split = keys_of(types, "T3")[:5]
    for key in split[:4]:
        other[key]["stop"] = "M"
    other[split[4]]["kill_codes"] = ["unproven:A1"]
    other[keys_of(types, "T5")[0]]["override"]["C"] = "fail"
    agree = pm.agreement([run, other])
    assert agree["split"] == sorted(split + [keys_of(types, "T5")[0]])
    assert agree["rate"] == pytest.approx(43 / 49)
    report = pm.score([run, other], types, idea_calls_cap=CAP, audit={"differences": []})
    assert not report["agreement"]["ok"] and not report["accepted"]
    assert pm.agreement([run])["rate"] is None
    assert not pm.score([run], types, idea_calls_cap=CAP, audit={"differences": []})["accepted"]


def test_any_check_hash_difference_fails_acceptance():
    run, types = perfect(MAIN_COUNTS)
    report = pm.score([run, run], types, idea_calls_cap=CAP,
                      audit={"files": 10, "math_checks": 4, "differences": ["after run 1: hybrid/math/cost.py changed"]})
    assert all(r["passed"] for r in report["runs"]) and report["agreement"]["ok"]
    assert not report["audit"]["ok"] and not report["accepted"]


def test_report_markdown_and_json_carry_every_metric():
    run, types = perfect(HELD_COUNTS)
    t4 = keys_of(types, "T4")[0]
    run[t4] = result("T4", stop="P", codes=["no_difference"])
    report = pm.score([run, run], types, idea_calls_cap=CAP,
                      audit={"files": 7, "math_checks": 2, "differences": ["after run 2: x changed"]})
    meta = {"set": "held_back", "cards": 21, "runs": 2, "started": "s", "finished": "f"}
    entries = {key: {"type": kind} for key, kind in types.items()}
    text = pm.render_markdown(report, [run, run], entries, meta)
    assert "Accepted (plan section 13.2): no." in text
    for metric in ("catch T1", "stop_accuracy", "reason_accuracy", "false_kills", "t7_survival", "termination",
                   "cost_cap"):
        assert f"| {metric} |" in text
    assert f"- {t4} T4 (card-00000001): stop P (expected M/C), codes no_difference" in text and "WRONG" in text
    assert "Repeatability: 100.0% of cards agree over 2 run(s) (needed 90.0%): met." in text
    # 17 of 18 caught cards stopped right; costs are numbers, not percentages.
    assert "| stop_accuracy | 94.4% | 85.0% | yes |" in text
    assert "executor_s 1.50 / 1.50" in text and "model_calls 10 / 10" in text
    assert "Check integrity: 1 difference(s) in 7 audited files and 2 math checks: not met." in text
    assert "- after run 2: x changed" in text
    back = json.loads(pm.dumps(report))
    assert back["accepted"] is False and back["thresholds"]["agreement"] == 0.9
