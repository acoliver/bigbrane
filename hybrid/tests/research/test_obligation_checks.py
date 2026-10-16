"""S3 executed obligation checks: classification, executor runs with replay, judge input, gate rules, report, plumb.

The stages run against the scripted LLM, LocalExecutor (check programs really execute) and the fake bench.
"""
import asyncio
import json

import pytest

from hybrid.research.inspect import inspect_run
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.plumb import PlumbError, parse_stages
from hybrid.research.stages import obligation_checks as oc
from hybrid.research.stages.common import ExecResult, load_prompt
from hybrid.research.stages.s3_feasibility import derived_vote, settle_split
from hybrid.tests.research.test_orchestrator_fakes import (make_orchestrator, obligation_check, orch_config,
                                                           program)

SHIPPED = ROOT / "hybrid" / "config" / "run.toml"


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261011T000000Z"


def run(orch):
    return asyncio.run(orch.run())


def only_card(orch):
    (card_id,) = orch.store.list_cards()
    return orch.store.get_card(card_id)


def s3(card):
    return card["status"]["verdicts"]["S3"]


def artifacts(orch, kind):
    return [json.loads((orch.run_dir / e["data"]["path"]).read_text())
            for e in orch.ledger.iter_events("artifact") if e["data"]["kind"] == kind]


def report(status, evidence="computed", counterexample=None):
    return program({"status": status, "evidence": evidence, "counterexample": counterexample})


def checker(o0=None, o1=None, kinds=("complexity", "numerical_stability", "learnability")):
    return {"checks": [obligation_check("o0", kinds[0], o0), obligation_check("o1", kinds[1], o1),
                       obligation_check("o2", kinds[2], None)]}


def judge(verdict, statuses, fatal=()):
    return {"verdict": verdict, "obligations": [{"name": n, "status": s, "argument": "arg"} for n, s in statuses],
            "fatal_issues": list(fatal), "confidence": 0.7}


def formalizer(optional=(False, False, False)):
    return {"obligations": [{"name": f"o{i}", "kind": kind, "claim": "c", "statement": "s", "argument": "a",
                             "known_bounds": "", "optional": flag}
                            for i, (kind, flag) in enumerate(zip(("complexity", "other", "learnability"), optional))]}


FEASIBLE = judge("feasible", [("o0", "discharged"), ("o1", "discharged"), ("o2", "discharged")])


# ----- the line --------------------------------------------------------------------------------


def test_checker_runs_between_disprover_and_judges_and_records_one_artifact_per_obligation(run_dir):
    orch, llm = make_orchestrator(run_dir)
    run(orch)
    prompts = llm.prompts()
    assert prompts.index("disprover") < prompts.index("obligation_checker") < prompts.index("feasibility_judge_a")
    checker_call = next(c for c in llm.calls if c["prompt"] == "obligation_checker")
    assert checker_call["role"] == "obligation_checker"
    assert checker_call["schema"]["properties"]["checks"]["items"]["properties"]["name"]["enum"] == ["o0", "o1", "o2"]
    assert '"name": "o0"' in checker_call["user"] and "CARD:" in checker_call["user"]
    records = {r["obligation"]: r for r in artifacts(orch, "obligation_check")}
    assert set(records) == {"o0", "o1", "o2"}
    assert records["o0"]["status"] == "discharged" and records["o0"]["decided"] and records["o0"]["replayed"]
    assert len(records["o0"]["runs"]) == 2
    assert records["o1"]["status"] == "not_checked"
    assert records["o1"]["problems"] == ["checkable kind, but the checker wrote no program"]
    assert records["o2"]["status"] == "not_checked" and records["o2"]["kind"] == "learnability"
    purposes = sorted(json.loads((orch.run_dir / e["data"]["path"]).read_text())["purpose"]
                      for e in orch.ledger.iter_events("artifact") if e["data"]["kind"] == "executor")
    assert purposes.count("obligation_check") == 1 and purposes.count("obligation_check_replay") == 1
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "pass" and verdict["reasons"][0] == "gate rule: judges"
    for job in (run["job_id"] for run in records["o0"]["runs"]):
        assert job in verdict["by_calls"]
    assert checker_call["call_id"] in verdict["by_calls"]
    assert any(r.startswith("obligation check o0 (complexity): discharged (replayed)") for r in verdict["reasons"])
    (gate,) = artifacts(orch, "s3_gate")
    assert gate["rule"] == "judges" and gate["verdict"] == "pass" and gate["all_checkable_discharged"] is False


def test_formalizer_is_asked_for_optional_and_judges_get_the_checks(run_dir):
    orch, llm = make_orchestrator(run_dir)
    run(orch)
    form = next(c for c in llm.calls if c["prompt"] == "formalizer")
    assert oc.OPTIONAL_NOTE in form["user"]
    assert "optional" in form["schema"]["properties"]["obligations"]["items"]["required"]
    for role in ("feasibility_judge_a", "feasibility_judge_b"):
        user = next(c["user"] for c in llm.calls if c["prompt"] == role)
        assert "EXECUTED OBLIGATION CHECKS" in user and "3 ops per token" in user
        assert '"reproduced_by_replay": true' in user
        assert '"verdict"' not in user
    disprover = next(c["user"] for c in llm.calls if c["prompt"] == "disprover")
    assert "EXECUTED OBLIGATION CHECKS" not in disprover


def test_replayed_violation_of_a_stated_obligation_is_infeasible_proved_despite_feasible_judges(run_dir):
    script = {"obligation_checker": checker(o1=report("violated", "state norm 1e39 at t=40",
                                                      {"dtype": "float32", "t": 40, "norm": 1e39}))}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "infeasible_proved"
    assert verdict["reasons"][0] == "gate rule: check_violation"
    assert any("violated stated obligation 'o1' (replayed)" in r and '"t": 40' in r for r in verdict["reasons"])
    (gate,) = artifacts(orch, "s3_gate")
    assert gate["rule"] == "check_violation" and gate["votes"] == {"feasibility_judge_a": "feasible",
                                                                   "feasibility_judge_b": "feasible"}
    assert "builder" not in llm.prompts()


def test_violation_of_an_optional_obligation_does_not_gate(run_dir):
    script = {"formalizer": formalizer(optional=(False, True, False)),
              "obligation_checker": checker(o1=report("violated", "fails", {"x": 1})),
              "feasibility_judge_a": FEASIBLE, "feasibility_judge_b": FEASIBLE}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "pass" and verdict["reasons"][0] == "gate rule: judges"
    assert any(r.startswith("obligation check o1 (numerical_stability, optional): violated") for r in verdict["reasons"])


def test_violation_without_counterexample_breaks_the_contract_and_is_unknown(run_dir):
    script = {"obligation_checker": checker(o1=report("violated", "fails", None))}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    assert s3(only_card(orch))["verdict"] == "pass"
    record = next(r for r in artifacts(orch, "obligation_check") if r["obligation"] == "o1")
    assert record["status"] == "unknown" and not record["decided"] and len(record["runs"]) == 1
    assert "violated without a counterexample" in record["problems"][0]


def test_violation_that_the_replay_does_not_reproduce_is_unknown(run_dir):
    flaky = ("import json, os\nprint(json.dumps({'status': 'violated', 'evidence': 'random',"
             " 'counterexample': {'seed': os.urandom(8).hex()}}))\n")
    script = {"obligation_checker": checker(o1=flaky)}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    assert s3(only_card(orch))["verdict"] == "pass"
    record = next(r for r in artifacts(orch, "obligation_check") if r["obligation"] == "o1")
    assert record["status"] == "unknown" and len(record["runs"]) == 2 and not record["replayed"]
    assert record["problems"][0].startswith("replay disagreed")


def test_crashing_program_is_unknown(run_dir):
    script = {"obligation_checker": checker(o1="raise SystemExit(3)\n")}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    record = next(r for r in artifacts(orch, "obligation_check") if r["obligation"] == "o1")
    assert record["status"] == "unknown" and record["problems"][0].startswith("first run ")
    assert len(record["runs"]) == 1


def test_program_for_a_judge_only_kind_is_not_run(run_dir):
    script = {"obligation_checker": checker(o1=report("violated", "x", {"a": 1}),
                                            kinds=("complexity", "learnability", "learnability"))}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    record = next(r for r in artifacts(orch, "obligation_check") if r["obligation"] == "o1")
    assert record["status"] == "not_checked" and record["runs"] == []
    assert record["problems"] == ["program ignored: kind learnability is left to the judges"]
    assert s3(only_card(orch))["verdict"] == "pass"


def test_max_obligation_checks_caps_the_programs_run(run_dir):
    cfg = orch_config(s3__max_obligation_checks=1)
    script = {"obligation_checker": checker(o0=report("discharged"), o1=report("discharged"))}
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script)
    run(orch)
    records = {r["obligation"]: r for r in artifacts(orch, "obligation_check")}
    assert records["o0"]["status"] == "discharged"
    assert records["o1"]["status"] == "skipped" and records["o1"]["problems"] == ["s3.max_obligation_checks reached"]


def test_switching_symbolic_checks_off_restores_the_old_stage(run_dir):
    orch, llm = make_orchestrator(run_dir, cfg=orch_config(s3__symbolic_checks=False))
    run(orch)
    assert "obligation_checker" not in llm.prompts()
    assert artifacts(orch, "obligation_check") == []
    for call in llm.calls:
        if call["prompt"].startswith("feasibility_judge"):
            assert "EXECUTED OBLIGATION CHECKS" not in call["user"]
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "pass" and verdict["reasons"][0] == "gate rule: judges"
    assert not any(r.startswith("obligation check") for r in verdict["reasons"])


# ----- split votes ------------------------------------------------------------------------------


def test_check_settles_a_split_on_the_obligation_it_discharged(run_dir):
    script = {"obligation_checker": checker(o1=report("discharged", "state bounded by 2.0 for t <= 4096")),
              "feasibility_judge_a": judge("undetermined", [("o0", "discharged"), ("o1", "open"), ("o2", "discharged")]),
              "feasibility_judge_b": FEASIBLE}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "pass" and verdict["reasons"][0] == "gate rule: check_settles_split"
    assert llm.prompts().count("feasibility_judge_a") == 1
    (gate,) = artifacts(orch, "s3_gate")
    assert gate["contested"] == ["o1"]
    assert gate["votes"]["feasibility_judge_a"] == "undetermined"
    assert gate["effective_votes"] == {"feasibility_judge_a": "feasible", "feasibility_judge_b": "feasible"}
    assert "feasibility_judge_a: o1 open -> discharged (check)" in gate["notes"]


def test_without_the_check_the_same_split_stays_undetermined(run_dir):
    script = {"obligation_checker": checker(o1=None),
              "feasibility_judge_a": judge("undetermined", [("o0", "discharged"), ("o1", "open"), ("o2", "discharged")]),
              "feasibility_judge_b": FEASIBLE}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "undetermined" and verdict["reasons"][0] == "gate rule: judges"
    assert llm.prompts().count("feasibility_judge_a") == 2


def test_split_like_card_46dbb349_open_against_refuted_with_a_discharging_check(run_dir):
    """Judge A leaves stability open, judge B refutes it; the check discharges it. Both judges still leave
    learnability open, so the card ends undetermined, but the judges no longer disagree."""
    script = {"obligation_checker": checker(o1=report("discharged", "finite in float64 over the stated protocol")),
              "feasibility_judge_a": judge("undetermined", [("o0", "discharged"), ("o1", "open"), ("o2", "open")]),
              "feasibility_judge_b": judge("infeasible", [("o0", "discharged"), ("o1", "refuted"), ("o2", "open")],
                                           ["float32 overflow"])}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "undetermined"
    assert verdict["reasons"][0] == "gate rule: check_settles_split"
    assert "both judges undetermined after 2 round(s)" in verdict["reasons"]
    gate = artifacts(orch, "s3_gate")[-1]
    assert gate["effective_votes"] == {"feasibility_judge_a": "undetermined", "feasibility_judge_b": "undetermined"}
    judges = artifacts(orch, "judges")
    assert [j["gate"]["check_settles_split"] for j in judges] == [True, True]


def test_check_does_not_decide_a_split_on_an_obligation_it_did_not_decide(run_dir):
    script = {"obligation_checker": checker(o0=report("discharged"), o1=report("unknown", "sampled only")),
              "feasibility_judge_a": judge("undetermined", [("o0", "discharged"), ("o1", "open"), ("o2", "open")]),
              "feasibility_judge_b": judge("infeasible", [("o0", "discharged"), ("o1", "refuted"), ("o2", "open")],
                                           ["overflow"])}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "undetermined" and verdict["reasons"][0] == "gate rule: judges"
    assert artifacts(orch, "s3_gate")[-1]["contested"] == []


def test_all_checkable_discharged_is_recorded(run_dir):
    script = {"obligation_checker": checker(o0=report("discharged"), o1=report("discharged"))}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    (gate,) = artifacts(orch, "s3_gate")
    assert gate["all_checkable_discharged"] is True and gate["rule"] == "judges"


def test_disproof_rule_comes_first(run_dir):
    script = {"disprover": {"attacks": [{"target_claim": "capacity", "counterexample": "K > d pairs",
                                         "status": "succeeds", "check_code": program({"disproved": True})}]},
              "obligation_checker": checker(o1=report("violated", "x", {"a": 1}))}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    verdict = s3(only_card(orch))
    assert verdict["verdict"] == "infeasible_proved" and verdict["reasons"][0] == "gate rule: disproof"


# ----- units ------------------------------------------------------------------------------------


def result(stdout, status="ok"):
    return ExecResult("job-000000000000", status, "", stdout, "", 0.1)


def test_parse_report_contract():
    ok, problems = oc.parse_report({"status": "discharged", "evidence": "e", "counterexample": None})
    assert ok == {"status": "discharged", "evidence": "e", "counterexample": None} and problems == []
    assert oc.parse_report(None) == (None, ["no JSON object on stdout"])
    _, problems = oc.parse_report({"status": "proved", "evidence": ""})
    assert problems == ["status 'proved' is not one of discharged, violated, unknown", "evidence is missing or empty"]
    _, problems = oc.parse_report({"status": "violated", "evidence": "e", "counterexample": {}})
    assert problems == ["violated without a counterexample"]


def test_outcome_needs_an_agreeing_replay_to_decide():
    line = json.dumps({"status": "violated", "evidence": "e", "counterexample": {"n": 3}})
    same = oc.outcome(result(line), result(line))
    assert (same["status"], same["decided"], same["replayed"]) == ("violated", True, True)
    other = json.dumps({"status": "violated", "evidence": "e2", "counterexample": {"n": 4}})
    assert oc.outcome(result(line), result(other))["status"] == "unknown"
    evidence_only = json.dumps({"status": "violated", "evidence": "different words", "counterexample": {"n": 3}})
    assert oc.outcome(result(line), result(evidence_only))["decided"] is True
    assert oc.outcome(result(line), result("", status="timeout"))["problems"][0].startswith("replay timeout")
    unknown = oc.outcome(result(json.dumps({"status": "unknown", "evidence": "e"})), None)
    assert unknown["status"] == "unknown" and unknown["problems"] == []
    with pytest.raises(AssertionError):
        oc.outcome(result(line), None)


def test_derived_vote_and_split_rule_units():
    assert derived_vote(["discharged", "open"]) == "undetermined"
    assert derived_vote(["open", "refuted"]) == "infeasible"
    assert derived_vote(["discharged"]) == "feasible"
    checks = [{"obligation": "o1", "decided": True, "status": "discharged", "optional": False}]
    # judge B says infeasible but lists nothing refuted: its vote is not derived from its list and is kept
    judged = {"a": judge("undetermined", [("o1", "open")]), "b": judge("infeasible", [("o1", "discharged")])}
    settled = settle_split(judged, checks)
    assert settled["applied"] and settled["contested"] == ["o1"]
    assert settled["votes"] == {"a": "feasible", "b": "infeasible"}
    assert any("does not follow from its own obligation statuses" in n for n in settled["notes"])
    # agreeing votes are never touched
    same = {"a": judge("undetermined", [("o1", "open")]), "b": judge("undetermined", [("o1", "refuted"), ("o2", "open")])}
    assert settle_split(same, checks)["applied"] is False


def test_config_defaults_and_validation(tmp_path):
    cfg = load_run_config(SHIPPED)
    assert cfg["s3"]["symbolic_checks"] is True and cfg["s3"]["max_obligation_checks"] == 6
    assert cfg["s3"]["max_checks"] == 3
    text = SHIPPED.read_text().replace("symbolic_checks = true\n", "").replace("max_obligation_checks = 6\n", "")
    old = tmp_path / "old.toml"
    old.write_text(text)
    cfg = load_run_config(old)
    assert cfg["s3"]["symbolic_checks"] is True and cfg["s3"]["max_obligation_checks"] == 6
    with pytest.raises(ValueError, match="s3.symbolic_checks must be true or false"):
        load_run_config(SHIPPED, {"s3.symbolic_checks": "yes"})
    with pytest.raises(ValueError, match="s3.max_obligation_checks must be an integer >= 0"):
        load_run_config(SHIPPED, {"s3.max_obligation_checks": -1})


def test_prompt_file_states_the_contract():
    text = load_prompt("obligation_checker")
    assert text.startswith("# prompt: obligation_checker")
    for phrase in ('"status": "discharged" | "violated" | "unknown"', "counterexample is required when status is "
                   "violated", "run twice", "numerical_stability", "learnability", "Do not build a straw version"):
        assert phrase in text, phrase


# ----- report and plumb -------------------------------------------------------------------------


def test_dossier_shows_the_obligation_checks_and_the_gate_rule(run_dir, capsys):
    script = {"obligation_checker": checker(o1=report("discharged", "state bounded by 2.0")),
              "feasibility_judge_a": judge("undetermined", [("o0", "discharged"), ("o1", "open"), ("o2", "discharged")]),
              "feasibility_judge_b": FEASIBLE}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    assert inspect_run(run_dir) == 0
    card = only_card(orch)
    html = (run_dir / "report" / "dossiers" / f"{card['id']}.html").read_text()
    assert "Obligation checks (S3)" in html
    assert "S3 gate rule: check_settles_split" in html
    assert "state bounded by 2.0" in html and "discharged (replayed)" in html
    assert "contested: o1" in html and "Program for o1" in html


def test_dossier_of_a_run_without_checks_has_no_section(run_dir, capsys):
    orch, llm = make_orchestrator(run_dir, cfg=orch_config(budgets__max_cards=1), script={
        "salad_critic": {"ok": False, "fatal": [{"kind": "dimension_error", "where": "u", "problem": "dimensions do not match"}],
                         "fixable": [], "notes": "n"}})
    run(orch)
    assert inspect_run(run_dir) == 0
    card = only_card(orch)
    assert "Obligation checks" not in (run_dir / "report" / "dossiers" / f"{card['id']}.html").read_text()


def test_parse_stages():
    assert parse_stages("S3") == ("S3",)
    assert parse_stages("s4, S5,S6") == ("S4", "S5", "S6")
    for bad in ("S2", "S3,S5", "S5,S4", "", "S7"):
        with pytest.raises(PlumbError):
            parse_stages(bad)
