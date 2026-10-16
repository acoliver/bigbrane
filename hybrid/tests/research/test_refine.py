"""Stage R, the refinement dialogue (Phase 7, DISCOVERY-PIPELINE-PLAN.md section 7), with fake models.

The challenger asks, the proposer revises; the dialogue ends when the challenger asks nothing, when the idea has used
its rounds (budgets.idea.refinement_rounds, 3) or on a stall (the same questions at the same places), and writes the
definitions and premises list in the vocabulary of stage M's statement fixer."""
import asyncio
import copy
import json

import pytest

from hybrid.research import lineage
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.outcomes import OutcomeStore, check_kill_code
from hybrid.research.schemas import validate
from hybrid.research.stages import m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages import r_refine
from hybrid.tests.research.test_orchestrator_fakes import draft, make_orchestrator, orch_config

EXPLORER_RULES_MARK = "## General card rules (shared with every explorer)"


def run(orch):
    return asyncio.run(orch.run())


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


def calls(llm, prompt):
    return [c for c in llm.calls if c["prompt"] == prompt]


def config(**overrides):
    cfg = orch_config(**overrides)
    cfg["refine"]["enabled"] = True
    return cfg


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261013T070000Z"


def question(qid="Q1", target="undefined_object", where="mechanism.update[0]", text="What is phi?"):
    return {"id": qid, "target": target, "where": where, "question": text}


def asks(*rounds):
    """A challenger that asks rounds[i] in round i (the last entry repeats)."""
    def challenger(i, user):
        questions = rounds[min(i, len(rounds) - 1)]
        return {"questions": questions, "notes": ""}
    return challenger


def always_new(i, user):
    return {"questions": [question(where=f"mechanism.state[{i}]", text=f"What is object {i}?")], "notes": ""}


DEFINITION = {"id": "D1", "kind": "sampling", "text": "keys are drawn uniformly from the 8 phase slots"}
PREMISE = {"id": "A1", "text": "keys of distinct pairs never share a phase slot", "kind": "other", "does_work": True,
           "why_needed": "exact recall of every stored pair needs collision-free keys", "testable_by": "code",
           "measurement_plan": {"measure": "slot collisions", "data": "the toy key stream",
                                "procedure": "count pairs mapped to one slot", "accept_if": "zero collisions"},
           "citation": None}


def proposer(card_fn=None, definitions=(DEFINITION,), premises=(PREMISE,)):
    """A proposer that answers every question asked in its user message and returns the card from card_fn."""
    def answer(i, user):
        card = card_fn(i, user) if card_fn else draft(0, user=user)
        ids = sorted(set(part.split(" ", 1)[0] for part in user.split("- Q")[1:]))
        return {"card": card, "answers": [{"question_id": f"Q{qid}", "answer": "defined now", "change": "added"}
                                          for qid in ids],
                "definitions": [dict(d) for d in definitions], "premises": [copy.deepcopy(p) for p in premises]}
    return answer


def grown(extra):
    def card(i, user):
        out = draft(0, user=user)
        out["mechanism"]["state"] += [{"name": f"H{k}", "type": "real scalar", "shape": [], "meaning": f"a helper object number {k}"}
                                      for k in range(extra)]
        return out
    return card


def card_of(orch):
    return orch.store.get_card(orch.store.list_cards()[0])


# ----- the dialogue's end ---------------------------------------------------------------------------------------------


def test_the_dialogue_ends_after_three_rounds_without_a_fourth_challenger_call(run_dir):
    orch, llm = make_orchestrator(run_dir, cfg=config(), script={"refine_challenger": always_new,
                                                                "refine_proposer": proposer()})
    assert run(orch)["reason"] == "complete"
    assert len(calls(llm, "refine_challenger")) == 3 and len(calls(llm, "refine_proposer")) == 3
    [stop] = events(orch, "loop_stop")
    assert stop["kill_code"] == "budget:refinement" and stop["stage"] == "R" and not stop["terminal"]
    charges = [e for e in events(orch, "budget_charge") if e["loop"] == "refinement"]
    assert [c["used"] for c in charges] == [1, 2, 3] and all(c["cap"] == 3 for c in charges)
    card = card_of(orch)
    r = card["status"]["verdicts"]["R"]
    assert r["verdict"] == "pass" and r["reasons"][0].startswith("dialogue ended: budget:refinement after 3")
    assert any("not shown to the challenger again" in reason for reason in r["reasons"])
    assert card["status"]["stage"] == "S6"
    assert orch.order == ("S0", "S1", "R", "S2", "S3", "S4", "S5", "S6")
    # the proposer answers in its own prompt with the shared card rules, and the challenger sees the earlier rounds
    assert EXPLORER_RULES_MARK in calls(llm, "refine_proposer")[0]["system"]
    assert "Round 1:" in calls(llm, "refine_challenger")[1]["user"]
    assert "What is object 0?" in calls(llm, "refine_challenger")[2]["user"]


def test_the_dialogue_ends_on_a_stall_when_the_same_questions_come_back(run_dir):
    same = [question(), question("Q2", "uncosted_step", "claims.complexity.memory", "What does the store cost?")]
    reworded = [dict(q, question=q["question"] + " Please say.") for q in same]
    orch, llm = make_orchestrator(run_dir, cfg=config(), script={"refine_challenger": asks(same, reworded),
                                                                "refine_proposer": proposer()})
    run(orch)
    assert len(calls(llm, "refine_challenger")) == 2 and len(calls(llm, "refine_proposer")) == 1
    [stop] = events(orch, "loop_stop")
    assert stop["kill_code"] == "stall:refinement"
    r = card_of(orch)["status"]["verdicts"]["R"]
    assert r["verdict"] == "pass" and r["reasons"][0].startswith("dialogue ended: stall:refinement after 2")
    assert sum(reason.startswith("open question") for reason in r["reasons"]) == 2
    record = orch.ctx.latest_json_artifact(card_of(orch)["id"], "dialogue_assumptions")
    assert [q["id"] for q in record["open_questions"]] == ["Q1", "Q2"] and record["ended"] == "stall:refinement"


def test_the_dialogue_ends_early_when_the_challenger_asks_nothing(run_dir):
    orch, llm = make_orchestrator(run_dir, cfg=config(), script={"refine_challenger": asks([question()], []),
                                                                "refine_proposer": proposer()})
    run(orch)
    assert len(calls(llm, "refine_challenger")) == 2 and len(calls(llm, "refine_proposer")) == 1
    assert events(orch, "loop_stop") == []
    r = card_of(orch)["status"]["verdicts"]["R"]
    assert r["reasons"][0].startswith("dialogue ended: converged after 2 challenger round(s), 1 of 1")


def test_a_question_outside_the_five_targets_is_refused_by_the_schema():
    schema = r_refine.challenger_schema(5)
    assert schema["properties"]["questions"]["items"]["properties"]["target"]["enum"] == list(r_refine.QUESTION_TARGETS)
    assert schema["properties"]["questions"]["maxItems"] == 5
    with pytest.raises(Exception):
        validate_json({"questions": [question(target="style")], "notes": ""}, schema)
    with pytest.raises(Exception):
        validate_json({"questions": [question(f"Q{k}") for k in range(6)], "notes": ""}, schema)


def validate_json(value, schema):
    import jsonschema
    jsonschema.validate(value, schema)


# ----- output in the vocabulary of stage M -----------------------------------------------------------------------------


def test_the_dialogue_writes_definitions_and_premises_in_the_statement_fixer_vocabulary(run_dir):
    orch, llm = make_orchestrator(run_dir, cfg=config(), script={"refine_challenger": asks([question()], []),
                                                                "refine_proposer": proposer()})
    run(orch)
    card = card_of(orch)
    record = orch.ctx.latest_json_artifact(card["id"], "dialogue_assumptions")
    assert record["definitions"] == [dict(DEFINITION, added_by="dialogue")]
    [premise] = record["premises"]
    validate("assumption", premise)
    assert premise["added_by"] == "dialogue" and premise["status"] == "open" and premise["evidence"] is None
    # the proposer's schema uses the statement fixer's own item schemas
    fixer = mc.statement_fixer_schema(["claim"], 1)["properties"]
    schema = r_refine.proposer_schema(card)["properties"]
    assert schema["definitions"]["items"] == fixer["definitions"]["items"]
    # FIX-6: the fixer alone says where a premise is settled (subject); everything else is the fixer's item schema
    fixer_premise = copy.deepcopy(fixer["premises"]["items"])
    assert "subject" in fixer_premise["required"]
    fixer_premise["required"].remove("subject")
    del fixer_premise["properties"]["subject"]
    assert schema["premises"]["items"] == fixer_premise
    # stage M's statement fixer starts from the list
    user = m_math.fixer_user(orch.ctx, card["id"], card, [{"id": "capability.1", "text": "recall"}])
    assert "FROM THE REFINEMENT DIALOGUE" in user
    block = json.loads(user.split("FROM THE REFINEMENT DIALOGUE", 1)[1].split("\n", 1)[1])
    assert block["premises"][0]["id"] == "A1" and block["definitions"][0]["id"] == "D1"
    # the outcome record of a card that never reached M carries the dialogue's list
    from hybrid.research.outcome_records import RunView, premise_ledger
    premises, definitions = premise_ledger(RunView(run_dir), card["id"])
    assert [p["id"] for p in premises] == ["A1"] and [d["id"] for d in definitions] == ["D1"]


def test_a_premise_without_a_measurement_plan_is_recorded_as_a_problem(run_dir):
    bare = dict(PREMISE, measurement_plan=None)
    orch, _ = make_orchestrator(run_dir, cfg=config(), script={"refine_challenger": asks([question()], []),
                                                              "refine_proposer": proposer(premises=(bare,))})
    run(orch)
    [answer] = events(orch, "refine_answer")
    assert answer["problems"] == ["premise A1: testable_by code needs a measurement_plan"]


# ----- growth limit, toy example and S1 as the entry check -------------------------------------------------------------


def test_a_revision_that_grows_the_core_past_the_limit_is_rejected_and_the_round_still_counts(run_dir):
    orch, llm = make_orchestrator(run_dir, cfg=config(), script={"refine_challenger": asks([question()]),
                                                                "refine_proposer": proposer(grown(3))})
    run(orch)
    [start] = events(orch, "refine_start")
    assert (start["core_size"], start["core_limit"]) == (5, 6)
    [answer] = events(orch, "refine_answer")
    assert answer["accepted"] is False and answer["core_size"] == 8 and "past the limit 6" in answer["rejected_because"]
    assert not [e for e in events(orch, "card_revised") if e.get("stage") == "R"]
    assert len(card_of(orch)["mechanism"]["state"]) == 2
    # the unchanged card draws the same question again: a stall
    assert [s["kill_code"] for s in events(orch, "loop_stop")] == ["stall:refinement"]


def test_a_revision_within_the_limit_is_accepted(run_dir):
    orch, _ = make_orchestrator(run_dir, cfg=config(), script={"refine_challenger": asks([question()], []),
                                                              "refine_proposer": proposer(grown(1))})
    run(orch)
    [revised] = [e for e in events(orch, "card_revised") if e.get("stage") == "R"]
    assert revised["state_objects"] == 3 and len(card_of(orch)["mechanism"]["state"]) == 3


def test_a_revision_that_breaks_the_toy_example_fails_the_card_at_r(run_dir, tmp_path):
    def broken(i, user):
        out = draft(0, user=user)
        out["toy_example"]["expected"]["state_norm"] = 99.0
        return out
    cfg = config()
    cfg["outcomes"].update(enabled=True, path=str(tmp_path / "outcomes"))
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"refine_challenger": asks([question()], []),
                                                           "refine_proposer": proposer(broken)})
    run(orch)
    card = card_of(orch)
    r = card["status"]["verdicts"]["R"]
    assert r["verdict"] == "fail" and r["reasons"][0].startswith("gate: ill_formed:toy:")
    assert card["status"]["stage"] == "R" and not calls(llm, "reduction_critic")
    [record] = [s.record for s in OutcomeStore(tmp_path / "outcomes").current()]
    assert record["kill_codes"] == ["ill_formed:toy"] and record["terminal_stage"] == "R"
    assert [p["id"] for p in record["premises"]] == ["A1"]


def test_s1_is_the_entry_check_and_passes_fixable_findings_to_the_challenger(run_dir):
    critic = {"ok": False, "fatal": [], "notes": "",
              "fixable": [{"kind": "missing_definition", "where": "mechanism.update[1]", "problem": "phase() undefined"}]}
    orch, llm = make_orchestrator(run_dir, cfg=config(), script={"salad_critic": critic,
                                                                "refine_challenger": asks([]),
                                                                "refine_proposer": proposer()})
    run(orch)
    assert len(calls(llm, "salad_critic")) == 1 and not calls(llm, "explorer")[1:]   # no S1 revision
    s1 = card_of(orch)["status"]["verdicts"]["S1"]
    assert s1["verdict"] == "pass" and s1["reasons"][0].startswith("gate: entry check passed; 1 fixable")
    assert "phase() undefined" in calls(llm, "refine_challenger")[0]["user"]


def test_s1_entry_check_fails_a_toy_mismatch_without_revising(run_dir):
    def bad_toy(i, user):
        out = draft(i, user=user)
        out["toy_example"]["expected"]["state_norm"] = 99.0
        return out
    orch, llm = make_orchestrator(run_dir, cfg=config(), script={"explorer": bad_toy})
    run(orch)
    card = card_of(orch)
    assert card["status"]["stage"] == "S1" and card["status"]["verdicts"]["S1"]["verdict"] == "fail"
    assert len(calls(llm, "explorer")) == 1 and not calls(llm, "refine_challenger")


# ----- roles and families ---------------------------------------------------------------------------------------------


def test_a_card_a_gpt_model_drafted_gets_the_claude_first_challenger(run_dir):
    orch, llm = make_orchestrator(run_dir, cfg=config(), models={"explorer": "gpt-6.1-sol"},
                                  script={"refine_challenger": asks([question()], []), "refine_proposer": proposer()})
    run(orch)
    assert {c["role"] for c in calls(llm, "refine_challenger")} == {"refine_challenger_alt"}


def test_a_challenger_in_the_proposers_family_is_recorded_not_refused(run_dir):
    orch, llm = make_orchestrator(run_dir, cfg=config(), models={"explorer": "claude-opus", "refine_challenger": "opus"},
                                  script={"refine_challenger": asks([question()], []), "refine_proposer": proposer()})
    run(orch)
    assert all(r["family_overlap"] for r in events(orch, "refine_round"))
    assert any(reason.startswith("family overlap") for reason in card_of(orch)["status"]["verdicts"]["R"]["reasons"])


def test_a_renamed_copy_shares_the_idea_rounds(run_dir):
    """Two cards with one mechanism signature: the second gets only the rounds the idea has left."""
    cfg = config()
    cfg["budgets"]["max_cards"] = 2
    cfg["diversity"]["threshold"] = 1.01
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={
        "explorer": lambda i, user: draft(0, distinct=False, user=user) | {"title": f"Name {i}"},
        "refine_challenger": always_new, "refine_proposer": proposer()})
    run(orch)
    assert len(calls(llm, "refine_proposer")) == 3
    assert sorted(s["kill_code"] for s in events(orch, "loop_stop")) == ["budget:refinement", "budget:refinement"]


# ----- config and resume ----------------------------------------------------------------------------------------------


def test_config_defaults_validation_and_the_shipped_switches():
    old = load_run_config(ROOT / "hybrid" / "config" / "run3.toml")
    assert old["refine"] == {"enabled": False, "growth_limit": 0.25, "max_questions": 5}
    assert old["s2"]["family_exclusion"] is False and old["s2"]["difference_statement"] is False
    shipped = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    assert shipped["refine"]["enabled"] is True and shipped["s2"]["difference_statement"] is True
    assert shipped["s2"]["family_exclusion"] is True and shipped["outcomes"]["enforce_r3"] is True
    for dotted, bad, message in (("refine.enabled", "yes", "refine.enabled"),
                                 ("refine.max_questions", 6, "refine.max_questions"),
                                 ("refine.growth_limit", -0.1, "refine.growth_limit"),
                                 ("s2.difference_statement", 1, "s2.difference_statement")):
        with pytest.raises(ValueError, match=message):
            load_run_config(ROOT / "hybrid" / "config" / "run.toml", {dotted: bad})


def test_resume_mid_dialogue_continues_without_granting_rounds_twice(run_dir):
    script = {"refine_challenger": always_new, "refine_proposer": proposer()}
    first = config()
    # explorer, salad critic, challenger, proposer, challenger: the run stops before the second answer
    first["budgets"]["max_calls"] = 5
    orch, llm = make_orchestrator(run_dir, cfg=first, script=script)
    run(orch)
    card_id = orch.store.list_cards()[0]
    assert orch.store.get_card(card_id)["status"]["stage"] == "S1"
    assert len(calls(llm, "refine_proposer")) == 1
    resumed, llm2 = make_orchestrator(run_dir, cfg=config(), script=script, resume=True)
    assert run(resumed)["reason"] == "complete"
    assert resumed.store.get_card(card_id)["status"]["stage"] == "S6"
    assert len(events(resumed, "refine_start")) == 1
    # one answer before the stop, two after: the idea's three rounds, no more
    assert len(calls(llm2, "refine_proposer")) == 2
    assert lineage.BudgetBook(load_run_config(ROOT / "hybrid" / "config" / "run.toml"), resumed.ledger,
                              resumed.store, run_id=run_dir.name).used(card_id, "refinement") == 3


def test_a_run_keeps_the_refine_setting_it_started_with(run_dir):
    orch, _ = make_orchestrator(run_dir, cfg=orch_config(budgets__max_calls=1))
    run(orch)
    with pytest.raises(ValueError, match="refine.enabled = false"):
        make_orchestrator(run_dir, cfg=config(), resume=True)
    resumed, llm = make_orchestrator(run_dir, cfg=orch_config(), resume=True)
    run(resumed)
    assert not calls(llm, "refine_challenger") and "R" not in resumed.order


def test_every_kill_code_r_can_record_is_in_the_enumeration():
    for code in ("ill_formed:toy", "budget:refinement", "stall:refinement"):
        check_kill_code(code)
