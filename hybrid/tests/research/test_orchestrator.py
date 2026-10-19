"""Orchestrator and stage gates with a scripted LLM, a fake classifier, LocalExecutor and a fake bench."""
import asyncio
import json

import pytest

from hybrid.llm.errors import TransportError
from hybrid.research.inspect import inspect_run
from hybrid.research.ledger import Ledger
from hybrid.research.store import Store
from hybrid.research.stages.s2_prior_art import BROADER_SEARCH_NOTE
from hybrid.tests.research.critic_answers import KNOWN, NOT_KNOWN
from hybrid.tests.research.test_orchestrator_fakes import (TEMPLATE, FakeBench, analyst_supports_every_claim,
                                                           comparison_check, declaring, draft, make_orchestrator,
                                                           orch_config, program)


def run(orch):
    return asyncio.run(orch.run())


def only_card(orch):
    cards = orch.store.list_cards()
    assert len(cards) == 1, cards
    return orch.store.get_card(cards[0])


def verdicts(card):
    return {stage: v["verdict"] for stage, v in card["status"]["verdicts"].items()}


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


def stage_starts(orch, stage):
    return [d["card_id"] for d in events(orch, "stage_start") if d["stage"] == stage]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261010T000000Z"


# ----- full line --------------------------------------------------------------------------------

def test_happy_path_reaches_s6_and_inspect_renders(run_dir, capsys):
    # S6 passes only when every claim is supported (s6_verdict.ledger_verdict).
    orch, llm = make_orchestrator(run_dir, script={"analyst": analyst_supports_every_claim})
    summary = run(orch)
    assert summary["reason"] == "complete"
    card = only_card(orch)
    assert card["status"]["stage"] == "S6"
    assert verdicts(card) == {s: "pass" for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")}
    assert llm.prompts() == ["explorer", "salad_critic", "prior_art_query", "reduction_critic", "formalizer",
                             "disprover", "obligation_checker", "feasibility_judge_a", "feasibility_judge_b", "builder",
                             "falsifier_designer", "analyst"]
    # prior_art goes through the injected wrapper with allow_web=False
    assert all(not c["allow_web"] for c in llm.calls)
    # analyst sees the bench JSON verbatim
    analyst_user = [c["user"] for c in llm.calls if c["prompt"] == "analyst"][0]
    raw_artifact = [a for a in events(orch, "artifact") if a["kind"] == "bench_raw"][0]
    assert (run_dir / raw_artifact["path"]).read_text() in analyst_user
    # supported claim spawns a mutation request (no card slot left to use it)
    assert events(orch, "mutation_requested") == [{"card_id": card["id"]}]
    assert Ledger(run_dir).verify() == (True, None)
    assert inspect_run(run_dir) == 0
    assert (run_dir / "report" / "dossiers" / f"{card['id']}.html").is_file()
    out = capsys.readouterr().out
    assert "stage S6: 1" in out
    receipts = [e for e in events(orch, "literature_receipt")]
    assert receipts and receipts[0]["card_id"] == card["id"]


def test_explorers_alternate_roles_and_run_in_parallel(run_dir):
    cfg = orch_config(run__n_explorers=2, budgets__max_cards=2)
    script = {"salad_critic": {"ok": False, "fatal": [], "notes": "x undefined",
                               "fixable": [{"kind": "missing_definition", "where": "mechanism.update[0]",
                                            "problem": "x is used but never defined"}]}}
    cfg["s1"]["max_revisions"] = 0
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script)
    run(orch)
    roles = sorted(c["role"] for c in llm.calls if c["prompt"] == "explorer")
    assert roles == ["explorer", "explorer_alt"]
    created = events(orch, "card_created")
    assert len(created) == 2 and {c["explorer"] for c in created} == {0, 1}
    assert all(3 <= len(c["papers"]) <= 6 for c in created)


# ----- S0 ---------------------------------------------------------------------------------------

def test_duplicate_card_is_recorded_and_dropped(run_dir):
    cfg = orch_config(budgets__max_cards=2)
    cfg["diversity"]["threshold"] = 0.6
    cfg["workers"]["S1"] = 1
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"explorer": lambda i, user: draft(0, distinct=False, user=user)})
    run(orch)
    dups = events(orch, "duplicate")
    assert len(dups) == 1 and dups[0]["cosine"] >= 0.99
    dropped = orch.store.get_card(dups[0]["card_id"])
    assert verdicts(dropped) == {"S0": "fail"}
    assert "duplicate" in dropped["status"]["verdicts"]["S0"]["reasons"][0]
    assert dups[0]["card_id"] not in stage_starts(orch, "S1")


def test_explorer_transport_failure_is_recorded_and_run_completes(run_dir):
    orch, llm = make_orchestrator(run_dir, fail={"explorer": TransportError("down")})
    summary = run(orch)
    assert summary["reason"] == "complete"
    assert orch.store.list_cards() == []
    failed = events(orch, "explore_failed")
    assert len(failed) == 1 and "TransportError" in failed[0]["error"]


def test_references_to_unseen_papers_are_dropped(run_dir):
    def explorer(i, user):
        card = draft(i, user=user)
        card["references"] = [{"source_id": "not-a-given-paper", "why": "invented"}]
        return card
    orch, llm = make_orchestrator(run_dir, script={"explorer": explorer})
    run(orch)
    card = only_card(orch)
    assert card["references"] == []
    assert events(orch, "card_created")[0]["dropped_references"][0]["source_id"] == "not-a-given-paper"


def test_transplant_sources_are_grounded_and_parsimony_is_recorded(run_dir):
    def explorer(i, user):
        card = draft(i, user=user)
        card["source_transplant"]["source_ids"].append("invented-paper")
        card["extensions"] = [{"name": "decay", "description": "an optional decay on the running sum"}]
        return card
    orch, llm = make_orchestrator(run_dir, script={"explorer": explorer})
    run(orch)
    card = only_card(orch)
    assert "invented-paper" not in card["source_transplant"]["source_ids"]
    assert card["core_idea"].startswith("Bind each token")
    created = events(orch, "card_created")[0]
    assert created["dropped_transplant_sources"] == ["invented-paper"]
    assert created["source_transplant"] == "phasor superposition"
    assert (created["state_objects"], created["update_equations"], created["extensions"]) == (2, 2, 1)
    assert any("transplant source" in r for r in card["status"]["verdicts"]["S0"]["reasons"])


def test_transplant_citing_only_unseen_papers_fails_the_explore_task(run_dir):
    def explorer(i, user):
        card = draft(i, user=user)
        card["source_transplant"]["source_ids"] = ["invented-paper"]
        return card
    orch, llm = make_orchestrator(run_dir, script={"explorer": explorer})
    run(orch)
    assert orch.store.list_cards() == []
    assert "source_transplant cites no paper" in events(orch, "explore_failed")[0]["error"]


# ----- S1 ---------------------------------------------------------------------------------------

BAD_CRITIC = {"ok": False, "fatal": [], "notes": "theta is undefined",
              "fixable": [{"kind": "missing_definition", "where": "mechanism.update[0]",
                           "problem": "theta is used in the update but never defined"}]}
FATAL_CRITIC = {"ok": False, "notes": "the update cannot type-check",
                "fatal": [{"kind": "dimension_error", "where": "mechanism.update[0]",
                           "problem": "W (d x d) multiplies x_t (V) and no projection exists"}],
                "fixable": [{"kind": "notation_clash", "where": "mechanism.state",
                             "problem": "L names both the loss and the code length"}]}
OK_CRITIC = {"ok": True, "fatal": [], "fixable": [], "notes": "ok"}
# Critic passes whose findings differ from round to round, so the revision loop runs to its cap instead of ending as a
# stall (Phase 8: a revision that leaves the same findings at the same places ends the loop). The last one is
# BAD_CRITIC, so the residual findings are BAD_CRITIC's.
BAD_CRITIC_B = {"ok": False, "fatal": [], "notes": "phi is undefined",
                "fixable": [{"kind": "missing_definition", "where": "mechanism.readout[0]",
                             "problem": "phi is used in the readout but never defined"}]}
BAD_CRITIC_C = {"ok": False, "fatal": [], "notes": "the step is vague",
                "fixable": [{"kind": "underspecified_step", "where": "mechanism.learning",
                             "problem": "the learning step does not say how k is chosen"}]}
CHANGING_BAD = [BAD_CRITIC_B, BAD_CRITIC_C, BAD_CRITIC]


def test_s1_revise_then_pass(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"salad_critic": [BAD_CRITIC, OK_CRITIC]})
    run(orch)
    card = only_card(orch)
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "pass"]
    assert s1[0]["reasons"][0] == "revision 1 of 2"
    assert "fixable missing_definition (mechanism.update[0]): theta is used in the update but never defined" \
        in s1[0]["reasons"]
    revise_user = [c["user"] for c in llm.calls if c["prompt"] == "explorer"][1]
    assert "OPERATOR: revise" in revise_user and "theta is undefined" in revise_user
    assert "Fix by tightening" in revise_user
    assert card["status"]["stage"] == "S6"
    revised = events(orch, "card_revised")
    assert len(revised) == 1 and revised[0]["state_objects"] == 2


def test_s1_fixable_findings_get_two_revisions_then_fail_when_residual_pass_is_off(run_dir):
    cfg = orch_config()
    cfg["s1"]["pass_with_residual_fixable"] = False
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"salad_critic": CHANGING_BAD})
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S1"] == "fail" and card["status"]["stage"] == "S1"
    assert llm.prompts().count("salad_critic") == 3
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "revise", "fail"]
    assert s1[1]["reasons"][0] == "revision 2 of 2"
    assert s1[2]["reasons"][0] == "gate: fixable findings remain after 2 revision(s)"
    assert len(events(orch, "card_revised")) == 2
    assert stage_starts(orch, "S2") == []


RESIDUAL_FINDING = "theta is used in the update but never defined"
OPEN_ISSUES_LABEL = "known open issues from S1"


def test_s1_residual_fixable_findings_pass_after_the_last_revision(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"salad_critic": CHANGING_BAD})
    run(orch)
    card = only_card(orch)
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "revise", "pass"]
    assert llm.prompts().count("salad_critic") == 3 and len(events(orch, "card_revised")) == 2
    reasons = s1[2]["reasons"]
    assert f"residual: missing_definition (mechanism.update[0]): {RESIDUAL_FINDING}" in reasons
    assert [r for r in reasons if r.startswith("residual:")] == reasons[2:3]
    assert reasons[1] == "loop s1_revision ended: budget:s1_revision: 2 of 2 revision(s) used"
    assert any("toy example reproduced" in r for r in reasons)
    assert card["status"]["stage"] == "S6" and verdicts(card)["S1"] == "pass"
    artifacts = [a for a in events(orch, "artifact") if a["kind"] == "s1_residual"]
    assert len(artifacts) == 1 and artifacts[0]["card_id"] == card["id"]
    stored = json.loads((run_dir / artifacts[0]["path"]).read_text())
    assert stored["findings"] == BAD_CRITIC["fixable"] and stored["revisions_used"] == 2
    assert stored["card_id"] == card["id"]


def test_s3_formalizer_and_disprover_get_the_residual_findings_in_the_user_message(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"salad_critic": BAD_CRITIC})
    run(orch)
    for name in ("formalizer", "disprover"):
        user = [c["user"] for c in llm.calls if c["prompt"] == name][0]
        assert OPEN_ISSUES_LABEL in user and RESIDUAL_FINDING in user
    assert user.index("PROOF OBLIGATIONS") < user.index(OPEN_ISSUES_LABEL)
    for call in llm.calls:
        if call["prompt"].startswith("feasibility_judge"):
            assert OPEN_ISSUES_LABEL not in call["user"]


def test_s3_user_messages_have_no_open_issues_for_a_clean_s1_pass(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"salad_critic": [BAD_CRITIC, OK_CRITIC]})
    run(orch)
    assert [a for a in events(orch, "artifact") if a["kind"] == "s1_residual"] == []
    for call in llm.calls:
        assert OPEN_ISSUES_LABEL not in call["user"]


def test_open_issues_label_comes_from_the_stage_not_from_a_prompt_file():
    from hybrid.research.stages.common import PROMPT_DIR
    for path in PROMPT_DIR.glob("*.md"):
        assert OPEN_ISSUES_LABEL not in path.read_text(encoding="utf-8"), path


def test_s1_with_no_revisions_allowed_passes_with_residual_at_once(run_dir):
    cfg = orch_config()
    cfg["s1"]["max_revisions"] = 0
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"salad_critic": BAD_CRITIC})
    run(orch)
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["pass"]
    assert llm.prompts().count("explorer") == 1 and events(orch, "card_revised") == []


def test_s1_fatal_finding_after_the_last_revision_still_fails(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"salad_critic": [BAD_CRITIC_B, BAD_CRITIC_C, FATAL_CRITIC]})
    run(orch)
    card = only_card(orch)
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "revise", "fail"]
    assert s1[2]["reasons"][0] == "gate: fatal salad-critic finding"
    assert not any(r.startswith("residual:") for r in s1[2]["reasons"])
    assert [a for a in events(orch, "artifact") if a["kind"] == "s1_residual"] == []
    assert card["status"]["stage"] == "S1" and stage_starts(orch, "S2") == []


def test_s1_toy_mismatch_after_the_last_revision_still_fails(run_dir):
    def explorer(i, user):
        card = draft(i, user=user)
        card["toy_example"]["expected"]["state_norm"] = 99.0
        return card
    orch, llm = make_orchestrator(run_dir, script={"explorer": explorer, "salad_critic": CHANGING_BAD})
    run(orch)
    card = only_card(orch)
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "revise", "fail"]
    assert s1[2]["reasons"][0] == "gate: fixable findings remain after 2 revision(s)"
    assert any("state_norm" in r for r in s1[2]["reasons"])
    assert [a for a in events(orch, "artifact") if a["kind"] == "s1_residual"] == []
    assert card["status"]["stage"] == "S1" and stage_starts(orch, "S2") == []


def test_s1_two_revisions_then_pass(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"salad_critic": [BAD_CRITIC_B, BAD_CRITIC, OK_CRITIC]})
    run(orch)
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "revise", "pass"]
    assert only_card(orch)["status"]["stage"] == "S6"


def test_s1_fatal_finding_fails_while_revisions_remain(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"salad_critic": FATAL_CRITIC})
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S1"] == "fail"
    assert llm.prompts().count("salad_critic") == 1 and llm.prompts().count("explorer") == 1
    reasons = card["status"]["verdicts"]["S1"]["reasons"]
    assert reasons[0] == "gate: fatal salad-critic finding"
    assert any(r.startswith("fatal dimension_error (mechanism.update[0])") for r in reasons)
    assert any(r.startswith("fixable notation_clash") for r in reasons)
    assert events(orch, "card_revised") == []


def test_s1_fatal_after_a_revision_fails(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"salad_critic": [BAD_CRITIC, FATAL_CRITIC]})
    run(orch)
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "fail"]


def test_s1_toy_mismatch_sends_back_then_revision_fixes_it(run_dir):
    def explorer(i, user):
        card = draft(i, user=user)
        if i == 0:
            card["toy_example"]["expected"]["state_norm"] = 99.0
        return card
    orch, llm = make_orchestrator(run_dir, script={"explorer": explorer})
    run(orch)
    s1 = [d for d in events(orch, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "pass"]
    assert any("state_norm" in r for r in s1[0]["reasons"])
    assert any(call.startswith("job-") for call in s1[0]["by_calls"])
    executor_artifacts = [a for a in events(orch, "artifact") if a["kind"] == "executor"]
    assert executor_artifacts


def test_s1_toy_crash_fails_after_revision(run_dir):
    def explorer(i, user):
        card = draft(i, user=user)
        card["toy_example"]["python"] = "raise SystemExit('boom: toy example crashed')"
        return card
    orch, llm = make_orchestrator(run_dir, script={"explorer": explorer})
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S1"] == "fail"
    assert any("did not run" in r for r in card["status"]["verdicts"]["S1"]["reasons"])


# ----- S2 ---------------------------------------------------------------------------------------

def reduction(reduces_to="ssm", confidence=0.9, difference="", radical=False, textbook=None):
    # textbook_transplant gates (decision R2C), so a card meant to pass S2 changes the construction by default
    textbook = (not radical) if textbook is None else textbook
    return declaring({"reduces_to": reduces_to, "mapping": "S <- h, theta <- A", "confidence": confidence,
            "difference_if_any": difference, **(NOT_KNOWN if radical else KNOWN), "why": "it is a diagonal recurrence",
            "textbook_transplant": textbook, "textbook_why": "a diagonal linear recurrence as the textbook states it"})


def test_s2_known_disproven_fails(run_dir):
    orch, llm = make_orchestrator(run_dir, labels="known_disproven")
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "fail" and card["status"]["stage"] == "S2"
    assert any("known_disproven" in r for r in card["status"]["verdicts"]["S2"]["reasons"])


def test_s2_textbook_transplant_gates_and_is_recorded(run_dir):
    script = {"reduction_critic": reduction(reduces_to="none", radical=True, textbook=True) | {"mapping": None}}
    orch, llm = make_orchestrator(run_dir, script=script)
    orch.routing = {"table": "llxprt", "source": "file", "gateways": []}
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "fail" and card["status"]["stage"] == "S2"
    reasons = card["status"]["verdicts"]["S2"]["reasons"]
    assert any(r.startswith("textbook transplant: yes: ") for r in reasons)
    assert any(r.startswith("gate: not radical enough: textbook transplant") for r in reasons)
    artifact = [a for a in events(orch, "artifact") if a["kind"] == "reduction_critic"][0]
    assert json.loads((run_dir / artifact["path"]).read_text())["textbook_transplant"] is True
    assert events(orch, "run_start")[0]["routing"] == {"table": "llxprt", "source": "file", "gateways": []}


def test_s2_textbook_transplant_is_only_recorded_when_the_radical_gates_are_off(run_dir):
    cfg = orch_config()
    cfg["s2"]["require_radical"] = False
    script = {"reduction_critic": reduction(reduces_to="none", radical=True, textbook=True) | {"mapping": None}}
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script)
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "pass" and card["status"]["stage"] == "S6"
    assert any(r.startswith("textbook transplant: yes: ") for r in card["status"]["verdicts"]["S2"]["reasons"])


def test_inspect_counts_textbook_transplants_and_lenses_per_stage(run_dir, capsys):
    cfg = orch_config(budgets__max_cards=3, run__n_explorers=1)
    critics = [reduction(reduces_to="none", radical=True, textbook=t) | {"mapping": None} for t in (True, False)]
    salad = [OK_CRITIC, OK_CRITIC, FATAL_CRITIC]
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"reduction_critic": critics, "salad_critic": salad})
    run(orch)
    from hybrid.research.inspect import creativity_metrics
    metrics = creativity_metrics(run_dir, list(orch.ledger.iter_events()), orch.store)
    textbook = metrics["textbook"]
    assert (len(textbook["transplant"]), len(textbook["modified"]), len(textbook["not_assessed"])) == (1, 1, 1)
    by_model = metrics["textbook_by_model"]
    assert set(by_model) <= {"fake-explorer", "fake-explorer_alt"}             # explorers alternate roles
    assert sum(c.get("transplant", 0) for c in by_model.values()) == 1
    assert sum(c.get("modified", 0) for c in by_model.values()) == 1
    assert sum(metrics["lens_stage"]["S1"].values()) == 3 and sum(metrics["lens_stage"]["S2"].values()) == 2
    assert metrics["core_size"]["state_objects"] == {"mean": 2.0, "max": 2}
    assert inspect_run(run_dir) == 0
    out = capsys.readouterr().out
    assert "textbook transplants 1, modified 1 (of 2 assessed); not assessed 1" in out
    assert "Lenses per stage" in out
    assert "Creativity" in (run_dir / "report" / "index.html").read_text()


def test_s2_verified_reduction_gives_reduces_to_known(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"reduction_critic": reduction()})
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "reduces_to_known"
    assert "reduction_check" in llm.prompts()
    by_calls = card["status"]["verdicts"]["S2"]["by_calls"]
    assert any(c.startswith("job-") for c in by_calls)
    assert stage_starts(orch, "S3") == []


def test_s2_reduction_refuted_by_numeric_check_passes(run_dir):
    script = {"reduction_critic": reduction(radical=True),
              "reduction_check": comparison_check(0.7)}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "pass"
    assert any("rejects the claimed reduction" in r for r in card["status"]["verdicts"]["S2"]["reasons"])


def test_s2_reduction_with_learning_rule_difference_passes(run_dir):
    script = {"reduction_critic": reduction(difference="LEARNING_RULE: structure grows by MDL", radical=True)}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    assert verdicts(only_card(orch))["S2"] == "pass"


def test_s2_reduction_without_check_uses_confidence(run_dir):
    script = {"reduction_critic": reduction(confidence=0.5, radical=True),
              "reduction_check": {"check_code": None, "explanation": "underspecified"}}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    assert verdicts(only_card(orch))["S2"] == "pass"


def test_s2_prior_art_tested_without_new_element_fails(run_dir):
    orch, llm = make_orchestrator(run_dir, labels="prior_art_tested")
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "fail"
    assert "prior_art_delta" in llm.prompts()


def test_s2_prior_art_tested_with_new_element_passes(run_dir):
    script = {"prior_art_delta": {"adds_untested_element": True, "element": "an MDL growth rule", "why": "new"}}
    orch, llm = make_orchestrator(run_dir, labels="prior_art_tested", script=script)
    run(orch)
    assert verdicts(only_card(orch))["S2"] == "pass"


def test_s2_undertested_passes_and_not_radical_fails(run_dir, tmp_path):
    orch, _ = make_orchestrator(run_dir, labels="prior_art_undertested")
    run(orch)
    assert verdicts(only_card(orch))["S2"] == "pass"
    script = {"reduction_critic": reduction(reduces_to="none", radical=False) | {"mapping": None}}
    orch2, _ = make_orchestrator(tmp_path / "run-20261010T000009Z", script=script)
    run(orch2)
    card = only_card(orch2)
    assert verdicts(card)["S2"] == "fail"
    assert any("not radical" in r for r in card["status"]["verdicts"]["S2"]["reasons"])


def prior_art_artifact(orch, run_dir):
    [artifact] = [a for a in events(orch, "artifact") if a["kind"] == "prior_art"]
    return json.loads((run_dir / artifact["path"]).read_text())


def test_s2_novel_label_with_no_papers_is_retried_then_undetermined(run_dir):
    orch, llm = make_orchestrator(run_dir, labels="novel", found=False)
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "undetermined" and card["status"]["stage"] == "S2"
    assert card["status"]["verdicts"]["S2"]["reasons"][-1] == "novelty unassessed: literature search returned no papers"
    assert stage_starts(orch, "S3") == []
    assert prior_art_artifact(orch, run_dir)["label"] == "novelty_unassessed"
    queries = [c["user"] for c in llm.calls if c["prompt"] == "prior_art_query"]
    assert len(queries) == 2
    assert BROADER_SEARCH_NOTE not in queries[0] and BROADER_SEARCH_NOTE in queries[1]
    [event] = events(orch, "novelty_unassessed")
    assert event["card_id"] == card["id"] and event["retry_papers"] == 0 and event["still_unassessed"] is True


def test_s2_retry_that_finds_papers_is_judged_on_them(run_dir):
    orch, llm = make_orchestrator(run_dir, labels=["novel", "prior_art_undertested"], found=[False, True])
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "pass"
    assert prior_art_artifact(orch, run_dir)["label"] == "prior_art_undertested"
    [event] = events(orch, "novelty_unassessed")
    assert event["retry_papers"] == 1 and event["still_unassessed"] is False


def test_s2_empty_search_counts_as_novel_when_configured(run_dir):
    cfg = orch_config(s2__empty_search_counts_as_novel=True)
    orch, llm = make_orchestrator(run_dir, cfg=cfg, found=False)
    run(orch)
    assert verdicts(only_card(orch))["S2"] == "pass"
    assert llm.prompts().count("prior_art_query") == 1
    assert events(orch, "novelty_unassessed") == []


def test_s2_empty_search_with_verified_reduction_gives_reduces_to_known(run_dir):
    orch, llm = make_orchestrator(run_dir, found=False, script={"reduction_critic": reduction()})
    run(orch)
    assert verdicts(only_card(orch))["S2"] == "reduces_to_known"
    assert llm.prompts().count("prior_art_query") == 2


def test_s2_empty_search_with_learning_rule_difference_is_undetermined(run_dir):
    script = {"reduction_critic": reduction(difference="LEARNING_RULE: structure grows by MDL", radical=True)}
    orch, llm = make_orchestrator(run_dir, found=False, script=script)
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S2"] == "undetermined"
    assert any("learning rule" in r for r in card["status"]["verdicts"]["S2"]["reasons"])


# ----- S3 ---------------------------------------------------------------------------------------

def judge(verdict, fatal=(), status="discharged"):
    return {"verdict": verdict, "obligations": [{"name": "o1", "status": status, "argument": "arg"}],
            "fatal_issues": list(fatal), "confidence": 0.6}


def test_s3_confirmed_disproof_gives_infeasible_proved(run_dir):
    script = {"disprover": {"attacks": [{"target_claim": "capacity", "counterexample": "K > d pairs",
                                         "status": "succeeds",
                                         "check_code": program({"disproved": True, "detail": "collision"})}]}}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S3"] == "infeasible_proved"
    assert any("confirmed disproof" in r for r in card["status"]["verdicts"]["S3"]["reasons"])
    assert stage_starts(orch, "S4") == []


def test_s3_attack_refuted_by_its_check_does_not_block(run_dir):
    script = {"disprover": {"attacks": [{"target_claim": "capacity", "counterexample": "K > d pairs",
                                         "status": "succeeds",
                                         "check_code": program({"disproved": False, "detail": "held"})}]}}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    assert verdicts(only_card(orch))["S3"] == "pass"


def test_s3_split_vote_gets_one_more_round_then_undetermined(run_dir):
    script = {"feasibility_judge_b": judge("undetermined", status="open")}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S3"] == "undetermined"
    assert llm.prompts().count("feasibility_judge_a") == 2
    assert llm.prompts().count("feasibility_judge_b") == 2
    assert stage_starts(orch, "S4") == []


def test_s3_split_then_agreement_passes(run_dir):
    script = {"feasibility_judge_b": [judge("undetermined", status="open"), judge("feasible")]}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S3"] == "pass"
    second_round = [c["user"] for c in llm.calls if c["prompt"] == "feasibility_judge_a"][1]
    assert "judging round 2" in second_round


def test_s3_judges_do_not_see_each_other(run_dir):
    script = {"feasibility_judge_b": [judge("undetermined", status="open"), judge("feasible")]}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    for call in llm.calls:
        if call["prompt"].startswith("feasibility_judge"):
            assert "fake-feasibility_judge" not in call["user"]
            assert '"verdict"' not in call["user"]


def test_s3_both_infeasible_with_surviving_attack_is_infeasible_proved(run_dir):
    script = {"disprover": {"attacks": [{"target_claim": "capacity", "counterexample": "bits", "status": "succeeds"}]},
              "feasibility_judge_a": judge("infeasible", ["capacity"], "refuted"),
              "feasibility_judge_b": judge("infeasible", ["capacity"], "refuted")}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    assert verdicts(only_card(orch))["S3"] == "infeasible_proved"


def test_s3_both_infeasible_without_refutation_fails(run_dir):
    script = {"feasibility_judge_a": judge("infeasible", ["vague"], "open"),
              "feasibility_judge_b": judge("infeasible", ["vague"], "open")}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    assert verdicts(only_card(orch))["S3"] == "fail"


# ----- S4 and the S3 invariant -----------------------------------------------------------------

def test_s4_repairs_forbidden_import_then_passes(run_dir):
    bad = "import os\n" + TEMPLATE
    orch, llm = make_orchestrator(run_dir, script={"builder": [{"candidate_py": bad, "notes": ""},
                                                               {"candidate_py": TEMPLATE, "notes": ""}]})
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S4"] == "pass"
    second = [c["user"] for c in llm.calls if c["prompt"] == "builder"][1]
    assert "forbidden import os" in second
    built = events(orch, "candidate_built")[0]
    assert built["describe"]["params"] > 0


def test_s4_fails_after_two_repairs_on_acausal_model(run_dir):
    acausal = TEMPLATE.replace(
        "last = np.array([p[-1] for p in prefixes], dtype=np.int64)",
        "last = np.array([prefixes[0][-1] for p in prefixes], dtype=np.int64)")
    assert acausal != TEMPLATE
    orch, llm = make_orchestrator(run_dir, script={"builder": {"candidate_py": acausal, "notes": ""}})
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S4"] == "fail"
    assert llm.prompts().count("builder") == 3
    assert any("causality" in r for r in card["status"]["verdicts"]["S4"]["reasons"])
    assert stage_starts(orch, "S5") == []


def test_no_card_reaches_s4_without_s3_pass(run_dir):
    cfg = orch_config(budgets__max_cards=3)
    script = {"feasibility_judge_b": [judge("infeasible", ["x"], "refuted"), judge("feasible"),
                                      judge("undetermined", status="open")]}
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script)
    run(orch)
    for card_id in stage_starts(orch, "S4"):
        assert orch.store.get_card(card_id)["status"]["verdicts"]["S3"]["verdict"] == "pass"
    for card_id in orch.store.list_cards():
        card = orch.store.get_card(card_id)
        if "S4" in card["status"]["verdicts"]:
            assert card["status"]["verdicts"]["S3"]["verdict"] == "pass"


def test_enqueue_refuses_s4_without_s3_pass(run_dir):
    orch, llm = make_orchestrator(run_dir, script={"feasibility_judge_b": judge("undetermined", status="open")})
    run(orch)
    card = only_card(orch)
    orch.queues = {stage: asyncio.Queue() for stage in orch.queues}
    with pytest.raises(AssertionError, match="S3"):
        orch._enqueue("S4", card["id"])


# ----- S5 and S6 --------------------------------------------------------------------------------

def test_s5_candidate_failure_ends_line(run_dir):
    orch, llm = make_orchestrator(run_dir, bench=FakeBench(status="job_failed"))
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S5"] == "fail"
    assert "analyst" not in llm.prompts()


def test_s6_refuted_claims_spawn_no_mutation(run_dir):
    script = {"analyst": {"claims": [{"claim_id": "capability.1", "claim": "c", "status": "refuted",
                                      "evidence": "kl 2.0 vs 0.1", "falsifier_tasks": []}]}}
    orch, llm = make_orchestrator(run_dir, script=script, cfg=orch_config(s5__falsifiers=False))
    run(orch)
    assert events(orch, "mutation_requested") == []


def test_mutation_request_becomes_mutate_card(run_dir):
    from hybrid.research.stages.s0_explore import explore
    orch, llm = make_orchestrator(run_dir)
    run(orch)
    parent = only_card(orch)["id"]
    orch.mutations = asyncio.Queue()
    orch.mutations.put_nowait(parent)
    orch.budget.max_cards = 2                                      # the run used its one slot; a mutation needs one
    task = orch._next_task(0, 1)
    assert (task.op, task.parents, task.role) == ("mutate", [parent], "explorer_alt")
    child = asyncio.run(explore(orch.ctx, task))
    assert orch.store.get_card(child)["parent_ids"] == [parent]
    user = [c["user"] for c in llm.calls if c["prompt"] == "explorer"][-1]
    assert "OPERATOR: mutate" in user and f"PARENT CARD A ({parent})" in user and "VERDICTS SO FAR" in user


def test_cross_operator_needs_two_eligible_parents(run_dir):
    cfg = orch_config(budgets__max_cards=2)
    cfg["explore"]["p_cross"] = 1.0
    orch, llm = make_orchestrator(run_dir, cfg=cfg)
    run(orch)
    orch.mutations = asyncio.Queue()
    task = orch._next_task(0, 2)
    assert task.op == "cross" and sorted(task.parents) == sorted(orch.store.list_cards())


# ----- lens order seed --------------------------------------------------------------------------

def test_lens_order_is_derived_from_the_run_id_unless_run_seed_is_set(tmp_path):
    from hybrid.research.orchestrator import derive_lens_seed
    assert derive_lens_seed("run-20261010T000000Z") == derive_lens_seed("run-20261010T000000Z")
    assert derive_lens_seed("run-20261010T000000Z") != derive_lens_seed("run-20261010T000001Z")

    ids = [f"run-2026101{n}T000000Z" for n in range(6)]
    perms = [make_orchestrator(tmp_path / run_id)[0].lens_perm for run_id in ids]
    assert len({tuple(perm) for perm in perms}) > 1
    assert sorted(perms[0]) == list(range(len(perms[0])))
    same = make_orchestrator(tmp_path / "other" / ids[0])[0]
    assert same.lens_perm == perms[0]

    cfg_a, cfg_b = orch_config(), orch_config()
    cfg_a["run"]["seed"] = cfg_b["run"]["seed"] = 0
    fixed = [make_orchestrator(tmp_path / "fixed" / run_id, cfg=cfg)[0]
             for run_id, cfg in ((ids[0], cfg_a), (ids[1], cfg_b))]
    assert fixed[0].lens_perm == fixed[1].lens_perm and fixed[0].lens_seed == 0


def test_run_start_records_the_lens_seed_and_resume_keeps_it(run_dir):
    orch, _ = make_orchestrator(run_dir, cfg=orch_config(budgets__max_calls=1))
    run(orch)
    start = events(orch, "run_start")[0]
    assert start["lens_seed"] == orch.lens_seed
    # a resumed run reads the seed back from its ledger
    orch2, _ = make_orchestrator(run_dir, resume=True)
    assert orch2.lens_seed == orch.lens_seed and orch2.lens_perm == orch.lens_perm


def test_resume_of_a_run_without_a_recorded_lens_seed_uses_seed_zero(run_dir):
    from hybrid.research.orchestrator import LEGACY_LENS_SEED
    run_dir.mkdir(parents=True)
    Ledger(run_dir).append("run_start", "orchestrator", {"run_id": run_dir.name, "resume": False})
    orch, _ = make_orchestrator(run_dir, resume=True)
    assert LEGACY_LENS_SEED == 0 and orch.lens_seed == 0
    cfg = orch_config()
    cfg["run"]["seed"] = 5
    assert make_orchestrator(run_dir, cfg=cfg, resume=True)[0].lens_seed == 5


# ----- budgets and resume -----------------------------------------------------------------------

def test_max_calls_stops_cleanly(run_dir):
    cfg = orch_config(budgets__max_calls=3)
    orch, llm = make_orchestrator(run_dir, cfg=cfg)
    summary = run(orch)
    assert summary["reason"].startswith("max_calls")
    assert len(llm.calls) == 3
    assert events(orch, "run_end")[0]["reason"].startswith("max_calls")
    assert Ledger(run_dir).verify() == (True, None)


def test_max_cards_limits_s0(run_dir):
    cfg = orch_config(budgets__max_cards=3, run__n_explorers=2)
    cfg["s1"]["max_revisions"] = 0
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"salad_critic": BAD_CRITIC})
    summary = run(orch)
    assert summary["reason"] == "complete"
    assert len(orch.store.list_cards()) == 3
    assert llm.prompts().count("explorer") == 3


def test_max_wall_stops_cleanly(run_dir):
    cfg = orch_config(budgets__max_wall_s=0.5)

    async def slow(*args, **kwargs):
        await asyncio.sleep(5)

    orch, llm = make_orchestrator(run_dir, cfg=cfg)
    llm.complete = slow
    summary = run(orch)
    assert summary["reason"].startswith("max_wall_s")


def test_resume_continues_from_ledger_and_cards(run_dir):
    cfg = orch_config(budgets__max_calls=3)   # explorer, critic, prior_art; the reduction critic is refused
    orch, llm = make_orchestrator(run_dir, cfg=cfg)
    first = run(orch)
    assert first["reason"].startswith("max_calls")
    card = only_card(orch)
    assert card["status"]["stage"] == "S1" and verdicts(card)["S1"] == "pass"

    cfg2 = orch_config(budgets__max_calls=50)
    orch2, llm2 = make_orchestrator(run_dir, cfg=cfg2, resume=True)
    second = run(orch2)
    assert second["reason"] == "complete"
    assert second["requeued_at_start"] == {"S2": [card["id"]]}
    assert "explorer" not in llm2.prompts()          # card budget already used
    assert orch2.budget.calls == 3 + len(llm2.calls)
    final = only_card(orch2)
    assert final["status"]["stage"] == "S6"
    assert Ledger(run_dir).verify() == (True, None)
    assert inspect_run(run_dir) == 0


def test_resume_requeues_s1_revise(run_dir):
    cfg = orch_config(budgets__max_calls=2)   # explorer, critic (revise) then the revision call is refused
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"salad_critic": BAD_CRITIC})
    run(orch)
    card = only_card(orch)
    assert verdicts(card)["S1"] == "revise"
    orch2, llm2 = make_orchestrator(run_dir, cfg=orch_config(), resume=True,
                                    script={"salad_critic": BAD_CRITIC})
    summary = run(orch2)
    assert summary["requeued_at_start"] == {"S1": [card["id"]]}
    # one of the two revisions was already used: critic, the second revision, critic, then the residual pass
    assert llm2.prompts()[:3] == ["salad_critic", "explorer", "salad_critic"]
    assert verdicts(only_card(orch2))["S1"] == "pass"
    assert [d["verdict"] for d in events(orch2, "verdict") if d["stage"] == "S1"] == ["revise", "revise", "pass"]


def test_resume_of_a_run_started_under_the_old_gate_applies_the_new_gate(run_dir):
    # Old rules: fixable findings after the last revision fail the card. Stop the run right after the
    # second revise verdict, before its revision call, which is where a live run can be killed.
    old = orch_config(budgets__max_calls=4)   # explorer, critic, revision, critic; the second revision is refused
    old["s1"]["pass_with_residual_fixable"] = False
    orch, llm = make_orchestrator(run_dir, cfg=old, script={"salad_critic": [BAD_CRITIC_B, BAD_CRITIC_C]})
    assert run(orch)["reason"].startswith("max_calls")
    card = only_card(orch)
    assert verdicts(card)["S1"] == "revise" and card["status"]["stage"] == "S1"
    assert [d["verdict"] for d in events(orch, "verdict") if d["stage"] == "S1"] == ["revise", "revise"]

    orch2, llm2 = make_orchestrator(run_dir, cfg=orch_config(), resume=True, script={"salad_critic": BAD_CRITIC})
    summary = run(orch2)
    assert summary["requeued_at_start"] == {"S1": [card["id"]]}
    assert llm2.prompts()[0] == "salad_critic"
    assert llm2.prompts().count("salad_critic") == 1 and llm2.prompts().count("explorer") == 0
    final = only_card(orch2)
    assert verdicts(final)["S1"] == "pass" and final["status"]["stage"] == "S6"
    s1 = [d for d in events(orch2, "verdict") if d["stage"] == "S1"]
    assert [d["verdict"] for d in s1] == ["revise", "revise", "pass"]
    assert any(r.startswith("residual:") for r in s1[2]["reasons"])
    assert any(OPEN_ISSUES_LABEL in c["user"] for c in llm2.calls if c["prompt"] == "formalizer")
    assert Ledger(run_dir).verify() == (True, None)


def test_stage_error_leaves_card_in_place(run_dir):
    orch, llm = make_orchestrator(run_dir, fail={"formalizer": TransportError("down")})
    summary = run(orch)
    assert summary["reason"] == "stage_errors"   # KD6: not "complete"; --resume requeues the card
    card = only_card(orch)
    assert summary["stage_errors"][0]["card_id"] == card["id"] and summary["stage_errors"][0]["stage"] == "S3"
    assert card["status"]["stage"] == "S2" and verdicts(card)["S2"] == "pass"
    errors = events(orch, "stage_error")
    assert errors[0]["stage"] == "S3" and "TransportError" in errors[0]["error"]
