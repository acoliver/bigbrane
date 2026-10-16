"""Phase 8: loop budgets and stall detection (DISCOVERY-PIPELINE-PLAN.md section 10, hybrid/research/lineage.py).

Three layers:
- the loop rule itself (decide_round, is_stall, fingerprint), checked exhaustively against an independent oracle on
  every failure sequence over a small alphabet;
- each loop of the orchestrator under an adversarial fake LLM: a critic that always asks for revision, one that
  alternates between two findings, an explorer that renames the same idea, a reviser that repeats the same failing
  repair, a reduction critic that keeps reducing escape cards (to the same family, or alternating), a prover that
  resends the same failing proof, model-call caps, and a restart of a recorded idea in a new run;
- a randomized property test: per-call random adversaries over many seeds, after which every run must have ended by
  itself with every count inside its cap.
Scripted LLM, LocalExecutor, no live model calls.
"""
import asyncio
import itertools
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pytest

from hybrid.research import lineage
from hybrid.research.inspect import budget_metrics, inspect_run
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.outcomes import OutcomeStore, check_kill_code
from hybrid.research.signature import card_signature
from hybrid.research.stages.common import ESCAPE_FIELD
from hybrid.research.store import Store
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.test_escape import ARGUMENT, NOT_REDUCED, REDUCED
from hybrid.tests.research.test_orchestrator_fakes import declaring, draft, make_orchestrator, orch_config

FIRST_LIVE_RUN = ROOT / "runs" / "run-20261010T000322Z"


def run(orch):
    return asyncio.run(orch.run())


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261013T000000Z"


def critic(*places):
    """A salad-critic pass with one fixable finding at each place."""
    return {"ok": False, "fatal": [], "notes": "findings",
            "fixable": [{"kind": "missing_definition", "where": place, "problem": f"a symbol at {place} is undefined"}
                        for place in places]}


OK_CRITIC = {"ok": True, "fatal": [], "fixable": [], "notes": "ok"}
SAME_CRITIC = critic("mechanism.update[0]")


def always_new_critic(index, user):
    """Always asks for another revision, each time about something new: never a stall."""
    return critic(f"mechanism.update[{index}]")


def alternating_critic(index, user):
    """Alternates between two findings: no two consecutive rounds are equal, so only the cap ends the loop."""
    return critic("mechanism.update[0]" if index % 2 == 0 else "mechanism.readout[0]")


def escape_aware(make):
    """Wrap an explorer fake so that escape tasks and revisions of escape cards carry the argument."""
    def explorer(index, user):
        card = make(index, user)
        if "OPERATOR: escape" in user or ("OPERATOR: revise" in user and f'"{ESCAPE_FIELD}"' in user):
            card[ESCAPE_FIELD] = ARGUMENT
        return card
    return explorer


def renamer(index, user):
    """The same mechanism under a new title and summary each time (one idea, one signature)."""
    card = draft(index, user=user)
    card["mechanism"] = draft(0)["mechanism"]
    return card


def repeater(index, user):
    """Returns the same card every time, so a revision repeats the failing card exactly."""
    return draft(0, user=user)


NOT_RADICAL = NOT_REDUCED | {"radical_enough": False, "why": "A known running sum with a new name."}


def reducing(*families):
    """A reduction critic that reduces every card, cycling through the families by call index."""
    def critic_reply(index, user):
        return declaring(REDUCED | {"reduces_to": families[index % len(families)]})
    return critic_reply


def config(**overrides) -> dict:
    cfg = orch_config(**overrides)
    cfg["explore"].update(p_mutate=0.0, p_cross=0.0)
    cfg["s5"]["falsifiers"] = False
    return cfg


def with_store(cfg: dict, store_dir: Path) -> dict:
    cfg["outcomes"].update(enabled=True, path=str(store_dir))
    return cfg


def own_records(store: OutcomeStore, run_id: str) -> dict[str, dict]:
    return {s.record["card_id"]: s.record for s in store.current() if s.record["run"] == run_id}


# ----- the loop rule, exhaustively ------------------------------------------------------------------------------

ALPHABET = ({}, {"a": None}, {"b": None}, {"a": [1, 2]}, {"a": None, "b": None})


def drive(sequence, cap):
    """The generic revision loop over a scripted sequence of round failures, as every loop uses it."""
    previous, used = None, 0
    for failing in sequence:
        if not failing:
            return "pass", used
        stalled = lineage.is_stall(previous, failing)
        stop = lineage.decide_round("loop", used=used, cap=cap, stalled=stalled, failing=failing)
        if stop is not None:
            return stop.kill_code, used
        used += 1
        previous = lineage.fingerprint(failing)
    return "unfinished", used


def oracle(sequence, cap):
    """Independent statement of the rule: the first empty round passes; a round equal to the one before stalls; the
    round after `cap` revisions ends on the budget; whichever comes first."""
    for k, failing in enumerate(sequence):
        if not failing:
            return "pass", k
        if k > 0 and failing == sequence[k - 1]:
            return "stall:loop", k
        if k == cap:
            return "budget:loop", k
    return "unfinished", len(sequence)


@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_every_failure_sequence_ends_within_the_cap_as_the_rule_says(cap):
    checked = 0
    for sequence in itertools.product(ALPHABET, repeat=cap + 2):
        outcome, used = drive(sequence, cap)
        assert (outcome, used) == oracle(sequence, cap), sequence
        # termination within budget: at most cap revisions, and never more than cap + 1 rounds
        assert used <= cap and outcome != "unfinished"
        checked += 1
    assert checked == len(ALPHABET) ** (cap + 2)


def test_the_fingerprint_names_the_failing_ids_and_each_detail():
    assert lineage.fingerprint({"a": None, "b": 1}) == lineage.fingerprint({"b": 1, "a": None})
    assert lineage.fingerprint({"a": None}) != lineage.fingerprint({"a": 0})
    assert lineage.fingerprint({"a": {"x": [1, 2]}}) != lineage.fingerprint({"a": {"x": [2, 1]}})
    assert not lineage.is_stall(lineage.fingerprint({}), {})          # nothing failing is never a stall
    assert lineage.LoopStop("s1_revision", "stall", "d").kill_code == "stall:s1_revision"
    with pytest.raises(ValueError):
        lineage.LoopStop("s1_revision", "tired", "d")


def test_observe_round_records_each_round_once_and_stalls_on_an_exact_repeat(tmp_path):
    ledger = Ledger(tmp_path / "run")
    kwargs = {"card_id": "card-00000001", "stage": "S1", "loop": "s1_revision", "key": "card-00000001"}
    assert not lineage.observe_round(ledger, round_id="r0", failures={"a": None}, **kwargs)
    assert not lineage.observe_round(ledger, round_id="r0", failures={"b": None}, **kwargs)  # same round again
    assert not lineage.observe_round(ledger, round_id="r1", failures={"b": None}, **kwargs)
    assert lineage.observe_round(ledger, round_id="r2", failures={"b": None}, **kwargs)
    assert lineage.observe_round(ledger, round_id="r2", failures={}, **kwargs)              # recorded answer
    other = dict(kwargs, key="card-00000002")
    assert not lineage.observe_round(ledger, round_id="r0", failures={"b": None}, **other)  # keys are separate
    rounds = [e["data"] for e in ledger.iter_events("loop_round")]
    assert [(r["key"], r["round_id"], r["stalled"]) for r in rounds] == [
        ("card-00000001", "r0", False), ("card-00000001", "r1", False), ("card-00000001", "r2", True),
        ("card-00000002", "r0", False)]
    assert rounds[2]["previous"] == "r1" and rounds[2]["failing"] == ["b"]


# ----- config and kill codes ------------------------------------------------------------------------------------

def test_shipped_config_has_the_plan_defaults_and_older_configs_get_them(tmp_path):
    shipped = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    assert shipped["budgets"]["idea"] == {"refinement_rounds": 3, "send_backs": 1, "math_revision_rounds": 3,
                                          "model_calls": 80}
    assert shipped["budgets"]["lineage"] == {"ideas": 4, "model_calls": 200}
    assert shipped["math"]["max_prover_attempts"] == 4
    for older in ("run2.toml", "run3.toml", "run4.toml", "run5.toml"):
        cfg = load_run_config(ROOT / "hybrid" / "config" / older)
        assert cfg["budgets"]["idea"] == lineage.IDEA_DEFAULTS and cfg["budgets"]["lineage"] == lineage.LINEAGE_DEFAULTS
    assert {loop: lineage.loop_cap(shipped, loop) for loop in lineage.IDEA_LOOPS} == {
        "s1_revision": 2, "refinement": 3, "send_back": 1, "math_revision": 3}


@pytest.mark.parametrize("old,new,message", [
    ("model_calls = 80", "model_calls = 0", "budgets.idea.model_calls must be an integer >= 1"),
    ("send_backs = 1", "send_backs = -1", "budgets.idea.send_backs must be an integer >= 0"),
    ("refinement_rounds = 3", "refinement_rounds = 1.5", "budgets.idea.refinement_rounds must be an integer"),
    ("send_backs = 1", "send_backs = 1\nrounds = 3", r"unknown \[budgets.idea\] keys \['rounds'\]"),
    ("ideas = 4", "ideas = 0", "budgets.lineage.ideas must be an integer >= 1"),
    ("model_calls = 200", "model_calls = true", "budgets.lineage.model_calls must be an integer"),
])
def test_bad_budget_values_are_refused(tmp_path, old, new, message):
    shipped = (ROOT / "hybrid" / "config" / "run.toml").read_text(encoding="utf-8")
    assert shipped.count(f"\n{old}\n") == 1
    path = tmp_path / "run.toml"
    path.write_text(shipped.replace(f"\n{old}\n", f"\n{new}\n"), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_run_config(path)


def test_budget_and_stall_kill_codes_name_their_loop():
    for code in ("budget", "stall", "budget:s1_revision", "budget:lineage_ideas", "stall:escape", "stall:prover",
                 "budget:model_calls"):
        assert check_kill_code(code) == code
    for bad in ("budget:", "stall:Prover", "budget s1"):
        with pytest.raises(ValueError):
            check_kill_code(bad)


# ----- S1 revisions under adversarial critics and revisers ---------------------------------------------------------

def s1_line(orch):
    return [d["verdict"] for d in events(orch, "verdict") if d["stage"] == "S1"]


def test_a_critic_that_always_asks_for_revision_stops_at_the_cap(run_dir, tmp_path):
    cfg = with_store(config(), tmp_path / "outcomes")
    cfg["s1"]["pass_with_residual_fixable"] = False
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"salad_critic": always_new_critic})
    assert run(orch)["reason"] == "complete"
    assert s1_line(orch) == ["revise", "revise", "fail"]
    assert llm.prompts().count("salad_critic") == 3
    [stop] = events(orch, "loop_stop")
    assert stop["kill_code"] == "budget:s1_revision" and not stop["terminal"]
    card_id = orch.store.list_cards()[0]
    record = own_records(OutcomeStore(tmp_path / "outcomes"), run_dir.name)[card_id]
    assert record["kill_codes"] == ["ill_formed:fixable", "budget:s1_revision"]
    assert record["budget"]["loops"] == {"s1_revision": 2} and record["budget"]["stops"][0]["reason"] == "budget"


def test_a_critic_that_alternates_never_stalls_and_ends_on_the_cap(run_dir):
    cfg = config()
    cfg["s1"]["max_revisions"] = 3
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"salad_critic": alternating_critic})
    run(orch)
    assert s1_line(orch) == ["revise", "revise", "revise", "pass"]          # residual pass after the cap
    assert [r["stalled"] for r in events(orch, "loop_round")] == [False, False, False, False]
    assert [s["kill_code"] for s in events(orch, "loop_stop")] == ["budget:s1_revision"]


def test_a_reviser_that_repeats_the_same_failing_repair_stalls_after_one_revision(run_dir, tmp_path):
    cfg = with_store(config(), tmp_path / "outcomes")
    cfg["s1"]["pass_with_residual_fixable"] = False
    cfg["s1"]["max_revisions"] = 3
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"explorer": repeater, "salad_critic": SAME_CRITIC})
    run(orch)
    assert s1_line(orch) == ["revise", "fail"]
    assert llm.prompts().count("explorer") == 2                            # the draft and one revision
    [stop] = events(orch, "loop_stop")
    assert stop["kill_code"] == "stall:s1_revision"
    assert "missing_definition@mechanism.update[0]" in stop["detail"]
    record = own_records(OutcomeStore(tmp_path / "outcomes"), run_dir.name)[orch.store.list_cards()[0]]
    assert record["kill_codes"] == ["ill_formed:fixable", "stall:s1_revision"]


def test_the_same_toy_mismatch_after_a_revision_is_a_stall(run_dir):
    def wrong_toy(index, user):
        card = draft(0, user=user)
        card["toy_example"]["expected"]["state_norm"] = 99.0
        return card
    orch, _ = make_orchestrator(run_dir, cfg=config(), script={"explorer": wrong_toy})
    run(orch)
    assert s1_line(orch) == ["revise", "fail"]
    assert [s["kill_code"] for s in events(orch, "loop_stop")] == ["stall:s1_revision"]
    assert events(orch, "loop_round")[-1]["failing"] == ["toy_example"]


# ----- lineages: renamed ideas, restarts, escapes, mutations --------------------------------------------------------

def test_a_renamed_idea_shares_one_lineage_and_one_revision_count(run_dir, tmp_path):
    cfg = with_store(config(budgets__max_cards=6), tmp_path / "outcomes")
    cfg["budgets"]["lineage"]["ideas"] = 3
    script = {"explorer": renamer, "salad_critic": always_new_critic, "reduction_critic": NOT_RADICAL}
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script)
    assert run(orch)["reason"] == "complete"
    assigned = events(orch, "lineage_assigned")
    assert len(assigned) == 6 and len({a["idea_key"] for a in assigned}) == 1
    assert len({a["lineage_id"] for a in assigned}) == 1
    assert [a["joined"] for a in assigned] == ["new"] + ["signature"] * 5
    assert [a["admitted"] for a in assigned] == [True, True, True, False, False, False]
    # the idea used its two revisions once; later copies get none (S1 passes them with residual findings at once)
    charges = [c for c in events(orch, "budget_charge") if c["loop"] == "s1_revision"]
    assert len(charges) == 2 and len({c["card_id"] for c in charges}) == 1
    refused = [a["card_id"] for a in assigned if not a["admitted"]]
    store = OutcomeStore(tmp_path / "outcomes")
    records = own_records(store, run_dir.name)
    for card_id in refused:
        card = orch.store.get_card(card_id)
        assert card["status"]["stage"] == "S0" and card["status"]["verdicts"]["S0"]["verdict"] == "fail"
        assert records[card_id]["kill_codes"] == ["budget:lineage_ideas"]
        assert records[card_id]["lineage_id"] == assigned[0]["lineage_id"]
    assert llm.prompts().count("salad_critic") == 3 + 2                      # 3 for the first card, 1 per later copy


def test_a_restart_in_a_new_run_inherits_the_recorded_lineage(tmp_path):
    store_dir = tmp_path / "outcomes"
    first = tmp_path / "run-20261013T000001Z"
    cfg = with_store(config(budgets__max_cards=3), store_dir)
    cfg["budgets"]["lineage"]["ideas"] = 3
    script = {"explorer": renamer, "reduction_critic": NOT_RADICAL}
    orch, _ = make_orchestrator(first, cfg=cfg, script=script)
    run(orch)
    lineage_id = events(orch, "lineage_assigned")[0]["lineage_id"]
    assert all(r["outcome"] == "dead" for r in own_records(OutcomeStore(store_dir), first.name).values())

    # "what if we tried" again: a new run, the same mechanism under new names
    second = tmp_path / "run-20261013T000002Z"
    cfg2 = with_store(config(budgets__max_cards=2), store_dir)
    cfg2["budgets"]["lineage"]["ideas"] = 3
    orch2, llm2 = make_orchestrator(second, cfg=cfg2, script=script)
    run(orch2)
    assigned = events(orch2, "lineage_assigned")
    assert [(a["joined"], a["via"], a["admitted"]) for a in assigned] == [("signature", lineage_id, False)] * 2
    assert assigned[0]["lineage_ideas"] == 3
    assert "salad_critic" not in llm2.prompts()                               # nothing past S0
    assert events(orch2, "run_start")[0]["loop_budgets"]["prior"] == {"records": 3, "lineages": 1}
    records = own_records(OutcomeStore(store_dir), second.name)
    assert {code for r in records.values() for code in r["kill_codes"]} == {"budget:lineage_ideas"}


def escape_run(run_dir, reduction, max_cards, reserve, ideas=4):
    """One new card, then only escapes: the reserve holds every other slot for escapes, so the explorer waits for
    reductions and stops when none can come."""
    cfg = config(budgets__max_cards=max_cards)
    cfg["explore"]["escape_reserve"] = reserve
    cfg["budgets"]["lineage"]["ideas"] = ideas
    explorer = escape_aware(lambda index, user: draft(index + 1, user=user))
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"explorer": explorer, "reduction_critic": reduction})
    assert run(orch)["reason"] == "complete"
    return orch, llm


def test_an_escape_reduced_to_its_parents_family_ends_the_escape_loop_as_a_stall(run_dir):
    orch, _ = escape_run(run_dir, reducing("ssm"), max_cards=3, reserve=0.6)
    created = events(orch, "card_created")
    assert [c["op"] for c in created] == ["new", "escape"]
    child = created[1]["card_id"]
    [stop] = events(orch, "loop_stop")
    assert (stop["card_id"], stop["kill_code"], stop["stage"]) == (child, "stall:escape", "S2")
    assert events(orch, "escape_skipped")[-1]["reason"].startswith("stall:escape")
    view = RunView(orch.run_dir)
    assert build_record(view, child, "orchestrator")["kill_codes"] == ["reduces_to_known:ssm", "stall:escape"]
    assert build_record(view, created[0]["card_id"], "orchestrator")["kill_codes"] == ["reduces_to_known:ssm"]


def test_escapes_that_keep_changing_family_end_at_the_lineage_cap(run_dir):
    orch, _ = escape_run(run_dir, reducing("ssm", "linear_attention"), max_cards=4, reserve=0.75, ideas=3)
    created = events(orch, "card_created")
    assert [c["op"] for c in created] == ["new", "escape", "escape"]
    assert len({a["lineage_id"] for a in events(orch, "lineage_assigned")}) == 1
    [stop] = events(orch, "loop_stop")
    assert stop["card_id"] == created[2]["card_id"] and stop["kill_code"] == "budget:escape"
    assert "holds 3 idea(s), cap 3" in stop["detail"]


def test_a_mutation_without_room_in_the_lineage_is_skipped(run_dir):
    cfg = config()
    cfg["budgets"]["lineage"]["ideas"] = 1
    orch, _ = make_orchestrator(run_dir, cfg=cfg)
    run(orch)
    card_id = orch.store.list_cards()[0]
    assert events(orch, "mutation_requested") == []
    [skipped] = events(orch, "mutation_skipped")
    assert skipped["card_id"] == card_id and skipped["reason"].startswith("budget:mutation")
    # the card finished; a mutation stop never counts as a cause of death
    record = build_record(RunView(orch.run_dir), card_id, "orchestrator")
    assert record["outcome"] == "finished" and record["kill_codes"] == []


def test_random_mutate_and_cross_draws_skip_parents_whose_lineage_is_full(run_dir):
    cfg = config(budgets__max_cards=4)
    cfg["explore"].update(p_mutate=1.0, p_cross=0.0)
    cfg["budgets"]["lineage"]["ideas"] = 1
    orch, _ = make_orchestrator(run_dir, cfg=cfg, script={"reduction_critic": NOT_RADICAL})
    run(orch)
    # every card is its own full lineage, so no card is ever drawn as a parent
    assert {c["op"] for c in events(orch, "card_created")} == {"new"}


# ----- model-call caps ---------------------------------------------------------------------------------------------

def test_the_idea_model_call_cap_ends_the_card_at_the_stage_that_reaches_it(run_dir, tmp_path):
    cfg = with_store(config(), tmp_path / "outcomes")
    cfg["budgets"]["idea"]["model_calls"] = 4        # explorer, salad critic, prior art, reduction critic
    orch, llm = make_orchestrator(run_dir, cfg=cfg)
    run(orch)
    assert llm.prompts() == ["explorer", "salad_critic", "prior_art_query", "reduction_critic"]
    card = orch.store.get_card(orch.store.list_cards()[0])
    assert card["status"]["stage"] == "S3"
    assert card["status"]["verdicts"]["S3"]["reasons"][0].startswith("gate: budget:model_calls: idea sig-")
    [stop] = events(orch, "loop_stop")
    assert stop["terminal"] and stop["stage"] == "S3" and stop["kill_code"] == "budget:model_calls"
    calls = [c for c in events(orch, "budget_charge") if c["loop"] == "model_calls"]
    assert [c["used"] for c in calls] == [1, 2, 3, 4] and [c["stage"] for c in calls] == ["S0", "S1", "S2", "S2"]
    record = own_records(OutcomeStore(tmp_path / "outcomes"), run_dir.name)[card["id"]]
    assert record["kill_codes"] == ["budget:model_calls"] and record["terminal_stage"] == "S3"
    assert record["budget"]["model_calls"] == 4


def test_the_lineage_model_call_cap_counts_every_card_of_the_lineage(run_dir):
    cfg = config(budgets__max_cards=2)
    cfg["explore"]["escape_reserve"] = 0.5
    cfg["budgets"]["lineage"]["model_calls"] = 6
    explorer = escape_aware(lambda index, user: draft(index + 1, user=user))
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"explorer": explorer, "reduction_critic": reducing("ssm")})
    run(orch)
    # parent: explorer, critic, prior art, reduction critic, check = 5; the escape child's explorer call is the 6th,
    # which leaves its lineage no call for S1, so it ends at S0
    child = [c for c in events(orch, "card_created") if c["op"] == "escape"][0]["card_id"]
    [stop] = [s for s in events(orch, "loop_stop") if s["card_id"] == child]
    assert stop["kill_code"] == "budget:lineage_calls" and stop["stage"] == "S0" and stop["terminal"]
    assert llm.prompts().count("salad_critic") == 1


# ----- the prover loop in stage M ------------------------------------------------------------------------------------

def test_a_prover_that_resends_the_same_failing_proof_stalls(tmp_path):
    from hybrid.tests.research import math_fakes as mf
    from hybrid.tests.research.test_math_stages import Harness

    h = Harness(tmp_path, mf.math_script(proofs=mf.SORRY_PROOF))
    h.ctx.cfg["math"]["max_prover_attempts"] = 4
    assert h.m() is None
    attempts = [e["data"]["attempt"] for e in h.events("proof_attempt") if e["data"]["statement_id"] == "O5.a"]
    assert attempts == [1, 2]                                   # the second round repeated the first exactly
    [stop] = [e["data"] for e in h.events("loop_stop")]
    assert stop["kill_code"] == "stall:prover" and stop["stage"] == "M" and not stop["terminal"]
    assert any(r.startswith("loop prover ended: stall:prover") for r in h.card_verdict("M")["reasons"])


def test_a_prover_that_changes_its_failing_proof_runs_to_the_attempt_cap(tmp_path):
    from hybrid.tests.research import math_fakes as mf
    from hybrid.tests.research.test_math_stages import Harness

    h = Harness(tmp_path, mf.math_script(proofs=[mf.SORRY_PROOF, mf.WEAKER_PROOF, mf.SORRY_PROOF]))
    h.ctx.cfg["math"]["max_prover_attempts"] = 3
    assert h.m() is None
    attempts = [e["data"]["attempt"] for e in h.events("proof_attempt") if e["data"]["statement_id"] == "O5.a"]
    assert attempts == [1, 2, 3]
    assert [e["data"]["kill_code"] for e in h.events("loop_stop")] == ["budget:prover"]


# ----- resume ------------------------------------------------------------------------------------------------------

def test_a_resumed_run_keeps_its_counts(run_dir):
    cfg = config(budgets__max_calls=3)          # explorer, critic (revise), revision; the second critic is refused
    orch, _ = make_orchestrator(run_dir, cfg=cfg, script={"salad_critic": always_new_critic})
    run(orch)
    orch2, llm2 = make_orchestrator(run_dir, cfg=config(), resume=True,
                                    script={"salad_critic": lambda i, u: always_new_critic(i + 10, u)})
    assert orch2.book.used(orch.store.list_cards()[0], "s1_revision") == 1
    assert orch2.book.idea_calls[events(orch, "lineage_assigned")[0]["idea_key"]] == 3
    run(orch2)
    assert s1_line(orch2) == ["revise", "revise", "pass"]       # one revision left after the resume, not two
    assert events(orch2, "lineages_assigned_at_resume") == []


def test_a_run_from_before_phase_8_resumes_with_lineages_and_its_used_revisions(tmp_path):
    """The first live run (four cards, two at S1 revise with one revision used, no budget events), copied so the
    original is never touched, resumes: every card gets a lineage at resume, the revision already used counts, so
    each revising card gets exactly one more."""
    run_dir = tmp_path / FIRST_LIVE_RUN.name
    shutil.copytree(FIRST_LIVE_RUN, run_dir, ignore=shutil.ignore_patterns("ledger.lock", "report"))
    revising = [c for c in Store(run_dir, Ledger(run_dir)).list_cards()
                if Store(run_dir, Ledger(run_dir)).get_card(c)["status"]["verdicts"]["S1"]["verdict"] == "revise"]
    assert len(revising) == 2
    cfg = config(budgets__max_cards=4)
    orch, llm = make_orchestrator(run_dir, cfg=cfg, resume=True,
                                  script={"salad_critic": always_new_critic, "reduction_critic": NOT_RADICAL})
    assert run(orch)["reason"] == "complete"
    assigned = events(orch, "lineage_assigned")
    assert len(assigned) == 4 and all(a["legacy"] for a in assigned)
    assert sorted(events(orch, "lineages_assigned_at_resume")[0]["cards"]) == sorted(orch.store.list_cards())
    charges = Counter(c["card_id"] for c in events(orch, "budget_charge") if c["loop"] == "s1_revision")
    assert charges == {card_id: 1 for card_id in revising}
    stops = {s["card_id"]: s for s in events(orch, "loop_stop")}
    for card_id in revising:
        assert stops[card_id]["kill_code"] == "budget:s1_revision" and "2 of 2" in stops[card_id]["detail"]
    assert Ledger(run_dir).verify() == (True, None)


# ----- inspect -----------------------------------------------------------------------------------------------------

def test_inspect_shows_budget_use_and_stall_kills(run_dir, capsys):
    cfg = config()
    cfg["s1"]["pass_with_residual_fixable"] = False
    orch, _ = make_orchestrator(run_dir, cfg=cfg, script={"explorer": repeater, "salad_critic": SAME_CRITIC})
    run(orch)
    card_id = orch.store.list_cards()[0]
    assert inspect_run(run_dir) == 0
    out = capsys.readouterr().out
    assert "Loop budgets: per idea 80 model calls, loops math_revision 3, refinement 3, s1_revision 2, " \
           "send_back 1; per lineage 4 ideas and 200 model calls" in out
    assert f"lineage {card_id}: 1 of 4 ideas" in out
    assert f"{card_id} (new) idea sig-" in out and "s1_revision 1;" in out
    assert "Loop stops: 1 stall(s), 0 budget stop(s)" in out
    assert f"{card_id} S1 stall:s1_revision (the gate decided after it)" in out
    overview = (run_dir / "report" / "index.html").read_text()
    assert "Loop budgets and stalls" in overview and "stall:s1_revision" in overview
    dossier = (run_dir / "report" / "dossiers" / f"{card_id}.html").read_text()
    assert "Loop budget" in dossier and "s1_revision 1" in dossier and "stall:s1_revision" in dossier


def test_inspect_of_a_run_without_budget_events_has_no_budget_block():
    assert budget_metrics([{"type": "run_start", "data": {}}]) == {}


# ----- the property test: randomized adversaries ----------------------------------------------------------------------

CAPS = {"max_revisions": 2, "ideas": 3, "idea_calls": 30, "lineage_calls": 60}


def adversary(seed: int) -> dict:
    """A fake LLM script whose every call picks its behaviour at random (seeded): critics that pass, repeat, always ask
    for something new, alternate or go fatal; explorers that make new ideas, rename one idea or repeat a card; reduction
    critics that pass, fail as not radical, or reduce to one of two families; an analyst that supports a claim."""
    rng = random.Random(seed)

    def salad(index, user):
        return rng.choice([OK_CRITIC, SAME_CRITIC, always_new_critic(index, user), alternating_critic(index, user),
                           SAME_CRITIC])

    def explore(index, user):
        return rng.choice([lambda: draft(index + 1, user=user), lambda: renamer(index + 1, user),
                           lambda: repeater(index, user)])()

    def reduction(index, user):
        return rng.choice([NOT_REDUCED, NOT_RADICAL, REDUCED, declaring(REDUCED | {"reduces_to": "linear_attention"}), REDUCED])

    return {"explorer": escape_aware(explore), "salad_critic": salad, "reduction_critic": reduction}


@pytest.mark.parametrize("seed", range(10))
def test_every_loop_terminates_within_budget_under_random_adversaries(tmp_path, seed):
    run_dir = tmp_path / f"run-2026101{seed % 10}T000000Z"
    cfg = with_store(config(budgets__max_cards=7, run__n_explorers=2, budgets__max_calls=10_000), tmp_path / "out")
    cfg["explore"].update(p_mutate=0.3, p_cross=0.1, escape_reserve=0.3)
    cfg["s1"]["max_revisions"] = CAPS["max_revisions"]
    cfg["s1"]["pass_with_residual_fixable"] = bool(seed % 2)
    cfg["budgets"]["lineage"].update(ideas=CAPS["ideas"], model_calls=CAPS["lineage_calls"])
    cfg["budgets"]["idea"]["model_calls"] = CAPS["idea_calls"]
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=adversary(seed))
    summary = run(orch)

    # The run ended by itself: no run budget (calls, tokens, wall time) had to stop it, nothing is left queued.
    assert summary["reason"] == "complete" and summary["queued_unfinished"] == {}
    every = list(orch.ledger.iter_events())
    assigned = {e["data"]["card_id"]: e["data"] for e in every if e["type"] == "lineage_assigned"}
    assert set(assigned) == set(orch.store.list_cards())
    # Every card ended: a terminal verdict, or S6.
    for card_id in orch.store.list_cards():
        status = orch.store.get_card(card_id)["status"]
        verdict = status["verdicts"].get(status["stage"], {}).get("verdict")
        assert status["stage"] == "S6" or verdict not in (None, "pass", "revise"), (card_id, status["stage"], verdict)
    # Per idea: revisions within the cap; model calls made by stages within the cap.
    revisions = Counter(e["data"]["idea_key"] for e in every
                        if e["type"] == "budget_charge" and e["data"]["loop"] == "s1_revision")
    assert all(n <= CAPS["max_revisions"] for n in revisions.values()), revisions
    for e in every:
        if e["type"] == "budget_charge" and e["data"]["loop"] == "model_calls" and e["data"]["stage"] != "S0":
            assert e["data"]["used"] <= CAPS["idea_calls"] and e["data"]["lineage_used"] <= CAPS["lineage_calls"]
    # An S0 charge past a cap belongs to a card that ended at S0.
    for e in every:
        data = e["data"]
        if e["type"] == "budget_charge" and data["loop"] == "model_calls" and data["stage"] == "S0" and (
                data["used"] >= CAPS["idea_calls"] or data["lineage_used"] >= CAPS["lineage_calls"]):
            assert orch.store.get_card(data["card_id"])["status"]["stage"] == "S0"
    # Per lineage: admitted ideas within the cap.
    admitted = Counter(a["lineage_id"] for a in assigned.values() if a["admitted"])
    assert all(n <= CAPS["ideas"] for n in admitted.values()), admitted
    # Stalls: a round marked stalled repeats the round before it, and ends its loop (no later round for that key).
    rounds: dict[tuple, list[dict]] = {}
    for e in every:
        if e["type"] == "loop_round":
            rounds.setdefault((e["data"]["key"], e["data"]["loop"]), []).append(e["data"])
    for sequence in rounds.values():
        for previous, current in zip(sequence, sequence[1:]):
            assert current["stalled"] == (bool(current["failing"]) and current["fingerprint"] == previous["fingerprint"])
        assert all(not r["stalled"] for r in sequence[:-1])
    # S1 never revised a card more often than the cap allows.
    for card_id in orch.store.list_cards():
        revise = sum(1 for e in every if e["type"] == "verdict" and e["data"]["card_id"] == card_id
                     and e["data"]["stage"] == "S1" and e["data"]["verdict"] == "revise")
        assert revise <= CAPS["max_revisions"]
    # Every stop carries a kill code from the enumeration, and every dead card's record carries one.
    for e in every:
        if e["type"] == "loop_stop":
            check_kill_code(e["data"]["kill_code"])
    records = own_records(OutcomeStore(tmp_path / "out"), run_dir.name)
    for card_id, record in records.items():
        assert record["outcome"] != "dead" or record["kill_codes"]
        for stop in (record["budget"] or {}).get("stops", []):
            if stop["terminal"]:
                assert stop["kill_code"] in record["kill_codes"]
    # Total model calls are bounded by the cards and the per-idea cap.
    assert len(llm.calls) <= cfg["budgets"]["max_cards"] * CAPS["idea_calls"]
    assert Ledger(run_dir).verify() == (True, None)


def test_a_model_call_cap_reached_at_s6_kills_the_card_with_the_stops_code(tmp_path):
    """FIX-14. The property test above failed for seed 7 when the idea's model calls ran out exactly at the S6
    analyst call (a sibling card of the same idea had spent its share first, so it depended on the interleaving of
    two explorers): the orchestrator recorded the terminal stop and an S6 fail, but the outcome record read S6 from
    the claims ledger first and came out finished with no kill code. Here one card, run once to count the calls it
    makes before S6, runs again with the idea's cap set to that count, so the analyst call is the one refused."""
    probe_dir = tmp_path / "run-20261014T000001Z"
    probe, _ = make_orchestrator(probe_dir, cfg=with_store(config(), tmp_path / "probe-out"))
    assert run(probe)["reason"] == "complete"
    [card_id] = probe.store.list_cards()
    assert probe.store.get_card(card_id)["status"]["stage"] == "S6"
    before_s6 = sum(1 for c in events(probe, "budget_charge")
                    if c["loop"] == "model_calls" and c["stage"] != "S6")
    assert before_s6 > 0

    run_dir = tmp_path / "run-20261014T000002Z"
    cfg = with_store(config(), tmp_path / "out")
    cfg["budgets"]["idea"]["model_calls"] = before_s6
    orch, llm = make_orchestrator(run_dir, cfg=cfg)
    assert run(orch)["reason"] == "complete"
    [card_id] = orch.store.list_cards()
    assert not [c for c in llm.calls if c["prompt"] == "analyst"]
    [stop] = events(orch, "loop_stop")
    assert (stop["stage"], stop["kill_code"], stop["terminal"]) == ("S6", "budget:model_calls", True)
    status = orch.store.get_card(card_id)["status"]
    assert status["stage"] == "S6" and status["verdicts"]["S6"]["verdict"] == "fail"
    [recorded] = events(orch, "outcome_recorded")
    assert (recorded["outcome"], recorded["kill_codes"]) == ("dead", ["budget:model_calls"])
    record = own_records(OutcomeStore(tmp_path / "out"), run_dir.name)[card_id]
    assert (record["terminal_stage"], record["outcome"], record["kill_codes"]) == ("S6", "dead", ["budget:model_calls"])
    assert f"used {before_s6} of {before_s6} model calls" in record["causes"][0]["detail"]
    assert [s["kill_code"] for s in record["budget"]["stops"] if s["terminal"]] == record["kill_codes"]
    # The backfill reads the same run the same way.
    assert build_record(RunView(run_dir), card_id, "backfill")["kill_codes"] == ["budget:model_calls"]


def test_an_s6_fail_without_a_terminal_stop_is_refused(tmp_path):
    """Only a terminal loop stop fails a card at S6 (S6 itself only passes); any other S6 fail has no kill-code
    mapping, so the record is refused rather than written as finished."""
    from hybrid.tests.research.outcome_fixtures import build_run

    stages = ("S0", "S1", "S2", "S3", "S4", "S5")
    run_dir = build_run(tmp_path / "run-20261014T000003Z", [(make_card("card-0000f014"), [
        *((stage, "pass", ["ok"]) for stage in stages), ("S6", "fail", ["gate: something"])], {})])
    with pytest.raises(ValueError, match="S6 fail without a terminal loop stop"):
        build_record(RunView(run_dir), "card-0000f014", "backfill")


# ----- loops of later phases plug in by naming their loop -------------------------------------------------------------
# Phase 7's refinement dialogue and prior-art send-back and stage V's math revisions are not built yet. These tests
# drive BudgetBook.after_round the way those stages will: one call per failing round, naming the loop.

PLAN_CAPS = {"refinement": 3, "send_back": 1, "math_revision": 3}


def book_for(tmp_path, *, ideas=4):
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    cfg["budgets"]["lineage"]["ideas"] = ideas
    run_dir = tmp_path / "run-20261013T000009Z"
    ledger = Ledger(run_dir)
    book = lineage.BudgetBook(cfg, ledger, Store(run_dir, ledger), run_id=run_dir.name)
    return book, ledger


def drive_loop(book, card_id, loop, rounds, *, stage="R"):
    """A later phase's revision loop over scripted round failures (an empty round passes). Returns the outcome and
    the revisions granted."""
    granted = 0
    for k, failing in enumerate(rounds):
        if not failing:
            return "pass", granted
        stop = book.after_round(card_id=card_id, stage=stage, loop=loop, round_id=f"round-{k}", failures=failing)
        if stop is not None:
            return stop.kill_code, granted
        granted += 1
    return "unfinished", granted


def new_questions(n):
    """A challenger that always asks something new."""
    return [{f"undefined_object@q{k}": None} for k in range(n)]


def alternating(n):
    return [{"unstated_assumption@a": None} if k % 2 == 0 else {"claim_without_test@b": None} for k in range(n)]


@pytest.mark.parametrize("loop", sorted(PLAN_CAPS))
def test_each_named_loop_stops_at_its_plan_cap_with_its_kill_code(tmp_path, loop):
    book, ledger = book_for(tmp_path)
    book.assign(make_card("card-0000000a"), op="new", parents=[])
    outcome, granted = drive_loop(book, "card-0000000a", loop, new_questions(10))
    assert (outcome, granted) == (f"budget:{loop}", PLAN_CAPS[loop])
    check_kill_code(outcome)
    [stop] = [e["data"] for e in ledger.iter_events("loop_stop")]
    assert stop["kill_code"] == f"budget:{loop}" and not stop["terminal"] and stop["stage"] == "R"
    charges = [e["data"] for e in ledger.iter_events("budget_charge") if e["data"]["loop"] == loop]
    assert [c["used"] for c in charges] == list(range(1, PLAN_CAPS[loop] + 1))
    assert all(c["cap"] == PLAN_CAPS[loop] for c in charges)


@pytest.mark.parametrize("loop", sorted(PLAN_CAPS))
def test_an_alternating_adversary_never_stalls_and_ends_on_the_cap(tmp_path, loop):
    book, ledger = book_for(tmp_path)
    book.assign(make_card("card-0000000a"), op="new", parents=[])
    assert drive_loop(book, "card-0000000a", loop, alternating(10)) == (f"budget:{loop}", PLAN_CAPS[loop])
    assert not any(e["data"]["stalled"] for e in ledger.iter_events("loop_round"))


def test_the_same_failing_obligations_and_counterexamples_end_math_revision_as_a_stall(tmp_path):
    book, ledger = book_for(tmp_path)
    book.assign(make_card("card-0000000a"), op="new", parents=[])
    same = {"obligation_failed:P2": {"counterexample": {"n": 7, "x": [1, 0]}}, "unproven:A1": None}
    assert drive_loop(book, "card-0000000a", "math_revision", [same, dict(same)]) == ("stall:math_revision", 1)
    [stop] = [e["data"] for e in ledger.iter_events("loop_stop")]
    assert "obligation_failed:P2" in stop["detail"] and "unproven:A1" in stop["detail"]


def test_a_new_counterexample_for_the_same_obligation_is_progress_not_a_stall(tmp_path):
    book, ledger = book_for(tmp_path)
    book.assign(make_card("card-0000000a"), op="new", parents=[])
    rounds = [{"obligation_failed:P2": {"counterexample": {"n": n}}} for n in (7, 8, 9, 10, 11)]
    assert drive_loop(book, "card-0000000a", "math_revision", rounds) == ("budget:math_revision", 3)


def test_renamed_variants_share_the_counts_of_a_named_loop(tmp_path):
    book, ledger = book_for(tmp_path)
    first = make_card("card-0000000a", title="Phase-bound sum")
    renamed = make_card("card-0000000b", title="Rotary ledger memory")
    assert card_signature(first)["signature"] == card_signature(renamed)["signature"]
    book.assign(first, op="new", parents=[])
    assert book.assign(renamed, op="new", parents=[]).joined == "signature"
    assert drive_loop(book, "card-0000000a", "refinement", new_questions(2) + [{}]) == ("pass", 2)
    # the renamed copy gets the one round its idea has left, keyed by its own card id for stall detection
    assert drive_loop(book, "card-0000000b", "refinement", new_questions(10)) == ("budget:refinement", 1)


def test_a_send_back_takes_an_idea_slot_of_the_lineage(tmp_path):
    book, ledger = book_for(tmp_path, ideas=2)
    root = make_card("card-0000000a")
    book.assign(root, op="new", parents=[])
    assert book.room("card-0000000a") is None
    assert book.after_round(card_id="card-0000000a", stage="S2", loop="send_back", round_id="difference-1",
                            failures={"no_difference": "closest prior X used Y"}) is None
    [charge] = [e["data"] for e in ledger.iter_events("budget_charge")]
    assert charge["lineage_ideas"] == 2 and charge["ideas_cap"] == 2
    # the lineage is now full: an escape or mutate child is refused
    child = book.assign(make_card("card-0000000b", parents=["card-0000000a"]), op="escape", parents=["card-0000000a"])
    assert not child.admitted
    # a rebuilt book (resume) counts the slot too
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    cfg["budgets"]["lineage"]["ideas"] = 2
    again = lineage.BudgetBook(cfg, ledger, book.store, run_id=book.run_id)
    assert again.lineage_ideas["card-0000000a"] == book.lineage_ideas["card-0000000a"]
    assert len(again.lineage_ideas["card-0000000a"]) == 2 and again.room("card-0000000a") is not None


def test_send_back_slots_recorded_in_the_outcome_store_count_in_a_later_run(tmp_path):
    """A send-back used in an earlier run still holds its lineage slot when the store seeds a new run's book."""
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    cfg["budgets"]["lineage"]["ideas"] = 2
    first_dir = tmp_path / "run-20261013T000010Z"
    first_ledger = Ledger(first_dir)
    first = lineage.BudgetBook(cfg, first_ledger, Store(first_dir, first_ledger), run_id=first_dir.name)
    card = make_card("card-0000000a")
    first.assign(card, op="new", parents=[])
    assert first.after_round(card_id=card["id"], stage="S2", loop="send_back", round_id="d1",
                             failures={"no_difference": None}) is None
    budget = lineage.card_budget(list(first_ledger.iter_events()), card["id"])
    assert budget["loops"] == {"send_back": 1}

    class Stored:
        record = {"run": first_dir.name, "card_id": card["id"], "lineage_id": card["id"],
                  "mechanism_signature": card_signature(card), "budget": budget, "cost": {"model_calls": 0}}

    class Outcomes:
        def current(self):
            return [Stored]

    second_dir = tmp_path / "run-20261013T000011Z"
    second_ledger = Ledger(second_dir)
    second = lineage.BudgetBook(cfg, second_ledger, Store(second_dir, second_ledger), run_id=second_dir.name,
                                outcomes=Outcomes())
    restart = second.assign(make_card("card-0000000c", title="A new name"), op="new", parents=[])
    assert restart.joined == "signature" and not restart.admitted       # card + its send-back fill the 2 slots
    assert second.used("card-0000000c", "send_back") == 1


def test_a_send_back_in_a_full_lineage_is_a_budget_stop_without_a_charge(tmp_path):
    book, ledger = book_for(tmp_path, ideas=1)
    book.assign(make_card("card-0000000a"), op="new", parents=[])
    stop = book.after_round(card_id="card-0000000a", stage="S2", loop="send_back", round_id="difference-1",
                            failures={"no_difference": "closest prior X used Y"})
    assert stop.kill_code == "budget:send_back" and "holds 1 idea(s), cap 1" in stop.detail
    assert list(ledger.iter_events("budget_charge")) == []
    assert [e["data"]["kill_code"] for e in ledger.iter_events("loop_stop")] == ["budget:send_back"]


def test_after_round_refuses_unknown_loops_and_passing_rounds(tmp_path):
    book, _ = book_for(tmp_path)
    book.assign(make_card("card-0000000a"), op="new", parents=[])
    with pytest.raises(ValueError, match="unknown budget loop 'refine'"):
        book.after_round(card_id="card-0000000a", stage="R", loop="refine", round_id="r0", failures={"a": None})
    with pytest.raises(ValueError, match="no failures"):
        book.after_round(card_id="card-0000000a", stage="R", loop="refinement", round_id="r0", failures={})


def test_a_resumed_stage_repeating_a_round_gets_the_recorded_answer_and_no_fresh_rounds(tmp_path):
    book, ledger = book_for(tmp_path)
    book.assign(make_card("card-0000000a"), op="new", parents=[])
    failing = {"undefined_object@q0": None}
    assert book.after_round(card_id="card-0000000a", stage="R", loop="refinement", round_id="r0",
                            failures=failing) is None
    rebuilt = lineage.BudgetBook(load_run_config(ROOT / "hybrid" / "config" / "run.toml"), ledger, book.store,
                                 run_id=book.run_id)
    assert rebuilt.used("card-0000000a", "refinement") == 1
    # the same answer to the same round is a stall now that one revision was spent on it
    stop = rebuilt.after_round(card_id="card-0000000a", stage="R", loop="refinement", round_id="r1",
                               failures=failing)
    assert stop.kill_code == "stall:refinement"


# ----- repair scope (plan section 10: a repair may change only what its failure names) --------------------------------

def test_repair_scope_names_changed_added_and_removed_ids_outside_the_failure():
    before = {"O1": "n + 0 = n", "O2": "cost <= 4 n", "O3": "rank <= d"}
    after = {"O1": "n + 0 = n", "O2": "cost <= 3 n", "O3": "rank <= d + 1", "O4": "new"}
    assert lineage.repair_scope(before, after, ["O2"]) == ["O3", "O4"]
    assert lineage.repair_scope(before, {k: v for k, v in before.items() if k != "O1"}, ["O2"]) == ["O1"]
    assert lineage.repair_scope(before, dict(before, O2="cost <= 3 n"), ["O2"]) == []


def test_a_reviser_that_keeps_touching_an_unnamed_obligation_is_rejected_then_stalls(tmp_path):
    """Stage V's use: the repair is checked before any check runs; a rejected repair uses a round, and the same
    rejected repair again is a stall."""
    book, ledger = book_for(tmp_path)
    book.assign(make_card("card-0000000a"), op="new", parents=[])
    failing = {"obligation_failed:P2": {"counterexample": {"n": 7}}}
    before = {"P1": "a", "P2": "b"}
    outcomes = []
    for k in range(5):
        repaired = {"P1": "a changed", "P2": f"b{k}"}          # touches P1, which the failure does not name
        outside = lineage.repair_scope(before, repaired, ["P2"])
        assert outside == ["P1"]
        counted = lineage.reject_repair(ledger, card_id="card-0000000a", stage="V", loop="math_revision",
                                        round_id=f"v{k}", outside=outside, failures=failing)
        stop = book.after_round(card_id="card-0000000a", stage="V", loop="math_revision", round_id=f"v{k}",
                                failures=counted)
        outcomes.append(stop and stop.kill_code)
        if stop:
            break
    assert outcomes == [None, "stall:math_revision"]
    rejected = [e["data"] for e in ledger.iter_events("repair_rejected")]
    assert [r["outside"] for r in rejected] == [["P1"], ["P1"]]
    metrics = budget_metrics(list(ledger.iter_events()))
    assert len(metrics["repairs_rejected"]) == 2
    from hybrid.research.inspect import render_budgets
    text = "\n".join(render_budgets(metrics))
    assert "Repairs rejected without running checks: 2" in text and "touched P1" in text
    with pytest.raises(ValueError):
        lineage.reject_repair(ledger, card_id="card-0000000a", stage="V", loop="math_revision", round_id="v9",
                              outside=[], failures=failing)


# ----- a later phase's stage under the orchestrator: kill codes reach the ledger and the outcome store -------------------

def refinement_stage(loop):
    """A stand-in for Phase 7's dialogue stage, put in S1's place: the salad-critic role plays the challenger, each
    fixable finding is a question, the dialogue ends when no question is left or the budget says so."""
    from hybrid.research.schemas import load_schema
    from hybrid.research.stages.common import card_for_prompt

    async def stage(ctx, card_id):
        k = 0
        while True:
            reply = await ctx.ask("salad_critic", "salad_critic", card_for_prompt(ctx.card(card_id)),
                                  load_schema("salad_critic"))
            questions = {f"{q['kind']}@{q['where']}": None for q in reply.parsed["fixable"]}
            if not questions:
                ctx.verdict(card_id, "S1", "pass", ["dialogue: no questions left"], [reply.call_id])
                return "S2"
            stop = ctx.book.after_round(card_id=card_id, stage="S1", loop=loop, round_id=f"round-{k}",
                                        failures=questions)
            if stop is not None:
                ctx.verdict(card_id, "S1", "fail", [f"gate: fixable questions remain; {stop.line()}"], [reply.call_id])
                return None
            k += 1
    return stage


@pytest.mark.parametrize("challenger,code,asked", [
    (always_new_critic, "budget:refinement", 4),
    (alternating_critic, "budget:refinement", 4),
    (SAME_CRITIC, "stall:refinement", 2),
])
def test_a_plugged_in_dialogue_ends_under_adversarial_challengers_with_its_kill_code(run_dir, tmp_path, monkeypatch,
                                                                                     challenger, code, asked):
    from hybrid.research.stages import STAGE_RUNNERS

    monkeypatch.setitem(STAGE_RUNNERS, "S1", refinement_stage("refinement"))
    cfg = with_store(config(), tmp_path / "outcomes")
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"salad_critic": challenger})
    assert run(orch)["reason"] == "complete"
    assert llm.prompts().count("salad_critic") == asked
    [stop] = events(orch, "loop_stop")
    assert stop["kill_code"] == code and stop["stage"] == "S1"
    record = own_records(OutcomeStore(tmp_path / "outcomes"), run_dir.name)[orch.store.list_cards()[0]]
    assert record["outcome"] == "dead" and code in record["kill_codes"]
    assert record["budget"]["loops"] == {"refinement": asked - 1 if code.startswith("budget") else 1}
