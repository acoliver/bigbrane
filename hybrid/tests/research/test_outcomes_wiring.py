"""The outcome store in the loop (outcomes.enabled): S0 lists nearby dead ideas, S2 searches own history and can
enforce R3, the orchestrator records every card whose line ends. With the switch off nothing changes, and runs
started without the switch resume."""
import asyncio

import pytest

from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.outcomes import OutcomeStore
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.outcome_fixtures import build_run
from hybrid.tests.research.test_orchestrator_fakes import draft, make_orchestrator, orch_config

EARLIER_RUN = "run-20260901T000000Z"
CRITIC = {"reduces_to": "other_known", "mapping": "S = sum v exp(i theta)", "confidence": 0.9,
          "difference_if_any": "", "radical_enough": False,
          "why": "It is a phase-keyed holographic sum, which is diagonal linear attention.",
          "textbook_transplant": True, "textbook_why": "-"}


def run(orch):
    return asyncio.run(orch.run())


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


def calls(llm, prompt):
    return [c for c in llm.calls if c["prompt"] == prompt]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261010T000000Z"


@pytest.fixture
def store(tmp_path):
    """A store holding one dead idea from an earlier run whose mechanism is the fixture card's, so the explorer's
    draft (the same mechanism under another title) matches its signature exactly."""
    earlier = make_card("card-0000d001", title="Phase Ring Memory")
    view = RunView(build_run(tmp_path / "runs" / EARLIER_RUN, [(earlier, [
        ("S0", "pass", ["ok"]), ("S1", "pass", ["ok"]),
        ("S2", "reduces_to_known", ["gate: update rule reduces to other_known: m"])], {"reduction_critic": CRITIC})]))
    store = OutcomeStore(tmp_path / "outcomes")
    store.append(build_record(view, "card-0000d001", "backfill"))
    return store


def enabled(store, **overrides):
    cfg = orch_config()
    cfg["outcomes"].update(enabled=True, path=str(store.dir), **overrides)
    return cfg


def test_switch_off_leaves_prompts_and_records_unchanged(run_dir, tmp_path):
    orch, llm = make_orchestrator(run_dir)
    assert orch.ctx.outcomes is None
    run(orch)
    assert "OUTCOME STORE" not in calls(llm, "explorer")[0]["user"]
    assert "OWN HISTORY" not in calls(llm, "reduction_critic")[0]["user"]
    assert not [a for a in events(orch, "artifact") if a["kind"] == "own_history"]
    assert events(orch, "outcome_recorded") == [] and events(orch, "own_history") == []
    assert "outcome_store_nearby" not in events(orch, "card_created")[0]
    assert events(orch, "run_start")[0]["outcomes"]["enabled"] is False


def test_switch_on_shows_dead_ideas_to_s0_and_own_history_to_s2_and_records_the_outcome(run_dir, store):
    orch, llm = make_orchestrator(run_dir, cfg=enabled(store))
    summary = run(orch)
    assert summary["reason"] == "complete"
    dead = store.current()[0]
    cause = dead.causes()[0]["id"]
    explorer_user = calls(llm, "explorer")[0]["user"]
    assert "OUTCOME STORE" in explorer_user and cause in explorer_user and "Phase Ring Memory" in explorer_user
    assert events(orch, "card_created")[0]["outcome_store_nearby"] == [dead.record_id]
    critic_user = calls(llm, "reduction_critic")[0]["user"]
    assert "OWN HISTORY" in critic_user and "EXACT signature" in critic_user and cause in critic_user
    history = events(orch, "own_history")[0]
    assert history["exact"] == [dead.record_id] and history["near"] == []
    card = orch.store.get_card(orch.store.list_cards()[0])
    assert any(r.startswith("own history: 1 exact") for r in card["status"]["verdicts"]["S2"]["reasons"])
    assert card["status"]["stage"] == "S6"                       # enforce_r3 is off: the match is shown, not gated
    recorded = events(orch, "outcome_recorded")
    assert [r["card_id"] for r in recorded] == [card["id"]] and recorded[0]["outcome"] == "finished"
    mine = [s.record for s in store.current() if s.record["run"] == run_dir.name]
    assert len(mine) == 1 and mine[0]["written_by"] == "orchestrator"
    assert mine[0]["mechanism_signature"]["signature"] == dead.record["mechanism_signature"]["signature"]
    assert store.verify() == (True, None)


def test_enforced_r3_kills_an_exact_match_that_names_no_cause(run_dir, store):
    orch, llm = make_orchestrator(run_dir, cfg=enabled(store, enforce_r3=True))
    run(orch)
    card = orch.store.get_card(orch.store.list_cards()[0])
    s2 = card["status"]["verdicts"]["S2"]
    assert s2["verdict"] == "fail" and any(r.startswith("gate: not_radical:R3") for r in s2["reasons"])
    assert card["status"]["stage"] == "S2" and not calls(llm, "formalizer")
    mine = [s.record for s in store.current() if s.record["run"] == run_dir.name][0]
    assert mine["kill_codes"] == ["not_radical:R3"] and mine["outcome"] == "dead"


def test_enforced_r3_lets_a_card_with_no_exact_match_through(run_dir, tmp_path):
    empty = OutcomeStore(tmp_path / "empty")
    orch, _ = make_orchestrator(run_dir, cfg=enabled(empty, enforce_r3=True))
    run(orch)
    card = orch.store.get_card(orch.store.list_cards()[0])
    assert card["status"]["verdicts"]["S2"]["verdict"] == "pass"
    assert "OUTCOME STORE" in orch.ctx.llm.calls[0]["user"]       # the block is there, with "(none)"


def test_a_duplicate_is_recorded_with_its_kill_code(run_dir, store):
    cfg = enabled(store)
    cfg["budgets"]["max_cards"] = 2
    cfg["diversity"]["threshold"] = 0.6
    orch, _ = make_orchestrator(run_dir, cfg=cfg,
                                script={"explorer": lambda i, user: draft(0, distinct=False, user=user)})
    run(orch)
    duplicate = events(orch, "duplicate")[0]["card_id"]
    mine = {s.record["card_id"]: s.record for s in store.current() if s.record["run"] == run_dir.name}
    assert mine[duplicate]["kill_codes"][0].startswith("duplicate:card-")


def test_a_run_started_without_the_store_resumes_with_it_on_and_off(run_dir, store):
    first = orch_config()
    first["budgets"]["max_calls"] = 1                         # the explorer call only: the card stops at S0
    orch, _ = make_orchestrator(run_dir, cfg=first)
    run(orch)
    card_id = orch.store.list_cards()[0]
    assert orch.store.get_card(card_id)["status"]["stage"] == "S0"
    off, _ = make_orchestrator(run_dir, cfg=orch_config(budgets__max_calls=2), resume=True)
    run(off)
    assert events(off, "outcome_recorded") == []
    on = enabled(store)
    on["budgets"]["max_calls"] = 200
    resumed, llm = make_orchestrator(run_dir, cfg=on, resume=True)
    summary = run(resumed)
    assert summary["reason"] == "complete"
    assert resumed.store.get_card(card_id)["status"]["stage"] == "S6"
    assert events(resumed, "run_start")[-1]["outcomes"]["enabled"] is True
    assert [r["card_id"] for r in events(resumed, "outcome_recorded")] == [card_id]


def test_config_defaults_and_validation(tmp_path):
    old = load_run_config(ROOT / "hybrid" / "config" / "run3.toml")
    assert old["outcomes"]["enabled"] is False and old["outcomes"]["path"] == "results/outcomes"
    shipped = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    # Phase 7 turned R3 on for new runs (it acts once outcomes.enabled gives it a store); run3.toml predates it.
    assert shipped["outcomes"]["enabled"] is False and shipped["outcomes"]["enforce_r3"] is True
    assert old["outcomes"]["enforce_r3"] is False
    for key, bad in (("enabled", "yes"), ("enforce_r3", 1), ("jaccard_min", 0), ("tfidf_min", 1.5),
                     ("history_max", -1), ("path", "")):
        with pytest.raises(ValueError, match=f"outcomes.{key}"):
            load_run_config(ROOT / "hybrid" / "config" / "run.toml", {f"outcomes.{key}": bad})
    text = (ROOT / "hybrid" / "config" / "run.toml").read_text()
    path = tmp_path / "run.toml"
    path.write_text(text.replace("[outcomes]\n", "[outcomes]\nsurprise = 1\n"))
    with pytest.raises(ValueError, match="unknown \\[outcomes\\] keys"):
        load_run_config(path)
