"""KD6: a run that drains with cards stopped on a stage_error ends with the resumable reason `stage_errors`, lists
the cards, writes no outcome record for them, and --resume requeues them."""
import asyncio

import pytest

from hybrid.llm.errors import TransportError
from hybrid.research.inspect import inspect_run
from hybrid.research.ledger import Ledger, unresolved_stage_errors
from hybrid.tests.research.test_orchestrator_fakes import analyst_supports_every_claim, make_orchestrator


def run(orch):
    return asyncio.run(orch.run())


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261017T000000Z"


def test_a_stage_error_ends_the_run_resumable_and_lists_the_card(run_dir, capsys):
    orch, _ = make_orchestrator(run_dir, fail={"salad_critic": TransportError("502 failed in transit")})
    summary = run(orch)
    card_id = orch.store.list_cards()[0]
    assert summary["reason"] == "stage_errors"
    assert [(e["card_id"], e["stage"]) for e in summary["stage_errors"]] == [(card_id, "S1")]
    assert "TransportError" in summary["stage_errors"][0]["error"]
    assert events(orch, "run_stop")[0]["reason"] == "stage_errors"
    end = events(orch, "run_end")[0]
    assert end["reason"] == "stage_errors" and end["stage_errors"] == summary["stage_errors"]
    # A transport failure is not a research verdict: no verdict, no kill code.
    assert [v["stage"] for v in events(orch, "verdict")] == ["S0"]
    assert orch.store.get_card(card_id)["status"]["stage"] == "S0"
    assert not events(orch, "outcome_recorded")
    # inspect names the card and says it is resumable
    assert inspect_run(run_dir) == 0
    out = capsys.readouterr().out
    assert "Stage errors without a later verdict: 1" in out and f"{card_id} at S1" in out


def test_a_clean_run_still_ends_complete_without_a_stage_errors_key(run_dir, capsys):
    orch, _ = make_orchestrator(run_dir, script={"analyst": analyst_supports_every_claim})
    summary = run(orch)
    assert summary["reason"] == "complete"
    assert "stage_errors" not in summary
    assert not events(orch, "stage_error")
    inspect_run(run_dir)
    assert "Stage errors" not in capsys.readouterr().out


def test_resume_requeues_the_card_and_the_error_is_resolved_by_its_verdict(run_dir):
    orch, _ = make_orchestrator(run_dir, fail={"salad_critic": TransportError("502 failed in transit")})
    assert run(orch)["reason"] == "stage_errors"
    card_id = orch.store.list_cards()[0]

    orch2, _ = make_orchestrator(run_dir, resume=True, script={"analyst": analyst_supports_every_claim})
    second = run(orch2)
    assert second["requeued_at_start"] == {"S1": [card_id]}
    assert second["reason"] == "complete" and "stage_errors" not in second
    assert orch2.store.get_card(card_id)["status"]["stage"] == "S6"
    assert unresolved_stage_errors(Ledger(run_dir).iter_events()) == []


def test_only_a_verdict_after_the_latest_error_resolves_it():
    def event(type, **data):
        return {"type": type, "data": data}

    log = [event("stage_error", card_id="a", stage="S1", error="x"), event("verdict", card_id="a", stage="S1"),
           event("stage_error", card_id="a", stage="S2", error="y"),
           event("stage_error", card_id="b", stage="S1", error="z"), event("verdict", card_id="c", stage="S1")]
    assert unresolved_stage_errors(log) == [{"card_id": "a", "stage": "S2", "error": "y"},
                                            {"card_id": "b", "stage": "S1", "error": "z"}]
