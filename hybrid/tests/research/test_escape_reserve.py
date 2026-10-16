"""The escape reserve (explore.escape_reserve): which card slots new, mutate and cross tasks may take, how an
escape request is judged against the free slots, how an explorer waits on the reserve, and what a resume uses.

The slot arithmetic is tested on the orchestrator's state directly. The integrated runs use one explorer, so
the card counts follow from the rule and do not depend on timing. Scripted LLM, LocalExecutor, no live calls.
"""
import asyncio

import pytest

from hybrid.research.orchestrator import ROOT, escape_reserve_slots, load_run_config
from hybrid.tests.research.test_escape import NOT_REDUCED, REDUCED, explorer, reduced_run, run_dir  # noqa: F401
from hybrid.tests.research.test_orchestrator_fakes import make_orchestrator, orch_config

CARD = "card-aaaaaaaa"


def run(orch):
    return asyncio.run(orch.run())


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


def reserve_config(reserve, **overrides):
    """A fake-run config with the reserve set and no mutate or cross draws, so the explorer's choices are fixed."""
    cfg = orch_config(**overrides)
    cfg["explore"]["escape_reserve"] = reserve
    cfg["explore"]["p_mutate"] = 0.0
    cfg["explore"]["p_cross"] = 0.0
    return cfg


def queues_only(orch) -> None:
    """The queues and wakeup that run() creates, for tests that drive the slot rules without running."""
    orch.mutations = asyncio.Queue()
    orch.escapes = asyncio.Queue()
    orch.slot_wakeup = asyncio.Event()


@pytest.mark.parametrize("max_cards,fraction,slots", [
    (12, 0.3, 4), (10, 0.3, 3), (1, 0.3, 1), (7, 0.5, 4), (12, 0.0, 0), (12, 1.0, 12),
])
def test_reserve_is_the_fraction_of_max_cards_rounded_up(max_cards, fraction, slots):
    assert escape_reserve_slots(max_cards, fraction) == slots


def test_normal_tasks_leave_the_unclaimed_reserve_and_queued_escapes_free(run_dir):
    orch, _ = make_orchestrator(run_dir, cfg=reserve_config(0.3, budgets__max_cards=10))
    queues_only(orch)
    orch.budget.cards = 6                                 # 4 slots free, 3 held for escapes
    assert orch._reserve_slots() == 3 and orch._escape_held() == 3
    assert orch._normal_slot_free()                       # a normal task leaves 3 free
    orch.budget.cards = 7                                 # 3 slots free
    assert not orch._normal_slot_free()                   # a normal task would leave 2 for 3 held

    orch.budget.cards = 6
    orch._request_escape(CARD)                            # the escape claims one reserved slot and queues
    assert orch.escape_claims == 1 and orch.escapes.qsize() == 1
    assert orch._escape_held() == 3                       # the claim moves from the reserve to the queue
    assert orch._normal_slot_free()

    orch.escapes.get_nowait()                             # the escape starts
    orch.budget.cards = 7
    assert orch._escape_held() == 2 and orch._normal_slot_free()
    orch.budget.cards = 8
    assert not orch._normal_slot_free()


def test_escape_takes_the_last_free_slot_and_skips_record_the_reserve(run_dir):
    orch, _ = make_orchestrator(run_dir, cfg=reserve_config(0.3, budgets__max_cards=10))
    queues_only(orch)
    orch.budget.cards = 9                                 # one slot free, all of it reserved
    assert not orch._normal_slot_free()
    orch._request_escape(CARD)
    assert orch.escapes.qsize() == 1 and orch.escape_claims == 1
    orch._request_escape(CARD)                            # the only free slot is already queued for an escape
    assert orch.escapes.qsize() == 1 and orch.escape_claims == 1
    skipped = events(orch, "escape_skipped")
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "no card slot left (9 of 10 used, 1 escapes queued)"
    assert skipped[0]["reserve"] == {"slots": 3, "claimed": 1, "queued": 1, "held": 3, "free": 1}


def test_explorer_waits_for_the_reduction_and_the_escape_takes_the_reserved_slot(run_dir):
    # max_cards 4 with a reserve of 2: two new cards start, then the explorer waits. The first card that
    # reduces at S2 queues an escape, which the explorer takes. Exactly three cards in every ordering.
    cfg = reserve_config(0.5, budgets__max_cards=4)
    orch, _ = make_orchestrator(run_dir, cfg=cfg,
                                script={"explorer": explorer(0), "reduction_critic": [REDUCED, NOT_REDUCED]})
    summary = run(orch)
    assert summary["reason"] == "complete"
    operators = [created["op"] for created in events(orch, "card_created")]
    assert len(operators) == 3 and operators.count("escape") == 1
    assert events(orch, "escape_skipped") == []
    assert len(events(orch, "escape_requested")) == 1
    assert events(orch, "run_start")[0]["escape_reserve"] == {"fraction": 0.5, "slots": 2, "max_cards": 4,
                                                              "claimed": 0}
    assert summary["budget"]["cards"] == 3


def test_explorers_stop_when_the_reserve_blocks_them_and_nothing_is_in_flight(run_dir):
    # No card reduces, so no escape can arrive. Two new cards start; the third would leave one slot for a
    # reserve of two. The blocked explorer waits for the two cards to leave the pipeline, then stops.
    orch, _ = make_orchestrator(run_dir, cfg=reserve_config(0.5, budgets__max_cards=4))
    summary = run(orch)
    assert summary["reason"] == "complete" and summary["budget"]["cards"] == 2
    assert len(orch.store.list_cards()) == 2
    assert events(orch, "escape_requested") == []


def test_resume_computes_the_reserve_against_the_new_max_cards(run_dir):
    first, _ = make_orchestrator(run_dir, cfg=reserve_config(0.5, budgets__max_cards=4))
    run(first)
    assert len(first.store.list_cards()) == 2
    # 0.5 of 10 is a reserve of 5: a normal task starts while at least 6 slots are free, so 5 cards in all.
    second, _ = make_orchestrator(run_dir, cfg=reserve_config(0.5, budgets__max_cards=10), resume=True)
    summary = run(second)
    assert events(second, "run_start")[-1]["escape_reserve"] == {"fraction": 0.5, "slots": 5, "max_cards": 10,
                                                                 "claimed": 0}
    # The budget counts attempts. One of the five failed before it made a card (a pairing lens with two papers),
    # and it still used a slot.
    assert summary["budget"]["cards"] == 5
    assert len(second.store.list_cards()) + len(events(second, "explore_failed")) == 5


def test_resume_counts_the_escape_tasks_the_ledger_already_started(run_dir):
    reduced_run(run_dir)                                  # max_cards 1: the parent's escape finds no slot
    second, _ = make_orchestrator(run_dir, cfg=orch_config(budgets__max_cards=2), resume=True,
                                  script={"explorer": explorer(30), "reduction_critic": NOT_REDUCED})
    run(second)
    assert [c["op"] for c in events(second, "card_created")].count("escape") == 1
    third, _ = make_orchestrator(run_dir, cfg=reserve_config(0.5, budgets__max_cards=4), resume=True)
    third._restore_budget()
    assert third.escape_claims == 1


def test_escape_reserve_defaults_to_0_3_and_is_validated(tmp_path):
    shipped = ROOT / "hybrid" / "config" / "run.toml"
    assert load_run_config(shipped)["explore"]["escape_reserve"] == 0.3
    assert load_run_config(ROOT / "hybrid" / "config" / "run2.toml")["explore"]["escape_reserve"] == 0.3
    text = shipped.read_text(encoding="utf-8")
    old = tmp_path / "old.toml"
    old.write_text("\n".join(line for line in text.splitlines() if not line.startswith("escape_reserve")),
                   encoding="utf-8")
    assert load_run_config(old)["explore"]["escape_reserve"] == 0.3     # a config from before the key existed
    for bad in (1.5, -0.1, True, "0.3"):
        with pytest.raises(ValueError, match="explore.escape_reserve must be a number in"):
            load_run_config(shipped, {"explore.escape_reserve": bad})
    for good in (0, 1, 0.0, 1.0):
        assert load_run_config(shipped, {"explore.escape_reserve": good})["explore"]["escape_reserve"] == good
