"""The S0 diversity check for escape cards: the parent and its escape ancestors are not compared with the new card,
every other card is, and mutate cards get the unchanged check. Scripted LLM, LocalExecutor, no live calls.
"""
import asyncio
import json

from hybrid.research.ledger import Ledger
from hybrid.research.stages.common import ESCAPE_FIELD
from hybrid.research.stages.escape import escape_chain
from hybrid.research.stages.s0_explore import ExploreTask, explore
from hybrid.tests.research.test_escape import ARGUMENT, REDUCED, run_dir  # noqa: F401
from hybrid.tests.research.test_orchestrator_fakes import draft, make_orchestrator, orch_config


def card_text_by_call(fixture_indexes):
    """Explorer output for the n-th explorer call: the fixture draft with index fixture_indexes[n] (the last
    index repeats). Escape drafts carry the non-reduction argument."""
    def make(n, user):
        card = draft(fixture_indexes[min(n, len(fixture_indexes) - 1)], user=user)
        if "OPERATOR: escape" in user:
            card[ESCAPE_FIELD] = ARGUMENT
        return card
    return make


def similarity_orchestrator(run_dir, fixture_indexes):
    cfg = orch_config()
    cfg["diversity"]["threshold"] = 0.6
    orch, _ = make_orchestrator(run_dir, cfg=cfg, script={"explorer": card_text_by_call(fixture_indexes)})
    # A real run writes run_start before its first explore; the graveyard reads the ledger from there.
    orch.ledger.append("run_start", "orchestrator", {"run_id": orch.run_id, "resume": False})
    return orch


def explore_once(orch, op, parents, seed):
    """One S0 explore task, run to its verdict. Returns the new card id, or None when it was a duplicate."""
    lens = orch.lens_order[0]
    task = ExploreTask(op=op, lens=lens, lens_sources=orch.lenses[lens]["sources"], role="explorer", explorer=0,
                       seed=seed, parents=parents)
    return asyncio.run(explore(orch.ctx, task))


def created_ids(orch):
    """Card ids in the order their card_created events were written, duplicates included."""
    return [event["data"]["card_id"] for event in orch.ledger.iter_events("card_created")]


def s0_verdict(orch, card_id):
    return orch.store.get_card(card_id)["status"]["verdicts"]["S0"]


def mark_reduced(orch, card_id):
    """The S2 reduction critic output that an escape task reads for its parent."""
    orch.store.put_artifact("reduction_critic", f"{card_id}.json", json.dumps(REDUCED).encode(), card_id=card_id)


def test_escape_chain_walks_consecutive_escape_steps(tmp_path):
    ledger = Ledger(tmp_path / "run")
    for card_id, op, parents in [("card-q", "new", []), ("card-p", "escape", ["card-q"]),
                                 ("card-x", "escape", ["card-p"]), ("card-m", "mutate", ["card-x"])]:
        ledger.append("card_created", "orchestrator", {"card_id": card_id, "op": op, "parents": parents})
    assert escape_chain(ledger, "card-x") == ["card-x", "card-p", "card-q"]
    assert escape_chain(ledger, "card-p") == ["card-p", "card-q"]
    assert escape_chain(ledger, "card-q") == ["card-q"]
    assert escape_chain(ledger, "card-m") == ["card-m"]      # a mutate card is not an escape step


def test_escape_child_identical_to_its_parent_passes_s0(run_dir):
    orch = similarity_orchestrator(run_dir, [0])
    parent = explore_once(orch, "new", [], seed=1)
    mark_reduced(orch, parent)
    child = explore_once(orch, "escape", [parent], seed=2)
    assert child is not None
    verdict = s0_verdict(orch, child)
    assert verdict["verdict"] == "pass"
    assert f"escape: parent and escape ancestors excluded from the nearest-card comparison: {parent}" \
        in verdict["reasons"]
    assert "nearest existing card - at TF-IDF cosine 0.000 (threshold 0.6)" in verdict["reasons"]


def test_escape_chain_excludes_the_ancestors_too(run_dir):
    orch = similarity_orchestrator(run_dir, [0])
    parent = explore_once(orch, "new", [], seed=1)
    mark_reduced(orch, parent)
    child = explore_once(orch, "escape", [parent], seed=2)
    mark_reduced(orch, child)
    grandchild = explore_once(orch, "escape", [child], seed=3)
    assert grandchild is not None
    assert (f"escape: parent and escape ancestors excluded from the nearest-card comparison: {child}, {parent}"
            in s0_verdict(orch, grandchild)["reasons"])


def test_mutate_child_identical_to_its_parent_is_still_a_duplicate(run_dir):
    orch = similarity_orchestrator(run_dir, [0])
    parent = explore_once(orch, "new", [], seed=1)
    assert explore_once(orch, "mutate", [parent], seed=2) is None
    verdict = s0_verdict(orch, created_ids(orch)[1])
    assert verdict["verdict"] == "fail"
    assert verdict["reasons"][0].startswith("duplicate: TF-IDF cosine 1.000 with ")
    assert not any("excluded" in reason for reason in verdict["reasons"])


def test_escape_child_is_still_compared_with_cards_off_the_chain(run_dir):
    # Card 0 is the parent (fixture 0). Card 1 is an unrelated card (fixture 5). The escape child repeats card 1's
    # text. Excluding the parent does not excuse the child from matching card 1.
    orch = similarity_orchestrator(run_dir, [0, 5, 5])
    parent = explore_once(orch, "new", [], seed=1)
    explore_once(orch, "new", [], seed=2)
    other = created_ids(orch)[1]
    mark_reduced(orch, parent)
    assert explore_once(orch, "escape", [parent], seed=3) is None
    verdict = s0_verdict(orch, created_ids(orch)[2])
    assert verdict["verdict"] == "fail"
    assert verdict["reasons"][0].startswith(f"duplicate: TF-IDF cosine 1.000 with {other} ")
    assert f"escape: parent and escape ancestors excluded from the nearest-card comparison: {parent}" \
        in verdict["reasons"]
