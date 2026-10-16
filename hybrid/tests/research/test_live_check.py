"""The Phase 7 live-check driver (hybrid/research/live_check.py) with fake models: cards copied from an earlier run
enter at S0, run S1, R and S2, and stop before the next stage; the source run and the source store are not written."""
import asyncio
import hashlib
from pathlib import Path

import pytest

from hybrid.bench.local import LocalExecutor
from hybrid.executor.lock import read_lock
from hybrid.research import live_check
from hybrid.research.ledger import Ledger
from hybrid.research.outcomes import JSONL, OutcomeStore
from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.store import Store
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.outcome_fixtures import build_run
from hybrid.tests.research.test_difference import DEAD_CRITIC, RENAMED_LA, checker, critic, statement
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, make_classifier, orch_config

SOURCE_MODEL = "claude-opus-5-5"


def fingerprint(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def source(tmp_path):
    """An earlier run with two cards that passed S2, drafted by a Claude model under the explorer_alt role."""
    run_dir = tmp_path / "runs" / "run-20261010T181300Z"
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    for card_id, title in (("card-0000a001", "Phase Ring Memory"), ("card-0000a002", "Parity Gate Ledger")):
        card = make_card(card_id, title=title)
        card["status"] = {"stage": "S0", "verdicts": {}}
        store.put_card(card)
        ledger.append("card_created", "orchestrator", {"card_id": card_id, "op": "new", "parents": [],
                                                       "role": "explorer_alt", "model": SOURCE_MODEL})
        for stage in ("S0", "S1", "S2"):
            store.set_verdict(card_id, stage, "pass", ["ok"], ["call-0123456789ab"])
    return run_dir


@pytest.fixture
def stores(tmp_path, source):
    """The source store (one dead record of the source run) and a fresh copy of it."""
    live = OutcomeStore(tmp_path / "outcomes")
    dead = make_card("card-0000d001", title="Phase Ring Memory")
    other = build_run(tmp_path / "runs" / "run-20260901T000000Z", [(dead, [
        ("S0", "pass", ["ok"]), ("S1", "pass", ["ok"]),
        ("S2", "reduces_to_known", ["gate: update rule reduces to other_known: m"])],
        {"reduction_critic": DEAD_CRITIC})])
    live.append(build_record(RunView(other), "card-0000d001", "backfill"))
    copy = live_check.copy_store(live.dir, tmp_path / "store-copy")
    return live, OutcomeStore(copy)


def make(tmp_path, source, store_copy, script, card_ids=("card-0000a001", "card-0000a002")):
    cfg = orch_config()
    cfg["refine"]["enabled"] = True
    cfg["s2"].update(family_exclusion=True, difference_statement=True)
    cfg["outcomes"]["enforce_r3"] = True
    cfg["workers"]["R"] = 1
    cfg = live_check.live_config(cfg, store_path=store_copy.dir, n_cards=len(card_ids), max_wall_s=600)
    run_dir = tmp_path / "runs" / "phase7-live-20261014T1800Z"
    ledger = Ledger(run_dir)
    llm = FakeLLM(run_dir, ledger, script, models={"explorer": SOURCE_MODEL, "refine_challenger": "gpt-6.1-sol"})
    orch = live_check.LiveCheckOrchestrator(cfg, run_dir, source_run=source, card_ids=list(card_ids), role="explorer",
                                            llm=llm, executor=LocalExecutor(run_dir.name, run_dir / "jobs" / "receipts"),
                                            bench=None, image=read_lock()["reference"], ledger=ledger,
                                            classify_prior_art=make_classifier("novel"))
    return orch, llm


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


def converged(i, user):
    return {"questions": [], "notes": "nothing left open"}


def test_copied_cards_run_s1_r_and_s2_and_stop_before_the_next_stage(tmp_path, source, stores):
    live, copy = stores
    before_source, before_live = fingerprint(source), fingerprint(live.dir)
    script = {"reduction_critic": critic(), "difference": statement(), "difference_check": checker(),
              "refine_challenger": converged}
    orch, llm = make(tmp_path, source, copy, script)
    summary = asyncio.run(orch.run())
    assert summary["reason"] == "complete"
    results = live_check.card_results(orch)
    assert sorted(results) == ["card-0000a001", "card-0000a002"]
    for card_id, result in results.items():
        assert {s: v["verdict"] for s, v in result["verdicts"].items()} == {"S0": "pass", "S1": "pass", "R": "pass",
                                                                           "S2": "pass"}
        assert result["stopped_before"] == "S3"
    assert not [e for e in events(orch, "stage_start") if e["stage"] not in ("S1", "R", "S2")]
    created = {e["card_id"]: e for e in events(orch, "card_created")}
    assert all(e["model"] == SOURCE_MODEL and e["role"] == "explorer" and e["parents"] == [] for e in created.values())
    # The challenger is chosen from the drafting model (Claude), and the proposer of the statement is the explorer role.
    assert {c["role"] for c in llm.calls if c["prompt"] == "refine_challenger"} == {"refine_challenger"}
    assert {c["role"] for c in llm.calls if c["prompt"] == "difference"} == {"explorer"}
    # Exact signature match with the dead record: R3 applies, and the statement cites its causes.
    judged = events(orch, "difference_judged")
    assert len(judged) == 2 and all(j["accepted"] and j["r3_targets"] for j in judged)
    # Alive cards get no outcome record; the source run and the source store are untouched.
    assert [s.record["card_id"] for s in copy.current()] == ["card-0000d001"]
    assert fingerprint(source) == before_source and fingerprint(live.dir) == before_live
    assert len(events(orch, "live_check_admit")) == 2 and len(events(orch, "live_check_stopped")) == 2


def test_a_card_killed_at_s2_is_recorded_in_the_store_copy_only(tmp_path, source, stores):
    live, copy = stores
    before_live = fingerprint(live.dir)
    script = {"reduction_critic": RENAMED_LA, "difference": statement(), "difference_check": checker(),
              "refine_challenger": converged}
    orch, llm = make(tmp_path, source, copy, script, card_ids=("card-0000a001",))
    asyncio.run(orch.run())
    result = live_check.card_results(orch)["card-0000a001"]
    assert result["verdicts"]["S2"]["verdict"] == "reduces_to_known"
    assert result["verdicts"]["S2"]["gate"].startswith("gate: excluded_family:linear_attention:")
    assert result["stopped_before"] is None
    mine = [s.record for s in OutcomeStore(copy.dir).current() if s.record["run"] == orch.run_id]
    assert [r["kill_codes"] for r in mine] == [["excluded_family:linear_attention"]]
    assert fingerprint(live.dir) == before_live


def test_a_store_copy_must_be_new(tmp_path, stores):
    live, copy = stores
    with pytest.raises(live_check.LiveCheckError):
        live_check.copy_store(live.dir, copy.dir)
    assert (copy.dir / JSONL).is_file()
