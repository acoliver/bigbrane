"""The outcome store: append-only JSONL with a hash chain, an SQLite index that equals a rebuild from the JSONL,
kill-code validation, and the step 1 and step 3 queries."""
from hybrid.tests.research.critic_answers import KNOWN, NOT_KNOWN
import json

import pytest

from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.outcomes import (OutcomeStore, OutcomeStoreError, cause_id, check_kill_code, nearby_dead,
                                      own_history, r3_problems, render_history, render_nearby)
from hybrid.research.schemas import SchemaViolation
from hybrid.research.signature import card_signature
from hybrid.tests.research.outcome_fixtures import KNOWN_MECHANISMS, build_run, renamed

LATER = "run-20261013T000000Z"
REDUCED = ["gate: update rule reduces to linear_attention: v k^T accumulation"]
CRITIC = {"reduces_to": "linear_attention", "mapping": "S = sum v k^T", "confidence": 0.9, "difference_if_any": "",
          **KNOWN, "why": "This is not new. It is diagonal linear attention with a phase map.",
          "textbook_transplant": False, "textbook_why": "-"}


def record_for(tmp_path, card, verdicts, artifacts=None, run="run-20261001T000000Z"):
    view = RunView(build_run(tmp_path / run, [(card, verdicts, artifacts or {})]))
    return build_record(view, card["id"], "backfill")


def reduced(tmp_path, card, run="run-20261001T000000Z"):
    return record_for(tmp_path, card, [("S0", "pass", ["ok"]), ("S1", "pass", ["ok"]), ("S2", "reduces_to_known",
                                                                                       REDUCED)],
                      {"reduction_critic": CRITIC}, run)


@pytest.fixture
def store(tmp_path):
    return OutcomeStore(tmp_path / "outcomes")


def test_append_chains_hashes_and_skips_a_record_already_stored(tmp_path, store):
    record = reduced(tmp_path, KNOWN_MECHANISMS["linear_attention"])
    first = store.append(record)
    assert first["seq"] == 1 and first["prev_hash"] == "0" * 64
    assert store.append(record) is None
    other = reduced(tmp_path, KNOWN_MECHANISMS["delta_rule"], run="run-20261002T000000Z")
    second = store.append(other)
    assert second["prev_hash"] == first["hash"]
    assert store.verify() == (True, None)


def test_an_edited_line_breaks_the_chain_at_its_seq(tmp_path, store):
    store.append(reduced(tmp_path, KNOWN_MECHANISMS["linear_attention"]))
    store.append(reduced(tmp_path, KNOWN_MECHANISMS["delta_rule"], run="run-20261002T000000Z"))
    lines = store.jsonl.read_text().splitlines()
    edited = json.loads(lines[1])
    edited["record"]["kill_codes"] = ["no_difference"]
    store.jsonl.write_text(lines[0] + "\n" + json.dumps(edited, sort_keys=True) + "\n")
    assert store.verify() == (False, 2)


def test_sqlite_index_equals_a_rebuild_from_the_jsonl(tmp_path, store):
    for index, name in enumerate(("linear_attention", "delta_rule", "ssm")):
        store.append(reduced(tmp_path, KNOWN_MECHANISMS[name], run=f"run-2026100{index}T000000Z"))
    indexed = store.sqlite_rows()
    assert indexed == store.jsonl_rows() and len(indexed) == 3
    store.sqlite.unlink()
    assert store.sqlite_rows() == indexed            # rebuilt on read from the JSONL
    assert store.rebuild_sqlite() == 3 and store.sqlite_rows() == indexed


def test_an_index_that_disagrees_with_the_jsonl_is_an_error_not_a_repair(tmp_path, store):
    import sqlite3

    store.append(reduced(tmp_path, KNOWN_MECHANISMS["linear_attention"]))
    with sqlite3.connect(store.sqlite) as connection:
        connection.execute("UPDATE outcomes SET hash = 'x' WHERE seq = 1")
    with pytest.raises(OutcomeStoreError, match="rebuild"):
        store.sqlite_rows()


def test_kill_codes_come_from_the_enumeration(tmp_path, store):
    for code in ("reduces_to_known:linear_attention", "not_radical:R3", "obligation_failed:P1", "magic_x:A2",
                 "conservation:storage", "falsified:copy_task", "duplicate:card-0123abcd", "ill_formed:toy",
                 "undetermined:S3", "stall", "budget"):
        assert check_kill_code(code) == code
    for bad in ("reduces_to_known", "free text reason", "not_radical:R4", "ill_formed:other", ""):
        with pytest.raises(ValueError):
            check_kill_code(bad)
    record = reduced(tmp_path, KNOWN_MECHANISMS["linear_attention"])
    with pytest.raises(ValueError, match="kill code"):
        store.append(dict(record, kill_codes=["it was bad"]))
    with pytest.raises(ValueError, match="needs at least one kill code"):
        store.append(dict(record, kill_codes=[], causes=[]))
    with pytest.raises(SchemaViolation):
        store.append(dict(record, outcome="buried"))
    assert store.lines() == []


def test_later_record_for_the_same_card_supersedes_in_queries(tmp_path, store):
    card = KNOWN_MECHANISMS["linear_attention"]
    store.append(record_for(tmp_path, card, [("S0", "pass", ["ok"])]))
    run_dir = tmp_path / "run-20261001T000000Z"
    from hybrid.research.ledger import Ledger
    from hybrid.research.store import Store
    Store(run_dir, Ledger(run_dir)).set_verdict(card["id"], "S1", "fail", ["gate: fatal salad-critic finding",
                                                                          "x"], ["call-0123456789ab"])
    store.append(build_record(RunView(run_dir), card["id"], "backfill"))
    current = store.current()
    assert len(store.lines()) == 2 and len(current) == 1
    assert current[0].record["kill_codes"] == ["ill_formed:fatal"]


def test_own_history_returns_exact_and_near_matches_with_the_recorded_kill_reason(tmp_path, store):
    dead = KNOWN_MECHANISMS["linear_attention"]
    store.append(reduced(tmp_path, dead))
    store.append(reduced(tmp_path, KNOWN_MECHANISMS["stack_machine"], run="run-20261002T000000Z"))
    variant = renamed(dead)
    variant["id"] = "card-0000b001"
    matches = own_history(store, variant, card_signature(variant), current_run=LATER, limit=8, jaccard_min=0.75,
                          tfidf_min=0.5)
    assert [m.stored.record["card_id"] for m in matches] == [dead["id"]]
    match = matches[0]
    assert match.exact and match.stored.dead
    found = match.as_dict()
    assert found["kill_codes"] == ["reduces_to_known:linear_attention"]
    assert found["causes"][0]["id"] == cause_id(match.stored.record_id, "reduces_to_known:linear_attention")
    assert "diagonal linear attention" in found["causes"][0]["detail"]
    text = render_history(matches)
    assert "EXACT signature" in text and found["causes"][0]["id"] in text


def test_own_history_finds_near_neighbours_and_leaves_out_only_the_cards_record_in_its_own_run(tmp_path, store):
    linear = KNOWN_MECHANISMS["linear_attention"]
    store.append(reduced(tmp_path, linear))
    # The delta rule shares most operators with linear attention but adds error correction: near, not exact.
    delta = KNOWN_MECHANISMS["delta_rule"]
    matches = own_history(store, delta, card_signature(delta), current_run=LATER, limit=8, jaccard_min=0.5,
                          tfidf_min=0.99)
    assert len(matches) == 1 and not matches[0].exact and matches[0].jaccard >= 0.5
    assert own_history(store, delta, card_signature(delta), current_run=LATER, limit=8, jaccard_min=0.99,
                       tfidf_min=0.99) == []
    # the card's own record in the querying run is left out; the same card id queried from another run finds it
    assert own_history(store, linear, card_signature(linear), current_run="run-20261001T000000Z", limit=8,
                       jaccard_min=0.1, tfidf_min=0.1) == []
    [earlier] = own_history(store, linear, card_signature(linear), current_run=LATER, limit=8, jaccard_min=0.1,
                            tfidf_min=0.1)
    assert earlier.exact and earlier.stored.record["run"] == "run-20261001T000000Z"


def test_r3_requires_the_difference_statement_to_cite_every_cause_of_an_exact_dead_match(tmp_path, store):
    dead = KNOWN_MECHANISMS["linear_attention"]
    store.append(reduced(tmp_path, dead))
    variant = renamed(dead)
    variant["id"] = "card-0000b001"
    matches = own_history(store, variant, card_signature(variant), current_run=LATER, limit=8, jaccard_min=0.75,
                          tfidf_min=0.5)
    problems = r3_problems(variant, matches)
    assert len(problems) == 1 and "reduces_to_known:linear_attention" in problems[0]
    cause = matches[0].stored.causes()[0]["id"]
    variant["difference_statement"] = {"cause_ids_addressed": [cause]}
    assert r3_problems(variant, matches) == []


def test_nearby_dead_ranks_by_task_text_lens_and_parent_tokens_one_line_per_card(tmp_path, store):
    linear = KNOWN_MECHANISMS["linear_attention"]
    stack = KNOWN_MECHANISMS["stack_machine"]
    store.append(reduced(tmp_path, linear, run="run-20261001T000000Z"))
    store.append(reduced(tmp_path, stack, run="run-20261002T000000Z"))
    store.append(reduced(tmp_path, linear, run="run-20261003T000000Z"))     # the same card in a later run
    store.append(record_for(tmp_path, KNOWN_MECHANISMS["ssm"], [("S0", "pass", ["ok"])], run="run-20261004T000000Z"))
    found = nearby_dead(store, text="pushdown stack closer brackets", tokens=card_signature(stack)["tokens"],
                        lens=None, limit=5)
    assert [m.stored.record["card_id"] for m in found] == [stack["id"], linear["id"]]   # the open ssm card is left out
    assert found[1].stored.record["run"] == "run-20261003T000000Z"
    assert nearby_dead(store, text="x", limit=5, exclude_runs=("run-20261002T000000Z", "run-20261003T000000Z",
                                                                "run-20261001T000000Z")) == []
    text = render_nearby(found)
    assert "killed at S2: reduces_to_known:linear_attention (cause oc-" in text
