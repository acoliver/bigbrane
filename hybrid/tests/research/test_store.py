import hashlib
import json
from pathlib import Path

import pytest

from hybrid.research.ledger import Ledger
from hybrid.research.schemas import SchemaViolation
from hybrid.research.store import Store
from hybrid.tests.research.builders import load_example, make_card

CARD_A = "card-aaaaaaaa"
CARD_B = "card-bbbbbbbb"
CALL = "call-0123456789ab"
JOB = "job-0123456789ab"


@pytest.fixture
def run_dir(tmp_path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def store(run_dir) -> Store:
    return Store(run_dir, Ledger(run_dir))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_put_card_writes_version_one_and_records_its_hash(store, run_dir):
    path = store.put_card(make_card(CARD_A))

    assert path == run_dir / "cards" / CARD_A / "v1.json"
    events = list(store.ledger.iter_events("card_version"))
    assert len(events) == 1
    assert events[0]["data"] == {
        "card_id": CARD_A, "version": 1, "path": "cards/card-aaaaaaaa/v1.json",
        "sha256": _sha(path), "stage": "S0",
    }
    assert store.get_card(CARD_A) == load_example() | {"id": CARD_A, "version": 1, "parent_ids": []}


def test_versions_increment_and_each_stays_readable(store):
    store.put_card(make_card(CARD_A, title="First"))
    store.put_card(make_card(CARD_A, version=2, title="Second"))

    assert store.versions(CARD_A) == [1, 2]
    assert store.next_version(CARD_A) == 3
    assert store.get_card(CARD_A)["title"] == "Second"
    assert store.get_card(CARD_A, version=1)["title"] == "First"


def test_put_card_rejects_a_version_that_is_not_next(store, run_dir):
    store.put_card(make_card(CARD_A))
    with pytest.raises(ValueError, match="next version is 2"):
        store.put_card(make_card(CARD_A, version=3))

    assert store.versions(CARD_A) == [1]
    assert len(list(store.ledger.iter_events())) == 1


def test_invalid_card_is_rejected_before_anything_is_written(store, run_dir):
    card = make_card(CARD_A)
    card["mechanism"]["state"][0].pop("shape")
    with pytest.raises(SchemaViolation):
        store.put_card(card)

    assert store.list_cards() == []
    assert list(store.ledger.iter_events()) == []


def test_put_card_rejects_unknown_parent(store):
    with pytest.raises(ValueError, match="unknown parent"):
        store.put_card(make_card(CARD_B, parents=[CARD_A]))


def test_put_card_rejects_a_parent_cycle(store):
    store.put_card(make_card(CARD_A))
    store.put_card(make_card(CARD_B, parents=[CARD_A]))
    with pytest.raises(ValueError, match="cycle"):
        store.put_card(make_card(CARD_A, version=2, parents=[CARD_B]))


def test_list_cards_is_sorted(store):
    store.put_card(make_card(CARD_B))
    store.put_card(make_card(CARD_A))
    assert store.list_cards() == [CARD_A, CARD_B]


def test_get_card_on_unknown_id_fails(store):
    with pytest.raises(KeyError):
        store.get_card(CARD_A)


def test_set_verdict_writes_a_new_version_and_a_verdict_event(store):
    store.put_card(make_card(CARD_A))
    record = store.set_verdict(CARD_A, "S1", "pass", ["toy example reproduced"], [JOB])

    assert record == {
        "card_id": CARD_A, "card_version": 2, "stage": "S1", "verdict": "pass",
        "reasons": ["toy example reproduced"], "by_calls": [JOB],
    }
    card = store.get_card(CARD_A)
    assert card["version"] == 2
    assert card["status"]["stage"] == "S1"
    assert card["status"]["verdicts"] == {
        "S1": {"verdict": "pass", "reasons": ["toy example reproduced"], "by_calls": [JOB]},
    }
    assert store.get_card(CARD_A, version=1)["status"]["verdicts"] == {}
    assert [event["data"] for event in store.ledger.iter_events("verdict")] == [record]
    assert len(list(store.ledger.iter_events("card_version"))) == 2


def test_verdicts_cannot_move_to_an_earlier_stage(store):
    store.put_card(make_card(CARD_A))
    store.set_verdict(CARD_A, "S2", "pass", ["novel"], [CALL])
    with pytest.raises(ValueError, match="after the card reached S2"):
        store.set_verdict(CARD_A, "S1", "pass", ["late"], [CALL])


def test_a_revision_at_the_same_stage_is_allowed(store):
    store.put_card(make_card(CARD_A))
    store.set_verdict(CARD_A, "S1", "revise", ["toy output mismatch"], [JOB])
    store.set_verdict(CARD_A, "S1", "pass", ["fixed"], [JOB])

    card = store.get_card(CARD_A)
    assert card["status"]["verdicts"]["S1"]["verdict"] == "pass"
    assert store.versions(CARD_A) == [1, 2, 3]


def test_verdict_with_unknown_value_or_no_reasons_is_rejected(store):
    store.put_card(make_card(CARD_A))
    with pytest.raises(SchemaViolation):
        store.set_verdict(CARD_A, "S1", "great", ["x"], [JOB])
    with pytest.raises(SchemaViolation):
        store.set_verdict(CARD_A, "S1", "pass", [], [JOB])
    assert store.versions(CARD_A) == [1]
    assert list(store.ledger.iter_events("verdict")) == []


def test_artifact_from_bytes_is_hashed_and_recorded(store, run_dir):
    payload = b'{"per_task": {}}'
    result = store.put_artifact("bench_result", "candidate.json", payload, card_id=CARD_A)

    assert result["path"] == run_dir / "artifacts" / "bench_result" / "candidate.json"
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["path"].read_bytes() == payload
    (event,) = list(store.ledger.iter_events("artifact"))
    assert event["data"] == {
        "kind": "bench_result", "name": "candidate.json", "path": "artifacts/bench_result/candidate.json",
        "sha256": result["sha256"], "bytes": len(payload), "card_id": CARD_A,
    }


def test_artifact_from_a_path_copies_the_content_hash(store, run_dir, tmp_path):
    source = tmp_path / "receipt.txt"
    source.write_bytes(b"executor receipt")
    result = store.put_artifact("receipt", "job-0123456789ab.txt", source)
    assert result["sha256"] == hashlib.sha256(b"executor receipt").hexdigest()
    assert "card_id" not in next(store.ledger.iter_events("artifact"))["data"]


def test_artifacts_are_immutable_and_names_are_plain(store):
    store.put_artifact("notes", "a.txt", b"one")
    with pytest.raises(FileExistsError):
        store.put_artifact("notes", "a.txt", b"two")
    with pytest.raises(ValueError):
        store.put_artifact("notes", "../escape.txt", b"x")
    with pytest.raises(ValueError):
        store.put_artifact("notes", "sub/dir.txt", b"x")


def test_lineage_lists_parents_and_children(store):
    store.put_card(make_card(CARD_A, title="Root"))
    store.put_card(make_card("card-cccccccc", parents=[CARD_A], title="Child one"))
    store.put_card(make_card(CARD_B, parents=[CARD_A], title="Child two"))

    lineage = store.lineage()
    assert lineage["roots"] == [CARD_A]
    assert lineage["nodes"][CARD_A]["child_ids"] == [CARD_B, "card-cccccccc"]
    assert lineage["nodes"][CARD_B]["parent_ids"] == [CARD_A]
    assert lineage["nodes"][CARD_B]["child_ids"] == []


def test_file_problems_detect_edits_and_unrecorded_files(store, run_dir):
    store.put_card(make_card(CARD_A))
    store.put_artifact("notes", "a.txt", b"one", card_id=CARD_A)
    assert store.file_problems() == []

    (run_dir / "artifacts" / "notes" / "a.txt").write_bytes(b"edited")
    stray = run_dir / "cards" / CARD_A / "v9.json"
    stray.write_text("{}", encoding="utf-8")

    problems = store.file_problems()
    assert "sha256 mismatch: artifacts/notes/a.txt" in problems
    assert "file not recorded in ledger: cards/card-aaaaaaaa/v9.json" in problems
    assert len(problems) == 2


def test_store_refuses_a_ledger_from_another_run(tmp_path):
    with pytest.raises(ValueError, match="belongs to"):
        Store(tmp_path / "one", Ledger(tmp_path / "two"))
