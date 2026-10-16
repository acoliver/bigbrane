import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hybrid.research.ledger import GENESIS_HASH, Ledger, event_hash, sorted_json

ROOT = Path(__file__).resolve().parents[3]
LEDGER_LINES = "ledger.jsonl"


def _read_lines(run_dir: Path) -> list[str]:
    return (run_dir / LEDGER_LINES).read_text(encoding="utf-8").splitlines()


def _write_lines(run_dir: Path, lines: list[str]) -> None:
    (run_dir / LEDGER_LINES).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fill(run_dir: Path, count: int) -> Ledger:
    ledger = Ledger(run_dir)
    for n in range(count):
        ledger.append("note", "tester", {"n": n})
    return ledger


def test_append_builds_a_hash_chain_from_genesis(tmp_path):
    ledger = Ledger(tmp_path)
    first = ledger.append("note", "tester", {"n": 1})
    second = ledger.append("note", "tester", {"n": 2})

    assert first["seq"] == 1
    assert first["prev_hash"] == GENESIS_HASH
    assert second["seq"] == 2
    assert second["prev_hash"] == first["hash"]
    assert set(first) == {"seq", "ts", "type", "actor", "data", "prev_hash", "hash"}
    body = {key: value for key, value in first.items() if key != "hash"}
    expected = hashlib.sha256((GENESIS_HASH + sorted_json(body)).encode("utf-8")).hexdigest()
    assert first["hash"] == expected
    assert ledger.verify() == (True, None)


def test_stored_lines_are_sorted_key_utf8_json(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.append("note", "tester", {"text": "café", "b": 1, "a": 2})
    line = _read_lines(tmp_path)[0]

    assert "café" in line
    assert line == sorted_json(json.loads(line))
    assert ":" in line and ": " not in line
    event = json.loads(line)
    assert event_hash(event["prev_hash"], {k: v for k, v in event.items() if k != "hash"}) == event["hash"]


def test_empty_ledger_verifies(tmp_path):
    assert Ledger(tmp_path).verify() == (True, None)


def test_edited_data_is_reported_at_its_seq(tmp_path):
    ledger = _fill(tmp_path, 3)
    lines = _read_lines(tmp_path)
    event = json.loads(lines[1])
    event["data"]["n"] = 99
    lines[1] = sorted_json(event)
    _write_lines(tmp_path, lines)

    assert ledger.verify() == (False, 2)


def test_recomputed_hash_still_breaks_the_next_link(tmp_path):
    ledger = _fill(tmp_path, 3)
    lines = _read_lines(tmp_path)
    event = json.loads(lines[1])
    event["data"]["n"] = 99
    body = {key: value for key, value in event.items() if key != "hash"}
    event["hash"] = event_hash(event["prev_hash"], body)
    lines[1] = sorted_json(event)
    _write_lines(tmp_path, lines)

    assert ledger.verify() == (False, 3)


def test_deleted_event_is_detected(tmp_path):
    ledger = _fill(tmp_path, 3)
    lines = _read_lines(tmp_path)
    del lines[1]
    _write_lines(tmp_path, lines)

    assert ledger.verify() == (False, 2)


def test_unparseable_line_is_detected(tmp_path):
    ledger = _fill(tmp_path, 2)
    lines = _read_lines(tmp_path)
    lines[1] = lines[1][:-5]
    _write_lines(tmp_path, lines)

    assert ledger.verify() == (False, 2)


def test_partial_trailing_write_is_detected(tmp_path):
    ledger = _fill(tmp_path, 3)
    with open(tmp_path / LEDGER_LINES, "a", encoding="utf-8") as f:
        f.write('{"seq":4,"ts":')

    assert ledger.verify() == (False, 4)


def test_iter_events_filters_by_type(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.append("alpha", "tester", {"n": 1})
    ledger.append("beta", "tester", {"n": 2})
    ledger.append("alpha", "tester", {"n": 3})

    assert [event["data"]["n"] for event in ledger.iter_events("alpha")] == [1, 3]
    assert len(list(ledger.iter_events())) == 3


def test_append_rejects_non_dict_data_without_writing(tmp_path):
    ledger = _fill(tmp_path, 1)
    size_before = (tmp_path / LEDGER_LINES).stat().st_size

    with pytest.raises(TypeError):
        ledger.append("note", "tester", ["not", "a", "dict"])
    assert (tmp_path / LEDGER_LINES).stat().st_size == size_before


def test_append_rejects_values_json_cannot_represent_without_writing(tmp_path):
    ledger = _fill(tmp_path, 1)
    size_before = (tmp_path / LEDGER_LINES).stat().st_size

    with pytest.raises(ValueError):
        ledger.append("note", "tester", {"x": float("nan")})
    assert (tmp_path / LEDGER_LINES).stat().st_size == size_before
    assert ledger.verify() == (True, None)


WORKER = """
import sys
from hybrid.research.ledger import Ledger
ledger = Ledger(sys.argv[1])
for i in range(int(sys.argv[3])):
    ledger.append("tick", sys.argv[2], {"i": i})
"""


def test_two_processes_appending_keep_the_chain_valid(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    workers = [
        subprocess.Popen([sys.executable, "-c", WORKER, str(tmp_path), f"proc{index}", "200"], env=env, cwd=ROOT)
        for index in range(2)
    ]
    for worker in workers:
        assert worker.wait(timeout=240) == 0

    ledger = Ledger(tmp_path)
    assert ledger.verify() == (True, None)
    events = list(ledger.iter_events("tick"))
    assert len(events) == 400
    assert [event["seq"] for event in events] == list(range(1, 401))
    assert {event["actor"] for event in events} == {"proc0", "proc1"}


def test_many_asyncio_tasks_keep_the_chain_valid(tmp_path):
    ledger = Ledger(tmp_path)

    async def write_all():
        await asyncio.gather(*(
            asyncio.to_thread(ledger.append, "tick", "task", {"i": index}) for index in range(300)
        ))

    asyncio.run(write_all())

    assert ledger.verify() == (True, None)
    events = list(ledger.iter_events())
    assert [event["seq"] for event in events] == list(range(1, 301))
    assert sorted(event["data"]["i"] for event in events) == list(range(300))
