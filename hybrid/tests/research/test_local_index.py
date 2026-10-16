"""S2 with the local arXiv index: config, receipts, and an end-to-end prior-art pass. No network."""
import asyncio
import tomllib
from types import SimpleNamespace

import httpx
import pytest

import hybrid.literature as literature_module
from hybrid.literature.client import Literature
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.schemas import validate
from hybrid.research.stages import s2_prior_art
from hybrid.tests.literature.oai_support import build_fixture_index
from hybrid.tests.research.builders import make_card

CARD_ID = "card-0000abcd"


class Ledger:
    def __init__(self):
        self.events = []

    def append(self, type, actor, data):
        self.events.append((type, data))
        return {"type": type, "data": data}

    def of_type(self, type):
        return [data for t, data in self.events if t == type]


def search_event(**overrides):
    event = {
        "query": "echo", "sources": ["arxiv_local", "arxiv"], "limit": 6, "ts": "2026-10-12T00:00:00+00:00",
        "per_source_ids": {"arxiv_local": ["2401.00001"], "arxiv": []}, "errors": {}, "result_count": 1,
        "local_index": {"path": "/x/arxiv.sqlite", "status": "ok", "record_count": 1234567,
                        "snapshot_date": "2026-10-11", "complete": False},
    }
    return {**event, **overrides}


# Config

def test_shipped_configs_name_the_default_index_path():
    for name in ("run.toml", "run2.toml"):
        with open(ROOT / "hybrid" / "config" / name, "rb") as handle:
            assert tomllib.load(handle)["literature"]["local_index_path"] == "data/arxiv-index/arxiv.sqlite"


def test_config_without_literature_section_gets_the_default(tmp_path):
    with open(ROOT / "hybrid" / "config" / "run.toml", "rb") as handle:
        text = handle.read().decode()
    start = text.index("[literature]")
    end = text.index("[s2]")
    old = tmp_path / "old.toml"
    old.write_text(text[:start] + text[end:])
    assert load_run_config(old)["literature"]["local_index_path"] == "data/arxiv-index/arxiv.sqlite"


def test_index_path_can_be_overridden_and_must_be_a_path_string():
    shipped = ROOT / "hybrid" / "config" / "run.toml"
    assert load_run_config(shipped, {"literature.local_index_path": "elsewhere.sqlite"})["literature"][
        "local_index_path"] == "elsewhere.sqlite"
    for bad in ("", 7, None):
        with pytest.raises(ValueError, match="literature.local_index_path"):
            load_run_config(shipped, {"literature.local_index_path": bad})


# Receipts

def sink_with_ledger():
    ledger = Ledger()
    ctx = SimpleNamespace(ledger=ledger)
    return s2_prior_art.CardLiteratureSink(ctx, CARD_ID), ledger


def test_local_receipt_states_source_snapshot_date_and_record_count():
    sink, ledger = sink_with_ledger()
    sink.append("literature_search", "literature", search_event())

    receipts = {r["source"]: r for r in ledger.of_type("literature_receipt")}
    assert set(receipts) == {"arxiv_local", "arxiv"}
    local = receipts["arxiv_local"]
    assert local["result_ids"] == ["2401.00001"]
    assert (local["snapshot_date"], local["record_count"], local["index_complete"]) == ("2026-10-11", 1234567, False)
    assert "snapshot_date" not in receipts["arxiv"] and "record_count" not in receipts["arxiv"]
    for receipt in receipts.values():
        validate("literature_receipt", receipt)


def test_semantic_scholar_events_named_s2_get_a_semantic_scholar_receipt():
    sink, ledger = sink_with_ledger()
    sink.append("literature_search", "literature", search_event(
        per_source_ids={"arxiv": [], "openalex": ["W1"], "s2": ["abc123", "def456"]}, errors={}))

    receipts = {r["source"]: r for r in ledger.of_type("literature_receipt")}
    assert set(receipts) == {"arxiv", "openalex", "semantic_scholar"}
    assert receipts["semantic_scholar"]["result_ids"] == ["abc123", "def456"]
    assert receipts["semantic_scholar"]["card_id"] == CARD_ID
    for receipt in receipts.values():
        validate("literature_receipt", receipt)


def test_receipt_records_that_a_missing_index_was_skipped():
    sink, ledger = sink_with_ledger()
    reason = "skipped: local index not found at /x/arxiv.sqlite"
    sink.append("literature_search", "literature", search_event(
        per_source_ids={"arxiv_local": [], "arxiv": []}, errors={"arxiv_local": reason},
        local_index={"path": "/x/arxiv.sqlite", "status": "missing"}))

    (local,) = [r for r in ledger.of_type("literature_receipt") if r["source"] == "arxiv_local"]
    assert local["result_ids"] == [] and local["note"] == reason
    assert "record_count" not in local
    validate("literature_receipt", local)


def test_schema_rejects_malformed_snapshot_fields():
    base = {"query": "q", "source": "arxiv_local", "ts": "2026-10-12T00:00:00+00:00", "result_ids": []}
    validate("literature_receipt", dict(base, snapshot_date="2026-10-11", record_count=0, index_complete=True))
    for bad in ({"snapshot_date": "yesterday"}, {"record_count": -1}, {"index_complete": "yes"}, {"note": ""}):
        with pytest.raises(Exception):
            validate("literature_receipt", dict(base, **bad))


# End to end through prior_art() with the configured index

class Completion:
    def __init__(self, parsed, call_id):
        self.parsed, self.call_id = parsed, call_id


class ScriptedCtx:
    """Just enough of StageContext for prior_art(): config, ledger, and a scripted model."""

    def __init__(self, index_path, grounded_id):
        self.cfg = {"literature": {"local_index_path": str(index_path)}}
        self.ledger = Ledger()
        self.classify_prior_art = None
        self._answers = [
            {"queries": ["echo state network", "reservoir computing chaos", "recurrent random features"]},
            {"label": "prior_art_undertested", "notes": "close",
             "evidence": [{"source": "arxiv", "source_id": grounded_id, "relation": "closest_prior_work"}]},
        ]

    async def call(self, role, system, user, schema, allow_web=False):
        return Completion(self._answers.pop(0), f"call-{len(self._answers)}")


def all_rate_limited_literature(index_path):
    web = httpx.MockTransport(lambda request: httpx.Response(429, text="{}"))
    return Literature(cache_root=index_path.parent / "cache", http=httpx.Client(transport=web),
                      sleep=lambda s: None, clock=lambda: 0.0, local_index_path=index_path, max_retries=0)


def test_prior_art_finds_and_grounds_local_hits_while_every_remote_source_is_rate_limited(tmp_path, monkeypatch):
    index = build_fixture_index(tmp_path / "arxiv.sqlite")
    shared = all_rate_limited_literature(index)
    monkeypatch.setattr(literature_module, "literature_for", lambda path: shared if path == index else pytest.fail(path))
    ctx = ScriptedCtx(index, "2401.00001")

    result, calls = asyncio.run(s2_prior_art.prior_art(ctx, make_card(CARD_ID)))

    assert "arxiv:2401.00001" in result["candidates"]
    assert [e["source_id"] for e in result["evidence"]] == ["2401.00001"]
    assert result["rejected_evidence"] == []
    receipts = ctx.ledger.of_type("literature_receipt")
    local = [r for r in receipts if r["source"] == "arxiv_local"]
    assert len(local) == 3 and all(r["record_count"] == 5 and r["snapshot_date"] == "2026-10-11" for r in local)
    assert any("2401.00001" in r["result_ids"] for r in local)
    remote_ids = [r["result_ids"] for r in receipts if r["source"] in ("arxiv", "openalex")]
    assert remote_ids and all(ids == [] for ids in remote_ids)


def test_prior_art_with_missing_index_still_runs_and_records_the_skip(tmp_path, monkeypatch):
    missing = tmp_path / "nowhere" / "arxiv.sqlite"
    shared = all_rate_limited_literature(missing)
    monkeypatch.setattr(literature_module, "literature_for", lambda path: shared)
    ctx = ScriptedCtx(missing, "2401.00001")
    ctx._answers[1]["evidence"] = []

    result, _ = asyncio.run(s2_prior_art.prior_art(ctx, make_card(CARD_ID)))

    assert result["candidates"] == []
    local = [r for r in ctx.ledger.of_type("literature_receipt") if r["source"] == "arxiv_local"]
    assert len(local) == 3
    assert all(r["note"].startswith("skipped: local index not found") and r["result_ids"] == [] for r in local)
