"""Harvest loop against a scripted OAI-PMH server: tokens, filtering, etiquette, resume, stop, deferral. No network."""
import functools
import json
import threading

import httpx
import pytest

from hybrid.literature import __main__ as cli
from hybrid.literature.arxiv_index import ArxivIndexReader, ArxivIndexWriter, PageCommit
from hybrid.literature.harvest import (
    MAX_ATTEMPTS_PER_REQUEST,
    USER_AGENT,
    HarvestError,
    Harvester,
    SetPlan,
    new_client,
)
from hybrid.tests.literature.oai_support import RESPONSE_DATE, FakeOai, oai_error, oai_page, record

CS = SetPlan("cs")
STAT = SetPlan("stat")
MATH = SetPlan("math", frozenset({"math.ST", "math.PR"}))


class Waits:
    """Records every pause and never sleeps. Requests a stop after `stop_after` pauses when given."""

    def __init__(self, stop: threading.Event | None = None, stop_after: int | None = None) -> None:
        self.seconds: list[float] = []
        self._stop, self._after = stop, stop_after

    def __call__(self, seconds: float) -> bool:
        self.seconds.append(seconds)
        if self._stop is not None and self._after is not None and len(self.seconds) >= self._after:
            self._stop.set()
        return self._stop.is_set() if self._stop is not None else False


def paged_server(pages: dict[str | None, str]) -> FakeOai:
    """Pages keyed by the resumption token that requests them (None for the first request)."""
    return FakeOai(lambda params, _req: pages[params.get("resumptionToken")])


def harvest(tmp_path, server, plans=(CS,), **kwargs):
    writer = ArxivIndexWriter(tmp_path / "arxiv.sqlite")
    waits = kwargs.pop("wait", None) or Waits()
    harvester = Harvester(writer, server.client(), plans, wait=waits, base_url="https://oai.test/oai", **kwargs)
    return writer, harvester, waits


def test_follows_resumption_tokens_to_the_last_page(tmp_path):
    server = paged_server({
        None: oai_page([record("2401.00001", "One"), record("2401.00002", "Two")], token="t1"),
        "t1": oai_page([record("2401.00003", "Three")], token="t2"),
        "t2": oai_page([record("2401.00004", "Four")]),
    })
    writer, harvester, waits = harvest(tmp_path, server)

    assert harvester.run() == "complete"

    assert server.params[0] == {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": "cs"}
    assert server.params[1] == {"verb": "ListRecords", "resumptionToken": "t1"}
    assert server.params[2] == {"verb": "ListRecords", "resumptionToken": "t2"}
    state = writer.state("cs")
    assert (state.status, state.pages, state.records_seen, state.records_kept, state.resumption_token) == ("done", 3, 4, 4, None)
    assert ArxivIndexReader(tmp_path / "arxiv.sqlite").info().record_count == 4
    assert waits.seconds == [5.0, 5.0]  # a pause between requests, none before the first


def test_real_client_identifies_itself_without_a_personal_address():
    with new_client() as client:
        agent = client.headers["User-Agent"]
    assert agent == USER_AGENT
    assert "@" not in agent and "bigbrane" in agent and "OAI-PMH" in agent


def test_category_filter_keeps_cross_lists_and_counts_the_rest(tmp_path):
    server = paged_server({None: oai_page([
        record("2401.00001", "In scope", categories=("math.ST", "stat.TH")),
        record("2401.00002", "Out of scope", categories=("math.AG",)),
        record("2401.00003", "Cross-listed", categories=("math.GM", "math.PR")),
    ])})
    writer, harvester, _ = harvest(tmp_path, server, plans=(MATH,))
    harvester.run()

    state = writer.state("math")
    assert (state.records_seen, state.records_kept, state.records_filtered) == (3, 2, 1)
    rows = ArxivIndexReader(tmp_path / "arxiv.sqlite").search('{title} : ("scope" OR "cross")', 10)
    assert {r.arxiv_id for r in rows} == {"2401.00001", "2401.00003"}


def test_record_in_two_sets_is_stored_once_and_updated(tmp_path):
    server = FakeOai(lambda params, _r: oai_page([record("2401.00001", f"Title from {params['set']}")]))
    writer, harvester, _ = harvest(tmp_path, server, plans=(CS, SetPlan("stat")))
    harvester.run()

    reader = ArxivIndexReader(tmp_path / "arxiv.sqlite")
    assert reader.info().record_count == 1
    assert [r.title for r in reader.search('{title} : ("title")', 5)] == ["Title from stat"]


def test_deleted_records_leave_the_index(tmp_path):
    first = harvest(tmp_path, paged_server({None: oai_page([record("2401.00001", "Doomed"), record("2401.00002", "Kept")])}))
    first[1].run()
    first[0].close()
    writer = ArxivIndexWriter(tmp_path / "arxiv.sqlite")
    again = Harvester(writer, paged_server({None: oai_page([], deleted=("2401.00001",))}).client(), (SetPlan("stat"),),
                      wait=Waits())
    again.run()

    reader = ArxivIndexReader(tmp_path / "arxiv.sqlite")
    assert reader.info().record_count == 1
    assert reader.search('{title} : ("doomed")', 5) == []
    assert writer.state("stat").records_deleted == 1


def test_honors_retry_after_on_503_and_retries(tmp_path):
    answers = iter([httpx.Response(503, headers={"Retry-After": "42"}), httpx.Response(503, headers={"Retry-After": "7"})])

    def handler(params, _request):
        try:
            return next(answers)
        except StopIteration:
            return oai_page([record("2401.00001", "One")])

    writer, harvester, waits = harvest(tmp_path, FakeOai(handler))
    assert harvester.run() == "complete"
    assert waits.seconds == [42.0, 7.0]
    assert writer.state("cs").records_kept == 1


def test_503_without_retry_after_backs_off_exponentially(tmp_path):
    answers = iter([httpx.Response(503), httpx.Response(503), httpx.Response(503)])

    def handler(params, _request):
        try:
            return next(answers)
        except StopIteration:
            return oai_page([record("2401.00001", "One")])

    _, harvester, waits = harvest(tmp_path, FakeOai(handler))
    harvester.run()
    assert waits.seconds == [30.0, 60.0, 120.0]


def test_set_that_keeps_failing_is_deferred_and_the_run_is_incomplete(tmp_path):
    server = FakeOai(lambda params, _r: httpx.Response(503, headers={"Retry-After": "1"}))
    writer, harvester, _ = harvest(tmp_path, server, manifest_paths=[tmp_path / "manifest.json"])
    assert harvester.run() == "deferred"
    assert len(server.requests) == 2 * MAX_ATTEMPTS_PER_REQUEST  # the first pass, then the retry pass
    assert (writer.state("cs").status, harvester.deferred_sets()) == ("deferred", ["cs"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run_status"] == "deferred" and manifest["complete"] is False


def test_client_error_on_a_set_is_not_retried_within_a_pass(tmp_path, caplog):
    server = FakeOai(lambda params, _r: httpx.Response(400))
    _, harvester, _ = harvest(tmp_path, server)
    assert harvester.run() == "deferred"
    assert len(server.requests) == 2  # one request in the first pass, one in the retry pass
    assert "HTTP 400" in caplog.text


def test_timed_out_set_is_deferred_and_the_next_set_still_runs(tmp_path, caplog):
    def handler(params, request):
        if params.get("resumptionToken") == "t1":
            raise httpx.ReadTimeout("The read operation timed out", request=request)
        if params.get("set") == "cs":
            return oai_page([record("2401.00001", "One")], token="t1")
        return oai_page([record("2401.00002", "Two")])

    server = FakeOai(handler)
    manifest = tmp_path / "manifest.json"
    writer, harvester, _ = harvest(tmp_path, server, plans=(CS, STAT), max_attempts=3, manifest_paths=[manifest])

    assert harvester.run() == "deferred"
    # cs gives up on its token, stat runs to completion, then cs gets one more pass.
    assert [p.get("set", p.get("resumptionToken")) for p in server.params] == [
        "cs", "t1", "t1", "t1", "stat", "t1", "t1", "t1",
    ]
    cs = writer.state("cs")
    assert (cs.status, cs.resumption_token, cs.pages, cs.records_kept) == ("deferred", "t1", 1, 1)
    assert writer.state("stat").status == "done"
    assert "set cs deferred" in caplog.text
    saved = json.loads(manifest.read_text())
    assert saved["run_status"] == "deferred" and saved["complete"] is False


def test_deferred_set_is_retried_after_the_other_sets_and_can_complete(tmp_path):
    calls = {"cs": 0}

    def handler(params, request):
        if params.get("set") == "stat":
            return oai_page([record("2401.00002", "Two")])
        calls["cs"] += 1
        if calls["cs"] <= 3:
            raise httpx.ReadTimeout("The read operation timed out", request=request)
        return oai_page([record("2401.00001", "One")])

    server = FakeOai(handler)
    writer, harvester, _ = harvest(tmp_path, server, plans=(CS, STAT), max_attempts=3)

    assert harvester.run() == "complete"
    assert [p.get("set") for p in server.params] == ["cs", "cs", "cs", "stat", "cs"]
    assert (writer.state("cs").status, writer.state("stat").status) == ("done", "done")
    assert harvester.deferred_sets() == []


def test_info_lists_deferred_sets(tmp_path, capsys):
    writer = ArxivIndexWriter(tmp_path / "arxiv.sqlite")
    writer.ensure_sets(["cs", "stat"])
    writer.mark_deferred("cs")
    writer.close()

    assert cli.main(["info", "--data-dir", str(tmp_path)]) == 0
    assert "deferred: cs" in capsys.readouterr().out


def test_harvest_command_exits_nonzero_with_a_summary_when_a_set_stays_deferred(tmp_path, monkeypatch, caplog):
    server = FakeOai(lambda params, _r: (
        httpx.Response(503, headers={"Retry-After": "1"}) if params.get("set") == "cs"
        else oai_page([record("2401.00002", "Two")])
    ))
    monkeypatch.setattr(cli, "new_client", server.client)
    monkeypatch.setattr(cli, "Harvester", functools.partial(Harvester, wait=Waits()))
    monkeypatch.setattr(cli, "_configure_logging", lambda log_file: None)
    monkeypatch.setattr(cli.signal, "signal", lambda signum, handler: None)

    code = cli.main([
        "harvest", "--data-dir", str(tmp_path / "data"), "--log-dir", str(tmp_path / "log"),
        "--sets", "cs", "stat", "--defer-after", "2",
    ])

    assert code == 1
    assert len(server.requests) == 5  # cs twice, stat once, then cs twice more in the retry pass
    assert "harvest incomplete: 1 set(s) still deferred: cs" in caplog.text


def test_stop_between_pages_persists_token_and_resume_continues(tmp_path):
    pages = {
        None: oai_page([record("2401.00001", "One")], token="t1"),
        "t1": oai_page([record("2401.00002", "Two")], token="t2"),
        "t2": oai_page([record("2401.00003", "Three")]),
    }
    stop = threading.Event()
    server = paged_server(pages)
    writer, harvester, _ = harvest(tmp_path, server, stop=stop, wait=Waits(stop, stop_after=1))

    assert harvester.run() == "stopped"
    state = writer.state("cs")
    assert (state.status, state.resumption_token, state.pages, state.records_kept) == ("running", "t1", 1, 1)
    writer.close()

    resumed_server = paged_server(pages)
    writer = ArxivIndexWriter(tmp_path / "arxiv.sqlite")
    resumed = Harvester(writer, resumed_server.client(), (CS,), wait=Waits())
    assert resumed.run() == "complete"
    assert resumed_server.params[0] == {"verb": "ListRecords", "resumptionToken": "t1"}  # no restart from page one
    assert writer.state("cs").status == "done"
    assert ArxivIndexReader(tmp_path / "arxiv.sqlite").info().record_count == 3


def test_stop_during_a_retry_wait_does_not_lose_progress(tmp_path):
    stop = threading.Event()
    calls = {"n": 0}

    def handler(params, _request):
        calls["n"] += 1
        if calls["n"] == 1:
            return oai_page([record("2401.00001", "One")], token="t1")
        return httpx.Response(503, headers={"Retry-After": "3"})

    writer, harvester, _ = harvest(tmp_path, FakeOai(handler), stop=stop, wait=Waits(stop, stop_after=2))
    assert harvester.run() == "stopped"
    assert writer.state("cs").resumption_token == "t1"


def test_finished_sets_are_skipped_on_a_later_run(tmp_path):
    first = harvest(tmp_path, paged_server({None: oai_page([record("2401.00001", "One")])}))
    first[1].run()
    first[0].close()
    server = paged_server({})
    writer = ArxivIndexWriter(tmp_path / "arxiv.sqlite")
    assert Harvester(writer, server.client(), (CS,), wait=Waits()).run() == "complete"
    assert server.requests == []


def test_expired_resumption_token_restarts_the_set(tmp_path):
    writer = ArxivIndexWriter(tmp_path / "arxiv.sqlite")
    writer.ensure_sets(["cs"])
    writer.commit_page(PageCommit("cs", (record("2401.00001", "One"),), (), 0, 0, "stale", RESPONSE_DATE))

    def handler(params, _request):
        if params.get("resumptionToken") == "stale":
            return oai_error("badResumptionToken", "expired")
        return oai_page([record("2401.00001", "One"), record("2401.00002", "Two")])

    server = FakeOai(handler)
    assert Harvester(writer, server.client(), (CS,), wait=Waits()).run() == "complete"
    state = writer.state("cs")
    assert (state.restarts, state.pages, state.records_kept, state.status) == (1, 1, 2, "done")
    assert server.params[-1]["metadataPrefix"] == "arXiv"


def test_no_records_match_finishes_the_set(tmp_path):
    server = FakeOai(lambda params, _r: oai_error("noRecordsMatch"))
    writer, harvester, _ = harvest(tmp_path, server, from_date="2030-01-01")
    assert harvester.run() == "complete"
    assert server.params == [{"verb": "ListRecords", "metadataPrefix": "arXiv", "set": "cs", "from": "2030-01-01"}]
    assert writer.state("cs").status == "done"


def test_page_limit_stops_without_finishing(tmp_path):
    server = FakeOai(lambda params, _r: oai_page([record("2401.00001", "One")], token="more"))
    writer, harvester, _ = harvest(tmp_path, server, max_pages=2)
    assert harvester.run() == "page_limit"
    assert writer.state("cs").pages == 2


def test_manifest_records_range_counts_and_filters(tmp_path):
    server = paged_server({
        None: oai_page([
            record("2401.00001", "One", created="2019-04-01", categories=("math.ST",)),
            record("2401.00002", "Two", created="2024-08-09", categories=("math.PR",), datestamp="2026-09-30"),
        ], token="t1", list_size=3),
        "t1": oai_page([record("2401.00003", "Three", categories=("math.AG",))]),
    })
    manifests = [tmp_path / "data" / "manifest.json", tmp_path / "log" / "manifest.json"]
    _, harvester, _ = harvest(tmp_path, server, plans=(MATH,), manifest_paths=manifests)
    harvester.run()

    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest["run_status"] == "complete" and manifest["complete"] is True
        assert manifest["record_count"] == 2
        assert manifest["created_range"] == ["2019-04-01", "2024-08-09"]
        assert manifest["datestamp_range"] == ["2026-09-30", "2026-10-01"]
        assert manifest["snapshot_ts"] == "2026-10-11T00:37:04Z"
        assert manifest["category_filters"]["math"] == ["math.PR", "math.ST"]
        math = manifest["sets"]["math"]
        assert (math["records_seen"], math["records_kept"], math["records_filtered"], math["complete_list_size"]) == (3, 2, 1, 3)
        assert math["status"] == "done" and math["completed_at"]
