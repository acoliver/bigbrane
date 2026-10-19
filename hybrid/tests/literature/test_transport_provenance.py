"""Measured attempt facts, with fixed policy and no live requests."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import logging
import threading
from types import SimpleNamespace

import httpx
import pytest

from hybrid.literature import provenance
from hybrid.literature.cache import ResponseCache
from hybrid.literature.client import LOCAL_SOURCE, Literature
from hybrid.literature.keys import ApiKey
from hybrid.literature.transport import Request, SourceError, Transport
from hybrid.tests.literature.support import FakeClock, RecordingSink, fixture_text

EPOCH = datetime(2026, 10, 18, tzinfo=timezone.utc)
URL = "https://export.arxiv.org/api/query"


def now(clock):
    return lambda: (EPOCH + timedelta(seconds=clock.now)).isoformat()


def transport(tmp_path, handler, clock=None, **kwargs):
    clock = clock or FakeClock()
    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return Transport(ResponseCache(tmp_path), http, clock=clock.clock, sleep=clock.sleep,
                     now=now(clock), **kwargs), clock


def test_200_and_pacer_wait_are_measured_without_extra_requests(tmp_path):
    clock = FakeClock()
    calls = []

    def handler(request):
        calls.append((request.url.params["q"], clock.now))
        clock.now += 0.25
        return httpx.Response(200, text="answer")

    tr, _ = transport(tmp_path, handler, clock)
    first = tr.get("arxiv", Request(URL, {"q": "alpha"})).provenance
    second = tr.get("arxiv", Request(URL, {"q": "beta"})).provenance
    assert calls == [("alpha", 0), ("beta", 3)]
    assert clock.sleeps == [2.75]
    assert first["attempt_count"] == first["http_request_count"] == first["http_response_count"] == 1
    assert first["outcome"] == "network" and first["terminal"] == "success"
    attempt = second["attempts"][0]
    assert attempt["started_at"] == "2026-10-18T00:00:03+00:00"
    assert attempt["ordinal"] == 1 and attempt["started_s"] == 3
    assert attempt["duration_s"] == attempt["exchanges"][0]["duration_s"] == 0.25
    assert attempt["pacer_wait_s"] == 2.75
    assert attempt["http_status"] == 200 and attempt["redirect_chain"] == []
    assert second["duration_s"] == 3


@pytest.mark.parametrize("header,parsed,kind,wait,pacer", [
    ("7", 7, "delta", 7, 0),
    ("1000", 1000, "delta", 120, 0),
    ("1", 1, "delta", 1, 2),
    ("Sun, 18 Oct 2026 00:00:40 GMT", 40, "date", 5, 0),
    ("Sat, 17 Oct 2026 23:00:00 GMT", 0, "date", 5, 0),
    ("not a date", None, "invalid", 5, 0),
])
def test_retry_after_is_observed_without_changing_policy(tmp_path, header, parsed, kind, wait, pacer):
    calls = []
    clock = FakeClock()

    def handler(request):
        calls.append((str(request.url), clock.now))
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": header})
        return httpx.Response(200, text="done")

    tr, _ = transport(tmp_path, handler, clock, max_retries=1)
    record = tr.get("arxiv", Request(URL, {"q": "alpha"})).provenance
    first, second = record["attempts"]
    assert len(calls) == record["http_request_count"] == record["http_response_count"] == 2
    assert [t for _, t in calls] == [0, wait + pacer]
    assert calls[0][0] == calls[1][0] == URL + "?q=alpha"
    assert clock.sleeps == ([wait, pacer] if pacer else [wait])
    assert first["retry_after_s"] == parsed and first["retry_after_kind"] == kind
    assert first["retry_delay_s"] == first["retry_wait_s"] == wait
    assert first["terminal"] == "retry" and second["terminal"] == "success"
    assert second["pacer_wait_s"] == pacer
    assert [a["ordinal"] for a in record["attempts"]] == [1, 2]


def test_exhaustion_cooldown_and_cache_before_cooldown(tmp_path):
    codes = iter([200, 429, 429, 429])
    calls = []
    clock = FakeClock()

    def handler(request):
        calls.append(clock.now)
        return httpx.Response(next(codes), text="answer")

    tr, _ = transport(tmp_path, handler, clock, max_retries=2)
    cached_request = Request(URL, {"q": "cached"})
    tr.get("arxiv", cached_request)
    with pytest.raises(SourceError) as exhausted:
        tr.get("arxiv", Request(URL, {"q": "exhausted"}))
    record = exhausted.value.provenance
    assert record["attempt_count"] == record["http_request_count"] == record["http_response_count"] == 3
    assert [a["http_status"] for a in record["attempts"]] == [429, 429, 429]
    assert [a["retry_wait_s"] for a in record["attempts"]] == [5, 10, 0]
    assert calls == [0, 3, 8, 18] and clock.sleeps == [3, 5, 10]
    assert record["terminal"] == "exhausted" and record["cooldown_remaining_s"] == 600
    before = list(clock.sleeps)
    with pytest.raises(SourceError) as skipped:
        tr.get("arxiv", Request(URL, {"q": "skip"}))
    skip = skipped.value.provenance
    hit = tr.get("arxiv", cached_request).provenance
    assert skip["outcome"] == "cooldown" and skip["terminal"] == "skipped"
    assert skip["cooldown_remaining_s"] == 600
    assert hit["outcome"] == "cache_hit" and hit["terminal"] == "success"
    for no_network in (skip, hit):
        assert no_network["attempt_count"] == no_network["http_request_count"] == no_network["http_response_count"] == 0
        assert no_network["attempts"] == []
    assert calls == [0, 3, 8, 18] and clock.sleeps == before


def test_redirect_is_two_http_requests_inside_one_logical_attempt(tmp_path):
    clock = FakeClock()
    calls = []

    def handler(request):
        calls.append((request.url.scheme, clock.now))
        clock.now += 0.25
        if request.url.scheme == "http":
            return httpx.Response(301, headers={"Location": URL + "?token=redirect-secret"})
        return httpx.Response(200, text="done")

    tr, _ = transport(tmp_path, handler, clock)
    record = tr.get("arxiv", Request("http://export.arxiv.org/api/query", {"q": "query-secret"})).provenance
    assert calls == [("http", 0), ("https", 0.25)] and clock.sleeps == []
    assert record["attempt_count"] == 1
    assert record["http_request_count"] == record["http_response_count"] == 2
    attempt = record["attempts"][0]
    assert attempt["duration_s"] == 0.5 and attempt["http_status"] == 200
    assert [e["http_status"] for e in attempt["exchanges"]] == [301, 200]
    assert attempt["redirect_chain"] == [{"endpoint": "http://export.arxiv.org", "http_status": 301}]
    assert "secret" not in json.dumps(record)


def test_redirect_then_timeout_keeps_completed_hop_and_safe_error(tmp_path):
    clock = FakeClock()

    def handler(request):
        clock.now += 0.5
        if request.url.scheme == "http":
            return httpx.Response(301, headers={"Location": URL + "?token=secret"})
        raise httpx.ReadTimeout(f"secret {request.url} {dict(request.headers)}", request=request)

    tr, _ = transport(tmp_path, handler, clock, max_retries=0)
    with pytest.raises(SourceError) as error:
        tr.get("arxiv", Request("http://user:password@export.arxiv.org/private/secret?token=secret", {}))
    record = error.value.provenance
    assert record["attempt_count"] == 1 and record["http_request_count"] == 2
    assert record["http_response_count"] == 1 and record["error_type"] == "ReadTimeout"
    assert [e["http_status"] for e in record["attempts"][0]["exchanges"]] == [301, None]
    assert record["attempts"][0]["duration_s"] == 1
    for sensitive in ("secret", "password", "user", "token="):
        assert sensitive not in str(error.value) + json.dumps(record)


def test_network_timeout_retries_have_no_invented_http_status(tmp_path):
    clock = FakeClock()
    calls = []

    def handler(request):
        calls.append(clock.now)
        clock.now += 2
        raise httpx.ReadTimeout("private details", request=request)

    tr, _ = transport(tmp_path, handler, clock, max_retries=1)
    with pytest.raises(SourceError) as error:
        tr.get("s2", Request("https://api.semanticscholar.org/search", {"q": "a"}))
    record = error.value.provenance
    assert calls == [0, 7] and clock.sleeps == [5]
    assert record["attempt_count"] == record["http_request_count"] == 2
    assert record["http_response_count"] == 0
    assert [a["duration_s"] for a in record["attempts"]] == [2, 2]
    assert all(a["http_status"] is None and a["error_type"] == "ReadTimeout" for a in record["attempts"])
    assert record["terminal"] == "exhausted"


def test_terminal_404_is_not_retried(tmp_path):
    tr, clock = transport(tmp_path, lambda r: httpx.Response(404))
    with pytest.raises(SourceError) as error:
        tr.get("arxiv", Request(URL, {}))
    record = error.value.provenance
    assert record["terminal"] == "http_error" and record["error_type"] == "HTTPStatus"
    assert record["attempt_count"] == record["http_request_count"] == 1
    assert clock.sleeps == []


def test_search_records_use_hashed_association_and_existing_sink(tmp_path, caplog):
    secret = "query-secret-and-account"
    clock = FakeClock()
    key = ApiKey("s2", "header-secret", "test origin")

    def leaky(request):
        raise httpx.ConnectError(f"{request.url} {dict(request.headers)}", request=request)

    lit = Literature(cache_root=tmp_path, http=httpx.Client(transport=httpx.MockTransport(leaky)),
                     clock=clock.clock, sleep=clock.sleep, now=now(clock), max_retries=0,
                     api_keys={"s2": key})
    sink = RecordingSink()
    with caplog.at_level(logging.WARNING):
        receipt = lit.search_with_receipt(secret, ("s2",), 3, sink)
    (record,) = sink.of_type("literature_transport")
    assert receipt.transport == (record,)
    assert sink.of_type("literature_search")[0]["transport_call_id"] == record["call_id"]
    assert record["operation"] == "search_with_receipt"
    assert record["association_id"] == provenance.identity_hash(secret)
    assert record["source"] == "s2" and record["error_type"] == "ConnectError"
    assert secret not in json.dumps(record) + caplog.text
    assert key.value not in json.dumps(record) + caplog.text + repr(receipt)


def test_local_search_has_zero_http_requests(tmp_path):
    clock = FakeClock()
    lit = Literature(cache_root=tmp_path, clock=clock.clock, sleep=clock.sleep, now=now(clock),
                     http=httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("no HTTP"))))
    lit._search_local = lambda query, limit, ts: ((), {"status": "ok"})
    result = lit.search_with_receipt("alpha", (LOCAL_SOURCE,))
    (record,) = result.transport
    assert record["outcome"] == "local" and record["terminal"] == "success"
    assert record["attempts"] == [] and record["http_request_count"] == 0
    assert record["association_id"] == provenance.identity_hash("alpha")


def test_missing_local_index_is_measured_without_http(tmp_path):
    clock = FakeClock()
    lit = Literature(cache_root=tmp_path, clock=clock.clock, sleep=clock.sleep, now=now(clock))
    result = lit.search_with_receipt("alpha", (LOCAL_SOURCE,))
    assert result.transport[0]["outcome"] == "local"
    assert result.transport[0]["terminal"] == "error"
    assert result.transport[0]["http_request_count"] == 0


def test_parsing_error_does_not_echo_external_body(tmp_path):
    clock = FakeClock()
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text='{"secret":bad}')))
    lit = Literature(cache_root=tmp_path, http=http, clock=clock.clock, sleep=clock.sleep, now=now(clock))
    result = lit.search_with_receipt("alpha", ("openalex",))
    assert result.transport[0]["terminal"] == "parse_error"
    assert result.transport[0]["error_type"] == "JSONDecodeError"
    assert "secret" not in json.dumps(result.transport) + str(result.errors)


def test_interleaved_cards_reach_real_ledger_without_cross_talk(tmp_path):
    from hybrid.research.ledger import Ledger
    from hybrid.research.stages.s2_prior_art import CardLiteratureSink

    first_entered, second_finished = threading.Event(), threading.Event()
    clock = FakeClock()
    calls = []

    def handler(request):
        query = request.url.params["search_query"]
        calls.append(query)
        if "alpha" in query:
            first_entered.set()
            assert second_finished.wait(10)
        else:
            assert first_entered.wait(10)
        return httpx.Response(200, text=fixture_text("arxiv_search.xml"))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    lit = Literature(cache_root=tmp_path / "cache", http=http, clock=clock.clock,
                     sleep=clock.sleep, now=now(clock))
    ledger = Ledger(tmp_path / "run")
    ctx = SimpleNamespace(ledger=ledger)

    def run(query, card):
        receipt = lit.search_with_receipt(query, ("arxiv",), 3, CardLiteratureSink(ctx, card))
        if query == "beta":
            second_finished.set()
        return receipt

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, "alpha", "card-A")
        assert first_entered.wait(10)
        second = pool.submit(run, "beta", "card-B")
        b, a = second.result(timeout=15), first.result(timeout=15)
    events = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    records = [e["data"] for e in events if e["type"] == "literature_transport"]
    assert [r["card_id"] for r in records] == ["card-B", "card-A"]
    assert [r["association_id"] for r in records] == [provenance.identity_hash("beta"), provenance.identity_hash("alpha")]
    assert len(calls) == 2 and a.transport[0]["call_id"] != b.transport[0]["call_id"]
    for record in records:
        assert record["attempt_count"] == record["http_request_count"] == record["http_response_count"] == 1
        assert record["attempts"][0]["http_status"] == 200
        summary = next(e["data"] for e in events if e["type"] == "literature_search"
                       and e["data"]["card_id"] == record["card_id"])
        assert summary["transport_call_id"] == record["call_id"]
    assert ledger.verify() == (True, None)
    snapshot = json.dumps(a.transport)
    http.get(URL, params={"search_query": "alpha"})
    assert json.dumps(a.transport) == snapshot  # no completed-call state retained by HTTP hooks


def test_lookup_and_related_calls_keep_actual_source_context(tmp_path):
    from hybrid.tests.literature.test_client import openalex_graph_responder

    clock = FakeClock()
    sink = RecordingSink()
    lit = Literature(cache_root=tmp_path, clock=clock.clock, sleep=clock.sleep, now=now(clock),
                     http=httpx.Client(transport=httpx.MockTransport(openalex_graph_responder)))
    paper = lit.get_paper("openalex", "W100", sink)
    graph = lit.related(paper, sink)
    records = sink.of_type("literature_transport")
    assert graph.via == "openalex"
    assert [r["operation"] for r in records] == ["get_paper", "related", "related", "related"]
    assert [r["outcome"] for r in records] == ["network", "cache_hit", "network", "network"]
    assert len({r["call_id"] for r in records[1:]}) == 1
    assert records[0]["call_id"] != records[1]["call_id"]
    assert {r["source"] for r in records} == {"openalex"}
    assert {r["association_id"] for r in records} == {provenance.identity_hash("openalex:W100")}
    assert sink.of_type("literature_related")[0]["transport_call_id"] == records[1]["call_id"]
    assert sum(r["http_request_count"] for r in records) == 3


def test_failed_lookup_still_emits_transport_event(tmp_path):
    clock = FakeClock()
    lit = Literature(cache_root=tmp_path, clock=clock.clock, sleep=clock.sleep, now=now(clock),
                     http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    sink = RecordingSink()
    with pytest.raises(SourceError):
        lit.get_paper("arxiv", "0000.00001", sink)
    (record,) = sink.of_type("literature_transport")
    assert record["operation"] == "get_paper" and record["terminal"] == "http_error"


def test_reused_http_client_does_not_duplicate_exchange_hooks(tmp_path):
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok")))
    clock = FakeClock()
    first = Transport(ResponseCache(tmp_path / "a"), http, clock=clock.clock, sleep=clock.sleep)
    second = Transport(ResponseCache(tmp_path / "b"), http, clock=clock.clock, sleep=clock.sleep)
    assert first.get("arxiv", Request(URL, {})).provenance["http_request_count"] == 1
    assert second.get("arxiv", Request(URL, {})).provenance["http_request_count"] == 1
    assert len(http.event_hooks["request"]) == len(http.event_hooks["response"]) == 1


def test_unhandled_http_error_still_fails_fast_with_measured_exchanges(tmp_path):
    http = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(301, headers={"Location": URL})), follow_redirects=True, max_redirects=2)
    clock = FakeClock()
    tr = Transport(ResponseCache(tmp_path), http, clock=clock.clock, sleep=clock.sleep)
    sink = RecordingSink()
    with pytest.raises(httpx.TooManyRedirects):
        with provenance.operation("get", "private-query", sink):
            tr.get("arxiv", Request(URL, {}))
    (record,) = sink.of_type("literature_transport")
    assert record["terminal"] == "error" and record["error_type"] == "TooManyRedirects"
    assert record["attempt_count"] == 1
    assert record["http_request_count"] == record["http_response_count"] == 3
    assert record["attempts"][0]["terminal"] == "error"
    assert clock.sleeps == []


def test_retry_records_requested_and_actual_sleep_without_repeating_it(tmp_path):
    class EarlySleepClock(FakeClock):
        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds - 1 if seconds > 1 else seconds

    clock = EarlySleepClock()
    calls = []

    def handler(request):
        calls.append(clock.now)
        return httpx.Response(429 if len(calls) == 1 else 200, headers={"Retry-After": "7"})

    tr, _ = transport(tmp_path, handler, clock, max_retries=1)
    record = tr.get("arxiv", Request(URL, {})).provenance
    assert calls == [0, 6] and clock.sleeps == [7]
    assert record["attempts"][0]["retry_delay_s"] == 7
    assert record["attempts"][0]["retry_wait_s"] == 6
