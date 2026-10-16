"""Caching, retries, failure recording, search events and citation lookups, against a fake web."""
import httpx
import pytest

from hybrid.literature.paper import Paper
from hybrid.tests.literature.support import FakeClock, FakeWeb, RecordingSink, fixture_text, make_literature, ok, status

ARXIV = "export.arxiv.org"
OPENALEX = "api.openalex.org"
S2 = "api.semanticscholar.org"


def refuse(request):
    raise httpx.ConnectError("connection refused", request=request)


def work(work_id, title):
    return {
        "id": f"https://openalex.org/{work_id}",
        "doi": None,
        "title": title,
        "display_name": title,
        "publication_year": 2024,
        "authorships": [],
        "primary_location": None,
        "cited_by_count": 1,
        "abstract_inverted_index": None,
        "referenced_works": [],
    }


def standard_web(**overrides):
    routes = {
        ARXIV: ok(fixture_text("arxiv_search.xml")),
        OPENALEX: ok(fixture_text("openalex_search.json")),
        S2: status(404),
    }
    routes.update(overrides)
    return FakeWeb(routes)


# Caching

def test_second_identical_search_makes_no_network_call(tmp_path):
    web = standard_web()
    literature = make_literature(tmp_path, web)
    first = literature.search("linear attention", sources=("arxiv", "openalex"), limit=3)
    calls_after_first = len(web.calls)
    second = literature.search("linear attention", sources=("arxiv", "openalex"), limit=3)
    assert len(web.calls) == calls_after_first == 2
    assert [p.source_id for p in first] == [p.source_id for p in second]
    assert (tmp_path / "arxiv").is_dir() and (tmp_path / "openalex").is_dir()


def test_cache_hit_does_not_wait_for_rate_limit(tmp_path):
    web = standard_web()
    clock = FakeClock()
    literature = make_literature(tmp_path, web, clock=clock)
    literature.search("linear attention", sources=("arxiv",), limit=3)
    sleeps_before = list(clock.sleeps)
    literature.search("linear attention", sources=("arxiv",), limit=3)
    assert clock.sleeps == sleeps_before


def test_arxiv_requests_are_at_least_three_seconds_apart(tmp_path):
    web = standard_web()
    clock = FakeClock()
    literature = make_literature(tmp_path, web, clock=clock)
    literature.search("alpha", sources=("arxiv",), limit=3)
    literature.search("beta", sources=("arxiv",), limit=3)
    assert clock.sleeps == [pytest.approx(3.0)]


def test_failed_responses_are_not_cached(tmp_path):
    web = FakeWeb({S2: status(404)})
    literature = make_literature(tmp_path, web)
    literature.search("anything", sources=("s2",), limit=3)
    assert list((tmp_path / "s2").glob("*.json")) == []


# Retries and failure recording

def test_429_is_retried_after_retry_after_seconds(tmp_path):
    web = FakeWeb({ARXIV: [status(429, {"Retry-After": "7"}), ok(fixture_text("arxiv_search.xml"))]})
    clock = FakeClock()
    literature = make_literature(tmp_path, web, clock=clock)
    receipt = literature.search_with_receipt("alpha", sources=("arxiv",), limit=3)
    assert receipt.errors == {}
    assert len(receipt.papers) == 3
    assert clock.sleeps[0] == 7.0
    assert len(web.calls_to(ARXIV)) == 2


def test_exhausted_429_is_recorded_then_source_cools_down_and_is_retried_later(tmp_path):
    web = FakeWeb({S2: status(429)})
    clock = FakeClock()
    literature = make_literature(tmp_path, web, clock=clock, max_retries=2)
    first = literature.search_with_receipt("alpha", sources=("s2",), limit=3)
    assert first.errors == {"s2": "HTTP 429 after 3 attempts"}
    assert clock.sleeps == [5.0, 10.0]
    assert len(web.calls_to(S2)) == 3

    second = literature.search_with_receipt("beta", sources=("s2",), limit=3)
    assert "cooling down" in second.errors["s2"]
    assert len(web.calls_to(S2)) == 3  # no network call during the cooldown

    clock.now += 600.0
    third = literature.search_with_receipt("gamma", sources=("s2",), limit=3)
    assert third.errors == {"s2": "HTTP 429 after 3 attempts"}
    assert len(web.calls_to(S2)) == 6


def test_cooldown_does_not_affect_other_sources(tmp_path):
    web = standard_web(**{S2: status(429)})
    literature = make_literature(tmp_path, web, max_retries=0)
    literature.search("alpha", sources=("s2",), limit=3)
    receipt = literature.search_with_receipt("beta", sources=("arxiv", "s2"), limit=3)
    assert receipt.errors.keys() == {"s2"}
    assert receipt.per_source["arxiv"]
    assert len(web.calls_to(ARXIV)) == 1


def test_one_failing_source_does_not_drop_the_others(tmp_path):
    web = standard_web(**{S2: status(429)})
    literature = make_literature(tmp_path, web, max_retries=0)
    sink = RecordingSink()
    papers = literature.search("linear attention", limit=3, sink=sink)
    assert {p.source for p in papers} == {"arxiv", "openalex"}
    (event,) = sink.of_type("literature_search")
    assert set(event["errors"]) == {"s2"}
    assert event["per_source_ids"]["arxiv"]


def test_not_found_is_not_retried(tmp_path):
    web = FakeWeb({S2: status(404)})
    literature = make_literature(tmp_path, web, max_retries=3)
    receipt = literature.search_with_receipt("alpha", sources=("s2",), limit=3)
    assert receipt.errors == {"s2": "HTTP 404"}
    assert len(web.calls_to(S2)) == 1


def test_transport_errors_are_retried_then_recorded(tmp_path):
    web = FakeWeb({S2: refuse})
    literature = make_literature(tmp_path, web, max_retries=2)
    receipt = literature.search_with_receipt("alpha", sources=("s2",), limit=3)
    assert receipt.errors["s2"].startswith("transport error")
    assert len(web.calls_to(S2)) == 3


def test_unparseable_body_is_recorded_as_source_error(tmp_path):
    web = FakeWeb({OPENALEX: ok("<html>maintenance</html>")})
    literature = make_literature(tmp_path, web)
    receipt = literature.search_with_receipt("alpha", sources=("openalex",), limit=3)
    assert receipt.errors["openalex"].startswith("unparseable response")
    assert receipt.papers == ()


# Search contract

def test_search_event_carries_query_sources_ts_ids_and_errors(tmp_path):
    web = standard_web(**{S2: status(404)})
    literature = make_literature(tmp_path, web)
    sink = RecordingSink()
    papers = literature.search("linear attention", sources=("arxiv", "openalex", "s2"), limit=3, sink=sink)
    (event,) = sink.of_type("literature_search")
    assert sink.events[0][0] == "literature_search" and sink.events[0][1] == "literature"
    assert event["query"] == "linear attention"
    assert event["sources"] == ["arxiv", "openalex", "s2"]
    assert event["ts"] == "2026-10-09T00:00:00+00:00"
    assert event["limit"] == 3
    assert event["per_source_ids"]["s2"] == []
    assert event["errors"] == {"s2": "HTTP 404"}
    assert event["result_count"] == len(papers)


def test_unknown_source_and_empty_query_are_rejected(tmp_path):
    literature = make_literature(tmp_path, standard_web())
    with pytest.raises(ValueError):
        literature.search("alpha", sources=("pubmed",))
    with pytest.raises(ValueError):
        literature.search("   ")


def test_openalex_gets_mailto_only_when_contact_is_configured(tmp_path):
    web = standard_web()
    make_literature(tmp_path / "with", web, contact_email="ops@example.org").search(
        "alpha", sources=("openalex",), limit=3
    )
    make_literature(tmp_path / "without", web).search("beta", sources=("openalex",), limit=3)
    with_email, without_email = web.calls_to(OPENALEX)
    assert with_email.url.params["mailto"] == "ops@example.org"
    assert "mailto" not in without_email.url.params


def test_search_result_order_is_source_priority_then_rank(tmp_path):
    web = standard_web()
    literature = make_literature(tmp_path, web)
    receipt = literature.search_with_receipt("alpha", sources=("arxiv", "openalex"), limit=3)
    arxiv_ids = [p.source_id for p in receipt.per_source["arxiv"]]
    assert [p.source_id for p in receipt.papers[: len(arxiv_ids)]] == arxiv_ids


# Lookup and citation graph

def test_get_paper_returns_the_feed_entry_and_records_an_event(tmp_path):
    web = standard_web()
    literature = make_literature(tmp_path, web)
    sink = RecordingSink()
    paper = literature.get_paper("arxiv", "2312.00752", sink=sink)
    assert paper.source == "arxiv"
    assert sink.of_type("literature_get")[0]["source_id"] == "2312.00752"


def openalex_graph_responder(request):
    path = request.url.path
    if path == "/works/W100":
        focal = work("W100", "Focal paper")
        focal["referenced_works"] = ["https://openalex.org/W1", "https://openalex.org/W2"]
        return httpx.Response(200, json=focal)
    if request.url.params.get("filter", "").startswith("openalex:"):
        return httpx.Response(200, json={"results": [work("W1", "Reference one"), work("W2", "Reference two")]})
    if request.url.params.get("filter", "").startswith("cites:"):
        return httpx.Response(200, json={"results": [work("W3", "Citing paper")]})
    raise AssertionError(f"unexpected OpenAlex request {request.url}")


def test_related_uses_openalex_references_and_citations(tmp_path):
    web = FakeWeb({OPENALEX: openalex_graph_responder, S2: status(404)})
    literature = make_literature(tmp_path, web)
    paper = literature.get_paper("openalex", "W100")
    related = literature.related(paper)
    assert related.via == "openalex"
    assert [p.source_id for p in related.references] == ["W1", "W2"]
    assert [p.source_id for p in related.citations] == ["W3"]
    assert related.errors == {}


def test_related_resolves_arxiv_papers_by_doi_and_falls_back_to_s2(tmp_path):
    s2_refs = {"data": [{"citedPaper": {"paperId": "p1", "title": "Ref one"}}, {"citedPaper": {"paperId": None}}]}
    s2_cites = {"data": [{"citingPaper": {"paperId": "c1", "title": "Cite one"}}]}

    def s2_responder(request):
        if request.url.path.endswith("/references"):
            return httpx.Response(200, json=s2_refs)
        return httpx.Response(200, json=s2_cites)

    web = FakeWeb({ARXIV: ok(fixture_text("arxiv_search.xml")), OPENALEX: status(500), S2: s2_responder})
    literature = make_literature(tmp_path, web, max_retries=0)
    arxiv_paper = literature.search("alpha", sources=("arxiv",), limit=3)[0]
    related = literature.related(arxiv_paper)
    assert related.via == "s2"
    assert "HTTP 500" in related.errors["openalex"]
    assert [p.source_id for p in related.references] == ["p1"]
    assert [p.source_id for p in related.citations] == ["c1"]
    assert any(call.url.path == f"/graph/v1/paper/ARXIV:{arxiv_paper.arxiv_id}/references" for call in web.calls_to(S2))


def test_related_records_each_failed_source_and_returns_empty(tmp_path):
    literature = make_literature(tmp_path, FakeWeb({OPENALEX: status(404), S2: status(404)}), max_retries=0)
    unresolved = Paper("arxiv", "0000.00001", "Untitled", (), None, None, "u", None, None, "t", arxiv_id="0000.00001")
    related = literature.related(unresolved)
    assert related.via is None
    assert related.references == () and related.citations == ()
    assert set(related.errors) == {"openalex", "s2"}
