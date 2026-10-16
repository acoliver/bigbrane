"""The arxiv_local source: query translation, BM25 ranking, Paper identity, and its place in Literature.search."""
import sqlite3

import httpx
import pytest

from hybrid.literature.client import LOCAL_SOURCE, Literature
from hybrid.literature.paper import paper_key_of
from hybrid.literature.sources.arxiv_local import ArxivLocal, match_expressions
from hybrid.tests.literature.oai_support import FIXTURE_RECORDS, build_fixture_index
from hybrid.tests.literature.support import FakeClock, FakeWeb, RecordingSink, fixture_text, make_literature, ok, status

NOW = "2026-10-12T00:00:00+00:00"


@pytest.fixture
def index(tmp_path):
    return build_fixture_index(tmp_path / "arxiv.sqlite")


def local(index):
    return ArxivLocal(index)


# Query translation

def test_keyword_query_becomes_all_terms_then_any_term():
    all_terms, any_term = match_expressions("echo state network")
    assert all_terms == '{title abstract} : ("echo" AND "state" AND "network")'
    assert any_term == '{title abstract} : ("echo" OR "state" OR "network")'


def test_quoted_phrases_and_stopwords():
    expressions = match_expressions('"sparse distributed memory" for the associative recall')
    assert expressions[0] == '{title abstract} : ("sparse distributed memory" AND "associative" AND "recall")'


def test_hyphens_and_punctuation_split_into_words():
    assert match_expressions("state-space models")[0] == '{title abstract} : ("state" AND "space" AND "models")'


def test_reserved_words_and_operators_are_neutralized():
    # FTS5 would read bare AND/OR/NOT/NEAR and a trailing * as syntax; every term is quoted instead.
    assert match_expressions("near NOT title:foo*")[0] == '{title abstract} : ("near" AND "not" AND "title" AND "foo")'


def test_single_term_has_no_any_term_fallback():
    assert len(match_expressions("hopfield")) == 1


def test_query_of_only_stopwords_is_searched_as_written():
    assert match_expressions("of the")[0] == '{title abstract} : ("of" AND "the")'


def test_query_without_words_is_rejected():
    with pytest.raises(ValueError, match="no searchable terms"):
        match_expressions("!!! ???")


# Search

def test_bm25_puts_the_best_title_and_abstract_match_first(index):
    result = local(index).search("reservoir computing", 5, NOW)
    assert [p.source_id for p in result.papers][:2] == ["2401.00004", "2401.00001"]


def test_stemming_matches_inflected_forms(index):
    assert "2401.00002" in {p.source_id for p in local(index).search("network memorize", 5, NOW).papers}


def test_all_terms_results_precede_any_term_results(index):
    ids = [p.source_id for p in local(index).search("sparse memory associative", 5, NOW).papers]
    # 00005 has "sparse" and "memory"; 00002 has "memory" and "associative"; none has all three,
    # so the any-term pass supplies every result.
    assert set(ids) >= {"2401.00005", "2401.00002"}
    both = [p.source_id for p in local(index).search("sparse memory", 5, NOW).papers]
    assert both[0] == "2401.00005"


def test_search_does_not_match_authors_or_categories(index):
    assert local(index).search("Hopfield", 5, NOW).papers[0].source_id == "2401.00002"
    assert local(index).search("Jaeger", 5, NOW).papers == ()
    assert local(index).search("nlin", 5, NOW).papers == ()


def test_limit_is_respected(index):
    assert len(local(index).search("networks OR memory OR models OR computing", 2, NOW).papers) == 2


def test_papers_carry_arxiv_identity_and_metadata(index):
    paper = local(index).search("selective state space", 1, NOW).papers[0]
    assert (paper.source, paper.source_id, paper.arxiv_id) == ("arxiv", "2401.00003", "2401.00003")
    assert paper.url == "https://arxiv.org/abs/2401.00003"
    assert paper.title == "State space models for long sequence modelling"
    assert paper.authors == ("Albert Gu",) and paper.year == 2024
    assert paper.venue == "Some Journal 12 (2024)" and paper.citation_count is None
    assert paper.fetched_at == "2026-10-11T00:37:04Z"
    assert paper_key_of(paper) == "arxiv:2401.00003"


def test_reports_snapshot_facts(index):
    info = local(index).search("hopfield", 1, NOW).info
    assert (info.record_count, info.snapshot_date, info.complete) == (5, "2026-10-11", True)


def test_incomplete_harvest_is_flagged(tmp_path):
    path = build_fixture_index(tmp_path / "partial.sqlite", complete=False)
    assert local(path).search("hopfield", 1, NOW).info.complete is False


def test_search_works_from_another_thread(index):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        papers = pool.submit(lambda: local(index).search("hopfield", 1, NOW).papers).result()
    assert papers[0].source_id == "2401.00002"


# Inside Literature

def remote_web():
    return FakeWeb({
        "export.arxiv.org": ok(fixture_text("arxiv_search.xml")),
        "api.openalex.org": ok(fixture_text("openalex_search.json")),
        "api.semanticscholar.org": status(404),
    })


def literature_with_local(tmp_path, web, path, **kwargs):
    clock = FakeClock()
    return Literature(
        cache_root=tmp_path / "cache",
        http=httpx.Client(transport=httpx.MockTransport(web.handler)),
        sleep=clock.sleep,
        clock=clock.clock,
        now=lambda: NOW,
        local_index_path=path,
        **kwargs,
    )


def test_search_with_receipt_states_source_snapshot_and_count(tmp_path, index):
    literature = literature_with_local(tmp_path, FakeWeb({}), index)
    sink = RecordingSink()
    receipt = literature.search_with_receipt("echo", (LOCAL_SOURCE,), 3, sink)

    assert [p.source_id for p in receipt.per_source[LOCAL_SOURCE]] == ["2401.00001"]
    assert receipt.errors == {}
    (event,) = sink.of_type("literature_search")
    assert event["sources"] == [LOCAL_SOURCE]
    assert event["per_source_ids"] == {LOCAL_SOURCE: ["2401.00001"]}
    assert event["local_index"] == {
        "path": str(index), "status": "ok", "record_count": 5, "snapshot_date": "2026-10-11", "complete": True,
    }


def test_default_sources_lead_with_the_local_index_only_when_configured(tmp_path, index):
    assert literature_with_local(tmp_path, FakeWeb({}), index).default_sources == (LOCAL_SOURCE, "arxiv", "openalex", "s2")
    assert make_literature(tmp_path, FakeWeb({})).default_sources == ("arxiv", "openalex", "s2")


def test_missing_index_is_skipped_and_recorded_while_remote_sources_answer(tmp_path):
    literature = literature_with_local(tmp_path, remote_web(), tmp_path / "absent.sqlite", max_retries=0)
    sink = RecordingSink()
    receipt = literature.search_with_receipt("mamba selective state spaces", (LOCAL_SOURCE, "arxiv"), 3, sink)

    assert receipt.per_source[LOCAL_SOURCE] == ()
    assert receipt.errors[LOCAL_SOURCE].startswith("skipped: local index not found at ")
    assert receipt.per_source["arxiv"] and len(receipt.papers) == len(receipt.per_source["arxiv"])
    (event,) = sink.of_type("literature_search")
    assert event["local_index"] == {"path": str(tmp_path / "absent.sqlite"), "status": "missing"}
    assert LOCAL_SOURCE in event["errors"]


def test_unconfigured_local_source_is_skipped(tmp_path):
    receipt = make_literature(tmp_path, FakeWeb({})).search_with_receipt("hopfield", (LOCAL_SOURCE,), 3)
    assert receipt.errors == {LOCAL_SOURCE: "skipped: no local index is configured"}


def test_unreadable_index_is_recorded_not_raised(tmp_path):
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"this is not a database" * 100)
    receipt = literature_with_local(tmp_path, FakeWeb({}), bad).search_with_receipt("hopfield", (LOCAL_SOURCE,), 3)
    assert "local index unreadable" in receipt.errors[LOCAL_SOURCE]


def test_wrong_schema_version_fails_loudly(tmp_path):
    other = tmp_path / "other.sqlite"
    sqlite3.connect(other).execute("PRAGMA user_version=99").connection.close()
    with pytest.raises(RuntimeError, match="schema version 99"):
        literature_with_local(tmp_path, FakeWeb({}), other).search("hopfield", (LOCAL_SOURCE,), 3)


def test_local_hits_survive_remote_rate_limit_cooldowns(tmp_path, index):
    web = FakeWeb({
        "export.arxiv.org": status(429),
        "api.openalex.org": status(429),
        "api.semanticscholar.org": status(429),
    })
    literature = literature_with_local(tmp_path, web, index, max_retries=0)
    receipt = literature.search_with_receipt("echo", literature.default_sources, 6)

    assert [p.source_id for p in receipt.papers] == ["2401.00001"]
    assert set(receipt.errors) == {"arxiv", "openalex", "s2"}


def test_merge_dedupes_by_arxiv_id_and_local_record_wins(tmp_path):
    from hybrid.literature.sources import arxiv as remote_arxiv
    from hybrid.tests.literature.oai_support import record
    from hybrid.literature.arxiv_index import ArxivIndexWriter, PageCommit

    remote = remote_arxiv.parse_search(fixture_text("arxiv_search.xml"), "t")[0]
    path = tmp_path / "overlap.sqlite"
    with ArxivIndexWriter(path) as writer:
        writer.ensure_sets(["cs"])
        local_copy = record(remote.arxiv_id, remote.title, "Local copy of the abstract.", authors=("Local Author",))
        writer.commit_page(PageCommit("cs", (local_copy,), (), 0, 0, None, "2026-10-11T00:00:00Z"))

    literature = literature_with_local(tmp_path, remote_web(), path, max_retries=0)
    receipt = literature.search_with_receipt(remote.title, (LOCAL_SOURCE, "arxiv"), 10)

    assert remote.arxiv_id in {p.source_id for p in receipt.per_source[LOCAL_SOURCE]}
    assert remote.arxiv_id in {p.source_id for p in receipt.per_source["arxiv"]}
    merged = [p for p in receipt.papers if p.arxiv_id == remote.arxiv_id]
    assert len(merged) == 1
    assert merged[0].abstract == "Local copy of the abstract."
    assert merged[0].authors == ("Local Author",)
    assert paper_key_of(merged[0]) == f"arxiv:{remote.arxiv_id}"


def test_unknown_source_names_still_rejected(tmp_path, index):
    with pytest.raises(ValueError, match="unknown literature source"):
        literature_with_local(tmp_path, FakeWeb({}), index).search("x", ("arxiv_remote",), 3)


def test_fixture_records_are_what_the_tests_assume():
    assert [r.arxiv_id for r in FIXTURE_RECORDS] == [f"2401.0000{i}" for i in range(1, 6)]
