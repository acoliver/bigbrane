"""Parsers against saved real responses, plus request construction."""
import json

import pytest

from hybrid.literature.paper import Paper, dedupe, normalize_arxiv_id
from hybrid.literature.sources import arxiv, openalex, semantic_scholar
from hybrid.tests.literature.support import fixture_text


def test_arxiv_fixture_parses_entries_with_versionless_ids():
    papers = arxiv.parse_search(fixture_text("arxiv_search.xml"), "2026-10-09T00:00:00+00:00")
    assert len(papers) == 3
    for paper in papers:
        assert paper.source == "arxiv"
        assert paper.arxiv_id == paper.source_id
        assert "v" not in paper.source_id.split(".")[-1]
        assert paper.url == f"https://arxiv.org/abs/{paper.source_id}"
        assert paper.authors and all(author for author in paper.authors)
        assert paper.year is not None and 1990 <= paper.year <= 2026
        assert paper.abstract and "\n" not in paper.abstract
        assert paper.citation_count is None


def test_arxiv_error_entry_is_a_parse_failure():
    body = (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        "<id>http://arxiv.org/api/errors#incorrect_id_format</id>"
        "<title>Error</title><summary>incorrect id format</summary></entry></feed>"
    )
    with pytest.raises(ValueError, match="arXiv API error"):
        arxiv.parse_search(body, "t")


def test_arxiv_search_request_uses_one_term_per_word():
    request = arxiv.search_request("Mamba/SSM test-time", 5)
    assert request.url == "https://export.arxiv.org/api/query"
    assert request.params["search_query"] == "all:Mamba AND all:SSM AND all:test AND all:time"
    assert request.params["max_results"] == "5"


def test_arxiv_get_request_strips_version_and_prefix():
    assert arxiv.get_request("arXiv:2312.00752v2").params["id_list"] == "2312.00752"


def test_arxiv_get_rejects_non_arxiv_ids():
    with pytest.raises(ValueError):
        arxiv.get_request("W4389713725")


def test_openalex_fixture_parses_works_and_rebuilds_abstracts():
    papers = openalex.parse_search(fixture_text("openalex_search.json"), "t")
    assert len(papers) == 3
    assert all(paper.source_id.startswith("W") for paper in papers)
    assert all(paper.abstract and "  " not in paper.abstract for paper in papers)
    by_arxiv = {paper.arxiv_id: paper for paper in papers if paper.arxiv_id}
    assert "2312.06635" in by_arxiv
    found = by_arxiv["2312.06635"]
    assert found.year == 2023
    assert found.doi == "10.48550/arxiv.2312.06635"
    assert found.url == "https://doi.org/10.48550/arxiv.2312.06635"
    assert found.citation_count is not None


def test_openalex_abstract_rebuilds_from_inverted_index():
    assert openalex.rebuild_abstract({"a": [0, 2], "b": [1]}) == "a b a"
    assert openalex.rebuild_abstract(None) is None


def test_openalex_work_without_title_is_skipped():
    work = {"id": "https://openalex.org/W1", "title": None, "display_name": None}
    assert openalex.parse_work(work, "t") is None


def test_openalex_rejects_malformed_work_id():
    with pytest.raises(ValueError):
        openalex.work_id_of({"id": "https://example.org/notes"})


def test_openalex_references_request_caps_or_filter():
    with pytest.raises(ValueError):
        openalex.references_request([f"W{i}" for i in range(101)])


def test_openalex_get_accepts_doi_lookup():
    request = openalex.get_request("doi:10.48550/arXiv.2312.00752")
    assert request.url == "https://api.openalex.org/works/doi:10.48550/arXiv.2312.00752"


def test_normalize_arxiv_id_handles_prefixes_versions_and_old_style_ids():
    assert normalize_arxiv_id("arXiv:2312.00752v3") == "2312.00752"
    assert normalize_arxiv_id("math/0101001v1") == "math/0101001"
    assert normalize_arxiv_id("W4389713725") is None


def test_paper_rejects_unknown_source():
    with pytest.raises(ValueError):
        Paper("pubmed", "1", "T", (), None, None, "u", None, None, "t")


def test_dedupe_keeps_order_for_a_single_source_fixture():
    papers = arxiv.parse_search(fixture_text("arxiv_search.xml"), "t")
    assert [p.source_id for p in dedupe(papers)] == [p.source_id for p in papers]


def test_s2_fixture_parses_real_search_response():
    papers = semantic_scholar.parse_search(fixture_text("s2_search.json"), "t")
    assert len(papers) == 10
    assert all(paper.source == "s2" for paper in papers)
    assert all(len(paper.source_id) == 40 for paper in papers)
    assert all(paper.title and paper.url for paper in papers)
    first = papers[0]
    assert first.title.startswith("A Correntropy-Based Echo State Network")
    assert first.doi == "10.1109/JAS.2024.124932"
    assert first.arxiv_id is None
    assert first.citation_count is not None


def test_s2_references_skip_unresolved_cited_papers():
    # Synthetic payload: the saved fixture is a search response, and this shape is the
    # documented references shape. Replace with a saved references response when one exists.
    body = json.dumps(
        {
            "data": [
                {"citedPaper": {"paperId": "p1", "title": "Ref one", "externalIds": {}}},
                {"citedPaper": {"paperId": None, "title": None}},
            ]
        }
    )
    assert [p.source_id for p in semantic_scholar.parse_references(body, "t")] == ["p1"]


def test_s2_request_paths_keep_prefixed_ids_readable():
    request = semantic_scholar.get_request("ARXIV:2312.00752")
    assert request.url.endswith("/paper/ARXIV:2312.00752")
