"""Live smoke test against the real keyless APIs. Runs only with BIGBRANE_LIVE=1 and `-m live`."""
import pytest

from hybrid.literature.client import Literature


@pytest.mark.live
def test_live_arxiv_and_openalex_search_return_titled_papers(tmp_path):
    literature = Literature(cache_root=tmp_path)
    receipt = literature.search_with_receipt(
        "Mamba selective state spaces", sources=("arxiv", "openalex"), limit=5
    )
    assert receipt.papers, receipt.errors
    assert all(paper.title and paper.url for paper in receipt.papers)
    assert any("mamba" in paper.title.lower() for paper in receipt.papers)
