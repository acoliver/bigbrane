"""The arXiv API endpoint must not add an automatic HTTP-to-HTTPS request."""
import httpx
import pytest

from hybrid.literature.client import Literature
from hybrid.literature.sources import arxiv
from hybrid.tests.literature.support import FakeClock, fixture_text


@pytest.mark.parametrize("built_request", [arxiv.search_request("sequence memory", 6), arxiv.get_request("1110.5667")])
def test_arxiv_request_builders_use_https(built_request):
    assert built_request.url == "https://export.arxiv.org/api/query"


def test_search_and_lookup_each_use_one_paced_https_exchange(tmp_path):
    clock = FakeClock()
    exchanges = []

    def handler(request):
        exchanges.append((str(request.url), clock.now))
        if request.url.scheme == "http":
            return httpx.Response(301, headers={"Location": str(request.url.copy_with(scheme="https"))})
        return httpx.Response(200, text=fixture_text("arxiv_search.xml"))

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as http:
        literature = Literature(cache_root=tmp_path, http=http, clock=clock.clock, sleep=clock.sleep)
        literature.search("sequence memory", sources=("arxiv",), limit=6)
        literature.get_paper("arxiv", "1110.5667")
        literature.search("sequence memory", sources=("arxiv",), limit=6)

    assert len(exchanges) == 2
    assert all(url.startswith("https://export.arxiv.org/api/query?") for url, _ in exchanges)
    assert [started for _, started in exchanges] == [0.0, 3.0]
