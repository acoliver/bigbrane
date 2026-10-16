"""Cross-source deduplication by arXiv id and normalized title."""
from hybrid.literature.paper import Paper, dedupe, normalize_title


def make(source, source_id, title, **fields):
    base = dict(
        authors=(),
        year=None,
        abstract=None,
        url="u",
        venue=None,
        citation_count=None,
        fetched_at="t",
    )
    base.update(fields)
    return Paper(source=source, source_id=source_id, title=title, **base)


def test_same_arxiv_id_from_two_sources_merges_and_keeps_first_source():
    from_arxiv = make("arxiv", "2312.00752", "Mamba: Linear-Time Sequence Modeling", arxiv_id="2312.00752")
    from_openalex = make(
        "openalex",
        "W4389326242",
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        arxiv_id="2312.00752",
        abstract="We propose a selective state space model.",
        citation_count=900,
        year=2023,
    )
    merged = dedupe([from_arxiv, from_openalex])
    assert len(merged) == 1
    assert merged[0].source == "arxiv"
    assert merged[0].abstract == "We propose a selective state space model."
    assert merged[0].citation_count == 900
    assert merged[0].year == 2023


def test_same_normalized_title_merges_without_identifiers():
    first = make("openalex", "W1", "Attention Is All You Need!")
    second = make("s2", "abc", "attention is all you need", venue="NeurIPS")
    merged = dedupe([first, second])
    assert len(merged) == 1
    assert merged[0].venue == "NeurIPS"


def test_distinct_papers_are_kept_in_order():
    papers = [make("arxiv", "1111.1111", "Alpha"), make("arxiv", "2222.2222", "Beta")]
    assert [p.title for p in dedupe(papers)] == ["Alpha", "Beta"]


def test_normalized_title_drops_accents_case_and_punctuation():
    assert normalize_title("Poincaré  Embeddings, Revisited!") == normalize_title("poincare embeddings revisited")
