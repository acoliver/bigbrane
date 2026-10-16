"""sample_for_lens draws only from its lens, from the seed list plus cached hits, deterministically."""
import pytest

from hybrid.literature.paper import Paper, paper_to_dict
from hybrid.literature.sampling import sample_for_lens
from hybrid.literature.seed import SeedEntry, write_build_log, write_seed


def paper(source_id, title, source="arxiv"):
    return Paper(source, source_id, title, ("A. Author",), 2024, f"Abstract of {title}.", "u", None, None, "t")


@pytest.fixture
def files(tmp_path):
    seed_path = tmp_path / "seed.toml"
    log_path = tmp_path / "log.jsonl"
    seeded = [
        SeedEntry("reservoir_computing", paper("1111.0001", "Seeded reservoir"), "relevance a"),
        SeedEntry("reservoir_computing", paper("1111.0002", "Second seeded reservoir"), "relevance b"),
        SeedEntry("morphogenesis", paper("2222.0001", "Seeded morphogenesis"), "relevance c"),
    ]
    write_seed(seeded, seed_path, "2026-10-09T00:00:00+00:00")
    hits = [paper("3333.0001", "Cached reservoir hit"), paper("1111.0001", "Seeded reservoir")]
    write_build_log(
        [
            {
                "lens": "reservoir_computing",
                "query": "echo state network",
                "per_source": {"arxiv": [paper_to_dict(p) for p in hits]},
            },
            {
                "lens": "morphogenesis",
                "query": "neural morphogenesis",
                "per_source": {"arxiv": [paper_to_dict(paper("4444.0001", "Morphogenesis hit"))]},
            },
        ],
        log_path,
    )
    return seed_path, log_path


def test_pool_is_seed_entries_plus_cached_hits_for_the_lens_only(files):
    seed_path, log_path = files
    drawn = sample_for_lens("reservoir_computing", 10, 0, seed_path=seed_path, build_log_path=log_path)
    assert sorted(p.source_id for p in drawn) == ["1111.0001", "1111.0002", "3333.0001"]


def test_same_seed_gives_same_draw_and_k_caps_at_pool_size(files):
    seed_path, log_path = files
    first = sample_for_lens("reservoir_computing", 2, 42, seed_path=seed_path, build_log_path=log_path)
    second = sample_for_lens("reservoir_computing", 2, 42, seed_path=seed_path, build_log_path=log_path)
    assert [p.source_id for p in first] == [p.source_id for p in second]
    assert len(first) == 2
    everything = sample_for_lens("morphogenesis", 50, 1, seed_path=seed_path, build_log_path=log_path)
    assert sorted(p.source_id for p in everything) == ["2222.0001", "4444.0001"]


def test_unknown_lens_is_rejected(files):
    seed_path, log_path = files
    with pytest.raises(ValueError, match="unknown lens"):
        sample_for_lens("astrology", 3, 0, seed_path=seed_path, build_log_path=log_path)


def test_negative_k_is_rejected(files):
    seed_path, log_path = files
    with pytest.raises(ValueError):
        sample_for_lens("reservoir_computing", -1, 0, seed_path=seed_path, build_log_path=log_path)
