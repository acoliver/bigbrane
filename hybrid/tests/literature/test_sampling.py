"""sample_for_lens draws only from its lens, from the seed list plus cached hits, deterministically."""
import json
from pathlib import Path

import pytest

from hybrid.literature import seed as seed_module

from hybrid.literature.paper import Paper, paper_key_of, paper_to_dict
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


@pytest.mark.parametrize("explicit_none", [False, True])
def test_default_pool_matches_shipped_seed_without_reading_raw_log(monkeypatch, explicit_none):
    original_open = Path.open

    def no_raw_log(path, *args, **kwargs):
        assert path != seed_module.BUILD_LOG_PATH, "default sampling opened ignored raw history"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_raw_log)
    entries = seed_module.load_seed()
    for lens in seed_module.LENSES:
        expected = {paper_key_of(entry.paper): entry.paper for entry in entries if entry.lens == lens}
        hits = seed_module.cached_search_papers(lens, **({"path": None} if explicit_none else {}))
        drawn = sample_for_lens(lens, 1000, 42, **({"build_log_path": None} if explicit_none else {}))
        assert len(hits) == len(expected)
        assert {paper_key_of(p): p for p in hits} == expected
        assert len(drawn) == len(expected)
        assert {paper_key_of(p): p for p in drawn} == expected


def test_explicit_missing_raw_log_does_not_fall_back_to_projection(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        seed_module.cached_search_papers("reservoir_computing", missing)
    with pytest.raises(FileNotFoundError):
        sample_for_lens("reservoir_computing", 10, 0, build_log_path=missing)


def test_explicit_raw_log_preserves_hits_from_all_historical_rows(files):
    seed_path, log_path = files
    rows = seed_module.read_build_log(log_path)
    rows.append({
        "lens": "reservoir_computing", "query": "another historical search",
        "per_source": {"arxiv": [paper_to_dict(paper("5555.0001", "Another historical hit"))]},
    })
    write_build_log(rows, log_path)
    hits = seed_module.cached_search_papers("reservoir_computing", log_path)
    assert sorted(p.source_id for p in hits) == ["1111.0001", "3333.0001", "5555.0001"]
    drawn = sample_for_lens("reservoir_computing", 100, 0, seed_path=seed_path, build_log_path=log_path)
    assert sorted(p.source_id for p in drawn) == ["1111.0001", "1111.0002", "3333.0001", "5555.0001"]


def test_corrupt_shipped_projection_fails_default_sampling(tmp_path, monkeypatch):
    document = json.loads(seed_module.SEED_PROVENANCE_PATH.read_bytes())
    document["payload"]["rows"][0]["query"] += " changed"
    corrupt = tmp_path / "corrupt-provenance.json"
    corrupt.write_text(json.dumps(document))
    original_open = Path.open

    def corrupt_projection(path, *args, **kwargs):
        return original_open(corrupt if path == seed_module.SEED_PROVENANCE_PATH else path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", corrupt_projection)
    with pytest.raises(ValueError, match="checksum"):
        seed_module.cached_search_papers("reservoir_computing")
    with pytest.raises(ValueError, match="checksum"):
        sample_for_lens("reservoir_computing", 10, 0)
