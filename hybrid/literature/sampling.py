"""Draw reading material for an explorer working on one lens."""
from __future__ import annotations

import random
from pathlib import Path

from hybrid.literature.paper import Paper, dedupe
from hybrid.literature.seed import BUILD_LOG_PATH, SEED_PATH, cached_search_papers, check_lens, load_seed


def sample_for_lens(
    lens: str,
    k: int,
    seed: int,
    *,
    seed_path: Path = SEED_PATH,
    build_log_path: Path = BUILD_LOG_PATH,
) -> list[Paper]:
    """Pool = the seed list entries for the lens plus cached search hits for the lens, deduplicated.
    The pool is sorted before sampling so the same seed always gives the same draw."""
    check_lens(lens)
    if k < 0:
        raise ValueError("k must be non-negative")
    seeded = [entry.paper for entry in load_seed(seed_path) if entry.lens == lens]
    pool = dedupe([*seeded, *cached_search_papers(lens, build_log_path)])
    pool.sort(key=lambda paper: (paper.source, paper.source_id))
    return random.Random(seed).sample(pool, min(k, len(pool)))
