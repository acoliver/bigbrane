"""Draw reading material for an explorer working on one lens."""
from __future__ import annotations

import random
from pathlib import Path

from hybrid.literature.paper import Paper, dedupe
from hybrid.literature.seed import SEED_PATH, cached_search_papers, check_lens, load_seed


def sample_for_lens(
    lens: str,
    k: int,
    seed: int,
    *,
    seed_path: Path = SEED_PATH,
    build_log_path: Path | None = None,
) -> list[Paper]:
    """Draw from lens seed entries plus validated shipped history, or an explicit raw log.
    The pool is deduplicated and sorted so the same seed always gives the same draw."""
    check_lens(lens)
    if k < 0:
        raise ValueError("k must be non-negative")
    seeded = [entry.paper for entry in load_seed(seed_path) if entry.lens == lens]
    pool = dedupe([*seeded, *cached_search_papers(lens, build_log_path)])
    pool.sort(key=lambda paper: (paper.source, paper.source_id))
    return random.Random(seed).sample(pool, min(k, len(pool)))
