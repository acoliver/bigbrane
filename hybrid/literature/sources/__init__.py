"""Keyless literature sources. Each module exposes request builders and pure parsers."""
from hybrid.literature.sources import arxiv, openalex, semantic_scholar

MODULES = {
    arxiv.NAME: arxiv,
    openalex.NAME: openalex,
    semantic_scholar.NAME: semantic_scholar,
}
