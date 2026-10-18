"""The S2 novelty gate: when a card counts as known work (decision R2A, R2C).

The reduction critic answers three separate questions (schemas/reduction_critic.schema.json):
- `known_work`: the card's general mechanism, its update, readout and learning rule with every learned parameter free,
  is already published or textbook. This is the only one of the three that gates here.
- exclusion from an out-of-scope family is decided by families.toml and the counted comparators (KD4), not here.
- `surprise`: whether a researcher in the excluded families would be surprised. Recorded, never gated.

A `known_work: true` answer kills automatically only with a host-counted, independently referenced comparator.
Citations establish at most identity, never coverage of the general mechanism. Paper leads keep their source-qualified
ids; unvalidated textbook/algorithm names and coverage prose go to the existing difference challenger, or remain
unevidenced when that stage is disabled. No keyword in `covers` decides a verdict. A resemblance that fixes a learned
part is not evidence about the general learned mechanism.
The textbook_transplant gate is separate: `true` means the source field's construction is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

SOURCE_KINDS = ("paper", "textbook", "standard_algorithm")


@dataclass(frozen=True)
class Judgment:
    kills: bool
    basis: str       # counted_comparator, coverage_unchecked, not_claimed, learned_part_fixed, no_evidence
    reason: str


def bare(ref: str) -> str:
    """A paper ref without its source prefix: `s2:712c...` and `712c...` name the same candidate."""
    return ref.split(":", 1)[-1]


def paper_key(ref: str, candidates: list[dict]) -> str | None:
    """Keep qualified identity. An old bare id resolves only when it has one source."""
    keys = {c["key"] for c in candidates}
    if ref in keys:
        return ref
    if ":" in ref:
        return None
    matches = {c["key"] for c in candidates if c["source_id"] == ref}
    return next(iter(matches)) if len(matches) == 1 else None


def candidate_ids(candidates: list[dict]) -> set[str]:
    return {c["key"] for c in candidates}


def accepted_sources(critic: dict, candidates: list[dict]) -> list[dict]:
    """Source leads for the challenger, NOT verified coverage of the mechanism.

    Names remain unvalidated leads; a paper must resolve to its actual source-qualified id.
    """
    sources = []
    for source in critic["known_work_sources"]:
        if source["kind"] != "paper":
            sources.append(source)
        else:
            key = paper_key(source["ref"], candidates)
            if key is not None:
                sources.append(dict(source, ref=key))
    return sources


def cited_papers(critic: dict, candidates: list[dict]) -> list[str]:
    """Source-qualified search-result ids to send to the difference challenger."""
    return [s["ref"] for s in accepted_sources(critic, candidates) if s["kind"] == "paper"]


def cited_names(critic: dict, candidates: list[dict]) -> list[dict]:
    """Textbook sections and standard algorithms the critic cites; the challenger checks the statement answers them."""
    return [s for s in accepted_sources(critic, candidates) if s["kind"] != "paper"]


def judge(critic: dict, counted: list[dict], candidates: list[dict]) -> Judgment:
    if not critic["known_work"]:
        return Judgment(False, "not_claimed", "the critic did not find the general mechanism to be known work")
    fixed = critic["fixed_learned_parts"]
    if fixed:
        parts = "; ".join(f"{p['part']} (card states: {p['capability'][:120]})" for p in fixed)
        return Judgment(False, "learned_part_fixed",
                        f"the resemblance holds only with a learned part fixed, zeroed or set to the witness: {parts}")
    if counted:
        ids = ", ".join(c["declared_id"] or critic["reduces_to"] for c in counted)
        return Judgment(True, "counted_comparator", f"host-counted comparator: {ids}")
    sources = accepted_sources(critic, candidates)
    if sources:
        cited = "; ".join(f"{s['kind']} {s['ref']}" for s in sources)
        return Judgment(False, "coverage_unchecked", f"source leads, mechanism coverage not checked: {cited}")
    return Judgment(False, "no_evidence", "known work claimed without an independently referenced comparator; "
                                          "no source-qualified paper lead resolved in the search results")


LISTED_TITLES = 80   # PA2: candidates are stored best-ranked first; titles are cheap, so the critic sees more than the writer


def search_listing(candidates: list[dict]) -> str:
    """Source-qualified results and supplied excerpts within the critic bound.
    S2 selects the field-balanced, required-evidence listing before calling this renderer."""
    lines = [f"- {c['key']} ({c['year'] if c['year'] is not None else 'n.d.'}) {c['title']}\n  {c.get('excerpt', '(excerpt not recorded)')}"
             for c in candidates[:LISTED_TITLES]]
    head = "SEARCH RESULTS (cite a paper by exactly one of these ids"
    head += f"; the {len(lines)} best-ranked of {len(candidates)}):\n" if len(candidates) > len(lines) else "):\n"
    return head + ("\n".join(lines) or "(no source returned)")
