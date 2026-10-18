"""The S2 novelty gate: when a card counts as known work (decision R2A, R2C).

The reduction critic answers three separate questions (schemas/reduction_critic.schema.json):
- `known_work`: the card's general mechanism, its update, readout and learning rule with every learned parameter free,
  is already published or textbook. This is the only one of the three that gates here.
- exclusion from an out-of-scope family is decided by families.toml and the counted comparators (KD4), not here.
- `surprise`: whether a researcher in the excluded families would be surprised. Recorded, never gated.

A `known_work: true` answer kills the card only when the host can check its evidence:
1. a host-counted comparator (the reduction check verified a comparator of the claimed family within the tolerance), or
2. at least one cited source that the host accepts, when the critic did not reach its mapping by fixing a learned part:
   a `paper` must be a candidate of this card's prior-art search; a `textbook` section or a `standard_algorithm` must
   be named. Its `covers` text says which steps of the card, including the ones the check typed `other` or could not
   verify, the source accounts for.
Anything else, such as "familiar", "would not surprise" or "moved to a new field" without a source covering the
mechanism, is recorded as unevidenced and does not gate. A resemblance that holds only with a learned part fixed,
zeroed or set to the hand witness (`fixed_learned_parts`) is a limit and cannot carry a cited-source kill. The critic
lists a part there only when the card states a capability the frozen form lacks, so a decorative knob does not count.
The textbook_transplant gate is separate: `true` means the source field's construction is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

SOURCE_KINDS = ("paper", "textbook", "standard_algorithm")


@dataclass(frozen=True)
class Judgment:
    kills: bool
    basis: str       # counted_comparator, cited_source, not_claimed, learned_part_fixed, no_evidence
    reason: str


def bare(ref: str) -> str:
    """A paper ref without its source prefix: `s2:712c...` and `712c...` name the same candidate."""
    return ref.split(":", 1)[-1]


def candidate_ids(candidates: list[dict]) -> set[str]:
    return {bare(c["key"]) for c in candidates} | {bare(c["source_id"]) for c in candidates}


def accepted_sources(critic: dict, candidates: list[dict]) -> list[dict]:
    """The cited sources the host accepts: papers present in the search results, named textbooks and algorithms."""
    ids = candidate_ids(candidates)
    return [s for s in critic["known_work_sources"] if s["kind"] != "paper" or bare(s["ref"]) in ids]


def cited_papers(critic: dict, candidates: list[dict]) -> list[str]:
    """Bare ids of the search-result papers the critic cites; each is a required closest_prior entry."""
    return [bare(s["ref"]) for s in accepted_sources(critic, candidates) if s["kind"] == "paper"]


def cited_names(critic: dict, candidates: list[dict]) -> list[dict]:
    """Textbook sections and standard algorithms the critic cites; the challenger checks the statement answers them."""
    return [s for s in accepted_sources(critic, candidates) if s["kind"] != "paper"]


def judge(critic: dict, counted: list[dict], candidates: list[dict]) -> Judgment:
    if not critic["known_work"]:
        return Judgment(False, "not_claimed", "the critic did not find the general mechanism to be known work")
    if counted:
        ids = ", ".join(c["declared_id"] or critic["reduces_to"] for c in counted)
        return Judgment(True, "counted_comparator", f"host-counted comparator: {ids}")
    fixed = critic["fixed_learned_parts"]
    if fixed:
        parts = "; ".join(f"{p['part']} (card states: {p['capability'][:120]})" for p in fixed)
        return Judgment(False, "learned_part_fixed",
                        f"the resemblance holds only with a learned part fixed, zeroed or set to the witness: {parts}")
    sources = accepted_sources(critic, candidates)
    if sources:
        cited = "; ".join(f"{s['kind']} {s['ref']}" for s in sources)
        return Judgment(True, "cited_source", f"cited source covering the mechanism: {cited}")
    return Judgment(False, "no_evidence", "known work claimed without a host-counted comparator or an accepted "
                                          "cited source (a paper id absent from the search results is not accepted)")


def search_listing(candidates: list[dict]) -> str:
    """The search results the critic may cite by id, titles only (the difference writer gets the excerpts)."""
    lines = [f"- {c['key']} ({c['year'] if c['year'] is not None else 'n.d.'}) {c['title']}" for c in candidates]
    return "SEARCH RESULTS (cite a paper by exactly one of these ids):\n" + ("\n".join(lines) or "(no source returned)")
