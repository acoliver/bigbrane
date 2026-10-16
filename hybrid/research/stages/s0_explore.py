"""S0 explore: one explorer call produces one theory card from a lens, papers and an operator.

Every explorer prompt carries the graveyard: the outcome of earlier cards in this run and in the runs named
by explore.graveyard_runs, so explorers see which families earlier cards were reduced to. The `escape`
operator takes a card that S2 reduced to a known family and asks for a mechanism that breaks the property
the reduction relies on; its system prompt adds prompts/explorer_escape.md to the explorer prompt, and its
draft must carry a non_reduction_argument.

With outcomes.enabled, the graveyard block also lists the dead ideas from the outcome store nearest to the task
(hybrid/research/outcomes.py `nearby_dead`: the lens, the papers and any parent cards against each record's idea,
the parents' signature tokens, and the lens), each with its kill codes and cause ids. Plan section 9 asks for the
causes of the target archive cell and its neighbours; in lens mode the lens and the text stand in for them. With the
switch off the prompt is byte for byte what it was.

Generation mode "arms" (Phase 6, `[generation] mode = "arms"`): `explore` hands every task to
`stages/g_generate.generate`, which drafts through a generation arm, classifies the card into the archive and shows
the archive's visited cells with their recorded causes. Both modes end in `admit` here, which stores the card and
gives its S0 verdict.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path

from hybrid.literature.paper import Paper
from hybrid.literature.sampling import sample_for_lens
from hybrid.literature.seed import KNOWN_ALTERNATIVES_LENS, LENSES, load_seed
from hybrid.research import graveyard, lineage
from hybrid.research import outcomes as outcome_store
from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.stages.common import (StageContext, StageError, card_for_prompt,
                                           card_text_for_similarity, draft_schema, ground_draft, load_prompt,
                                           load_section, new_card_id, parsimony)
from hybrid.research.stages.diversity import most_similar
from hybrid.research.stages.escape import escape_chain, parent_reduction_block

OPERATORS = ("new", "mutate", "cross", "escape")
PARENTS_PER_OPERATOR = {"new": 0, "mutate": 1, "cross": 2, "escape": 1}
GRAVEYARD_HEADER = ("GRAVEYARD (earlier cards and how they ended, reductions first). A card reduced at S2 marks a "
                    "basin: a new card whose state update or retrieval has the same form will be reduced the same "
                    "way. Do not resubmit these ideas under new names:")


@dataclass
class ExploreTask:
    op: str
    lens: str                      # lens or pairing name, recorded on the card
    lens_sources: list[str]        # seed lenses to draw papers from
    role: str                      # explorer | explorer_alt
    explorer: int
    seed: int
    parents: list[str] = field(default_factory=list)
    # Set by explore() once the card exists, so the orchestrator can record the outcome of a duplicate.
    card_id: str | None = None
    # Generation arm (generation.mode = "arms", Phase 6): set on new tasks only; mutate, cross and escape carry none.
    arm: str | None = None

    def __post_init__(self):
        if self.op not in OPERATORS:
            raise ValueError(f"unknown operator {self.op!r}")
        if self.arm is not None and self.op != "new":
            raise ValueError(f"only a new task carries a generation arm, not {self.op} ({self.arm})")
        expected = PARENTS_PER_OPERATOR[self.op]
        if len(self.parents) != expected:
            raise ValueError(f"operator {self.op} needs {expected} parents, got {self.parents}")


def _unique(papers: list[Paper], seen: set | None = None) -> list[Paper]:
    seen = set() if seen is None else seen
    unique = []
    for paper in papers:
        key = (paper.source, paper.source_id)
        if key not in seen:
            seen.add(key)
            unique.append(paper)
    return unique


def sample_papers(task: ExploreTask, k: int) -> list[Paper]:
    """k papers split over the task's lenses, the remainder going to the first lenses: a pairing asked for 3 papers
    draws 2 + 1. (Flooring k / 2 gave 1 + 1 = 2 < papers_min and lost the slot: Phase 6 run
    phase6-drafts-20261013T0611Z, ledger line 217, `only 2 papers for lens pair:kernel+program_induction`.)"""
    papers: list[Paper] = []
    count = len(task.lens_sources)
    for offset, lens in enumerate(task.lens_sources):
        share = max(1, k // count + (1 if offset < k % count else 0))
        papers.extend(sample_for_lens(lens, share, task.seed + offset))
    return _unique(papers)


def neighbour_lenses(sources: list[str]) -> list[str]:
    """The lenses next to each source lens in the seed registry order (seed.LENSES, wrapping), without the sources
    and the known-alternatives lens."""
    order = [lens for lens in LENSES if lens != KNOWN_ALTERNATIVES_LENS]
    out: list[str] = []
    for lens in sources:
        if lens not in order:
            continue
        i = order.index(lens)
        for near in (order[(i - 1) % len(order)], order[(i + 1) % len(order)]):
            if near not in sources and near not in out:
                out.append(near)
    return out


def top_up_papers(task: ExploreTask, papers: list[Paper], k: int) -> list[tuple[Paper, str]]:
    """Papers to add when the task's lenses gave fewer than k: first from the neighbouring lenses, then from the seed
    reading list of every explore lens. Each comes with its origin (`neighbour:<lens>` or `seed_list:<lens>`).
    Deterministic in the task seed."""
    seen = {(paper.source, paper.source_id) for paper in papers}
    added: list[tuple[Paper, str]] = []
    for offset, lens in enumerate(neighbour_lenses(task.lens_sources)):
        if len(papers) + len(added) >= k:
            return added
        drawn = _unique(sample_for_lens(lens, k - len(papers) - len(added), task.seed + 1000 + offset), seen)
        added += [(paper, f"neighbour:{lens}") for paper in drawn]
    if len(papers) + len(added) < k:
        entries = sorted((entry for entry in load_seed() if entry.lens != KNOWN_ALTERNATIVES_LENS),
                         key=lambda entry: (entry.lens, entry.paper.source, entry.paper.source_id))
        random.Random(f"seed-list:{task.seed}").shuffle(entries)
        for entry in entries:
            if len(papers) + len(added) >= k:
                break
            if _unique([entry.paper], seen):
                added.append((entry.paper, f"seed_list:{entry.lens}"))
    return added[:k - len(papers)]


def draw_papers(ctx: StageContext, task: ExploreTask, k: int) -> list[Paper]:
    """The task's papers. When its lenses give fewer than explore.papers_min, the draw is topped up to k from the
    neighbouring lenses and then the seed list (`top_up_papers`), and a `paper_fallback` event records what was
    added and from where, so the slot still drafts. StageError only if even the fallback stays short."""
    minimum = ctx.cfg["explore"]["papers_min"]
    papers = sample_papers(task, k)
    if len(papers) >= minimum:
        return papers
    added = top_up_papers(task, papers, k)
    ctx.emit("paper_fallback", {
        "lens": task.lens, "lens_sources": list(task.lens_sources), "arm": task.arm, "op": task.op,
        "explorer": task.explorer, "requested": k, "papers_min": minimum, "from_lens": len(papers),
        "added": [{"paper": f"{paper.source}:{paper.source_id}", "origin": origin} for paper, origin in added]})
    papers = papers + [paper for paper, _ in added]
    if len(papers) < minimum:
        raise StageError(f"only {len(papers)} papers for lens {task.lens}, after the neighbouring-lens and seed-list "
                         f"fallback")
    return papers


def known_alternatives() -> list[str]:
    return [f"{entry.paper.title} ({entry.paper.year}): {entry.relevance}"
            for entry in load_seed() if entry.lens == KNOWN_ALTERNATIVES_LENS]


def _paper_block(papers: list[Paper], chars: int) -> str:
    lines = []
    for paper in papers:
        abstract = (paper.abstract or "(no abstract)").strip()[:chars]
        lines.append(f"- source_id: {paper.source_id} ({paper.source}, {paper.year or 'n.d.'})\n"
                     f"  title: {paper.title}\n  abstract: {abstract}")
    return "\n".join(lines)


def existing_summaries(ctx: StageContext, limit: int, chars: int) -> str:
    rows = []
    for card_id in ctx.store.list_cards()[-limit:]:
        card = ctx.store.get_card(card_id)
        stage = card["status"]["stage"]
        verdict = card["status"]["verdicts"].get(stage, {}).get("verdict", "-")
        rows.append(f"- {card_id} [{card['lens']}; {stage} {verdict}] {card['title']}: {card['summary'][:chars]}")
    return "\n".join(rows) or "(none yet)"


def graveyard_block(ctx: StageContext) -> str:
    """This run's cards and those of explore.graveyard_runs, bounded by explore.graveyard_max."""
    earlier = [grave for run_dir in ctx.graveyard_runs for grave in graveyard.graves(run_dir)]
    return graveyard.render(graveyard.graves(ctx.run_dir), earlier, ctx.cfg["explore"]["graveyard_max"])


def task_text(ctx: StageContext, task: ExploreTask, papers: list[Paper]) -> str:
    """What a generation task is about, for the outcome-store query: lens, papers and parent cards."""
    parts = [task.lens, ctx.lenses[task.lens]["description"]]
    parts += [f"{paper.title} {(paper.abstract or '')[:600]}" for paper in papers]
    parts += [card_text_for_similarity(ctx.store.get_card(parent)) for parent in task.parents]
    return "\n".join(parts)


def nearby_dead(ctx: StageContext, task: ExploreTask, papers: list[Paper]) -> list:
    """Step 1 query: the dead ideas of earlier runs nearest to this task (this run's own cards are in the graveyard
    block already, and so are the runs listed in explore.graveyard_runs)."""
    from hybrid.research.signature import card_signature

    tokens = sorted({token for parent in task.parents
                     for token in card_signature(ctx.store.get_card(parent))["tokens"]}) or None
    exclude = (ctx.run_id, *(Path(run).name for run in ctx.graveyard_runs))
    return outcome_store.nearby_dead(ctx.outcomes, text=task_text(ctx, task, papers), tokens=tokens, lens=task.lens,
                                     limit=ctx.cfg["outcomes"]["prompt_max"], exclude_runs=exclude)


def system_prompt(task: ExploreTask) -> str:
    base = load_prompt("explorer")
    if task.op != "escape":
        return base
    return base.rstrip() + "\n\n" + load_prompt("explorer_escape")


def build_user_prompt(ctx: StageContext, task: ExploreTask, papers: list[Paper], nearby: list | None = None) -> str:
    ex = ctx.cfg["explore"]
    lens_info = ctx.lenses[task.lens]
    parts = [
        f"OPERATOR: {task.op}",
        f"LENS: {task.lens}\n{lens_info['description']}",
        "PAPERS (cite by source_id in references when you use one):\n" + _paper_block(papers, ex["abstract_chars"]),
        "KNOWN ALTERNATIVES TO AVOID REINVENTING:\n" + "\n".join(f"- {line}" for line in known_alternatives()),
        "CARDS ALREADY IN THIS RUN (do not repeat them):\n"
        + existing_summaries(ctx, ex["existing_summaries"], ex["summary_chars"]),
        GRAVEYARD_HEADER + "\n" + graveyard_block(ctx),
    ]
    if nearby is not None:
        parts.append(load_section("outcome_store") + "\n" + outcome_store.render_nearby(nearby))
    for label, parent in zip(("PARENT CARD A", "PARENT CARD B"), task.parents):
        parent_card = ctx.store.get_card(parent)
        parts.append(f"{label} ({parent}):\n{card_for_prompt(parent_card)}")
        verdicts = parent_card["status"]["verdicts"]
        if verdicts:
            notes = "\n".join(f"  {stage}: {v['verdict']}: {'; '.join(v['reasons'])[:800]}"
                              for stage, v in sorted(verdicts.items()))
            parts.append(f"{label} VERDICTS SO FAR:\n{notes}")
    if task.op == "escape":
        parts.append(parent_reduction_block(ctx, task.parents[0]))
        parts.append("Return one theory card draft as JSON that escapes this reduction, with a "
                     "non_reduction_argument. Do not include id, version, parent_ids, created_by_call, lens or "
                     "status; the orchestrator fills them in.")
    else:
        parts.append("Return one theory card draft as JSON. Do not include id, version, parent_ids, "
                     "created_by_call, lens or status; the orchestrator fills them in.")
    return "\n\n".join(parts)


async def explore(ctx: StageContext, task: ExploreTask) -> str | None:
    if ctx.generation is not None:
        # Generation mode "arms" (Phase 6): arms, classifier, archive (hybrid/research/stages/g_generate.py).
        from hybrid.research.stages.g_generate import generate

        return await generate(ctx, task)
    ex = ctx.cfg["explore"]
    k = ex["papers_min"] + (task.seed % (ex["papers_max"] - ex["papers_min"] + 1))
    papers = draw_papers(ctx, task, k)
    nearby = nearby_dead(ctx, task, papers) if ctx.outcomes is not None else None
    user = build_user_prompt(ctx, task, papers, nearby)
    result = await ctx.call(task.role, system_prompt(task), user, draft_schema(escape=task.op == "escape"))
    card, dropped, dropped_sources = card_from_draft(ctx, task, result, papers)
    admitted = admit(ctx, task, card, call_ids=[result.call_id], model=result.model, papers=papers, nearby=nearby,
                     user=user, dropped=dropped, dropped_sources=dropped_sources)
    return card["id"] if admitted.passed else None


def card_from_draft(ctx: StageContext, task: ExploreTask, result, papers: list[Paper],
                    extra: dict | None = None) -> tuple[dict, list[dict], list[str]]:
    """The card for an explorer draft: references and transplant sources grounded in the given papers (and the
    parents' references), orchestrator fields filled in, validated against theory_card (StageError otherwise)."""
    allowed = {paper.source_id for paper in papers}
    for parent in task.parents:
        allowed |= {ref["source_id"] for ref in ctx.store.get_card(parent)["references"]}
    draft, dropped, dropped_sources = ground_draft(result.parsed, allowed)
    card = dict(draft, **(extra or {}), id=new_card_id(), version=1, parent_ids=list(task.parents),
                created_by_call=result.call_id, lens=task.lens, status={"stage": "S0", "verdicts": {}})
    try:
        validate("theory_card", card)
    except SchemaViolation as exc:
        raise StageError(f"explorer card fails theory_card schema: {exc}") from exc
    return card, dropped, dropped_sources


@dataclass(frozen=True)
class Admission:
    passed: bool
    # None when the card passed S0; else the kill code it failed with ("duplicate:<card>", a G gate's code, or the
    # budget stop's code).
    kill_code: str | None
    nearest: str | None
    cosine: float


def admit(ctx: StageContext, task: ExploreTask, card: dict, *, call_ids: list[str], model: str | None,
          papers: list[Paper], nearby: list | None, user: str, dropped: list[dict], dropped_sources: list[str],
          gate: tuple[str, str] | None = None, created_extra: dict | None = None,
          extra_reasons: list[str] | None = None) -> Admission:
    """Store a drafted card and give it its S0 verdict: a gate decided before (generation stage G: kill code and
    reason), else a TF-IDF duplicate, else a lineage at its idea cap or out of model calls, else pass. The card's
    model calls are charged to its idea and lineage."""
    # An escape child reuses its parent's vocabulary by design, so the parent and its escape ancestors are not
    # compared against it. Mutate and cross children are compared with every card.
    excluded = escape_chain(ctx.ledger, task.parents[0]) if task.op == "escape" else []
    existing = {card_id: card_text_for_similarity(ctx.store.get_card(card_id))
                for card_id in ctx.store.list_cards() if card_id not in excluded}
    nearest, cosine = most_similar(card_text_for_similarity(card), existing)
    chain_reasons = ([f"escape: parent and escape ancestors excluded from the nearest-card comparison: "
                      f"{', '.join(excluded)}"] if excluded else [])
    ctx.store.put_card(card)
    task.card_id = card["id"]
    store_fields = ({"outcome_store_nearby": [match.stored.record_id for match in nearby]}
                    if nearby is not None else {})
    ctx.emit("card_created", {
        "card_id": card["id"], "op": task.op, "lens": task.lens, "role": task.role, "explorer": task.explorer,
        "parents": list(task.parents), "call_id": call_ids[0], "model": model,
        "papers": [f"{p.source}:{p.source_id}" for p in papers],
        "dropped_references": dropped, "dropped_transplant_sources": dropped_sources,
        "source_transplant": card["source_transplant"]["object"] if "source_transplant" in card else None,
        **parsimony(card),
        "nearest": nearest, "cosine": round(cosine, 4),
        "prompt_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(), **store_fields, **(created_extra or {}),
    })
    threshold = ctx.cfg["diversity"]["threshold"]
    duplicate = nearest is not None and cosine >= threshold
    # Lineage and idea (Phase 8): a child joins its parent's lineage, a card with a known signature joins that
    # signature's lineage, and a lineage at budgets.lineage.ideas admits no more cards. The explorer call is charged
    # to the new card.
    assignment = ctx.book.assign(card, op=task.op, parents=list(task.parents), admit=not duplicate and gate is None)
    for call_id in call_ids:
        ctx.book.charge_call(card["id"], call_id, "S0")
    if gate is not None:
        code, reason = gate
        ctx.verdict(card["id"], "S0", "fail", [f"gate: {code}: {reason}"] + chain_reasons, list(call_ids))
        return Admission(False, code, nearest, cosine)
    if duplicate:
        ctx.emit("duplicate", {"card_id": card["id"], "nearest": nearest, "cosine": round(cosine, 4),
                               "threshold": threshold})
        reason = f"duplicate: TF-IDF cosine {cosine:.3f} with {nearest} is at or above {threshold}"
        ctx.verdict(card["id"], "S0", "fail", [reason] + chain_reasons, list(call_ids))
        return Admission(False, f"duplicate:{nearest}", nearest, cosine)
    # A lineage at its idea cap admits no more cards; an idea or lineage whose model calls are used up (a renamed copy
    # of a costly idea, or this very call reaching the cap) can make no further call, so the card ends here as well.
    stop = ctx.book.room(card["id"]) if not assignment.admitted else ctx.book.calls_stop(card["id"])
    if stop is not None:
        lineage.record_stop(ctx.ledger, card_id=card["id"], stage="S0", stop=stop, terminal=True)
        how = f"joined by {assignment.joined}" + (f" ({assignment.via})" if assignment.via else "")
        ctx.verdict(card["id"], "S0", "fail", [f"gate: {stop.kill_code}: {stop.detail}; this card {how}",
                                               *chain_reasons], list(call_ids))
        return Admission(False, stop.kill_code, nearest, cosine)
    reasons = (["card validates against the theory_card schema",
                f"nearest existing card {nearest or '-'} at TF-IDF cosine {cosine:.3f} (threshold {threshold})"]
               + chain_reasons + list(extra_reasons or []))
    if dropped:
        reasons.append(f"dropped {len(dropped)} reference(s) to papers the explorer was not given: "
                       + ", ".join(ref["source_id"] for ref in dropped))
    if dropped_sources:
        reasons.append("dropped transplant source(s) the explorer was not given: " + ", ".join(dropped_sources))
    ctx.verdict(card["id"], "S0", "pass", reasons, list(call_ids))
    return Admission(True, None, nearest, cosine)


def lens_seed(run_id: str, index: int) -> int:
    return int(hashlib.sha256(f"{run_id}:{index}".encode()).hexdigest()[:8], 16)
