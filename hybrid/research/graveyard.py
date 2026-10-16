"""Graveyard: the outcome of every earlier card, compact enough for the S0 explorer prompt.

Each card becomes one line of about 250 characters: title, core idea, and its outcome, which is one of
- reduced: S2 gave reduces_to_known; the known family and one sentence of the critic's argument;
- died: the stage and verdict where the card's line ended, with the deciding reason;
- finished: the card completed S6;
- alive: the card is still moving (alive at Sk).

Reading is read-only and touches only files: ledger.jsonl (parsed line by line, without the ledger
lock, so a run that another process is writing can be read), and the card versions and
reduction_critic artifacts the ledger records. That makes it safe to point `explore.graveyard_runs`
at earlier or still-running runs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ENTRY_CHARS = 250
TITLE_CHARS = 45
IDEA_CHARS = 70
# Order in which entries are kept when the graveyard is longer than explore.graveyard_max.
KIND_RANK = {"reduced": 0, "died": 1, "finished": 2, "alive": 3}
DEAD_VERDICTS = ("fail", "infeasible_proved", "undetermined", "reduces_to_known")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# Opening sentences that only say what the card is not ("This is not a renamed recurrence ...").
DENIAL = re.compile(r"^(This|It|The card) (is|does) not\b|^There is no\b")


@dataclass(frozen=True)
class Grave:
    card_id: str
    title: str
    core_idea: str
    kind: str        # reduced | died | finished | alive
    outcome: str
    order: int       # position of the card's card_created event in its run; higher is newer

    def line(self) -> str:
        idea = clip(self.core_idea, IDEA_CHARS) if self.core_idea else "(no core_idea)"
        return clip(f"- {clip(self.title, TITLE_CHARS)} | idea: {idea} | {self.outcome}", ENTRY_CHARS)


def clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit - 3].rstrip() + "..."


def reduction_summary(why: str) -> str:
    """The first sentence of the critic's argument that says what the card is, skipping opening
    sentences that only say what it is not."""
    sentences = SENTENCE_END.split(" ".join(why.split()))
    return next((s for s in sentences if not DENIAL.match(s)), sentences[0])


def read_events(run_dir: Path) -> list[dict]:
    """Events of ledger.jsonl in order. A last line that another process is still writing is skipped."""
    lines = (Path(run_dir) / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            if index == len(lines) - 1:
                break
            raise
    return events


def deciding_reason(reasons: list[str]) -> str:
    """The gate reason when the stage gave one, else the first reason. A short generic gate line
    ("fatal salad-critic finding") is followed by the reason after it, which names the finding."""
    index = next((i for i, reason in enumerate(reasons) if reason.startswith("gate:")), 0)
    text = reasons[index].removeprefix("gate:").strip()
    if len(text) < 60 and index + 1 < len(reasons):
        text += "; " + reasons[index + 1]
    return text


def outcome(card: dict, reduction: dict | None) -> tuple[str, str]:
    verdicts = card["status"]["verdicts"]
    stage = card["status"]["stage"]
    if verdicts.get("S2", {}).get("verdict") == "reduces_to_known":
        if reduction is None:
            raise ValueError(f"{card['id']} reduces_to_known at S2 but has no reduction_critic artifact")
        return "reduced", f"reduced at S2 to {reduction['reduces_to']}: {reduction_summary(reduction['why'])}"
    current = verdicts.get(stage)
    if current is None:
        return "alive", f"alive at {stage}"
    if current["verdict"] in DEAD_VERDICTS:
        return "died", f"died at {stage} ({current['verdict']}): {deciding_reason(current['reasons'])}"
    if stage == "S6":
        return "finished", f"finished S6: {current['reasons'][0]}"
    return "alive", f"alive at {stage}" + (" (revising)" if current["verdict"] == "revise" else "")


def graves(run_dir: Path) -> list[Grave]:
    """One Grave per card in the run, oldest first. Cards and artifacts are found through the ledger
    (card_version and artifact events), which the store appends only after the file is written."""
    run_dir = Path(run_dir)
    order: dict[str, int] = {}
    latest: dict[str, str] = {}
    reductions: dict[str, str] = {}
    for index, event in enumerate(read_events(run_dir)):
        data = event["data"]
        if event["type"] == "card_version":
            order.setdefault(data["card_id"], index)
            latest[data["card_id"]] = data["path"]
        elif event["type"] == "artifact" and data["kind"] == "reduction_critic" and data.get("card_id"):
            reductions[data["card_id"]] = data["path"]
    out = []
    for card_id, path in latest.items():
        card = json.loads((run_dir / path).read_text(encoding="utf-8"))
        reduction_path = reductions.get(card_id)
        reduction = json.loads((run_dir / reduction_path).read_text(encoding="utf-8")) if reduction_path else None
        kind, text = outcome(card, reduction)
        out.append(Grave(card_id, card["title"], card.get("core_idea", ""), kind, text, order[card_id]))
    return sorted(out, key=lambda grave: grave.order)


def select(current: list[Grave], earlier: list[Grave], limit: int) -> tuple[list[Grave], list[Grave]]:
    """Up to `limit` entries: reductions first, then deaths, finished and alive cards; within a kind this
    run before earlier runs and newest first. Returns (this run's entries, earlier runs' entries)."""
    ranked = sorted([(grave, 0) for grave in current] + [(grave, 1) for grave in earlier],
                    key=lambda item: (KIND_RANK[item[0].kind], item[1], -item[0].order))[:limit]
    return [g for g, source in ranked if source == 0], [g for g, source in ranked if source == 1]


def render(current: list[Grave], earlier: list[Grave], limit: int) -> str:
    mine, theirs = select(current, earlier, limit)
    if not mine and not theirs:
        return "(none yet)"
    parts = []
    if mine:
        parts.append("This run:\n" + "\n".join(grave.line() for grave in mine))
    if theirs:
        parts.append("Earlier runs:\n" + "\n".join(grave.line() for grave in theirs))
    return "\n".join(parts)
