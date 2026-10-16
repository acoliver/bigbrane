"""Card builders for research tests. Every card starts from the labeled toy fixture."""
import copy
import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "example_card.json"


def load_example() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def make_card(card_id: str, *, version: int = 1, parents=(), title: str | None = None) -> dict:
    card = copy.deepcopy(load_example())
    card["id"] = card_id
    card["version"] = version
    card["parent_ids"] = list(parents)
    if title is not None:
        card["title"] = title
    return card
