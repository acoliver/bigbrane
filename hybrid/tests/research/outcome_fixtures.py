"""Fixtures for the outcome store tests: mechanism cards of known families, a renamer, and a small run built through
the real Store so the backfill reads the same layout as runs/."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from hybrid.research.ledger import Ledger
from hybrid.research.store import Store
from hybrid.tests.research.builders import load_example, make_card

REAL_CARDS = Path(__file__).resolve().parent / "fixtures" / "outcomes"


def real_card(card_id: str) -> dict:
    """A card copied from runs/ (latest version) for the signature tests."""
    return json.loads((REAL_CARDS / f"{card_id}.json").read_text(encoding="utf-8"))


def eq(plain: str, latex: str | None = None, name: str = "step") -> dict:
    return {"name": name, "plain": plain, "latex": latex or plain}


def mechanism_card(card_id: str, state: list[tuple[str, list]], update: list[dict], readout: list[dict],
                   learning: str, summary: str, title: str = "Fixture mechanism") -> dict:
    card = make_card(card_id, title=title)
    card["summary"] = summary
    card["core_idea"] = summary.split(".")[0]
    card["mechanism"] = {
        "state": [{"name": name, "type": "array", "shape": shape, "meaning": f"state object {name} of the fixture"}
                  for name, shape in state],
        "update": update, "readout": readout,
        "learning": {"description": learning, "few_shot_argument": "Fixture argument, not used by the signature."},
    }
    return card


# Eight mechanisms from families the plan names (section 1.2 and the planted-card types). Each pair differs in what
# it does, so each must get its own signature.
KNOWN_MECHANISMS = {
    "linear_attention": mechanism_card(
        "card-0000a001", [("S", ["d", "d"])],
        [eq("S = S + outer(v_t, k_t)", r"S_t = S_{t-1} + v_t k_t^\top")],
        [eq("y = S @ q_t", r"y_t = S_t q_t")],
        "Key, value and query projections are trained by gradient descent with backprop.",
        summary="Additive key value outer products summed into a matrix memory. Queries read it linearly."),
    "softmax_attention": mechanism_card(
        "card-0000a002", [("K", ["n", "d"]), ("V", ["n", "d"])],
        [eq("append k_t to K; append v_t to V")],
        [eq("y = softmax(K @ q_t / sqrt(d)) @ V", r"y_t = \mathrm{softmax}(K q_t / \sqrt{d})^\top V")],
        "Key, value and query projections are trained by gradient descent with backprop.",
        summary="Every key and value is kept. Queries weigh all stored keys by exponentiated scores."),
    "delta_rule": mechanism_card(
        "card-0000a003", [("S", ["d", "d"]), ("beta", [1])],
        [eq("S = S + beta * outer(v_t - S @ k_t, k_t)", r"S_t = S_{t-1} + \beta (v_t - S_{t-1} k_t) k_t^\top")],
        [eq("y = S @ q_t", r"y_t = S_t q_t")],
        "Projections and beta are trained by gradient descent with backprop.",
        summary="A matrix memory corrected toward each target by its own prediction error."),
    "ssm": mechanism_card(
        "card-0000a004", [("h", ["d"]), ("A", ["d", "d"])],
        [eq("h = A @ h + B @ x_t", r"h_t = A h_{t-1} + B x_t")],
        [eq("y = C @ h", r"y_t = C h_t")],
        "A, B and C are trained by gradient descent with backprop.",
        summary="A fixed linear recurrence carries a hidden vector forward through time."),
    "union_find": mechanism_card(
        "card-0000a005", [("parent", ["n"])],
        [eq("if x_t is linked to y_t: union(find(x_t), find(y_t))")],
        [eq("predict the label stored at find(q)")],
        "No training: the structure is fixed by hand.",
        summary="Linked symbols join one equivalence class, labels are read from class representatives."),
    "hmm_forward": mechanism_card(
        "card-0000a006", [("alpha", ["k"]), ("A", ["k", "k"]), ("B", ["k", "V"])],
        [eq("alpha = (alpha @ A) * B[:, x_t]; alpha = alpha / alpha.sum()")],
        [eq("p = (alpha @ A) @ B")],
        "A and B are the posterior mean under a Dirichlet prior on transition and emission counts.",
        summary="Hidden state beliefs propagate forward through transition and emission matrices."),
    "ngram_counts": mechanism_card(
        "card-0000a007", [("C", ["V", "V"])],
        [eq("C[x_prev, x_t] += 1")],
        [eq("p = C[x_t] / C[x_t].sum()")],
        "Learning is counting: each bigram increments its count.",
        summary="Pair frequencies tallied in a table give the conditional next symbol distribution."),
    "stack_machine": mechanism_card(
        "card-0000a008", [("stack", ["depth"])],
        [eq("if x_t is an opener: push(x_t) else pop()")],
        [eq("predict the closer that matches the top of the stack")],
        "No training: the transition table is fixed by hand.",
        summary="Opening brackets go on a pushdown store, closers pop the most recent opener."),
}


def _rename_text(text: str, mapping: dict[str, str]) -> str:
    for old in sorted(mapping, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", mapping[old], text)
    return text


def _walk(value, fn):
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_walk(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _walk(item, fn) for key, item in value.items()}
    return value


def renamed(card: dict, *, extra: dict[str, str] | None = None) -> dict:
    """The same mechanism under new names: every state object renamed everywhere it appears (equations, meanings,
    claims), the identifiers in `extra` renamed too, and the title, summary, lens, core idea, equation names (the
    roles of the steps) and the transplant object replaced."""
    out = copy.deepcopy(card)
    mapping = {item["name"]: f"zq{i}w" for i, item in enumerate(card["mechanism"]["state"])}
    mapping.update(extra or {})
    out = _walk(out, lambda text: _rename_text(text, mapping))
    out["title"] = "Entirely Different Title: the Lantern Archivist"
    out["summary"] = "A lantern archivist keeps ledgers for travelling merchants and answers their questions."
    out["lens"] = "a_lens_nobody_used"
    out["core_idea"] = "The archivist and the merchant swap roles each night."
    for section in ("update", "readout"):
        for index, equation in enumerate(out["mechanism"][section]):
            equation["name"] = f"role of the {('keeper', 'courier', 'scribe', 'oracle')[index % 4]} {index}"
    if "source_transplant" in out:
        out["source_transplant"]["object"] = "the archivist's lantern"
    return out


def build_run(run_dir: Path, cards: list[tuple[dict, list[tuple[str, str, list[str]]], dict]]) -> Path:
    """A run directory written through the real Ledger and Store. Each entry: (card, [(stage, verdict, reasons)],
    {artifact kind: JSON value}). Artifacts are written before the verdicts, as the stages do."""
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    for card, verdicts, artifacts in cards:
        card = copy.deepcopy(card)
        card["status"] = {"stage": "S0", "verdicts": {}}
        store.put_card(card)
        ledger.append("card_created", "orchestrator", {"card_id": card["id"], "op": "new", "parents": []})
        for kind, value in artifacts.items():
            store.put_artifact(kind, f"{card['id']}-{kind}.json", json.dumps(value).encode(), card_id=card["id"])
        for stage, verdict, reasons in verdicts:
            store.set_verdict(card["id"], stage, verdict, reasons, ["call-0123456789ab", "job-0123456789ab"])
    ledger.append("llm_call", "llm:fake", {"call_id": "call-0123456789ab", "usage": {"input": 70, "output": 30}})
    return run_dir


def example_variant(card_id: str, title: str) -> dict:
    card = make_card(card_id, title=title)
    card["core_idea"] = load_example().get("core_idea") or "Bind each token to a phase and read the running sum."
    return card
