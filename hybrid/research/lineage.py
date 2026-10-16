"""Budgets and stall detection (DISCOVERY-PIPELINE-PLAN.md section 10, Phase 8).

Every loop that can repeat work on one idea has a cap, and a revision loop whose rounds stop changing ends early.

Ideas and lineages. An idea is keyed by the mechanism signature its card had when it was created
(`hybrid.research.signature`), so a card that renames an earlier card's mechanism is the same idea and shares its
counts. A lineage is a root idea plus every child made from it (mutate, cross, escape, and the repair children of
later phases): a child inherits its first parent's lineage, and a new card whose signature is already known joins the
lineage that signature belongs to. With the outcome store on, signatures and lineages of earlier runs count too, so a
"what if we tried" restart of a recorded idea in a new run inherits the recorded counts instead of starting fresh.

Caps (run.toml):
- per idea, `[budgets.idea]`: `model_calls` (every model call made while a stage works on the card), and the
  revision loops named in IDEA_LOOPS. The S1 revision loop keeps its existing key `s1.max_revisions`; the loops of
  later phases (refinement dialogue, prior-art send-back, math revision) have their keys here already and charge
  the book by naming the loop.
- per lineage, `[budgets.lineage]`: `ideas` (cards admitted into the lineage) and `model_calls`.
- prover attempts per statement stay `math.max_prover_attempts`, counted by stage M from its `proof_attempt` events.

Plugging a loop in. A stage with a revision loop calls `BudgetBook.after_round` after every round that still fails,
naming its loop, the round and the round's failures; it revises only when that returns None. S1 does this for
`s1_revision`, Phase 7's refinement dialogue for `refinement` (stages/r_refine.py) and the S2 difference statement's
one send-back for `send_back` (stages/difference.py); stage V's math revisions (`math_revision`) need only that call. A granted `send_back` round also takes an idea slot of the lineage
(plan section 8, step 5), so a lineage at its idea cap grants no send-back. A repair checked with `repair_scope` that
touches ids its failures did not name is rejected without running checks (`reject_repair`, event `repair_rejected`)
and its round counts with the old failures plus the out-of-scope ids.

Stall. After each round of a revision loop the set of failing ids and a hash of each failure's detail (a
counterexample, the repeated proof, the toy-example mismatch) is recorded as a `loop_round` event. A round whose
set and hashes equal the previous round's for the same loop key ends the loop with `stall`.

Kill codes. A loop ended by its cap records `budget:<loop>`, one ended by a stall `stall:<loop>`, in a `loop_stop`
event. `terminal: true` means the stop itself ended the card (a model-call cap reached mid-stage, a lineage at its
idea cap at S0); otherwise the stage's gate decided afterwards, and the code is added to the card's kill codes when
the card dies at that stage (`outcome_records.classify`).

Ledger events: `lineage_assigned`, `budget_charge`, `loop_round`, `loop_stop`, `repair_rejected`, `mutation_skipped`. A resumed run
rebuilds every count from them. A run started before Phase 8 has none: its cards are assigned lineages at resume
(`lineage_assigned` with `legacy: true`), and the S1 revisions they used before that are counted from their S1
`revise` verdicts, so no revision is granted twice.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from hybrid.research.ledger import Ledger
from hybrid.research.signature import card_signature

ACTOR = "orchestrator"
IDEA_DEFAULTS = {"refinement_rounds": 3, "send_backs": 1, "math_revision_rounds": 3, "model_calls": 80}
LINEAGE_DEFAULTS = {"ideas": 4, "model_calls": 200}
# Lowest allowed value of each key: a revision loop may be switched off (0), a call or idea cap may not.
IDEA_MINIMUM = {"refinement_rounds": 0, "send_backs": 0, "math_revision_rounds": 0, "model_calls": 1}
LINEAGE_MINIMUM = {"ideas": 1, "model_calls": 1}
# Per-idea revision loops and the config key holding each one's cap. A loop charges the book with its name.
IDEA_LOOPS = {
    "s1_revision": ("s1", "max_revisions"),
    "refinement": ("budgets.idea", "refinement_rounds"),
    "send_back": ("budgets.idea", "send_backs"),
    "math_revision": ("budgets.idea", "math_revision_rounds"),
}
# Loops whose every granted round also takes an idea slot of the card's lineage (plan section 8, step 5: "the
# send-back costs a slot from the lineage budget"). A lineage at budgets.lineage.ideas grants no such round.
LINEAGE_SLOT_LOOPS = frozenset({"send_back"})
# The failure key a rejected repair adds to its round (see reject_repair).
REPAIR_SCOPE = "repair_scope"
# Loop names of the other caps and stops.
MODEL_CALLS = "model_calls"
LINEAGE_IDEAS = "lineage_ideas"
LINEAGE_CALLS = "lineage_calls"
ESCAPE = "escape"
MUTATION = "mutation"
PROVER = "prover"
REASONS = ("budget", "stall")


# ----- config ------------------------------------------------------------------------------------


def fill_defaults(cfg: dict) -> None:
    """Add `[budgets.idea]` and `[budgets.lineage]` keys a config file leaves out (every config before Phase 8)."""
    for name, defaults in (("idea", IDEA_DEFAULTS), ("lineage", LINEAGE_DEFAULTS)):
        table = cfg["budgets"].setdefault(name, {})
        for key, value in defaults.items():
            table.setdefault(key, value)


def validate_budgets(cfg: dict) -> None:
    for name, minimum in (("idea", IDEA_MINIMUM), ("lineage", LINEAGE_MINIMUM)):
        table = cfg["budgets"][name]
        unknown = set(table) - set(minimum)
        if unknown:
            raise ValueError(f"unknown [budgets.{name}] keys {sorted(unknown)}")
        for key, low in minimum.items():
            value = table[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"budgets.{name}.{key} must be an integer >= {low}, got {value!r}")


def slot_name(card_id: str, loop: str, n: int) -> str:
    """The lineage member that stands for the n-th round of a LINEAGE_SLOT_LOOPS loop on a card."""
    return f"{card_id}#{loop}{n}"


def loop_cap(cfg: dict, loop: str) -> int:
    section, key = IDEA_LOOPS[loop]
    table = cfg
    for part in section.split("."):
        table = table[part]
    return table[key]


# ----- stops and stalls ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopStop:
    loop: str
    reason: str      # "budget" or "stall"
    detail: str

    def __post_init__(self):
        if self.reason not in REASONS:
            raise ValueError(f"a loop stops by {REASONS}, not {self.reason!r}")

    @property
    def kill_code(self) -> str:
        return f"{self.reason}:{self.loop}"

    def line(self) -> str:
        return f"loop {self.loop} ended: {self.kill_code}: {self.detail}"


class LoopBudgetExhausted(Exception):
    """A card reached a model-call cap (its idea's or its lineage's) while a stage worked on it. The orchestrator
    ends the card at that stage with the stop's kill code."""

    def __init__(self, card_id: str, stop: LoopStop):
        super().__init__(f"{card_id}: {stop.kill_code}: {stop.detail}")
        self.card_id = card_id
        self.stop = stop


def detail_hash(value: Any) -> str | None:
    if value is None:
        return None
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def fingerprint(failures: Mapping[str, Any]) -> str:
    """The failing ids and the hash of each one's detail, as one hash."""
    pairs = sorted((key, detail_hash(value)) for key, value in failures.items())
    return hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()[:16]


def is_stall(previous: str | None, failures: Mapping[str, Any]) -> bool:
    """A round stalls when something still fails and the failures are exactly those of the previous round."""
    return bool(failures) and previous is not None and previous == fingerprint(failures)


def decide_round(loop: str, *, used: int, cap: int, stalled: bool, failing: Mapping[str, Any]) -> LoopStop | None:
    """After a failing round of a revision loop: the stop that ends the loop, or None to grant one more revision.
    A stall wins over the cap because it names the more specific cause."""
    if stalled:
        return LoopStop(loop, "stall", "the revision left the same failures: " + ", ".join(sorted(failing)))
    if used >= cap:
        return LoopStop(loop, "budget", f"{used} of {cap} revision(s) used")
    return None


def observe_round(ledger: Ledger, *, card_id: str, stage: str, loop: str, key: str, round_id: str,
                  failures: Mapping[str, Any]) -> bool:
    """Record one round of a loop (`loop_round`) and return whether it stalled against the previous round of the same
    loop and key. A round id already recorded (a resumed stage seeing the same round again) is not recorded twice;
    its recorded answer is returned."""
    rounds = [event["data"] for event in ledger.iter_events("loop_round")
              if event["data"]["loop"] == loop and event["data"]["key"] == key]
    for recorded in rounds:
        if recorded["round_id"] == round_id:
            return recorded["stalled"]
    previous = rounds[-1] if rounds else None
    stalled = is_stall(previous["fingerprint"] if previous else None, failures)
    ledger.append("loop_round", ACTOR, {
        "card_id": card_id, "stage": stage, "loop": loop, "key": key, "round_id": round_id,
        "failing": sorted(failures), "details": {k: detail_hash(v) for k, v in sorted(failures.items())},
        "fingerprint": fingerprint(failures), "previous": previous["round_id"] if previous else None,
        "stalled": stalled})
    return stalled


def record_stop(ledger: Ledger, *, card_id: str, stage: str, stop: LoopStop, terminal: bool) -> dict:
    return ledger.append("loop_stop", ACTOR, {"card_id": card_id, "stage": stage, "loop": stop.loop,
                                              "reason": stop.reason, "kill_code": stop.kill_code,
                                              "detail": stop.detail, "terminal": terminal})


def repair_scope(before: Mapping[str, Any], after: Mapping[str, Any], named: Iterable[str]) -> list[str]:
    """The ids a repair changed, added or removed although the failure it answers did not name them (plan section 10:
    "a repair that changes anything other than what the failure names is rejected without running checks").
    `before` and `after` map ids (obligations, statements, premises) to their content."""
    allowed = set(named)
    changed = {key for key in set(before) | set(after)
               if (key in before) != (key in after) or detail_hash(before[key]) != detail_hash(after[key])}
    return sorted(changed - allowed)


def reject_repair(ledger: Ledger, *, card_id: str, stage: str, loop: str, round_id: str, outside: list[str],
                  failures: Mapping[str, Any]) -> dict:
    """Record a repair rejected for touching `outside` (`repair_rejected`) and return the failures its round counts
    as: the failures it answered, which still stand since no check ran, plus the out-of-scope ids. The caller passes
    them to BudgetBook.after_round, so a rejected repair uses a round, and the same rejected repair twice stalls."""
    if not outside:
        raise ValueError("reject_repair needs the ids the repair touched outside the named failures")
    ledger.append("repair_rejected", ACTOR, {"card_id": card_id, "stage": stage, "loop": loop, "round_id": round_id,
                                             "outside": sorted(outside), "named": sorted(failures)})
    return {**failures, REPAIR_SCOPE: sorted(outside)}


# ----- the book ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Assignment:
    card_id: str
    lineage_id: str
    idea_key: str
    joined: str          # "parent", "signature" or "new"
    via: str | None      # the parent card, or the lineage the signature belonged to
    admitted: bool


class BudgetBook:
    """Counts per idea and per lineage, rebuilt from the run ledger (and the outcome store of earlier runs when
    given), and written back as ledger events as the run charges them."""

    def __init__(self, cfg: dict, ledger: Ledger, store, *, run_id: str, outcomes=None):
        self.ledger = ledger
        self.store = store
        self.run_id = run_id
        self.idea_caps = dict(cfg["budgets"]["idea"])
        self.lineage_caps = dict(cfg["budgets"]["lineage"])
        self.loop_caps = {loop: loop_cap(cfg, loop) for loop in IDEA_LOOPS}
        self.assigned: dict[str, Assignment] = {}
        self.idea_used: Counter = Counter()      # (idea key, loop) -> revisions used
        self.card_used: Counter = Counter()      # (card id, loop) -> revisions charged to this card
        self.idea_calls: Counter = Counter()     # idea key -> model calls
        self.lineage_calls: Counter = Counter()  # lineage id -> model calls
        self.lineage_ideas: dict[str, set[str]] = {}
        self.sig_lineage: dict[str, str] = {}
        self.prior = {"records": 0, "lineages": 0}
        if outcomes is not None:
            self._seed(outcomes)
        self._replay()

    # --- rebuilding ----------------------------------------------------------------------------

    def _seed(self, outcomes) -> None:
        """Signatures, lineages and counts of earlier runs from the outcome store (this run's records are left out:
        the ledger holds them). A record without a `budget` block (backfilled, or written before Phase 8) counts its
        recorded model calls and no revisions."""
        lineages = set()
        for stored in outcomes.current():
            record = stored.record
            if record["run"] == self.run_id:
                continue
            budget = record.get("budget") or {}
            sig = record["mechanism_signature"]["signature"]
            lineage = budget.get("lineage_id") or record["lineage_id"]
            idea = budget.get("idea_key") or sig
            self.sig_lineage.setdefault(sig, lineage)
            self.lineage_ideas.setdefault(lineage, set()).add(record["card_id"])
            calls = budget["model_calls"] if budget else record["cost"]["model_calls"]
            self.lineage_calls[lineage] += calls
            self.idea_calls[idea] += calls
            for loop, used in budget.get("loops", {}).items():
                self.idea_used[(idea, loop)] += used
                if loop in LINEAGE_SLOT_LOOPS:
                    self.lineage_ideas[lineage].update(slot_name(record["card_id"], loop, n)
                                                       for n in range(1, used + 1))
            lineages.add(lineage)
            self.prior["records"] += 1
        self.prior["lineages"] = len(lineages)

    def _replay(self) -> None:
        events = list(self.ledger.iter_events())
        revisions: dict[str, list[int]] = {}
        for event in events:
            data = event["data"]
            if event["type"] == "verdict" and data["stage"] == "S1" and data["verdict"] == "revise":
                revisions.setdefault(data["card_id"], []).append(event["seq"])
        self._legacy_revisions = revisions
        for event in events:
            data = event["data"]
            if event["type"] == "lineage_assigned":
                self._apply(Assignment(data["card_id"], data["lineage_id"], data["idea_key"], data["joined"],
                                       data["via"], data["admitted"]), event["seq"])
            elif event["type"] == "budget_charge":
                card = self.assigned[data["card_id"]]
                if data["loop"] == MODEL_CALLS:
                    self.idea_calls[card.idea_key] += 1
                    self.lineage_calls[card.lineage_id] += 1
                else:
                    self._count_round(card, data["loop"])

    def _count_round(self, card: Assignment, loop: str) -> None:
        self.idea_used[(card.idea_key, loop)] += 1
        self.card_used[(card.card_id, loop)] += 1
        if loop in LINEAGE_SLOT_LOOPS:
            self.lineage_ideas.setdefault(card.lineage_id, set()).add(
                slot_name(card.card_id, loop, self.card_used[(card.card_id, loop)]))

    def _apply(self, assignment: Assignment, seq: int | None) -> None:
        self.assigned[assignment.card_id] = assignment
        self.sig_lineage.setdefault(assignment.idea_key, assignment.lineage_id)
        if assignment.admitted:
            self.lineage_ideas.setdefault(assignment.lineage_id, set()).add(assignment.card_id)
        # S1 revisions a card used before it had an assignment (a run started before Phase 8) were never charged.
        legacy = [s for s in self._legacy_revisions.get(assignment.card_id, []) if seq is None or s < seq]
        if legacy:
            self.idea_used[(assignment.idea_key, "s1_revision")] += len(legacy)
            self.card_used[(assignment.card_id, "s1_revision")] += len(legacy)

    # --- assignment ----------------------------------------------------------------------------

    def assign(self, card: dict, *, op: str, parents: list[str], admit: bool = True, legacy: bool = False) -> Assignment:
        """Give a new card its lineage and idea key and decide whether the lineage admits it (fewer than
        budgets.lineage.ideas admitted ideas). Recorded as `lineage_assigned`."""
        if card["id"] in self.assigned:
            raise ValueError(f"{card['id']} already has a lineage")
        sig = card_signature(card)["signature"]
        if parents:
            lineage, joined, via = self.ensure(parents[0]).lineage_id, "parent", parents[0]
        elif sig in self.sig_lineage:
            lineage, joined, via = self.sig_lineage[sig], "signature", self.sig_lineage[sig]
        else:
            lineage, joined, via = card["id"], "new", None
        members = self.lineage_ideas.get(lineage, set())
        admitted = admit and (legacy or len(members) < self.lineage_caps["ideas"])
        assignment = Assignment(card["id"], lineage, sig, joined, via, admitted)
        event = self.ledger.append("lineage_assigned", ACTOR, {
            **asdict(assignment), "signature": sig, "op": op, "legacy": legacy,
            "lineage_ideas": len(members) + (1 if admitted else 0), "ideas_cap": self.lineage_caps["ideas"],
            "idea_calls": self.idea_calls[sig], "lineage_calls": self.lineage_calls[lineage]})
        self._apply(assignment, event["seq"] if not legacy else None)
        return assignment

    def ensure(self, card_id: str) -> Assignment:
        """The card's assignment; a card of a run started before Phase 8 gets one now from its file (its parents,
        else its signature), admitted whatever the lineage holds, since it is already in the pipeline."""
        if card_id not in self.assigned:
            card = self.store.get_card(card_id)
            created = [e["data"] for e in self.ledger.iter_events("card_created") if e["data"]["card_id"] == card_id]
            self.assign(card, op=created[0]["op"] if created else "unknown", parents=list(card["parent_ids"]),
                        legacy=True)
        return self.assigned[card_id]

    def assign_missing(self) -> list[str]:
        """Assign every card of the run that has no lineage yet, in creation order (resume of an older run)."""
        order = [e["data"]["card_id"] for e in self.ledger.iter_events("card_created")]
        order += [c for c in self.store.list_cards() if c not in order]
        missing = [card_id for card_id in order if card_id not in self.assigned]
        for card_id in missing:
            self.ensure(card_id)
        return missing

    # --- caps ------------------------------------------------------------------------------------

    def used(self, card_id: str, loop: str) -> int:
        return self.idea_used[(self.ensure(card_id).idea_key, loop)]

    def cap(self, loop: str) -> int:
        return self.loop_caps[loop]

    def charge(self, card_id: str, loop: str, stage: str) -> int:
        """One more round of a per-idea revision loop. Returns the idea's count after the charge."""
        if loop not in IDEA_LOOPS:
            raise ValueError(f"unknown budget loop {loop!r}; known loops {sorted(IDEA_LOOPS)}")
        card = self.ensure(card_id)
        self._count_round(card, loop)
        used = self.idea_used[(card.idea_key, loop)]
        extra = {}
        if loop in LINEAGE_SLOT_LOOPS:
            extra = {"lineage_ideas": len(self.lineage_ideas[card.lineage_id]),
                     "ideas_cap": self.lineage_caps["ideas"]}
        self.ledger.append("budget_charge", ACTOR, {"card_id": card_id, "stage": stage, "loop": loop,
                                                    "idea_key": card.idea_key, "lineage_id": card.lineage_id,
                                                    "used": used, "cap": self.loop_caps[loop], **extra})
        return used

    def after_round(self, *, card_id: str, stage: str, loop: str, round_id: str, failures: Mapping[str, Any],
                    key: str | None = None) -> LoopStop | None:
        """The one call a revision loop makes after a round that still fails, naming its loop (a key of IDEA_LOOPS).

        Records the round (`loop_round`) under `key` (default the card id) and `round_id` (the caller names the
        round so that a resumed stage repeating it is recognized), then decides: a stall, or the idea's count at the
        loop's cap, or for a loop in LINEAGE_SLOT_LOOPS a lineage with no idea slot left, ends the loop; the stop is
        recorded (`loop_stop`, not terminal: the stage's gate decides the card) and returned. Otherwise one more
        revision is granted and charged (`budget_charge`) before the caller makes it, and None is returned. The
        counts are per idea, so cards that share a signature share them."""
        if loop not in IDEA_LOOPS:
            raise ValueError(f"unknown budget loop {loop!r}; known loops {sorted(IDEA_LOOPS)}")
        if not failures:
            raise ValueError("a round with no failures ends its loop by passing; it does not consult the budget")
        stalled = observe_round(self.ledger, card_id=card_id, stage=stage, loop=loop, key=key or card_id,
                                round_id=round_id, failures=failures)
        stop = decide_round(loop, used=self.used(card_id, loop), cap=self.cap(loop), stalled=stalled,
                            failing=failures)
        if stop is None and loop in LINEAGE_SLOT_LOOPS:
            full = self.room(card_id)
            if full is not None:
                stop = LoopStop(loop, "budget", full.detail)
        if stop is not None:
            record_stop(self.ledger, card_id=card_id, stage=stage, stop=stop, terminal=False)
            return stop
        self.charge(card_id, loop, stage)
        return None

    def room(self, card_id: str) -> LoopStop | None:
        """None when the card's lineage can admit another idea, else the budget stop naming the full lineage."""
        card = self.ensure(card_id)
        held = len(self.lineage_ideas.get(card.lineage_id, ()))
        if held < self.lineage_caps["ideas"]:
            return None
        return LoopStop(LINEAGE_IDEAS, "budget",
                        f"lineage {card.lineage_id} holds {held} idea(s), cap {self.lineage_caps['ideas']}")

    def calls_stop(self, card_id: str) -> LoopStop | None:
        """The budget stop when the card's idea or lineage has used its model calls (no further call may be made for
        it), else None."""
        card = self.ensure(card_id)
        idea_cap, lineage_cap = self.idea_caps["model_calls"], self.lineage_caps["model_calls"]
        if self.idea_calls[card.idea_key] >= idea_cap:
            return LoopStop(MODEL_CALLS, "budget",
                            f"idea {card.idea_key} used {self.idea_calls[card.idea_key]} of {idea_cap} model calls")
        if self.lineage_calls[card.lineage_id] >= lineage_cap:
            return LoopStop(LINEAGE_CALLS, "budget", f"lineage {card.lineage_id} used "
                                                     f"{self.lineage_calls[card.lineage_id]} of {lineage_cap} model calls")
        return None

    def reserve_call(self, card_id: str) -> None:
        """Before a stage's model call: raise LoopBudgetExhausted when the card's idea or lineage has used its model
        calls, else count the call now, so concurrent workers on cards of one idea or lineage cannot pass the cap
        together. A call that then fails is released; one that returns is recorded with charge_call(reserved=True)."""
        stop = self.calls_stop(card_id)
        if stop is not None:
            raise LoopBudgetExhausted(card_id, stop)
        card = self.assigned[card_id]
        self.idea_calls[card.idea_key] += 1
        self.lineage_calls[card.lineage_id] += 1

    def release_call(self, card_id: str) -> None:
        card = self.assigned[card_id]
        self.idea_calls[card.idea_key] -= 1
        self.lineage_calls[card.lineage_id] -= 1

    def charge_call(self, card_id: str, call_id: str, stage: str, *, reserved: bool = False) -> None:
        """Record a model call made for the card (`budget_charge`). The explorer's call is charged after the fact,
        once its card exists; a stage's call was reserved before it was made."""
        card = self.ensure(card_id)
        if not reserved:
            self.idea_calls[card.idea_key] += 1
            self.lineage_calls[card.lineage_id] += 1
        self.ledger.append("budget_charge", ACTOR, {"card_id": card_id, "stage": stage, "loop": MODEL_CALLS,
                                                    "call_id": call_id, "idea_key": card.idea_key,
                                                    "lineage_id": card.lineage_id,
                                                    "used": self.idea_calls[card.idea_key],
                                                    "cap": self.idea_caps["model_calls"],
                                                    "lineage_used": self.lineage_calls[card.lineage_id],
                                                    "lineage_cap": self.lineage_caps["model_calls"]})

    def caps_record(self) -> dict:
        """The caps in force, for run_start and inspect."""
        return {"idea": dict(self.idea_caps), "lineage": dict(self.lineage_caps), "loops": dict(self.loop_caps),
                "prior": dict(self.prior)}


# ----- reading a run's ledger (inspect, outcome records) ------------------------------------------


def card_budget(events: list[dict], card_id: str) -> dict | None:
    """The budget block of one card's outcome record: its lineage and idea key, the revisions charged to it per
    loop, its model calls, and its loop stops. None for a card with no lineage_assigned event (a run before Phase 8,
    never resumed since)."""
    assigned = None
    loops: Counter = Counter()
    calls = 0
    stops = []
    for event in events:
        data = event["data"]
        if data.get("card_id") != card_id:
            continue
        if event["type"] == "lineage_assigned":
            assigned = data
        elif event["type"] == "budget_charge":
            if data["loop"] == MODEL_CALLS:
                calls += 1
            else:
                loops[data["loop"]] += 1
        elif event["type"] == "loop_stop":
            stops.append({key: data[key] for key in ("stage", "loop", "reason", "kill_code", "terminal")})
    if assigned is None:
        return None
    return {"lineage_id": assigned["lineage_id"], "idea_key": assigned["idea_key"], "joined": assigned["joined"],
            "admitted": assigned["admitted"], "loops": dict(sorted(loops.items())), "model_calls": calls,
            "stops": stops}
