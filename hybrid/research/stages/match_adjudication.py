"""How stage C settles a statement-match objection (FIX-7, results/phase4-20261013T073748Z/match-diag/REPORT.md).

In the 2026-10-14 calibration (runs/calibration-math-20261014T053341Z/k3-run1) the local matcher called card k3's
definition D1, which restates the card's sixteen states, its update and its readout, a substantive premise, and read
the P4 solvable-case ceiling ("L at most 4") as a requirement to succeed at every size up to 4. The host turned the
one definition verdict into sixteen mismatched statements and hid seven scored cells. Both findings were false, and
neither quoted anything that differed.

The rules here are pure functions over the matcher's answer and the texts it was shown.

Anchoring. A statement objection names its rubric item (or ``direction``) and quotes the claim text and the formal
statement or definition text that differ. The claim side is the item's ``instantiated_claim`` when it has one (P3: the
selected limit; P4: the selected solvable case), else the claim and its quote. The statement side is the frozen
statement and the text of the definitions it uses. A quote that is not exact text (up to whitespace, case and the
usual symbol spellings) of its side, or two quotes that are the same text, do not anchor anything. A rubric item
marked ``differs`` or a direction other than ``match`` with no anchored objection is recorded as rejected and blocks
nothing.

Definitions. A definition that faithfully restates the card's own state, update, readout or learning rule defines
the object the claims are about: it is part of the claim (verdict ``mechanism``). An objection to a definition quotes
the clause it objects to and says either which card mechanism text the clause contradicts (``contradicts_card``,
with that card text quoted) or what the clause asserts beyond the card (``beyond_card``): performance, a capacity, a
cost, a property of the data or of the world. A ``beyond_card`` objection whose clause only specifies the card's
state, update, readout or learning rule is rejected by that rule; an addition to those parts changes the card's
mechanism and is argued as a contradiction of the card text it changes.

Adjudication. Every anchored objection goes to a separately recorded cross-check (role ``statement_crosscheck``).
Both readings agree on a mismatch: ``confirmed``, the statement is mismatched and its claim blocked. They disagree:
``disputed``, and the statement is ``unresolved`` (mathrecords, Statement match at stage C). A host-computed
disagreement (the finite-domain predicate check) or a missing comparison is a mismatch without a cross-check.

Tiebreak (FIX-17, decision XC). A dispute (the matcher upheld an anchored objection, the cross-check rejected it) goes
to a third judge from a family other than both (role ``statement_tiebreak``, gateway ChatGPT). Upheld: confirmed, the
statement is mismatched. Rejected: the objection blocks nothing. A failed tiebreak call settles nothing: the statement
stays unresolved and its claim is not shown. Every dispute is recorded (``dispute_record``).

Unread replies (FIX-7 item 6). A cross-check reply that fails its schema after the client's retry reads nothing: each
objection it was to settle is ``unread`` and leaves its statement unresolved, as a disputed one does. Stage C treats a
matcher reply that fails its schema the same way, for every statement (c_challenge.matcher_reading).
"""
from __future__ import annotations

import json

from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.falsify import _normal

CONFIRMED, DISPUTED, REJECTED, UNREAD = "confirmed", "disputed", "rejected", "unread"
MATCHED, MISMATCHED, UNRESOLVED = "matched", "mismatched", "unresolved"


def _needle(quote: str | None) -> str:
    return _normal(quote or "").strip(" .;:,'\"`")


def _leaves(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [f"{k}: {x}" for k, v in value.items() for x in _leaves(v)] + [str(k) for k in value]
    if isinstance(value, list):
        return [x for v in value for x in _leaves(v)]
    return [json.dumps(value)]


def corpus(*values) -> list[str]:
    """Normalized texts a quote may come from: each value as JSON (as the roles were shown it) and its leaves."""
    texts = []
    for value in values:
        if isinstance(value, str):
            texts.append(_normal(value))
            continue
        texts.append(_normal(json.dumps(value, indent=1, ensure_ascii=False).replace("\\n", " ")))
        texts.append(_normal(" | ".join(_leaves(value))))
    return texts


def quoted(quote: str | None, texts: list[str]) -> bool:
    needle = _needle(quote)
    return bool(needle) and any(needle in text for text in texts)


def claim_texts(item: dict) -> list[str]:
    """The claim side of a statement objection: the instantiated claim when the item has one, else the claim."""
    if item.get("instantiated_claim"):
        return corpus(item["instantiated_claim"])
    return corpus(item["claim"], item["claim_quote"])


def statement_texts(spec, definition_texts: list[str]) -> list[str]:
    return corpus(spec, *definition_texts)


def card_texts(card: dict) -> list[str]:
    shown = {key: value for key, value in card.items() if key not in ("status", "created_by_call", "version")}
    return corpus(shown)


def statement_objections(sid: str, entry: dict, item: dict, spec, definition_texts: list[str]) -> list[dict]:
    """Every objection the matcher raised on one statement, each with ``host`` ``anchored`` or ``rejected`` and the
    reason, plus a rejected entry for each flagged rubric item or direction that no anchored objection covers."""
    claim_side = claim_texts(item)
    statement_side = statement_texts(spec, definition_texts)
    where = "its instantiated_claim" if item.get("instantiated_claim") else "the claim or its quote"
    out = []
    for index, objection in enumerate(entry.get("objections") or [], start=1):
        problems = []
        if not quoted(objection["claim_quote"], claim_side):
            problems.append(f"the claim quote is not text of {where}")
        if not quoted(objection["statement_quote"], statement_side):
            problems.append("the statement quote is not text of the formal statement or its definitions")
        if not problems and _needle(objection["claim_quote"]) == _needle(objection["statement_quote"]):
            problems.append("the two quotes are the same text")
        out.append(dict(objection, objection_id=f"{sid}#{index}", on=f"statement {sid}",
                        host="rejected" if problems else "anchored", host_reason="; ".join(problems)))
    covered = {o["item"] for o in out if o["host"] == "anchored"}
    flagged = [name for name in mc.RUBRIC if entry["rubric"][name]["verdict"] == "differs"]
    if entry["direction"] != "match":
        flagged.append("direction")
    for name in flagged:
        if name not in covered:
            note = (entry["rubric"][name]["note"] if name in mc.RUBRIC
                    else f"{entry['direction']}: {entry['why']}")
            out.append({"objection_id": None, "on": f"statement {sid}", "item": name, "claim_quote": None,
                        "statement_quote": None, "difference": note or "differs", "host": "rejected",
                        "host_reason": "marked as a difference with no anchored objection quoting the claim text and "
                                       "the statement text that differ"})
    return out


def definition_objection(finding: dict, definition_text: str, card_side: list[str]) -> dict:
    """The host's reading of one definition verdict: ``host`` ``none`` (convention or mechanism), ``anchored`` or
    ``rejected``, with the reason (module docstring, Definitions)."""
    did = finding["definition_id"]
    base = {"definition_id": did, "on": f"definition {did}", "verdict": finding["verdict"], "why": finding["why"],
            "objection": finding.get("objection"), "objection_id": None, "host": "none", "host_reason": ""}
    if finding["verdict"] != "substantive":
        return base
    objection = finding.get("objection")
    if objection is None:
        return dict(base, host="rejected", host_reason="substantive with no objection quoting the clause")
    problems = []
    if not quoted(objection["definition_quote"], corpus(definition_text)):
        problems.append("the definition quote is not text of the definition")
    if objection["kind"] == "contradicts_card":
        if not quoted(objection["card_quote"], card_side):
            problems.append("a contradiction of the card needs the card text it contradicts, quoted exactly")
    else:
        if objection["asserts"] in mc.CARD_DEFINING:
            problems.append(f"the clause specifies the card's own {objection['asserts'].replace('_', ' ')}: a "
                            "definition that restates the card's state, update, readout or learning rule is part of "
                            "the claim; an addition to them is argued as a contradiction of the card text it changes")
        if objection["card_quote"] and not quoted(objection["card_quote"], card_side):
            problems.append("the card quote is not text of the card")
    return dict(base, objection_id=f"{did}#1", host="rejected" if problems else "anchored",
                host_reason="; ".join(problems))


def pending(statement_items: dict[str, list[dict]], definition_items: dict[str, dict]) -> list[dict]:
    """The anchored objections the cross-check must read, in a stable order."""
    found = [o for sid in sorted(statement_items) for o in statement_items[sid] if o["host"] == "anchored"]
    found += [d for did, d in sorted(definition_items.items()) if d["host"] == "anchored"]
    return found


def adjudicate(objections: list[dict], verdicts: list[dict]) -> dict[str, dict]:
    """{objection id: {adjudication, crosscheck verdict, why}}: an objection the cross-check upheld is confirmed, one it
    rejected is disputed. Every anchored objection must have a verdict (the schema asks for one each)."""
    by_id = {}
    for verdict in verdicts:
        by_id.setdefault(verdict["objection_id"], verdict)
    missing = [o["objection_id"] for o in objections if o["objection_id"] not in by_id]
    if missing:
        raise ValueError(f"the cross-check gave no verdict for {', '.join(missing)}")
    return {o["objection_id"]: {"adjudication": CONFIRMED if by_id[o["objection_id"]]["verdict"] == "upheld"
                                else DISPUTED, "crosscheck": by_id[o["objection_id"]]["verdict"],
                                "crosscheck_why": by_id[o["objection_id"]]["why"]}
            for o in objections}


def tiebreak(disputes: list[dict], verdicts: list[dict]) -> dict[str, dict]:
    """FIX-17 (decision XC): {objection id: {adjudication, tiebreak verdict, why}} for the objections the matcher and
    the cross-check disagree on. A third judge from a family other than both (role ``statement_tiebreak``) settles
    each: upheld is ``confirmed`` (the statement is mismatched), rejected is ``rejected`` (the objection blocks
    nothing). Every dispute must have a verdict (the schema asks for one each)."""
    by_id = {}
    for verdict in verdicts:
        by_id.setdefault(verdict["objection_id"], verdict)
    missing = [o["objection_id"] for o in disputes if o["objection_id"] not in by_id]
    if missing:
        raise ValueError(f"the tiebreak gave no verdict for {', '.join(missing)}")
    return {o["objection_id"]: {"adjudication": CONFIRMED if by_id[o["objection_id"]]["verdict"] == "upheld"
                                else REJECTED, "tiebreak": by_id[o["objection_id"]]["verdict"],
                                "tiebreak_why": by_id[o["objection_id"]]["why"]}
            for o in disputes}


def tiebreak_failed(disputes: list[dict], why: str) -> dict[str, dict]:
    """The tiebreak call failed (transport, schema or budget of the call): every dispute stays ``disputed``, its
    statement unresolved and its claim not shown. A failed tiebreak never settles a dispute either way."""
    return {o["objection_id"]: {"adjudication": DISPUTED, "tiebreak": None, "tiebreak_why": why} for o in disputes}


def dispute_record(objection: dict) -> dict:
    """What is kept of every matcher/cross-check dispute: both readings, the tiebreak's and the outcome."""
    keys = ("objection_id", "on", "item", "claim_quote", "statement_quote", "difference", "definition_id")
    entry = {k: objection[k] for k in keys if k in objection}
    entry.update(matcher="upheld", matcher_why=objection.get("why"), crosscheck=objection.get("crosscheck"),
                 crosscheck_why=objection.get("crosscheck_why"), tiebreak=objection.get("tiebreak"),
                 tiebreak_why=objection.get("tiebreak_why"),
                 outcome={CONFIRMED: "mismatch", REJECTED: "matched"}.get(objection.get("adjudication"),
                                                                           "unresolved"))
    return entry


def unread(objections: list[dict], why: str) -> dict[str, dict]:
    """{objection id: adjudication ``unread``}: the cross-check's reply failed its schema after the retry (FIX-7 item
    6), so no objection is corroborated or rejected; each leaves its statement unresolved."""
    return {o["objection_id"]: {"adjudication": UNREAD, "crosscheck": None, "crosscheck_why": why}
            for o in objections}


def settle(own: list[dict], definitions: list[dict], hard: list[str]) -> tuple[str, list[str], list[str]]:
    """A statement's match after adjudication: (matched, mismatched or unresolved; the reasons that block it; the
    disputed objections). ``own`` are its objections and ``definitions`` the readings of the definitions it uses, each
    with ``adjudication`` set when the cross-check read it; ``hard`` are host-computed mismatch reasons."""
    reasons, disputed = list(hard), []
    for objection in own:
        text = f"{objection['item']}: {objection['difference']} (claim: {objection['claim_quote']!r}; statement: " \
               f"{objection['statement_quote']!r})"
        if objection.get("adjudication") == CONFIRMED:
            reasons.append(text)
        elif objection.get("adjudication") in (DISPUTED, UNREAD):
            disputed.append(f"{text}; cross-check: {objection['crosscheck_why']}" + _tiebreak_note(objection))
    for item in definitions:
        did, objection = item["definition_id"], item.get("objection") or {}
        text = (f"definition {did} is a substantive premise, not a convention ({objection.get('kind')}, "
                f"{objection.get('asserts')}): {objection.get('difference')} (clause: "
                f"{objection.get('definition_quote')!r})")
        if item.get("adjudication") == CONFIRMED:
            reasons.append(text)
        elif item.get("adjudication") in (DISPUTED, UNREAD):
            disputed.append(f"{text}; cross-check: {item['crosscheck_why']}" + _tiebreak_note(item))
    if reasons:
        return MISMATCHED, reasons, disputed
    if disputed:
        return UNRESOLVED, [], disputed
    return MATCHED, [], []


def _tiebreak_note(objection: dict) -> str:
    if "tiebreak_why" not in objection:
        return ""
    return f"; tiebreak not settled: {objection['tiebreak_why']}"
