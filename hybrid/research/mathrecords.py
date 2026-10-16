"""Tier records, definitions, the premise ledger and the gate rules of stages M and C (DISCOVERY-PIPELINE-PLAN.md
sections 4.1 to 4.3 and 5). Pure functions over plain dicts; the stage modules do the calls, the checks and the I/O.

Definitions and premises. The statement fixer separates two things the first live run (card-46dbb349, 2026-10-12)
had mixed into one "assumption" list:
- a **definition** fixes a convention or protocol choice (how inputs are sampled, how evaluation reads the model,
  the accounting units, how an underspecified baseline is built). Definitions are part of the frozen formal
  statements (hashed with them, shown to the prover and to the statement-match checker) and carry no status;
- a **premise** is a substantive claim about the mechanism, the data or the world that a proof relies on but does
  not prove. It says why it is needed and how it can be settled: ``testable_by`` ``code`` (with a measurement plan
  that test design must schedule) or ``citation`` (a source in the arXiv index or OpenAlex).

Statement result, from its check record (the check decides, never a model):
- ``holds``: Lean pass (tier lean), SymPy discharged (sympy_exact), Z3 discharged (smt_bounded, with its bound),
  an executed check discharged (executed, observed on the listed sizes);
- ``refuted``: a decided counterexample from Z3 or SymPy;
- ``failed``: an executed check decided violated (the construction or simulation did not meet its threshold);
- ``not_shown``: nothing decided (a rejected Lean proof, an unknown result, no attempt left). Tier ``argued``.

Obligation result: ``refuted`` if any statement is refuted, else ``holds`` at the best tier of its holding
statements, else ``failed`` if any failed, else ``not_shown``.

Premise status, from evidence only: ``proven`` when an obligation that discharges it holds at lean or sympy_exact;
``measured`` when it holds at smt_bounded or executed (the evidence names the executed check artifact);
``cited`` when the challenger verified its source against the literature record (stage C); ``circular`` when the
statement-match role found that it restates or implies a claim it supports (stage C); ``open`` otherwise. The
proposer and the prover never set a status.

Settled premises (FIX-9). A premise that a discharging obligation settles at stage M (``proven`` or ``measured``: its
check held) is evidence, not an assumption, and its ledger entry carries ``settled`` ("settled at M by <statement
id>"). The statement-match finding that it restates or implies a claim it supports does not make it ``circular``
(circularity is for premises assumed without support; the finding is kept in its basis), and stage C does not require
magic-X code for it (``settled_at_m``). A discharging check that failed or was not shown settles nothing: the premise
stays open or circular and every check applies. In the 2026-10-14 calibration (runs/calibration-math-20261014T121552Z)
the fixer's conformance premise A1 was measured by a held conformance statement in every finished run, and stage C
still raised ``magic_x_missing:A1`` in all of them and, in k1-run2, ``circular:A1``, which scored P1 not shown.

Claim record (one per card claim). Status words, in order of precedence: ``blocked`` (statement mismatch, stage
C), ``refuted``, ``failed``, ``circular`` (it rests on a circular premise), ``unproven`` (an obligation was not
shown), ``conditional`` (every obligation holds at some tier, but open premises remain; they are listed),
``proven`` (lean, sympy_exact), ``proven_up_to_bound`` (smt_bounded), ``observed`` (executed); and
``not_formalized``. Executed results are never called proven: the phrase comes from
``hybrid.math.tiers.verdict_phrase``, and a conditional phrase only prefixes it.

Statement match at stage C (FIX-7). Every stage C claim record also says how its statements matched it (``match``):
``matched``; ``mismatched`` (a mismatch the host anchored and the cross-check upheld, or a finite-domain predicate
disagreement; the claim is ``blocked`` as above); or ``unresolved`` (the matcher raised an anchored objection and the
separately recorded cross-check rejected it or its reply failed its schema; or the matcher's own reply failed its
schema, FIX-7 item 6). An unresolved claim keeps the status its checks give, failed or
passed, with ``match_detail`` naming the disputed objections: the dispute is recorded, not turned into a block or a
pass. An unresolved statement cannot discharge a premise. The stage C gate lists ``match_unresolved:<claim>``; a card
with unresolved claims and no failure is ``undetermined``, never ``pass``.

Advancing. A conditional claim advances only when each of its open premises can still be settled: testable by code
with a measurement plan (test design must schedule it), or, at stage M only, testable by citation with a source
offered (stage C verifies it; a source that does not verify leaves the premise open, and it then blocks).

Where a premise is settled (FIX-6). The fixer says what a premise is about (``subject``): ``frozen_definitions``
(the card's own code, its equations or its frozen definitions: that the cost reference executes the card's equations
with the definitions' spellings and buffers, that a witness keeps the buffers a definition names), ``mechanism`` (a
capacity or component the mechanism relies on, the magic-X subject) or ``world`` (external data, hardware, a
published result). The host settles the subject: a premise whose text or measurement plan names one of the record's
frozen definitions is about them whatever label it carries. Such a premise is settled at stage M only
(``settle_at`` ``M``): the record must hold an obligation with claim_id null that discharges it, frozen with the
statements and checked in M (for the cost reference's conformance, a ``conformance`` statement). Without one it is
refused at freeze (``refused`` says why) and, like one whose discharging check did not hold, it never advances and is
never deferred to test design: calibration runs only M and C, so a deferred plan for the card's own definitions would
never run. Mechanism and world premises keep the routes above (``settle_at`` ``deferrable``). In the 2026-10-14
calibration (card k1, results/phase4-20261013T073748Z/k1-calib-diag/REPORT.md, section 3) the fixer's A1 ("the
reference executes the card's equations with D1's product spellings and D2's retained buffers") had no discharging
obligation and its plan was deferred to test design, which left the three complexity claims and P3 conditional.

Premises stage C adds (FIX-4). Only a dependency the host has substantiated becomes a premise: a component named only
in prose or supplied from outside the mechanism, a component whose cost P1 does not cover (an accounting premise on
the cost obligation), a missing target-rule class in P5, or a challenger finding whose verification code, derivation
or citation confirmed it. The stage gives such a premise an ``open_basis`` saying what settled it and what is still
missing; the advancing rule is unchanged, so a premise without a measurement plan still blocks. An allegation the
challenger could not substantiate is recorded in the challenge artifact and is not a premise.
"""
from __future__ import annotations

from hybrid.math import tiers

TIER_RANK = {tiers.LEAN: 4, tiers.SYMPY_EXACT: 3, tiers.SMT_BOUNDED: 2, tiers.EXECUTED: 1, tiers.ARGUED: 0}
TIER_LABEL = {tiers.LEAN: "lean-proven", tiers.SYMPY_EXACT: "sympy-exact", tiers.SMT_BOUNDED: "z3-checked (bounded)",
              tiers.EXECUTED: "executed-on-examples", tiers.ARGUED: "argued (prose only)"}
FORM_TIER = {"lean": tiers.LEAN, "sympy": tiers.SYMPY_EXACT, "z3": tiers.SMT_BOUNDED, "simulation": tiers.EXECUTED,
             "construction": tiers.EXECUTED, "cost": tiers.EXECUTED,
             # Phase 4, profile obligations P5 and P4 (hybrid/math/profile.py)
             "conservation": tiers.EXECUTED, "learnability": tiers.EXECUTED,
             # FIX-6: a cost reference run against the card's frozen equations (spec_checks.reference_conformance)
             "conformance": tiers.EXECUTED}
PROVER_FORMS = ("lean", "simulation", "construction", "cost", "learnability")
AUTO_FORMS = ("z3", "sympy", "conservation")
FORMS = tuple(FORM_TIER)
# The forms a statement fixer may use when the card is not held to the profile (the Phase 3 forms).
BASE_FORMS = ("lean", "sympy", "z3", "simulation", "construction", "cost", "conformance")
# FIX-7: ``mechanism`` labels a definition that restates the card's own state, update, readout or learning rule. It
# defines the object the claims are about and is part of the claim, not a hidden premise.
DEFINITION_KINDS = ("sampling", "evaluation", "accounting", "baseline", "notation", "mechanism", "other")
PREMISE_KINDS = ("oracle", "store", "prior", "other")
PREMISE_STATUSES = ("proven", "cited", "measured", "open", "circular")
TESTABLE_BY = ("code", "citation")
# FIX-6: what a premise is about, and where it is settled (module docstring, Where a premise is settled).
SUBJECTS = ("frozen_definitions", "mechanism", "world")
SETTLE_AT = ("M", "deferrable")
CITATION_SOURCES = ("arxiv", "openalex")
CLAIM_HOLDS = {tiers.LEAN: "proven", tiers.SYMPY_EXACT: "proven", tiers.SMT_BOUNDED: "proven_up_to_bound",
               tiers.EXECUTED: "observed"}
# Claims the statement fixer formalizes: everything falsify.claim_items lists except the mechanism description
# (not a claim) and the falsifiers (conditions under which the card fails, decided at S5 and S6).
SKIPPED_CLAIM_PREFIXES = ("falsifier.", "sample_efficiency.mechanism")


def formal_claims(items: list[dict]) -> list[dict]:
    return [item for item in items if not item["id"].startswith(SKIPPED_CLAIM_PREFIXES)]


# ----- statements --------------------------------------------------------------------------------------------------

def statement_sizes(statement: dict) -> list[str]:
    spec = statement["spec"]
    if statement["form"] == "simulation":
        return [f"size={s}" for s in spec["sizes"]]
    if statement["form"] == "construction":
        return [f"{spec['task']} size={s}" for s in spec["sizes"]]
    if statement["form"] == "cost":
        return [f"n={n}" for n in spec.get("ns") or (64, 128, 256, 512, 1024, 2048, 4096)]
    if statement["form"] == "learnability":
        return [f"{spec['task']} size={spec['size']} m={spec['examples']} seed={seed}" for seed in spec["seeds"]]
    if statement["form"] == "conformance":
        return [f"n={point['n']} {stream}" for point in spec["points"] for stream in spec["streams"]]
    if statement["form"] == "conservation":
        sizes = {"copy": "L", "recall": "K", "dyck": "depth", "index": "n", "reads": "n", "storage": "facts",
                 "sample": "examples"}
        return [f"{section} {sizes[section]}={entry[sizes[section]]}" for section in sizes
                for entry in spec.get(section) or []]
    return []


def statement_result(statement: dict, check: dict | None) -> dict:
    """The result of one frozen statement from its deciding check record (None when it was never checked)."""
    tier = FORM_TIER[statement["form"]]
    base = {"statement_id": statement["id"], "form": statement["form"], "tier": tiers.ARGUED,
            "label": TIER_LABEL[tiers.ARGUED], "evidence": "", "check": None, "bound": statement.get("bound")}
    if check is None:
        return dict(base, result="not_shown", phrase=tiers.verdict_phrase(tiers.ARGUED), evidence="never checked")
    base.update(evidence=check.get("evidence") or "; ".join(check.get("reasons") or []), check=check.get("artifact"))
    status, decided = check["status"], check["decided"]
    if decided and status in ("pass", "discharged"):
        phrase = tiers.verdict_phrase(tier, bound=statement.get("bound"), sizes=statement_sizes(statement) or None)
        return dict(base, result="holds", tier=tier, label=TIER_LABEL[tier], phrase=phrase)
    if decided and status == "violated":
        result = "failed" if tier == tiers.EXECUTED else "refuted"
        return dict(base, result=result, counterexample=check.get("counterexample"),
                    phrase=f"{result}: {str(check.get('evidence'))[:300]}")
    return dict(base, result="not_shown", phrase=tiers.verdict_phrase(tiers.ARGUED))


def obligation_result(obligation: dict, results: list[dict], *, require_all: bool = False) -> dict:
    """An obligation holds when any statement holds, at the best tier. With ``require_all`` (profile slots, whose
    statements are separate parts of one requirement, such as the two variants of P2.composition) a failed statement
    fails it, it holds only when every statement holds, and its tier is the weakest of them."""
    holding = sorted((r for r in results if r["result"] == "holds"), key=lambda r: -TIER_RANK[r["tier"]])
    if require_all:
        holding = holding[::-1] if len(holding) == len(results) else []
    if any(r["result"] == "refuted" for r in results):
        outcome = "refuted"
    elif require_all and any(r["result"] == "failed" for r in results):
        outcome = "failed"
    elif holding:
        outcome = "holds"
    elif any(r["result"] == "failed" for r in results):
        outcome = "failed"
    else:
        outcome = "not_shown"
    best = holding[0] if holding else None
    return {"obligation_id": obligation["id"], "claim_id": obligation.get("claim_id"), "result": outcome,
            "tier": best["tier"] if best else tiers.ARGUED, "label": TIER_LABEL[best["tier"] if best else tiers.ARGUED],
            "phrase": best["phrase"] if best else tiers.verdict_phrase(tiers.ARGUED),
            "by_statement": best["statement_id"] if best else None, "check": best["check"] if best else None,
            "premises": list(obligation["premises"]), "definitions": list(obligation["definitions"]),
            "statements": results}


# ----- the premise ledger -------------------------------------------------------------------------------------------

def premise_ledger(premises: list[dict], obligations: list[dict], outcomes: dict[str, dict], *,
                   extra_edges: list[dict] = (), cited: dict[str, str] | None = None,
                   citation_problems: dict[str, str] | None = None,
                   circular: dict[str, str] | None = None, citations_checked: bool = False) -> list[dict]:
    """One entry per premise, status from evidence.

    ``outcomes`` maps obligation id to obligation_result. ``extra_edges`` are {premise_id, obligation_id} edges the
    challenger added (edges are only ever added). ``cited`` maps a premise id to the evidence of a source the
    challenger verified against the literature record; ``citation_problems`` maps a premise id to why its source
    did not verify. ``circular`` maps a premise id to the statement-match finding that it restates or implies a
    claim it supports. ``citations_checked`` is False at stage M (sources are verified at stage C)."""
    cited, problems, circular = cited or {}, citation_problems or {}, circular or {}
    used_by: dict[str, list[str]] = {p["id"]: [] for p in premises}
    for obligation in obligations:
        for pid in obligation["premises"]:
            used_by[pid].append(obligation["id"])
    for edge in extra_edges:
        if edge["obligation_id"] not in used_by[edge["premise_id"]]:
            used_by[edge["premise_id"]].append(edge["obligation_id"])
    ledger = []
    for premise in premises:
        pid = premise["id"]
        dischargers = [o for o in obligations if pid in o["discharges"]]
        held = sorted((outcomes[o["id"]] for o in dischargers if outcomes[o["id"]]["result"] == "holds"),
                      key=lambda r: -TIER_RANK[r["tier"]])
        # FIX-9: a discharging check that held settles the premise, so a circularity finding does not apply to it.
        if pid in circular and not held:
            status, evidence = "circular", circular[pid]
            basis = "restates or implies a claim it supports (statement-match role); the claim fails"
        elif held and held[0]["tier"] in tiers.PROVEN_TIERS:
            status, evidence = "proven", f"{held[0]['by_statement']} ({held[0]['label']})"
            basis = f"obligation {held[0]['obligation_id']} holds at {held[0]['label']}"
        elif held:
            if not held[0]["check"]:
                raise ValueError(f"premise {pid}: obligation {held[0]['obligation_id']} holds without a check "
                                 "artifact")
            status = "measured"
            evidence = f"{held[0]['by_statement']} ({held[0]['label']}), math_check artifact {held[0]['check']}"
            basis = f"obligation {held[0]['obligation_id']} holds at {held[0]['label']}: {held[0]['phrase']}"
        elif pid in cited:
            status, evidence = "cited", cited[pid]
            basis = "source resolved in the literature record and verified by the challenger"
        else:
            status, evidence = "open", None
            basis = premise.get("open_basis") or _open_basis(premise, problems.get(pid), citations_checked)
        entry = {"id": pid, "text": premise["text"], "kind": premise["kind"], "does_work": premise["does_work"],
                 "why_needed": premise["why_needed"], "testable_by": premise["testable_by"],
                 "measurement_plan": premise["measurement_plan"], "citation": premise["citation"],
                 "added_by": premise["added_by"], "status": status, "evidence": evidence,
                 "used_by": sorted(set(used_by[pid])), "basis": basis}
        # FIX-6: recorded only for premises frozen with a settlement decision (older records have none).
        entry.update({key: premise[key] for key in ("subject", "settle_at", "refused") if key in premise})
        if status == "open" and premise.get("settle_at") == "M" and not premise.get("open_basis"):
            entry["basis"] = _m_basis(premise, dischargers)
        if held:
            entry["settled"] = f"settled at M by {held[0]['by_statement']}"
            if pid in circular:
                entry["basis"] += (f"; {entry['settled']}, so the statement-match finding that it "
                                   f"{circular[pid]} is not a circularity")[:1000]
        ledger.append(entry)
    return ledger


def settled_at_m(ledger: list[dict]) -> dict[str, str]:
    """The premises a discharging check settled (status proven or measured), each with "settled at M by <statement
    id>" (FIX-9). Such a premise is evidence, not an assumption: stage C exempts it from the magic-X code requirement
    and from the circularity verdict. A premise whose discharging check did not hold is not here."""
    return {entry["id"]: entry["settled"] for entry in ledger if entry.get("settled")}


def _m_basis(premise: dict, dischargers: list[dict]) -> str:
    """Why an open premise about the card's own frozen definitions stays open: refused at freeze, or its discharging
    check at stage M did not hold. It is never deferred to test design."""
    if premise.get("refused"):
        return f"refused at freeze: {premise['refused']}"
    return ("about the card's own frozen definitions, settled only at stage M; its discharging obligation "
            f"{', '.join(o['id'] for o in dischargers)} did not hold, so it stays open and is not deferred to test "
            "design")


def _open_basis(premise: dict, citation_problem: str | None, citations_checked: bool) -> str:
    if premise["testable_by"] == "code":
        if premise["measurement_plan"] is None:
            return "testable by code but no measurement plan was given; it cannot be scheduled"
        return "testable by code; its measurement plan is for test design to schedule (not yet measured)"
    if premise["citation"] is None:
        return "testable by citation but no source was given"
    if not citations_checked:
        return "testable by citation; the source is verified at stage C"
    return f"source not verified: {citation_problem or 'no verification was recorded'}"


def schedulable(entry: dict) -> bool:
    """An open premise test design can settle: testable by code, with a measurement plan, and not about the card's own
    frozen definitions (those are settled at stage M only, FIX-6)."""
    return (entry["status"] == "open" and entry["testable_by"] == "code" and entry["measurement_plan"] is not None
            and entry.get("settle_at") != "M")


def can_advance(entry: dict, *, stage: str) -> bool:
    """Whether a premise lets a conditional claim advance from ``stage`` (section docstring, Advancing)."""
    if entry["status"] != "open":
        return entry["status"] != "circular"
    if entry.get("settle_at") == "M":
        return False
    if schedulable(entry):
        return True
    return stage == "M" and entry["testable_by"] == "citation" and entry["citation"] is not None


def premises_of(obligation_ids: list[str], ledger: list[dict], status: str) -> list[str]:
    return sorted({entry["id"] for entry in ledger if entry["status"] == status
                   and set(entry["used_by"]) & set(obligation_ids)})


def scheduled_measurements(records: list[dict], ledger: list[dict]) -> list[dict]:
    """The measurement plans of the open, code-testable premises of conditional claims: test design (a later phase)
    must schedule each before the claims count as more than conditional."""
    by_id = {entry["id"]: entry for entry in ledger}
    needed: dict[str, list[str]] = {}
    for record in records:
        if record["status"] == "conditional":
            for pid in record["open_premises"]:
                if schedulable(by_id[pid]):
                    needed.setdefault(pid, []).append(record["claim_id"])
    return [{"premise_id": pid, "text": by_id[pid]["text"], "measurement_plan": by_id[pid]["measurement_plan"],
             "claims": claims, "status": "deferred_to_test_design"} for pid, claims in sorted(needed.items())]


# ----- claim records ----------------------------------------------------------------------------------------------

def claim_records(claims: list[dict], obligations: list[dict], outcomes: dict[str, dict], ledger: list[dict], *,
                  not_formalized: list[dict] = (), blocked: dict[str, str] | None = None,
                  failed: dict[str, str] | None = None, unresolved: dict[str, str] | None = None) -> list[dict]:
    """One tier record per claim. ``blocked`` maps claim id to the statement-match finding that blocks it; ``failed``
    maps claim id to a host finding that fails it although its checks held (a profile consistency failure, Phase 4).
    ``unresolved`` (stage C, FIX-7) maps claim id to the disputed statement-match objections; when it is given, every
    record carries ``match`` (module docstring, Statement match at stage C), and an unresolved claim keeps its status."""
    match_known = unresolved is not None
    blocked, failed, unresolved = blocked or {}, failed or {}, unresolved or {}
    why_not = {item["claim_id"]: item["why"] for item in not_formalized}
    by_id = {entry["id"]: entry for entry in ledger}
    records = []
    for claim in claims:
        cid = claim["id"]
        mine = [o for o in obligations if o.get("claim_id") == cid]
        results = [outcomes[o["id"]] for o in mine]
        base = {"claim_id": cid, "text": claim["text"], "obligations": [o["id"] for o in mine],
                "tier": tiers.ARGUED, "label": TIER_LABEL[tiers.ARGUED], "open_premises": [],
                "circular_premises": [], "detail": ""}
        if not mine:
            records.append(dict(base, status="not_formalized", phrase=tiers.verdict_phrase(tiers.ARGUED),
                                detail=why_not.get(cid, "no obligation was written for this claim")))
            continue
        weakest = min((r["tier"] for r in results), key=lambda t: TIER_RANK[t])
        obligation_ids = [o["id"] for o in mine]
        open_deps = premises_of(obligation_ids, ledger, "open")
        circular_deps = premises_of(obligation_ids, ledger, "circular")
        all_hold = all(r["result"] == "holds" for r in results)
        if cid in blocked:
            status, detail = "blocked", blocked[cid]
        elif any(r["result"] == "refuted" for r in results):
            status = "refuted"
            detail = "; ".join(s["phrase"] for r in results for s in r["statements"] if s["result"] == "refuted")
        elif any(r["result"] == "failed" for r in results):
            status = "failed"
            detail = "; ".join(s["phrase"] for r in results for s in r["statements"] if s["result"] == "failed")
        elif cid in failed:
            status, detail = "failed", failed[cid]
        elif circular_deps:
            status = "circular"
            detail = "rests on circular premises: " + "; ".join(f"{pid}: {by_id[pid]['evidence']}"
                                                                for pid in circular_deps)
        elif not all_hold:
            status, detail = "unproven", "obligations not shown: " + ", ".join(
                r["obligation_id"] for r in results if r["result"] != "holds")
        elif open_deps:
            status = "conditional"
            detail = "conditional on open premises " + "; ".join(
                f"{pid} ({by_id[pid]['basis']})" for pid in open_deps)
        else:
            status, detail = CLAIM_HOLDS[weakest], ""
        shown = all_hold and status in (*CLAIM_HOLDS.values(), "conditional", "circular")
        weakest_phrase = min(results, key=lambda r: TIER_RANK[r["tier"]])["phrase"]
        if status in CLAIM_HOLDS.values():
            phrase = weakest_phrase
        elif status == "conditional":
            phrase = f"conditional on {', '.join(open_deps)}: {weakest_phrase}"
        else:
            phrase = f"{status}: {detail}"[:400]
        tier = weakest if shown or status == "unproven" else tiers.ARGUED
        entry = dict(base, status=status, tier=tier, label=TIER_LABEL[tier], open_premises=open_deps,
                     circular_premises=circular_deps, phrase=phrase, detail=detail[:1000])
        if match_known:
            entry["match"] = "mismatched" if cid in blocked else "unresolved" if cid in unresolved else "matched"
            if entry["match"] == "unresolved":
                entry["match_detail"] = unresolved[cid][:1000]
        records.append(entry)
    for record in records:
        if match_known and "match" not in record:
            record["match"] = "matched"
    return records


def claim_line(record: dict) -> str:
    """The verdict reason for one claim. The tier label and phrase never call an executed result proven."""
    text = f"claim {record['claim_id']}: {record['status']} [{record['label']}]"
    if record["status"] in CLAIM_HOLDS.values():
        return f"{text}: {record['phrase']}"[:600]
    return f"{text}: {record['detail']}"[:600]


# ----- gates --------------------------------------------------------------------------------------------------------

def math_gate(records: list[dict], ledger: list[dict]) -> tuple[str, list[str]]:
    """Stage M verdict and kill codes (section 9 vocabulary). Precedence: a refutation, then a failed executed
    check, then anything unproven or conditional on a premise that cannot advance, then pass (conditional claims
    whose premises can still be settled go to stage C)."""
    refuted = [r["claim_id"] for r in records if r["status"] == "refuted"]
    failed = [r["claim_id"] for r in records if r["status"] == "failed"]
    if refuted:
        return "infeasible_proved", [f"obligation_failed:{cid}" for cid in refuted]
    if failed:
        return "fail", [f"obligation_failed:{cid}" for cid in failed]
    return _unproven_or_pass(records, ledger, stage="M")


def _unproven_or_pass(records: list[dict], ledger: list[dict], *, stage: str) -> tuple[str, list[str]]:
    by_id = {entry["id"]: entry for entry in ledger}
    codes = []
    for r in records:
        if r["status"] in ("unproven", "not_formalized"):
            codes.append(f"unproven:{r['claim_id']}")
        elif r["status"] == "conditional":
            codes += [f"unproven:{pid}" for pid in r["open_premises"] if not can_advance(by_id[pid], stage=stage)]
    if codes:
        return "undetermined", sorted(set(codes))
    return "pass", []


def challenge_gate(records: list[dict], ledger: list[dict], *, conservation: list[str], magic_x: list[str],
                   magic_x_missing: list[str], counterexamples: list[str],
                   mismatched: list[str] = ()) -> tuple[str, list[str]]:
    """Stage C verdict and kill codes. ``conservation`` lists violated laws, ``magic_x`` the premise ids whose
    baseline + X reached the claim, ``magic_x_missing`` the doing-work, code-testable premises the challenger gave
    no runnable magic-X for, ``counterexamples`` the claim ids a widened Z3 search refuted, ``mismatched`` the
    statement ids the statement-match role found different from their claim (section 5.4: any mismatch fails the
    card, including a statement that discharges a premise). Precedence: conservation, counterexample, then every
    failure together (magic-X, missing magic-X, circular premises, statement mismatch, failed executed checks),
    then the unproven rules with stage C's advancing rule. A claim whose record says ``match`` ``unresolved``
    (FIX-7) adds ``match_unresolved:<claim>`` to a failing verdict, and turns a pass into ``undetermined``."""
    unresolved = sorted({f"match_unresolved:{r['claim_id']}" for r in records if r.get("match") == "unresolved"})
    verdict, codes = _challenge_gate(records, ledger, conservation=conservation, magic_x=magic_x,
                                     magic_x_missing=magic_x_missing, counterexamples=counterexamples,
                                     mismatched=mismatched)
    if not unresolved:
        return verdict, codes
    return ("undetermined" if verdict == "pass" else verdict), sorted(set(codes) | set(unresolved))


def _challenge_gate(records: list[dict], ledger: list[dict], *, conservation: list[str], magic_x: list[str],
                    magic_x_missing: list[str], counterexamples: list[str],
                    mismatched: list[str] = ()) -> tuple[str, list[str]]:
    if conservation:
        return "infeasible_proved", sorted({f"conservation:{law}" for law in conservation})
    refuted = sorted(set(counterexamples) | {r["claim_id"] for r in records if r["status"] == "refuted"})
    if refuted:
        return "infeasible_proved", [f"obligation_failed:{cid}" for cid in refuted]
    codes = {f"magic_x:{pid}" for pid in magic_x} | {f"magic_x_missing:{pid}" for pid in magic_x_missing}
    codes |= {f"circular:{pid}" for r in records if r["status"] == "circular" for pid in r["circular_premises"]}
    if mismatched or any(r["status"] == "blocked" for r in records):
        codes.add("statement_mismatch")
    codes |= {f"obligation_failed:{r['claim_id']}" for r in records if r["status"] == "failed"}
    if codes:
        return "fail", sorted(codes)
    return _unproven_or_pass(records, ledger, stage="C")
