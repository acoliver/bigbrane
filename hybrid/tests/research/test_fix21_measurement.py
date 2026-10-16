"""FIX-21 regressions for stage C (decision MO, results/phase4-20261013T073748Z/FIX-21.md): measurement objects are a
component role of their own.

Cause, from the recorded calibration line runs/calibration-math-20261016T051933Z/k1-run3 (sanitized extract:
fixtures/calibration_math_20261016T051933Z/k1_run3_measurement.json, written by tmp/fix21/scripts/
make_fix21_fixture.py). The challenger listed the P5 restricted hypothesis classes as a component with role
"external", kind "prior", no statement and no function, used by O13 (P5), and said itself that they were declared
classes for the information experiment, frozen in O13.a and not supplied to the trained model. The host made the
hidden-component premise C-6f545ab1 from it ("it exists as costed mechanism code, or the claims that use it hold
without it"), the statement-match role read that premise as implying P5, and P5 became circular (not shown).

A measurement object (a P5 hypothesis class, a P2 task generator, an evaluation set) belongs to the test or bound. It
never becomes a premise; it is settled by the frozen statement that defines it, only when that statement held. A part
the model is given at inference or training is still a mechanism or external part.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from hybrid.research import mathrecords
from hybrid.research.calibration import load_cards
from hybrid.research.schemas import validate
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import load_section
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_profile_stage import Harness

FIX = json.loads((Path(__file__).parent / "fixtures" / "calibration_math_20261016T051933Z" /
                  "k1_run3_measurement.json").read_text(encoding="utf-8"))
CARD = FIX["card"]
RECORD = FIX["record"]
RECORDED = FIX["recorded_c"]
CLASSES = next(c for c in FIX["challenger_components"] if c["name"] == "Restricted target-rule classes")
PREMISE_ID = "C-6f545ab1"
CARD_QUOTE = ("This is a restricted-class information bound, not a claim that learning the parsing or lookup "
              "operation is free.")


def recorded_checks() -> dict[str, dict]:
    return copy.deepcopy(FIX["m_checks"])


def disable_role_check(monkeypatch):
    """The component rule as it stood before FIX-21: no measurement role, no correction."""
    monkeypatch.setattr(c_challenge, "measurement_role", lambda component, record, runnable: (None, None))


def oid_of(record: dict, claim_id: str) -> str:
    return next(o["id"] for o in record["obligations"] if o["claim_id"] == claim_id)


def stage_c(components: list[dict] | None = None, *, record: dict | None = None, checks: dict | None = None,
            readings: dict[str, list[str]] | None = None, mismatched: dict | None = None) -> dict:
    """Stage C's component rule, findings, ledger, records and gate on the recorded k1-run3 stage M, with the
    recorded statement-match reading of each premise the stage shows (C-6f545ab1 implies P5, A1 implies P1) plus
    ``readings``. The challenger's own findings in the recorded run were out of scope (explanation objections), so
    they block nothing and are left out."""
    record = copy.deepcopy(RECORD) if record is None else record
    checks = recorded_checks() if checks is None else checks
    statements = {s["id"]: s for s in m_math.all_statements(record)}
    coverage = c_challenge.p1_coverage(record, checks)
    raw = copy.deepcopy(FIX["challenger_components"] if components is None else components)
    items = [c_challenge.component_disposition(c, CARD, record, checks, statements, coverage) for c in raw]
    findings = c_challenge.merge_findings(c_challenge.host_findings(record, items)
                                          + c_challenge.component_findings(items, record))
    new_premises, new_edges = [], []
    for item in findings:
        if item["disposition"] in c_challenge.BLOCKING:
            premise = c_challenge.finding_premise(item)
            new_premises.append(premise)
            new_edges += [{"premise_id": premise["id"], "obligation_id": oid} for oid in item["used_by"]]
    implies = {r["premise_id"]: r["implies"] for r in RECORDED["circularity"]}
    implies.update(readings or {})
    circular, circularity = {}, []
    for premise in list(record["premises"]) + new_premises:
        pid = premise["id"]
        supports = c_challenge.supported_claims(record, pid, new_edges)
        hit = sorted(set(implies.get(pid, [])) & supports)
        if hit:
            circular[pid] = f"restates or implies {', '.join(hit)}, which it supports: recorded reading"
        circularity.append({"premise_id": pid, "implies": sorted(set(implies.get(pid, []))),
                            "supports": sorted(supports), "circular": bool(hit), "why": "recorded reading"})
    shows_nothing = {sid: f"mismatched: {why}" for sid, why in (mismatched or {}).items()}
    c_challenge.settle_measurements(items, shows_nothing)
    blocked = {m_math.obligation_of(record, sid)["claim_id"]: f"statement {sid} does not match the claim"
               for sid in mismatched or {}}
    _, ledger, records = m_math.evaluate(record, checks, extra_edges=new_edges, extra_premises=new_premises,
                                         circular=circular, citations_checked=True, blocked=blocked,
                                         mismatched=mismatched, unresolved_claims={})
    settled = mathrecords.settled_at_m(ledger)
    missing = c_challenge.settle_magic_x(copy.deepcopy(RECORDED["magic_x"]), settled)
    c_challenge.settle_circularity(circularity, settled)
    gate = mathrecords.challenge_gate(records, ledger, conservation=[], magic_x=[], magic_x_missing=missing,
                                      counterexamples=[], mismatched=sorted(mismatched or {}))
    return {"components": {c["name"]: c for c in items}, "findings": {f["id"]: f for f in findings},
            "new_premises": new_premises, "new_edges": new_edges, "circularity": circularity,
            "ledger": {e["id"]: e for e in ledger}, "statuses": {r["claim_id"]: r["status"] for r in records},
            "records": {r["claim_id"]: r for r in records}, "gate": gate}


def component(name: str, *, used_by, role: str = "external", kind: str = "oracle", subject: str | None = None,
              statement_id=None, function=None, quote: str = CARD_QUOTE, why: str = "the conclusion needs it",
              magic_x=None) -> dict:
    return mf.component(name, quote=quote, statement_id=statement_id, function=function, used_by=used_by, role=role,
                        kind=kind, subject=subject, why=why, magic_x=magic_x)


# ----- the recorded run --------------------------------------------------------------------------------------------

def test_the_fixture_is_the_recorded_k1_run3_challenge():
    assert FIX["source"] == "runs/calibration-math-20261016T051933Z/k1-run3"
    assert FIX["sha256"]["challenge"] == "fd86a8c499f1dfe8ac246cbba81f04f3776391121d77884c7f94a3a104e24090"
    assert (CLASSES["role"], CLASSES["kind"], CLASSES["statement_id"], CLASSES["function"], CLASSES["used_by"]) == \
        ("external", "prior", None, None, ["O13"])
    assert "frozen in O13.a" in CLASSES["why"] and "not executable oracles supplied" in CLASSES["why"]
    recorded = next(c for c in RECORDED["components"] if c["name"] == CLASSES["name"])
    assert recorded["disposition"] == "substantiated" and recorded["covered_by"] is None
    finding = next(f for f in RECORDED["findings"] if f["id"] == PREMISE_ID)
    assert finding["origin"] == "component Restricted target-rule classes" and finding["does_work"] is True
    assert finding["text"].endswith("it exists as costed mechanism code, or the claims that use it hold without it")
    assert [p["id"] for p in RECORDED["new_premises"]] == [PREMISE_ID]
    reading = next(c for c in RECORDED["circularity"] if c["premise_id"] == PREMISE_ID)
    assert reading["circular"] is True and reading["implies"] == ["P5"]
    assert RECORDED["kill_codes"] == [f"circular:{PREMISE_ID}", "obligation_failed:P4.composition"]
    assert RECORDED["claims"]["P5"] == "circular" and RECORDED["ledger"][0]["status"] == "circular"
    # O13 is the P5 conservation obligation; O13.a held at M and freezes every hypothesis class it counts.
    o13 = next(o for o in RECORD["obligations"] if o["id"] == "O13")
    assert o13["claim_id"] == "P5" and [s["form"] for s in o13["statements"]] == ["conservation"]
    sample = o13["statements"][0]["spec"]["sample"]
    assert len(sample) == 4 and all(e["hypothesis_class"]["definition"] and e["hypothesis_class"]["log2_size"]
                                    for e in sample)
    check = FIX["m_checks"]["O13.a"]
    assert (check["form"], check["status"], check["decided"]) == ("conservation", "discharged", True)


def test_without_the_role_check_the_replay_reproduces_the_recorded_outcome(monkeypatch):
    """The replay is faithful: with the FIX-21 role check switched off, the recorded inputs give the recorded
    component dispositions, the recorded premise and the recorded verdict, so the tests below change only the rule."""
    disable_role_check(monkeypatch)
    out = stage_c()
    for recorded in RECORDED["components"]:
        assert out["components"][recorded["name"]]["disposition"] == recorded["disposition"], recorded["name"]
    assert out["new_premises"] == RECORDED["new_premises"]
    assert out["new_edges"] == [e for e in RECORDED["new_edges"] if e["premise_id"] == PREMISE_ID]
    assert out["ledger"][PREMISE_ID]["status"] == "circular"
    assert out["gate"] == (RECORDED["verdict"], RECORDED["kill_codes"])
    assert out["statuses"] == RECORDED["claims"]


# ----- the replay with FIX-21 -----------------------------------------------------------------------------------------

def test_the_replayed_k1_run3_gives_no_hidden_component_premise_and_p5_is_not_circular():
    out = stage_c()
    classes = out["components"][CLASSES["name"]]
    assert (classes["role"], classes["disposition"], classes["settled_by"]) == ("measurement", "measurement", ["O13.a"])
    assert classes["role_correction"]["from"] == "external" and classes["role_correction"]["to"] == "measurement"
    assert "O13.a define the P5 sample entries' hypothesis_class definitions" in classes["role_correction"]["reason"]
    assert "settled by O13.a, which held at stage M" in classes["reason"]
    assert PREMISE_ID not in out["findings"] and out["new_premises"] == [] and out["new_edges"] == []
    assert PREMISE_ID not in out["ledger"]
    assert not any(c["circular"] for c in out["circularity"])
    p5 = out["records"]["P5"]
    assert (p5["status"], p5["circular_premises"], p5["open_premises"]) == ("observed", [], [])
    assert out["gate"] == ("fail", ["obligation_failed:P4.composition"])
    # Only P5 changes; every other component keeps its recorded disposition and role.
    for cid, status in RECORDED["claims"].items():
        if cid != "P5":
            assert out["statuses"][cid] == status, cid
    for recorded in RECORDED["components"]:
        if recorded["name"] != CLASSES["name"]:
            item = out["components"][recorded["name"]]
            assert (item["disposition"], item["role"]) == (recorded["disposition"], recorded["role"])
            assert "role_correction" not in item


def test_a_measurement_component_has_no_magic_x_item_and_no_finding():
    out = stage_c()
    assert not any(f["origin"].startswith("component Restricted") for f in out["findings"].values())
    items = [c_challenge.component_disposition(copy.deepcopy(CLASSES), CARD, RECORD, recorded_checks(),
                                               {s["id"]: s for s in m_math.all_statements(RECORD)},
                                               c_challenge.p1_coverage(RECORD, recorded_checks()))]
    assert c_challenge.component_findings(items, RECORD) == []
    assert "measurement" not in c_challenge.BLOCKING and "measurement_open" not in c_challenge.BLOCKING


# ----- a measurement object whose defining statement did not hold stays open -----------------------------------------

def failed_o13(status: str, decided: bool) -> dict:
    checks = recorded_checks()
    checks["O13.a"].update(status=status, decided=decided, evidence="sample: violated: 3 examples cannot identify one "
                           "of 256 rules")
    return checks


@pytest.mark.parametrize("status,decided", [("violated", True), ("unknown", False)], ids=["violated", "undecided"])
def test_a_measurement_object_whose_defining_statement_did_not_hold_stays_open(status, decided):
    out = stage_c(checks=failed_o13(status, decided))
    classes = out["components"][CLASSES["name"]]
    assert classes["role"] == "measurement" and classes["disposition"] == "measurement_open"
    assert classes["settled_by"] == [] and f"which did not hold at stage M (O13.a {status})" in classes["reason"]
    assert "it stays open" in classes["reason"]
    # No premise, no auto-pass: P5 rests on O13.a, as without the component.
    assert out["new_premises"] == [] and PREMISE_ID not in out["ledger"]
    without = stage_c([c for c in FIX["challenger_components"] if c["name"] != CLASSES["name"]],
                      checks=failed_o13(status, decided))
    p5 = out["statuses"]["P5"]
    assert p5 != "observed" and p5 == without["statuses"]["P5"]
    assert out["gate"] == without["gate"]
    if decided:
        assert p5 == "failed" and "obligation_failed:P5" in out["gate"][1]
    assert out["gate"][0] == "fail"


def test_a_missing_defining_check_leaves_the_measurement_object_open():
    checks = recorded_checks()
    del checks["O13.a"]
    out = stage_c(checks=checks)
    classes = out["components"][CLASSES["name"]]
    assert classes["disposition"] == "measurement_open" and "O13.a not checked" in classes["reason"]
    assert out["statuses"]["P5"] != "observed"


def test_a_defining_statement_mismatched_at_stage_c_settles_nothing():
    out = stage_c(mismatched={"O13.a": "the sample entries count a different class"})
    classes = out["components"][CLASSES["name"]]
    assert classes["disposition"] == "measurement_open" and classes["settled_by"] == []
    assert "shows nothing at stage C (O13.a: mismatched: the sample entries count" in classes["reason"]
    assert out["statuses"]["P5"] == "blocked" and out["new_premises"] == []


# ----- a real external helper is still caught -------------------------------------------------------------------------

def hidden_part_cases() -> list:
    return [
        pytest.param(component("Content lookup oracle", used_by=["O4"], kind="oracle", subject="recall oracle",
                               why="an oracle supplied to the model returns the value stored after the matching key"),
                     "P2.recall", id="oracle supplied to a P2 witness"),
        pytest.param(component("Selector prior in the embeddings", used_by=["O13"], kind="prior",
                               subject="hidden prior over selectors",
                               why="the embeddings encode the identity selector before any example is seen"),
                     "P5", id="prior given to the model, used by P5 only"),
        pytest.param(component("Restricted target-rule classes", used_by=["O13", "O8"], kind="prior",
                               subject="sample information classes",
                               why="the classes are given to the learner as a restricted parameterization"),
                     "P5", id="class named, but also used by a learnability obligation"),
        pytest.param(component("Restricted target-rule classes", used_by=["O1", "O13"], kind="prior",
                               subject="sample information classes"),
                     "P5", id="class named, but also used by the cost obligation"),
    ]


@pytest.mark.parametrize("part,claim", hidden_part_cases())
def test_a_real_external_helper_still_becomes_a_magic_x_premise(part, claim):
    out = stage_c([*FIX["challenger_components"][:9], part, *FIX["challenger_components"][10:]],
                  readings={})
    item = out["components"][part["name"]]
    assert item["role"] == "external" and item["disposition"] == "substantiated" and "role_correction" not in item
    fid = c_challenge.finding_id(RECORD, part["subject"], part["used_by"])
    finding = out["findings"][fid]
    assert finding["does_work"] is True and finding["kind"] == "missing_assumption"
    assert finding["text"].endswith("it exists as costed mechanism code, or the claims that use it hold without it")
    assert out["ledger"][fid]["status"] == "open"
    # The gate lists failures first (P4.composition failed in this run), so the open premise shows on the claim.
    assert out["statuses"][claim] == "conditional" and out["records"][claim]["open_premises"] == [fid]
    assert out["gate"][0] == "fail"


def test_a_real_external_helper_with_magic_x_code_is_still_run_by_magic_x():
    """The magic-X loop runs on a substantiated finding that does work and carries X code: the helper keeps both."""
    code = "def x(prefix, layout, ops):\n    return 0\n"
    oracle = component("Content lookup oracle", used_by=["O4"], kind="oracle", subject="recall oracle",
                       magic_x={"code": code, "claim": None, "quote": CARD_QUOTE, "why": "an oracle"})
    out = stage_c([*FIX["challenger_components"], oracle])
    finding = next(f for f in out["findings"].values() if f["subject"] == "recall oracle")
    assert finding["disposition"] == "substantiated" and finding["does_work"] and finding["magic_x"]["code"] == code


def test_a_runnable_part_of_a_witness_is_never_a_measurement_object():
    """A function the witness source defines is code the model runs, whatever it is called."""
    generator = component("Copy task generator", used_by=["O3"], kind="other", subject="copy task generator",
                          statement_id="O3.a", function="build")
    out = stage_c([*FIX["challenger_components"], generator])
    item = out["components"]["Copy task generator"]
    assert item["runnable"] and item["role"] == "external" and item["disposition"] != "measurement"
    assert "role_correction" not in item


# ----- the host corrects and records a role ---------------------------------------------------------------------------

def test_a_mislabeled_measurement_object_is_corrected_and_recorded():
    prior = component("P5 hypothesis classes", used_by=["O13"], role="mechanism", kind="prior",
                      subject="hypothesis class sizes")
    generator = component("Recall task generator", used_by=["O4"], kind="other", subject="recall instances")
    out = stage_c([*FIX["challenger_components"], prior, generator])
    item = out["components"]["P5 hypothesis classes"]
    assert item["role_correction"]["from"] == "mechanism" and item["role"] == "measurement"
    assert item["disposition"] == "measurement" and item["settled_by"] == ["O13.a"]
    item = out["components"]["Recall task generator"]
    assert (item["role_correction"]["from"], item["role"], item["disposition"], item["settled_by"]) == \
        ("external", "measurement", "measurement", ["O4.a"])
    assert "the P2 task generator, its frozen instances and protocol constants" in item["role_correction"]["reason"]
    assert out["new_premises"] == [] and out["gate"] == ("fail", ["obligation_failed:P4.composition"])


def test_a_component_labelled_measurement_that_the_model_uses_is_corrected_to_external():
    labelled = component("Restricted target-rule classes", used_by=["O1"], role="measurement", kind="prior",
                         subject="selector classes in the reference")
    out = stage_c([labelled])
    item = out["components"][labelled["name"]]
    assert item["role_correction"] == {"from": "measurement", "to": "external", "reason": (
        "host role check: labelled a measurement object, but O1 define no measurement object (only P5 conservation "
        "statements with defined hypothesis classes and P2 task statements do), so it is a part of what the model "
        "uses; it is settled as a part supplied from outside the mechanism")}
    assert item["role"] == "external" and item["disposition"] == "substantiated"
    fid = c_challenge.finding_id(RECORD, labelled["subject"], ["O1"])
    assert out["ledger"][fid]["status"] == "open" and out["statuses"]["P1"] == "conditional"


def test_a_measurement_label_on_a_p5_entry_whose_class_is_undefined_is_corrected_to_external():
    record = copy.deepcopy(RECORD)
    o13 = next(o for o in record["obligations"] if o["id"] == "O13")
    del o13["statements"][0]["spec"]["sample"][0]["hypothesis_class"]
    labelled = dict(CLASSES, role="measurement")
    out = stage_c([labelled], record=record)
    item = out["components"][CLASSES["name"]]
    assert item["role"] == "external" and item["role_correction"]["from"] == "measurement"
    assert "O13 define no measurement object" in item["role_correction"]["reason"]
    assert item["disposition"] == "substantiated"


@pytest.mark.parametrize("name,subject,kind,expected", [
    ("Restricted target-rule classes", "sample information classes", "hypothesis_class", True),
    ("256-rule hypothesis class", "class", "hypothesis_class", True),
    ("selectors", "hypothesis_class sizes", "hypothesis_class", True),
    ("Rule classes", "information bound", "hypothesis_class", True),
    ("sample information classes", "information bound", "hypothesis_class", False),
    ("Selector prior", "prior over target rules", "hypothesis_class", False),
    ("Class labels oracle", "label oracle", "hypothesis_class", False),
    ("Copy task generator", "copy", "task_generator", True),
    ("Held-out test set", "evaluation", "task_generator", True),
    ("P2 accuracy threshold", "protocol", "task_generator", True),
    ("Instance count", "protocol constant", "task_generator", True),
    ("Parse-tree generator", "nesting parser", "task_generator", False),
    ("Answer table", "precomputed answers", "task_generator", False),
    ("Depth decoder", "decoder", "task_generator", False),
])
def test_the_name_test_is_narrow(name, subject, kind, expected):
    pattern = c_challenge.MEASUREMENT_OBJECTS[kind][1]
    assert bool(pattern.search(f"{name} | {subject}")) is expected


def test_the_role_check_reads_structure_not_the_challengers_reasoning():
    """The recorded component's reasoning ("not executable oracles supplied to the trained model") decides nothing:
    the same reasoning on a component used by P2 and P5 together is not a measurement object, and the recorded
    component with a reasoning that claims the opposite still is one."""
    mixed = dict(CLASSES, used_by=["O4", "O13"])
    definition, why = c_challenge.measurement_definition(mixed, RECORD, False)
    assert definition is None and "different measurement objects" in why
    opposite = dict(CLASSES, why="an external prior")
    definition, why = c_challenge.measurement_definition(opposite, RECORD, False)
    assert definition["object"] == "hypothesis_class" and definition["statement_ids"] == ["O13.a"]


def test_a_card_claim_obligation_never_defines_a_measurement_object():
    for obligation in RECORD["obligations"]:
        expected = {"P5": "hypothesis_class"}.get(obligation["claim_id"]) or (
            "task_generator" if (obligation["claim_id"] or "").startswith("P2.") else None)
        assert c_challenge.measured_object(RECORD, obligation) == expected, obligation["id"]
    unprofiled = dict(copy.deepcopy(RECORD), profile=None)
    assert all(c_challenge.measured_object(unprofiled, o) is None for o in unprofiled["obligations"])


# ----- the contract and the prompt ----------------------------------------------------------------------------------

def test_the_challenger_schema_accepts_the_measurement_role():
    schema = mc.challenger_schema(["A1"], ["O1", "O13"], ["O1.a"], [], [], statement_ids=["O1.a", "O13.a"])
    roles = schema["properties"]["components"]["items"]["properties"]["role"]["enum"]
    assert roles == ["mechanism", "external", "measurement"]
    assert mc.COMPONENT_ROLES == ("mechanism", "external", "measurement")


def test_the_prompts_define_the_measurement_role_with_the_p5_hypothesis_class():
    section = load_section("challenge_findings")
    assert section.startswith("ANSWER FORMAT FOR COMPONENTS, FINDINGS AND CONSERVATION (FIX-4).")
    paragraph = section.split("MEASUREMENT OBJECTS (FIX-21).", 1)[1].split("\n\n", 1)[0]
    for phrase in ('`role` "measurement"', "never supplied to the model at inference or training",
                   "a P5 hypothesis class", "a P2 task generator", "an evaluation set", "a protocol constant",
                   "`hypothesis_class`", "only when that statement held", "never becomes a premise",
                   "stays open", "is settled as external"):
        assert phrase in paragraph, phrase
    system = (Path(c_challenge.__file__).parents[1] / "prompts" / "challenger.md").read_text(encoding="utf-8")
    assert "measurement object" in system and '`role` "measurement"' in system
    for text in (paragraph, system):
        assert "\u2014" not in text and not re.search(r"\bcanonical\b|load-bearing", text)


# ----- stage C end to end (scripted LLMs, the k1 calibration card) -----------------------------------------------

def run_k1(tmp_path, components: list[dict], implies: dict[str, list[str]]) -> Harness:
    card = load_cards(("k1",))["k1"]
    script = cf.script("k1", card)
    script["challenger"] = mf.challenger_output(components=components)
    script["statement_match"] = mf.match_output(implies=implies)
    h = Harness(tmp_path, script=script)
    h.m()
    h.c()
    return h


def k1_classes() -> tuple[dict, str]:
    """The recorded component, used by the P5 obligation of the scripted k1 record (O11 there)."""
    card = load_cards(("k1",))["k1"]
    p5 = oid_of(cf.fixer_output("k1", card), "P5")
    return dict(copy.deepcopy(CLASSES), used_by=[p5]), p5


def challenge_artifact(h: Harness) -> dict:
    return [json.loads((h.run_dir / e["data"]["path"]).read_text()) for e in h.ledger.iter_events("artifact")
            if e["data"]["kind"] == "challenge"][-1]


def c_ledger(h: Harness) -> dict[str, dict]:
    verdict = h.verdict("C")
    data = json.loads((h.run_dir / "artifacts" / "assumption_ledger" / verdict["assumption_ledger"]).read_text())
    return {p["id"]: p for p in data["premises"]}


def test_stage_c_before_the_fix_reproduces_the_circular_p5(tmp_path, monkeypatch):
    disable_role_check(monkeypatch)
    classes, p5 = k1_classes()
    h = run_k1(tmp_path, [classes], {PREMISE_ID: ["P5"]})
    assert c_challenge.finding_id(h.frozen(), classes["subject"], [p5]) == PREMISE_ID
    assert h.slot("M", "P5")["status"] == "observed"
    assert h.slot("C", "P5")["status"] == "circular" and f"circular:{PREMISE_ID}" in h.verdict("C")["kill_codes"]
    assert c_ledger(h)[PREMISE_ID]["status"] == "circular"


def test_stage_c_records_the_correction_and_keeps_p5_observed(tmp_path):
    classes, p5 = k1_classes()
    h = run_k1(tmp_path, [classes], {PREMISE_ID: ["P5"]})
    verdict = h.verdict("C")
    validate("math_verdict", verdict)
    assert h.slot("C", "P5")["status"] == "observed" and h.slot("C", "P5")["circular_premises"] == []
    assert not any(code.endswith(PREMISE_ID) for code in verdict["kill_codes"])
    assert PREMISE_ID not in c_ledger(h)
    assert all(c["premise_id"] != PREMISE_ID for c in verdict["circularity"])
    assert all(m["premise_id"] != PREMISE_ID for m in verdict["magic_x"])
    # The challenge artifact keeps the challenger's label in the correction, with the host's reason.
    item = next(c for c in challenge_artifact(h)["components"] if c["name"] == CLASSES["name"])
    assert (item["role"], item["disposition"], item["settled_by"]) == ("measurement", "measurement", [f"{p5}.a"])
    assert item["role_correction"]["from"] == "external" and "host role check" in item["role_correction"]["reason"]
    assert item["measurement"]["object"] == "hypothesis_class"
    # The run ledger records it too, and so does the stage C verdict.
    events = [e["data"] for e in h.ledger.iter_events("challenge_check") if e["data"]["check"] == "component_role"]
    assert len(events) == 1
    assert (events[0]["component"], events[0]["role"], events[0]["disposition"]) == \
        (CLASSES["name"], "measurement", "measurement")
    assert events[0]["role_correction"]["from"] == "external"
    assert verdict["component_roles"] == [{k: item.get(k) for k in ("name", "subject", "role", "disposition",
                                                                     "settled_by", "role_correction", "reason")}]
    # The challenger was told about the role.
    user = next(c["user"] for c in h.llm.calls if c["role"] == "challenger")
    assert "MEASUREMENT OBJECTS (FIX-21)." in user


def test_stage_c_still_blocks_on_an_oracle_supplied_to_a_p2_witness(tmp_path):
    card = load_cards(("k1",))["k1"]
    recall = oid_of(cf.fixer_output("k1", card), "P2.recall")
    oracle = component("Content lookup oracle", used_by=[recall], kind="oracle", subject="recall oracle",
                       why="an oracle supplied to the model returns the value stored after the matching key")
    h = run_k1(tmp_path, [oracle], {})
    fid = c_challenge.finding_id(h.frozen(), "recall oracle", [recall])
    assert c_ledger(h)[fid]["status"] == "open"
    slot = h.slot("C", "P2.recall")
    assert slot["status"] == "conditional" and slot["open_premises"] == [fid]
    item = next(c for c in challenge_artifact(h)["components"] if c["name"] == "Content lookup oracle")
    assert item["role"] == "external" and "role_correction" not in item
    assert not [e for e in h.ledger.iter_events("challenge_check") if e["data"]["check"] == "component_role"]
