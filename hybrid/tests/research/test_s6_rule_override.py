"""S6: a rule-decided outcome overrides the analyst, and a falsifier task where the candidate decided nothing gates
every analyst entry that uses its numbers.

The unit tests build the falsifiers block with the real falsifier_eval.decide on scripted scores. The line test
runs the whole loop with the scripted LLM and FakeBench. The shapes follow runs/run-20261011T042400Z
card-03fa1d38: the only rule of three_atom_composition names a capability, no rule names the card falsifier
that the same protocol tests, the candidate did not train, and the analyst marked that falsifier fired from the
task's composed-token accuracy (0.0078)."""
import asyncio
import json

import pytest

from hybrid.bench import falsifier_eval
from hybrid.research.stages import falsify
from hybrid.research.stages.s6_verdict import (SOURCES_NOTE, claim_line, falsifier_section, merge, task_gate,
                                               task_gates)
from hybrid.tests.research.builders import load_example
from hybrid.tests.research.test_orchestrator_fakes import (FakeBench, falsifier_design, falsifier_rules,
                                                           fake_scores, make_orchestrator)

LABELS = ["m=1", "m=2"]
REFS = {"uniform": fake_scores(0.0078125, 4.85, LABELS), "unigram": fake_scores(0.004, 18.0, LABELS)}
UNTRAINED = {"candidate": (0.0078125, 18.52), "ngram": (0.38, 2.0)}
TRAINED_MEETS = {"candidate": (0.95, 0.1), "ngram": (0.3, 2.0)}
TRAINED_MISSES = {"candidate": (0.5, 1.0), "ngram": (0.45, 1.2)}


def block(scripted: dict, rules: list[dict], blocked: str | None = None) -> dict:
    """A bench_raw falsifiers entry decided by the real evaluator."""
    scores = {name: (None if value is None else fake_scores(*value, LABELS)) for name, value in scripted.items()}
    decisions = [{"rule_index": i, "claims": r["claims"], "quote": r["quote"], "rule": r,
                  **falsifier_eval.decide(r, {**{k: v for k, v in scores.items() if v is not None}, **REFS}, rules,
                                          blocked)}
                 for i, r in enumerate(rules)]
    entrants = {name: ({"status": "ok", "metrics": s} if s is not None else {"status": "job_failed"})
                for name, s in scores.items()}
    return {"title": "three atom composition", "protocol": "p", "deviations": "d", "entrants": entrants,
            "decisions": decisions}


def entry(claim_id: str, status: str, tasks=(), evidence: str = "numbers") -> dict:
    return {"claim_id": claim_id, "claim": claim_id, "status": status, "evidence": evidence,
            "falsifier_tasks": list(tasks)}


def ledger_for(falsifiers: dict, *entries: dict) -> dict:
    return merge("card-03fa1d38", falsify.claim_items(load_example()), {"claims": list(entries)}, falsifiers)


def by_id(ledger: dict, claim_id: str) -> dict:
    return next(c for c in ledger["claims"] if c["claim_id"] == claim_id)


CAPABILITY_ONLY = [falsifier_rules()[0]]
DID_NOT_TRAIN = "the candidate did not train on this task"


# ----- the card-03fa1d38 case ------------------------------------------------------------------------------

def test_untrained_task_keeps_a_falsifier_without_a_rule_inconclusive_and_notes_the_analyst():
    falsifiers = {"three_atom_composition": block(UNTRAINED, CAPABILITY_ONLY)}
    ledger = ledger_for(falsifiers, entry("capability.1", "untested"),
                        entry("falsifier.2", "refuted", ["three_atom_composition"],
                              "composed-token accuracy 0.0078 in three_atom_composition"))
    fals = by_id(ledger, "falsifier.2")
    assert fals["rules"] == []                                  # no rule names it: the link is the analyst's task list
    assert (fals["status"], fals["decided_by"], fals["analyst_status"]) == ("inconclusive", "rule", "refuted")
    assert fals["analyst_tasks"] == ["three_atom_composition"]
    (gate,) = fals["gated_by"]
    assert gate["task_id"] == "three_atom_composition" and gate["reason"].startswith(DID_NOT_TRAIN)
    assert fals["analyst_override"].startswith(
        "analyst would have marked fired; overridden by rule: three_atom_composition: " + DID_NOT_TRAIN)
    assert ledger["falsified_by"] == []
    assert ledger["counts"] == {"supported": 0, "refuted": 0, "inconclusive": 2, "untested": 7}
    line = claim_line(fals)
    assert line.startswith("falsifier 2 inconclusive (neither fired nor ruled out) [rule]: ")
    assert "(analyst would have marked fired; overridden by rule: three_atom_composition: " in line
    cap = by_id(ledger, "capability.1")             # the rule that does exist keeps its own inconclusive outcome
    assert (cap["status"], cap["decided_by"]) == ("inconclusive", "rule") and "analyst_override" not in cap


def test_untrained_rule_for_a_falsifier_wins_over_the_analyst_marking_it_fired():
    falsifiers = {"iterated_shift": block(UNTRAINED, falsifier_rules())}
    ledger = ledger_for(falsifiers, entry("falsifier.1", "refuted", ["iterated_shift"]))
    fals = by_id(ledger, "falsifier.1")
    assert (fals["status"], fals["decided_by"], fals["gated_by"]) == ("inconclusive", "rule", [])
    assert fals["rules"][0]["reason"].startswith(DID_NOT_TRAIN)
    assert fals["analyst_override"].startswith("analyst would have marked fired; overridden by rule: iterated_shift "
                                               "rule 1: inconclusive (" + DID_NOT_TRAIN)
    assert ledger["counts"]["refuted"] == 0 and ledger["falsified_by"] == []


def test_a_task_gate_applies_without_the_analyst_citing_a_rule_named_claim_and_to_supported_readings():
    falsifiers = {"three_atom_composition": block(UNTRAINED, CAPABILITY_ONLY)}
    ledger = ledger_for(falsifiers, entry("capability.2", "supported", ["three_atom_composition"]))
    cap = by_id(ledger, "capability.2")
    assert (cap["status"], cap["decided_by"]) == ("inconclusive", "rule")
    assert cap["analyst_override"].startswith("analyst would have marked supported; overridden by rule: ")


# ----- rule outcomes against the analyst ---------------------------------------------------------------------

def test_rule_supported_overrides_analyst_refuted():
    falsifiers = {"iterated_shift": block(TRAINED_MEETS, CAPABILITY_ONLY)}
    ledger = ledger_for(falsifiers, entry("capability.1", "refuted", ["iterated_shift"]))
    cap = by_id(ledger, "capability.1")
    assert (cap["status"], cap["decided_by"], cap["analyst_status"]) == ("supported", "rule", "refuted")
    assert cap["analyst_override"] == ("analyst would have marked refuted; overridden by rule: iterated_shift rule 0: "
                                       "supported (rule met, so the claim is supported)")
    assert ledger["counts"]["supported"] == 1 and ledger["counts"]["refuted"] == 0
    assert claim_line(cap).startswith("supported [rule]: ") and "analyst would have marked refuted" in claim_line(cap)


def test_rule_refuted_overrides_analyst_supported():
    falsifiers = {"iterated_shift": block(TRAINED_MISSES, falsifier_rules())}
    ledger = ledger_for(falsifiers, entry("falsifier.1", "supported", ["iterated_shift"]),
                        entry("capability.1", "refuted", ["iterated_shift"]))
    fals = by_id(ledger, "falsifier.1")
    assert (fals["status"], fals["decided_by"]) == ("refuted", "rule")
    assert fals["analyst_override"].startswith("analyst would have marked not fired; overridden by rule: "
                                               "iterated_shift rule 1: refuted")
    assert ledger["falsified_by"] == [{"claim_id": "falsifier.1", "task_id": "iterated_shift", "rule_index": 1}]
    cap = by_id(ledger, "capability.1")             # the analyst agreed with the rule: no note
    assert cap["status"] == "refuted" and "analyst_override" not in cap
    assert ledger["counts"] == {"supported": 0, "refuted": 2, "inconclusive": 0, "untested": 7}


def test_without_a_rule_or_a_gated_task_the_analyst_decides():
    falsifiers = {"iterated_shift": block(TRAINED_MEETS, CAPABILITY_ONLY)}
    ledger = ledger_for(falsifiers, entry("falsifier.2", "refuted", ["iterated_shift"]),
                        entry("capability.2", "supported"), entry("complexity.memory", "refuted"))
    for claim_id, status in (("falsifier.2", "refuted"), ("capability.2", "supported"),
                             ("complexity.memory", "refuted")):
        item = by_id(ledger, claim_id)
        assert (item["status"], item["decided_by"], item["gated_by"]) == (status, "analyst", [])
        assert "analyst_override" not in item
    assert claim_line(by_id(ledger, "falsifier.2")).startswith("falsifier 2 fired, card refuted on this point [analyst]")
    assert ledger["falsified_by"] == []                       # falsified_by lists rule decisions only
    assert ledger["counts"] == {"supported": 2, "refuted": 2, "inconclusive": 0, "untested": 5}


def test_an_untested_entry_that_cites_a_gated_task_stays_untested():
    falsifiers = {"three_atom_composition": block(UNTRAINED, CAPABILITY_ONLY)}
    ledger = ledger_for(falsifiers, entry("falsifier.2", "untested", ["three_atom_composition"]))
    fals = by_id(ledger, "falsifier.2")
    assert (fals["status"], fals["decided_by"], fals["gated_by"]) == ("untested", "analyst", [])


def test_an_entry_citing_a_task_s5_did_not_run_is_an_error():
    with pytest.raises(ValueError, match="did not run"):
        ledger_for({}, entry("falsifier.2", "refuted", ["three_atom_composition"]))


# ----- the task gate --------------------------------------------------------------------------------------------

def test_task_gate_reads_the_evaluators_task_level_decisions():
    assert task_gate(block(UNTRAINED, CAPABILITY_ONLY)).startswith(DID_NOT_TRAIN)
    assert task_gate(block({"candidate": None, "ngram": (0.3, 2.0)}, CAPABILITY_ONLY)) == \
        "the candidate produced no valid predictions on this task"
    assert task_gate(block(TRAINED_MEETS, CAPABILITY_ONLY, blocked="the protocol needs d = 64")) == \
        "the protocol needs d = 64"
    assert task_gate(block(TRAINED_MEETS, falsifier_rules())) is None
    # a rule-level gate (an entrant named in the rule without predictions) does not gate the task
    assert task_gate(block({"candidate": (0.95, 0.1), "ngram": None}, falsifier_rules())) is None
    gates = task_gates({"a": block(UNTRAINED, CAPABILITY_ONLY), "b": block(TRAINED_MEETS, CAPABILITY_ONLY)})
    assert list(gates) == ["a"]


def test_the_analyst_sees_the_gate_and_is_told_to_list_its_tasks():
    section = falsifier_section({"three_atom_composition": block(UNTRAINED, CAPABILITY_ONLY)})
    assert "candidate gate: " + DID_NOT_TRAIN in section
    assert "candidate gate" not in falsifier_section({"t": block(TRAINED_MEETS, CAPABILITY_ONLY)})
    assert "falsifier_tasks" in SOURCES_NOTE


# ----- the whole line -------------------------------------------------------------------------------------------

def test_card_03fa1d38_shape_through_the_loop(run_dir):
    script = {"falsifier_designer": {"tasks": [falsifier_design(rules=[dict(falsifier_rules()[0])])],
                                     "untestable": [{"claim_id": "falsifier.2", "why": "no numeric threshold"}]},
              "analyst": {"claims": [entry("capability.1", "untested", ["iterated_shift"]),
                                     entry("falsifier.2", "refuted", ["iterated_shift"],
                                           "composed-token accuracy 0.0078")]}}
    orch, llm = make_orchestrator(run_dir, script=script,
                                  bench=FakeBench(falsifier_scores={"candidate": (0.0078125, 18.52),
                                                                    "ngram": (0.38, 2.0)}))
    asyncio.run(orch.run())
    (card_id,) = orch.store.list_cards()
    reasons = orch.store.get_card(card_id)["status"]["verdicts"]["S6"]["reasons"]
    assert reasons[0] == ("claims ledger: 0 supported, 0 refuted, 2 inconclusive, 7 untested; "
                          "2 decided by falsifier rules")
    (line,) = [r for r in reasons if r.startswith("falsifier 2 ")]
    assert line.startswith("falsifier 2 inconclusive (neither fired nor ruled out) [rule]: ")
    assert "analyst would have marked fired; overridden by rule: iterated_shift: " + DID_NOT_TRAIN in line
    assert not any("fired, card refuted" in r for r in reasons)
    analyst_user = [c["user"] for c in llm.calls if c["prompt"] == "analyst"][0]
    assert SOURCES_NOTE in analyst_user and "candidate gate: " + DID_NOT_TRAIN in analyst_user
    recorded = [e["data"] for e in orch.ledger.iter_events("artifact") if e["data"]["kind"] == "claims_ledger"]
    ledger = json.loads((orch.run_dir / recorded[-1]["path"]).read_text())
    assert by_id(ledger, "falsifier.2")["gated_by"][0]["task_id"] == "iterated_shift"


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261011T000000Z"
