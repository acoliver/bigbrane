"""Decisions P4P and P4W: card-supplied P4 learning protocols.

Calibration P4 failed because the prover wrote a fresh training recipe and learner in every card-run
(results/p4-diag-20261016/REPORT.md). Cards K1 and K2 now supply a measured learner for every P4 task: K2 composition
at d = 64 and recall at d = 128 (results/p4-protocols-20261016/REPORT.md; P4W moves K2's recall learning width from 64
to 128), and K1 copy, recall, nesting and composition plus K2 copy and nesting (results/p4-protocols2-20261016/REPORT.md).
The K1 composition learner is the measured learner that does not reliably learn composition (decision P4K): it passed 2
of 3 check seeds, so its recorded verdict is violated. Stage M checks such a learner verbatim through the same check as
a prover learner and asks the prover for none; a hash mismatch stops the load and the stage; without a protocol
nothing changes.

The full checks of the learners are answered with their recorded host results in the stage tests
(calibration_fakes.FastExecutor). ``test_the_k2_recall_protocol_replays_through_the_stage_m_learnability_path`` trains
the recall learner for real on check seed 0 through the stage's runner; it takes minutes, so it is marked slow and runs
only with BIGBRANE_RUN_SLOW=1.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from hybrid.math import constructions, learnability, profile
from hybrid.math.runner import LIBRARY_CAPS, MathRunner
from hybrid.research import calibration as cal
from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.stages import c_challenge, card_resources as cr, m_math
from hybrid.research.stages.common import StageError, draft_schema, load_prompt, load_section
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_profile_stage import Harness

# The sha256 values results/p4-protocols-20261016/REPORT.md gives for the K2 composition and recall sources.
COMPOSITION_SHA = "37beb27be234fd101a66d51cfe818e9210eeb7e75cb017adfe85b85126925be8"
RECALL_SHA = "5905fdfb33939f82f8d50300cf62153d1b4f6d1c332c209a90de86a7e9e8b038"
PROTOCOL_DIR = cal.FIXTURES / "protocols"
SUPPLIED_SLOTS = ("P4.recall", "P4.composition")

# Every card protocol: (card, learning task) -> (file name stem, sha256 of the validated source, the report that
# validated it). The file stem names the task as the card does ("nesting" for the dyck learning task). The K1 and K2
# copy, K1 recall, nesting and composition and K2 nesting hashes are those of results/p4-protocols2-20261016/protocols.json.
P1 = "results/p4-protocols-20261016"
P2 = "results/p4-protocols2-20261016"
PROTOCOLS = {
    ("k1", "copy"): ("k1-copy", "e264cfca5b7c15a156a233cdf0b1ae87c73c9b0b7233e4deaefb58f6a93dbc6a", P2),
    ("k1", "recall"): ("k1-recall", "ead21867ef878030a7fa0488975fe713f35fb327c0ac54b28e5635521b0595ab", P2),
    ("k1", "dyck"): ("k1-nesting", "760d56fa2728c06d8eb51cef0c683424ab286e0e23fe53dcdd8c77d68f41988e", P2),
    ("k1", "composition"): ("k1-composition", "80d8d44382ef404ceb1257d82db39501f8677856d9b9655166b4b8d150c67d46", P2),
    ("k2", "copy"): ("k2-copy", "cf38d4628ab6cf2d850316aa98fb64edaa946fc5da8433ecc337087da16911c9", P2),
    ("k2", "recall"): ("k2-recall", RECALL_SHA, P1),
    ("k2", "dyck"): ("k2-nesting", "1ff702d960a4610dfdf1ed06e05abb3d96b9aff0ff8824ae0094910e027e4b63", P2),
    ("k2", "composition"): ("k2-composition", COMPOSITION_SHA, P1),
}
# The card's frozen learning case each protocol was validated for: size, examples m, d, rounds.
LEARNING_CASES = {
    ("k1", "copy"): (4, 3000, "64", "2"), ("k1", "recall"): (4, 3000, "64", "2"),
    ("k1", "dyck"): (2, 1500, "64", "2"), ("k1", "composition"): (4, 4000, "64", "2"),
    ("k2", "copy"): (4, 1500, "64", "1"), ("k2", "recall"): (4, 2500, "128", "1"),
    ("k2", "dyck"): (2, 1200, "64", "1"), ("k2", "composition"): (4, 3000, "64", "1"),
}
PRECISION = {"k1": 64, "k2": 32}
# The recorded host verdict of each protocol: K1 composition is the measured learner that fails (decision P4K).
RECORDED_STATUS = {key: "violated" if key == ("k1", "composition") else "discharged" for key in PROTOCOLS}
PROTOCOL_CASES = sorted(PROTOCOLS)
P4_SLOT_OF_TASK = {tasks[0]: slot for slot, tasks in profile.P4_SLOTS.items()}


def card_of(key: str) -> dict:
    return copy.deepcopy(cal.load_cards((key,))[key])


def k2() -> dict:
    return card_of("k2")


def only_protocols(card: dict, tasks) -> dict:
    """The card with the protocols of the other learning tasks removed."""
    card = copy.deepcopy(card)
    for task, entry in card["resources"]["learning"].items():
        if task not in tasks:
            entry.pop("protocol", None)
    return card


without_protocols = cf.without_protocols


def slot_statements(record: dict, slots) -> dict[str, dict]:
    return {s["id"]: s for o in record["obligations"] if o["claim_id"] in slots for s in o["statements"]}


def prover_items(h: Harness) -> list[dict]:
    return [item for call in h.llm.calls if call["prompt"] == "prover" for item in cf._shown(call["user"])]


def check_artifacts(h: Harness, statement_id: str) -> list[dict]:
    return [json.loads((h.run_dir / "artifacts" / "math_check" / e["data"]["artifact"]).read_text())
            for e in h.ledger.iter_events("math_check") if e["data"]["statement_id"] == statement_id]


# ----- 1. the card field and the K2 fixture ------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["k1", "k2"])
def test_each_card_carries_a_protocol_for_every_p4_task_and_their_sha256_match_the_reports(key):
    protocols = cr.learning_protocols(card_of(key))
    assert sorted(protocols) == ["composition", "copy", "dyck", "recall"]
    for task, protocol in protocols.items():
        stem, sha, report = PROTOCOLS[(key, task)]
        path = PROTOCOL_DIR / f"{stem}.py"
        assert protocol.sha256 == sha and hashlib.sha256(path.read_bytes()).hexdigest() == sha
        assert protocol.path == f"hybrid/fixtures/calibration/protocols/{stem}.py"
        assert protocol.source == path.read_text(encoding="utf-8")
        assert protocol.provenance.startswith(f"Measured learner from {report} ")
    record = protocols["recall"].record()
    stem = PROTOCOLS[(key, "recall")][0]
    assert record == {"origin": "card", "task": "recall", "path": f"hybrid/fixtures/calibration/protocols/{stem}.py",
                      "sha256": PROTOCOLS[(key, "recall")][1], "provenance": protocols["recall"].provenance}


def test_the_protocol_directory_holds_exactly_the_cards_sources():
    assert sorted(path.name for path in PROTOCOL_DIR.glob("*.py")) == sorted(f"{s}.py" for s, _, _ in PROTOCOLS.values())


@pytest.mark.parametrize("key, task", PROTOCOL_CASES)
def test_each_protocol_is_wired_to_the_cards_frozen_learning_case(key, task):
    """The protocol applies only when the card's learning entry is the case it was validated for."""
    entry = card_of(key)["resources"]["learning"][task]
    assert (entry["size"], entry["examples"], entry["d"], entry["rounds"]) == LEARNING_CASES[(key, task)]


def test_the_k1_composition_provenance_says_the_learner_does_not_reliably_learn_composition():
    protocol = cr.learning_protocols(card_of("k1"))["composition"]
    assert "does not reliably learn composition" in protocol.provenance and "P4K" in protocol.provenance
    assert "passed 2 of 3 check seeds" in protocol.provenance
    assert "the expected verdict is violated" in protocol.provenance
    for key, task in PROTOCOL_CASES:
        if (key, task) != ("k1", "composition"):
            provenance = cr.learning_protocols(card_of(key))[task].provenance
            assert "violated" not in provenance and "P4K" not in provenance


def test_k2_recall_learns_at_width_128_and_k1_stays_at_width_64():
    cards = cal.load_cards(("k1", "k2"))
    learning = cards["k2"]["resources"]["learning"]
    assert learning["recall"]["d"] == "128"
    assert {task: entry["d"] for task, entry in learning.items() if task != "recall"} == {
        "copy": "64", "dyck": "64", "composition": "64"}
    assert "recall, which learns at a store width of 128" in cards["k2"]["resources"]["why"]
    assert {entry["d"] for entry in cards["k1"]["resources"]["learning"].values()} == {"64"}
    assert not cal.anonymization_problems(cards["k1"]) and not cal.anonymization_problems(cards["k2"])


@pytest.mark.parametrize("key, task", PROTOCOL_CASES)
def test_each_protocol_source_states_the_cards_frozen_width_and_precision(key, task):
    """The learner's own RECIPE width and its arithmetic equal the card's learning d and its precision contract."""
    card = card_of(key)
    protocol = cr.learning_protocols(card)[task]
    recipe = learnability.recipe_of(learnability.load_learner(protocol.source))
    entry = card["resources"]["learning"][task]
    width = recipe["d"] if "d" in recipe else recipe["width_d"]
    assert width == cr.evaluate(entry["d"], {constructions.SIZE_NAME[task]: entry["size"]})
    assert card["resources"]["architecture"] == {"hidden": "d", "precision_bits": PRECISION[key], "state_bits": None}
    wrong, right = ("float64", "float32") if key == "k2" else ("float32", "float64")
    assert right in protocol.source and wrong not in protocol.source


@pytest.mark.parametrize("key, task", PROTOCOL_CASES)
def test_the_stage_m_request_is_the_experiment_the_protocol_was_confirmed_on(key, task):
    """The statement the host freezes from the card, sent through m_math.library_request with the frozen architecture
    contract, is the experiment of the recorded confirmation: same source hash, task, size, m, seeds, threshold,
    layout, widths at the size and contract (the recorded request wrote the widths as numbers)."""
    name = PROTOCOLS[(key, task)][0]
    card = card_of(key)
    protocol = cr.learning_protocols(card)[task]
    spec, problems = cr.normalize_statement(card, P4_SLOT_OF_TASK[task], "learnability", {"task": task})
    assert problems == [] and cr.protocol_applies(protocol, spec)
    contract, problems = cr.normalize_architecture(card, None)
    assert problems == []
    request = m_math.library_request({"form": "learnability", "spec": spec}, protocol.source, contract)
    recorded = json.loads((cf.P4_PROTOCOLS / f"{name}-confirmation.json").read_text())
    assert recorded["sha256"] == protocol.sha256 == hashlib.sha256(request["source"].encode()).hexdigest()
    assert cf.experiment_key(request["spec"], request["architecture"]) == cf.experiment_key(
        recorded["spec"], recorded["architecture_contract"])
    assert recorded["result"]["status"] == RECORDED_STATUS[(key, task)]
    assert recorded["result"]["architecture"]["violations"] == []


def test_a_protocol_applies_only_to_the_cards_learning_case():
    card = k2()
    protocol = cr.learning_protocols(card)["recall"]
    spec, _ = cr.normalize_statement(card, "P4.recall", "learnability", {"task": "recall"})
    assert cr.protocol_applies(protocol, spec)
    for change in ({"size": 3}, {"examples": 1000}, {"task": "copy"},
                   {"widths": dict(spec["widths"], d="64")}, {"widths": dict(spec["widths"], rounds="2")}):
        assert not cr.protocol_applies(protocol, dict(spec, **change)), change


# ----- 2. a hash mismatch fails fast -------------------------------------------------------------------------------

def _copy_fixtures(tmp_path: Path) -> Path:
    directory = tmp_path / "calibration"
    shutil.copytree(cal.FIXTURES, directory)
    return directory


def test_a_hash_mismatch_stops_the_calibration_load(tmp_path):
    directory = _copy_fixtures(tmp_path)
    card = json.loads((directory / "k2.json").read_text())
    card["resources"]["learning"]["recall"]["protocol"]["sha256"] = "0" * 64
    (directory / "k2.json").write_text(json.dumps(card))
    with pytest.raises(cr.ProtocolError, match=r"calibration card k2: .*k2-recall\.py has sha256 5905fdfb"):
        cal.load_cards(("k2",), directory)


def test_a_changed_or_missing_source_or_a_path_outside_the_repo_is_refused(tmp_path):
    root = tmp_path / "repo"
    (root / "protocols").mkdir(parents=True)
    source = (PROTOCOL_DIR / "k2-recall.py").read_bytes()
    (root / "protocols" / "k2-recall.py").write_bytes(source + b"\n")
    card = k2()
    card["resources"]["learning"]["recall"]["protocol"]["source"] = "protocols/k2-recall.py"
    card = only_protocols(card, ("recall",))
    with pytest.raises(cr.ProtocolError, match="the card states " + RECALL_SHA):
        cr.learning_protocols(card, root)
    (root / "protocols" / "k2-recall.py").write_bytes(source)
    assert cr.learning_protocols(card, root)["recall"].sha256 == RECALL_SHA
    for path, problem in (("protocols/absent.py", "does not exist"), ("../repo/protocols/k2-recall.py", "inside"),
                          (str(root / "protocols" / "k2-recall.py"), "inside")):
        card["resources"]["learning"]["recall"]["protocol"]["source"] = path
        with pytest.raises(cr.ProtocolError, match=problem):
            cr.learning_protocols(card, root)


def test_the_schema_requires_the_three_protocol_fields():
    card = k2()
    validate("theory_card", card)
    for broken in ({"sha256": "abc"}, {"source": "/abs/path.py"}, {"provenance": ""}, {"extra": 1}):
        changed = copy.deepcopy(card)
        changed["resources"]["learning"]["recall"]["protocol"].update(broken)
        with pytest.raises(SchemaViolation):
            validate("theory_card", changed)
    changed = copy.deepcopy(card)
    del changed["resources"]["learning"]["recall"]["protocol"]["provenance"]
    with pytest.raises(SchemaViolation):
        validate("theory_card", changed)


def test_stage_m_stops_on_a_hash_mismatch_before_any_call_or_job(tmp_path):
    card = k2()
    card["resources"]["learning"]["composition"]["protocol"]["sha256"] = "f" * 64
    h = Harness(tmp_path, "k2", card=card)
    with pytest.raises(StageError, match="card learning protocol refused; nothing was run: .*k2-composition.py"):
        h.m()
    assert h.llm.calls == [] and h.executor.jobs == []
    assert list(h.ledger.iter_events("math_statements_frozen")) == []


# ----- 3. a supplied protocol skips the prover and is checked by the same runner -----------------------------------

def test_a_supplied_protocol_is_checked_verbatim_by_the_same_runner_and_the_prover_is_never_asked(tmp_path):
    """K2 as it was first wired: learners for composition and recall, prover learners for copy and nesting."""
    h = Harness(tmp_path, "k2", card=only_protocols(k2(), ("composition", "recall")))
    h.m()
    record = h.frozen()
    supplied = slot_statements(record, SUPPLIED_SLOTS)
    prover_learned = slot_statements(record, ("P4.copy", "P4.nesting"))
    assert len(supplied) == 2 and len(prover_learned) == 2
    protocols = cr.learning_protocols(h.card)

    # The prover was asked for the other learnability statements and never for the supplied ones.
    shown = {item["statement_id"] for item in prover_items(h)}
    assert set(prover_learned) <= shown and not set(supplied) & shown
    assert not [e for e in h.ledger.iter_events("proof_attempt") if e["data"]["statement_id"] in supplied]

    # The same runner path: timing probe, then the full job and its replay, each with the frozen spec and contract,
    # the library caps and the supplied source byte for byte.
    contract = m_math.architecture_contract(record)

    def caps(jobs, statement) -> bool:
        """The probe under the runner's probe caps, the full job and its replay under the library caps."""
        return ([job.caps for _, job in jobs]
                == [h.ctx.math.probe_caps(statement["spec"], LIBRARY_CAPS), LIBRARY_CAPS, LIBRARY_CAPS])

    by_source: dict[str, list] = {}
    for job in h.executor.jobs:
        request = json.loads((Path(job.input_dir) / "request.json").read_text())
        if request["kind"].startswith("learnability"):
            by_source.setdefault(request["source"], []).append((request, job))
    for sid, statement in supplied.items():
        protocol = protocols[statement["spec"]["task"]]
        jobs = by_source[protocol.source]
        assert [r["kind"] for r, _ in jobs] == ["learnability_probe", "learnability", "learnability"]
        assert all(r["spec"] == statement["spec"] for r, _ in jobs)
        assert jobs[1][0]["architecture"] == contract and caps(jobs, statement)
        [check] = check_artifacts(h, sid)
        assert check["text"] == protocol.source and check["purpose"] == "card_protocol"
        assert check["learner"] == protocol.record() and check["learner"]["sha256"] == protocol.sha256
        assert check["probe"]["status"] == "ok" and check["status"] == "discharged" and check["decided"]
        assert check["details"]["architecture"]["violations"] == []
    for sid, statement in prover_learned.items():
        jobs = by_source[cf.LEARNERS[statement["spec"]["task"]]]
        assert [r["kind"] for r, _ in jobs] == ["learnability_probe", "learnability", "learnability"]
        assert jobs[1][0]["architecture"] == contract and caps(jobs, statement)
        assert all("learner" not in check for check in check_artifacts(h, sid))
    assert len(h.executor.measured) == 4

    # The ledger records the learner's origin and hash, before the check and with it.
    events = [e["data"] for e in h.ledger.iter_events("learning_protocol")]
    assert {e["statement_id"]: e["sha256"] for e in events} == {
        sid: protocols[s["spec"]["task"]].sha256 for sid, s in supplied.items()}
    assert all(e["origin"] == "card" for e in events)
    checked = {e["data"]["statement_id"]: e["data"].get("learner") for e in h.ledger.iter_events("math_check")}
    assert {sid: checked[sid]["sha256"] for sid in supplied} == {
        sid: protocols[s["spec"]["task"]].sha256 for sid, s in supplied.items()}
    verdict = h.verdict("M")
    assert {sid: verdict["checks"][sid]["learner"]["origin"] for sid in supplied} == {sid: "card" for sid in supplied}
    assert not any("learner" in verdict["checks"][sid] for sid in prover_learned)
    for slot in ("P4.copy", "P4.recall", "P4.nesting", "P4.composition"):
        assert h.slot("M", slot)["status"] == "observed", slot
    validate("math_verdict", verdict)


@pytest.mark.parametrize("key", ["k1", "k2"])
def test_every_p4_slot_of_a_calibration_card_runs_its_supplied_learner_and_the_prover_writes_none(tmp_path, key):
    h = Harness(tmp_path, key)
    h.m()
    supplied = slot_statements(h.frozen(), tuple(profile.P4_SLOTS))
    protocols = cr.learning_protocols(h.card)
    assert sorted(s["spec"]["task"] for s in supplied.values()) == sorted(protocols) == [
        "composition", "copy", "dyck", "recall"]
    assert not set(supplied) & {item["statement_id"] for item in prover_items(h)}
    assert not [e for e in h.ledger.iter_events("proof_attempt") if e["data"]["statement_id"] in supplied]
    assert len(h.executor.measured) == 8
    verdict = h.verdict("M")
    for sid, statement in supplied.items():
        task = statement["spec"]["task"]
        protocol = protocols[task]
        [check] = check_artifacts(h, sid)
        assert check["text"] == protocol.source and check["purpose"] == "card_protocol"
        assert check["learner"] == protocol.record() and check["learner"]["sha256"] == PROTOCOLS[(key, task)][1]
        assert check["details"]["architecture"]["violations"] == []
        assert check["status"] == RECORDED_STATUS[(key, task)]
        assert verdict["checks"][sid]["learner"]["origin"] == "card"
    # K1 composition is the measured learner that does not reliably learn composition (decision P4K): its slot fails.
    for slot, tasks in profile.P4_SLOTS.items():
        expected = "failed" if (key, tasks[0]) == ("k1", "composition") else "observed"
        assert h.slot("M", slot)["status"] == expected, slot
    validate("math_verdict", verdict)


def test_the_statements_still_freeze_from_the_card(tmp_path):
    """The frozen P4 statements are what the card states (recall at d = 128 and m = 2500, composition at d = 64 and
    m = 3000), with the protocol seeds and threshold, and the freeze event comes before any check."""
    h = Harness(tmp_path, "k2")
    h.m()
    supplied = {s["spec"]["task"]: s["spec"] for s in slot_statements(h.frozen(), SUPPLIED_SLOTS).values()}
    assert (supplied["recall"]["widths"]["d"], supplied["recall"]["examples"]) == ("128", 2500)
    assert (supplied["composition"]["widths"]["d"], supplied["composition"]["examples"]) == ("64", 3000)
    assert all(spec["seeds"] == list(profile.PROTOCOL["p4_seeds"]) and spec["threshold"] == profile.P4_THRESHOLD
               for spec in supplied.values())
    types = [e["type"] for e in h.ledger.iter_events() if e["type"] in ("math_statements_frozen", "learning_protocol",
                                                                        "math_check")]
    assert types[0] == "math_statements_frozen" and "learning_protocol" in types


def test_a_failed_protocol_check_stays_failed_and_the_prover_is_not_asked(tmp_path, monkeypatch):
    """No retries: the supplied learner's check is the slot's result even when it misses the threshold."""
    measured = cf.measured_protocols()
    for key, result in measured.items():
        if result["task"] == "recall":
            row = dict(result["results"][1], accuracy=0.95, passed=False, reason="below_threshold")
            measured[key] = dict(result, status="violated", results=[result["results"][0], row,
                                                                     result["results"][2]],
                                 counterexample={"task": "recall", "seed": 1, "reason": "below_threshold"})
    monkeypatch.setattr(cf, "measured_protocols", lambda: measured)
    h = Harness(tmp_path, "k2")
    h.m()
    recall = slot_statements(h.frozen(), ("P4.recall",))
    [sid] = recall
    assert [c["status"] for c in check_artifacts(h, sid)] == ["violated"]
    assert sid not in {item["statement_id"] for item in prover_items(h)}
    assert h.slot("M", "P4.recall")["status"] == "failed"


# ----- 4. no supplied protocol: the prover writes the learner as before ---------------------------------------------

@pytest.mark.parametrize("key", ["k1", "k2"])
def test_without_a_protocol_the_prover_writes_every_learner(tmp_path, key):
    card = without_protocols(cal.load_cards((key,))[key])
    h = Harness(tmp_path, key, card=card)
    h.m()
    learned = slot_statements(h.frozen(), profile.P4_SLOTS)
    assert len(learned) == 4
    assert set(learned) <= {item["statement_id"] for item in prover_items(h)}
    assert list(h.ledger.iter_events("learning_protocol")) == [] and h.executor.measured == []
    for sid, statement in learned.items():
        [check] = check_artifacts(h, sid)
        assert check["purpose"] == "prove" and "learner" not in check
        assert check["text"] == cf.LEARNERS[statement["spec"]["task"]]
    fixer = next(c for c in h.llm.calls if c["prompt"] == "statement_fixer")
    assert "CARD LEARNING PROTOCOLS (decision P4P). The card supplies" not in fixer["user"]


def test_a_prover_message_without_protocols_is_unchanged_by_the_feature(tmp_path):
    """The prover's message for a learnability statement is the same function of the card and record as before: the
    protocol text appears only for the fixer and the challenger."""
    h = Harness(tmp_path, "k1", card=without_protocols(card_of("k1")))
    h.m()
    prover = [c["user"] for c in h.llm.calls if c["prompt"] == "prover"]
    assert prover and not any("CARD LEARNING PROTOCOLS" in user or "decision P4P" in user for user in prover)


# ----- 5. the fixer and the challenger cannot override a supplied protocol ------------------------------------------

def test_the_fixer_is_told_the_protocol_is_used_verbatim(tmp_path):
    h = Harness(tmp_path, "k2")
    h.m()
    fixer = next(c for c in h.llm.calls if c["prompt"] == "statement_fixer")["user"]
    assert "CARD LEARNING PROTOCOLS (decision P4P)" in fixer and "statement of composition, copy, dyck, recall" in fixer
    assert "A card-supplied protocol is used verbatim" in fixer
    assert "write no definition that sets or changes its training recipe" in fixer
    assert '"sha256": "' + RECALL_SHA + '"' in fixer
    # Additive prompt text: the rule sits next to the old text, which is unchanged.
    prompt = load_prompt("statement_fixer")
    assert "unless the card itself states it: the prover chooses it. A card-supplied learning protocol" in prompt
    section = load_section("profile_obligations")
    assert "chosen by the prover within the P4 budget stated below" in section
    assert "the card's learner is used verbatim" in section
    challenger = load_section("challenge_profile")
    assert "A P4 check whose `learner` has origin card ran the card's own supplied learning protocol" in challenger


def test_the_fixer_cannot_change_a_protocol_tasks_width_or_m():
    card = k2()
    _, problems = cr.normalize_statement(card, "P4.recall", "learnability",
                                         {"task": "recall", "widths": {"d": "64"}, "examples": 1000})
    assert any("widths.d must be \"128\"" in p for p in problems)
    assert any("examples is a card value, 2500" in p for p in problems)


def test_the_challenger_sees_the_origin_and_no_variant_replaces_a_card_learner(tmp_path):
    """K2 as it was first wired: copy has a prover learner, recall a card learner."""
    h = Harness(tmp_path, "k2", card=only_protocols(k2(), ("composition", "recall")))
    h.m()
    record = h.frozen()
    verdict = h.verdict("M")
    checks = {sid: dict(c) for sid, c in verdict["checks"].items()}
    [recall] = slot_statements(record, ("P4.recall",))
    [copy_sid] = slot_statements(record, ("P4.copy",))
    view = c_challenge._check_view(checks[recall])
    assert view["learner"]["origin"] == "card" and view["learner"]["sha256"] == RECALL_SHA
    assert "learner" not in c_challenge._check_view(checks[copy_sid])
    user = c_challenge.challenger_user(h.card, record, verdict, [], {}, {})
    assert '"origin": "card"' in user and "A P4 check whose `learner` has origin card" in user

    oids = [m_math.obligation_of(record, sid)["id"] for sid in (recall, copy_sid)]
    deviation = "augmentation"
    finding = mf.finding("the recipe relabels tokens", used_by=oids,
                         evidence={"source": "card", "ref": None, "quote": h.card["mechanism"]["learning"]
                                   ["description"][:40]},
                         targets=[mf.target(recall, deviation, [(deviation, "augment")]),
                                  mf.target(copy_sid, "no such text", [("no such text", "x")])])
    item = dict(finding, id="C-test", used_by=oids, verification={"evidence": "confirmed by the host"})
    before = len(list(h.ledger.iter_events("math_check")))
    out = asyncio.run(c_challenge.settle_materiality(h.ctx, h.card["id"], finding, item, record, checks))
    by_statement = {piece["statement_id"]: piece for piece in out}
    assert by_statement[recall]["disposition"] == "out_of_scope" and by_statement[recall]["materiality"] is None
    assert "supplied learning protocol (sha256 " + RECALL_SHA in by_statement[recall]["reason"]
    # A prover learner is still settled by the materiality rules (here: the deviation is not in its source).
    assert by_statement[copy_sid]["disposition"] == "rejected"
    assert len(list(h.ledger.iter_events("math_check"))) == before


@pytest.mark.parametrize("key", ["k1", "k2"])
def test_every_check_shows_the_challenger_its_card_learner(tmp_path, key):
    h = Harness(tmp_path, key)
    h.m()
    record = h.frozen()
    verdict = h.verdict("M")
    for sid, statement in slot_statements(record, tuple(profile.P4_SLOTS)).items():
        view = c_challenge._check_view(dict(verdict["checks"][sid]))
        assert view["learner"]["origin"] == "card"
        assert view["learner"]["sha256"] == PROTOCOLS[(key, statement["spec"]["task"])][1]


@pytest.mark.parametrize("key", ["k1", "k2"])
def test_authors_keep_all_four_protocols_of_a_card_and_lose_one_on_a_changed_case(key):
    card = card_of(key)
    own = copy.deepcopy(card["resources"])
    kept, dropped = cr.keep_card_protocols(card, own)
    assert kept == card["resources"] and dropped == []
    answer = copy.deepcopy(own)
    answer["learning"]["copy"]["examples"] += 100
    kept, dropped = cr.keep_card_protocols(card, answer)
    assert dropped == ["copy"] and "protocol" not in kept["learning"]["copy"]
    assert sorted(cr.learning_protocols({"resources": kept})) == ["composition", "dyck", "recall"]


def test_explorers_cannot_state_a_protocol_and_authors_keep_only_the_cards_own():
    learning = draft_schema()["$defs"]["learning_resources"]["properties"]
    assert "protocol" not in learning and "learning_protocol" not in draft_schema()["$defs"]
    from hybrid.research.schemas import load_schema
    assert "protocol" in load_schema("theory_card")["$defs"]["learning_resources"]["properties"]
    card = only_protocols(k2(), ("composition", "recall"))
    own = copy.deepcopy(card["resources"])
    # An author who restates the card's resources keeps its protocols unchanged.
    kept, dropped = cr.keep_card_protocols(card, own)
    assert kept == card["resources"] and dropped == []
    # An author cannot add a protocol, change one, or keep one on a changed learning case.
    answer = copy.deepcopy(own)
    answer["learning"]["copy"]["protocol"] = dict(own["learning"]["recall"]["protocol"])
    answer["learning"]["composition"]["protocol"]["sha256"] = "1" * 64
    answer["learning"]["recall"]["d"] = "64"
    kept, dropped = cr.keep_card_protocols(card, answer)
    assert sorted(dropped) == ["composition", "copy", "recall"]
    assert "protocol" not in kept["learning"]["copy"] and "protocol" not in kept["learning"]["recall"]
    assert kept["learning"]["composition"]["protocol"] == own["learning"]["composition"]["protocol"]
    # A card that states none gets none from its author.
    kept, dropped = cr.keep_card_protocols(without_protocols(card), own)
    assert sorted(dropped) == ["composition", "recall"] and cr.learning_protocols({"resources": kept}) == {}


# ----- 6. replay of the K2 recall protocol through the stage M learnability path ------------------------------------

SLOW = os.environ.get("BIGBRANE_RUN_SLOW") == "1"


@pytest.mark.slow
@pytest.mark.skipif(not SLOW, reason="trains the K2 recall learner for minutes; set BIGBRANE_RUN_SLOW=1")
def test_the_k2_recall_protocol_replays_through_the_stage_m_learnability_path(tmp_path, monkeypatch):
    """The frozen P4.recall statement as stage M freezes it from the card (production test count 500), on check seed
    0 only, sent through m_math.library_request and the stage's MathRunner (timing probe, admission, full job and its
    replay, architecture survey) with the card's learner. Nothing is recorded or scripted: the learner trains."""
    monkeypatch.setitem(profile.PROTOCOL, "p4_test_count", 500)
    card = k2()
    protocol = cr.learning_protocols(card)["recall"]
    spec, problems = cr.normalize_statement(card, "P4.recall", "learnability", {"task": "recall"})
    assert problems == [] and spec["test_count"] == 500 and cr.protocol_applies(protocol, spec)
    contract, _ = cr.normalize_architecture(card, None)
    request = m_math.library_request({"form": "learnability", "spec": dict(spec, seeds=[0])}, protocol.source, contract)
    executor = cf.FastExecutor(tmp_path / "exec")
    executor._measured = {}
    runner = MathRunner(executor, mf.IMAGE, tmp_path / "work")
    record = runner.check_library(request)
    assert executor.measured == []
    assert record["probe"]["status"] == "ok" and record["resource"] is None
    assert record["probe"]["projected_cpu_s"] <= learnability.PROBE_FIT * LIBRARY_CAPS.cpu_s
    printed = record["runs"][0]["printed"]
    (tmp_path / "replay-result.json").write_text(json.dumps({"probe": record["probe"], "printed": printed},
                                                            default=str, indent=1))
    assert record["status"] == "discharged" and record["decided"] and record["replayed"], record["evidence"]
    [row] = printed["results"]
    assert row["seed"] == 0 and row["accuracy"] >= 0.99 and row["passed"]
    assert printed["architecture"]["violations"] == []
