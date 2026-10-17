"""FIX-16: card completeness. A frozen statement may contain no value that is neither in the card nor set by the
protocol (results/phase4-20261013T073748Z/FIX-16.md).

Cause: calibration card K2 never stated its store width, so the statement fixer chose one per run; at d = 64 the
P2.recall witness shares cells between keys and scored about 0.757 in every such run, while d = 128, 4K+8 and 8K+32
passed (STATUS.md, "K2 recall: cause found"). The audit fix16/audit-varying-frozen-values.txt found the same freedom
in widths, rounds, positions, hidden width, state_bits, state_words, count, the P1 grid and the threshold."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from hybrid.math import constructions, profile
from hybrid.math.runner import MathRunner
from hybrid.research import calibration as cal
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.outcomes import check_kill_code
from hybrid.research.schemas import validate
from hybrid.research.stages import card_resources as cr
from hybrid.research.stages import m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import Budget, CodeRunner, StageContext
from hybrid.research.store import Store
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM

CARDS = cal.load_cards()
K2 = CARDS["k2"]
RUN_K2 = ROOT / "runs" / "calibration-math-20261015T134559Z" / "k2-run2"
RECORDED = RUN_K2 / "artifacts" / "math_statements" / "card-ca11b002-v1-43de31.json"
RECORDED_SHA256 = "a4fed3e1006bbf4f755afda46cf991bf78eeb1e41ed5b0cbecefde4c95acf3ee"
# O7.a of that record, P2.recall, as the fixer froze it: d = 64 (128 state words), the width at which the witness
# mapped keys k % 63 and scored 0.757 at K = 64.
RECORDED_RECALL = {"count": 200, "params": {"V": 16}, "sizes": [8, 32, 64], "state_words": "128", "task": "recall",
                   "threshold": 0.99, "widths": {"d": "64", "positions": "3*K+1", "rounds": "1"}}


@pytest.fixture
def production(monkeypatch):
    """The production protocol (the research conftest sets small counts for the scripted runs)."""
    monkeypatch.undo()
    return profile.PROTOCOL


def claim_ids(card):
    from hybrid.research.mathrecords import formal_claims
    from hybrid.research.stages.falsify import claim_items
    return [c["id"] for c in formal_claims(claim_items(card))] + list(profile.SLOTS)


def texts(card):
    from hybrid.research.mathrecords import formal_claims
    from hybrid.research.stages.falsify import claim_items
    claims = formal_claims(claim_items(card)) + profile.profile_claims()
    return {c["id"]: c["text"] for c in claims}


def freeze(card, answer):
    return mc.freeze(answer, card, claim_ids(card), 60, texts(card))


def problems(card, answer):
    return mc.fixer_problems(answer, card, claim_ids(card), 60, texts(card))


def statement(record, claim_id, task=None):
    return next(s for o in record["obligations"] if o["claim_id"] == claim_id for s in o["statements"]
                if task is None or s["spec"].get("task") == task)


def answer_for(key):
    """The scripted fixer's answer for a K card under the production protocol (it leaves ns out)."""
    answer = cf.fixer_output(key, CARDS[key])
    for obligation in answer["obligations"]:
        for item in obligation["statements"]:
            if item["form"] == "cost":
                item["cost"].pop("ns", None)
    return answer


def set_spec(answer, claim_id, form, change, task=None):
    for obligation in answer["obligations"]:
        if obligation["claim_id"] == claim_id:
            for item in obligation["statements"]:
                if task is None or item[form].get("task") == task:
                    change(item[form])
                    return answer
    raise AssertionError(f"no {form} statement for {claim_id}")


# ----- the classification and the protocol -----------------------------------------------------------------------

def test_the_production_protocol_values(production):
    assert production == {"p2_count": 200, "p4_test_count": 500, "p4_seeds": (0, 1, 2), "p1_ns": None,
                          "cost_grid": (8, 32), "conformance_n": 256, "conformance_tolerance": 0.0}
    assert profile.P2_THRESHOLD == profile.P4_THRESHOLD == 0.99
    text = profile.protocol_text()
    for needle in ("count 200", "test_count 500", "threshold 0.99", "[8, 32]", "seeds [0, 1, 2]",
                   "leave ns out (the plan's lengths 64, 128, 256, 512, 1024, 2048, 4096)"):
        assert needle in text, needle
    assert profile.protocol_positions("composition") == "5*T+3"
    assert profile.protocol_positions("dyck") is None


def test_every_frozen_field_is_classified_as_protocol_card_derived_or_claim():
    kinds = {kind for fields in profile.FIELD_KINDS.values() for kind in fields.values()}
    assert kinds == {"protocol", "card", "derived", "claim"}
    assert profile.FIELD_KINDS["construction"]["count"] == "protocol"
    assert profile.FIELD_KINDS["construction"]["threshold"] == "protocol"
    assert profile.FIELD_KINDS["construction"]["widths.d"] == "card"
    assert profile.FIELD_KINDS["architecture"] == {"hidden": "card", "precision_bits": "card", "state_bits": "card"}
    assert profile.FIELD_KINDS["conservation"]["entry state_bits"] == "derived"


# ----- the completeness gate ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", cal.CARD_KEYS)
def test_every_calibration_card_passes_the_completeness_gate(key):
    card = CARDS[key]
    assert cr.missing(card) == []
    validate("theory_card", card)
    assert cal.anonymization_problems(card) == []


def test_k2_recall_width_covers_the_key_universe_at_every_required_k():
    """The stated width gives every possible key its own store entry: 4K+8 >= n_keys = max(2K, 16)."""
    formula = K2["resources"]["tasks"]["recall"]["d"]
    for K in profile.REQUIRED_SIZES["recall"]:
        assert cr.evaluate(formula, {"K": K}) >= constructions.layout("recall", K)["n_keys"]
    assert cr.evaluate(formula, {"K": 64}) == 264


def incomplete_k2():
    card = copy.deepcopy(K2)
    del card["resources"]["tasks"]["recall"]["d"]
    del card["resources"]["learning"]["copy"]["examples"]
    card["resources"]["architecture"].pop("precision_bits")
    return card


def test_a_card_missing_resources_lists_each_missing_item():
    assert cr.missing(incomplete_k2()) == [
        "resources.architecture.precision_bits (one of [16, 32, 64])",
        "resources.tasks.recall.d (a formula in K)",
        "resources.learning.copy.examples (the example count m the card states)"]
    bare = {key: value for key, value in K2.items() if key != "resources"}
    found = cr.missing(bare)
    assert found[0] == "resources: the card states no resources section"
    assert "resources.tasks.binding.state_words (a formula in hops)" in found
    assert "resources.cost.sizes (the size parameters other than n that the P1 cost formula uses)" in found


def test_a_stated_architecture_the_contract_refuses_is_missing():
    # A carried-state budget above the architecture check's maximum could never be frozen: every fixer answer would
    # fail on it, so the gate asks for it again.
    card = copy.deepcopy(K2)
    card["resources"]["architecture"]["state_bits"] = 64
    assert cr.missing(card) == ["resources.architecture.state_bits must be between 0 and 32"]
    card["resources"]["architecture"]["state_bits"] = 32
    assert cr.missing(card) == []


def test_an_unstated_shape_symbol_is_missing_until_the_card_states_it():
    card = copy.deepcopy(CARDS["k3"])
    card["mechanism"]["state"][0]["shape"] = ["m"]
    found = cr.missing(card)
    assert "resources.tasks.copy.symbols.m (a formula in L)" in found
    assert "resources.learning.dyck.symbols.m (a formula in depth)" in found
    card["resources"]["dimensions"]["m"] = "4"
    assert cr.missing(card) == []


def test_calibration_refuses_an_incomplete_fixture_before_any_call(tmp_path):
    with pytest.raises(ValueError, match=r"calibration fixture error \(card_incomplete\): k2 lacks "
                                         r"resources.architecture.precision_bits"):
        cal.check_complete({"k2": incomplete_k2()})
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    with pytest.raises(ValueError, match="card_incomplete"):
        asyncio.run(cal.calibrate(cards={"k2": incomplete_k2()}, runs=1, run_root=tmp_path / "r",
                                  out_dir=tmp_path / "o", cfg=cfg, reason="x", deps_factory=None))
    assert not (tmp_path / "r").exists()


def test_stage_m_ends_an_incomplete_card_with_card_incomplete_and_no_fixer_call(tmp_path):
    card = incomplete_k2()
    cfg = cal.calibration_config(load_run_config(ROOT / "hybrid" / "config" / "run.toml"))
    calls = []

    def build(ledger):
        executor = cf.FastExecutor(ledger.run_dir / "exec")
        llm = FakeLLM(ledger.run_dir, ledger, cf.script("k2", K2))
        calls.append(llm)
        return cal.Deps(llm, executor, "image", MathRunner(executor, mf.IMAGE, ledger.run_dir))

    outcome = asyncio.run(cal.run_card(key="k2", card=card, run_index=1, run_dir=tmp_path / "k2-run1", cfg=cfg,
                                       reason="FIX-16 test", deps_factory=build,
                                       target_rule={"max_cost_exponent": 1.5, "state_growth": "sublinear"}))
    assert outcome["stopped"].startswith("card_incomplete at M (fixture error): resources.architecture.precision_bits")
    assert outcome["m_kill_codes"] == ["card_incomplete"]
    assert outcome["stages"]["M"]["verdict"] == "undetermined" and "C" not in outcome["stages"]
    assert calls[0].calls == []
    verdict = json.loads(next((tmp_path / "k2-run1" / "artifacts" / "math_verdict").iterdir()).read_text())
    assert verdict["missing_resources"] == cr.missing(card)
    assert check_kill_code("card_incomplete") == "card_incomplete"


class FakeBook:
    def __init__(self, grants: int):
        self.grants = grants
        self.rounds: list[dict] = []

    def after_round(self, *, card_id, stage, loop, round_id, failures):
        from hybrid.research import lineage
        self.rounds.append({"stage": stage, "loop": loop, "round_id": round_id, "failing": sorted(failures)})
        if len(self.rounds) > self.grants:
            return lineage.LoopStop(loop, "budget", f"{self.grants} of {self.grants} revision(s) used")
        return None


def stage_context(tmp_path, card, script, *, book=None, cfg=None):
    ledger = Ledger(tmp_path / "run")
    store = Store(tmp_path / "run", ledger)
    store.put_card(card)
    ledger.append("card_created", "orchestrator", {"card_id": card["id"], "role": "explorer", "model": "fake-gpt",
                                                   "op": "new", "parents": []})
    cfg = cfg or cal.calibration_config(load_run_config(ROOT / "hybrid" / "config" / "run.toml"))
    b = cfg["budgets"]
    executor = cf.FastExecutor(tmp_path / "exec")
    llm = FakeLLM(tmp_path / "run", ledger, script)
    ctx = StageContext(run_id="run", run_dir=tmp_path / "run", ledger=ledger, store=store, llm=llm,
                       runner=CodeRunner(executor, tmp_path / "run", store, "image", None), bench=None, cfg=cfg,
                       budget=Budget(b["max_cards"], b["max_calls"], b["max_tokens"], b["max_wall_s"]), lenses={},
                       math=MathRunner(executor, mf.IMAGE, tmp_path / "run"), book=book)
    return ctx, llm


def resources_reply(index, user):
    return {"resources": copy.deepcopy(K2["resources"]), "why": "the store width grows with the amount to hold"}


def test_in_a_normal_run_the_author_is_asked_for_the_missing_resources_under_the_revision_budget(tmp_path):
    book = FakeBook(grants=1)
    # The test is about the revision round, not the learners: a card without protocols keeps the author's resources
    # as stated (a protocol stays only on an unchanged learning case, and this card's copy case is incomplete).
    incomplete = cf.without_protocols(incomplete_k2())
    ctx, llm = stage_context(tmp_path, incomplete, {"card_resources": resources_reply}, book=book)
    card = asyncio.run(m_math.complete_card(ctx, K2["id"], ctx.card(K2["id"])))
    assert card is not None and cr.missing(card) == [] and card["resources"] == cf.without_protocols(K2)["resources"]
    assert [c["prompt"] for c in llm.calls] == ["card_resources"]
    assert "resources.tasks.recall.d (a formula in K)" in llm.calls[0]["user"]
    assert book.rounds == [{"stage": "M", "loop": "math_revision", "round_id": "resources-v1",
                            "failing": sorted(cr.missing(incomplete))}]
    assert ctx.card(K2["id"])["version"] == 2
    asked = ctx.card_events(K2["id"], "card_resources_requested")
    assert asked[0]["missing_after"] == [] and asked[0]["stage"] == "M"


def test_an_author_who_leaves_items_unstated_ends_the_card_at_the_revision_cap(tmp_path):
    book = FakeBook(grants=1)
    partial = copy.deepcopy(incomplete_k2()["resources"])

    def reply(index, user):
        return {"resources": copy.deepcopy(partial), "why": "unchanged"}

    ctx, llm = stage_context(tmp_path, incomplete_k2(), {"card_resources": reply}, book=book)
    assert asyncio.run(m_math.complete_card(ctx, K2["id"], ctx.card(K2["id"]))) is None
    verdict = ctx.card(K2["id"])["status"]["verdicts"]["M"]
    assert verdict["verdict"] == "undetermined"
    assert verdict["reasons"][0] == "math gate (M): undetermined; kill codes card_incomplete, budget:math_revision"
    assert any(r.startswith("gate: card_incomplete: the card's resources leave 3 item(s) unstated")
               for r in verdict["reasons"])
    assert len(book.rounds) == 2


def test_stage_r_asks_for_missing_resources_once_after_the_dialogue(tmp_path):
    from hybrid.research.stages import r_refine
    cfg = cal.calibration_config(load_run_config(ROOT / "hybrid" / "config" / "run.toml"))
    ctx, llm = stage_context(tmp_path, incomplete_k2(), {"card_resources": resources_reply}, cfg=cfg)
    calls = []
    lines = asyncio.run(r_refine.resources_round(ctx, K2["id"], calls))
    assert lines == ["card resources: 3 item(s) were unstated; the author was asked once; now complete"]
    assert len(calls) == 1 and cr.missing(ctx.card(K2["id"])) == []
    assert asyncio.run(r_refine.resources_round(ctx, K2["id"], calls)) == ["card resources: complete"]
    cfg["math"]["profile"] = False
    assert asyncio.run(r_refine.resources_round(ctx, K2["id"], calls)) == []


def test_a_t5_override_runs_the_sealed_card_with_the_resources_stated_in_r(tmp_path):
    # A sealed planted card states no resources; the main line's stage R asked its author for them. The T5 override
    # (M and C, no stage R) runs the card as it entered with those resources, else M would end it card_incomplete.
    from hybrid.research import planted as pl
    sealed = {key: value for key, value in K2.items() if key != "resources"}
    card_id = "card-5eed0001"
    run_dir = (tmp_path / "main").resolve()
    store = Store(run_dir, Ledger(run_dir))
    asked_never, version = pl.override_card(sealed, card_id, store)
    assert version is None and "resources" not in asked_never and cr.missing(asked_never)
    entered = pl.entering_card(sealed, card_id)
    store.put_card(entered)
    store.put_card(dict(copy.deepcopy(entered), version=2, resources=copy.deepcopy(K2["resources"])))
    store.put_card(dict(copy.deepcopy(entered), version=3, resources=copy.deepcopy(K2["resources"]),
                        title=entered["title"] + " (revised)"))
    card, version = pl.override_card(sealed, card_id, store)
    assert version == 3 and card["resources"] == K2["resources"] and cr.missing(card) == []
    assert {key: value for key, value in card.items() if key != "resources"} == entered


# ----- host verification at freeze ----------------------------------------------------------------------------------

def test_a_fixer_that_copies_the_card_freezes_the_cards_values(production):
    answer = answer_for("k2")
    assert problems(K2, answer) == []
    record = freeze(K2, answer)
    recall = statement(record, "P2.recall")["spec"]
    assert recall["widths"] == {"d": "4*K+8", "rounds": "1", "positions": "3*K+1"}
    assert recall["state_words"] == "8*K+16" and recall["count"] == 200 and recall["threshold"] == 0.99
    assert record["profile"]["architecture"] == {"hidden": "d", "precision_bits": 32}
    p4 = statement(record, "P4.copy")["spec"]
    assert (p4["size"], p4["examples"], p4["seeds"], p4["test_count"]) == (4, 1500, [0, 1, 2], 500)
    p1 = statement(record, "P1")["spec"]
    assert p1["sizes"] == {"d": [8, 32], "V": [8, 32]} and "ns" not in p1
    assert p1["widths"] == {"d": "d", "rounds": "1", "positions": "n"}
    p5 = statement(record, "P5")["spec"]
    assert [e["state_bits"] for e in p5["recall"]] == [(8 * K + 16) * 32 for K in (8, 32, 64)]
    assert record["resources"] == K2["resources"]
    validate("math_statement", dict(record, card_id=K2["id"], card_version=1, fixer_calls=["call-000000000000"],
                                    fixer_model="fake", claims=[], first_problems=[]))


@pytest.mark.parametrize("change,needle", [
    (lambda s: s["widths"].update(d="64"), 'widths.d must be "4*K+8"'),
    (lambda s: s.update(state_words="128"), 'state_words must be "8*K+16"'),
    (lambda s: s["widths"].update(rounds="2"), 'widths.rounds must be "1"'),
])
def test_a_fixer_output_that_changes_a_card_value_is_refused(production, change, needle):
    answer = answer_for("k2")
    set_spec(answer, "P2.recall", "construction", lambda s: (s.setdefault("widths", {"d": "4*K+8", "rounds": "1"}),
                                                             change(s)))
    found = problems(K2, answer)
    assert any(needle in p for p in found), found
    record = freeze(K2, answer)
    assert not any(o["claim_id"] == "P2.recall" for o in record["obligations"])
    dropped = next(d for d in record["dropped"] if d.get("statement_id", "").startswith("O3"))
    assert any(needle in p for p in dropped["problems"])


def test_formulas_equal_after_simplification_are_the_cards_value(production):
    answer = answer_for("k2")
    set_spec(answer, "P2.recall", "construction",
             lambda s: s.update(widths={"d": "4*(K+2)", "rounds": "1"}, state_words="2*(4*K+8)"))
    assert problems(K2, answer) == []
    assert statement(freeze(K2, answer), "P2.recall")["spec"]["widths"]["d"] == "4*K+8"


@pytest.mark.parametrize("claim_id,form,change,needle", [
    ("P2.recall", "construction", lambda s: s.update(count=500), "count is a protocol value, 200"),
    ("P2.recall", "construction", lambda s: s.update(threshold=1), "threshold is a protocol value, 0.99"),
    ("P2.recall", "construction", lambda s: s.update(sizes=[8, 16, 32, 64]), "sizes are protocol values"),
    ("P2.recall", "construction", lambda s: s.update(params={"V": 8}), "params are protocol values"),
    ("P2.nesting", "construction", lambda s: s.update(widths={"d": "4*depth+8", "rounds": "1", "positions": "4*depth"}),
     "widths.positions is a protocol value that is left out"),
    ("P4.copy", "learnability", lambda s: s.update(test_count=200), "test_count is a protocol value, 500"),
    ("P4.copy", "learnability", lambda s: s.update(examples=3000), "examples is a card value, 1500"),
    ("P4.copy", "learnability", lambda s: s.update(seeds=[1, 2, 3]), "seeds are protocol values"),
    ("P1", "cost", lambda s: s.update(sizes={"d": [32, 64], "V": [5, 33]}), "sizes must be"),
    ("P1", "cost", lambda s: s.update(ns=[64, 128, 256, 512]), "ns is a protocol value"),
])
def test_a_fixer_output_that_changes_a_protocol_value_is_refused(production, claim_id, form, change, needle):
    answer = answer_for("k2")
    answer = json.loads(json.dumps(answer))
    if form == "cost":
        set_spec(answer, claim_id, form, lambda s: s.pop("ns", None))
    set_spec(answer, claim_id, form, change)
    found = problems(K2, answer)
    assert any(needle in p for p in found), found
    record = freeze(K2, answer)
    assert not any(o["claim_id"] == claim_id for o in record["obligations"])


def test_an_architecture_that_differs_from_the_card_is_refused_and_the_cards_is_frozen(production):
    answer = answer_for("k2")
    answer["architecture"] = {"hidden": "2*d", "precision_bits": 64}
    found = problems(K2, answer)
    assert 'architecture.hidden must be "d" (the card\'s resources); the answer gives "2*d"' in found
    assert "architecture.precision_bits must be 32 (the card's resources); the answer gives 64" in found
    record = freeze(K2, answer)
    assert record["profile"]["architecture"] == {"hidden": "d", "precision_bits": 32}
    assert any("architecture" in d for d in record["dropped"])


def test_a_p5_entry_with_invented_state_bits_is_refused(production):
    answer = answer_for("k2")
    set_spec(answer, "P5", "conservation", lambda s: s["recall"][2].update(state_bits=4096))
    found = problems(K2, answer)
    assert any("state_bits is derived from the card's resources, 16896; the entry gives 4096" in p for p in found)


def test_protocol_values_are_identical_across_two_freezes_of_the_same_card(production):
    """One fixer leaves every protocol value out, another writes them all: the frozen values are the same."""
    silent = answer_for("k1")
    explicit = json.loads(json.dumps(silent))
    for obligation in explicit["obligations"]:
        for item in obligation["statements"]:
            spec = item.get(item["form"])
            if item["form"] == "construction":
                spec.update(count=200, threshold=0.99, params=dict(constructions.DEFAULTS[spec["task"]]) or None)
                if spec["params"] is None:
                    spec.pop("params")
            if item["form"] == "learnability":
                spec.update(test_count=500, threshold=0.99, seeds=[2, 1, 0])
            if item["form"] == "cost":
                spec.pop("ns", None)
    for answer in (silent, explicit):
        set_spec(answer, "P1", "cost", lambda s: s.pop("ns", None))
    first, second = freeze(CARDS["k1"], silent), freeze(CARDS["k1"], explicit)

    def protocol(record):
        found = {}
        for obligation in record["obligations"]:
            for item in obligation["statements"]:
                spec = item["spec"]
                if item["form"] in ("construction", "learnability", "cost"):
                    keys = ("count", "threshold", "sizes", "params", "seeds", "test_count", "ns")
                    found[item["id"]] = ({k: spec.get(k) for k in keys}, (spec.get("widths") or {}).get("positions"))
        return found

    assert protocol(first) == protocol(second)
    assert [s["sha256"] for o in first["obligations"] for s in o["statements"]] == \
        [s["sha256"] for o in second["obligations"] for s in o["statements"]]
    composition = statement(first, "P2.composition", "composition")["spec"]
    assert composition["count"] == 200 and composition["widths"]["positions"] == "5*T+3"


# ----- replay of the recorded K2 failure -----------------------------------------------------------------------------

def recorded_recall():
    if RECORDED.is_file():
        raw = RECORDED.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == RECORDED_SHA256
        spec = statement(json.loads(raw), "P2.recall")["spec"]
        assert spec == RECORDED_RECALL
    return copy.deepcopy(RECORDED_RECALL)


def test_replay_the_recorded_k2_recall_statement_is_refused_and_the_card_width_is_frozen(production):
    """The recorded run froze d = 64 for P2.recall; under FIX-16 that value is refused, and a fixer that copies the
    card freezes d = 4K+8 (264 at K = 64, above the 128 possible keys), which the earlier passing runs used."""
    recorded = recorded_recall()
    answer = answer_for("k2")
    set_spec(answer, "P2.recall", "construction", lambda s: (s.clear(), s.update(copy.deepcopy(recorded))))
    found = problems(K2, answer)
    assert any('widths.d must be "4*K+8" (the card\'s resources.tasks.recall); the statement gives "64"' in p
               for p in found), found
    assert any('state_words must be "8*K+16"' in p for p in found)
    record = freeze(K2, answer)
    assert not any(o["claim_id"] == "P2.recall" for o in record["obligations"])
    # The rewrite the problems ask for: the recorded statement with the card's values copied.
    rewritten = dict(recorded, widths=dict(recorded["widths"], d="4*K+8"), state_words="8*K+16")
    set_spec(answer, "P2.recall", "construction", lambda s: (s.clear(), s.update(rewritten)))
    assert problems(K2, answer) == []
    spec = statement(freeze(K2, answer), "P2.recall")["spec"]
    assert spec["widths"]["d"] == K2["resources"]["tasks"]["recall"]["d"] == "4*K+8"
    assert [cr.evaluate(spec["widths"]["d"], {"K": K}) for K in spec["sizes"]] == [40, 136, 264]
    assert all(cr.evaluate(spec["widths"]["d"], {"K": K}) >= constructions.layout("recall", K)["n_keys"]
               for K in spec["sizes"])


# ----- the prompts ---------------------------------------------------------------------------------------------------

def test_the_fixer_prompt_states_the_copy_rule_and_the_resources_block():
    from hybrid.research.stages.common import load_prompt
    text = load_prompt("statement_fixer")
    assert "You copy them; you invent none." in text
    block = cr.resources_text(K2)
    assert block.startswith("CARD RESOURCES (FIX-16). Copy these values; invent none.")
    assert '"4*K+8"' in block
    for name in ("statement_fixer", "card_resources", "refine_proposer", "refine_challenger", "explorer"):
        body = load_prompt(name)
        assert "FIX-16" in body and "\u2014" not in body and "canonical" not in body.lower(), name


def test_the_resources_request_schema_validates_the_k_cards_resources():
    import jsonschema
    for card in CARDS.values():
        jsonschema.validate({"resources": card["resources"], "why": "x"}, cr.request_schema())
