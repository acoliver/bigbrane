"""FIX-4 regressions for stage C, built from the recorded calibration smoke runs/calibration-smoke-20261013T170308Z,
card k1 (results/phase4-20261013T073748Z/k1-c-diag/REPORT.md and FIX-4.md).

Fixture fixtures/calibration_smoke_20261013/k1_challenge.json (sanitized by tmp/fix4/make_fixtures.py: run, call and
job ids, receipts, hashes and paths dropped): the card, the frozen record, the stage M check summaries with every
prover source, the stage M claim records, the three parsed stage C replies (challenger, back-translation,
comparison) and the recorded stage C outcome.

The old logic is reproduced from the recorded replies: the pre-FIX-4 component rule (copied below from
c_challenge.py as it stood) with the gate that is still in place. The new logic runs the real stage C on the
recorded M results, with the recorded challenger reply restated in the FIX-4 schema (``converted_reply``), and the
recorded back-translation and comparison.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from hybrid.math import architecture, findings, learnability, profile, spec_checks
from hybrid.math.runner import MathRunner
from hybrid.research import calibration, mathrecords
from hybrid.research.calibration import load_cards
from hybrid.research.ledger import Ledger
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import Budget, StageContext, load_section
from hybrid.research.store import Store
from hybrid.tests.math.test_learnability import HAND_SET
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_math_stages import Harness
from hybrid.tests.research.test_profile_stage import Harness as ProfileHarness
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, orch_config

FIX = json.loads((Path(__file__).parent / "fixtures" / "calibration_smoke_20261013" / "k1_challenge.json")
                 .read_text(encoding="utf-8"))
CARD_ID = FIX["card"]["id"]
REPLY = FIX["replies"]["challenger"]
SQUARE = {"hidden": "d", "precision_bits": 32}
O7_WIDTHS = {"d": "32*ceiling(log(4*depth+1)/log(2))", "rounds": "depth"}


def record() -> dict:
    return copy.deepcopy(FIX["record"])


def statements(rec: dict) -> dict:
    return {s["id"]: s for s in m_math.all_statements(rec)}


def checks() -> dict:
    return copy.deepcopy(FIX["m_checks"])


# ----- the pre-FIX-4 component rule, copied from c_challenge.py (2026-10-13, before this fix) ---------------------

def old_rule(reply: dict, m_checks: dict, stmts: dict) -> tuple[list[dict], list[dict]]:
    new_premises, new_edges = [], []
    counter = 0
    for component in reply["components"]:
        sid = component["statement_id"]
        check = m_checks.get(sid) if sid else None
        runnable = bool(check) and m_math.holds(check) and c_challenge.defines(check.get("text"),
                                                                                component["function"])
        costed = runnable and stmts[sid]["form"] == "cost"
        if not (runnable and costed):
            counter += 1
            pid = f"C{counter}"
            missing = "runnable code" if not runnable else "a measured cost"
            new_premises.append({"id": pid, "text": f"component {component['name']} works as the card says "
                                                    f"(no {missing} in a check that held)", "kind": "other",
                                 "does_work": True, "why_needed": component["why"], "testable_by": "code",
                                 "measurement_plan": None, "citation": None, "added_by": "challenger"})
            new_edges += [{"premise_id": pid, "obligation_id": oid} for oid in component["used_by"]]
    for extra in reply["new_premises"]:
        counter += 1
        pid = f"C{counter}"
        new_premises.append({"id": pid, "text": extra["text"], "kind": extra["kind"], "does_work": True,
                             "why_needed": extra["why_needed"], "testable_by": "code", "measurement_plan": None,
                             "citation": None, "added_by": "challenger"})
        new_edges += [{"premise_id": pid, "obligation_id": oid} for oid in extra["used_by"]]
    return new_premises, new_edges


def slot_cells(records: list[dict]) -> dict:
    by_id = {r["claim_id"]: r["status"] for r in records}
    return {slot: calibration.cell_outcome(by_id.get(slot)) for slot in profile.SLOTS}


# ----- 1. the recorded findings reproduce on the old logic ---------------------------------------------------------

def test_the_old_rule_reproduces_all_eleven_recorded_challenger_premises():
    rec = record()
    premises, edges = old_rule(REPLY, checks(), statements(rec))
    recorded = FIX["recorded_c"]
    assert [p["id"] for p in premises] == [f"C{i}" for i in range(1, 12)]
    assert premises == recorded["new_premises"]
    assert edges == recorded["new_edges"]
    # C1 to C8 restate executed components; C9 has no code; C10 and C11 are the challenger's own premises.
    assert all("a measured cost" in p["text"] for p in premises[:8]) and "runnable code" in premises[8]["text"]


def test_the_rubric_marks_only_o15a_and_the_old_gate_gives_the_recorded_verdict():
    reasons = {e["statement_id"]: mc.mismatch_reasons(e) for e in FIX["replies"]["compare"]["statements"]}
    assert [sid for sid, r in reasons.items() if r] == ["O15.a"] == FIX["recorded_c"]["mismatched"]
    assert any(r.startswith("quantifiers:") for r in reasons["O15.a"])
    rec = record()
    premises, edges = old_rule(REPLY, checks(), statements(rec))
    mismatched = {"O15.a": "; ".join(reasons["O15.a"])}
    _, ledger, records = m_math.evaluate(rec, checks(), extra_edges=edges, extra_premises=premises,
                                         citations_checked=True, blocked={"P3": "mismatch"}, mismatched=mismatched)
    statuses = {r["claim_id"]: r["status"] for r in records}
    assert statuses == FIX["recorded_c"]["claims"]
    verdict = mathrecords.challenge_gate(records, ledger, conservation=[], magic_x=[], magic_x_missing=[],
                                         counterexamples=[], mismatched=["O15.a"])
    assert verdict == (FIX["recorded_c"]["verdict"], FIX["recorded_c"]["kill_codes"]) == ("fail",
                                                                                          ["statement_mismatch"])
    # The conditional-to-not_shown cell map stays strict.
    cells = slot_cells(records)
    assert cells["P1"] == "pass" and cells["P3"] == "not_shown" and cells["P5"] == "not_shown"
    assert all(cells[c] == "not_shown" for c in ("P2.copy", "P2.nesting", "P4.copy", "P4.composition"))
    assert calibration.cell_outcome("conditional") == "not_shown" == calibration.cell_outcome("blocked")


# ----- 2. the FIX-4 stage C on the recorded k1 results ---------------------------------------------------------------

WIDTH_CODE = '''
def verify(witness):
    mech = witness.load("O7.a")
    state = mech["init"](witness.layout("dyck", 16))
    found = witness.survey(mech["PARAMS"])
    widest = max(max(a["shape"]) for a in found["arrays"])
    d = len(state["h"])
    return (widest > d, f"at depth 16 the widest O7.a parameter axis is {widest} against d = {d}")
'''
SUBJECTS = {"256-rule hypothesis class": "hypothesis class",
            "Dyck depth decoder and recency retrieval": "nesting decoder width"}


def conservation_numbers() -> dict:
    """The recorded conservation numbers with quotes from the card, D5 and the frozen statements, and derivations of
    the three state sizes (32 bits per word times the frozen state-words formulas)."""
    claims = copy.deepcopy(REPLY["conservation"]["claims"])
    d5 = next(d["text"] for d in FIX["record"]["definitions"] if d["id"] == "D5")
    quotes = [{"source": "definition", "ref": "D5", "text": d5[d5.index("Hand-set construction"):]},
              {"source": "statement", "ref": "O5.a", "text": '"params": {"D": 4}, "sizes": [8, 32, 64]'},
              {"source": "statement", "ref": "O5.a", "text": "(8*L+1)*32*ceiling(log(2*L+1)/log(2))+1"},
              {"source": "statement", "ref": "O6.a", "text": '"params": {"V": 16}, "sizes": [8, 32, 64]'},
              {"source": "statement", "ref": "O6.a", "text": "(4*(3*K+1)+1)*32*ceiling(log(3*K+2)/log(2))+1"},
              {"source": "statement", "ref": "O7.a", "text": '"sizes": [2, 4, 8, 16]'},
              {"source": "statement", "ref": "O7.a", "text": "(8*depth**2+1)*32*ceiling(log(4*depth+1)/log(2))+1"},
              {"source": "card", "ref": None, "text": REPLY["conservation"]["quotes"][7]},
              {"source": "card", "ref": None, "text": REPLY["conservation"]["quotes"][8]}]
    derivations = [
        {"value": 4202528, "expression": "32*((8*L+1)*32*ceiling(log(2*L+1)/log(2))+1)", "inputs": {"L": 64}},
        {"value": 6332448, "expression": "32*((4*(3*K+1)+1)*32*ceiling(log(3*K+2)/log(2))+1)", "inputs": {"K": 64}},
        {"value": 14687264, "expression": "32*((8*depth**2+1)*32*ceiling(log(4*depth+1)/log(2))+1)",
         "inputs": {"depth": 16}}]
    return {"claims": claims, "quotes": quotes, "derivations": derivations, "why": REPLY["conservation"]["why"]}


def converted_reply(*, extra_components=(), extra_findings=(), magic_x=()) -> dict:
    """The recorded challenger reply in the FIX-4 schema: every component as part of the mechanism (the recorded
    reasons say so), C10 and C11 as findings with an exact anchor; C10 with verification code, C11 claiming code
    without giving any (it is then unsupported, and the host's own P5 finding covers it)."""
    components = [mf.component(c["name"], quote=c["quote"], statement_id=c["statement_id"], function=c["function"],
                               used_by=c["used_by"], subject=SUBJECTS.get(c["name"]), why=c["why"])
                  for c in REPLY["components"]]
    c10, c11 = REPLY["new_premises"]
    new = [mf.finding("nesting decoder width", claim=c10["text"], dependency=c10["why_needed"], used_by=["O7"],
                      evidence={"source": "check", "ref": "O7.a", "quote": "H = N * K * 2"}, code=WIDTH_CODE,
                      premise_kind="other", does_work=False),
           mf.finding("hypothesis class", claim=c11["text"], dependency=c11["why_needed"], used_by=["O14"],
                      evidence={"source": "statement", "ref": "O14.a", "quote": '"log2_H": 8'}, code=None,
                      premise_kind="prior", does_work=False)]
    return mf.challenger_output(magic_x=magic_x, conservation=conservation_numbers(),
                                components=[*components, *extra_components], findings=[*new, *extra_findings])


def empty_reply() -> dict:
    return mf.challenger_output()


class Replay:
    """Stage C on the recorded k1 stage M: the frozen record and the M verdict are stored as stage M stored them, the
    challenger answers ``replies`` (one per check batch), the matcher gives the recorded answers."""

    def __init__(self, tmp_path, replies, *, rec=None, m_checks=None, compare=None):
        self.run_dir = (tmp_path / "run").resolve()
        self.ledger = Ledger(self.run_dir)
        self.store = Store(self.run_dir, self.ledger)
        self.store.put_card(copy.deepcopy(FIX["card"]))
        self.executor = mf.MathExecutor(tmp_path / "exec")
        script = {"challenger": list(replies),
                  "statement_backtranslate": mf.recorded_backtranslation(FIX["replies"]["backtranslate"]),
                  "statement_match": recorded_compare(compare or FIX["replies"]["compare"])}
        self.llm = FakeLLM(self.run_dir, self.ledger, script,
                           models={"challenger": "gpt-6.1-sol", "statement_match": "qwen3.6-35b-a3b-mlx"})
        self.ctx = StageContext(run_id=self.run_dir.name, run_dir=self.run_dir, ledger=self.ledger, store=self.store,
                                llm=self.llm, runner=None, bench=None, cfg=orch_config(),
                                budget=Budget(1, 500, 10**9, 3600.0), lenses={},
                                math=MathRunner(self.executor, mf.IMAGE, self.run_dir))
        rec = rec or record()
        for statement in m_math.all_statements(rec):
            statement["sha256"] = m_math.statement_digest(statement)
        for definition in rec["definitions"]:
            definition["sha256"] = mc.definition_hash(definition)
        rec.update(card_version=1, fixer_calls=[], fixer_model=FIX["m_models"]["statement_fixer"][0], dropped=[],
                   first_problems=[])
        stored = self.ctx.put_json_artifact("math_statements", self.ctx.artifact_name(CARD_ID), rec, CARD_ID)
        hashes = {s["id"]: s["sha256"] for s in m_math.all_statements(rec)}
        hashes.update({d["id"]: d["sha256"] for d in rec["definitions"]})
        self.ctx.emit("math_statements_frozen", {"card_id": CARD_ID, "path": stored["path"].relative_to(
            self.run_dir).as_posix(), "sha256": stored["sha256"], "hashes": hashes})
        verdict = {"card_id": CARD_ID, "stage": "M", "verdict": "undetermined", "kill_codes": [],
                   "claims": FIX["m_claims"], "models": FIX["m_models"], "checks": m_checks or checks()}
        self.ctx.put_json_artifact("math_verdict", self.ctx.artifact_name(CARD_ID), verdict, CARD_ID)

    def run(self):
        return asyncio.run(c_challenge.run(self.ctx, CARD_ID))

    def verdict(self) -> dict:
        return [v for v in c_challenge._verdicts(self.ctx, CARD_ID) if v["stage"] == "C"][-1]

    def statuses(self) -> dict:
        return {c["claim_id"]: c["status"] for c in self.verdict()["claims"]}

    def challenge(self) -> dict:
        return [json.loads((self.run_dir / e["data"]["path"]).read_text()) for e in self.ledger.iter_events("artifact")
                if e["data"]["kind"] == "challenge"][-1]

    def ledger_entries(self) -> dict:
        name = self.verdict()["assumption_ledger"]
        data = json.loads((self.run_dir / "artifacts" / "assumption_ledger" / name).read_text())
        return {p["id"]: p for p in data["premises"]}

    def challenger_users(self) -> list[str]:
        return [c["user"] for c in self.llm.calls if c["role"] == "challenger"]


def recorded_compare(reply: dict):
    """The recorded comparison: statements and definitions as recorded; the premises are the ones this stage shows
    (their ids differ from the recorded C1 to C11), each implying no claim, as the recorded answer said of all eleven.
    FIX-7: each recorded difference carries an anchored objection (math_fakes.anchored), and the default cross-check
    upholds it, so a recorded mismatch still blocks here (test_fix7_match.py replays the dispute)."""
    def answer(index: int, user: str) -> dict:
        shown = json.JSONDecoder().raw_decode(user.split("PREMISES:\n", 1)[1])[0]
        return mf.anchored(dict(copy.deepcopy(reply), premises=[{"premise_id": p["premise_id"], "implies": [],
                                                                 "why": "states a dependency, not a claim"}
                                                                for p in shown]), user)
    return answer


def matching_compare() -> dict:
    """The recorded comparison with O15.a's two objections taken back (the diagnosis found both to be false)."""
    compare = copy.deepcopy(FIX["replies"]["compare"])
    entry = next(e for e in compare["statements"] if e["statement_id"] == "O15.a")
    entry["direction"] = "match"
    for item in entry["rubric"].values():
        item["verdict"] = "same"
    return compare


def fid(subject: str, oids) -> str:
    return c_challenge.finding_id(FIX["record"], subject, oids)


def test_new_logic_keeps_observed_evidence_and_blocks_only_the_real_gaps(tmp_path):
    replay = Replay(tmp_path, [converted_reply(), empty_reply()])
    assert replay.run() is None
    statuses = replay.statuses()
    # Executed components keep their evidence: copy, recall, composition and every P4 slot are observed again.
    for slot in ("P2.copy", "P2.recall", "P2.composition", "P4.copy", "P4.recall", "P4.nesting", "P4.composition",
                 "sample_efficiency.prediction"):
        assert statuses[slot] == "observed", slot
    # The nesting width (C3 and C10) and the undefined hypothesis class (C9 and C11) still block; P1's coverage of the
    # witnesses is an accounting gap, because no architecture contract was frozen.
    assert statuses["P2.nesting"] == "conditional" and statuses["P5"] == "conditional"
    assert statuses["P1"] == "conditional" and statuses["P3"] == "blocked"
    verdict = replay.verdict()
    assert (verdict["verdict"], verdict["kill_codes"]) == ("fail", ["statement_mismatch"])
    by_id = {f["id"]: f for f in verdict["findings"]}
    width, cls, arch = fid("nesting decoder width", ["O7"]), fid("hypothesis_class", ["O14"]), fid(
        "architecture_contract", ["O4"])
    assert set(by_id) == {width, cls, arch}
    assert by_id[width]["disposition"] == "substantiated" and by_id[width]["sources"] == ["challenger finding"]
    # C9 (a prose component), C11 (code-testable without code) and the host's own P5 check are one finding.
    assert by_id[cls]["disposition"] == "substantiated"
    assert sorted(by_id[cls]["sources"]) == ["challenger finding", "component 256-rule hypothesis class", "host"]
    assert by_id[arch]["disposition"] == "accounting" and by_id[arch]["used_by"] == ["O4"]
    challenge = replay.challenge()
    dispositions = {c["name"]: c["disposition"] for c in challenge["components"]}
    assert dispositions["Cost-reference address and content stores"] == "covered"
    assert [dispositions[c["name"]] for c in REPLY["components"][1:5]] == ["accounting"] * 4
    assert [dispositions[c["name"]] for c in REPLY["components"][5:9]] == ["training_scope"] * 4
    ledger = replay.ledger_entries()
    assert set(ledger) == {width, cls, arch}
    assert "accounting (section 5.3)" in ledger[arch]["basis"] and ledger[arch]["used_by"] == ["O4"]
    # The width finding was confirmed by running its code on the recorded O7.a source.
    assert "2176 against d = 224" in ledger[width]["basis"]
    # Conservation ran on numbers quoted from D5 and the frozen statements, with derived state sizes.
    assert challenge["conservation"]["ran"] and challenge["conservation"]["status"] == "discharged"


def test_without_the_p3_false_alarm_the_card_is_undetermined_on_the_real_gaps_only(tmp_path):
    replay = Replay(tmp_path, [converted_reply(), empty_reply()], compare=matching_compare())
    assert replay.run() is None
    verdict = replay.verdict()
    gaps = [fid("architecture_contract", ["O4"]), fid("nesting decoder width", ["O7"]), fid("hypothesis_class",
                                                                                           ["O14"])]
    assert verdict["verdict"] == "undetermined"
    assert verdict["kill_codes"] == sorted([f"unproven:{g}" for g in gaps]
                                           + [f"unproven:capability.{i}" for i in range(1, 6)])
    cells = slot_cells(verdict["claims"])
    assert cells["P3"] == "pass" and cells["P2.copy"] == "pass" and cells["P4.copy"] == "pass"
    assert cells["P2.nesting"] == "not_shown" and cells["P5"] == "not_shown" and cells["P1"] == "not_shown"
    assert verdict["profile"]["passed"] is False


def test_an_uncosted_external_index_in_a_witness_still_blocks(tmp_path):
    index = mf.component("content lookup index", quote="the second retrieves the item stored after the matching "
                         "one", statement_id="O6.a", function="init", used_by=["O6"], role="external", kind="store",
                         why="the binary key codes act as an index the mechanism does not compute")
    replay = Replay(tmp_path, [converted_reply(extra_components=[index]), empty_reply()],
                    compare=matching_compare())
    replay.run()
    pid = fid("content lookup index", ["O6"])
    assert replay.statuses()["P2.recall"] == "conditional"
    assert f"unproven:{pid}" in replay.verdict()["kill_codes"]
    assert "supplied from outside the mechanism" in replay.ledger_entries()[pid]["basis"]


# ----- 3. evidence delivery and conservation provenance -------------------------------------------------------------

def test_the_checks_go_to_the_challenger_complete_in_bounded_batches(tmp_path):
    full = json.dumps({sid: c_challenge._check_view(c) for sid, c in FIX["m_checks"].items()}, indent=1,
                      ensure_ascii=False, default=str)
    assert len(full) > c_challenge.CHECK_BATCH_CHARS   # the recorded checks did not fit in one prompt
    batches = c_challenge.check_batches(FIX["m_checks"])
    assert len(batches) == 2
    assert [sid for batch in batches for sid in batch] == list(FIX["m_checks"])
    for batch in batches:
        assert len(json.dumps(batch, indent=1, ensure_ascii=False, default=str)) <= c_challenge.CHECK_BATCH_CHARS
    replay = Replay(tmp_path, [converted_reply(), empty_reply()])
    replay.run()
    users = replay.challenger_users()
    assert len(users) == 2 and "BATCH 1 OF 2" in users[0] and "BATCH 2 OF 2" in users[1]
    # Every check arrives once and whole, including O8.b's source (cut in the recorded prompt) and O9.a's check
    # (left out of it).
    delivered = {}
    for user in users:
        shown = user.split("in the other batches", 1)[1].split("):\n", 1)[1].split("\n\nPREMISE LEDGER:", 1)[0]
        part = json.loads(shown)
        assert not set(part) & set(delivered)
        delivered.update(part)
    assert delivered == json.loads(full)
    assert delivered["O8.b"]["text"] == FIX["m_checks"]["O8.b"]["text"] and "O9.a" in delivered
    assert load_section("challenge_findings").splitlines()[0] in users[0]
    assert load_section("challenge_profile").splitlines()[0] in users[0]


def test_conservation_numbers_from_frozen_definitions_are_accepted_and_invented_ones_refused():
    numbers = conservation_numbers()
    assert c_challenge.conservation_problems(numbers, FIX["card"], FIX["record"]) == []
    invented = copy.deepcopy(numbers)
    invented["claims"]["copy"][0]["L"] = 128
    assert any("128" in p for p in c_challenge.conservation_problems(invented, FIX["card"], FIX["record"]))
    altered = copy.deepcopy(numbers)
    altered["quotes"][0]["text"] = "A state word contributes 64 bits to conservation accounting."
    assert any("not text of definition D5" in p for p in c_challenge.conservation_problems(altered, FIX["card"],
                                                                                           FIX["record"]))
    wrong = copy.deepcopy(numbers)
    wrong["derivations"][0]["value"] = 4202529
    wrong["claims"]["copy"][0]["state_bits"] = 4202529
    problems = c_challenge.conservation_problems(wrong, FIX["card"], FIX["record"])
    assert any("is 4.20253e+06, not 4202529" in p for p in problems)
    unquoted = copy.deepcopy(numbers)
    unquoted["derivations"][0]["expression"] = "33*((8*L+1)*32*ceiling(log(2*L+1)/log(2))+1)"
    assert any("[33.0]" in p for p in c_challenge.conservation_problems(unquoted, FIX["card"], FIX["record"]))


# ----- 4. the architecture contract ------------------------------------------------------------------------------

def o7_spec(**changes) -> dict:
    spec = next(s["spec"] for s in m_math.all_statements(FIX["record"]) if s["id"] == "O7.a")
    return dict(copy.deepcopy(spec), **dict({"widths": O7_WIDTHS, "count": 20}, **changes))


def test_the_recorded_nesting_witness_breaks_a_square_contract_at_depth_8_and_16():
    result = spec_checks.audited_construction(o7_spec(), FIX["m_checks"]["O7.a"]["text"], SQUARE)
    assert result["status"] == "violated"
    bad = {p["point"]["depth"]: p for p in result["architecture"]["points"] if p["violations"]}
    assert sorted(bad) == [8, 16] and bad[16]["widest_axis"] == 2176 and bad[16]["dimensions"]["d"] == "224"
    assert "(2176, 224)" in result["evidence"]
    # The accuracy itself was not the problem: every size met the threshold. Since FIX-8 a size that breaks the
    # architecture contract is not a P2 success, so depths 8 and 16 fail on the contract alone.
    assert all(r["accuracy"] >= 0.99 for r in result["results"])
    rows = {r["size"]: r for r in result["results"]}
    assert rows[2]["passed"] and rows[4]["passed"]
    for depth in (8, 16):
        assert not rows[depth]["passed"]
        assert any(v.startswith(f"architecture at depth={depth}") for v in rows[depth]["resource_violations"])


def test_a_declared_and_counted_hidden_width_admits_the_same_witness():
    wide = {"hidden": "8*d", "precision_bits": 32}
    widths = {"d": "32*ceiling(log(4*depth+1)/log(2))", "rounds": "depth", "positions": "4*depth"}
    result = spec_checks.audited_construction(o7_spec(widths=widths), FIX["m_checks"]["O7.a"]["text"], wide)
    # 2176 > 8 * 224 = 1792 still breaks it at depth 16; 576 <= 8 * 192 passes at depth 8.
    assert result["status"] == "violated"
    assert [p["point"]["depth"] for p in result["architecture"]["points"] if p["violations"]] == [16]


def test_the_cost_reference_must_realize_a_declared_hidden_width():
    source = FIX["m_checks"]["O4.a"]["text"]
    spec = next(s["spec"] for s in m_math.all_statements(FIX["record"]) if s["id"] == "O4.a")
    claim = m_math.cost.CostClaim(**{k: spec.get(k) for k in m_math.cost.CLAIM_FIELDS})
    square = m_math.cost.verify_cost(source, claim, ns=(64, 128, 256, 512),
                                     architecture={"shared": SQUARE, "widths": {"d": "d", "rounds": "2"}})
    assert square["architecture"]["violations"] == []
    wider = m_math.cost.verify_cost(source, claim, ns=(64, 128, 256, 512),
                                    architecture={"shared": {"hidden": "8*d", "precision_bits": 32},
                                                  "widths": {"d": "d", "rounds": "2"}})
    assert wider["status"] == "violated" and wider["failed_checks"]["architecture"]
    assert "does not realize the declared architecture" in " ".join(wider["architecture"]["violations"])


def nesting_script(hidden: str) -> tuple[dict, dict]:
    """Calibration card k1 scripted as in test_profile_stage, with the recorded O7.a statement (widths added) and
    the recorded O7.a source as its P2.nesting witness, under a contract of hidden width ``hidden``.

    FIX-16: the contract and the nesting widths are the card's own values, which the fixer copies. The drift is
    therefore stated in the card: its resources carry the contract (hidden width ``hidden``, 32-bit precision) and,
    for dyck, the recorded O7.a widths and state words."""
    card = copy.deepcopy(load_cards(("k1",))["k1"])
    recorded = o7_spec()
    card["resources"]["architecture"] = {"hidden": hidden, "precision_bits": 32, "state_bits": None}
    card["resources"]["tasks"]["dyck"] = dict(card["resources"]["tasks"]["dyck"], **recorded["widths"],
                                              state_words=recorded["state_words"])
    fixer = cf.fixer_output("k1", card)
    fixer["architecture"] = {"hidden": hidden, "precision_bits": 32}
    next(o for o in fixer["obligations"] if o["claim_id"] == "P2.nesting")["statements"] = [
        {"form": "construction", "construction": o7_spec()}]
    base = cf.prover_output("k1")

    def prover(index: int, user: str) -> dict:
        answer = base(index, user)
        for item, attempt in zip(cf._shown(user), answer["attempts"]):
            if item["form"] == "construction" and item["frozen_statement"]["task"] == "dyck":
                attempt["source"] = FIX["m_checks"]["O7.a"]["text"]
        return answer

    script = cf.script("k1", card, fixer=fixer)
    script["prover"] = prover
    return card, script


@pytest.mark.parametrize("hidden", ["d", "16*d"])
def test_the_nesting_width_drift_blocks_through_the_contract_in_stage_m(tmp_path, hidden):
    """End to end: under a square contract the recorded witness fails P2.nesting with every accuracy at 1.0; under a
    contract wide enough for it, the square cost reference no longer realizes the declared width and P1 fails. The
    drift cannot pass both."""
    card, script = nesting_script(hidden)
    h = ProfileHarness(tmp_path, script=script, card=card)
    h.m()
    verdict = h.verdict("M")
    slots = {slot: entry["status"] for slot, entry in verdict["profile"]["slots"].items()}
    nesting = next(c for sid, c in verdict["checks"].items() if c["form"] == "construction"
                   and "dyck:" in c["evidence"])
    assert "depth=16 instance accuracy 1.0000" in nesting["evidence"]
    assert h.frozen()["profile"]["architecture"] == {"hidden": hidden, "precision_bits": 32}
    assert all(c["details"]["architecture"]["checked"] for c in verdict["checks"].values()
               if c["form"] in ("cost", "construction", "learnability"))
    if hidden == "d":
        assert slots["P2.nesting"] == "failed" and slots["P1"] == "observed"
        assert verdict["kill_codes"] == ["obligation_failed:P2.nesting"]
        assert "architecture contract broken" in nesting["evidence"]
    else:
        assert slots["P2.nesting"] == "observed" and slots["P1"] == "failed"
        assert verdict["kill_codes"] == ["obligation_failed:P1"]
        assert verdict["checks"]["O1.a"]["details"]["failed_checks"]["architecture"]
    assert all(status in ("observed", "proven") for slot, status in slots.items() if slot not in ("P1", "P2.nesting"))


def test_the_p4_controls_stay_enforced_under_the_architecture_contract():
    spec = {"task": "recall", "size": 3, "examples": 40, "seeds": [0, 1, 2], "test_count": 40,
            "widths": {"d": "64", "rounds": "1"}}
    result = learnability.run_learnability(spec, HAND_SET, {"hidden": "d", "precision_bits": 32})
    assert result["status"] == "violated" and result["counterexample"]["reason"] == "data_independent"
    assert result["architecture"]["checked"]


def test_profile_statements_need_widths_and_p5_needs_a_defined_hypothesis_class():
    rec = FIX["record"]
    spec = {s["id"]: s["spec"] for s in m_math.all_statements(rec)}
    assert any("widths" in p for p in profile.statement_problems("P2.nesting", "construction", spec["O7.a"]))
    assert any("widths" in p for p in profile.statement_problems("P1", "cost", spec["O4.a"]))
    assert any("widths" in p for p in profile.statement_problems("P4.copy", "learnability", spec["O10.a"]))
    assert not any("widths" in p for p in profile.statement_problems("P2.nesting", "construction",
                                                                      dict(spec["O7.a"], widths=O7_WIDTHS)))
    p5 = profile.statement_problems("P5", "conservation", spec["O14.a"])
    assert p5 and all("hypothesis_class" in p for p in p5)
    entry = dict(spec["O14.a"]["sample"][0], hypothesis_class={
        "definition": "every readout table from 4 symbols to 4 answers", "size_justification": "4**4 = 256 tables, "
        "and the mechanism supplies no prior bits", "log2_size": 8})
    assert profile.hypothesis_class_problems(entry) == []
    assert profile.hypothesis_class_problems(dict(entry, log2_H=9))
    assert any("architecture must be {hidden, precision_bits}" in p
               for p in profile.record_problems([], [{"task": "copy"}], None))
    assert architecture.shared_problems(SQUARE) == []
    assert architecture.shared_problems({"hidden": "8*depth", "precision_bits": 32})


# ----- 5. findings: settlement on the example card ------------------------------------------------------------------

HOST_KINDS = ("audited_construction", "conservation", "magic_x", "finding", "sympy")


def run_with(tmp_path, *, findings=(), components=(), magic_x=(), fixer=None):
    """Stages M and C on the example card, every finding verification and SymPy derivation run for real."""
    challenger = mf.challenger_output(findings=findings, components=components, magic_x=magic_x)
    h = Harness(tmp_path, mf.math_script(fixer=fixer, challenger=challenger), host_kinds=HOST_KINDS)
    h.m()
    return h, h.c()


CONFIRM = "def verify(witness):\n    return (True, 'the O6.a source defines predict')\n"
DENY = "def verify(witness):\n    return (False, 'the O6.a table is filled from the stream it reads')\n"
CRASH = "def verify(witness):\n    return witness.load('O99.z')\n"


def test_unsupported_and_rejected_challenges_do_not_block_a_valid_card(tmp_path):
    allegations = [mf.finding("no code given", code=None),
                   mf.finding("code that crashes", code=CRASH),
                   mf.finding("code that says no", code=DENY),
                   mf.finding("misquoted", code=CONFIRM, evidence={"source": "card", "ref": None,
                                                                    "quote": "a sentence the card never wrote"}),
                   mf.finding("a derivation that fails", testable_by="derivation",
                              derivation={"symbols": {"n": "positive_integer"}, "lhs": "n + 1", "rhs": "n",
                                          "relation": "=="}),
                   mf.finding("a citation with no source", testable_by="citation")]
    h, nxt = run_with(tmp_path, findings=allegations)
    assert nxt == "S4" and h.verdict("C")["verdict"] == "pass"
    dispositions = {f["subject"]: f["disposition"] for f in h.verdict("C")["findings"]}
    assert dispositions == {"no code given": "unsupported", "code that crashes": "unsupported",
                            "code that says no": "rejected", "misquoted": "unsupported",
                            "a derivation that fails": "rejected", "a citation with no source": "unsupported"}
    assert not any(pid.startswith("C") for pid in h.ledger_of("C"))


def test_a_confirmed_missing_assumption_blocks_until_settled_and_a_plan_lets_it_advance(tmp_path):
    h, nxt = run_with(tmp_path / "a", findings=[mf.finding("hidden table", code=CONFIRM)])
    pid = c_challenge.finding_id({"obligations": [{"id": "O5", "claim_id": "capability.1"}]}, "hidden table", ["O5"])
    assert nxt is None and h.verdict("C")["kill_codes"] == [f"unproven:{pid}"]
    assert h.claim("C", "capability.1")["status"] == "conditional"
    h, nxt = run_with(tmp_path / "b", findings=[mf.finding("hidden table", code=CONFIRM, plan=mf.PLAN)])
    assert nxt == "S4" and h.verdict("C")["scheduled_measurements"][0]["premise_id"] == pid


def test_a_confirmed_witness_violation_fails_the_claim_only_when_the_conforming_witness_fails(tmp_path):
    """FIX-13: confirmation shows the deviation exists. Without a conforming replacement to replay it is unresolved
    and blocks; a replacement that fails the frozen check makes it material and fails the claim."""
    h, nxt = run_with(tmp_path / "a", findings=[mf.finding("wider table than declared", kind="witness_violation",
                                                           code=CONFIRM, used_by=["O6"])])
    pid = c_challenge.finding_id({"obligations": [{"id": "O6", "claim_id": "capability.2"}]},
                                 "wider table than declared O6", ["O6"])
    assert nxt is None
    assert h.verdict("C")["kill_codes"] == [f"unproven:{pid}"]
    assert h.claim("C", "capability.2")["status"] == "conditional"
    store = 'state["table"][state["key"]] = token'
    material = mf.finding("wider table than declared", kind="witness_violation", code=CONFIRM, used_by=["O6"],
                          targets=[mf.target("O6.a", store, [(store, "state")])])
    h, nxt = run_with(tmp_path / "b", findings=[material])
    assert nxt is None
    assert h.verdict("C")["kill_codes"] == ["obligation_failed:capability.2"]
    assert h.claim("C", "capability.2")["status"] == "failed"


def test_an_explanation_objection_does_not_attach_to_a_profile_slot():
    rec = {"obligations": [{"id": "O8", "claim_id": "P2.composition"}, {"id": "O20", "claim_id": "capability.4"}]}
    used, why = c_challenge.finding_scope(mf.finding("round order", kind="explanation", used_by=["O8"]), rec)
    assert used == [] and "profile slot" in why
    used, why = c_challenge.finding_scope(mf.finding("round order", kind="explanation", used_by=["O8", "O20"]), rec)
    assert used == ["O20"] and why is None


# ----- 6. T3-style oracles, original and newly discovered -----------------------------------------------------------

ORACLE_X = {"code": mf.references.RETRIEVAL_ORACLE, "claim": dict(mf.references.MAGIC_X_CLAIM),
            "quote": mf.CAPABILITY_1_QUOTE, "why": "the index alone answers recall"}


def test_an_original_oracle_premise_is_caught_by_magic_x(tmp_path):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "a perfect retrieval index", "store")],
                            capability_1_premises=["A1"])
    h, nxt = run_with(tmp_path, fixer=fixer, magic_x=[dict(ORACLE_X, premise_id="A1")])
    assert nxt is None and h.verdict("C")["kill_codes"] == ["magic_x:A1"]


def test_a_newly_discovered_oracle_is_caught_by_magic_x(tmp_path):
    oracle = mf.finding("perfect retrieval index", code=CONFIRM, premise_kind="store", magic_x=ORACLE_X)
    h, nxt = run_with(tmp_path, findings=[oracle])
    pid = c_challenge.finding_id({"obligations": [{"id": "O5", "claim_id": "capability.1"}]},
                                 "perfect retrieval index", ["O5"])
    assert nxt is None and h.verdict("C")["kill_codes"] == [f"magic_x:{pid}"]
    request = next(r for r in h.executor.requests() if r["kind"] == "magic_x")
    assert request["assumption"]["id"] == pid and request["assumption"]["cost_in_p1"] is False


def test_a_newly_discovered_oracle_without_x_code_still_blocks(tmp_path):
    h, nxt = run_with(tmp_path, findings=[mf.finding("perfect retrieval index", code=CONFIRM, premise_kind="store")])
    assert nxt is None and h.verdict("C")["verdict"] == "undetermined"
    assert h.claim("C", "capability.1")["status"] == "conditional"


# ----- 7. cost coverage through the shared contract -----------------------------------------------------------------

def contract_setup(*, covered: bool):
    """The recorded k1 run as if it had frozen a square contract and every witness passed it (P1 too when
    ``covered``); a premise A1 that the copy table implements."""
    rec = record()
    rec["profile"]["architecture"] = SQUARE
    rec["premises"] = [mf.premise("A1", "the copy address table", "store")]
    rec["premises"][0]["added_by"] = "statement_fixer"
    next(o for o in rec["obligations"] if o["id"] == "O5")["premises"] = ["A1"]
    m_checks = checks()
    passed = {"checked": True, "points": [], "violations": []}
    for sid in ("O5.a", "O6.a", "O8.a", "O8.b") + (("O4.a",) if covered else ()):
        m_checks[sid].setdefault("details", {})
        m_checks[sid]["details"] = dict(m_checks[sid]["details"] or {}, architecture=passed)
    return rec, m_checks


@pytest.mark.parametrize("covered", [True, False])
def test_linked_components_are_covered_only_when_p1_held_under_the_same_contract(tmp_path, covered):
    rec, m_checks = contract_setup(covered=covered)
    table = next(c for c in REPLY["components"] if c["name"] == "Hand-set copy position-address table")
    component = mf.component(table["name"], quote=table["quote"], statement_id="O5.a", function="init",
                             used_by=["O5"], premise_id="A1")
    reply = mf.challenger_output(components=[component], magic_x=[{
        "premise_id": "A1", "code": mf.references.POSITION_PARITY, "claim": dict(mf.references.MAGIC_X_CLAIM),
        "quote": "", "why": "the table as X"}])
    replay = Replay(tmp_path, [reply, empty_reply()], rec=rec, m_checks=m_checks, compare=matching_compare())
    replay.run()
    disposition = next(c for c in replay.challenge()["components"] if c["name"] == table["name"])["disposition"]
    request = next(r for r in replay.executor.requests() if r["kind"] == "magic_x")
    if covered:
        assert disposition == "covered" and request["assumption"]["cost_in_p1"] is True
        assert replay.statuses()["P1"] == "observed"
    else:
        # The witness passed the contract, but P1 did not hold under it: an accounting finding on P1, and no
        # exemption for A1; the copy result keeps its observed evidence through A1's own status.
        assert disposition == "accounting" and request["assumption"]["cost_in_p1"] is False
        assert replay.statuses()["P1"] == "conditional"


def test_a_witness_without_a_contract_check_does_not_inherit_p1_cost(tmp_path):
    rec, m_checks = contract_setup(covered=True)
    m_checks["O7.a"]["details"].pop("architecture", None)
    decoder = next(c for c in REPLY["components"] if c["name"].startswith("Dyck depth decoder"))
    component = mf.component(decoder["name"], quote=decoder["quote"], statement_id="O7.a", function="build",
                             used_by=["O7"], subject="nesting decoder width")
    item = c_challenge.component_disposition(component, FIX["card"], rec, m_checks, statements(rec),
                                             c_challenge.p1_coverage(rec, m_checks))
    assert item["disposition"] == "accounting" and "no verified P1 cost covers it" in item["reason"]


# ----- 8. P3 presentation ---------------------------------------------------------------------------------------------

def test_the_p3_comparison_sees_the_selected_limit_and_the_back_translation_stays_blind(tmp_path):
    replay = Replay(tmp_path, [converted_reply(), empty_reply()])
    replay.run()
    back = next(c["user"] for c in replay.llm.calls if c["prompt"] == "statement_backtranslate")
    compare = next(c["user"] for c in replay.llm.calls if c["prompt"] == "statement_match")
    limit = FIX["record"]["profile"]["limits"][0]
    # The blind translation sees the frozen statements and definitions only (D9 defines the P3 task family).
    assert limit["resources"] not in back and limit["why"] not in back and "instantiated_claim" not in back
    assert "profile P3:" not in back
    items = json.loads(compare.split("STATEMENTS TO COMPARE:\n", 1)[1].split("\n\nDEFINITIONS:", 1)[0])
    p3 = next(i for i in items if i["statement_id"] == "O15.a")
    assert p3["profile_slot"]["selected_limits"] == FIX["record"]["profile"]["limits"]
    assert p3["instantiated_claim"].startswith("There is a task family the mechanism cannot solve")
    assert limit["resources"] in p3["instantiated_claim"]
    assert load_section("profile_match_limits").splitlines()[0] in compare
    # FIX-7: the P4 statements carry their selected solvable case as an instantiated claim; nothing else does.
    p4 = [i for i in items if i.get("profile_slot", {}).get("slot", "").startswith("P4.")]
    assert p4 and all(i["instantiated_claim"].startswith("From its initial state") for i in p4)
    assert not any("instantiated_claim" in i for i in items if i["statement_id"] != "O15.a" and i not in p4)


def test_finding_code_runs_against_the_recorded_sources():
    sources = {sid: c["text"] for sid, c in FIX["m_checks"].items() if c.get("text")}
    result = findings.verify_finding(WIDTH_CODE, sources, {"O7.a": {"form": "construction", "spec": {}}})
    assert result["status"] == "discharged" and "2176 against d = 224" in result["evidence"]
    refused = findings.verify_finding("import os\ndef verify(witness):\n    return (True, 'x')\n", sources, {})
    assert refused["status"] == "unknown" and refused["prover_fault"]
    wrong = findings.verify_finding("def verify(witness):\n    return 'yes'\n", sources, {})
    assert wrong["status"] == "unknown" and wrong["prover_fault"]
