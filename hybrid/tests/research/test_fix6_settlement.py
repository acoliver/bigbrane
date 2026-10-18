"""FIX-6 regressions for stage M (results/phase4-20261013T073748Z/FIX-6.md), from the recorded calibration line
runs/calibration-math-20261014T053341Z/k1-run1 (sanitized extract: fixtures/calibration_math_20261014/k1_run1.json,
written by results/phase4-20261013T073748Z/fix6-logs/make_fix6_fixture.py).

1. The statement fixer's own conformance premise. Both fixer replies of k1-run1 wrote premise A1, "the reference
   executes the card's equations with D1's product spellings and D2's retained buffers", with no obligation that
   discharges it; its measurement plan was deferred to test design, which calibration never runs, and the three
   complexity claims and P3 stayed conditional. A premise about the card's own frozen definitions is now settled at
   stage M by a frozen discharging statement (a conformance statement for the cost reference) or refused at freeze.
2. Within-run best-attempt memory for P2 constructions, and the frozen contract as the authority. The prover's four
   composition and binding constructions of k1-run1 failed; each round saw only the latest source, and it argued that
   three rounds were needed while submitting witnesses it expected to fail.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from hybrid.math import spec_checks
from hybrid.research import mathrecords
from hybrid.research.calibration import load_cards
from hybrid.research.schemas import validate
from hybrid.research.stages import m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages import prover_memory as pm
from hybrid.research.stages import r_refine
from hybrid.research.stages.common import load_section
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_profile_stage import Harness

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "calibration_math_20261014" / "k1_run1.json")
                     .read_text(encoding="utf-8"))
RECORD = FIXTURE["frozen_record"]
CLAIMS = RECORD["claims"]
CLAIM_IDS = [c["id"] for c in CLAIMS]
TEXTS = {c["id"]: c["text"] for c in CLAIMS}
MAX_STATEMENTS = 40
DEPENDENT = ("complexity.time_per_token", "complexity.memory", "complexity.total_sequence", "P3")

# The card's equations as the fixer would freeze them from D1 and D2 of k1-run1, reading the parameters by the names
# a definition fixes (E, B, O and per round Wa, Wc, Wp, Wo, U, Vm), as the recorded P1 reference holds them.
K1_EQUATIONS = '''
def expected(params, sizes, tokens):
    d = int(sizes["d"])
    sq = float(d) ** 0.5
    stores = [[np.zeros((0, d)), np.zeros((0, d))], [np.zeros((0, d)), np.zeros((0, d))]]
    out = []
    for t, token in enumerate(tokens):
        h = params["E"][token] + params["B"][t % d]
        for r, name in enumerate(["R1", "R2"]):
            R = params[name]
            A = np.concatenate([stores[r][0], (R["Wa"] @ h).reshape(1, d)], axis=0)
            C = np.concatenate([stores[r][1], (R["Wc"] @ h).reshape(1, d)], axis=0)
            stores[r] = [A, C]
            e = np.exp((A @ (R["Wp"] @ h)) / sq)
            g = h + R["Wo"] @ ((e / np.sum(e)) @ C)
            h = g + R["U"] @ np.maximum(R["Vm"] @ g, 0.0)
        state = {"stores": [stores[0][0], stores[0][1], stores[1][0], stores[1][1]], "h": h, "t": t + 1}
        out.append({"state": state, "output": int(np.argmax(h @ params["O"]))})
    return out
'''
# FIX-20: precision_bits and tolerance are host values (the card's declared precision, k1's binary64, and the
# protocol's exact tolerance). The fixture card predates the card's resources section, so the host leaves this spec as
# written and the test gives the values the host sets for a card that declares its precision.
CONFORMANCE = {"reference": "O4.a", "equations": K1_EQUATIONS, "points": [{"n": 24, "d": 8, "V": 4},
                                                                           {"n": 48, "d": 16, "V": 16}],
               "streams": ["repeated", "distinct", "random"], "precision_bits": 64, "tolerance": 0.0}
PARAMETER_NAMES = {"id": "D9", "kind": "notation",
                   "text": "The cost reference returns params as a dict with E, B, O and per round R1 and R2 the "
                           "matrices Wa, Wc, Wp, Wo, U and Vm."}


def recorded_checks() -> dict[str, dict]:
    """The deciding (latest) stage M check of every statement of k1-run1."""
    latest: dict[str, dict] = {}
    for check in FIXTURE["checks"]:
        latest[check["statement_id"]] = check
    return latest


def reference_source() -> str:
    return recorded_checks()["O4.a"]["text"]


def with_discharger(parsed: dict, spec: dict | None = None) -> dict:
    """The recorded second fixer reply plus an obligation with claim_id null that discharges A1 by a conformance
    statement against the P1 cost statement."""
    answer = copy.deepcopy(parsed)
    answer["definitions"].append(dict(PARAMETER_NAMES))
    answer["obligations"].append({"id": "O16", "claim_id": None, "claim_quote": "premise A1",
                                  "statements": [{"form": "conformance", "conformance": dict(spec or CONFORMANCE)}],
                                  "definitions": ["D1", "D2", "D9"], "premises": [], "discharges": ["A1"]})
    return answer


def settlement(problems: list[str]) -> list[str]:
    return [p for p in problems if p.startswith("premise A1")]


# FIX-17 (DECISIONS.md P3) refuses a P3 limit with no mechanism basis at freeze. k1-run1 predates that field; its one
# limit (four retained rows of width d per token) follows from the card's write update, so the reply is frozen with
# that basis and P3 stays a claim that depends on A1.
K1_LIMIT_BASIS = {"part": "update", "anchor": "mechanism.update.write", "quote": "appended as new rows of A_l and C_l"}


def with_limit_basis(parsed: dict) -> dict:
    answer = copy.deepcopy(parsed)
    for limit in answer["limits"]:
        limit.setdefault("basis", dict(K1_LIMIT_BASIS))
    return answer


def freeze(parsed: dict) -> dict:
    return mc.freeze(with_limit_basis(parsed), FIXTURE["card"], CLAIM_IDS, MAX_STATEMENTS, TEXTS)


def evaluate(frozen: dict, checks: dict[str, dict]):
    record = dict(RECORD, obligations=frozen["obligations"], premises=frozen["premises"],
                  definitions=frozen["definitions"])
    return m_math.evaluate(record, checks)


def conformance_check(spec: dict, source: str) -> dict:
    result = spec_checks.reference_conformance(spec, source)
    return {"statement_id": "O16.a", "form": "conformance", "status": result["status"], "decided":
            result["status"] in ("discharged", "violated"), "evidence": result["evidence"],
            "counterexample": result["counterexample"], "artifact": "card-ca11b001-v1-fix6aa"}


# ----- 1. the fixer's own conformance premise is settled at M --------------------------------------------------------

def test_the_recorded_run_deferred_a1_to_test_design():
    recorded = FIXTURE["recorded_verdict"]
    assert [m["premise_id"] for m in recorded["scheduled_measurements"]] == ["A1"]
    assert recorded["scheduled_measurements"][0]["status"] == "deferred_to_test_design"
    assert set(DEPENDENT) <= set(recorded["conditional"])
    a1 = RECORD["premises"][0]
    assert a1["id"] == "A1" and a1["testable_by"] == "code"
    assert all("A1" not in o["discharges"] for o in RECORD["obligations"])


@pytest.mark.parametrize("reply", ["fixer_first", "fixer_second"])
def test_a1_names_the_frozen_definitions_and_is_settled_at_m(reply):
    a1 = next(p for p in FIXTURE[reply]["premises"] if p["id"] == "A1")
    ids = [d["id"] for d in FIXTURE[reply]["definitions"]]
    assert mc.named_definitions(a1, ids) == ["D1", "D2"]
    assert mc.settle_at(a1, ids) == "M"


@pytest.mark.parametrize("reply", ["fixer_first", "fixer_second"])
def test_both_recorded_fixer_replies_are_told_that_a1_needs_a_discharger_at_m(reply):
    problems = settlement(mc.fixer_problems(FIXTURE[reply], FIXTURE["card"], CLAIM_IDS, MAX_STATEMENTS, TEXTS))
    assert len(problems) == 1
    assert "no obligation settles it at stage M" in problems[0] and "conformance statement" in problems[0]
    assert "would never run" in problems[0]


def test_without_a_discharger_a1_is_refused_at_freeze_and_not_deferred():
    frozen = freeze(FIXTURE["fixer_second"])
    a1 = frozen["premises"][0]
    assert a1["subject"] == "frozen_definitions" and a1["settle_at"] == "M" and "no frozen obligation" in a1["refused"]
    validate("math_statement", dict(RECORD, premises=frozen["premises"]))
    _, ledger, records = evaluate(frozen, recorded_checks())
    entry = next(e for e in ledger if e["id"] == "A1")
    validate("assumption", entry)
    assert entry["status"] == "open" and entry["basis"].startswith("refused at freeze")
    assert not mathrecords.schedulable(entry) and not mathrecords.can_advance(entry, stage="M")
    assert not mathrecords.can_advance(entry, stage="C")
    assert mathrecords.scheduled_measurements(records, ledger) == []
    by_claim = {r["claim_id"]: r for r in records}
    for cid in DEPENDENT:
        assert by_claim[cid]["status"] == "conditional" and by_claim[cid]["open_premises"] == ["A1"]
    assert by_claim["P1"]["status"] == "observed"
    # With P2 and P4 aside, the refused premise keeps the card from passing M instead of advancing on a plan.
    _, codes = mathrecords._unproven_or_pass(records, ledger, stage="M")
    assert "unproven:A1" in codes


def test_a_conformance_obligation_discharges_a1_against_the_recorded_p1_reference():
    parsed = with_discharger(FIXTURE["fixer_second"])
    assert settlement(mc.fixer_problems(parsed, FIXTURE["card"], CLAIM_IDS, MAX_STATEMENTS, TEXTS)) == []
    frozen = freeze(parsed)
    a1 = frozen["premises"][0]
    assert a1["settle_at"] == "M" and "refused" not in a1
    discharger = next(o for o in frozen["obligations"] if o["id"] == "O16")
    assert discharger["claim_id"] is None and [s["form"] for s in discharger["statements"]] == ["conformance"]
    validate("math_statement", dict(RECORD, obligations=frozen["obligations"], premises=frozen["premises"],
                                    definitions=frozen["definitions"]))
    check = conformance_check(discharger["statements"][0]["spec"], reference_source())
    assert check["status"] == "discharged", check["evidence"]
    _, ledger, records = evaluate(frozen, dict(recorded_checks(), **{"O16.a": check}))
    entry = next(e for e in ledger if e["id"] == "A1")
    assert entry["status"] == "measured" and "O16.a" in entry["evidence"]
    by_claim = {r["claim_id"]: r for r in records}
    for cid in DEPENDENT:
        assert by_claim[cid]["status"] == "proven" and by_claim[cid]["open_premises"] == []
    _, codes = mathrecords._unproven_or_pass(records, ledger, stage="M")
    assert not any("A1" in code for code in codes)


@pytest.mark.parametrize("label,old,new", [
    ("no second residual", "hn = ops.np.add(g, u)", "hn = u"),
    ("an extra retained buffer", "'t': t + 1}", "'t': t + 1, 'x': ops.np.zeros(3)}"),
    ("work skipped for one token", "    h2, A2, C2 = round_step(",
     "    if int(token) == 1:\n        return {'A1': state['A1'], 'C1': state['C1'], 'A2': state['A2'], "
     "'C2': state['C2'], 'h': state['h'], 't': t + 1}\n    h2, A2, C2 = round_step("),
])
def test_a_reference_that_departs_from_the_equations_leaves_a1_open(label, old, new):
    source = reference_source()
    assert old in source
    frozen = freeze(with_discharger(FIXTURE["fixer_second"]))
    spec = next(o for o in frozen["obligations"] if o["id"] == "O16")["statements"][0]["spec"]
    check = conformance_check(spec, source.replace(old, new))
    assert check["status"] == "violated", label
    assert check["counterexample"]["stream"] in spec_checks.CONFORMANCE_STREAMS
    _, ledger, _ = evaluate(frozen, dict(recorded_checks(), **{"O16.a": check}))
    entry = next(e for e in ledger if e["id"] == "A1")
    assert entry["status"] == "open" and "did not hold" in entry["basis"]
    assert not mathrecords.can_advance(entry, stage="M") and not mathrecords.schedulable(entry)


def test_matching_counts_is_not_conformance():
    """The recorded P1 check matched every operation and state count; the cost check does not see an output layer
    that reads the wrong matrix, the conformance check does."""
    source = reference_source().replace("logits = state['h'] @ params['O']", "logits = state['h'] @ params['B']")
    spec = dict(CONFORMANCE, points=[{"n": 24, "d": 4, "V": 4}])
    assert conformance_check(spec, source)["status"] == "violated"


def test_a_profile_slot_that_lists_a1_does_not_settle_it():
    parsed = copy.deepcopy(FIXTURE["fixer_second"])
    next(o for o in parsed["obligations"] if o["claim_id"] == "P1")["discharges"] = ["A1"]
    assert len(settlement(mc.fixer_problems(parsed, FIXTURE["card"], CLAIM_IDS, MAX_STATEMENTS, TEXTS))) == 1
    assert freeze(parsed)["premises"][0]["refused"]


def test_a_world_premise_may_still_defer_to_test_design():
    parsed = copy.deepcopy(FIXTURE["fixer_second"])
    world = mf.premise("A2", "Natural text corpora have a Zipf exponent near 1 over the first ten thousand ranks.",
                       "prior", subject="world")
    parsed["premises"].append(world)
    next(o for o in parsed["obligations"] if o["claim_id"] == "complexity.memory")["premises"].append("A2")
    assert [p for p in mc.fixer_problems(parsed, FIXTURE["card"], CLAIM_IDS, MAX_STATEMENTS, TEXTS)
            if p.startswith("premise A2")] == []
    frozen = freeze(with_discharger(parsed))
    a2 = next(p for p in frozen["premises"] if p["id"] == "A2")
    assert a2["settle_at"] == "deferrable" and "refused" not in a2
    spec = next(o for o in frozen["obligations"] if o["id"] == "O16")["statements"][0]["spec"]
    _, ledger, records = evaluate(frozen, dict(recorded_checks(), **{"O16.a": conformance_check(
        spec, reference_source())}))
    entry = next(e for e in ledger if e["id"] == "A2")
    assert mathrecords.schedulable(entry) and mathrecords.can_advance(entry, stage="M")
    assert [m["premise_id"] for m in mathrecords.scheduled_measurements(records, ledger)] == ["A2"]


def test_a_world_label_on_a_premise_that_names_the_definitions_is_a_problem():
    parsed = copy.deepcopy(FIXTURE["fixer_second"])
    parsed["premises"][0]["subject"] = "world"
    problems = settlement(mc.fixer_problems(parsed, FIXTURE["card"], CLAIM_IDS, MAX_STATEMENTS, TEXTS))
    assert any("subject world but it names the frozen definitions D1, D2" in p for p in problems)
    assert freeze(parsed)["premises"][0]["subject"] == "frozen_definitions"


def test_a_conformance_reference_must_be_a_cost_statement():
    parsed = with_discharger(FIXTURE["fixer_second"], dict(CONFORMANCE, reference="O1.a"))
    assert any("conformance reference 'O1.a' is not a cost statement" in p
               for p in mc.fixer_problems(parsed, FIXTURE["card"], CLAIM_IDS, MAX_STATEMENTS, TEXTS))
    frozen = freeze(parsed)
    assert all(o["id"] != "O16" for o in frozen["obligations"])
    assert frozen["premises"][0]["refused"]


def test_the_fixer_schema_asks_for_the_subject_and_offers_the_conformance_form():
    schema = mc.statement_fixer_schema(CLAIM_IDS, 30, profile_required=True)
    premise = schema["properties"]["premises"]["items"]
    assert "subject" in premise["required"] and premise["properties"]["subject"]["enum"] == list(mathrecords.SUBJECTS)
    forms = schema["properties"]["obligations"]["items"]["properties"]["statements"]["items"]["properties"]
    assert "conformance" in forms["form"]["enum"]
    jsonschema.validate({"form": "conformance", "conformance": CONFORMANCE},
                        schema["properties"]["obligations"]["items"]["properties"]["statements"]["items"])
    assert "subject" not in r_refine._fixer_items()[1]["properties"]


def test_the_settlement_section_is_a_new_file_shown_to_the_fixer(tmp_path):
    section = load_section("premise_settlement")
    assert "frozen_definitions" in section and "conformance" in section and "would never run" in section
    h = Harness(tmp_path)
    h.m()
    call = next(c for c in h.llm.calls if c["prompt"] == "statement_fixer")
    assert section in call["user"]


# ----- 1b. the conformance audit runs in stage M ---------------------------------------------------------------------

FAKE_K1_EQUATIONS = '''
def expected(params, sizes, tokens):
    d = int(sizes["d"])
    K = np.zeros((0, d))
    W = np.zeros((0, d))
    out = []
    for token in tokens:
        x = params["E"][token]
        K = np.concatenate([K, (x @ params["Wk"]).reshape(1, -1)])
        W = np.concatenate([W, (x @ params["Wv"]).reshape(1, -1)])
        q = x @ params["Wq"]
        s = K @ q
        e = np.exp(s - s.max())
        out.append({"state": {"K": K, "V": W, "q": q}, "output": int((((e / e.sum()) @ W) @ params["O"]).argmax())})
    return out
'''


def fixer_with_conformance(card: dict, *, equations: str = FAKE_K1_EQUATIONS) -> dict:
    answer = cf.fixer_output("k1", card)
    answer["definitions"] = [{"id": "D1", "kind": "accounting", "text": "Parameters are E, Wq, Wk, Wv and O; the "
                              "state retains the key and value rows and the last query."}]
    answer["premises"] = [mf.premise("A1", "The cost reference executes the card's update and readout with D1's "
                                     "retained rows.", "other", subject="frozen_definitions")]
    p3 = next(o for o in answer["obligations"] if o["claim_id"] == "P3")
    p3["premises"] = ["A1"]
    # FIX-20: the points, the tolerance and the precision are host values; the fixer leaves them out.
    answer["obligations"].append(mf.obligation("O12", None, "premise A1", [{"form": "conformance", "conformance": {
        "reference": "O1.a", "equations": equations, "streams": ["repeated", "distinct", "random"]}}],
        discharges=["A1"], definitions=["D1"]))
    return answer


def test_stage_m_runs_the_conformance_audit_and_measures_the_premise(tmp_path):
    card = load_cards(("k1",))["k1"]
    h = Harness(tmp_path, fixer=fixer_with_conformance(card))
    h.m()
    audits = [e["data"] for e in h.ledger.iter_events("math_check") if e["data"]["purpose"] == "conformance"]
    assert [(a["statement_id"], a["status"]) for a in audits] == [("O12.a", "discharged")]
    verdict = h.verdict("M")
    ledger = json.loads((h.run_dir / "artifacts" / "assumption_ledger" / verdict["assumption_ledger"]).read_text())
    a1 = next(p for p in ledger["premises"] if p["id"] == "A1")
    assert a1["status"] == "measured" and a1["settle_at"] == "M"
    assert verdict["scheduled_measurements"] == []
    cost_call = next(c["user"] for c in h.llm.calls if c["prompt"] == "prover" and "O1.a" in
                     [i["statement_id"] for i in cf._shown(c["user"])])
    item = next(i for i in cf._shown(cost_call) if i["statement_id"] == "O1.a")
    assert item["conformance_audit"][0]["statement_id"] == "O12.a"
    assert load_section("conformance_audit") in cost_call


def test_a_failed_audit_keeps_the_premise_open_and_not_deferred(tmp_path):
    card = load_cards(("k1",))["k1"]
    wrong = FAKE_K1_EQUATIONS.replace('q = x @ params["Wq"]', 'q = x @ params["Wk"]')
    # K1 without its P4 protocols: with them the verdict is the failed composition learner (decision P4K), and a failed
    # check takes precedence over every unproven code. This test is about the open premise A1 after a failed audit.
    h = Harness(tmp_path, fixer=fixer_with_conformance(card, equations=wrong), protocols=False)
    h.m()
    audit = next(e["data"] for e in h.ledger.iter_events("math_check") if e["data"]["purpose"] == "conformance")
    assert audit["status"] == "violated"
    verdict = h.verdict("M")
    assert "unproven:A1" in verdict["kill_codes"] and verdict["scheduled_measurements"] == []
    assert next(c for c in verdict["claims"] if c["claim_id"] == "P3")["open_premises"] == ["A1"]


# ----- 2. P2 construction memory and the frozen contract -------------------------------------------------------------

def composition_statement() -> dict:
    return next(s for o in RECORD["obligations"] for s in o["statements"] if s["id"] == "O9.a")


def attempts(sid: str) -> list[dict]:
    return [c for c in FIXTURE["checks"] if c["statement_id"] == sid]


def beating(check: dict) -> dict:
    """A recorded attempt with its smallest size raised to 1.0, above the query-independent baseline (FIX-7: only an
    attempt that beats the baseline somewhere is shown as best_attempt; test_fix7_match.py covers the recorded ones,
    which never do)."""
    out = copy.deepcopy(check)
    out["details"]["results"][0]["accuracy"] = 1.0
    return out


def test_the_recorded_composition_attempts_no_longer_rank_the_modal_witness_as_best():
    # FIX-6 ranked attempt 4, the most-frequent-output witness, best; FIX-7 ranks against the baseline and shows none.
    statement = composition_statement()
    best = pm.BestAttempts()
    for check in attempts("O9.a"):
        best.note(statement, check["attempt"], check, check["text"])
    assert best.prompt_block("O9.a") is None
    ranking = best.ranking_block("O9.a")
    assert not any(a["beats_baseline"] for a in ranking["attempts"])
    assert [r["size"] for r in ranking["attempts"][0]["sizes"]] == [4, 16, 32]


def test_a_best_construction_block_has_sizes_state_and_source():
    statement = composition_statement()
    best = pm.BestAttempts()
    check = beating(attempts("O9.a")[3])
    best.note(statement, 4, check, check["text"])
    block = best.prompt_block("O9.a")
    assert block["attempt"] == 4 and block["held"] is False
    assert [r["size"] for r in block["sizes"]] == [4, 16, 32]
    assert [round(r["accuracy"], 5) for r in block["sizes"]] == [1.0, 0.19344, 0.10984]
    assert block["source"] == attempts("O9.a")[-1]["text"]
    assert all({"state_peak_words", "declared_state_words", "passed"} <= set(r) for r in block["sizes"])


def test_an_attempt_that_measured_nothing_does_not_replace_the_best():
    statement = composition_statement()
    best = pm.BestAttempts()
    first = beating(attempts("O9.a")[0])
    best.note(statement, 1, first, first["text"])
    best.note(statement, 2, {"status": "unknown", "decided": False, "details": {}}, "def init(layout): pass")
    assert best.prompt_block("O9.a")["attempt"] == 1


def test_the_frozen_contract_states_rounds_widths_and_the_hidden_width_per_size():
    statement = composition_statement()
    facts = pm.frozen_contract(statement, RECORD["profile"]["architecture"])
    assert facts["widths"] == statement["spec"]["widths"] and facts["architecture"] == {"hidden": "d",
                                                                                         "precision_bits": 64}
    assert [p["T"] for p in facts["sizes"]] == [4, 16, 32]
    for point in facts["sizes"]:
        assert point["rounds"] == 2 and point["d"] == 64 and point["hidden"] == 64 and point["precision_bits"] == 64
        assert point["positions"] == point["longest_stream"] == 5 * point["T"] + 3
        assert point["state_words"] == 4 * (5 * point["T"] + 3) * 64 + 65


def test_the_prover_sees_the_best_construction_and_the_contract_after_a_failure():
    statement = composition_statement()
    best = pm.BestAttempts()
    checks = [beating(attempts("O9.a")[0]), attempts("O9.a")[1]]
    for check in checks:
        best.note(statement, check["attempt"], check, check["text"])
    record = dict(RECORD)
    user = m_math.prover_user(FIXTURE["card"], record, [statement], {"O9.a": "status violated"}, None,
                              {"O9.a": checks[1]["text"]}, {"O9.a": ["attempt 1: ...", "attempt 2: ..."]}, best)
    item = cf._shown(user)[0]
    assert item["best_attempt"]["attempt"] == 1 and item["best_attempt"]["source"] == checks[0]["text"]
    assert item["best_attempt"]["sizes"][0]["accuracy"] == pytest.approx(1.0)
    assert [a["attempt"] for a in item["attempts_vs_baseline"]["attempts"]] == [1, 2]
    assert item["previous_source"] == checks[1]["text"]
    assert item["frozen_contract"]["sizes"][0]["rounds"] == 2
    assert load_section("construction_contract") in user


def test_the_contract_section_forbids_acting_on_an_insufficiency_claim_with_a_known_failing_witness():
    section = load_section("construction_contract")
    assert "THE FROZEN CONTRACT IS THE AUTHORITY" in section and "finding with evidence" in section
    assert "Do not act on that claim by submitting a witness you know cannot meet the threshold" in section
    assert "\u2014" not in section and "\u2014" not in load_section("premise_settlement")


# Exact at T = 4, short of the threshold at 16 and 32: above the query-independent baseline at T = 4 (FIX-7).
CAPPED_AT_4 = cf.CAPPED["composition"].replace("CAP = 2", "CAP = 4")


def test_stage_m_injects_the_construction_memory_on_a_failed_construction(tmp_path):
    script = cf.script("k1", load_cards(("k1",))["k1"])
    scripted = script["prover"]

    def prover(index, user):
        answer = scripted(index, user)
        for item, entry in zip(cf._shown(user), answer["attempts"]):
            spec = item["frozen_statement"]
            if item["form"] == "construction" and spec["task"] == "composition" and "best_attempt" not in item:
                entry["source"] = CAPPED_AT_4                        # a witness that fails at the larger sizes
        return answer

    script["prover"] = prover
    h = Harness(tmp_path, script=script)
    h.m()
    record = h.frozen()
    sid = next(s["id"] for o in record["obligations"] for s in o["statements"]
               if s["form"] == "construction" and s["spec"]["task"] == "composition")
    calls = [c["user"] for c in h.llm.calls if c["prompt"] == "prover" and sid in [i["statement_id"] for i in
                                                                                    cf._shown(c["user"])]]
    first, second = (next(i for i in cf._shown(user) if i["statement_id"] == sid) for user in calls[:2])
    assert "best_attempt" not in first and first["frozen_contract"]["task"] == "composition"
    assert second["best_attempt"]["attempt"] == 1 and second["best_attempt"]["source"] == CAPPED_AT_4
    assert second["best_attempt"]["sizes"] and all("accuracy" in r for r in second["best_attempt"]["sizes"])
    assert load_section("construction_contract") in calls[1]
    assert h.slot("M", "P2.composition")["status"] == "observed"
