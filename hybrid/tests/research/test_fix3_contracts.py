"""FIX-3 regressions for stage M, built from the recorded calibration smoke runs/calibration-smoke-20261013T133146Z
(results/phase4-20261013T073748Z/FIX-3.md and k1-smoke-diag/REPORT.md).

Recorded fixtures in fixtures/calibration_smoke_20261013/ (sanitized by tmp/fix3/make_fixtures.py: ids, hashes,
receipts and paths dropped):

- k1_smoke.json: card k1's frozen definitions D1, D2, D5, D7 and statements O4.a (P1 cost), O5.a (copy learner),
  O7.a (P2.recall), O9.a and O9.b (P2.composition), O13.a (P4.composition), each with every recorded attempt: status,
  evidence, counterexample, per-size results and the submitted source.
- k2_smoke.json: card k2's D1, D2, D3, D5, D6 and statements O1.a (leading-term cost with a declared 9nd remainder),
  O5.a (P1 cost), O6.a (P2.copy), O9.a (composition) and O11.a (P4.copy) with their attempts. O1.a, O6.a, O11.a, D3
  and D5 were added for the findings of k2-smoke-diag/REPORT.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hybrid.math import constructions, cost, learnability, profile, spec_checks
from hybrid.math.runner import LIBRARY_CAPS
from hybrid.research.calibration import load_cards
from hybrid.research.stages import m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import load_prompt, load_section
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research.test_profile_stage import Harness

SMOKE = Path(__file__).parent / "fixtures" / "calibration_smoke_20261013"
K1 = json.loads((SMOKE / "k1_smoke.json").read_text(encoding="utf-8"))
K2 = json.loads((SMOKE / "k2_smoke.json").read_text(encoding="utf-8"))
SMALL_NS = (64, 128, 256, 512)


def spec_of(card: dict, sid: str) -> dict:
    return json.loads(json.dumps(card["statements"][sid]["spec"]))


def source_of(card: dict, sid: str, attempt: int) -> str:
    return next(a["source"] for a in card["attempts"][sid] if a["attempt"] == attempt)


def cost_claim(spec: dict, **changes) -> cost.CostClaim:
    fields = {k: spec.get(k) for k in ("ops", "state", "sizes", "exponent", "ops_remainder", "state_remainder")}
    return cost.CostClaim(**dict(fields, **changes))


def statement(sid: str, form: str, spec: dict) -> dict:
    return {"id": sid, "form": form, "spec": spec}


# ----- 1. host-generated task contracts ------------------------------------------------------------------------------

def test_the_composition_contract_states_shuffled_tables_and_the_real_stream_length():
    text = constructions.task_contract("composition", [4, 16, 32])
    assert "Pairs in the g table are shuffled" in text and "pairs in the f table are shuffled" in text
    assert "T=4: 23 tokens" in text and "T=16: 83 tokens" in text and "T=32: 163 tokens" in text
    assert "(exactly 5*T+3 at every size)" in text
    assert "A lookup must find a pair by its key token, not by its position" in text
    assert "'GSEP': 12, 'FSEP': 13, 'QSEP': 14, 'vocab': 15" in text


def test_the_binding_contract_states_two_chains_query_timing_and_the_size_field():
    text = constructions.task_contract("binding", [1, 4, 8])
    assert "Each instance has 2 chains" in text
    assert "hops=1: [1], hops=4: [4], hops=8: [8] variable steps" in text
    assert "The layout has no 'hops' field: read layout['size']" in text
    assert "hops=1: 11 tokens" in text and "(exactly 4*hops+7 at every size)" in text
    assert "The queries are the last variable of each chain, in random order, after QSEP" in text


def test_the_recall_contract_separates_the_pair_count_from_the_key_universe():
    text = constructions.task_contract("recall", [8, 32, 64], {"V": 4})
    assert "K is the number of pairs; n_keys is the key universe" in text
    assert "K=64: n_keys=128" in text and "K=64: 193 tokens" in text
    assert "pairs are in random key order" in text


def test_the_contract_is_measured_from_the_generator_so_it_follows_a_changed_generator(monkeypatch):
    def sorted_tables(lay, rng):
        T = lay["T"]
        g = {x: rng.randrange(T) for x in range(T)}
        f = {y: rng.randrange(T) for y in range(T)}
        tokens = [lay["GSEP"]] + [t for x in range(T) for t in (lay["x_base"] + x, lay["y_base"] + g[x])]
        tokens += [lay["FSEP"]] + [t for y in range(T) for t in (lay["y_base"] + y, lay["z_base"] + f[y])]
        tokens += [lay["QSEP"]]
        queries = []
        for x in range(T):
            tokens.append(lay["x_base"] + x)
            queries.append((len(tokens) - 1, lay["z_base"] + f[g[x]]))
        return constructions.Instance(tuple(tokens), tuple(queries))

    monkeypatch.setitem(constructions._GENERATORS, "composition", sorted_tables)
    text = constructions.task_contract("composition", [4])
    assert "Pairs in the g table are sorted by x" in text and "shuffled" not in text


def test_the_fixer_template_and_the_prover_get_the_contracts():
    template = profile.template_text(LIBRARY_CAPS)
    for task in profile.REQUIRED_SIZES:
        assert f"TASK CONTRACT {task} " in template
    assert "P1 COUNTING CONVENTION" in template and "token_store" in template
    o9a = statement("O9.a", "construction", spec_of(K1, "O9.a"))
    o13 = statement("O13.a", "learnability", spec_of(K1, "O13.a"))
    o4 = statement("O4.a", "cost", spec_of(K1, "O4.a"))
    text = m_math.task_contracts([o9a, o13, o4])
    assert "For O9.a: TASK CONTRACT composition" in text
    assert "measured over 500 instances at seed 0 (the instances the check runs)" in text
    assert "For O13.a: TASK CONTRACT composition" in text and "layout['vocab'] = 15" in text
    assert "P1 COUNTING CONVENTION" in text


def test_the_stage_sends_the_contracts_to_the_prover(tmp_path):
    h = Harness(tmp_path, "k1")
    h.m()
    users = [c["user"] for c in h.llm.calls if c["prompt"] == "prover"]
    assert any("TASK CONTRACT composition" in u and "TASK CONTRACT binding" in u for u in users)
    assert any("P1 COUNTING CONVENTION" in u for u in users if '"form": "cost"' in u)
    fixer = next(c["user"] for c in h.llm.calls if c["prompt"] == "statement_fixer")
    assert "TASK CONTRACT binding" in fixer and "P1 COUNTING CONVENTION" in fixer


# ----- 2. pre-freeze validation of declared capacities ---------------------------------------------------------------

def k1_store(sid: str, tokens: str) -> dict:
    """The k1 statement with D5's capacity and width written as a token_store (four address and content stores)."""
    spec = spec_of(K1, sid)
    name = constructions.SIZE_NAME[spec["task"]]
    return dict(spec, token_store={"arrays": 4, "tokens": tokens,
                                   "width": f"32*ceiling(log({name}+1)/log(2))"})


def test_the_recorded_composition_capacity_is_refused_before_freezing_with_the_real_length():
    problems = spec_checks.statement_problems("construction", k1_store("O9.a", "4*T+3"))
    assert any("token_store.tokens 4*T+3 = 19 at T=4, but the composition generator feeds 23 tokens there" in p
               and "5*T+3" in p for p in problems)
    assert any("T=32" in p and "163 tokens" in p for p in problems)


def test_the_corrected_capacity_still_exposes_stores_that_exceed_the_declared_state():
    problems = spec_checks.statement_problems("construction", k1_store("O9.a", "5*T+3"))
    assert not any("token_store.tokens" in p for p in problems)
    assert any("4 x 23 x 96 = 8832 words at T=4, above state_words" in p and "= 8300" in p for p in problems)


def test_the_recorded_binding_capacity_is_refused_with_the_real_length():
    problems = spec_checks.statement_problems("construction", k1_store("O9.b", "3*hops+3"))
    assert any("token_store.tokens 3*hops+3 = 6 at hops=1, but the binding generator feeds 11 tokens there" in p
               and "4*hops+7" in p for p in problems)


def test_a_consistent_declaration_passes_and_the_fixer_retry_carries_the_problem():
    spec = dict(spec_of(K1, "O9.a"), state_words="4*(5*T+3)*(32*ceiling(log(T+1)/log(2)))+10*96+2*(5*T+3)+3*T+3",
                token_store={"arrays": 4, "tokens": "5*T+3", "width": "32*ceiling(log(T+1)/log(2))"})
    assert spec_checks.statement_problems("construction", spec) == []
    bad = {"form": "construction", "construction": k1_store("O9.a", "4*T+3")}
    assert any("feeds 23 tokens" in p for p in mc.statement_problems(bad))


def test_the_fixer_schema_accepts_a_token_store():
    schema = mc.FORM_SCHEMAS["construction"]
    assert set(schema["properties"]["token_store"]["required"]) == {"arrays", "tokens", "width"}


# ----- 3. P1 accounting: the counter's convention is the one stated --------------------------------------------------

def test_the_counting_text_states_what_the_counter_counts():
    text = cost.counting_text()
    assert "a length-8 dot product counts 8 mac" in text
    assert "multiply then sum counts 15 arith" in text
    assert "init(sizes, ops) receives the sizes ['V', 'd']" in text and "n is never passed" in text
    assert "the probe state {'t': 3} measures 1" in text
    assert "Not in ops.np: expit, softmax" in text
    ops = cost.OpCounter()
    ops.np.dot(ops.np.ones(8), ops.np.ones(8))
    assert dict(ops.counts) == {"mac": 8}


def test_k2_reference_code_fails_its_frozen_scalar_dot_formula_and_passes_in_the_counter_units():
    spec, source = spec_of(K2, "O5.a"), source_of(K2, "O5.a", 4)
    assert "q multiplications and q-1 additions" in K2["definitions"]["D1"]["text"]
    recorded = K2["attempts"]["O5.a"][-1]["counterexample"]
    assert (recorded["check"], recorded["counted"], recorded["formula"]) == ("ops_exact", 44736, "74176")
    frozen = cost.verify_cost(source, cost_claim(spec), ns=SMALL_NS)
    assert frozen["status"] == "violated"
    ce = frozen["counterexample"]
    # The source probes for a sigmoid first; ops.np.sigmoid now exists and counts one unit per element, so the three
    # gates' expanded sigmoid (exp, add, divide: 3 units) costs 1: 2 units x 3 gates x d=8 x n=64 = 3072 fewer than
    # recorded. The MAC-counted dots still miss the frozen scalar-dot formula by a leading-order term.
    assert (ce["check"], ce["counted"], ce["formula"]) == ("ops_exact", 41664, "74176")
    assert recorded["counted"] - ce["counted"] == 2 * 3 * 8 * 64
    assert ce["counted_by_kind"]["mac"] == 34816 and set(frozen["failed_checks"]) == {"ops_exact", "exponent_d"}
    # The same source, the formula written in the counter's units: every exact check passes, nothing loosened.
    exact = cost.verify_cost(source, cost_claim(spec, ops="n*(8*d**2 + 13*d + d*V + V - 1)"), ns=SMALL_NS)
    assert exact["status"] == "discharged" and exact["tolerance"] == cost.DEFAULT_TOLERANCE
    feedback = m_math.failure_text(dict(check_from(frozen), form="cost"))
    assert "COUNTEREXAMPLE: ops_exact at {'V': 4, 'd': 8, 'n': 64}: counted operations 41664, formula 74176" in feedback
    assert "counted by kind {'mac': 34816" in feedback and "OTHER FAILING CHECKS" in feedback


def test_k2_scalar_dot_attempt_meets_its_frozen_formula_once_the_unit_sigmoid_exists():
    """The k2 report's intervention, now with the real counter: attempt 3 wrote scalar dots as D1 counts them and
    was 4608 over at n=64 only because sigmoid had to be expanded; with ops.np.sigmoid every exact check passes."""
    spec, source = spec_of(K2, "O5.a"), source_of(K2, "O5.a", 3)
    recorded = K2["attempts"]["O5.a"][2]["counterexample"]
    assert (recorded["check"], recorded["counted"], recorded["formula"]) == ("ops_exact", 78784, "74176")
    assert "ops.np.sigmoid" in source
    result = cost.verify_cost(source, cost_claim(spec), ns=SMALL_NS)
    assert result["status"] == "discharged", result["evidence"]
    first = result["measurements"][0]
    assert (first["point"], first["ops"], first["state_peak_words"]) == ({"V": 4, "d": 8, "n": 64}, 74176, 67)


def test_k1_p1_state_counter_word_is_reported_as_the_counterexample():
    spec, source = spec_of(K1, "O4.a"), source_of(K1, "O4.a", 4)
    result = cost.verify_cost(source, cost_claim(spec), ns=SMALL_NS)
    ce = result["counterexample"]
    assert (ce["check"], ce["counted"], ce["formula"], ce["difference"]) == ("state_exact", 4389, "4388", "1")
    assert list(result["failed_checks"]) == ["state_exact"]
    feedback = m_math.failure_text(dict(check_from(result), form="cost"))
    assert "state_exact at {'V': 4, 'd': 16, 'n': 64}: counted peak state words 4389, formula 4388 (difference 1)" \
        in feedback


def test_the_prompts_state_the_counter_units_and_drop_the_free_choice_of_units():
    fixer = load_prompt("statement_fixer")
    assert "A definition cannot change the counter's units" in fixer
    assert "unit-cost scalar operations, stored words" not in fixer
    section = load_section("profile_obligations")
    assert "units of the P1 COUNTING CONVENTION" in section and "`init` never receives n" in section
    assert "`init` does not receive n" in load_section("profile_prover")


def check_from(result: dict) -> dict:
    """A stage check record as run_check builds it from a printed cost report."""
    return {"status": result["status"], "evidence": result["evidence"], "counterexample": result["counterexample"],
            "reasons": [], "runs": [{"stdout_tail": json.dumps(result)[-2500:]}],
            "details": {"failed_checks": result.get("failed_checks")}}


# ----- 4. sandbox pre-check without an executor job -------------------------------------------------------------------

def test_the_recorded_str_learner_is_refused_on_the_host_with_the_line():
    source = source_of(K1, "O5.a", 4)
    recorded = K1["attempts"]["O5.a"][-1]
    assert "NameError: name 'str' is not defined" in recorded["evidence"]
    line = next(i for i, text in enumerate(source.splitlines(), 1) if "str(" in text)
    problems = m_math.source_problems("learnability", source)
    assert f"line {line}: builtin 'str' does not exist in the learnability sandbox; it would raise NameError when " \
           "that line runs" in problems
    assert any("RECIPE" in p for p in problems)


def test_an_absent_counted_function_is_refused_on_the_host_and_the_recorded_sigmoid_now_runs():
    source = source_of(K2, "O5.a", 1)
    assert "has no attribute 'sigmoid'" in K2["attempts"]["O5.a"][0]["output_tail"]
    # The recorded crash is gone: ops.np.sigmoid exists, so the first attempt is measured instead of crashing.
    assert cost.sandbox_problems(source, "cost") == []
    assert cost.verify_cost(source, cost_claim(spec_of(K2, "O5.a")), ns=SMALL_NS)["status"] in ("discharged",
                                                                                             "violated")
    # A function the counted namespace does not have is still refused without a job.
    assert "np = ops.np" in source and "np.sigmoid(" in source
    problems = cost.sandbox_problems(source.replace("np.sigmoid(", "np.expit("), "cost")
    assert any("ops.np has no 'expit'" in p for p in problems)
    # A probe inside try is allowed: the source catches the AttributeError itself.
    assert cost.sandbox_problems(source_of(K2, "O5.a", 4), "cost") == []


def test_sandbox_names_follow_the_checker_namespaces():
    assert cost.sandbox_problems("def run(g):\n    return str(g)\n", "construction") == []
    assert cost.sandbox_problems("def init(sizes, ops):\n    str = 3\n    return str, {}\n", "cost") == []
    assert any("'np'" in p for p in cost.sandbox_problems("def init(sizes, ops):\n    return np.zeros(2), {}\n",
                                                          "cost"))
    assert any("class statements" in p for p in cost.sandbox_problems("class A:\n    pass\n", "simulation"))
    assert any("'Exception'" in p for p in cost.sandbox_problems(
        "def run(g):\n    try:\n        return g\n    except Exception:\n        return 0\n", "simulation"))


def test_a_refused_source_costs_an_attempt_but_no_job_and_the_next_round_names_the_line(tmp_path):
    str_learner = cf.COPY_LEARNER.replace("    if not examples:\n", "    tag = str(seed)\n    if not examples:\n")

    def prover(index, user):
        attempts = []
        for item in cf._shown(user):
            spec = item["frozen_statement"]
            if item["form"] == "learnability" and spec["task"] == "copy":
                source = str_learner if "previous_attempt_failed" not in item else cf.COPY_LEARNER
            else:
                source = _scripted_source(item)
            attempts.append({"statement_id": item["statement_id"], "proof": None, "source": source, "notes": ""})
        return {"attempts": attempts}

    script = cf.script("k1", load_cards(("k1",))["k1"])
    script["prover"] = prover
    h = Harness(tmp_path, "k1", script=script)
    h.m()
    copy_jobs = [j for j in h.executor.jobs if _request(j).get("kind", "").startswith("learnability")
                 and _request(j)["spec"]["task"] == "copy"]
    # One probe and the full check with its replay, all from the second attempt: the refused first attempt ran none.
    assert all("str(seed)" not in _request(j)["source"] for j in copy_jobs) and copy_jobs
    second = [json.loads(c["user"].split("FROZEN STATEMENTS:\n", 1)[1].split("\n\n", 1)[0])
              for c in h.llm.calls if c["prompt"] == "prover" and "previous_attempt_failed" in c["user"]
              and '"task": "copy"' in c["user"] and '"form": "learnability"' in c["user"]]
    item = next(i for batch in second for i in batch if i["form"] == "learnability"
                and i["frozen_statement"]["task"] == "copy")
    assert "refused before any executor job" in item["previous_attempt_failed"]
    assert "builtin 'str' does not exist in the learnability sandbox" in item["previous_attempt_failed"]
    assert item["previous_source"] == str_learner
    assert item["attempt_history"][0].startswith("attempt 1: unknown (prover fault): refused before any executor job")
    assert m_math.holds(next(c for c in _checks(h) if c["statement_id"] == item["statement_id"]
                             and c["attempt"] == 2))


def _scripted_source(item: dict) -> str:
    spec = item["frozen_statement"]
    if item["form"] == "cost":
        return cf.COST["k1"][0]
    if item["form"] == "construction":
        return cf.OPEN[spec["task"]]
    return cf.LEARNERS[spec["task"]]


def _request(job) -> dict:
    return json.loads((Path(job.input_dir) / "request.json").read_text())


def _checks(h) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((h.run_dir / "artifacts" / "math_check").glob("*.json"))]


# ----- 5. the prover's training recipe, applied to the controls as well ----------------------------------------------

COPY_SPEC = {"task": "copy", "size": 3, "examples": 40, "seeds": [0, 1, 2], "test_count": 40}


def test_the_template_leaves_the_recipe_to_the_prover_and_keeps_the_fixed_terms():
    text = profile.template_text(LIBRARY_CAPS)
    assert "They do not fix the training recipe" in text
    assert "the same learn and the same RECIPE to the true data and to both controls" in text
    assert "fewer than 3 seeds, lowering the threshold below 0.99 or dropping a control" in text
    assert "optimizer schedule you freeze in a definition" not in text
    assert "Do not freeze an optimizer schedule the card does not state" in load_section("profile_obligations")
    assert "RECIPE" in load_prompt("prover") and "RECIPE" in load_section("profile_prover")
    assert "training recipe" in load_section("profile_match")


def test_the_recorded_k1_frozen_recipe_is_what_the_template_now_forbids():
    d7 = K1["definitions"]["D7"]["text"]
    assert "learning rate 0.01, 50 full-batch epochs" in d7
    assert not learnability.recipe_problems(cf.COPY_LEARNER)
    assert learnability.recipe_problems(source_of(K1, "O13.a", 4))


def test_the_recipe_is_recorded_and_the_same_for_every_seed():
    result = learnability.run_learnability(COPY_SPEC, cf.COPY_LEARNER)
    assert result["status"] == "discharged"
    recipe = json.loads(cf.COPY_LEARNER.split("RECIPE = ", 1)[1].split("\n", 1)[0])
    assert result["recipe"] == recipe and all(r["recipe"] == recipe for r in result["results"])
    assert "applied to the true data and both controls" in result["evidence"]


def test_a_learner_that_switches_recipe_for_a_control_is_stopped():
    sneaky = cf.COPY_LEARNER.replace("    if not examples:\n        return {\"offset\": 0}\n",
                                     "    if not examples:\n        RECIPE[\"epochs\"] = 0\n"
                                     "        return {\"offset\": 0}\n")
    assert sneaky != cf.COPY_LEARNER
    result = learnability.run_learnability(COPY_SPEC, sneaky)
    assert result["status"] == "unknown" and result["prover_fault"]
    assert "the training recipe changed during the empty-data control training" in result["evidence"]


def test_a_learner_without_a_recipe_is_refused():
    plain = "\n".join(line for line in cf.COPY_LEARNER.splitlines() if not line.startswith("RECIPE"))
    result = learnability.run_learnability(COPY_SPEC, plain)
    assert result["status"] == "unknown" and "RECIPE = {...} at module level" in result["evidence"]


# ----- 6. feedback: counterexample, crash line and size, previous source, history ------------------------------------

def test_the_recorded_composition_failure_reaches_the_prover_with_an_input_and_expected_against_got():
    spec = dict(spec_of(K1, "O9.a"), count=20)
    result = spec_checks.audited_construction(spec, source_of(K1, "O9.a", 4))
    assert result["status"] == "violated"
    error = result["counterexample"]["first_error"]
    assert error["tokens"] and error["tokens_total"] == 23 and error["expected"] != error["predicted"]
    check = {"form": "construction", "status": result["status"], "evidence": result["evidence"],
             "counterexample": result["counterexample"], "reasons": [], "runs": [], "details": {}}
    feedback = m_math.failure_text(check)
    assert "COUNTEREXAMPLE: composition at size 4: accuracy" in feedback
    assert f"predict returned {error['predicted']} but the answer is {error['expected']}; input tokens [" in feedback


def test_the_recorded_recall_assert_reaches_the_prover_with_its_size_and_line():
    recorded = K1["attempts"]["O7.a"][-1]
    assert recorded["evidence"] == "construction failed: AssertionError: "
    source = source_of(K1, "O7.a", 4)
    result = spec_checks.audited_construction(dict(spec_of(K1, "O7.a"), count=3), source)
    assert result["status"] == "unknown" and result["prover_fault"]
    line = result["crash"]["raised_at"]
    number = int(line.split()[1])
    assert source.splitlines()[number - 1].strip().startswith("assert")
    assert result["evidence"].startswith("construction failed at K=64, instance 0: AssertionError (no message); "
                                         f"raised at {line}")


def test_a_learner_crash_names_the_seed_the_phase_and_the_line():
    broken = cf.COPY_LEARNER.replace("    best, best_score = 0, -1\n", "    best, best_score = 0, -1\n    1 / 0\n")
    result = learnability.run_learnability(COPY_SPEC, broken)
    assert result["status"] == "unknown" and result["prover_fault"]
    assert result["evidence"].startswith("learner failed at seed 0 during training on the true examples: "
                                         "ZeroDivisionError: division by zero; raised at line")
    assert "in learn(): 1 / 0" in result["evidence"]


def test_the_attempt_history_of_k1_p1_keeps_both_halves_of_the_oscillation():
    lines = []
    for attempt in K1["attempts"]["O4.a"]:
        check = {"form": "cost", "status": attempt["status"], "counterexample": attempt["counterexample"],
                 "evidence": attempt["evidence"], "details": {}, "prover_fault": attempt["prover_fault"]}
        lines.append(m_math.attempt_line(attempt["attempt"], check))
    assert lines[0].startswith("attempt 1: violated: ops_exact at {'V': 4, 'd': 16, 'n': 64}: counted 357696 vs "
                               "formula 672768")
    assert "state_exact" in lines[1] and "ops_exact" in lines[2] and "state_exact at" in lines[3]
    o4 = statement("O4.a", "cost", spec_of(K1, "O4.a"))
    record = {"obligations": [{"id": "O4", "claim_id": "P1", "claim_quote": "exact cost", "definitions": ["D1"],
                               "statements": [o4]}],
              "definitions": [{"id": "D1", "kind": "accounting", "text": K1["definitions"]["D1"]["text"]}]}
    user = m_math.prover_user({"id": "card-k1", "version": 1, "title": "k1"}, record, [o4],
                              {"O4.a": "status violated"}, LIBRARY_CAPS, {"O4.a": source_of(K1, "O4.a", 3)},
                              {"O4.a": lines[:3]})
    shown = json.loads(user.split("FROZEN STATEMENTS:\n", 1)[1].split("\n\n", 1)[0])[0]
    assert shown["attempt_history"] == lines[:3] and shown["previous_source"] == source_of(K1, "O4.a", 3)


def test_a_later_crash_keeps_the_earlier_measurement_in_the_history(tmp_path):
    crashing = cf.OPEN["composition"].replace("def init(layout):\n",
                                              "def init(layout):\n    assert layout['size'] < 16\n", 1)
    assert crashing != cf.OPEN["composition"]
    sent = {}

    def prover(index, user):
        attempts = []
        for item in cf._shown(user):
            sid, spec = item["statement_id"], item["frozen_statement"]
            n = sent[sid] = sent.get(sid, 0) + 1
            if item["form"] == "construction" and spec["task"] == "composition":
                source = cf.CAPPED["composition"] if n == 1 else crashing
            else:
                source = _scripted_source(item)
            attempts.append({"statement_id": sid, "proof": None, "source": source, "notes": ""})
        return {"attempts": attempts}

    script = cf.script("k1", load_cards(("k1",))["k1"])
    script["prover"] = prover
    h = Harness(tmp_path, "k1", script=script)
    h.ctx.cfg["math"]["max_prover_attempts"] = 3
    h.m()
    third = [json.loads(c["user"].split("FROZEN STATEMENTS:\n", 1)[1].split("\n\n", 1)[0])
             for c in h.llm.calls if c["prompt"] == "prover" and '"task": "composition"' in c["user"]]
    item = [i for batch in third for i in batch if i["form"] == "construction"
            and i["frozen_statement"]["task"] == "composition"][-1]
    history = item["attempt_history"]
    assert len(history) == 2 and history[0].startswith("attempt 1: violated: size 4 accuracy")
    assert history[1].startswith("attempt 2: unknown (prover fault): construction failed at T=16, instance 0: "
                                 "AssertionError (no message); raised at line")
    assert "assert layout['size'] < 16" in item["previous_attempt_failed"]
    assert item["previous_source"] == crashing


def test_the_job_driver_reports_where_a_cost_source_raised(tmp_path):
    from hybrid.math.runner import MathRunner
    from hybrid.tests.research import math_fakes as mf
    source = ("def init(sizes, ops):\n    return {}, {}\n"
              "def update(params, state, token, ops):\n    return state['missing']\n"
              "def predict(params, state, ops):\n    return 0\n")
    runner = MathRunner(cf.FastExecutor(tmp_path / "exec"), mf.IMAGE, tmp_path / "work")
    record = runner.check_library({"kind": "cost", "source": source,
                                   "claim": {"ops": "n", "state": "1", "sizes": {}, "exponent": 1},
                                   "ns": [8, 16, 32]})
    assert record["status"] == "unknown"
    printed = record["runs"][0]["printed"]
    assert printed["prover_fault"] and printed["raised_at"] == "line 4 in update(): return state['missing']"
    assert "KeyError: 'missing'; raised at line 4 in update()" in printed["evidence"]


@pytest.mark.parametrize("task", sorted(constructions.SIZE_NAME))
def test_every_task_contract_measures_lengths_that_match_the_instances(task):
    sizes = profile.REQUIRED_SIZES[task]
    text = constructions.task_contract(task, sizes, count=30, seed=7)
    for size in sizes:
        low, high = constructions.stream_lengths(task, size, 30, 7)
        shown = str(low) if low == high else f"{low} to {high}"
        assert f"{constructions.SIZE_NAME[task]}={size}: {shown} tokens" in text


# ----- 7. the k2 diagnosis: unit sigmoid, declared remainders, self-fed copy, learner vocabulary ----------------------
# results/phase4-20261013T073748Z/k2-smoke-diag/REPORT.md, recommendations 1, 2, 3 and 6.

def test_ops_np_sigmoid_is_sigma_of_x_at_one_unit_per_element():
    ops = cost.OpCounter()
    out = ops.np.sigmoid(ops.np.array([-2.0, 0.0, 2.0, 800.0, -800.0]))
    values = [float(out[i]) for i in range(5)]
    assert values[1] == 0.5 and values[2] == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert values[0] == pytest.approx(1 - values[2]) and values[3] == 1.0 and values[4] == 0.0
    assert dict(ops.counts) == {"transcendental": 5}


def test_the_counting_text_states_the_unit_sigmoid_and_its_sign():
    text = cost.counting_text()
    assert ("ops.np.sigmoid(x) is the logistic sigma(x) = 1/(1+exp(-x)) and counts one transcendental per element, "
            "as tanh (8 transcendental on length 8; sigma(2) = 0.8808)") in text
    assert "a definition that charges a sigmoid one operation must be implemented with ops.np.sigmoid" in text
    assert "costs 24 arith + 8 transcendental on length 8 (negative, exp, add, divide)" in text
    assert "1/(1+exp(x)) is sigma(-x) (0.1192 at x = 2), not sigma(x)" in text
    for prompt in (load_section("profile_prover"), load_prompt("prover")):
        assert "1/(1+exp(x)) is sigma(-x)" in prompt and "ops.np.sigmoid" in prompt
    assert "`ops.np.sigmoid` for the logistic sigma(x) = 1/(1+exp(-x))" in load_section("profile_obligations")
    assert "(`ops.np.tanh`, `ops.np.sigmoid`)" in load_prompt("statement_fixer")


O1 = spec_of(K2, "O1.a")


def test_the_recorded_k2_remainder_must_name_where_it_is_lower_order_before_freezing():
    assert (O1["ops"], O1["ops_remainder"]) == ("n*(16*d**2 + 2*d*V)", "9*n*d")
    problems = spec_checks.statement_problems("cost", O1)
    assert problems == ["cost: ops_remainder 9*n*d needs ops_remainder_lower_in: the sizes (from ['n', 'V', 'd']) in "
                        "which it is of strictly lower order than the ops formula, stated before freezing"]
    assert any("needs ops_remainder_lower_in" in p for p in mc.statement_problems({"form": "cost", "cost": O1}))
    assert spec_checks.statement_problems("cost", dict(O1, ops_remainder_lower_in=["d"])) == []
    # 9nd is lower order in d at fixed V, but as linear in n as the formula: it cannot be declared lower order in n.
    in_n = spec_checks.statement_problems("cost", dict(O1, ops_remainder_lower_in=["n"]))
    assert any("grows as n^1, not slower than the ops formula's n^1" in p and "not of lower order in n" in p
               for p in in_n)
    with pytest.raises(ValueError, match="needs ops_remainder_lower_in"):
        cost.verify_cost(source_of(K2, "O5.a", 3), cost_claim(O1), ns=SMALL_NS)


def test_a_remainder_cannot_absorb_leading_work_or_be_negative():
    # The MAC shortfall of the k2 reference code is 8nd^2: same order in d as the formula, refused.
    leading = cost.remainder_problems(cost_claim(O1, ops_remainder="8*n*d**2", ops_remainder_lower_in=["d"]))
    assert any("grows as d^2, not slower than the ops formula's d^2" in p for p in leading)
    large = cost.remainder_problems(cost_claim(O1, ops_remainder="20*n*d**2", ops_remainder_lower_in=["V"]))
    assert any("not below the ops formula's 69632: a remainder that large can absorb leading work" in p for p in large)
    negative = cost.remainder_problems(cost_claim(O1, ops_remainder="-n", ops_remainder_lower_in=["d"]))
    assert any("cannot be negative" in p for p in negative)
    stray = cost.remainder_problems(cost_claim(O1, ops_remainder=None, ops_remainder_lower_in=["d"]))
    assert stray == ["ops_remainder_lower_in is set but no ops_remainder is declared"]


def test_declared_lower_order_work_passes_and_missing_leading_work_does_not():
    claim = cost_claim(O1, ops_remainder_lower_in=["d"])
    # Attempt 3 counts n*(16d^2 + 9d + 2dV - 1) with scalar dots and the unit sigmoid: within 9nd of the leading term.
    scalar = cost.verify_cost(source_of(K2, "O5.a", 3), claim, ns=SMALL_NS)
    assert scalar["status"] == "discharged", scalar["evidence"]
    assert all("formula_band" in s for s in scalar["slopes"])
    # Attempt 4's MAC dots are 8nd^2 short; the declared 9nd cannot cover it.
    mac = cost.verify_cost(source_of(K2, "O5.a", 4), claim, ns=SMALL_NS)
    ce = mac["counterexample"]
    assert (ce["check"], ce["counted"], ce["formula"], ce["difference"]) == ("ops_exact", 41664, "69632", "-27968")
    assert 27968 > 9 * 64 * 8


SLOPED = ("def init(sizes, ops):\n    d = sizes['d']\n    return {'W': ops.np.eye(d)}, {'y': ops.np.ones(d)}\n"
          "def update(params, state, token, ops):\n    return {'y': ops.np.add(params['W'] @ state['y'], 0.0)}\n"
          "def predict(params, state, ops):\n    return 0\n")


def test_the_slope_band_admits_declared_lower_order_work_that_a_bare_slope_would_reject():
    """n*(d^2 + d) counted against n*d^2 with remainder n*d: the two-point d slope over d = 2, 4 is
    log(20/6)/log(2) = 1.737, outside 2 +- 0.05, but inside the band [1, 3.32] of slopes that counts within the
    remainder can have, while the exact checks still hold pointwise."""
    base = dict(ops="n*d**2", state="d", sizes={"d": [2, 4]}, exponent=1)
    ns = (16, 32, 64, 128)
    loose = cost.verify_cost(SLOPED, cost.CostClaim(**base, ops_remainder="n*d", ops_remainder_lower_in=["d"]), ns=ns)
    assert loose["status"] == "discharged", loose["evidence"]
    in_d = next(s for s in loose["slopes"] if s["parameter"] == "d")
    assert (in_d["measured"], in_d["formula"], in_d["formula_band"]) == (1.737, 2.0, [1.0, 3.3219])
    bare = cost.verify_cost(SLOPED, cost.CostClaim(**base), ns=ns)
    assert {"ops_exact", "exponent_d"} <= set(bare["failed_checks"])
    xs, f = [2, 4, 8], [4.0, 16.0, 64.0]
    assert cost.slope_band(xs, f, [0, 0, 0]) == pytest.approx((2.0, 2.0))
    assert cost.slope_band(xs, f, [0, 0, 0])[0] == pytest.approx(cost.loglog_slope(xs, f))


def test_copy_feeds_the_mechanism_its_own_predictions_after_the_separator():
    assert "never unrevealed target symbols" in K2["definitions"]["D5"]["text"]
    reads = []

    def update(state, token):
        reads.append(token)
        return state

    mechanism = {"init": lambda layout: {}, "update": update, "predict": lambda state: 3}
    lay = constructions.layout("copy", 5, {"D": 4})
    instance = constructions.generate("copy", 5, 1, 0, {"D": 4})[0]
    right, _, error = constructions.run_instance(mechanism, lay, instance)
    assert reads[:6] == list(instance.tokens[:6]) and instance.tokens[5] == lay["SEP"]
    assert reads[6:] == [3, 3, 3, 3]
    # The instance still stores the true continuation, which P4 training examples carry.
    assert list(instance.tokens[6:]) == list(instance.tokens[:4]) and reads[6:] != list(instance.tokens[6:])
    # Other tasks read every token as stored.
    reads.clear()
    recall = constructions.generate("recall", 3, 1, 0, None)[0]
    constructions.run_instance(mechanism, constructions.layout("recall", 3), recall)
    assert reads == list(recall.tokens)
    text = constructions.task_contract("copy", [8], {"D": 4})
    assert "the harness feeds, after SEP, the mechanism's own previous prediction instead (self-fed continuation, " \
           "no teacher forcing)" in text
    assert "self-fed continuation" in load_section("profile_prover") and "self-fed continuation" in load_prompt("prover")


def test_the_recorded_k2_copy_construction_scores_the_same_when_self_fed():
    recorded = K2["attempts"]["O6.a"][-1]
    assert [r["accuracy"] for r in recorded["results"]] == [1.0, 0.0, 0.0]
    result = constructions.run_construction(source_of(K2, "O6.a", 4), "copy", [8, 32], count=20, params={"D": 4})
    assert [r["accuracy"] for r in result["results"]] == [1.0, 0.0]
    first, then = result["results"][1]["first_error"], recorded["results"][1]["first_error"]
    assert (first["position"], first["expected"], first["predicted"]) == (then["position"], then["expected"],
                                                                            then["predicted"]) == (42, 1, 0)


def test_the_recorded_k2_learner_sized_its_vocabulary_from_the_largest_integer():
    spec, source = spec_of(K2, "O11.a"), source_of(K2, "O11.a", 4)
    namespace = {"np": np}
    exec(compile(source, "<recorded>", "exec"), namespace)
    lay = constructions.layout("copy", spec["size"], spec["params"])
    found = []
    namespace["collect"](lay, found)
    assert "nv = mx + 1" in source and max(found) + 1 == 6 and lay["vocab"] == 5
    contract = learnability.task_contract(spec)
    assert "the vocabulary is layout['vocab'] = 5 token ids (0..4); size it from the layout, not from the largest id" \
        in contract
    for prompt in (load_section("profile_prover"), load_prompt("prover")):
        assert "Size the output layer from `layout['vocab']`" in prompt
    # The recorded learner has no RECIPE either, so it is refused on the host before any job.
    assert any("RECIPE" in p for p in m_math.source_problems("learnability", source))
