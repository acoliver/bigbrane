"""Card falsifier tasks in the research loop: design, validation with one retry, S5 wiring, S6 mechanical verdicts.

Scripted LLM, real validation harness through LocalExecutor, FakeBench that decides the real rules with the
real falsifier_eval.decide on scripted scores."""
import asyncio
import json

import pytest

from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.schemas import validate
from hybrid.research.stages import falsify
from hybrid.research.stages.contracts import analyst_schema, falsifier_design_schema
from hybrid.research.stages.s6_verdict import combine, merge
from hybrid.tests.bench.falsifier_programs import MARKOV, SHIFT, variant
from hybrid.tests.research.builders import load_example
from hybrid.tests.research.test_orchestrator_fakes import (CAPABILITY_QUOTE, TEMPLATE, FakeBench, falsifier_design,
                                                           falsifier_rules, make_orchestrator, orch_config)

NONDETERMINISTIC = variant(SHIFT, "r = np.random.default_rng(seed)",
                           "import random\n    r = np.random.default_rng(seed + random.randrange(10**6))")


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261011T000000Z"


def run(orch):
    return asyncio.run(orch.run())


def only_card(orch):
    (card_id,) = orch.store.list_cards()
    return orch.store.get_card(card_id)


def artifact(orch, kind):
    recorded = [e["data"] for e in orch.ledger.iter_events("artifact") if e["data"]["kind"] == kind]
    return json.loads((orch.run_dir / recorded[-1]["path"]).read_text()) if recorded else None


def designer_calls(llm):
    return [c for c in llm.calls if c["prompt"] == "falsifier_designer"]


def ledger_entry(ledger, claim_id):
    return next(c for c in ledger["claims"] if c["claim_id"] == claim_id)


# ----- claim ids, quotes, schemas ---------------------------------------------------------------------

def test_claim_items_give_every_card_claim_a_stable_id():
    card = load_example()
    ids = [item["id"] for item in falsify.claim_items(card)]
    assert ids == ["complexity.time_per_token", "complexity.memory", "complexity.total_sequence",
                   "sample_efficiency.mechanism", "sample_efficiency.prediction", "capability.1", "capability.2",
                   "falsifier.1", "falsifier.2"]


def test_quotes_must_be_card_text():
    card = load_example()
    card["falsifiers"][0] = "Recall stays \u2265 0.9 at K = 256 pairs, which would refute the capacity analysis."
    assert falsify.quote_in_card(CAPABILITY_QUOTE, card)
    assert falsify.quote_in_card("recall stays >= 0.9 at K = 256   pairs", card)   # symbols and spacing
    assert not falsify.quote_in_card("accuracy reaches 0.99 on every task", card)
    assert not falsify.quote_in_card("  ", card)


def test_designer_and_analyst_schemas_limit_claim_ids_and_entrants():
    import jsonschema
    schema = falsifier_design_schema(["capability.1", "falsifier.1"], ["candidate", "ngram", "uniform", "unigram"])
    good = {"tasks": [falsifier_design()], "untestable": []}
    jsonschema.validate(good, schema)
    for broken in ({"rules": [dict(falsifier_rules()[0], claims=["capability.9"])]},
                   {"rules": [dict(falsifier_rules()[1], conditions=[dict(falsifier_rules()[1]["conditions"][0],
                                                                           minus="rwkv")])]},
                   {"task_id": "Bad Id"}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"tasks": [dict(falsifier_design(), **broken)], "untestable": []}, schema)
    analyst = analyst_schema(["capability.1"], ["iterated_shift"])
    entry = {"claim_id": "capability.1", "claim": "c", "status": "untested", "evidence": "e",
             "falsifier_tasks": ["iterated_shift"]}
    jsonschema.validate({"claims": [entry]}, analyst)
    for broken in ({"claim_id": None}, {"falsifier_tasks": None}, {"falsifier_tasks": ["other_task"]}):
        bad = {k: v for k, v in dict(entry, **broken).items() if v is not None}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"claims": [bad]}, analyst)
    without_tasks = analyst_schema(["capability.1"], [])
    jsonschema.validate({"claims": [dict(entry, falsifier_tasks=[])]}, without_tasks)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"claims": [entry]}, without_tasks)


# ----- the full line ----------------------------------------------------------------------------------

def test_designed_task_is_validated_run_and_decides_claims_in_s6(run_dir):
    bench = FakeBench()
    orch, llm = make_orchestrator(run_dir, bench=bench)
    assert run(orch)["reason"] == "complete"
    card = only_card(orch)
    # the designer saw the card and the claim ids, never the candidate code
    (call,) = designer_calls(llm)
    assert TEMPLATE.splitlines()[0] not in call["user"] and "deliberately simple bigram" not in call["user"]
    assert "capability.1:" in call["user"] and "falsifier.2:" in call["user"] and card["title"] in call["user"]
    assert call["role"] == "falsifier_designer"
    tasks = artifact(orch, "falsifier_tasks")
    assert [t["task_id"] for t in tasks["validated"]] == ["iterated_shift"] and tasks["dropped"] == []
    assert tasks["untestable"] == [{"claim_id": "complexity.memory", "why": "internal state size"}]
    (validation,) = [a for a in tasks["attempts"] if a["ok"]]
    executor_artifact = [e["data"] for e in orch.ledger.iter_events("artifact") if e["data"]["kind"] == "executor"]
    assert any(validation["job_id"] in a["path"] for a in executor_artifact)
    assert [[d.resolve() for d in dirs] for dirs in bench.falsifier_dirs] == \
        [[(run_dir / tasks["validated"][0]["dir"]).resolve()]]
    s5 = card["status"]["verdicts"]["S5"]
    assert validation["job_id"] in s5["by_calls"] and call["call_id"] in s5["by_calls"]
    assert any("falsifier iterated_shift rule for capability.1: supported" in r for r in s5["reasons"])
    summary = artifact(orch, "bench_result")
    assert summary["falsifiers"]["iterated_shift"]["rules"][0]["outcome"] == "supported"
    # S6: capability.1 decided by its rule (supported), falsifier.1 not observed (supported), the rest by the analyst
    ledger = artifact(orch, "claims_ledger")
    validate("claims_ledger", ledger)
    cap = ledger_entry(ledger, "capability.1")
    assert (cap["status"], cap["decided_by"], cap["analyst_status"]) == ("supported", "rule", "supported")
    assert cap["rules"][0]["quote"] == CAPABILITY_QUOTE and cap["rules"][0]["conditions"][0]["lhs"] == 0.95
    fals = ledger_entry(ledger, "falsifier.1")
    assert (fals["status"], fals["decided_by"], fals["analyst_status"]) == ("supported", "rule", None)
    other = ledger_entry(ledger, "capability.2")
    assert (other["status"], other["decided_by"]) == ("untested", "analyst")
    assert ledger["counts"] == {"supported": 2, "refuted": 0, "inconclusive": 0, "untested": 7}
    s6 = card["status"]["verdicts"]["S6"]["reasons"]
    assert s6[0].startswith("claims ledger: 2 supported, 0 refuted, 0 inconclusive, 7 untested; 2 decided by")
    analyst_user = [c["user"] for c in llm.calls if c["prompt"] == "analyst"][0]
    assert "FALSIFIER TASKS AND PRE-REGISTERED RULES" in analyst_user and "decision: supported" in analyst_user


def test_a_trained_candidate_that_misses_the_threshold_is_refuted(run_dir):
    bench = FakeBench(falsifier_scores={"candidate": (0.5, 1.0), "ngram": (0.45, 1.2)})
    orch, _ = make_orchestrator(run_dir, bench=bench)
    run(orch)
    ledger = artifact(orch, "claims_ledger")
    assert ledger_entry(ledger, "capability.1")["status"] == "refuted"
    falsifier = ledger_entry(ledger, "falsifier.1")      # beat the n-gram by only 0.05 < 0.1: falsifier observed
    assert falsifier["status"] == "refuted" and falsifier["rules"][0]["outcome"] == "refuted"
    assert [e for e in orch.ledger.iter_events("mutation_requested")] == []


def test_an_untrained_or_failed_candidate_leaves_the_claims_inconclusive(run_dir, tmp_path):
    orch, _ = make_orchestrator(run_dir, bench=FakeBench(falsifier_scores={"candidate": (0.0, 9.0),
                                                                           "ngram": (0.3, 2.0)}))
    run(orch)
    ledger = artifact(orch, "claims_ledger")
    for claim_id in ("capability.1", "falsifier.1"):
        entry = ledger_entry(ledger, claim_id)
        assert entry["status"] == "inconclusive" and "did not train" in entry["rules"][0]["reason"]
    assert ledger["counts"]["inconclusive"] == 2
    other = tmp_path / "second" / "run-20261011T000000Z"   # same run id, so the same lens
    orch, _ = make_orchestrator(other, bench=FakeBench(falsifier_scores={"candidate": None, "ngram": (0.3, 2.0)}))
    run(orch)
    assert ledger_entry(artifact(orch, "claims_ledger"), "capability.1")["status"] == "inconclusive"


def test_a_failed_task_is_retried_once_with_the_failure_quoted(run_dir):
    script = {"falsifier_designer": [{"tasks": [falsifier_design(NONDETERMINISTIC)], "untestable": []},
                                     {"tasks": [falsifier_design()], "untestable": []}]}
    orch, llm = make_orchestrator(run_dir, script=script)
    run(orch)
    first, second = designer_calls(llm)
    assert "not deterministic on replay" in second["user"] and "THESE TASKS FAILED VALIDATION" in second["user"]
    assert "random.randrange" in second["user"]
    tasks = artifact(orch, "falsifier_tasks")
    assert [t["task_id"] for t in tasks["validated"]] == ["iterated_shift"] and tasks["dropped"] == []
    assert [(a["round"], a["ok"], a["stage"]) for a in tasks["attempts"]] == [(0, False, "executor"),
                                                                             (1, True, "executor")]
    assert ledger_entry(artifact(orch, "claims_ledger"), "capability.1")["decided_by"] == "rule"


def test_a_task_that_fails_twice_is_dropped_and_s5_runs_the_standard_tasks(run_dir):
    bad_quote = falsifier_design(rules=[dict(falsifier_rules()[0], quote="the model is better than everything")])
    script = {"falsifier_designer": [{"tasks": [bad_quote, falsifier_design(NONDETERMINISTIC, task_id="second")],
                                      "untestable": []},
                                     {"tasks": [bad_quote, falsifier_design(MARKOV, task_id="second")],
                                      "untestable": []}]}
    bench = FakeBench()
    orch, llm = make_orchestrator(run_dir, script=script, bench=bench)
    run(orch)
    tasks = artifact(orch, "falsifier_tasks")
    assert len(designer_calls(llm)) == 2
    assert tasks["validated"] == []
    dropped = {d["task_id"]: d for d in tasks["dropped"]}
    assert set(dropped) == {"iterated_shift", "second"}
    assert "quote is not text from the card" in dropped["iterated_shift"]["problems"][0]
    host = [a for a in tasks["attempts"] if a["task_id"] == "iterated_shift"]
    assert all(a["stage"] == "host" and a["job_id"] is None for a in host)     # never sent to the executor
    assert any("group" in p or "rule 0" in p for p in dropped["second"]["problems"])   # MARKOV has no group m=2
    assert bench.falsifier_dirs == [[]]
    card = only_card(orch)
    assert card["status"]["verdicts"]["S5"]["verdict"] == "pass"
    ledger = artifact(orch, "claims_ledger")
    assert all(c["decided_by"] == "analyst" for c in ledger["claims"]) and ledger["falsifier_tasks"] == []


def test_falsifiers_can_be_switched_off(run_dir):
    bench = FakeBench()
    orch, llm = make_orchestrator(run_dir, bench=bench, cfg=orch_config(s5__falsifiers=False))
    run(orch)
    assert designer_calls(llm) == [] and bench.falsifier_dirs == [[]]
    assert artifact(orch, "falsifier_tasks") is None
    assert not any("falsifier" in r for r in only_card(orch)["status"]["verdicts"]["S5"]["reasons"])


def test_a_resumed_s5_reuses_the_validated_tasks(run_dir):
    orch, llm = make_orchestrator(run_dir)
    run(orch)
    card_id = only_card(orch)["id"]
    dirs, ids = asyncio.run(falsify.prepare(orch.ctx, card_id))
    assert len(designer_calls(llm)) == 1 and ids == []
    assert [d.name.split("-", 1)[1] for d in dirs] == ["iterated_shift"] and all(d.is_dir() for d in dirs)


# ----- S6 merging -------------------------------------------------------------------------------------

def test_rules_that_disagree_leave_a_claim_inconclusive():
    base = {"task_id": "t", "rule_index": 0, "reason": "r", "quote": "q", "conditions": [], "if_met": "supported"}
    assert combine([dict(base, outcome="supported"), dict(base, outcome="inconclusive")])[0] == "supported"
    status, note = combine([dict(base, outcome="supported"), dict(base, task_id="u", outcome="refuted")])
    assert status == "inconclusive" and "disagree" in note
    assert combine([dict(base, outcome="inconclusive")])[0] == "inconclusive"


def test_merge_fills_claims_the_analyst_skipped():
    card = load_example()
    claims = falsify.claim_items(card)
    ledger = merge("card-00000001", claims, {"claims": [{"claim_id": "falsifier.2", "claim": "x", "status": "refuted",
                                                          "evidence": "observed", "falsifier_tasks": []}]}, {})
    assert ledger["counts"] == {"supported": 0, "refuted": 1, "inconclusive": 0, "untested": 8}
    skipped = ledger_entry(ledger, "capability.1")
    assert skipped["analyst_status"] is None and skipped["decision_note"] == "the analyst gave no entry for this claim"


# ----- config -----------------------------------------------------------------------------------------

def test_s5_falsifier_settings_default_and_validate(tmp_path):
    shipped = ROOT / "hybrid" / "config" / "run.toml"
    cfg = load_run_config(shipped)
    assert cfg["s5"]["falsifiers"] is True and cfg["s5"]["falsifier_wall_s"] == 300
    old = tmp_path / "old.toml"
    old.write_text(shipped.read_text().replace("falsifiers = true\n", "").replace("falsifier_wall_s = 300\n", ""))
    assert load_run_config(old)["s5"]["falsifiers"] is True
    assert load_run_config(shipped, {"s5.falsifiers": False})["s5"]["falsifiers"] is False
    with pytest.raises(ValueError, match="s5.falsifiers must be true or false"):
        load_run_config(shipped, {"s5.falsifiers": "yes"})
    with pytest.raises(ValueError, match="falsifier_wall_s"):
        load_run_config(shipped, {"s5.falsifier_wall_s": 0})


def test_designer_routes_avoid_the_builder_model_family():
    import tomllib
    with open(ROOT / "hybrid" / "config" / "models.toml", "rb") as handle:
        routes = tomllib.load(handle)["routes"]

    def family(model):
        return "claude" if any(k in model for k in ("claude", "opus", "sonnet", "haiku")) else model.split("/")[0]

    for name, table in routes.items():
        builders = {family(e["model"]) for e in table["builder"]}
        designers = {family(e["model"]) for e in table["falsifier_designer"]}
        assert not builders & designers, name
