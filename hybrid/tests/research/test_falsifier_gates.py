"""Gates on card falsifier tasks (results/falsifier-20261011/recall-investigation.md, section 5).

The regression fixture is the assoc_recall_capacity task and bench scores of the plumbing run
runs/plumb-20261011T012002Z (card-537e161c): both of its rules were decided on a candidate that learned
nothing. Under the gates, the recorded design is rejected at design time and the recorded scores decide
nothing."""
import asyncio
import json

import pytest

from hybrid.bench import falsifier as bench_falsifier
from hybrid.bench import falsifier_eval as FE
from hybrid.research.stages import falsify
from hybrid.research.stages.contracts import falsifier_design_schema
from hybrid.research.stages.s4_construct import builder_prompt
from hybrid.research.stages.s6_verdict import claim_line, falsified_by, merge
from hybrid.tests.bench.falsifier_programs import SHIFT, variant
from hybrid.tests.research.builders import load_example
from hybrid.tests.research.test_falsifiers import artifact, designer_calls, only_card, run
from hybrid.tests.research.test_orchestrator_fakes import (TEMPLATE, FakeBench, falsifier_design, falsifier_rules,
                                                           make_orchestrator)
from hybrid.tests.research.test_orchestrator import events, verdicts
from hybrid.research.orchestrator import ROOT

FIXTURES = ROOT / "hybrid" / "tests" / "research" / "fixtures" / "assoc_recall"
BASELINES = ["ngram", "gru", "transformer", "ssm", "linear_attention"]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261011T000000Z"


def recorded():
    card = json.loads((FIXTURES / "card.json").read_text())
    design = json.loads((FIXTURES / "design.json").read_text())
    scores = json.loads((FIXTURES / "bench_scores.json").read_text())
    return card, dict(design, protocol_params=[]), scores


def recorded_scores(scores):
    out = {name: entry["metrics"] for name, entry in scores["entrants"].items()}
    return {**out, **scores["references"]}


# ----- regression: the recorded assoc_recall_capacity task ---------------------------------------------

def test_the_recorded_design_is_rejected_at_design_time_with_the_stated_texts():
    card, design, _ = recorded()
    problems = falsify.host_problems(design, card, BASELINES)
    text = "\n".join(problems)
    for phrase in (
            "rule 1 mixes falsifier claims with other claims; a met falsifier refutes the card while a met "
            "capability supports it, so write separate rules",
            "rule 1 decides falsifier.3 with if_met 'supported'; a falsifier rule states the falsifying observation "
            "from the falsifier's own text and if_met must be 'refuted'",
            "rule 1 quote is not text from falsifier.3",
            "rule 0 condition 0 value 0.5 is not a number in the quote",
            "rule 0 condition 0 subtracts unigram; the card states absolute thresholds, so subtracting a trivial "
            "reference changes them",
            "rule 1 can support the card by a low score; name a control_rule in this task where the candidate must "
            "first show the capability"):
        assert phrase in text, phrase
    assert "rule 1 quote is not text from capability.2" not in text      # the quote is capability 2's own sentence


def test_the_recorded_scores_decide_nothing_under_the_trained_gate():
    _, design, scores = recorded()
    table = recorded_scores(scores)
    for index, rule in enumerate(design["rules"]):
        decision = FE.decide(rule, table, design["rules"])
        assert decision["outcome"] == "inconclusive", (index, decision["reason"])
        assert "did not train on this task" in decision["reason"]
        assert decision["training"]["kl_gain_over_best_reference"] < 0
        assert decision["training"]["acc_gain_over_best_reference"] <= 0.0
    # the uniform predictor in the candidate's place passes the old unigram-only check and fails the new one
    uniform = scores["references"]["uniform"]
    assert not FE.trained(uniform, scores["references"])["trained"]
    assert uniform["kl"] - scores["references"]["unigram"]["kl"] < 0


def test_the_recorded_candidate_is_blocked_on_length_and_on_d_even_before_training_is_considered():
    _, design, scores = recorded()
    table = recorded_scores(scores)
    entry = {"max_train_len": 148, "protocol_params": [{"name": "d", "card": 64, "task": 64, "model_side": True}]}
    old_describe = {"params": 929, "state_floats_per_sequence": 2048, "notes": "d=32, M=4"}
    assert "no integer train_max_len" in FE.task_blocked(entry, old_describe)
    truncating = {"hyper": {"d": 32, "M": 4}, "train_max_len": 128}
    assert FE.task_blocked(entry, truncating) == (
        "the task has training sequences of 148 tokens but the candidate trains on at most 128 tokens, so the "
        "answers at the end of longer sequences are never trained")
    full = {"hyper": {"d": 32, "M": 4}, "train_max_len": 2048}
    assert FE.task_blocked(entry, full) == "the protocol needs d = 64 and the candidate was built with d = 32"
    assert FE.task_blocked(entry, {"hyper": {"M": 4}, "train_max_len": 2048}).startswith(
        "the protocol needs d = 64 and the candidate's describe()['hyper'] does not report an integer d")
    assert FE.task_blocked(entry, {"hyper": {"d": 64}, "train_max_len": 2048}) is None
    reason = FE.task_blocked(entry, full)
    for rule in design["rules"]:
        decision = FE.decide(rule, table, design["rules"], reason)
        assert decision["outcome"] == "inconclusive" and decision["reason"] == reason


def test_a_well_formed_replacement_task_is_inconclusive_until_the_candidate_shows_the_capability():
    """Rule 1 as the card states the falsifier (acc >= 0.9 at K = 256, refuted) with rule 0 as its control."""
    capability = {"claims": ["capability.2"], "quote": "Accuracy \u2265 0.95 for K \u2264 48.", "if_met": "supported",
                  "conditions": [{"entrant": "candidate", "minus": None, "metric": "acc", "group": g, "op": ">=",
                                  "value": 0.95} for g in ("K=16", "K=32", "K=48")]}
    falsifier = {"claims": ["falsifier.3"], "quote": "stays \u2265 0.9 at K = 4d = 256", "if_met": "refuted",
                 "conditions": [{"entrant": "candidate", "minus": None, "metric": "acc", "group": "K=256",
                                 "op": ">=", "value": 0.9}], "control_rule": 0}
    rules = [capability, falsifier]
    card, design, scores = recorded()
    assert falsify.host_problems(dict(design, rules=rules), card, BASELINES) == []
    table = recorded_scores(scores)
    for rule in rules:
        assert FE.decide(rule, table, rules)["outcome"] == "inconclusive"      # the recorded run: nothing learned

    def trained_scores(low):
        block = {"acc": 0.5, "kl": 5.0, "n": 200,
                 "groups": {g: {"acc": low if g != "K=256" else 0.0, "kl": 5.0, "n": 50}
                            for g in ("K=16", "K=32", "K=48", "K=256")}}
        return {**table, "candidate": block}

    # trained overall but the control is not met: the survived falsifier says nothing
    decision = FE.decide(falsifier, trained_scores(0.2), rules)
    assert decision["outcome"] == "inconclusive"
    assert decision["reason"] == ("the positive control (rule 0) was not met, so a low score here does not separate a "
                                  "capacity limit from a model that cannot do the task")
    # control met and K = 256 low: the falsifier did not fire
    decision = FE.decide(falsifier, trained_scores(0.97), rules)
    assert decision["outcome"] == "supported" and decision["control"]["met"] is True
    # control met and K = 256 high: the falsifier fired
    high = trained_scores(0.97)
    high["candidate"]["groups"]["K=256"]["acc"] = 0.95
    assert FE.decide(falsifier, high, rules)["outcome"] == "refuted"


# ----- host gates (falsify.host_problems) --------------------------------------------------------------

def problems_of(rules=None, **changes):
    card = load_example()
    design = falsifier_design(rules=rules)
    design.update(changes)
    return falsify.host_problems(design, card, ["ngram"])


def test_the_fixture_design_has_no_host_problems():
    assert problems_of() == []


def test_polarity_mixing_and_claim_scoped_quotes():
    cap, fal = falsifier_rules()
    flipped = problems_of([cap, dict(fal, if_met="supported")])
    assert any("rule 1 decides falsifier.1 with if_met 'supported'" in p for p in flipped)
    mixed = problems_of([cap, dict(fal, claims=["capability.1", "falsifier.1"])])
    assert any(p.startswith("rule 1 mixes falsifier claims with other claims") for p in mixed)
    elsewhere = problems_of([dict(cap, claims=["capability.2"]), fal])
    assert "rule 0 quote is not text from capability.2" in elsewhere


def test_condition_values_groups_and_trivial_reference_subtraction_come_from_the_quote():
    cap, fal = falsifier_rules()

    def with_condition(**changes):
        return [dict(cap, conditions=[dict(cap["conditions"][0], **changes)]), fal]

    assert "rule 0 condition 0 value 0.5 is not a number in the quote" in problems_of(with_condition(value=0.5))
    assert any("subtracts unigram" in p for p in problems_of(with_condition(minus="unigram")))
    assert any("subtracts uniform" in p for p in problems_of(with_condition(minus="uniform")))
    assert "rule 0 condition 0 group m=3 is not covered by the quote" in problems_of(with_condition(group="m=3"))
    assert problems_of(with_condition(group=None)) == []
    # a bound in the quote covers the groups under it: "K <= 48" covers K=16
    quote = {"claims": ["capability.1"], "quote": "Accuracy >= 0.9 at m <= 2.", "if_met": "supported",
             "conditions": [dict(cap["conditions"][0], group="m=1")]}
    card = load_example()
    card["claims"]["capabilities"][0]["prediction"] = "Exact recall. Accuracy >= 0.9 at m <= 2."
    design = falsifier_design(rules=[quote, falsifier_rules()[1]])
    assert [p for p in falsify.host_problems(design, card, ["ngram"]) if "group" in p] == []
    over = dict(quote, conditions=[dict(cap["conditions"][0], group="m=3")])
    assert any("group m=3 is not covered" in p
               for p in falsify.host_problems(falsifier_design(rules=[over, falsifier_rules()[1]]), card, ["ngram"]))


def test_limit_and_survivable_rules_need_a_valid_control_rule():
    cap, fal = falsifier_rules()
    limit = dict(cap, conditions=[dict(cap["conditions"][0], op="<=")])
    text = "rule 0 can support the card by a low score; name a control_rule in this task where the candidate must first show the capability"
    assert text in problems_of([limit, fal])
    performance = dict(fal, conditions=[dict(cap["conditions"][0], group="m=1", value=0.1)])   # acc >= 0.1 refutes
    assert "rule 1 can support the card by a low score" in "\n".join(problems_of([cap, performance]))
    ok_limit = dict(limit, claims=["capability.1"], control_rule=1)
    assert any("rule 0 control_rule 1 must be a 'supported' rule" in p for p in problems_of([ok_limit, fal]))
    assert any("rule 0 control_rule 0 does not name another rule" in p for p in problems_of([dict(limit, control_rule=0), fal]))
    assert any("rule 0 control_rule 5 does not name another rule" in p for p in problems_of([dict(limit, control_rule=5), fal]))
    third = dict(cap, conditions=[dict(cap["conditions"][0], group="m=1")])
    assert not any("control" in p for p in problems_of([third, fal, dict(limit, control_rule=0)]))
    with_baseline = dict(cap, conditions=[dict(cap["conditions"][0], minus="ngram")])
    assert any("control_rule 1 must be a 'supported' rule" in p
               for p in problems_of([dict(limit, control_rule=1), with_baseline]))


def test_protocol_parameters_must_match_the_claim_text_and_deviations():
    card = load_example()
    card["claims"]["capabilities"][0]["prediction"] = "Exact recall. Accuracy >= 0.9 at m = 2. Single layer, d = 64."
    cap, fal = falsifier_rules()

    def check(params, deviations="none"):
        design = falsifier_design(rules=[cap, fal])
        design.update(protocol_params=params, deviations=deviations)
        return falsify.host_problems(design, card, ["ngram"])

    good = {"name": "d", "card": 64, "task": 64, "model_side": True}
    assert check([good]) == []
    assert any("protocol parameter d = 32 is not stated" in p for p in check([dict(good, card=32)]))
    assert any("protocol parameter d = 6 is not stated" in p for p in check([dict(good, card=6, task=6)]))
    shrunk = dict(good, task=32, model_side=False)
    assert any("deviations does not mention d" in p for p in check([shrunk]))
    assert check([shrunk], "d scaled from 64 to 32 to fit the caps") == []


def test_a_retry_may_not_change_rules_the_failure_did_not_name():
    cap, fal = falsifier_rules()
    first = falsifier_design(rules=[cap, fal])
    softened = falsifier_design(rules=[dict(cap, conditions=[dict(cap["conditions"][0], value=0.5)]), fal])
    failure = ["rule 1 quote is not text from falsifier.1"]
    assert falsify.retry_problems(first, softened, failure) == [
        "the retry changed rule 0's threshold, operator, quote or claims; the failure did not name this rule"]
    assert falsify.retry_problems(first, softened, ["rule 0 condition 0 value 0.9 is not a number in the quote"]) == []
    assert falsify.retry_problems(first, first, failure) == []


def test_schema_accepts_the_new_fields_and_requires_protocol_params():
    import jsonschema
    schema = falsifier_design_schema(["capability.1", "falsifier.1"], ["candidate", "ngram", "uniform", "unigram"])
    cap, fal = falsifier_rules()
    design = falsifier_design(rules=[dict(fal, control_rule=0), cap])
    design.update(protocol_params=[{"name": "d", "card": 64, "task": 32, "model_side": True}], solvable_by="lookup")
    jsonschema.validate({"tasks": [design], "untestable": []}, schema)
    del design["protocol_params"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"tasks": [design], "untestable": []}, schema)


def test_a_declared_reference_must_use_the_cards_d_and_the_schema_accepts_it():
    import jsonschema

    card = load_example()
    card["claims"]["capabilities"][0]["prediction"] = "Exact recall. Accuracy >= 0.9 at m = 2. Single layer, d = 64."
    cap, fal = falsifier_rules()
    design = falsifier_design(rules=[cap, fal])
    design.update(protocol_params=[{"name": "d", "card": 64, "task": 64, "model_side": True}],
                  reference={"kind": "pair_memory", "d": 64, "min_token": 3})
    assert falsify.host_problems(design, card, ["ngram"]) == []
    design["reference"]["d"] = 32
    assert "reference d = 32 is not the card's d; list protocol_params d with card = 32" in \
        falsify.host_problems(design, card, ["ngram"])
    design["protocol_params"] = []
    design["reference"]["d"] = 64
    assert "reference d = 64 is not the card's d; list protocol_params d with card = 64" in \
        falsify.host_problems(design, card, ["ngram"])
    schema = falsifier_design_schema(["capability.1", "falsifier.1"], ["candidate", "ngram", "uniform", "unigram"])
    jsonschema.validate({"tasks": [design], "untestable": []}, schema)
    for bad in ({"kind": "pair_memory", "d": 64}, {"kind": "other", "d": 64, "min_token": 3},
                {"kind": "pair_memory", "d": 1, "min_token": 3}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"tasks": [dict(design, reference=bad)], "untestable": []}, schema)


def test_the_designer_prompt_states_the_reference_gate_and_the_reference_is_staged():
    prompt = falsify.designer_prompt(load_example(), falsify.claim_items(load_example()),
                                     {"baselines": ["ngram"], "seed": 0, "budget": {}}, 2048)
    assert "9. Reference mechanism" in prompt
    files = bench_falsifier.harness_files("x", 0, [], 1.0, reference={"kind": "pair_memory", "d": 8, "min_token": 3})
    assert "code/hybrid/bench/falsifier_reference.py" in files
    assert json.loads(files["harness.json"])["reference"] == {"kind": "pair_memory", "d": 8, "min_token": 3}


# ----- candidate contract (S4) and prompt additions ------------------------------------------------------

def test_the_builder_prompt_carries_the_contract_section_and_the_prompt_files_are_untouched():
    prompt = builder_prompt(load_example(), None, None)
    assert "CANDIDATE CONTRACT ADDITIONS" in prompt and "train_max_len" in prompt and '"hyper"' in prompt
    builder_md = (ROOT / "hybrid" / "research" / "prompts" / "builder.md").read_text()
    assert "train_max_len" not in builder_md


@pytest.mark.parametrize("missing, phrase", [
    ('                "hyper": {"context": 1},          # integer hyperparameters by name (state width d, slots M, ...)\n',
     "describe()['hyper'] must be a non-empty dict of integer hyperparameters"),
    ('                "train_max_len": 2048,            # fit() uses every token of sequences up to this length\n',
     "describe()['train_max_len'] must be a positive int"),
])
def test_s4_construct_check_enforces_hyper_and_train_max_len(run_dir, missing, phrase):
    assert missing in TEMPLATE
    bad = TEMPLATE.replace(missing, "")
    orch, llm = make_orchestrator(run_dir, script={"builder": [{"candidate_py": bad, "notes": ""},
                                                               {"candidate_py": TEMPLATE, "notes": ""}]})
    run(orch)
    assert verdicts(only_card(orch))["S4"] == "pass"
    second = [c["user"] for c in llm.calls if c["prompt"] == "builder"][1]
    assert phrase in second
    built = events(orch, "candidate_built")[0]
    assert built["describe"]["hyper"] == {"context": 1} and built["describe"]["train_max_len"] == 2048


# ----- provenance and the S5 line -------------------------------------------------------------------------

def test_attempts_and_the_artifact_record_the_staged_code_hashes(run_dir):
    orch, llm = make_orchestrator(run_dir)
    run(orch)
    tasks = artifact(orch, "falsifier_tasks")
    now = bench_falsifier.current_code_hashes()
    assert tasks["gates"] == falsify.GATES_VERSION and tasks["code_sha256"] == now
    (attempt,) = tasks["attempts"]
    assert attempt["code_sha256"] == now and attempt["report"]["code_sha256"] == now
    assert tasks["validated"][0]["code_sha256"] == now
    assert set(now) == {"main.py", "falsifier_eval.py", "metrics.py", "falsifier_reference.py"} and all(len(h) == 64 for h in now.values())
    task_json = json.loads((run_dir / tasks["validated"][0]["dir"] / "task.json").read_text())
    assert task_json["validation"]["code_sha256"] == now
    # the designer is told the candidate's training length and the gates
    (call,) = designer_calls(llm)
    assert "train_max_len = 2048" in call["user"] and "VALIDATION GATES ADDED" in call["user"]
    assert tasks["candidate_train_max_len"] == 2048


def test_an_edit_to_the_evaluator_between_attempts_shows_in_the_recorded_hashes(run_dir, monkeypatch):
    nondeterministic = variant(SHIFT, "r = np.random.default_rng(seed)",
                               "import random\n    r = np.random.default_rng(seed + random.randrange(10**6))")
    real = bench_falsifier.staged_code
    calls = []

    def edited_after_the_first_job():
        code = real()
        calls.append(1)
        if len(calls) > 1:       # the first call stages round 0; every later one stages the edited code
            code["code/hybrid/bench/falsifier_eval.py"] += "\n# edited between validation attempts\n"
        return code

    monkeypatch.setattr(bench_falsifier, "staged_code", edited_after_the_first_job)
    script = {"falsifier_designer": [{"tasks": [falsifier_design(nondeterministic)], "untestable": []},
                                     {"tasks": [falsifier_design()], "untestable": []}]}
    orch, _ = make_orchestrator(run_dir, script=script)
    run(orch)
    first, second = artifact(orch, "falsifier_tasks")["attempts"]
    assert (first["round"], second["round"]) == (0, 1)
    assert first["code_sha256"]["falsifier_eval.py"] != second["code_sha256"]["falsifier_eval.py"]
    assert first["code_sha256"]["main.py"] == second["code_sha256"]["main.py"]


def test_a_retry_that_changes_an_unnamed_threshold_is_rejected_on_the_host(run_dir):
    nondeterministic = variant(SHIFT, "r = np.random.default_rng(seed)",
                               "import random\n    r = np.random.default_rng(seed + random.randrange(10**6))")
    cap, fal = falsifier_rules()
    softened = dict(cap, conditions=[dict(cap["conditions"][0], value=0.5)])
    script = {"falsifier_designer": [{"tasks": [falsifier_design(nondeterministic)], "untestable": []},
                                     {"tasks": [falsifier_design(rules=[softened, fal])], "untestable": []}]}
    orch, _ = make_orchestrator(run_dir, script=script)
    run(orch)
    tasks = artifact(orch, "falsifier_tasks")
    assert tasks["validated"] == []
    retry = tasks["attempts"][-1]
    assert retry["stage"] == "host" and retry["job_id"] is None and retry["code_sha256"] is None
    assert any("the retry changed rule 0's threshold" in p for p in retry["problems"])


def test_an_artifact_from_older_gates_is_not_reused(run_dir):
    orch, llm = make_orchestrator(run_dir)
    run(orch)
    card_id = only_card(orch)["id"]
    recorded_tasks = artifact(orch, "falsifier_tasks")
    stale = {k: v for k, v in recorded_tasks.items() if k != "gates"}
    orch.ctx.put_json_artifact("falsifier_tasks", orch.ctx.artifact_name(card_id), stale, card_id)
    dirs, _ = asyncio.run(falsify.prepare(orch.ctx, card_id))
    assert len(designer_calls(llm)) == 2 and len(dirs) == 1




# ----- S6 presentation -------------------------------------------------------------------------------------

def ledger_entry(ledger, claim_id):
    return next(c for c in ledger["claims"] if c["claim_id"] == claim_id)


def verdict_lines(orch, stage="S6"):
    return only_card(orch)["status"]["verdicts"][stage]["reasons"]


def test_a_falsifier_that_fired_headlines_the_s6_verdict_and_is_recorded_in_the_ledger(run_dir):
    orch, _ = make_orchestrator(run_dir, bench=FakeBench(falsifier_scores={"candidate": (0.5, 1.0),
                                                                           "ngram": (0.45, 1.2)}))
    run(orch)
    ledger = artifact(orch, "claims_ledger")
    assert ledger["falsified_by"] == [{"claim_id": "falsifier.1", "task_id": "iterated_shift", "rule_index": 1}]
    lines = verdict_lines(orch)
    assert lines[0] == "card falsified by falsifier.1 (task iterated_shift rule 1)"
    assert lines[1].startswith("claims ledger:")
    fired = [line for line in lines if line.startswith("falsifier 1 fired, card refuted on this point [rule]: ")]
    assert len(fired) == 1 and "With one pair stored" in fired[0] and "falsifier 1: " not in fired[0]


def test_a_falsifier_that_did_not_fire_reads_as_such_and_does_not_queue_a_mutation(run_dir):
    # capability 1 is refuted (0.5 < 0.9); the falsifier needs the candidate within 0.1 of the n-gram and is not
    orch, _ = make_orchestrator(run_dir, bench=FakeBench(falsifier_scores={"candidate": (0.5, 1.0),
                                                                           "ngram": (0.05, 2.0)}))
    run(orch)
    ledger = artifact(orch, "claims_ledger")
    assert ledger_entry(ledger, "falsifier.1")["status"] == "supported" and ledger["falsified_by"] == []
    assert ledger_entry(ledger, "capability.1")["status"] == "refuted"
    lines = verdict_lines(orch)
    assert not lines[0].startswith("card falsified")
    assert any(line.startswith("falsifier 1 did not fire (the card survives on this point) [rule]: ")
               for line in lines)
    assert not any(line.startswith("supported [rule]: falsifier") for line in lines)
    assert events(orch, "mutation_requested") == []


def test_claim_line_and_falsified_by_on_synthetic_entries():
    base = {"claim": "falsifier 3: recall stays >= 0.9 at K = 256.", "claim_id": "falsifier.3", "decided_by": "rule",
            "evidence": "", "decision_note": "decided by rule (t rule 1: inconclusive)", "rules": []}
    assert claim_line(dict(base, status="inconclusive")).startswith(
        "falsifier 3 inconclusive (neither fired nor ruled out) [rule]: recall stays >= 0.9 at K = 256.")
    assert claim_line(dict(base, status="supported")).startswith("falsifier 3 did not fire (the card survives")
    plain = dict(base, claim_id="capability.2", claim="capability 2: x", status="refuted")
    assert claim_line(plain).startswith("refuted [rule]: capability 2: x")
    fired = dict(base, status="refuted", rules=[{"task_id": "t", "rule_index": 1, "outcome": "refuted"},
                                                {"task_id": "t", "rule_index": 2, "outcome": "inconclusive"}])
    assert falsified_by([fired, plain, dict(base, status="supported")]) == [
        {"claim_id": "falsifier.3", "task_id": "t", "rule_index": 1}]
    assert falsified_by([dict(fired, decided_by="analyst")]) == []
