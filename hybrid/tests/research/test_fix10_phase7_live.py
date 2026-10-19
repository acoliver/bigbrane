"""FIX-10: the four defects of the Phase 7 live check (results/phase7-live-20261014T1752Z/REPORT.md).

D2. S2 trusted an `equivalent` flag the LLM-written check program computed: card-c9615069's check
(runs/phase7-live-20261014T1752Z/artifacts/executor/job-137fb29ee378.json, copied to
fixtures/phase7_live_20261014T1752Z/) measured a 0.240 probability difference from the hard bimap it claimed, set
equivalent from a corrected soft rewrite instead, and S2 accepted reduces_to_known. The check now reports comparisons
as data and the host decides, counting only claimed or repaired forms of the stated family.
D1. Own history left out every record with the card's id, so a card copied into a new run never met its own earlier
deaths; fixture: the recorded outcome records of card-1184f7a7 and the card version S2 saw.
D3. Every cause of a record shared one clipped detail; each cause now carries its own deciding explanation.
D4. A dialogue stopped by the round cap recorded no open questions."""
from hybrid.tests.research.critic_answers import KNOWN, NOT_KNOWN, novelty_fields
import asyncio
import json
import math

import numpy as np
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid.research.outcome_records import CAUSE_CHARS, RunView, build_record
from hybrid.research.outcomes import OutcomeStore, own_history, r3_problems, r3_targets
from hybrid.research.signature import card_signature
from hybrid.research.stages import difference, s2_prior_art
from hybrid.research.stages.common import ESCAPE_FIELD, ExecResult
from hybrid.research.stages.s2_prior_art import EQUIVALENCE_TOL, judge_comparisons, report_problem
from hybrid.tests.research.reference_replays import FIXTURES as REPLAY_FIXTURES, load as replay_evidence
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.outcome_fixtures import build_run
from hybrid.tests.research.test_orchestrator_fakes import (comparison, declared, make_orchestrator, orch_config,
                                                           program, synthetic_declared, SYNTHETIC_REFERENCES)
from hybrid.tests.research.test_refine import always_new, asks, grown, proposer, question
from hybrid.tests.research.test_refine import config as refine_config

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "phase7_live_20261014T1752Z"
LIVE_RUN = "phase7-live-20261014T1752Z"
SOURCE_RUN = "run-20261010T181300Z"
CRITIC_KEYS = ("reduces_to", "mapping", "confidence", "difference_if_any", "why",
               "textbook_transplant", "textbook_why", "family", "family_why")
# The recorded program ends by setting its own verdict from the soft rewrite alone.
VERDICT_LINE = "equiv = maxd < 1e-6"
# FIX-12: the critic declares the hard bimap it names as its one comparator; the corrected soft rewrite is reported under
# no declared id (a correction term is not a repair of any type).
HARD_BIMAP = {"id": "hard_bimap", "base_family": "other_known",
              "base_form": "bidirectional map (forcePut) used as an exact-match cache",
              "update": "forcePut(w_{t-1}, w_t): bind the pair, unbind any other key of the value; LRU cap",
              "read": "p(v) = 1 for the value bound to the query, fallback mass over unbound values",
              "transformations": [], "parameter_mapping": "key w_{t-1}, value w_t, cap P_max", "limit_only": False,
              "components": [{"role": "update", "name": "bimap forcePut", "origin": "published", "cite": "Guava BiMap"}]}
HARD_BIMAP['reference_spec'] = replay_evidence('fix10')['hard_bimap']['reference_spec']
HONEST_TAIL = '''
print(json.dumps({"comparisons": [
    {"declared_id": None, "comparator": "two-sided state-gated one-hot delta rule with a -rho*x*y correction at (i, j)",
     "family": "other_known", "transformations": [], "limit_only": False,
     "reference": "the card's update rearranged; the correction has no published counterpart",
     "inputs_tested": 300, "max_abs_diff": maxd},
    {"declared_id": "hard_bimap", "comparator": "hard bimap with forcePut, LRU cap and fallback over unbound values",
     "family": "other_known", "transformations": [], "limit_only": False,
     "reference": "bidirectional map (forcePut), exact-match cache", "inputs_tested": 300, "max_abs_diff": bd,
     "reference_samples": json.load(open(REPLAY_FIXTURE))["hard_bimap"]["reference_samples"]}],
    "notes": "recorded job-137fb29ee378 with its own numbers reported as comparisons"}))
'''


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def recorded_job():
    return load("job-137fb29ee378.json")


def recorded_critic():
    record = load("reduction_critic-card-c9615069-v7-b1ef30.json")
    return {key: record[key] for key in CRITIC_KEYS} | novelty_fields(record) | {"comparators": [HARD_BIMAP]}


def honest_program() -> str:
    code = recorded_job()["files"]["main.py"]
    assert code.count(VERDICT_LINE) == 1
    return code.split(VERDICT_LINE, 1)[0] + HONEST_TAIL.replace("REPLAY_FIXTURE", repr(str(REPLAY_FIXTURES / "fix10.json")))


def run(orch):
    return asyncio.run(orch.run())


def only_card(orch):
    return orch.store.get_card(orch.store.list_cards()[0])


# ----- D2: the host decides equivalence -------------------------------------------------------------------------------


def test_the_recorded_check_verdict_is_outside_the_contract_and_is_not_read():
    job = recorded_job()
    printed = ExecResult(job["job_id"], job["status"], job["detail"], job["stdout"], job["stderr"],
                         job["run_s"]).printed_json()
    assert printed["equivalent"] is True and "max p diff 0.240" in printed["notes"]   # what S2 accepted
    assert report_problem(printed, [HARD_BIMAP]) is not None


def test_the_recorded_programs_own_numbers_refute_the_hard_cache_claim(tmp_path):
    path = tmp_path / "main.py"
    path.write_text(honest_program(), encoding="utf-8")
    done = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=100, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    printed = json.loads(done.stdout.strip().splitlines()[-1])
    assert report_problem(printed, [HARD_BIMAP]) is None
    rewrite, claimed = printed["comparisons"]
    assert rewrite["max_abs_diff"] < 1e-12                 # the soft rewrite equals the card
    assert 0.23 < claimed["max_abs_diff"] < 0.25           # the recorded 0.240 against the claimed hard bimap
    outcome, reasons, judged = judge_comparisons(printed["comparisons"], [HARD_BIMAP], recorded_critic()["reduces_to"])
    assert outcome == "refuted"
    assert reasons[0].startswith("numeric check rejects the claimed reduction") and "0.24" in reasons[0]
    assert reasons[1].startswith("not counted (no declared comparator") and "rewrite" in reasons[1]
    assert [c["counted"] for c in judged] == [False, True]


# Explicit synthetic numerical specimens for downstream routing, not historical card evidence.
PUBLISHED = dict(HARD_BIMAP, id='known')
REPAIRED = dict(PUBLISHED, id='repaired', transformations=[{'type':'normalizer', 'what':'L1 normalize the read'}],
                reference_spec=dict(PUBLISHED['reference_spec'], operation='bimap_hard_l1_read'))


def synthetic_rows(rows, declarations):
    declarations = {c['id']: c for c in declarations}
    result = []
    for row in rows:
        c = declarations.get(row['declared_id'])
        if c is None or 'reference_spec' not in c:
            result.append(row)
            continue
        if c['base_family'] == 'ssm':
            _, _, inputs, values, _ = SYNTHETIC_REFERENCES['ssm']
        else:
            sample = replay_evidence('fix10')['hard_bimap']['reference_samples'][0]
            inputs, values = sample['inputs'][:1], sample['card_outputs'][:1]
        offset = row['max_abs_diff'] if math.isfinite(row['max_abs_diff']) else 1.0
        # The nonfinite reported scalar is diagnostic; these explicit synthetic output values decide.
        trace = {'inputs': inputs, 'card_outputs': (np.asarray(values) + offset).tolist()}
        result.append(dict(row, reference_samples=[trace]))
    return result


@pytest.mark.parametrize("declared_list, comparisons, outcome", [
    ([], [comparison("other_known", 0.0, declared_id=None)], "unverified"),            # a rewrite never counts
    ([declared("linear_attention")], [comparison("linear_attention", 0.0)], "unverified"),   # another family
    ([PUBLISHED], [comparison("other_known", 2e-6)], "refuted"),
    ([PUBLISHED], [comparison("other_known", 1e-7)], "verified"),
    ([PUBLISHED], [comparison("other_known", float("nan"))], "refuted"),
    ([PUBLISHED, REPAIRED], [comparison("other_known", 0.3),
                             comparison("other_known", 0.0, "repaired", transformations=("normalizer",))], "verified"),
    ([PUBLISHED], [comparison("other_known", 0.3), comparison("other_known", 0.0, declared_id=None)], "refuted"),
    ([], [], "unverified"),
])
def test_only_declared_comparators_of_the_stated_family_decide(declared_list, comparisons, outcome):
    assert judge_comparisons(synthetic_rows(comparisons, declared_list), declared_list, "other_known")[0] == outcome


def test_a_verified_repair_still_reports_the_claimed_form_that_failed():
    parent = dict(synthetic_declared('ssm'), id='parent')
    lifted = dict(parent, id='lifted', transformations=[{'type':'lifted_state', 'what':'direct sum with a zero channel'}],
                  reference_spec=dict(parent['reference_spec'], operation='linear_ssm_zero_lift'))
    outcome, reasons, _ = judge_comparisons(
        synthetic_rows([comparison("ssm", 0.3, "parent", comparator="parent mapping"),
         comparison("ssm", 1e-9, "lifted", comparator="lifted state", transformations=("lifted_state",))], [parent, lifted]),
        [parent, lifted], "ssm")
    assert outcome == "verified" and "lifted state" in reasons[0] and f"{EQUIVALENCE_TOL:g}" in reasons[0]
    assert reasons[1].startswith("other comparisons with ssm") and "parent mapping" in reasons[1]


def test_the_report_contract_refuses_a_program_verdict_and_a_repeated_comparator():
    ssm = [declared("ssm")]
    good = {"comparisons": [comparison("ssm", 0.0)], "notes": ""}
    assert report_problem(good, ssm) is None
    assert "equivalent" in report_problem(dict(good, equivalent=True), ssm)
    assert "more than once" in report_problem({"comparisons": [comparison("ssm", 0.0)] * 2, "notes": ""}, ssm)
    assert report_problem({"comparisons": [dict(comparison("ssm", 0.0), inputs_tested=0)], "notes": ""}, ssm)


def s2_script(check_code):
    return {"reduction_critic": recorded_critic(),
            "reduction_check": {"check_code": check_code, "explanation": "recorded check of card-c9615069"}}


def exclusion_config():
    cfg = orch_config()
    cfg["s2"]["family_exclusion"] = True       # the recorded critic answers the family question
    return cfg


@pytest.mark.parametrize("which, outcome, phrase", [
    ("recorded", "unverified", "outside the comparison contract"),
    ("honest", "refuted", "rejects the claimed reduction"),
])
def test_s2_no_longer_accepts_the_recorded_reduction_of_c9615069(run_dir_fix10, which, outcome, phrase):
    code = recorded_job()["files"]["main.py"] if which == "recorded" else honest_program()
    orch, llm = make_orchestrator(run_dir_fix10, cfg=exclusion_config(), script=s2_script(code))
    run(orch)
    card = only_card(orch)
    s2 = card["status"]["verdicts"]["S2"]
    assert s2["verdict"] != "reduces_to_known"
    assert any(r.startswith(f"reduction claim other_known ({outcome}): ") and phrase in r for r in s2["reasons"])
    artifact = orch.ctx.latest_json_artifact(card["id"], "reduction_critic")
    assert artifact["reduction_outcome"] == outcome
    if which == "honest":
        assert [c["counted"] for c in artifact["reduction_check"]["comparisons"]] == [False, True]
        assert [c["eligibility"] for c in artifact["reduction_check"]["comparisons"]] == ["undeclared", "counted"]
        assert artifact["reduction_check"]["tolerance"] == EQUIVALENCE_TOL
    # the critic said radical_enough false, so with the reduction refused the card dies as not radical (R2)
    assert s2["verdict"] == "fail" and any(r.startswith("gate: not radical enough") for r in s2["reasons"])


def test_a_comparison_with_the_claimed_form_still_gives_reduces_to_known(run_dir_fix10):
    sample = replay_evidence('fix10')['hard_bimap']['reference_samples'][0]
    trace = dict(sample, inputs=sample['inputs'][:1], card_outputs=sample['card_outputs'][:1])
    report = {"comparisons": [dict(comparison("other_known", 0.0, "hard_bimap"), reference_samples=[trace])],
              "notes": "routing only: the actual first saved prefix matches; this does not preserve the full card reduction"}
    orch, _ = make_orchestrator(run_dir_fix10, cfg=exclusion_config(), script=s2_script(program(report)))
    run(orch)
    assert only_card(orch)["status"]["verdicts"]["S2"]["verdict"] == "reduces_to_known"


@pytest.fixture
def run_dir_fix10(tmp_path):
    return tmp_path / "run-20261014T200000Z"


# ----- D1: a copied card meets its own earlier deaths ------------------------------------------------------------------


@pytest.fixture
def recorded_store(tmp_path):
    """The recorded outcome records of card-1184f7a7: its death in the source run (store seq 10) and the record the
    live run wrote for it (seq 291), appended in that order."""
    store = OutcomeStore(tmp_path / "outcomes")
    for entry in sorted(load("outcomes-card-1184f7a7.json")["records"], key=lambda e: e["seq"]):
        store.append(entry["record"])
    return store


def test_own_history_finds_card_1184f7a7s_own_recorded_death_in_the_source_run(recorded_store):
    card = load("card-1184f7a7-v7.json")
    signature = card_signature(card)
    assert signature["signature"] == "sig-421616f3354d2f5f"        # unchanged by R, as the report says
    matches = own_history(recorded_store, card, signature, current_run=LIVE_RUN, limit=8, jaccard_min=0.6,
                          tfidf_min=0.3)
    [match] = matches
    assert match.exact and match.stored.record["run"] == SOURCE_RUN
    assert match.stored.record["kill_codes"] == ["reduces_to_known:linear_attention"]
    # R3 applies to it: without a statement that addresses the cause, the card fails the rule
    assert r3_targets(matches) == [match]
    [problem] = r3_problems(card, matches)
    cause = match.stored.causes()[0]["id"]
    assert cause in problem and SOURCE_RUN in problem
    # and the difference statement's requirements name that cause
    ctx = SimpleNamespace(outcomes=recorded_store, cfg={"outcomes": {"enforce_r3": True, "jaccard_min": 0.6}})
    families, _ = difference.families_for(SimpleNamespace(cfg=orch_config()))
    critic = {"family": "none", "known_work_sources": []}
    req = difference.requirements(ctx, {"evidence": []}, {"candidates": []}, matches, critic, families)
    assert req.cause_ids() == [cause]


def test_s2_leaves_out_only_the_current_runs_own_record(recorded_store):
    card = load("card-1184f7a7-v7.json")
    emitted = []
    ctx = SimpleNamespace(run_id=SOURCE_RUN, outcomes=recorded_store,
                          cfg={"outcomes": {"history_max": 8, "jaccard_min": 0.6, "tfidf_min": 0.3}},
                          put_json_artifact=lambda *args: None, artifact_name=lambda card_id: "a",
                          emit=lambda kind, data: emitted.append(data),
                          card_events=lambda card_id, kind: [])          # no refinement dialogue (FIX-12, F3)
    matches, block = s2_prior_art.own_history(ctx, card)
    # queried from the source run, its own record there is left out and the live run's record is found
    assert [m.stored.record["run"] for m in matches] == [LIVE_RUN]
    assert emitted[0]["exact"] == [matches[0].stored.record_id] and "EXACT signature" in block


# ----- D3: each cause carries its own deciding explanation -------------------------------------------------------------


ESCAPE_WHY = ("Escape test, step 1: the broken invariant is real in the card's own equations. Step 2: the strongest "
              "repair is a sequential greedy CRP mixture of conjugate regressions, and it reproduces every step. "
              + "The repair closes the escape. " * 40)
ESCAPE_CRITIC = {"reduces_to": "other_known", "mapping": "Known form: SUGS, a sequential greedy CRP mixture of "
                 "conjugate Bayesian linear regressions (Wang and Dunson 2011).", "confidence": 0.7,
                 "difference_if_any": "", **KNOWN, "why": ESCAPE_WHY, "textbook_transplant": False,
                 "textbook_why": "-"}
UP_TO_S2 = [("S0", "pass", ["ok"]), ("S1", "pass", ["ok"])]


def records(tmp_path, cards):
    view = RunView(build_run(tmp_path / "run-20261011T000000Z", cards))
    return {card_id: build_record(view, card_id, "backfill") for card_id in view.card_ids()}


def test_two_violated_obligations_get_their_own_explanations(tmp_path):
    card = make_card("card-0000e001")
    reasons = ["gate rule: check_violation",
               "executed check violated stated obligation 'capacity recall' (replayed): random codes recall 0.765",
               "executed check violated stated obligation 'elimination inference' (replayed): accuracy 0.563"]
    record = records(tmp_path, [(card, UP_TO_S2 + [("S2", "pass", ["ok"]), ("S3", "infeasible_proved", reasons)],
                                 {"s3_gate": {"rule": "check_violation", "verdict": "infeasible_proved"}})])[card["id"]]
    causes = {c["code"]: c["detail"] for c in record["causes"]}
    assert causes == {"obligation_failed:capacity_recall": reasons[1],
                      "obligation_failed:elimination_inference": reasons[2]}


def test_an_escape_reduction_records_the_mapping_and_the_whole_argument(tmp_path):
    card = make_card("card-0000e002")
    card[ESCAPE_FIELD] = {"known_family": "other_known: a count cache",
                          "broken_invariant": "The state is not a sum of per-token count writes in update u[0].",
                          "why_unrepairable": "No count cache reproduces the regime switches the card infers online."}
    gate = ["reduction claim other_known (verified): numeric check confirms equivalence to other_known: SUGS",
            "gate: update rule reduces to other_known: " + ESCAPE_CRITIC["mapping"]]
    record = records(tmp_path, [(card, UP_TO_S2 + [("S2", "reduces_to_known", gate)],
                                 {"reduction_critic": ESCAPE_CRITIC})])[card["id"]]
    [cause] = record["causes"]
    assert "Known form: SUGS" in cause["detail"] and "numeric check confirms equivalence" in cause["detail"]
    assert cause["detail"].endswith("The repair closes the escape.")       # the critic's argument in full
    assert record["kill_detail"].startswith("reduced to other_known: Known form: SUGS")


def test_not_radical_r2_records_the_critics_whole_argument(tmp_path):
    card = make_card("card-0000e003")
    why = "Check of the escape argument: the parent's mapping fails. " + "x " * 400 + "The closest repair is Hopfield."
    critic = dict(ESCAPE_CRITIC, reduces_to="none", mapping=None, why=why)
    gate = [f"gate: not radical enough: {why[:500]}"]
    record = records(tmp_path, [(card, UP_TO_S2 + [("S2", "fail", gate)], {"reduction_critic": critic})])[card["id"]]
    [cause] = record["causes"]
    assert cause["code"] == "not_radical:R2" and cause["detail"].endswith("The closest repair is Hopfield.")
    assert len(record["kill_detail"]) <= 600


def test_an_explanation_past_the_bound_is_cut_with_a_marker(tmp_path):
    card = make_card("card-0000e004")
    critic = dict(ESCAPE_CRITIC, reduces_to="none", mapping=None, why="y " * (CAUSE_CHARS + 100))
    record = records(tmp_path, [(card, UP_TO_S2 + [("S2", "fail", ["gate: not radical enough: y"])],
                                 {"reduction_critic": critic})])[card["id"]]
    detail = record["causes"][0]["detail"]
    assert "[truncated: 8000 of" in detail and len(detail) < CAUSE_CHARS + 120


def test_the_difference_context_shows_each_cause_in_full():
    families, _ = difference._families("hybrid/config/families.toml", "hybrid/config/archive.toml")
    long = "the deciding argument " * 60 + "ends here"
    match = {"card_id": "card-0000e005", "title": "t", "exact_signature": True, "jaccard": 1.0, "outcome": "dead",
             "terminal_stage": "S2", "kill_codes": ["not_radical:R2"],
             "causes": [{"id": "oc-aaaaaaaaaaaa/not_radical:R2", "detail": long}]}
    req = difference.Requirements(literature={}, outcomes={"oc-aaaaaaaaaaaa": match}, r3=[], family=None, need=1,
                                  families=families)
    block = difference.context_block({"label": "novel"}, req, {"reduces_to": "none"})
    assert long in block


# ----- D4: the round cap keeps the last round's questions --------------------------------------------------------------


def refine_dir(tmp_path):
    return tmp_path / "run-20261014T210000Z"


def dialogue(orch):
    card_id = orch.store.list_cards()[0]
    return (orch.ctx.latest_json_artifact(card_id, "dialogue_assumptions"),
            orch.ctx.latest_json_artifact(card_id, "refinement"), orch.store.get_card(card_id))


def test_the_cap_records_the_last_rounds_questions_as_answered_but_unchecked(tmp_path):
    orch, _ = make_orchestrator(refine_dir(tmp_path), cfg=refine_config(),
                                script={"refine_challenger": always_new, "refine_proposer": proposer()})
    run(orch)
    assumptions, summary, card = dialogue(orch)
    assert assumptions["ended"] == "budget:refinement" and summary["last_revision_unchecked"]
    [question] = assumptions["open_questions"]
    assert question["where"] == "mechanism.state[2]" and question["status"] == "answered_unchecked"
    assert summary["open_questions"] == assumptions["open_questions"]
    reasons = card["status"]["verdicts"]["R"]["reasons"]
    assert any(r.startswith("open question Q1 (answered, unchecked)") for r in reasons)


def test_the_cap_marks_questions_a_rejected_revision_left_unanswered(tmp_path):
    orch, _ = make_orchestrator(refine_dir(tmp_path), cfg=refine_config(),
                                script={"refine_challenger": always_new, "refine_proposer": proposer(grown(3))})
    run(orch)
    assumptions, summary, _ = dialogue(orch)
    assert assumptions["ended"] == "budget:refinement" and not summary["last_revision_unchecked"]
    assert [q["status"] for q in assumptions["open_questions"]] == ["unanswered"]


def test_a_converged_dialogue_leaves_nothing_open(tmp_path):
    orch, _ = make_orchestrator(refine_dir(tmp_path), cfg=refine_config(),
                                script={"refine_challenger": asks([question()], []), "refine_proposer": proposer()})
    run(orch)
    assumptions, _, _ = dialogue(orch)
    assert assumptions["ended"] == "converged" and assumptions["open_questions"] == []
