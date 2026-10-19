"""Offline review regressions. Synthetic cases are not migrated saved reference proof."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid.research.schemas import validate
from hybrid.research.stages import difference, known_work, s2_prior_art as s2
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.test_orchestrator_fakes import comparison, declared, defaults


def elman(name="Elman RNN"):
    c = declared("rnn")
    c.update(base_form=name, components=[
        {"role": "activation", "name": "tanh", "origin": "published", "cite": "Elman 1990"},
        {"role": "update", "name": "affine recurrence", "origin": "published", "cite": "Elman 1990"},
        {"role": "readout", "name": "linear", "origin": "published", "cite": "Elman 1990"}],
        reference_spec={"operation": "elman_tanh", "parameters": {
            "W": [[1.0]], "U": [[1.0]], "b": [0.0], "C": [[1.0]], "h0": [0.0]}})
    return c


def row(c, outputs, inputs=None):
    return dict(comparison(c["base_family"], 0.0),
                reference_samples=[{"inputs": inputs or [[0.2], [0.4]], "card_outputs": outputs}])


def test_model_written_square_copy_cannot_verify_without_reference():
    c = elman()
    c.pop("reference_spec")
    outcome, _, judged = s2.judge_comparisons([comparison("rnn", 0.0)], [c], "rnn")
    assert outcome == "unverified"
    assert not judged[0]["counted"]


def test_declared_tanh_executed_square_is_compared_to_host_tanh():
    c = elman()
    outcome, _, judged = s2.judge_comparisons([row(c, [[0.04], [0.1936]])], [c], "rnn")
    assert outcome == "refuted"
    assert judged[0]["max_abs_diff"] > s2.EQUIVALENCE_TOL
    assert judged[0]["reported_max_abs_diff"] == 0.0


@pytest.mark.parametrize("name", ["Elman RNN", "Elman recurrent neural network (1990)"])
def test_real_tanh_reference_is_independent_of_published_name_spelling(name):
    c = elman(name)
    h = 0.0
    outputs = []
    for x in (0.2, 0.4):
        h = float(np.tanh(h + x))
        outputs.append([h])
    outcome, _, judged = s2.judge_comparisons([row(c, outputs)], [c], "rnn")
    assert outcome == "verified"
    assert judged[0]["reference_operation"] == "elman_tanh"


def test_reference_parameters_cannot_hide_unknown_repairs():
    c = elman()
    c["transformations"] = [{"type": "sparsity", "what": "topK chosen from current hidden scores"}]
    assert s2.judge_comparisons([row(c, [[0.0], [0.0]])], [c], "rnn")[0] == "unverified"


@pytest.mark.parametrize("failure", ["null", "crashed", "missing"])
@pytest.mark.parametrize("kind", ["restatement", "unresolved", "limit_only", "learned_part_fixed"])
def test_failed_execution_never_verifies_at_high_confidence(failure, kind):
    c = elman()
    if kind == "restatement":
        c["components"][0]["origin"] = "card"
    if kind == "unresolved":
        c["transformations"] = [{"type": "other", "what": "state-selected search"}]
    if kind == "limit_only":
        c["limit_only"] = True
    critic = dict(defaults()["reduction_critic"], reduces_to="rnn", comparators=[c], confidence=1.0,
                  fixed_learned_parts=[{"part": "sigma", "capability": "adaptive width"}]
                  if kind == "learned_part_fixed" else [])

    async def ask(*args):
        return SimpleNamespace(call_id="fake-call", parsed={"check_code": None if failure == "null" else "pass",
                                                           "explanation": "synthetic failure"})

    async def run(*args):
        return SimpleNamespace(ok=failure != "crashed", status=failure, job_id="fake-job", stderr="", stdout="",
                               printed_json=lambda: None)

    cfg = {"s2": {"reduction_confidence": 0.9}, "generation": {
        "families_config": "hybrid/config/families.toml", "archive_config": "hybrid/config/archive.toml"}}
    ctx = SimpleNamespace(cfg=cfg, ask=ask, runner=SimpleNamespace(run=run))
    checked = asyncio.run(s2.run_reduction_check(ctx, make_card("card-abcdef12"), critic))
    assert checked.outcome == "unverified"
    assert s2.counted_comparisons(critic, checked.report) == []


def test_fixed_learned_part_blocks_even_a_numerical_match():
    c = elman()
    critic = dict(defaults()["reduction_critic"], reduces_to="rnn", comparators=[c], confidence=1.0,
                  fixed_learned_parts=[{"part": "sigma", "capability": "adaptive width"}])
    assert s2.declaration_problem(critic) is not None


@pytest.mark.parametrize("source", [
    {"kind": "standard_algorithm", "ref": "not a real algorithm", "covers": "all"},
    {"kind": "textbook", "ref": "invented section", "covers": "all"},
    {"kind": "paper", "ref": "s2:abc", "covers": "only initialization"},
    {"kind": "paper", "ref": "arxiv:abc", "covers": "only initialization"}])
def test_source_identity_and_coverage_assertions_are_not_proof(source):
    candidates = [{"key": "arxiv:abc", "source_id": "abc"}]
    critic = dict(defaults()["reduction_critic"], known_work=True, known_work_sources=[source])
    assert not known_work.judge(critic, [], candidates).kills
    if source["ref"] == "s2:abc":
        assert known_work.accepted_sources(critic, candidates) == []


def test_qualified_identity_is_not_collapsed_or_ambiguous():
    candidates = [{"key": "arxiv:abc", "source_id": "abc"}, {"key": "s2:abc", "source_id": "abc"}]
    critic = dict(defaults()["reduction_critic"], known_work_sources=[
        {"kind": "paper", "ref": "arxiv:abc", "covers": "unverified"}])
    assert known_work.cited_papers(critic, candidates) == ["arxiv:abc"]
    critic["known_work_sources"][0]["ref"] = "abc"
    assert known_work.accepted_sources(critic, candidates) == []


def test_two_distinct_prior_entries_remain_required():
    from hybrid.tests.research.test_known_work import requirements, statement
    req, families = requirements()
    card = make_card("card-abcdef12")
    one = statement(("family", families.known[0].id))
    assert "closest_prior:total" in difference.host_problems(one, card, req)
    assert "closest_prior:total" in difference.host_problems(
        statement(("family", families.known[0].id), ("family", families.known[0].id)), card, req)
    assert not difference.host_problems(
        statement(("family", families.known[0].id), ("family", families.known[1].id)), card, req)


def test_accepted_excerpt_gaps_scope_the_final_persisted_card(tmp_path):
    from hybrid.tests.research.test_difference import (card_of, checker, config, run, script, statement)
    from hybrid.tests.research.test_orchestrator_fakes import make_orchestrator
    from hybrid.research.outcomes import OutcomeStore

    unsupported = "The prior paper has no sequential mechanism."
    gap = "The excerpt does not establish absence of a sequential mechanism."

    def write(i, user):
        result = statement(refs=[{"kind": "family", "ref": "exact_match_cache", "title": "cache",
                                  "what_was_tried": "lookup", "why_it_failed": "no learning"}])(i, user)
        result["closest_prior"][0]["why_it_failed"] = unsupported
        result["why_it_failed"] = unsupported
        result["precise_change"] = unsupported
        return result

    def check(i, user):
        return dict(checker()(i, user), excerpt_gaps=[gap])

    empty = OutcomeStore(tmp_path / "outcomes")
    orch, _ = make_orchestrator(tmp_path / "run-20261018T000000Z", cfg=config(empty),
                                script=script(write, check))
    run(orch)
    card = card_of(orch)
    accepted = card["difference_statement"]
    assert card["status"]["verdicts"]["S2"]["verdict"] == "pass"
    assert unsupported not in json.dumps(accepted)
    assert accepted["evidence_scope"]["excerpt_gaps"] == [gap]
    assert accepted["evidence_scope"]["status"] == "excerpt_limited"
    assert accepted["closest_prior"][0]["ref"] == "2401.00001"
    validate("theory_card", card)
    artifacts = list((tmp_path / "run-20261018T000000Z" / "artifacts" / "difference").glob("*.json"))
    saved_artifacts = [json.loads(p.read_text()) for p in artifacts]
    assert any(unsupported in json.dumps(record["statement"]) for record in saved_artifacts)
    assert all(unsupported not in json.dumps(record["accepted_statement"]) for record in saved_artifacts
               if record["accepted"])
    assert any(record["accepted_statement"] == accepted for record in saved_artifacts if record["accepted"])


def test_saved_state_selected_topk_ssm_cannot_count_as_input_sparsity():
    from pathlib import Path
    saved = json.loads((Path(__file__).parent / "fixtures" / "review_gate_state_topk.json").read_text())
    # These are the saved measured numbers, with no independent reference added by the adapter.
    comparisons = saved["reduction_check"]["comparisons"]
    outcome, _, judged = s2.judge_comparisons(comparisons, saved["comparators"], saved["reduces_to"])
    assert outcome == "unverified"
    assert not any(c["counted"] for c in judged)
    assert judged[0]["declared_id"] == "topk_selective_diag_ssm"


def test_legacy_radical_answer_does_not_manufacture_a_source():
    from hybrid.tests.research.critic_answers import novelty_fields
    for radical in (True, False):
        migrated = novelty_fields({"radical_enough": radical})
        assert migrated["known_work_sources"] == []
        assert not migrated["known_work"]


def test_reference_schema_binds_declaration_and_executed_samples():
    c = elman()
    critic = dict(defaults()["reduction_critic"], reduces_to="rnn", mapping="h maps to h", comparators=[c])
    validate("reduction_critic", critic)
    report = {"comparisons": [row(c, [[0.04], [0.1936]])], "notes": "synthetic square execution"}
    assert s2.report_problem(report, [c]) is None
    assert s2.judge_comparisons(report["comparisons"], [c], "rnn")[0] == "refuted"


def test_executed_square_copy_cannot_verify(tmp_path):
    import subprocess
    import sys
    code = '''import json
h = 0.0
outputs = []
for x in (0.2, 0.4):
    h = (h + x) ** 2
    outputs.append([h])
print(json.dumps(outputs))
'''
    file = tmp_path / "square.py"
    file.write_text(code)
    actual = json.loads(subprocess.check_output([sys.executable, str(file)], text=True, cwd=tmp_path))
    assert s2.judge_comparisons([row(elman(), actual)], [elman()], "rnn")[0] == "refuted"


@pytest.mark.parametrize("operation,parameters,inputs,expected", [
    ("jeffrey_ipf", {"p0": [0.1, 0.2, 0.3, 0.4], "masks": [[1, 1, 0, 0], [1, 0, 1, 0]],
                     "targets": [0.6, 0.5], "labels": [0, 1, 0, 1], "label_count": 2}, [0, 1], [[1/3, 2/3], [1, 0]]),
    ("dyck2", {}, [0, 1, 3, 2], [[0], [0], [0], [1]]),
    ("rule110", {"h0": [0, 0, 1, 0]}, [0, 0], [[0, 1, 1, 0], [1, 1, 1, 0]]),
    ("viterbi", {"initial": [0.6, 0.4], "transition": [[0.7, 0.3], [0.4, 0.6]],
                  "emission": [[0.5, 0.5], [0.1, 0.9]]}, [0, 1], [[0.3, 0.04], [0.105, 0.081]]),
    ("linear_ssm", {"h0": [0.0], "A": [[0.5]], "B": [[2.0]], "C": [[1.0]]},
     [[1.0], [2.0]], [[2.0], [5.0]]),
    ("input_ssm", {"h0": [0.0, 0.0], "A": [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
                   "B": [[1, 2], [3, 4]], "C": [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
                   "mask": [[1, 0], [0, 1]]}, [0, 1], [[1, 0], [0, 4]]),
    ("linear_attention", {"S0": [[0.0]]}, [{"key": [2], "value": [3], "query": [0.5]}], [[3]]),
    ("delta_rule", {"S0": [[1.0]]}, [{"key": [2], "value": [3], "query": [0.5], "beta": 0.5}], [[1]])])
def test_published_operations_have_independently_specified_answers(operation, parameters, inputs, expected):
    # Constants are derived from the published equations, not generated by the production reference.
    np.testing.assert_allclose(s2.reference_outputs(operation, parameters, inputs), expected, rtol=0, atol=1e-15)
    family, _, repairs = s2.REFERENCE_OPERATIONS[operation]
    c = declared(family, transformations=repairs)
    c.update(base_form=operation, components=[{"role": "update", "name": operation,
                                             "origin": "published", "cite": "published control"}],
             reference_spec={"operation": operation, "parameters": parameters})
    report = dict(comparison(family, 0.0, transformations=repairs),
                  reference_samples=[{"inputs": inputs, "card_outputs": expected}])
    assert s2.judge_comparisons([report], [c], family)[0] == "verified"


def test_actual_saved_jeffrey_execution_matches_independent_host_reference():
    import contextlib
    import hashlib
    import io
    from pathlib import Path
    import runpy

    path = Path(__file__).parent / "fixtures" / "review_gate_jeffrey_saved.py"
    provenance = json.loads(path.with_suffix(".provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["sha256"]
    with contextlib.redirect_stdout(io.StringIO()):
        saved = runpy.run_path(str(path))
    rng = np.random.default_rng(12345)
    for _ in range(saved["n_inputs"]):
        masks, targets, labels = saved["make_params"](rng)
        inputs = rng.integers(0, saved["V"], size=saved["n"])
        actual = saved["card_run"](masks, targets, labels, inputs).tolist()
        c = declared("other_known")
        c.update(base_form="Jeffrey conditioning / IPF (Jeffrey 1965; Csiszar 1975)",
                 components=[{"role": "update", "name": "Jeffrey conditioning", "origin": "published",
                              "cite": "Jeffrey 1965; Csiszar 1975"}],
                 reference_spec={"operation": "jeffrey_ipf", "parameters": {
                     "p0": [1.0 / saved["d"]] * saved["d"], "masks": masks.tolist(),
                     "targets": targets.tolist(), "labels": labels.tolist(), "label_count": saved["V"]}})
        report = dict(comparison("other_known", 0.0), reference_samples=[{
            "inputs": inputs.tolist(), "card_outputs": actual}])
        assert s2.judge_comparisons([report], [c], "other_known")[0] == "verified"


def test_saved_counted_labels_without_reference_are_not_revalidated():
    c = elman()
    c.pop("reference_spec")
    critic = dict(defaults()["reduction_critic"], reduces_to="rnn", comparators=[c])
    old = dict(comparison("rnn", 0.0), eligibility="counted", within_tolerance=True)
    assert s2.counted_comparisons(critic, {"comparisons": [old]}) == []


def test_counted_argument_cannot_override_an_admitted_fixed_learned_part():
    critic = dict(defaults()["reduction_critic"], known_work=True,
                  fixed_learned_parts=[{"part": "sigma", "capability": "adaptive width"}])
    assert not known_work.judge(critic, [{"declared_id": "known"}], []).kills


def test_unrelated_retrieved_paper_cannot_fill_a_required_relevant_slot():
    from hybrid.tests.research.test_known_work import requirements, statement
    req, families = requirements(literature={"arxiv:close": {"key": "arxiv:close"},
                                            "arxiv:far": {"key": "arxiv:far"}},
                                 relevant_papers=("arxiv:close",), need=1)
    card = make_card("card-abcdef12")
    bad = statement(("literature", "arxiv:far"), ("family", families.known[0].id))
    assert "closest_prior:count" in difference.host_problems(bad, card, req)
    assert "closest_prior:total" in difference.host_problems(bad, card, req)
    good = statement(("literature", "arxiv:close"), ("family", families.known[0].id))
    assert not difference.host_problems(good, card, req)


def test_difference_listing_and_required_citation_preserve_source_prefix():
    from hybrid.tests.research.test_known_work import requirements, statement
    candidates = [{"key": "arxiv:abc", "source_id": "abc", "title": "one", "year": 2020, "excerpt": "a"},
                  {"key": "s2:abc", "source_id": "abc", "title": "two", "year": 2021, "excerpt": "b"}]
    refs = difference.literature_refs({"evidence": []}, {"candidates": candidates}, ("arxiv:abc",))
    assert refs["arxiv:abc"]["title"] == "one" and refs["s2:abc"]["title"] == "two"
    assert "abc" not in refs
    req, families = requirements(literature=refs, cited_papers=("arxiv:abc",))
    wrong = statement(("literature", "s2:abc"), ("family", families.known[0].id))
    assert "cited_source:arxiv:abc" in difference.host_problems(wrong, make_card("card-abcdef12"), req)


def test_arbitrary_algorithm_name_does_not_force_an_impossible_difference_citation():
    from hybrid.tests.research.test_known_work import requirements
    source = {"kind": "standard_algorithm", "ref": "fabricated algorithm", "covers": "entire mechanism"}
    req, _ = requirements(cited_names=(source,))
    text = difference.context_block({"label": "novel"}, req, {})
    assert "UNVALIDATED SOURCE LEAD" in text
    assert "not a mandatory closest_prior entry" in text
    assert "Name it in a closest_prior entry" not in text


def test_scoping_preserves_original_and_full_gap_list():
    # The full gate/persisted-card path is exercised above. Here check the deterministic transformation does
    # not mutate the raw statement or truncate late limitations.
    from hybrid.tests.research.test_known_work import statement
    original = statement(("literature", "arxiv:a"), ("family", "exact_match_cache"))
    gaps = ["gap " + str(i) + "x" * 100 for i in range(12)]
    attempt = difference.Attempt(original, {}, {"accepted": True, "excerpt_gaps": gaps}, ["writer", "checker"])
    scoped = difference.scoped_statement(attempt)
    assert scoped["evidence_scope"]["excerpt_gaps"] == gaps
    assert scoped["evidence_scope"]["source_calls"] == ["writer", "checker"]
    assert original["closest_prior"][0]["why_it_failed"] == "f"
    assert scoped["closest_prior"][0]["why_it_failed"] != "f"
    validate("difference", scoped)
