import copy
import json
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from hybrid.research.schemas import SchemaViolation, load_schema, validate
from hybrid.tests.research.builders import load_example

SCHEMA_NAMES = [
    "theory_card", "verdict", "literature_receipt", "bench_result", "salad_critic",
    "reduction_critic", "prior_art", "feasibility_judge", "disprover", "analyst", "claims_ledger",
]
CALL = "call-000000000001"


def _drop_state_shape(card):
    card["mechanism"]["state"][0].pop("shape")


def _one_falsifier(card):
    card["falsifiers"] = card["falsifiers"][:1]


def _no_toy_example(card):
    card.pop("toy_example")


def _toy_without_python(card):
    card["toy_example"].pop("python")


def _equation_without_plain_form(card):
    card["mechanism"]["update"][0].pop("plain")


def _readout_without_latex(card):
    card["mechanism"]["readout"][0].pop("latex")


def _complexity_without_big_o(card):
    card["claims"]["complexity"]["memory"] = "constant"


def _no_known_relatives(card):
    card["known_relatives"] = []


def _no_capabilities(card):
    card["claims"]["capabilities"] = []


def _state_without_meaning(card):
    card["mechanism"]["state"][0].pop("meaning")


def _verdict_outside_enum(card):
    card["status"]["verdicts"]["S1"] = {"verdict": "maybe", "reasons": ["x"], "by_calls": [CALL]}


def _verdict_without_calls(card):
    card["status"]["verdicts"]["S1"] = {"verdict": "pass", "reasons": ["x"], "by_calls": []}


def _unknown_top_level_field(card):
    card["vibes"] = "high"


def _bad_card_id(card):
    card["id"] = "card-XYZ"


SALAD_CASES = {
    "state object without shape": _drop_state_shape,
    "state object without meaning": _state_without_meaning,
    "only one falsifier": _one_falsifier,
    "no toy example": _no_toy_example,
    "toy example without python": _toy_without_python,
    "update equation without plain form": _equation_without_plain_form,
    "readout equation without LaTeX": _readout_without_latex,
    "complexity without big-O form": _complexity_without_big_o,
    "no known relatives": _no_known_relatives,
    "no capability claims": _no_capabilities,
    "verdict outside the enum": _verdict_outside_enum,
    "verdict without the calls behind it": _verdict_without_calls,
    "unknown top-level field": _unknown_top_level_field,
    "malformed card id": _bad_card_id,
}


def test_example_card_is_accepted():
    assert validate("theory_card", load_example()) is not None


@pytest.mark.parametrize("name", SALAD_CASES)
def test_salad_like_cards_are_rejected(name):
    card = load_example()
    SALAD_CASES[name](card)
    with pytest.raises(SchemaViolation):
        validate("theory_card", card)


def test_every_schema_file_is_valid_draft_2020_12():
    for name in SCHEMA_NAMES:
        schema = load_schema(name)
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_verdict_enum_agrees_between_card_and_verdict_schemas():
    expected = ["pass", "fail", "revise", "infeasible_proved", "reduces_to_known", "undetermined"]
    card_enum = load_schema("theory_card")["$defs"]["verdict_entry"]["properties"]["verdict"]["enum"]
    record_enum = load_schema("verdict")["properties"]["verdict"]["enum"]
    assert card_enum == expected
    assert record_enum == expected


def test_toy_example_runs_under_the_venv_python_and_reproduces_expected_values():
    toy = load_example()["toy_example"]
    completed = subprocess.run(
        [sys.executable, "-c", toy["python"]], capture_output=True, text=True, timeout=60, check=True,
    )
    actual = json.loads(completed.stdout)
    assert set(actual) == set(toy["expected"])
    for key, value in toy["expected"].items():
        assert actual[key] == pytest.approx(value, abs=1e-9), key


FIXABLE = {"kind": "missing_definition", "where": "mechanism.update[0]", "problem": "theta is never defined"}
FATAL = {"kind": "dimension_error", "where": "mechanism.update[1]", "problem": "W (d x d) times x_t (V) cannot agree"}


def test_salad_critic_requires_findings_when_not_ok():
    valid = {"ok": True, "fatal": [], "fixable": [], "notes": ""}
    validate("salad_critic", valid)

    for key, finding in (("fixable", FIXABLE), ("fatal", FATAL)):
        ok_with_finding = dict(valid, **{key: [finding]})
        with pytest.raises(SchemaViolation):
            validate("salad_critic", ok_with_finding)
        validate("salad_critic", dict(ok_with_finding, ok=False))

    with pytest.raises(SchemaViolation):
        validate("salad_critic", dict(valid, ok=False))


@pytest.mark.parametrize("bad", [
    {"ok": False, "fatal": [], "fixable": [dict(FIXABLE, kind="dimension_error")], "notes": ""},   # fatal kind
    {"ok": False, "fatal": [dict(FATAL, kind="missing_definition")], "fixable": [], "notes": ""},  # fixable kind
    {"ok": False, "fatal": [], "fixable": [dict(FIXABLE, severity="low")], "notes": ""},
    {"ok": False, "fatal": [], "fixable": [{"kind": "notation_clash", "where": "x"}], "notes": ""},
    {"ok": False, "fatal": [], "fixable": [dict(FIXABLE, where="")], "notes": ""},
    {"ok": False, "fatal": [], "fixable": [FIXABLE] * 9, "notes": ""},
    {"ok": False, "undefined_symbols": ["theta"], "dimension_errors": [], "unfalsifiable_claims": [], "notes": ""},
])
def test_salad_critic_findings_are_strict(bad):
    with pytest.raises(SchemaViolation):
        validate("salad_critic", bad)


REDUCTION = {
    "reduces_to": "rnn", "mapping": "h_t = tanh(W h + U x) equals S_t up to the readout",
    "confidence": 0.8, "difference_if_any": "", "radical_enough": False, "why": "Same state update.",
    "textbook_transplant": True, "textbook_why": "An Elman network as the textbook gives it.",
    "comparators": [{"id": "elman", "base_family": "rnn", "base_form": "Elman network", "update": "h_t = tanh(W h + U x)",
                     "read": "y_t = V h_t", "transformations": [], "parameter_mapping": "W, U and V as in the card",
                     "limit_only": False}],
}


def test_reduction_critic_records_the_textbook_question():
    for key in ("textbook_transplant", "textbook_why"):
        missing = {k: v for k, v in REDUCTION.items() if k != key}
        with pytest.raises(SchemaViolation, match=key):
            validate("reduction_critic", missing)
    with pytest.raises(SchemaViolation):
        validate("reduction_critic", dict(REDUCTION, textbook_transplant="yes"))
    with pytest.raises(SchemaViolation):
        validate("reduction_critic", dict(REDUCTION, textbook_why=""))


def test_theory_card_accepts_optional_core_idea_transplant_and_extensions():
    card = load_example()
    assert "core_idea" not in card and "source_transplant" not in card
    validate("theory_card", card)
    card.update(core_idea="Tokens bind to phases and a running complex sum is read out.",
                source_transplant={"object": "phasor superposition", "source_ids": ["2401.00001"],
                                   "change": "the phases are learned"},
                extensions=[{"name": "decay", "description": "optional decay",
                             "equations": [{"latex": "S_t = g S_{t-1}", "plain": "S = g * S"}]}])
    validate("theory_card", card)
    for broken in ({"source_transplant": {"object": "x", "source_ids": ["a"], "change": "c"}},     # object too short
                   {"source_transplant": {"object": "phasor", "source_ids": [], "change": "c"}},
                   {"source_transplant": {"object": "phasor", "source_ids": ["a"]}},
                   {"core_idea": "short"},
                   {"extensions": [{"name": "decay"}]}):
        with pytest.raises(SchemaViolation):
            validate("theory_card", dict(card, **broken))


def test_reduction_critic_needs_a_mapping_when_it_claims_a_reduction():
    validate("reduction_critic", REDUCTION)
    validate("reduction_critic", dict(REDUCTION, reduces_to="none", mapping=None, comparators=[]))

    with pytest.raises(SchemaViolation):
        validate("reduction_critic", dict(REDUCTION, mapping=None))
    with pytest.raises(SchemaViolation):
        validate("reduction_critic", dict(REDUCTION, reduces_to="unknown_thing"))
    with pytest.raises(SchemaViolation):
        validate("reduction_critic", dict(REDUCTION, confidence=1.5))


def test_prior_art_needs_evidence_unless_novel():
    validate("prior_art", {"label": "novel", "evidence": [], "notes": "no hits"})
    evidence = [{"source_id": "arxiv:0000.00001", "title": "T", "url": "https://example.org/x", "relation": "same"}]
    validate("prior_art", {"label": "prior_art_tested", "evidence": evidence, "notes": ""})

    with pytest.raises(SchemaViolation):
        validate("prior_art", {"label": "prior_art_tested", "evidence": [], "notes": ""})
    with pytest.raises(SchemaViolation):
        validate("prior_art", {"label": "unclear", "evidence": evidence, "notes": ""})


def test_feasible_judgement_cannot_carry_fatal_issues():
    obligation = {"name": "complexity", "status": "discharged", "argument": "One update per token."}
    validate("feasibility_judge", {"verdict": "feasible", "obligations": [obligation],
                                   "fatal_issues": [], "confidence": 0.7})
    with pytest.raises(SchemaViolation):
        validate("feasibility_judge", {"verdict": "feasible", "obligations": [obligation],
                                       "fatal_issues": ["capacity bound violated"], "confidence": 0.7})
    validate("feasibility_judge", {"verdict": "infeasible", "obligations": [obligation],
                                   "fatal_issues": ["capacity bound violated"], "confidence": 0.9})


def test_disprover_and_analyst_need_at_least_one_item():
    attack = {"target_claim": "recall K=3", "counterexample": "keys 0,3,5", "status": "succeeds"}
    validate("disprover", {"attacks": [attack]})
    with pytest.raises(SchemaViolation):
        validate("disprover", {"attacks": []})

    claim = {"claim": "recall K=3 is exact", "status": "untested", "evidence": "no bench run"}
    validate("analyst", {"claims": [claim]})
    with pytest.raises(SchemaViolation):
        validate("analyst", {"claims": [dict(claim, status="maybe")]})


RECEIPT = {"query": "modern hopfield networks", "source": "arxiv", "ts": "2026-10-09T12:00:00+00:00",
           "result_ids": ["arxiv:2008.02217"]}


def test_literature_receipt_shape():
    validate("literature_receipt", RECEIPT)
    validate("literature_receipt", dict(RECEIPT, card_id="card-00000001", result_ids=[]))
    with pytest.raises(SchemaViolation):
        validate("literature_receipt", dict(RECEIPT, source="google"))
    with pytest.raises(SchemaViolation):
        validate("literature_receipt", {k: v for k, v in RECEIPT.items() if k != "result_ids"})


def test_bench_result_requires_per_task_baselines_and_slopes():
    summary = {"per_task": {"recall": {"kl": 0.01}}, "baselines": {"ngram": {"kl": 0.2}},
               "scaling_slopes": {"time": 1.02}}
    validate("bench_result", summary)
    with pytest.raises(SchemaViolation):
        validate("bench_result", {k: v for k, v in summary.items() if k != "scaling_slopes"})
    with pytest.raises(SchemaViolation):
        validate("bench_result", dict(summary, scaling_slopes={"time": "fast"}))


def test_load_schema_addresses_schema_files_by_name():
    assert load_schema("theory_card")["title"] == "Bigbrane theory card"
    with pytest.raises(FileNotFoundError):
        load_schema("no_such_schema")
