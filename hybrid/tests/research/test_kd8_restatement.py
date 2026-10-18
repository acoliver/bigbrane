"""KD8: a comparator that fills a universal template's free slot with the card's own computation is a restatement
and never counts. Replays recorded reduction_critic artifacts (hybrid/tests/research/fixtures/kd8, built by
results-wt/kd8/tools/replay_kd8.py from the s2-probe and novelty-probe runs) against the host rule; no live calls."""
import json
from pathlib import Path

import pytest

from hybrid.research.stages.s2_prior_art import component_problem, eligibility, judge_comparisons

RECORDS = json.loads((Path(__file__).parent / "fixtures" / "kd8" / "replay_records.json").read_text())
BY_CARD = {r["card"]: r for r in RECORDS}


def replay(card):
    r = BY_CARD[card]
    outcome, _, judged = judge_comparisons(r["comparisons"], r["comparators"], r["reduces_to"])
    return outcome, sorted(c["declared_id"] for c in judged if c["counted"] and c["within_tolerance"])


@pytest.mark.parametrize("card", ["card-8f2e71cf", "card-136a1e9b", "card-383bd914"])
def test_recorded_restatements_were_counted_and_no_longer_are(card):
    assert BY_CARD[card]["recorded_counted"], "the record must have been counted when it ran"
    outcome, counted = replay(card)
    assert counted == []
    assert outcome == "unverified"


@pytest.mark.parametrize("card", ["card-ea50345f", "card-17100001", "card-17100002", "card-17100003", "card-17100004",
                                  "card-17100005", "card-17100006", "card-17100007", "card-17100009",
                                  "card-1710000a"])
def test_real_reductions_and_probe_mechanisms_still_count(card):
    outcome, counted = replay(card)
    assert outcome == "verified"
    assert counted == BY_CARD[card]["recorded_counted"]


def test_rule_110_as_elman_rnn_is_not_a_standard_rnn():
    outcome, counted = replay("card-17100008")
    assert (outcome, counted) == ("unverified", [])


def comparator(family="rnn", base_form="Elman RNN h_t = f(W h + U x) (Elman 1990)", components=None):
    return {"id": "c", "base_family": family, "base_form": base_form, "update": "u", "read": "r",
            "transformations": [], "parameter_mapping": "m", "limit_only": False, "components": components}


def published(role, name, cite="Elman 1990"):
    return {"role": role, "name": name, "origin": "published", "cite": cite}


def test_standard_elman_cell_counts():
    c = comparator(components=[published("activation", "tanh"), published("readout", "identity")])
    assert component_problem(c) is None and eligibility(c, "rnn") == "counted"


def test_nonstandard_activation_is_a_restatement_even_when_labelled_published():
    c = comparator(components=[published("activation", "H^2/tr(H^2)")])
    assert "not a standard rnn activation" in component_problem(c)
    assert eligibility(c, "rnn") == "restatement"


def test_lstm_gru_allows_only_sigmoid_and_tanh():
    ok = comparator("lstm_gru", "LSTM (Hochreiter and Schmidhuber 1997)",
                    [published("activation", "sigmoid"), published("activation", "tanh")])
    bad = comparator("lstm_gru", "LSTM (Hochreiter and Schmidhuber 1997)", [published("activation", "relu")])
    assert component_problem(ok) is None
    assert component_problem(bad) is not None


def test_no_components_is_not_counted():
    assert eligibility(comparator(components=None), "rnn") == "restatement"
    assert eligibility(comparator(components=[]), "rnn") == "restatement"


def test_card_origin_readout_is_a_restatement():
    c = comparator(components=[published("activation", "tanh"),
                               {"role": "readout", "name": "d^d det", "origin": "card", "cite": ""}])
    assert "readout" in component_problem(c)


@pytest.mark.parametrize("form", ["Moore machine (Hopcroft and Ullman)", "Finite-state automaton",
                                  "Turing machine", "Generic recurrence h_t = f(h_{t-1}, x_t)",
                                  "Universal function approximator"])
def test_universal_classes_never_count(form):
    c = comparator("other_known", form, [published("transition", "named rule", "Someone 2000")])
    assert eligibility(c, "other_known") == "restatement"


def test_named_pushdown_automaton_counts():
    c = comparator("other_known", "Deterministic pushdown automaton for Dyck-2 (Ginsburg and Greibach 1966)",
                   [published("transition", "PDA transition table delta(q, top, x)", "Ginsburg and Greibach 1966")])
    assert eligibility(c, "other_known") == "counted"


def test_other_known_needs_a_cited_published_transition():
    c = comparator("other_known", "Jeffrey's rule", [published("readout", "marginal", "Jeffrey 1965")])
    assert "no published, cited transition" in component_problem(c)
    c = comparator("other_known", "Jeffrey's rule", [published("update", "IPF step", "")])
    assert component_problem(c) is not None


def test_other_families_are_not_checked():
    c = comparator("ssm", "Mamba selective SSM", None)
    assert component_problem(c) is None and eligibility(c, "ssm") == "counted"
