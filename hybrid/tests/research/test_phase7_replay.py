"""Phase 7 exit criterion (DISCOVERY-PIPELINE-PLAN.md section 17): the S2 reductions of runs/run-20261010T181300Z,
replayed offline through the real S2 stage with the recorded critic answers and check outcomes, get the expected kill
codes. Fixture: fixtures/phase7_replay/run-20261010T181300Z.json, written by phase7_replay.write_fixture (read-only on
the source run)."""
from collections import Counter

import pytest

from hybrid.tests.research import phase7_replay


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    return phase7_replay.replay(phase7_replay.load_fixture(), tmp_path_factory.mktemp("phase7-replay"))


def test_the_fixture_holds_every_card_that_reached_the_reduction_critic():
    fixture = phase7_replay.load_fixture()
    assert fixture["source_run"] == phase7_replay.SOURCE_RUN and len(fixture["cards"]) == 21
    assert Counter(e["original"]["kill_codes"][0] for e in fixture["cards"]
                   if e["original"]["terminal_stage"] == "S2") == {
        "reduces_to_known:other_known": 12, "reduces_to_known:linear_attention": 3, "not_radical:R2": 3,
        "reduces_to_known:hopfield_or_kernel_memory": 1}


def test_every_s2_death_gets_its_expected_phase7_code(rows):
    died = [r for r in rows if r["expected_codes"]]
    assert len(died) == 19
    assert [(r["card_id"], r["replay_codes"]) for r in died] == [(r["card_id"], r["expected_codes"]) for r in died]


def test_the_linear_attention_and_delta_rule_reductions_are_excluded_as_families(rows):
    by_card = {r["card_id"]: r for r in rows}
    assert by_card["card-1184f7a7"]["replay_codes"] == ["excluded_family:linear_attention"]   # H3, diagonal
    assert by_card["card-537e161c"]["replay_codes"] == ["excluded_family:delta_rule"]         # MesaNet / RLS
    assert by_card["card-6de11b01"]["replay_codes"] == ["excluded_family:delta_rule"]         # gated delta rule
    for card_id in ("card-1184f7a7", "card-537e161c", "card-6de11b01"):
        assert by_card[card_id]["replay_gate"].startswith("gate: excluded_family:")
    # the other reductions and the not-radical verdicts keep their original codes
    unchanged = [r for r in rows if r["expected_codes"] and not r["expected_codes"][0].startswith("excluded")]
    assert len(unchanged) == 16 and all(r["replay_codes"] == r["original_codes"] for r in unchanged)


def test_the_two_cards_that_passed_s2_still_pass_its_reduction_gates(rows):
    passed = {r["card_id"]: r for r in rows if not r["expected_codes"]}
    assert sorted(passed) == ["card-46dbb349", "card-c9615069"]
    assert all(r["replay_s2"] == "pass" for r in passed.values())
