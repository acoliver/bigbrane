"""KD5: an out-of-scope family named by the S0 arm classifier is a flag, not a kill. It rests on argument alone, and a
lifted or growing state can make any mechanism fit a family's form (KD4). S2's counted comparison decides
`excluded_family:<id>`.

The three batch cards killed at S0 as `excluded_family:ssm` (discovery-live-20261017) are replayed from their saved
classifier outputs (fixtures/kd5_s0_flags_20261017, sha256 in SHA256SUMS), with no model call. The plainly excluded
control, the gated diagonal recurrence card-17100009 of the novelty probe, is stopped at S2 from its saved reduction
critic artifact (the KD4 fixtures) and, end to end with scripted models, after a flagged S0 pass."""
from hybrid.tests.research.critic_answers import KNOWN, NOT_KNOWN
import asyncio
import json
from pathlib import Path

import pytest

from hybrid.research import archive as arch
from hybrid.research.stages import g_generate as gg
from hybrid.tests.research.test_generation_arms import (CLOSED_CELL, OPEN_CELL, arms_cfg, classifier, events,
                                                        make_orchestrator, script)
from hybrid.tests.research.test_kd4_counted_family import excluded_id, probe
from hybrid.tests.research.test_orchestrator_fakes import comparison_check, declared
from hybrid.tests.research.test_difference import critic

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kd5_s0_flags_20261017"
BATCH_CARDS = ("card-62785c10", "card-c5524960", "card-97baae43")
FLAG = "S0 family flag, unconfirmed by any executed check; S2's counted comparison decides: "


def saved(card_id: str) -> dict:
    return json.loads((FIXTURES / f"classifier-{card_id}.json").read_text(encoding="utf-8"))


def replaying(card_id: str):
    return lambda _index, _user: saved(card_id)["parsed"]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261017T120000Z"


@pytest.mark.parametrize("card_id", BATCH_CARDS)
def test_the_batch_classifier_outputs_no_longer_kill_at_s0(tmp_path, run_dir, card_id):
    recorded = saved(card_id)
    assert recorded["parsed"]["family"] == "ssm" and recorded["recorded_gate"] == "excluded_family:ssm"
    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1}), script=script(cls=replaying(card_id)))
    asyncio.run(orch.run())
    stored = orch.store.get_card(orch.store.list_cards()[0])
    s0 = stored["status"]["verdicts"]["S0"]
    assert s0["verdict"] == "pass"
    assert not any("excluded_family" in reason for reason in s0["reasons"][:1])
    assert FLAG + "ssm: " + recorded["parsed"]["family_why"] in s0["reasons"]
    assigned = events(orch, "archive_assigned")[0]
    assert assigned["admitted"] and assigned["gate"] is None and assigned["closed_by"] is None
    assert events(orch, "arm_draft")[0]["outcome"] == "admitted"
    assert events(orch, "card_created")[0]["family"] == "ssm"


def test_a_card_with_no_family_has_no_flag(tmp_path, run_dir):
    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1}), script=script())
    asyncio.run(orch.run())
    reasons = orch.store.get_card(orch.store.list_cards()[0])["status"]["verdicts"]["S0"]["reasons"]
    assert not any(reason.startswith("S0 family flag") for reason in reasons)


def test_the_closed_cell_gate_still_stops_an_excluded_family_card_at_s0(tmp_path, run_dir):
    """Not an argument about a mapping: the card's commitments lie in the reference cells of an excluded family."""
    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1}),
                                script=script(cls=classifier([CLOSED_CELL], "ssm")))
    asyncio.run(orch.run())
    s0 = orch.store.get_card(orch.store.list_cards()[0])["status"]["verdicts"]["S0"]
    assert s0["verdict"] == "fail" and s0["reasons"][0].startswith("gate: not_radical:R1: ")


def test_the_duplicate_gate_still_applies_to_a_flagged_card(tmp_path, run_dir):
    from hybrid.tests.research.test_generation_arms import Drafts

    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 2}),
                                script=script(drafts=Drafts(same=True), cls=classifier(family="ssm")))
    asyncio.run(orch.run())
    codes = [d["kill_code"] for d in events(orch, "arm_draft")]
    assert codes[0] is None and codes[1] is not None and codes[1].startswith("duplicate:")


# ----- the control: a plainly excluded card is stopped, at S2 -----------------------------------------------------------


def test_the_gated_diagonal_recurrence_control_is_stopped_at_s2_by_its_counted_comparison():
    assert excluded_id(probe("card-17100009")) == "ssm"


def full_line(tmp_path, run_dir, crit):
    cfg = arms_cfg(tmp_path, quota={"A1": 1}, draft_only=False)
    cfg["outcomes"]["enabled"] = True
    cfg["s2"]["family_exclusion"] = True
    orch, _ = make_orchestrator(run_dir, cfg=cfg, script={**script(cls=classifier(family="ssm")),
                                                          "reduction_critic": crit,
                                                          "reduction_check": comparison_check(0.0)})
    asyncio.run(orch.run())
    return orch, orch.store.get_card(orch.store.list_cards()[0])["status"]["verdicts"]


def test_a_flagged_card_whose_reduction_is_counted_in_ssm_dies_at_s2(tmp_path, run_dir):
    crit = critic(reduces_to="ssm", family="ssm", **KNOWN, mapping="h_t = a_t h_{t-1} + b_t x_t",
                  comparators=[declared("ssm", "diag_ssm")])
    _, verdicts = full_line(tmp_path, run_dir, crit)
    assert verdicts["S0"]["verdict"] == "pass" and verdicts["S1"]["verdict"] == "pass"
    assert verdicts["S2"]["reasons"][-1].startswith("gate: excluded_family:ssm:")


def test_a_flagged_card_with_no_reduction_goes_on_past_s2(tmp_path, run_dir):
    crit = critic(reduces_to="none", family="ssm", **NOT_KNOWN, mapping=None, confidence=0.1,
                  family_why="a lifted state could be argued")
    _, verdicts = full_line(tmp_path, run_dir, crit)
    assert verdicts["S0"]["verdict"] == "pass"
    assert verdicts["S2"]["verdict"] == "pass"


def test_the_families_block_asks_for_a_mapping_without_a_lifted_or_growing_state():
    families = arch.load_families(Path("hybrid/config/families.toml"), arch.load_archive_config(Path("hybrid/config/archive.toml")))
    block = gg.families_block(families)
    assert "dies at generation" not in block and "one-hot lift" in block
