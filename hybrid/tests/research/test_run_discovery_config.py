"""hybrid/config/run-discovery.toml, the Phase 11 live batch of about 30 ideas (DISCOVERY-PIPELINE-PLAN.md section 14,
DECISIONS.md row SR): it loads with the project's loader, drafts the arms of the plan under decision D4, runs stages M
and C in place of S3, and differs from run.toml only where its comments say so."""
import tomllib
from collections import Counter

import pytest

from hybrid.math.runner import MathRunner
from hybrid.research.orchestrator import ROOT, escape_reserve_slots, load_run_config
from hybrid.research.stages import g_generate as gg
from hybrid.research.stages.common import pipeline_order
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_orchestrator_fakes import make_orchestrator

CONFIG = ROOT / "hybrid" / "config"
DISCOVERY = CONFIG / "run-discovery.toml"

# Keys that run-discovery.toml sets on purpose (section, key); every other key equals run.toml.
CHANGED = {
    ("budgets", "max_cards"), ("budgets", "max_calls"), ("budgets", "max_tokens"), ("budgets", "max_wall_s"),
    ("workers", "M"), ("workers", "C"),
    ("generation", "mode"), ("generation", "quota"),
    ("s3", "mode"),
    ("outcomes", "enabled"), ("outcomes", "path"),
}


@pytest.fixture
def run_dir(tmp_path):
    return (tmp_path / "discovery-run").resolve()


def raw(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def test_arms_quota_gives_thirty_ideas_with_the_base_arm_slots_moved_to_a1_and_a2():
    cfg = load_run_config(DISCOVERY)
    quota = cfg["generation"]["quota"]
    assert cfg["generation"]["mode"] == "arms" and not cfg["generation"]["draft_only"]
    # Plan section 14: 6 ideas per arm; decision D4 with no base model: A5's 6 slots go to A1 and A2, 3 each.
    assert quota == {"A1": 9, "A2": 9, "A3": 6, "A4": 6}
    assert sum(quota.values()) == 30
    gen = gg.load_generation_config(ROOT / cfg["generation"]["config"])
    assert gen.arms[gg.BASE_ARM].base_model == ""
    weights = gg.effective_weights(gen, False)
    assert set(quota) == set(weights)
    for arm_id, count in quota.items():
        assert count == pytest.approx(30 * weights[arm_id])


def test_escape_reserve_leaves_exactly_thirty_slots_for_the_quota():
    cfg = load_run_config(DISCOVERY)
    reserve = escape_reserve_slots(cfg["budgets"]["max_cards"], cfg["explore"]["escape_reserve"])
    assert cfg["budgets"]["max_cards"] - reserve == sum(cfg["generation"]["quota"].values())
    assert cfg["explore"]["p_escape"] == 1.0


def test_stages_m_and_c_replace_s3_and_the_dialogue_stage_runs():
    cfg = load_run_config(DISCOVERY)
    assert cfg["s3"]["mode"] == "math"
    assert pipeline_order(cfg) == ("S0", "S1", "R", "S2", "M", "C", "S4", "S5", "S6")
    assert cfg["math"]["profile"] is True and cfg["target"]["enabled"] is True
    assert cfg["workers"]["M"] >= 1 and cfg["workers"]["C"] >= 1


def test_outcomes_are_on_and_the_s2_difference_rules_are_enforced():
    cfg = load_run_config(DISCOVERY)
    assert cfg["outcomes"]["enabled"] is True and cfg["outcomes"]["enforce_r3"] is True
    assert cfg["s2"]["difference_statement"] is True and cfg["s2"]["family_exclusion"] is True


def test_budgets_cover_the_measured_upper_case_and_stay_under_the_backend_ceiling():
    cfg = load_run_config(DISCOVERY)
    budgets = cfg["budgets"]
    # Upper case from the ledgers (see the comments in run-discovery.toml): 43 cards at 11 calls and 0.37M tokens to
    # the end of S2, 12 cards in M and C at 35 calls and 1.0M tokens, 150 calls and 3M tokens for S4 to S6.
    assert budgets["max_calls"] >= 43 * 11 + 12 * 35 + 150
    assert budgets["max_tokens"] >= int(43 * 0.37e6 + 12 * 1.0e6 + 3e6)
    # The calibration copy gave one card-run 36000 s; a batch of many card-runs needs more.
    assert budgets["max_wall_s"] >= 60400
    models = raw(CONFIG / "models.toml")["budgets"]
    assert budgets["max_calls"] < models["max_calls"] and budgets["max_tokens"] < models["max_tokens"]


def test_only_the_documented_keys_differ_from_run_toml():
    shipped, discovery = raw(CONFIG / "run.toml"), raw(DISCOVERY)
    assert shipped.keys() == discovery.keys()
    differing = set()
    for section in shipped:
        assert shipped[section].keys() == discovery[section].keys(), section
        differing |= {(section, key) for key in shipped[section] if shipped[section][key] != discovery[section][key]}
    assert differing == CHANGED


def test_an_orchestrator_builds_from_the_config_with_the_quota_schedule_and_the_math_order(tmp_path, run_dir):
    cfg = load_run_config(DISCOVERY)
    # The store path is the real results/ directory when a run starts; the test must not open it for writing.
    cfg["outcomes"]["path"] = str(tmp_path / "outcomes")
    orch, _ = make_orchestrator(run_dir, cfg=cfg, script=mf.math_script(),
                                math=MathRunner(mf.MathExecutor(tmp_path / "exec"), mf.IMAGE, run_dir))
    assert orch.order == ("S0", "S1", "R", "S2", "M", "C", "S4", "S5", "S6")
    assert orch.budget.max_cards == cfg["budgets"]["max_cards"]
    schedule = orch.generation.schedule
    assert len(schedule) == 30
    assert Counter(schedule) == Counter({"A1": 9, "A2": 9, "A3": 6, "A4": 6})
    assert orch.generation.weights == pytest.approx({"A1": 0.3, "A2": 0.3, "A3": 0.2, "A4": 0.2})
    assert orch.generation.base_status[0] is False


def test_a_quota_naming_the_base_arm_would_stop_the_run_at_start(tmp_path, run_dir):
    cfg = load_run_config(DISCOVERY)
    cfg["outcomes"]["path"] = str(tmp_path / "outcomes")
    cfg["generation"]["quota"] = {"A1": 9, "A2": 9, "A3": 6, "A5": 6}
    with pytest.raises(ValueError, match="cannot run"):
        make_orchestrator(run_dir, cfg=cfg, script=mf.math_script(),
                          math=MathRunner(mf.MathExecutor(tmp_path / "exec"), mf.IMAGE, run_dir))
