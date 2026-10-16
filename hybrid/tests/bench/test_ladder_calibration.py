"""Rung 1 calibration acceptance (IMPROVEMENT-PLAN.md B.5).

Two parts:
1. The criterion functions on hand-made results, so each verdict rule is pinned down. Criteria 2, 4, 6 and 9 and
   the execution policy follow decision B10 (2026-10-16); the stored diag3 results are read back under them.
2. The calibration gate (`python -m hybrid.bench.ladder gate`) on fixture reports. Whether rung 1 is calibrated is
   an experiment result, read from the stored report (hybrid/bench/calibration/rung1.json) by the gate command,
   not a property of the code under test. These tests pin down the gate's rule: the report must belong to the
   current ladder code (a change to the runner, the baselines or the tasks needs a new calibration), and every
   required criterion must be recorded as passed; anything else, including a missing report, is "not calibrated"
   and a non-zero exit.
"""
from __future__ import annotations

import copy
import json

import pytest

from hybrid.bench import ladder, ladder_criteria as C

CFG = ladder.load_config(ladder.BENCH_DIR / "configs" / "ladder.toml")
SEEDS = C.SEEDS
KS = CFG["tasks"]["mqar256"]["eval_k"]


def result(sets, lr=0.001):
    return {"key": {"lr": lr}, "sets": {name: {"acc": acc, "seq_acc": acc, **extra} for name, (acc, extra) in sets.items()}}


def points_with(cells):
    return {cell: {seed: result(sets) for seed, sets in seeds.items()} for cell, seeds in cells.items()}


def per_seed(sets):
    """The same eval sets on every acceptance seed."""
    return {s: dict(sets) for s in SEEDS}


def mqar_cell(k8, k64):
    return per_seed({"K8": (k8, {}), "K64": (k64, {})})


# ------------------------------------------------------------------------------------------------ criterion 1

def test_the_acceptance_seeds_are_the_post_selection_seeds_after_the_seed_0_sweep():
    assert C.SEEDS == (1, 2, 3) and ladder.SWEEP_SEED == 0 and ladder.DEFAULT_SEEDS == [0, 1, 2, 3]


def test_criterion_1_needs_every_seed_and_every_k_at_or_above_095():
    good = per_seed({"K%d" % k: (1.0, {}) for k in KS})
    assert C.criterion_1(CFG, points_with({("mqar256", "tf-d64"): good}))["status"] == "pass"
    bad = {**good, 3: {**good[3], "K64": (0.9, {})}}
    assert C.criterion_1(CFG, points_with({("mqar256", "tf-d64"): bad}))["status"] == "fail"
    partial = {s: good[s] for s in SEEDS[:2]}
    assert C.criterion_1(CFG, points_with({("mqar256", "tf-d64"): partial}))["status"] == "incomplete"


def test_a_sweep_seed_0_run_is_not_an_acceptance_seed():
    only_sweep_and_two = {0: {"K%d" % k: (1.0, {}) for k in KS}, 1: {"K%d" % k: (1.0, {}) for k in KS},
                          2: {"K%d" % k: (1.0, {}) for k in KS}}
    assert C.criterion_1(CFG, points_with({("mqar256", "tf-d64"): only_sweep_and_two}))["status"] == "incomplete"


# ------------------------------------------------------------------------------------------------ criterion 2

def c2_cells(la=(0.9, 0.5), gdn=(0.9, 0.5), m2=(0.9, 0.5), tf=(1.0, 0.97)):
    cells = {("mqar512", "la-d32"): mqar_cell(*la), ("mqar512", "gdn-d32"): mqar_cell(*gdn),
             ("mqar512", "m2-s16-d32"): mqar_cell(*m2), ("mqar512", "tf-d32"): mqar_cell(*tf)}
    return cells


def test_criterion_2_passes_when_every_model_learns_and_the_drops_hold():
    crit = C.criterion_2(CFG, points_with(c2_cells()))
    assert crit["status"] == "pass", crit
    assert crit["numbers"]["gdn-d32"]["outcome"] == {"1": "pass", "2": "pass", "3": "pass"}


def test_criterion_2_fails_a_model_that_learned_and_keeps_its_accuracy_at_high_k():
    crit = C.criterion_2(CFG, points_with(c2_cells(gdn=(0.9, 0.8))))
    assert crit["status"] == "fail" and crit["numbers"]["gdn-d32"]["verdict"] == "fail"


def test_criterion_2_fails_attention_that_loses_more_than_005():
    assert C.criterion_2(CFG, points_with(c2_cells(tf=(1.0, 0.9))))["status"] == "fail"


def test_criterion_2_is_unresolved_for_a_fixed_state_model_that_never_learns():
    # Under the old rule K8 .002 and K64 .001 passed the attention drop rule and failed the fixed-state one on noise.
    crit = C.criterion_2(CFG, points_with(c2_cells(gdn=(0.002, 0.001))))
    assert crit["status"] == "unresolved", crit
    assert crit["numbers"]["gdn-d32"]["outcome"] == {"1": "unlearned", "2": "unlearned", "3": "unlearned"}
    assert "never learned the control" in crit["detail"]


def test_criterion_2_is_unresolved_for_attention_that_never_learns():
    crit = C.criterion_2(CFG, points_with(c2_cells(tf=(0.002, 0.002))))
    assert crit["status"] == "unresolved" and crit["numbers"]["tf-d32"]["verdict"] == "unresolved"


def test_criterion_2_is_unresolved_when_only_some_seeds_learn_the_control():
    cells = c2_cells()
    cells[("mqar512", "m2-s16-d32")] = {1: {"K8": (0.02, {}), "K64": (0.01, {})}, 2: {"K8": (0.9, {}), "K64": (0.3, {})},
                                       3: {"K8": (0.9, {}), "K64": (0.3, {})}}
    crit = C.criterion_2(CFG, points_with(cells))
    assert crit["status"] == "unresolved"
    assert crit["numbers"]["m2-s16-d32"]["outcome"] == {"1": "unlearned", "2": "pass", "3": "pass"}


def test_criterion_2_a_learned_seed_that_misses_the_drop_fails_even_beside_an_unlearned_one():
    cells = c2_cells()
    cells[("mqar512", "gdn-d32")] = {1: {"K8": (0.002, {}), "K64": (0.001, {})},
                                    2: {"K8": (0.9, {}), "K64": (0.85, {})}, 3: {"K8": (0.9, {}), "K64": (0.3, {})}}
    assert C.criterion_2(CFG, points_with(cells))["status"] == "fail"


def test_criterion_2_low_k_control_threshold_is_the_predeclared_one():
    just_above = C.criterion_2(CFG, points_with(c2_cells(la=(C.LEARNED_MIN_ACC, 0.0))))
    assert just_above["numbers"]["la-d32"]["verdict"] == "pass"
    just_below = C.criterion_2(CFG, points_with(c2_cells(la=(C.LEARNED_MIN_ACC - 0.01, 0.0))))
    assert just_below["numbers"]["la-d32"]["verdict"] == "unresolved"


def test_criterion_2_is_incomplete_without_every_seed_or_model():
    cells = c2_cells()
    cells[("mqar512", "la-d32")] = {s: v for s, v in cells[("mqar512", "la-d32")].items() if s != 3}
    assert C.criterion_2(CFG, points_with(cells))["status"] == "incomplete"
    del cells[("mqar512", "tf-d32")]
    assert C.criterion_2(CFG, points_with(cells))["status"] == "incomplete"


# ------------------------------------------------------------------------------------------------ criterion 3

def test_criterion_3_allows_one_inversion_within_noise():
    def cell(acc):
        return per_seed({"K64": (acc, {})})
    cells = {("mqar512", "la-d32"): cell(0.2), ("mqar512", "la-d64"): cell(0.5), ("mqar512", "la-d128"): cell(0.49)}
    assert C.criterion_3(CFG, points_with(cells))["status"] == "pass"
    cells[("mqar512", "la-d128")] = cell(0.1)
    assert C.criterion_3(CFG, points_with(cells))["status"] == "fail"


# ------------------------------------------------------------------------------------------------ criterion 4

def c4_cell(within, in_field, beyond):
    """Buckets for a w 32, two-layer window model: 1-8 and 17-32 are inside the window, 33-64 is reachable
    through the second layer, 65-128 is wholly beyond the 64-token receptive field."""
    dist = {"distance": {"1-8": {"acc": within, "n": 100}, "33-64": {"acc": in_field, "n": 100},
                         "65-128": {"acc": beyond, "n": 100}}}
    return per_seed({"K%d" % k: (0.5, dist) for k in KS})


def c4_cells(window=(1.0, 0.8, 0.01), gdn=(1.0, 0.9, 0.4), m2=(1.0, 0.9, 0.2)):
    return {("mqar256", "swa-w32-d64"): c4_cell(*window), ("mqar256", "gdn-d64"): c4_cell(*gdn),
            ("mqar256", "m2-s32-d64"): c4_cell(*m2)}


def test_criterion_4_measures_beyond_the_end_to_end_receptive_field_not_one_window():
    crit = C.criterion_4(CFG, points_with(c4_cells()))
    assert crit["status"] == "pass", crit
    assert crit["numbers"]["receptive_field"] == {"window": 32, "layers": 2, "tokens": 64}
    # The 33-64 bucket is reachable by the two window layers: its .8 accuracy is not counted as a limit failure.
    assert crit["numbers"]["swa-w32-d64"]["1"]["beyond_field"] == 0.01
    assert crit["numbers"]["swa-w32-d64"]["1"]["within_field"] == pytest.approx(0.9, abs=1e-4)


def test_criterion_4_fails_a_window_model_that_still_answers_beyond_its_field():
    assert C.criterion_4(CFG, points_with(c4_cells(window=(1.0, 0.8, 0.5))))["status"] == "fail"


def test_criterion_4_fails_when_no_rival_beats_the_window_on_a_seed():
    cells = c4_cells(gdn=(1.0, 0.9, 0.001), m2=(1.0, 0.9, 0.002))
    assert C.criterion_4(CFG, points_with(cells))["status"] == "fail"


def test_criterion_4_compares_all_retained_state_including_convolution_buffers():
    crit = C.criterion_4(CFG, points_with(c4_cells()))
    window_spec = CFG["models"]["swa-w32-d64"]
    assert crit["numbers"]["swa-w32-d64"]["state_floats_total"] == 8192
    assert crit["numbers"]["gdn-d64"]["state_floats_total"] == 9344
    assert crit["numbers"]["m2-s32-d64"]["state_floats_total"] == 9344
    assert crit["numbers"]["gdn-d64"]["state_floats_mixer"] == 8192
    assert crit["numbers"]["state_ratio_to_window"] == {"gdn-d64": 1.1406, "m2-s32-d64": 1.1406}
    assert window_spec["hyper"]["window"] == 32


def test_criterion_4_fails_when_the_rivals_hold_much_more_state_than_the_window():
    cfg = copy.deepcopy(CFG)
    cfg["models"]["gdn-d64"]["d"] = 256
    cfg["models"]["m2-s32-d64"]["hyper"] = {"d_state": 128}
    crit = C.criterion_4(cfg, points_with(c4_cells()))
    assert crit["status"] == "fail" and max(crit["numbers"]["state_ratio_to_window"].values()) > C.STATE_MATCH_MAX_RATIO


def test_criterion_4_is_incomplete_without_seeds_or_without_queries_beyond_the_field():
    cells = c4_cells()
    cells[("mqar256", "gdn-d64")] = {s: v for s, v in cells[("mqar256", "gdn-d64")].items() if s != 2}
    assert C.criterion_4(CFG, points_with(cells))["status"] == "incomplete"
    short = {"distance": {"1-8": {"acc": 1.0, "n": 100}, "33-64": {"acc": 0.8, "n": 100}}}
    short_cell = per_seed({"K%d" % k: (0.5, short) for k in KS})
    cells = {cell: short_cell for cell in c4_cells()}
    crit = C.criterion_4(CFG, points_with(cells))
    assert crit["status"] == "incomplete" and "receptive field" in crit["detail"]


# ------------------------------------------------------------------------------------------------ criterion 6

COPY_MODELS = ("tf-d64", "tf-halibi-d64", "la-d64", "gdn-d64", "m2-s32-d64")
FIXED = ("la-d64", "gdn-d64", "m2-s32-d64")


LONG_LENGTH = CFG["tasks"]["copy_long"]["eval_lengths"][0]


def c6_points(tf=1.0, fixed_copy=0.0, lookup=0.95, long_set=None, long_acc=0.0, with_long_task=True):
    """Copy L32 and L64 (plus an optional long set) for every model, prefix lookup L30 for the fixed-state ones and,
    unless with_long_task is False, the fixed-state models' whole-string accuracy `long_acc` on the copy_long task."""
    cells = {}
    if with_long_task:
        for model in FIXED:
            cells[("copy_long", model)] = per_seed({"L%d" % LONG_LENGTH: (long_acc, {})})
    for model in COPY_MODELS:
        acc = tf if model.startswith("tf") else fixed_copy
        sets = {"L32": (1.0, {}), "L64": (acc, {})}
        if long_set:
            sets[long_set[0]] = (acc if model.startswith("tf") else long_set[1], {})
        cells[("copy", model)] = per_seed(sets)
    for model in FIXED:
        cells[("lookup_prefix", model)] = per_seed({"L30": (lookup, {}), "L100": (lookup, {})})
    return points_with(cells)


def cfg_without_long_copy():
    cfg = copy.deepcopy(CFG)
    del cfg["tasks"][C.LONG_COPY_TASK]
    return cfg


def test_criterion_6_passes_with_the_empirical_part_and_the_tested_bound_on_the_long_copy_task():
    crit = C.criterion_6(CFG, c6_points())
    assert crit["status"] == "pass", crit
    assert crit["parts"]["empirical"]["status"] == "pass" and crit["parts"]["bound"]["status"] == "pass"
    checked = crit["numbers"]["bound_part"]["checked"]
    assert sorted((row["model"], row["set"]) for row in checked) == sorted((m, "L%d" % LONG_LENGTH) for m in FIXED)
    assert all(row["max_string_acc"] <= C.BOUND_TESTABLE_ACC for row in checked)


def test_criterion_6_without_a_long_copy_task_reports_the_unreached_bound():
    crit = C.criterion_6(cfg_without_long_copy(), c6_points(with_long_task=False))
    assert crit["status"] == "pass", crit
    assert crit["parts"]["empirical"]["status"] == "pass" and crit["parts"]["bound"]["status"] == "info"
    assert "reported, not tested" in crit["parts"]["bound"]["detail"]


def test_criterion_6_is_incomplete_when_the_long_copy_runs_are_missing():
    crit = C.criterion_6(CFG, c6_points(with_long_task=False))
    assert crit["parts"]["bound"]["status"] == "incomplete" and crit["status"] == "incomplete"
    assert "copy_long" in crit["parts"]["bound"]["detail"]


def test_criterion_6_fails_a_fixed_state_model_above_the_cap_on_the_long_copy_task():
    crit = C.criterion_6(CFG, c6_points(long_acc=0.4))
    assert crit["parts"]["bound"]["status"] == "fail" and crit["status"] == "fail"
    assert {v["model"] for v in crit["numbers"]["bound_part"]["violations"]} == set(FIXED)


def test_criterion_6_empirical_part_fails_when_copy_does_not_separate_the_families():
    crit = C.criterion_6(CFG, c6_points(tf=0.0, fixed_copy=0.0))     # transformer ties the fixed-state models at 0
    assert crit["status"] == "fail" and crit["parts"]["empirical"]["status"] == "fail"
    assert crit["parts"]["bound"]["status"] == "pass"       # the long copy task tests the bound whatever the empirical part says


def test_criterion_6_empirical_part_fails_when_a_fixed_state_model_misses_prefix_lookup():
    crit = C.criterion_6(CFG, c6_points(lookup=0.85))
    assert crit["status"] == "fail" and crit["parts"]["empirical"]["status"] == "fail"


def test_criterion_6_is_incomplete_when_a_seed_is_missing():
    pts = c6_points()
    del pts[("copy", "gdn-d64")][3]
    crit = C.criterion_6(CFG, pts)
    assert crit["status"] == "incomplete" and crit["parts"]["empirical"]["status"] == "incomplete"


def test_criterion_6_state_bits_and_payload_bits_are_counted_for_the_summary_grid():
    bound = C.criterion_6(CFG, c6_points())["numbers"]["theorem_2_7_bound"]
    assert {m: b["state_bits_fp32"] for m, b in bound.items()} == {"la-d64": 303104, "gdn-d64": 299008,
                                                                    "m2-s32-d64": 299008}
    assert {m: b["copy_length_at_bound"] for m, b in bound.items()} == {"la-d64": 75776, "gdn-d64": 74752,
                                                                         "m2-s32-d64": 74752}
    top = bound["la-d64"]["lengths"]["L256"]
    assert top["payload_bits"] == 1024 and top["max_string_acc"] == 1.0   # L256 at D16 carries 1024 bits


def test_criterion_6_bound_part_is_tested_where_the_payload_exceeds_the_state_with_margin():
    cfg = cfg_without_long_copy()
    cfg["tasks"]["copy"]["eval_lengths"] = [16, 32, 64, 128, 256, 100000]    # 400000 payload bits > 303104 state bits
    ok = C.criterion_6(cfg, c6_points(long_set=("L100000", 0.0), with_long_task=False))
    assert ok["status"] == "pass" and ok["parts"]["bound"]["status"] == "pass", ok
    over = C.criterion_6(cfg, c6_points(long_set=("L100000", 0.4), with_long_task=False))   # above its cap
    assert over["parts"]["bound"]["status"] == "fail" and over["status"] == "fail"
    assert over["numbers"]["bound_part"]["violations"][0]["set"] == "L100000"


def test_criterion_6_crossing_the_state_count_without_a_bit_margin_is_not_a_test():
    cfg = cfg_without_long_copy()
    cfg["tasks"]["copy"]["eval_lengths"] = [32, 64, 75777]    # La d64 holds 303104 bits; L75777 carries 303108
    bound = C._copy_bound(cfg, ["la-d64"])["la-d64"]["lengths"]["L75777"]
    assert bound["payload_bits"] == 303108 and bound["max_string_acc"] == pytest.approx(0.0625)
    assert bound["max_string_acc"] > C.BOUND_TESTABLE_ACC


# ------------------------------------------------------------------------------------------------ criterion 9

A, B = ("mqar512", "la-d32"), ("mqar512", "tf-d32")


def test_criterion_9_flags_a_seed_that_reverses_a_clear_ordering():
    a = {1: {"K4": (0.9, {})}, 2: {"K4": (0.95, {})}, 3: {"K4": (0.05, {})}}
    b = {s: {"K4": (0.1, {})} for s in SEEDS}
    crit = C.criterion_9(CFG, points_with({A: a, B: b}))
    assert crit["status"] == "fail" and crit["numbers"]["violations"][0]["a"] == "la-d32"
    assert crit["parts"]["order"]["status"] == "fail"
    a[3] = {"K4": (0.8, {})}
    crit = C.criterion_9(CFG, points_with({A: a, B: b}))
    assert crit["status"] == "pass" and crit["parts"]["repeatability"]["status"] == "pass"


def order_status(a, b):
    """Order part of criterion 9 for two models with the given K4 accuracies per seed; the bimodal inputs are
    unstable cells, so the repeatability part is judged in its own tests."""
    pa = {s: {"K4": (v, {})} for s, v in zip(SEEDS, a)}
    pb = {s: {"K4": (v, {})} for s, v in zip(SEEDS, b)}
    crit = C.criterion_9(CFG, points_with({A: pa, B: pb}))
    return {**crit["parts"]["order"], "numbers": crit["numbers"]}


def test_criterion_9_a_difference_within_the_evaluation_resolution_is_a_tie():
    # Mean gap .53 makes the pair eligible; the third seed differs by .005 against the order and is a tie.
    assert order_status((0.9, 0.9, 0.205), (0.1, 0.1, 0.2))["status"] == "pass"
    assert order_status((0.9, 0.9, 0.205), (0.1, 0.1, 0.2))["numbers"]["tied_seed_orders"] == 1


def test_criterion_9_a_reversal_beyond_the_tie_margin_is_not_excused():
    crit = order_status((0.9, 0.9, 0.18), (0.1, 0.1, 0.2))      # .02 against the order, twice the margin
    assert crit["status"] == "fail" and len(crit["numbers"]["violations"]) == 1


def test_criterion_9_two_models_at_the_ceiling_tie_whatever_their_small_difference():
    assert order_status((1.0, 1.0, 0.96), (0.2, 0.2, 0.99))["status"] == "pass"        # .03 reversal, both >= .95
    assert order_status((1.0, 1.0, 0.94), (0.2, 0.2, 0.99))["status"] == "fail"        # a is below the ceiling


def test_criterion_9_pairs_within_the_gap_threshold_are_not_checked_and_no_eligible_pair_is_incomplete():
    crit = order_status((0.5, 0.5, 0.5), (0.45, 0.45, 0.55))
    assert crit["status"] == "incomplete"


def c9_repeat_points(repeat_acc=None, repeat_lr=0.001):
    """GDN d32 learns on two seeds and not on one (unstable onset) next to an LA d32 cell at chance, which ties
    GDN on the seed where GDN does not learn."""
    gdn = {1: {"K8": (0.98, {})}, 2: {"K8": (0.002, {})}, 3: {"K8": (0.98, {})}}
    la = {s: {"K8": (0.0015, {})} for s in SEEDS}
    cells = {("mqar512", "gdn-d32"): gdn, ("mqar512", "la-d32"): la}
    pts = points_with(cells)
    if repeat_acc is not None:
        pts[("mqar512", "gdn-d32-replay1")] = {2: result({"K8": (repeat_acc, {})}, lr=repeat_lr)}
    return pts


def test_criterion_9_repeatability_is_incomplete_for_an_unstable_cell_without_a_same_seed_repeat():
    crit = C.criterion_9(CFG, c9_repeat_points())
    assert crit["parts"]["order"]["status"] == "pass"
    assert crit["parts"]["repeatability"]["status"] == "incomplete" and crit["status"] == "incomplete"
    assert crit["numbers"]["repeatability"]["unstable_cells"] == ["mqar512/gdn-d32"]


def test_criterion_9_repeatability_passes_when_the_repeat_agrees_and_fails_when_it_does_not():
    agree = C.criterion_9(CFG, c9_repeat_points(repeat_acc=0.01))
    assert agree["parts"]["repeatability"]["status"] == "pass" and agree["status"] == "pass", agree
    differ = C.criterion_9(CFG, c9_repeat_points(repeat_acc=0.99))     # chance in the original, learned on the repeat
    assert differ["parts"]["repeatability"]["status"] == "fail" and differ["status"] == "fail"
    assert differ["numbers"]["repeatability"]["checks"][0]["agree"] is False


def test_criterion_9_a_repeat_at_another_learning_rate_is_not_a_same_seed_repeat():
    crit = C.criterion_9(CFG, c9_repeat_points(repeat_acc=0.01, repeat_lr=0.0003))
    assert crit["parts"]["repeatability"]["status"] == "incomplete"


def test_criterion_9_repeat_attempts_are_not_ranked_against_their_own_model():
    pts = c9_repeat_points(repeat_acc=0.99)
    pts[("mqar512", "gdn-d32-replay1")] = {s: result({"K8": (0.99, {})}) for s in SEEDS}
    crit = C.criterion_9(CFG, pts)
    assert crit["numbers"]["pairs_checked"] == 1       # gdn-d32 against la-d32; the replay is not another model


def test_criterion_9_ranks_neither_the_acceptance_repeats_nor_their_originals_twice():
    """Every *-rep1 model of the acceptance slice is a repeat attempt: it is compared with its original, not ranked."""
    repeats = [m for m, spec in CFG["models"].items() if m.endswith("-rep1")]
    assert repeats and all(C._is_repeat(CFG, m) for m in repeats)
    pts = c9_repeat_points()
    pts[("mqar512", "gdn-d32-rep1")] = {s: result({"K8": (0.99, {})}) for s in SEEDS}     # learns on seed 2 too
    pts[("mqar512", "la-d32-rep1")] = {s: result({"K8": (0.9, {})}) for s in SEEDS}       # would reverse la-d32
    crit = C.criterion_9(CFG, pts)
    assert crit["numbers"]["pairs_checked"] == 1 and crit["numbers"]["violations"] == []
    assert crit["parts"]["order"]["status"] == "pass"
    checks = crit["numbers"]["repeatability"]["checks"]
    assert {c["repeat"] for c in checks} == {"gdn-d32-rep1"} and len(checks) == 3
    assert [c["agree"] for c in checks] == [True, False, True]      # seed 2: chance in the original, learned again
    assert crit["parts"]["repeatability"]["status"] == "fail"


def test_criterion_9_steady_cells_need_no_repeat():
    steady = points_with({("mqar512", "gdn-d32"): {s: {"K8": (0.9, {})} for s in SEEDS},
                          ("mqar512", "la-d32"): {s: {"K8": (0.4, {})} for s in SEEDS}})
    crit = C.criterion_9(CFG, steady)
    assert crit["status"] == "pass" and crit["numbers"]["repeatability"]["unstable_cells"] == []


# ------------------------------------------------------------------------------------------------ execution policy

def test_only_container_cpu_runs_are_trusted_acceptance_evidence():
    assert C.result_trusted({"execution": {"mode": "container", "device": "cpu"}, "device": "cpu"})
    assert C.result_trusted({})                                                  # written before host_mps existed
    assert not C.result_trusted({"execution": {"mode": "host_mps"}, "device": "mps"})
    assert not C.result_trusted({"execution": {"mode": "container"}, "device": "mps"})
    assert not C.result_trusted({"execution": {"mode": "host_mps"}, "device": "cpu"})


def test_the_policy_records_the_seeds_and_every_predeclared_threshold():
    policy = C.policy()
    assert policy["seeds"] == [1, 2, 3] and policy["sweep_seed"] == 0
    assert policy["criterion_2"]["learned_min_acc"] == 0.25 and policy["criterion_9"]["tie_margin"] == 0.01


# ------------------------------------------------------------------------------------------------ stored result
# ------------------------------------------------------------------------------------------------ gate

HASH = "a" * 64


def report(code=HASH, statuses=None, devices=("cpu",), trusted=True, seeds=None):
    statuses = statuses or {}
    return {"schema": C.SCHEMA, "code_hash": code, "devices": list(devices), "trusted": trusted,
            "scope": "acceptance" if trusted else "diagnostic", "policy": {"seeds": list(seeds or C.SEEDS)},
            "criteria": {name: {"status": statuses.get(name, "pass"), "detail": "d%s" % name, "numbers": {}}
                         for name in C.REQUIRED_CRITERIA}}


def test_gate_requires_every_b5_stored_result_criterion():
    assert C.REQUIRED_CRITERIA == ("1", "2", "3", "4", "6", "7", "8", "9")


def test_gate_passes_only_a_complete_passing_report_on_the_current_code():
    verdict = C.gate(report(), HASH)
    assert verdict["calibrated"] and verdict["reasons"] == [] and verdict["code_hash"] == HASH


def test_gate_rejects_a_report_from_other_code():
    verdict = C.gate(report(code="b" * 64), HASH)
    assert not verdict["calibrated"]
    assert len(verdict["reasons"]) == 1 and "current ladder code" in verdict["reasons"][0]


@pytest.mark.parametrize("criterion", C.REQUIRED_CRITERIA)
@pytest.mark.parametrize("status", ["fail", "incomplete", "unresolved", "info"])
def test_gate_rejects_any_required_criterion_that_did_not_pass(criterion, status):
    verdict = C.gate(report(statuses={criterion: status}), HASH)
    assert not verdict["calibrated"]
    assert verdict["reasons"] == ["criterion %s is %s: d%s" % (criterion, status, criterion)]


def test_gate_refuses_a_diagnostic_report_even_when_every_criterion_passed():
    verdict = C.gate(report(devices=("cpu", "mps"), trusted=False), HASH)
    assert not verdict["calibrated"]
    assert len(verdict["reasons"]) == 1 and "diagnostic" in verdict["reasons"][0]
    assert "Podman CPU" in verdict["reasons"][0]


def test_gate_refuses_a_report_that_does_not_record_trusted_true():
    rep = report()
    del rep["trusted"]
    assert not C.gate(rep, HASH)["calibrated"]
    rep["trusted"] = "yes"
    assert not C.gate(rep, HASH)["calibrated"]


def test_gate_refuses_a_report_judged_on_other_seeds():
    verdict = C.gate(report(seeds=(0, 1, 2)), HASH)
    assert not verdict["calibrated"] and "acceptance seeds are [1, 2, 3]" in verdict["reasons"][0]
    rep = report()
    del rep["policy"]
    assert not C.gate(rep, HASH)["calibrated"]


def test_gate_rejects_a_missing_criterion_and_a_wrong_schema():
    rep = report()
    del rep["criteria"]["4"]
    rep["schema"] = "something/0"
    verdict = C.gate(rep, HASH)
    assert not verdict["calibrated"]
    assert any("criterion 4 missing" in r for r in verdict["reasons"])
    assert any("schema" in r for r in verdict["reasons"])


def test_gate_command_exit_status(work_dir, monkeypatch, capsys):
    monkeypatch.setattr(ladder, "code_hash", lambda: HASH)
    missing = work_dir / "rung1.json"
    assert ladder.main(["gate", "--report", str(missing)]) == 1
    assert "no stored calibration report" in capsys.readouterr().out
    missing.write_text(json.dumps(report()))
    assert ladder.main(["gate", "--report", str(missing)]) == 0
    assert capsys.readouterr().out.startswith("rung 1 calibrated")
    missing.write_text(json.dumps(report(devices=("cpu", "mps"), trusted=False)))
    assert ladder.main(["gate", "--report", str(missing)]) == 1
    assert "diagnostic" in capsys.readouterr().out
    missing.write_text(json.dumps(report(statuses={"9": "fail"})))
    assert ladder.main(["gate", "--report", str(missing)]) == 1
    assert "criterion 9 is fail" in capsys.readouterr().out


def test_gate_command_defaults_to_the_stored_report_path():
    assert ladder.STORED_CALIBRATION == ladder.BENCH_DIR / "calibration" / "rung1.json"


def _stored_result(out_dir, rid, code, device=None, execution=None):
    result = {"run_id": rid, "status": "ok", "code_hash": code,
              "key": {"task": "mqar256", "model": "tf-d64", "lr": 0.001, "seed": 0, "steps": 1},
              "sets": {"K4": {"acc": 1.0, "seq_acc": 1.0}}, "train": {"steps_run": 1, "params": 1},
              "state_floats": {}}
    if device:
        result["device"] = device
    if execution:
        result["execution"] = execution
    path = out_dir / "runs" / rid / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))


def test_report_takes_its_code_hash_and_devices_from_the_results(work_dir):
    _stored_result(work_dir, "r-1", HASH)
    _stored_result(work_dir, "r-2", HASH, device="mps")
    rep = C.build_report(CFG, work_dir)
    assert rep["code_hash"] == HASH and rep["devices"] == ["cpu", "mps"]
    assert not C.gate(rep, HASH)["calibrated"]                          # one run decides nothing
    _stored_result(work_dir, "r-3", "b" * 64)
    with pytest.raises(ValueError, match="2 code hashes"):
        C.build_report(CFG, work_dir)


def test_a_report_over_host_mps_runs_is_labeled_diagnostic_and_a_container_report_acceptance(work_dir):
    mps_dir, cpu_dir = work_dir / "mps", work_dir / "cpu"
    _stored_result(mps_dir, "r-1", HASH, device="mps", execution={"mode": "host_mps", "device": "mps"})
    mps = C.build_report(CFG, mps_dir)
    assert mps["trusted"] is False and mps["scope"] == "diagnostic" and mps["policy"]["seeds"] == [1, 2, 3]
    assert "scope: diagnostic" in C.render_markdown(mps)
    _stored_result(cpu_dir, "r-1", HASH, execution={"mode": "container", "device": "cpu"})
    cpu = C.build_report(CFG, cpu_dir)
    assert cpu["trusted"] is True and cpu["scope"] == "acceptance"
    mixed = work_dir / "mixed"
    _stored_result(mixed, "r-1", HASH, execution={"mode": "container", "device": "cpu"})
    _stored_result(mixed, "r-2", HASH, device="mps", execution={"mode": "host_mps", "device": "mps"})
    assert C.build_report(CFG, mixed)["trusted"] is False


# ------------------------------------------------------------------------------------------------ stored diag3 results
# The diag3 tranche (results/phase10-diag3-20261013T2009Z, host MPS, code 8e4c6caba4c7) is the evidence behind the
# B10 proposals; its summary.md predicts how the revised criteria read it. The attention d32 control is the diag2 run
# on the same code hash. Both directories are read only; the checks skip when they are absent.

DIAG3 = ladder.ROOT / "results" / "phase10-diag3-20261013T2009Z"
DIAG2 = ladder.ROOT / "results" / "phase10-diag2-20261013T1334Z"
stored = pytest.mark.skipif(not (DIAG3 / "runs").exists() or not (DIAG2 / "runs").exists(),
                            reason="stored diag2/diag3 results are not on this machine")


def diag3_points(with_attention_control=False):
    results = ladder.load_results(DIAG3)
    points = ladder.chosen_points(results, DIAG3)
    if with_attention_control:
        control = [r for r in ladder.load_results(DIAG2) if r["key"]["model"] == "tf-d32"]
        points.update(ladder.chosen_points(control, DIAG2))
    return points


@stored
def test_stored_diag3_criterion_2_reads_five_learned_passes_and_four_unresolved_seeds():
    crit = C.criterion_2(CFG, diag3_points(with_attention_control=True))
    outcomes = {m: crit["numbers"][m]["outcome"] for m in ("la-d32", "gdn-d32", "m2-s16-d32", "tf-d32")}
    assert outcomes == {"la-d32": {"1": "pass", "2": "pass", "3": "pass"},
                        "gdn-d32": {"1": "unlearned", "2": "pass", "3": "unlearned"},
                        "m2-s16-d32": {"1": "unlearned", "2": "unlearned", "3": "pass"},
                        "tf-d32": {"1": "pass", "2": "pass", "3": "pass"}}
    fixed_passes = sum(o == "pass" for m in ("la-d32", "gdn-d32", "m2-s16-d32") for o in outcomes[m].values())
    assert fixed_passes == 5                                      # the summary's 5 of 9 fixed-state checks
    assert crit["status"] == "unresolved"                         # failed on noise before; now an optimization result
    # Drops of the learned seeds are the summary's: GDN seed 2 .834351, Mamba seed 3 .689575, LA .296/.354/.353.
    assert crit["numbers"]["gdn-d32"]["drop"]["2"] == pytest.approx(0.834351, abs=1e-4)
    assert crit["numbers"]["m2-s16-d32"]["drop"]["3"] == pytest.approx(0.689575, abs=1e-4)
    assert [crit["numbers"]["la-d32"]["drop"][s] for s in "123"] == pytest.approx([0.2959, 0.3542, 0.3535], abs=1e-4)


@stored
def test_stored_diag3_criterion_2_without_the_diag2_attention_control_is_incomplete():
    assert C.criterion_2(CFG, diag3_points())["status"] == "incomplete"


@stored
def test_stored_diag3_criterion_9_still_reverses_the_three_eligible_cross_family_comparisons():
    crit = C.criterion_9(CFG, diag3_points())
    assert crit["status"] == "fail" and crit["parts"]["order"]["status"] == "fail"
    assert crit["numbers"]["pairs_checked"] == 3
    assert sorted((v["set"], v["a"], v["b"]) for v in crit["numbers"]["violations"]) == [
        ("K16", "gdn-d32", "la-d32"), ("K4", "gdn-d32", "la-d32"), ("K4", "la-d32", "m2-s16-d32")]
    # Three cells have unstable onset and diag3 holds no same-seed repeat at the chosen rate for them.
    assert crit["parts"]["repeatability"]["status"] == "incomplete"
    assert crit["numbers"]["repeatability"]["unstable_cells"] == ["mqar256/gdn-d64", "mqar512/gdn-d32",
                                                                  "mqar512/m2-s16-d32"]


@stored
def test_stored_diag3_report_is_a_diagnostic_the_gate_refuses():
    rep = C.build_report(CFG, DIAG3)
    assert rep["devices"] == ["mps"] and rep["trusted"] is False and rep["scope"] == "diagnostic"
    verdict = C.gate(rep, rep["code_hash"])
    assert not verdict["calibrated"] and any("diagnostic" in r for r in verdict["reasons"])
