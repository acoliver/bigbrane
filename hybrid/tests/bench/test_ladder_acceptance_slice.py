"""The Phase 10 acceptance slice (decisions B10 and B10t): ladder.toml slice `accept`.

Pins down that the slice builds, that every cell has a same-seed repeat attempt at its own learning rate, that run
ids tell attempts and models apart, that the long-copy length makes the criterion 6 bound testable, and the job
count and runtime estimate that go into B10-SLICE-STATUS.md. Nothing here trains a model.
"""
from __future__ import annotations

import pytest

from hybrid import slots
from hybrid.bench import ladder, ladder_criteria as C, ladder_estimate as E

CONFIG = ladder.BENCH_DIR / "configs" / "ladder.toml"
SLICE = "accept"
FIXED_COPY_MODELS = ("la-d64", "gdn-d64", "m2-s32-d64")


@pytest.fixture(scope="module")
def cfg():
    return ladder.load_config(CONFIG)


@pytest.fixture(scope="module")
def settings(cfg):
    return ladder.slice_settings(cfg, SLICE)


def cells(cfg):
    """(own cells, follower cells) of the slice as {(task, model): cell}."""
    own, followers = {}, {}
    for cell in cfg["slices"][SLICE]["cells"]:
        for model in cell["models"]:
            (followers if "lr_from" in cell else own)[(cell["task"], model)] = cell
    return own, followers


def planned_runs(cfg, work_dir):
    """Every run the Scheduler queues for the slice, in order, with each sweep won by the lowest learning rate."""
    sched = ladder.Scheduler(cfg, None, work_dir, SLICE, 4, log=lambda message: None)
    keys = []
    while sched.queue:
        key = sched.queue.pop(0)
        keys.append(key)
        sched._finished(key, {"train": {"best_val_acc": -key["lr"]}})
    assert not sched.pending
    return keys


# ------------------------------------------------------------------------------------------------ the slice builds

def test_the_acceptance_slice_builds_in_the_container_on_the_acceptance_seeds(cfg, settings):
    assert settings["execution"] == ladder.CONTAINER
    assert settings["seeds"] == [ladder.SWEEP_SEED, *ladder.ACCEPTANCE_SEEDS]
    assert settings["train"] == cfg["train"]
    assert settings["caps"] == {**cfg["caps"], "memory_mb": 4096}
    assert ladder.slice_workers(cfg, SLICE, None) == ladder.DEFAULT_CONTAINER_WORKERS
    assert ladder.slice_workers(cfg, SLICE, 4) == 4


def test_a_job_of_the_slice_takes_one_executor_slot_and_its_tmpfs_fits_the_memory_cap(cfg, settings):
    job_caps = ladder.caps(settings)                       # Caps validates tmp_mb against memory_mb
    assert slots.job_weight(job_caps.memory_mb, 4) == 1


def test_the_slice_covers_the_cells_every_calibration_criterion_reads(cfg):
    own, _ = cells(cfg)
    needed = {("mqar256", m) for m in ("tf-d64", "swa-w32-d64", "la-d64", "gdn-d64", "m2-s32-d64")}       # 1, 4, 8
    needed |= {("mqar512", m) for m in ("tf-d32", "la-d32", "gdn-d32", "m2-s16-d32")}                       # 2
    needed |= {("mqar512", m) for m in ("la-d64", "gdn-d64", "m2-s64-d64")}                                 # 3
    needed |= {("copy", m) for m in ("tf-d64", "tf-halibi-d64", *FIXED_COPY_MODELS)}                       # 6, 8
    needed |= {("lookup_prefix", m) for m in FIXED_COPY_MODELS}                                             # 6
    assert needed <= set(own)
    assert set(own) - needed == {("lookup_prefix", "tf-d64")}       # the one cell kept from "first" beyond them
    assert all(cell["lrs"] == [0.0003, 0.001, 0.003] for cell in own.values())


def test_the_job_count_is_207_and_matches_the_scheduler(cfg, work_dir):
    keys = planned_runs(cfg, work_dir)
    own, followers = cells(cfg)
    sweep = [k for k in keys if (k["task"], k["model"]) in own]
    after = [k for k in keys if (k["task"], k["model"]) in followers]
    assert (len(own), len(followers)) == (21, 27)      # 21 cells, 21 repeats of them, 3 long copy, 3 long-copy repeats
    assert len(sweep) == 21 * (3 + 3) and len(after) == 27 * 3
    assert len(keys) == 207


# ------------------------------------------------------------------------------------------------ repeat attempts

def test_every_cell_has_a_same_seed_repeat_attempt_on_the_acceptance_seeds(cfg):
    own, followers = cells(cfg)
    expected = {(task, model) for (task, model) in own} | {("copy_long", m) for m in FIXED_COPY_MODELS}
    repeated = {}
    for (task, repeat), cell in followers.items():
        if repeat.endswith("-rep1"):
            repeated[(task, repeat[:-len("-rep1")])] = cell
    assert set(repeated) == expected
    for (task, model), cell in repeated.items():
        assert cell["seeds"] == list(ladder.ACCEPTANCE_SEEDS) and cell["steps"] == (
            own[(task, model)]["steps"] if (task, model) in own else own[("copy", model)]["steps"])
        assert ladder.parse_cell_ref(cell["lr_from"]) == (("copy", model) if task == "copy_long" else (task, model))


def test_a_repeat_model_differs_from_its_original_only_by_its_attempt_label(cfg):
    repeats = [m for m in cfg["models"] if m.endswith("-rep1")]
    assert len(repeats) == 11
    for name in repeats:
        base = cfg["models"][name[:-len("-rep1")]]
        spec = cfg["models"][name]
        assert spec["attempt"] == "acc-rep1" and "attempt" not in base
        assert {k: v for k, v in spec.items() if k != "attempt"} == base
        assert E.base_model(cfg, name) == name[:-len("-rep1")]


def test_a_repeat_runs_at_the_learning_rate_chosen_for_its_original_after_the_originals_seeds(cfg, work_dir):
    keys = planned_runs(cfg, work_dir)
    own, followers = cells(cfg)
    chosen = {(k["task"], k["model"]): k["lr"] for k in keys if k["seed"] == 1 and (k["task"], k["model"]) in own}
    assert len(chosen) == 21 and set(chosen.values()) == {0.0003}      # the lowest rate won every simulated sweep
    for (task, model), cell in followers.items():
        base_task, base_model = ladder.parse_cell_ref(cell["lr_from"])
        runs = [k for k in keys if (k["task"], k["model"]) == (task, model)]
        assert [k["seed"] for k in runs] == [1, 2, 3]
        assert {k["lr"] for k in runs} == {chosen[(base_task, base_model)]}
        original_last = max(i for i, k in enumerate(keys) if (k["task"], k["model"]) == (base_task, base_model))
        assert min(i for i, k in enumerate(keys) if (k["task"], k["model"]) == (task, model)) > original_last


def test_a_learning_rate_that_is_chosen_not_assumed_reaches_the_repeats(cfg, work_dir):
    sched = ladder.Scheduler(cfg, None, work_dir, SLICE, 4, log=lambda message: None)
    sweep = [k for k in sched.queue if (k["task"], k["model"]) == ("mqar256", "la-d64")]
    assert [k["lr"] for k in sweep] == [0.0003, 0.001, 0.003]
    for k, acc in zip(sweep, (0.2, 0.6, 0.4)):
        sched._finished(k, {"train": {"best_val_acc": acc}})
    assert sched.queue[:6] == [ladder.run_key("mqar256", model, 0.001, seed, 6000)
                               for model in ("la-d64", "la-d64-rep1") for seed in (1, 2, 3)]


def test_the_long_copy_models_follow_the_copy_cells_rate_and_the_repeats_the_same(cfg, work_dir):
    sched = ladder.Scheduler(cfg, None, work_dir, SLICE, 4, log=lambda message: None)
    sweep = [k for k in sched.queue if (k["task"], k["model"]) == ("copy", "gdn-d64")]
    for k, acc in zip(sweep, (0.9, 0.1, 0.2)):
        sched._finished(k, {"train": {"best_val_acc": acc}})
    queued = {(k["task"], k["model"], k["lr"]) for k in sched.queue[:12]}
    assert queued == {("copy", "gdn-d64", 0.0003), ("copy", "gdn-d64-rep1", 0.0003),
                      ("copy_long", "gdn-d64", 0.0003), ("copy_long", "gdn-d64-rep1", 0.0003)}


# ------------------------------------------------------------------------------------------------ run ids

def test_run_ids_are_unique_across_cells_attempts_learning_rates_and_seeds(cfg, settings, work_dir):
    keys = planned_runs(cfg, work_dir)
    ids = [ladder.run_id(cfg, key, settings) for key in keys]
    assert len(ids) == 207 and len(set(ids)) == 207
    # a sweep of every rate, not only the winning one, also has distinct ids
    own, _ = cells(cfg)
    swept = [ladder.run_key(t, m, lr, ladder.SWEEP_SEED, c["steps"]) for (t, m), c in own.items() for lr in c["lrs"]]
    assert len({ladder.run_id(cfg, k, settings) for k in swept}) == len(swept)


def test_an_attempt_label_alone_separates_the_run_ids_of_the_same_job(cfg, settings):
    original = ladder.run_key("mqar256", "gdn-d64", 0.001, 2, 6000)
    repeat = ladder.run_key("mqar256", "gdn-d64-rep1", 0.001, 2, 6000)
    assert ladder.run_id(cfg, original, settings) != ladder.run_id(cfg, repeat, settings)
    a, b = ladder.build_spec(cfg, original, ["K4"], settings, "cpu"), ladder.build_spec(cfg, repeat, ["K4"], settings, "cpu")
    assert a == b                                   # the job sees no difference: same weights, same data streams


def test_the_long_copy_job_trains_exactly_what_the_copy_job_trains(cfg, settings):
    for model in FIXED_COPY_MODELS:
        copy_key = ladder.run_key("copy", model, 0.001, 1, 6000)
        long_key = ladder.run_key("copy_long", model, 0.001, 1, 6000)
        copy_spec = ladder.build_spec(cfg, copy_key, ["L16"], settings, "cpu")
        long_spec = ladder.build_spec(cfg, long_key, ["L76000"], settings, "cpu")
        for field in ("family", "d", "layers", "hyper", "train", "state_lengths"):
            assert copy_spec[field] == long_spec[field]
        assert copy_spec["task"] == long_spec["task"]       # module, train params and vocabulary
        assert ladder.run_id(cfg, copy_key, settings) != ladder.run_id(cfg, long_key, settings)


# ------------------------------------------------------------------------------------------------ lr_from validation

def write_config(work_dir, cells_text):
    path = work_dir / "ladder.toml"
    path.write_text(CONFIG.read_text() + "\n[slices.bad]\ncells = [\n%s\n]\n" % cells_text)
    return path


@pytest.mark.parametrize("cells_text,message", [
    ('{ task = "copy", models = ["tf-d64-rep1"], lr_from = "copy/tf-d64", lrs = [0.001], steps = 10 }',
     "takes no lrs"),
    ('{ task = "copy", models = ["tf-d64-rep1"], lr_from = "copy/tf-d64", steps = 10 }', "names no cell"),
    ('{ task = "copy", models = ["tf-d64"], lrs = [0.001], steps = 10 },\n'
     '{ task = "copy", models = ["tf-d64-rep1"], lr_from = "copy", steps = 10 }', "not task/model"),
    ('{ task = "copy", models = ["tf-d64"], lrs = [0.001], steps = 10 },\n'
     '{ task = "copy", models = ["tf-d64-rep1"], lr_from = "copy/tf-d64", steps = 10 },\n'
     '{ task = "copy", models = ["tf-d32-rep1"], lr_from = "copy/tf-d64-rep1", steps = 10 }', "names no cell"),
], ids=["lrs", "missing", "malformed", "chain"])
def test_config_rejects_malformed_lr_from_cells(work_dir, cells_text, message):
    with pytest.raises(ValueError, match=message):
        ladder.load_config(write_config(work_dir, cells_text))


def test_a_fixed_rate_cell_queues_its_followers_with_it(work_dir):
    path = write_config(work_dir, '{ task = "copy", models = ["tf-d64"], lrs = [0.001], seeds = [1, 2], steps = 10 },\n'
                                  '{ task = "copy", models = ["tf-d64-rep1"], lr_from = "copy/tf-d64", seeds = [1, 2], '
                                  'steps = 10 }')
    sched = ladder.Scheduler(ladder.load_config(path), None, work_dir, "bad", 1, log=lambda message: None)
    assert [(k["model"], k["lr"], k["seed"]) for k in sched.queue] == [
        ("tf-d64", 0.001, 1), ("tf-d64", 0.001, 2), ("tf-d64-rep1", 0.001, 1), ("tf-d64-rep1", 0.001, 2)]


# ------------------------------------------------------------------------------------------------ long copy length

def test_the_long_copy_length_makes_the_information_bound_testable_for_every_fixed_state_model(cfg):
    task = cfg["tasks"][C.LONG_COPY_TASK]
    (length,) = task["eval_lengths"]
    assert length == 76000 and task["alphabet"] == cfg["tasks"]["copy"]["alphabet"] == 16
    bound = C._copy_bound(cfg, FIXED_COPY_MODELS)
    for model, row in bound.items():
        cap = row["lengths"]["L%d" % length]["max_string_acc"]
        assert cap <= C.BOUND_TESTABLE_ACC, model
        assert row["lengths"]["L%d" % length]["payload_bits"] - row["state_bits_fp32"] >= 128      # a real margin
        assert length >= C.bound_testable_length(row["state_bits_fp32"], 16)


def test_the_shortest_testable_length_is_derived_from_the_counted_state_bits(cfg):
    bits = {m: C.counted_state_bits(cfg, m) for m in FIXED_COPY_MODELS}
    assert bits == {"la-d64": 303104, "gdn-d64": 299008, "m2-s32-d64": 299008}
    assert {m: C.bound_testable_length(b, 16) for m, b in bits.items()} == {
        "la-d64": 75778, "gdn-d64": 74754, "m2-s32-d64": 74754}
    # one symbol shorter than the bound's length leaves the cap above 0.05
    cap = 2.0 ** (bits["la-d64"] - 4 * 75777)
    assert cap > C.BOUND_TESTABLE_ACC >= 2.0 ** (bits["la-d64"] - 4 * 75778)


def test_the_long_copy_length_covers_every_fixed_state_model_that_copies_in_the_slice(cfg):
    own, _ = cells(cfg)
    fixed = {m for (task, m) in own if task == "copy" and cfg["models"][m]["family"] in C.FIXED_FAMILIES}
    assert fixed == set(FIXED_COPY_MODELS)
    (length,) = cfg["tasks"][C.LONG_COPY_TASK]["eval_lengths"]
    for model in fixed:
        assert length >= C.bound_testable_length(C.counted_state_bits(cfg, model), 16), model


def test_only_fixed_state_models_run_the_long_copy_task(cfg):
    own, followers = cells(cfg)
    long_models = {m for (t, m) in followers if t == C.LONG_COPY_TASK}
    assert {m.removesuffix("-rep1") for m in long_models} == set(FIXED_COPY_MODELS)
    assert all(cfg["models"][m]["family"] in C.FIXED_FAMILIES for m in long_models)


def test_a_long_copy_prediction_file_fits_the_job_output_cap(cfg):
    task = cfg["tasks"][C.LONG_COPY_TASK]
    (length,) = task["eval_lengths"]
    pred_bytes = task["n_eval"] * length * 2          # int16 predictions at the L scored positions of each row
    assert pred_bytes < 64 << 20 and pred_bytes < 20 << 20


# ------------------------------------------------------------------------------------------------ estimate

def test_the_estimate_counts_the_scheduler_jobs_and_orders_by_the_measured_rates(cfg, work_dir):
    own, _ = cells(cfg)
    walls = {cell: 100.0 for cell in own}
    one = E.simulate(cfg, SLICE, walls, 1, work_dir / "one")
    four = E.simulate(cfg, SLICE, walls, 4, work_dir / "four")
    assert len(one["jobs"]) == len(four["jobs"]) == 207
    total = 207 * 100.0 + 18 * E.LONG_COPY_EVAL_S
    assert one["elapsed_s"] == pytest.approx(total)
    assert total / 4 <= four["elapsed_s"] < total / 4 + 2 * 100.0 + E.LONG_COPY_EVAL_S


def test_a_cell_nobody_measured_is_costed_from_the_measured_cell_it_derives_from(cfg):
    walls = {("mqar256", "gdn-d64"): 3000.0, ("mqar256", "m2-s32-d64"): 3100.0, ("mqar256", "la-d64"): 2300.0,
             ("copy", "la-d64"): 500.0}
    key = ladder.run_key("mqar512", "gdn-d64-rep1", 0.001, 1, 6000)
    assert E.job_seconds(cfg, walls, key) == 6000.0
    assert E.job_seconds(cfg, walls, ladder.run_key("mqar512", "m2-s64-d64", 0.001, 0, 6000)) == 6200.0
    assert E.job_seconds(cfg, walls, ladder.run_key("copy_long", "la-d64-rep1", 0.001, 1, 6000)) == pytest.approx(
        500.0 + E.LONG_COPY_EVAL_S)
