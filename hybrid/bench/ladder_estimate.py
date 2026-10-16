"""Job count and runtime estimate of a ladder slice, from measured one-thread Podman CPU runs.

The count comes from the Scheduler itself (ladder.Scheduler), played forward on a clock: a worker takes the next
queued run, a finished sweep run lets `Scheduler._finished` choose the learning rate and queue the post-selection
seeds and the `lr_from` followers, exactly as a real sweep does. Only the durations are modelled.

Durations (seconds per job) are the mean wall time of the measured runs of the same (task, model) cell in earlier
result directories (results/phase10-20261012: the `first` slice, 6000 steps, six workers on the 8-CPU Podman VM, so
the rates include that contention). A cell nobody measured is derived from a measured one by a stated rule
(DERIVED_FROM_MEASURED), and a copy_long run adds the long evaluation (LONG_COPY_EVAL_S). Repeat attempts cost what
their base model costs: the job spec is the same.

    python -m hybrid.bench.ladder_estimate --config hybrid/bench/configs/ladder.toml --slice accept \
        --rates results/phase10-20261012 --scratch results/<unique dir> --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ladder

# Occupied-slot ratio of results/phase10-diag3-20261013T2009Z (63,426 job-seconds over 3 workers in 21,604 s): the
# share of worker time that held a job, the rest being launches, staging and the tail. The elapsed estimate divides
# the simulated elapsed time by it.
OCCUPIED_SLOT_RATIO = 0.978617
# Mean seconds per row of the 152,003-token forward pass, worst of the three fixed-state models (results/b10-slice-
# 20261016/longcopy-probe-75000-isolated.log: 1.89 s linear attention), times 128 rows, times 1.5 for the container.
LONG_COPY_EVAL_S = 1.89 * 128 * 1.5
# (task, model) with no measured run -> (measured task, measured model, factor). N 512 carries twice the tokens per
# step of N 256 at the same width; Mamba-2 state 64 is costed as state 32 (a lower bound for it).
DERIVED_FROM_MEASURED = {("mqar512", "gdn-d64"): ("mqar256", "gdn-d64", 2.0),
                         ("mqar512", "la-d64"): ("mqar256", "la-d64", 2.0),
                         ("mqar512", "m2-s64-d64"): ("mqar256", "m2-s32-d64", 2.0)}
LONG_COPY_TASK = "copy_long"


def base_model(cfg, model):
    """The model a repeat attempt repeats (the table without `attempt` that has its family, width and hyper)."""
    spec = cfg["models"][model]
    if "attempt" not in spec:
        return model
    twins = [name for name, other in cfg["models"].items()
             if "attempt" not in other and {k: v for k, v in other.items()} == {k: v for k, v in spec.items()
                                                                                if k != "attempt"}]
    if len(twins) != 1:
        raise ValueError("repeat model %s has %d base models %s" % (model, len(twins), twins))
    return twins[0]


def measured_walls(*result_dirs):
    """{(task, model): mean wall seconds} over the successful runs in the given result directories."""
    sums = {}
    for out_dir in result_dirs:
        for result in ladder.load_results(Path(out_dir)):
            if result.get("status") != "ok":
                continue
            cell = (result["key"]["task"], result["key"]["model"])
            total, count = sums.get(cell, (0.0, 0))
            sums[cell] = (total + result["wall_s"], count + 1)
    return {cell: total / count for cell, (total, count) in sums.items()}


def job_seconds(cfg, walls, key):
    """Modelled wall seconds of one run."""
    task, model = key["task"], base_model(cfg, key["model"])
    extra = 0.0
    if task == LONG_COPY_TASK:
        task, extra = "copy", LONG_COPY_EVAL_S
    if (task, model) in walls:
        return walls[(task, model)] + extra
    src_task, src_model, factor = DERIVED_FROM_MEASURED[(task, model)]
    return factor * walls[(src_task, src_model)] + extra


def simulate(cfg, slice_name, walls, workers, scratch_dir):
    """Play the slice forward with `workers` workers. Returns {"jobs": [(key, start_s, end_s)], "elapsed_s": ...}.
    The Scheduler writes its lr_choice files under scratch_dir; the learning rate it picks has no bearing on the
    durations, so the lowest rate of a sweep wins."""
    sched = ladder.Scheduler(cfg, None, Path(scratch_dir), slice_name, workers, log=lambda message: None)
    clock, running, jobs = 0.0, [], []
    while sched.queue or running:
        while sched.queue and len(running) < workers:
            key = sched.queue.pop(0)
            end = clock + job_seconds(cfg, walls, key)
            running.append((end, key))
            jobs.append((key, clock, end))
        running.sort(key=lambda entry: entry[0])
        clock, key = running.pop(0)
        sched._finished(key, {"train": {"best_val_acc": -key["lr"]}})
    if sched.pending:
        raise RuntimeError("cells never chose a learning rate: %s" % sorted(sched.pending))
    return {"jobs": jobs, "elapsed_s": clock}


def estimate(cfg, slice_name, walls, workers, scratch_dir):
    sim = simulate(cfg, slice_name, walls, workers, scratch_dir)
    job_s = sum(end - start for _, start, end in sim["jobs"])
    by_task = {}
    for key, start, end in sim["jobs"]:
        row = by_task.setdefault(key["task"], {"jobs": 0, "job_s": 0.0})
        row["jobs"] += 1
        row["job_s"] += end - start
    return {"slice": slice_name, "workers": workers, "jobs": len(sim["jobs"]), "job_seconds": round(job_s),
            "simulated_elapsed_s": round(sim["elapsed_s"]),
            "elapsed_s": round(sim["elapsed_s"] / OCCUPIED_SLOT_RATIO),
            "elapsed_hours": round(sim["elapsed_s"] / OCCUPIED_SLOT_RATIO / 3600, 1),
            "by_task": {t: {"jobs": r["jobs"], "job_seconds": round(r["job_s"])} for t, r in sorted(by_task.items())}}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hybrid.bench.ladder_estimate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--slice", required=True)
    parser.add_argument("--rates", nargs="+", required=True, help="result directories with measured runs")
    parser.add_argument("--scratch", required=True, help="empty directory for the simulated scheduler's files")
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)
    cfg = ladder.load_config(args.config)
    print(json.dumps(estimate(cfg, args.slice, measured_walls(*args.rates), args.workers, args.scratch), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
