"""Bench tasks with exact oracles. `build_task(name, seed, params)` returns a TaskData."""
from __future__ import annotations

from . import compose, fewshot, hmm, recall, vom
from .common import ANSWER, DIST, TaskData

BUILDERS = {"hmm": hmm.build, "vom": vom.build, "recall": recall.build, "compose": compose.build,
            "fewshot": fewshot.build}
TASK_NAMES = tuple(BUILDERS)


def build_task(name, seed, train_pool, n_test, params):
    """params: task-specific keyword arguments from the bench config (unknown keys fail)."""
    return BUILDERS[name](seed, train_pool, n_test, **params)


__all__ = ["ANSWER", "DIST", "TaskData", "TASK_NAMES", "build_task"]
