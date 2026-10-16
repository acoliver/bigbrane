"""Run independent card-runs at once, at most ``parallel`` together (``calibrate --parallel``, ``planted --parallel``).

Each card-run has its own run directory and builds its own dependencies (LLM client, executor, math runner), so runs
share no prover memory or stage state; they share only the host-wide job slots (hybrid/slots.py) and LLM lanes
(hybrid/llm/lanes.py), which bound what reaches the Podman VM and each model route.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def check_parallel(parallel: int) -> int:
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
        raise ValueError(f"--parallel must be an integer >= 1, got {parallel!r}")
    return parallel


async def run_limited(thunks: list[Callable[[], Awaitable[T]]], parallel: int,
                      on_done: Callable[[int, T], None] | None = None) -> list[T]:
    """Await every thunk, at most ``parallel`` at a time, started in list order; return the results in list order.
    ``on_done(index, result)`` runs as each one finishes. The first failure cancels the others and is raised as it
    was (not as an exception group), so callers see the same exceptions as a serial loop."""
    check_parallel(parallel)
    results: list = [None] * len(thunks)
    gate = asyncio.Semaphore(parallel)

    async def one(index: int) -> None:
        async with gate:
            results[index] = await thunks[index]()
        if on_done is not None:
            on_done(index, results[index])

    try:
        async with asyncio.TaskGroup() as group:
            for index in range(len(thunks)):
                group.create_task(one(index))
    except BaseExceptionGroup as failed:
        raise failed.exceptions[0] from None
    return results
