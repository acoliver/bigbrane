"""Host-wide request lanes: where the LLM client queues, so that waiting is never counted against a timeout.

Without lanes, ``backends.<name>.concurrency`` bounds the requests of one LLMClient, and a request's timeout clock
runs while it waits at the server. Concurrent card-runs each have their own client, and the gateway serializes its
ChatGPT route (hybrid/gateway/chatgpt_pacing.py: one request at a time, at least ``min_start_interval_s`` between
starts, at most ``max_requests_per_hour`` per sliding hour, extra requests wait). Requests piled up there would time
out at the client although the gateway was only queueing them.

With lanes (``LLMClient(lanes_dir=...)``), a request first holds its lanes, then starts its timeout clock:

* ``chatgpt`` (a gateway backend's ``chatgpt/...`` models): the gateway's own ChatGPT settings, read from its config
  file (hybrid.gateway.paths.load_config; never written): ``max_concurrency`` slots, and before each start the client
  waits as the gateway's pacer would, using a start log shared by every process. The gateway then admits the request
  at once unless a client without lanes sent one first, so the request's timeout gets an allowance: one pacing gap
  plus one backend timeout (a single foreign request ahead of it).
* ``backend:<name>``: the backend's ``concurrency``, shared by every client on the host (LM Studio serves one request
  at a time whichever card-run sends it).

A request takes the ChatGPT lane before its backend lane, and holds both until its reply, so no request waits for a
lane while holding one that a request ahead of it needs. Slots are hybrid/slots.py file semaphores under the
workspace.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from hybrid.slots import FileSemaphore

from .config import BackendConfig, ModelsConfig

WINDOW_S = 3600.0
CHATGPT_LANE = "chatgpt"
CHATGPT_PREFIX = "chatgpt/"


@dataclass(frozen=True)
class Pacing:
    min_start_interval_s: float
    max_starts_per_hour: int


@dataclass(frozen=True)
class Lane:
    name: str
    limit: int
    pacing: Pacing | None = None
    # Seconds added to the request's timeout for waits at the server the lane cannot see.
    allowance_s: float = 0.0


def next_start(starts: list[float], now: float, pacing: Pacing) -> float:
    """The earliest start the gateway's pacer allows after ``starts`` (its rule: a gap after the last start and at most
    ``max_starts_per_hour`` starts in any sliding hour; Pacer._next_start in hybrid/gateway/chatgpt_pacing.py)."""
    recent = [t for t in starts if t > now - WINDOW_S]
    start = now
    if recent:
        start = max(start, recent[-1] + pacing.min_start_interval_s)
    if len(recent) >= pacing.max_starts_per_hour:
        start = max(start, recent[-pacing.max_starts_per_hour] + WINDOW_S)
    return start


def gateway_chatgpt_settings():
    """The running gateway's ChatGPT settings, from its config file (defaults when there is none)."""
    from hybrid.gateway.paths import load_config

    root = Path(os.environ.get("BIGBRANE_GATEWAY_HOME") or Path.home() / ".bigbrane-gateway")
    return load_config(root).chatgpt


class Lanes:
    def __init__(self, config: ModelsConfig, directory: Path, *, chatgpt=None, poll_s: float = 0.25,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.config = config
        self.directory = Path(directory)
        self.poll_s = poll_s
        self.clock = clock
        self.sleep = sleep
        self._chatgpt = chatgpt
        self._semaphores: dict[str, FileSemaphore] = {}

    def chatgpt_lane(self, backend: BackendConfig) -> Lane:
        settings = self._chatgpt if self._chatgpt is not None else gateway_chatgpt_settings()
        self._chatgpt = settings
        pacing = Pacing(float(settings.min_start_interval_s), int(settings.max_requests_per_hour))
        return Lane(CHATGPT_LANE, int(settings.max_concurrency), pacing,
                    allowance_s=pacing.min_start_interval_s + backend.timeout_s)

    def lanes_for(self, backend_name: str, model: str) -> list[Lane]:
        """The lanes of one request, in the order they are taken."""
        backend = self.config.backends[backend_name]
        lanes = []
        if backend.kind == "gateway" and model.startswith(CHATGPT_PREFIX):
            lanes.append(self.chatgpt_lane(backend))
        lanes.append(Lane(f"backend-{backend_name}", backend.concurrency))
        return lanes

    def semaphore(self, lane: Lane) -> FileSemaphore:
        if lane.name not in self._semaphores:
            self._semaphores[lane.name] = FileSemaphore(self.directory, "lane-" + lane.name, lane.limit,
                                                        poll_s=self.poll_s)
        return self._semaphores[lane.name]

    async def pace(self, lane: Lane) -> float:
        """Wait for this lane's next start under its pacing, record the start, and return the seconds waited. The
        start log is shared by every process and updated under its own lock."""
        log = self.directory / f"lane-{lane.name}.starts"
        fd = os.open(self.directory / f"lane-{lane.name}.starts.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await self.sleep(self.poll_s)
            try:
                now = self.clock()
                starts = json.loads(log.read_text()) if log.exists() else []
                start = next_start(starts, now, lane.pacing)
                kept = [t for t in starts if t > now - WINDOW_S] + [start]
                log.write_text(json.dumps(kept))
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        wait = start - now
        if wait > 0:
            await self.sleep(wait)
        return max(wait, 0.0)

    @asynccontextmanager
    async def hold(self, backend_name: str, model: str):
        """Hold every lane of a request (pacing included); yields the seconds to add to the request's timeout."""
        lanes = self.lanes_for(backend_name, model)
        async with AsyncExitStack() as stack:
            for lane in lanes:
                await stack.enter_async_context(self.semaphore(lane).hold_async())
            for lane in lanes:
                if lane.pacing is not None:
                    await self.pace(lane)
            yield sum(lane.allowance_s for lane in lanes)
