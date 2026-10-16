"""Test doubles and helpers for hybrid/llm tests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hybrid.llm.backends import BackendReply, BackendRequest
from hybrid.llm.errors import TransportError

CONFIG_TEMPLATE = """
routing = "main"

[budgets]
max_calls = {max_calls}
max_tokens = {max_tokens}

[backends.a]
kind = "openai_compat"
base_url = "http://127.0.0.1:1/v1"
concurrency = {concurrency_a}
timeout_s = 30

[backends.b]
kind = "openai_compat"
base_url = "http://127.0.0.1:2/v1"
concurrency = {concurrency_b}
timeout_s = 30

[routes.main]
r = [ {{ backend = "a", model = "m1" }}, {{ backend = "b", model = "m2" }} ]
solo = [ {{ backend = "a", model = "m1" }} ]
solo_b = [ {{ backend = "b", model = "m2" }} ]
"""


def write_config(tmp_path: Path, *, max_calls=100, max_tokens=1_000_000, concurrency_a=2, concurrency_b=2,
                 extra: str = "") -> str:
    path = tmp_path / "models.toml"
    path.write_text(CONFIG_TEMPLATE.format(max_calls=max_calls, max_tokens=max_tokens,
                                           concurrency_a=concurrency_a, concurrency_b=concurrency_b) + extra)
    return str(path)


class ListSink:
    def __init__(self):
        self.events: list[dict] = []

    def append(self, type: str, actor: str, data: dict) -> dict:
        event = {"type": type, "actor": actor, "data": data}
        self.events.append(event)
        return event

    def of(self, type: str) -> list[dict]:
        return [event for event in self.events if event["type"] == type]


class FakeBackend:
    """Scripted backend. Each script item is a str reply text or an Exception to raise."""

    def __init__(self, kind="openai_compat", script=None, default="ok", delay=0.0, usage=None):
        self.kind = kind
        self.script = list(script or [])
        self.default = default
        self.delay = delay
        self.usage = usage or {"input": 10, "output": 5, "cached": 0}
        self.requests: list[BackendRequest] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, request: BackendRequest) -> BackendReply:
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            item = self.script.pop(0) if self.script else self.default
            if isinstance(item, Exception):
                raise item
            return BackendReply(text=item, model=f"resolved-{request.model}", usage=dict(self.usage),
                                detail={"fake": True})
        finally:
            self.in_flight -= 1


def transport_failure(message="boom") -> TransportError:
    return TransportError(message)


def llxprt_stdout(response="ok", *, tools_total=0, tools_by_name=None, model="claude-haiku-5-5",
                  prompt=1000, candidates=20, cached=0) -> bytes:
    by_name = {name: {"count": 1} for name in (tools_by_name or [])}
    return json.dumps({
        "session_id": "s-1", "response": response,
        "stats": {
            "models": {model: {"api": {"totalRequests": 1},
                               "tokens": {"input": prompt - cached, "prompt": prompt, "candidates": candidates,
                                          "total": prompt + candidates, "cached": cached, "thoughts": 0, "tool": 0}}},
            "tools": {"totalCalls": tools_total, "byName": by_name},
        }}).encode()
