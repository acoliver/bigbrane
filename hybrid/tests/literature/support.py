"""Shared test doubles: a fake web routed by host, a fake clock, and an event sink."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import httpx

from hybrid.literature.client import Literature

FIXTURES = Path(__file__).parent / "fixtures"
Responder = Callable[[httpx.Request], httpx.Response]


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def ok(body: str) -> Responder:
    return lambda request: httpx.Response(200, text=body)


def status(code: int, headers: dict[str, str] | None = None) -> Responder:
    return lambda request: httpx.Response(code, text="{}", headers=headers or {})


class FakeWeb:
    """Routes each request to a responder chosen by host, and records every network call."""

    def __init__(self, routes: dict[str, Responder | list[Responder]]) -> None:
        self._routes = routes
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        route = self._routes[request.url.host]
        if isinstance(route, list):
            responder = route.pop(0) if len(route) > 1 else route[0]
        else:
            responder = route
        return responder(request)

    def calls_to(self, host: str) -> list[httpx.Request]:
        return [call for call in self.calls if call.url.host == host]


class FakeClock:
    """Time that advances only when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def append(self, type: str, actor: str, data: dict[str, Any]) -> None:
        self.events.append((type, actor, data))

    def of_type(self, type: str) -> list[dict[str, Any]]:
        return [data for event_type, _, data in self.events if event_type == type]


def make_literature(
    cache_root: Path,
    web: FakeWeb,
    clock: FakeClock | None = None,
    contact_email: str | None = None,
    max_retries: int = 3,
) -> Literature:
    clock = clock or FakeClock()
    return Literature(
        cache_root=cache_root,
        http=httpx.Client(transport=httpx.MockTransport(web.handler)),
        sleep=clock.sleep,
        clock=clock.clock,
        now=lambda: "2026-10-09T00:00:00+00:00",
        contact_email=contact_email,
        max_retries=max_retries,
    )
