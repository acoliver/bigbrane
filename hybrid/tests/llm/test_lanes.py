"""Host-wide request lanes (hybrid/llm/lanes.py): a fake slow gateway that serializes and paces its ChatGPT route the
way hybrid/gateway does, several clients (concurrent card-runs) sending at once, and the timeout that must not count
the queue."""
import asyncio
import json
import time

import httpx
import pytest

from hybrid.llm import LLMClient, TransportError
from hybrid.llm.config import load_config
from hybrid.llm.lanes import Lanes, Pacing, next_start

from .fakes import FakeBackend, ListSink, write_config

GATEWAY_CONFIG = """
routing = "main"

[budgets]
max_calls = 100
max_tokens = 1000000

[backends.gateway]
kind = "gateway"
base_url = "http://127.0.0.1:8765/v1"
api_key_env = "GW_TOKEN"
concurrency = 10
timeout_s = {timeout_s}

[routes.main]
gpt = [ {{ backend = "gateway", model = "chatgpt/gpt-test" }} ]
claude = [ {{ backend = "gateway", model = "anthropic/claude-test" }} ]
"""

INTERVAL_S = 0.05
REPLY_S = 0.3


class SlowGateway:
    """The gateway's ChatGPT route as the client sees it: one request at a time, at least INTERVAL_S between starts,
    every reply REPLY_S long; the other routes answer at once. Requests wait inside, as the real gateway's do."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.starts: list[float] = []
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"].startswith("chatgpt/"):
            async with self.lock:
                if self.starts:
                    await asyncio.sleep(max(0.0, self.starts[-1] + INTERVAL_S - time.time()))
                self.starts.append(time.time())
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
                await asyncio.sleep(REPLY_S)
                self.in_flight -= 1
        return httpx.Response(200, json={"model": body["model"], "choices": [
            {"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


@pytest.fixture
def gateway_home(tmp_path, monkeypatch):
    """The gateway's config file, as the client reads it (the running gateway's own settings)."""
    home = tmp_path / "gateway-home"
    home.mkdir()
    (home / "config.toml").write_text(f"[chatgpt]\nmax_concurrency = 1\nmin_start_interval_s = {INTERVAL_S}\n"
                                      "max_requests_per_hour = 60\n")
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(home))
    monkeypatch.setenv("GW_TOKEN", "t")
    return home


def clients(tmp_path, gateway, count, *, timeout_s, lanes_dir):
    config = tmp_path / "models.toml"
    config.write_text(GATEWAY_CONFIG.format(timeout_s=timeout_s))
    transport = httpx.MockTransport(gateway)
    return [LLMClient(str(config), str(tmp_path / f"run{i}"), ListSink(), http_transport=transport,
                      lanes_dir=lanes_dir) for i in range(count)]


async def send_together(made, per_client):
    calls = [client.complete("gpt", "s", f"u{i}") for client in made for i in range(per_client)]
    return await asyncio.gather(*calls, return_exceptions=True)


@pytest.mark.asyncio
async def test_without_lanes_requests_queued_at_a_slow_gateway_time_out(tmp_path, gateway_home):
    """The failure the lanes prevent: three card-runs send two ChatGPT calls each; the gateway answers one at a time,
    so the later requests spend longer than their timeout waiting there."""
    gateway = SlowGateway()
    made = clients(tmp_path, gateway, 3, timeout_s=1.0, lanes_dir=None)
    results = await send_together(made, 2)
    assert any(isinstance(r, TransportError) and "timed out" in str(r) for r in results)


@pytest.mark.asyncio
async def test_with_lanes_the_queue_is_held_by_the_client_and_nothing_times_out(tmp_path, gateway_home):
    gateway = SlowGateway()
    made = clients(tmp_path, gateway, 3, timeout_s=1.0, lanes_dir=tmp_path / "lanes")
    results = await send_together(made, 2)
    assert [getattr(r, "text", r) for r in results] == ["ok"] * 6
    # One request at the gateway at a time, started at least the gateway's gap apart.
    assert gateway.peak == 1
    gaps = [b - a for a, b in zip(gateway.starts, gateway.starts[1:])]
    assert len(gaps) == 5 and min(gaps) >= INTERVAL_S - 0.01
    # The six requests took about six replies end to end, longer than one timeout: the waiting happened before each
    # request's own timeout started, and each request records how long it waited and its allowance.
    attempts = [json.loads(path.read_text())["attempts"][0] for client in made
                for path in client.calls_dir.glob("*.json")]
    assert max(a["lane_wait_s"] for a in attempts) > 1.0
    assert all(a["timeout_allowance_s"] == pytest.approx(INTERVAL_S + 1.0) for a in attempts)


@pytest.mark.asyncio
async def test_other_gateway_routes_do_not_wait_for_the_chatgpt_lane(tmp_path, gateway_home):
    gateway = SlowGateway()
    made = clients(tmp_path, gateway, 1, timeout_s=5.0, lanes_dir=tmp_path / "lanes")
    started = time.monotonic()
    slow = asyncio.gather(*(made[0].complete("gpt", "s", f"g{i}") for i in range(3)))
    await asyncio.sleep(0.05)
    quick = await made[0].complete("claude", "s", "c")
    assert quick.text == "ok" and time.monotonic() - started < REPLY_S * 2
    await slow


def test_the_pacing_rule_mirrors_the_gateway():
    pacing = Pacing(min_start_interval_s=10.0, max_starts_per_hour=3)
    assert next_start([], 100.0, pacing) == 100.0
    assert next_start([95.0], 100.0, pacing) == 105.0
    assert next_start([90.0], 100.0, pacing) == 100.0
    # Three starts in the last hour: the next waits until the oldest of them leaves the window.
    assert next_start([1000.0, 1100.0, 1200.0], 1300.0, pacing) == 1000.0 + 3600.0
    # Starts older than an hour no longer count.
    assert next_start([0.0, 100.0, 4000.0], 4100.0, pacing) == 4100.0


@pytest.mark.asyncio
async def test_pacing_is_shared_by_every_client_through_the_start_log(tmp_path, gateway_home):
    config = tmp_path / "models.toml"
    config.write_text(GATEWAY_CONFIG.format(timeout_s=5))
    models = load_config(config)
    clock = [1000.0]
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    first, second = (Lanes(models, tmp_path / "lanes", clock=lambda: clock[0], sleep=sleep) for _ in range(2))
    async with first.hold("gateway", "chatgpt/gpt-test") as allowance:
        assert allowance == pytest.approx(INTERVAL_S + 5)
    async with second.hold("gateway", "chatgpt/gpt-test"):
        pass
    # The second client, another Lanes over the same directory, waited the gap after the first one's start.
    assert slept == [pytest.approx(INTERVAL_S)]
    assert json.loads((tmp_path / "lanes" / "lane-chatgpt.starts").read_text()) == [1000.0, 1000.0 + INTERVAL_S]
    async with second.hold("gateway", "anthropic/claude-test") as allowance:
        assert allowance == 0.0


@pytest.mark.asyncio
async def test_a_backend_lane_bounds_requests_from_every_client(tmp_path):
    shared = FakeBackend(delay=0.05)
    config = write_config(tmp_path, concurrency_a=1)
    made = [LLMClient(config, str(tmp_path / f"run{i}"), ListSink(), backends={"a": shared, "b": FakeBackend()},
                      lanes_dir=tmp_path / "lanes") for i in range(3)]
    await asyncio.gather(*(client.complete("solo", "s", "u") for client in made))
    assert shared.max_in_flight == 1
    # Without lanes each client keeps its own semaphore, so three clients send three at once.
    shared = FakeBackend(delay=0.05)
    made = [LLMClient(config, str(tmp_path / f"free{i}"), ListSink(), backends={"a": shared, "b": FakeBackend()})
            for i in range(3)]
    await asyncio.gather(*(client.complete("solo", "s", "u") for client in made))
    assert shared.max_in_flight == 3
