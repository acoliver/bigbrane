"""/v1/models: each provider's list is fetched under a timeout and cached for a TTL. A provider whose live fetch times
out or fails contributes its last good list, else its static ids, else nothing. A client that disconnects mid-response
costs one log line and no traceback. Fakes only: no upstream is contacted."""
import asyncio
import logging
import re
import time

import httpx
import pytest

from hybrid.gateway import app as app_module
from hybrid.gateway.errors import GatewayError, LoginRequired, UpstreamError
from hybrid.gateway.gateway import Gateway
from hybrid.gateway.meter import Meter
from hybrid.gateway.paths import load_config
from hybrid.gateway.redact import Redactor
from hybrid.gateway.server import GatewayServer
from .support import NOW, Sleeper, make_store, run, seed

TOKEN = "run-token-for-models-listing-tests"
TIMEOUT_S = 0.05
TTL_S = 600.0
HANG_S = 30.0          # longer than any test waits; the timeout cancels the call


class FakeClock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class ListBackend:
    """A provider whose model list answers with ``names`` after ``delay`` seconds, or fails with ``error``."""

    def __init__(self, names=(), *, delay: float = 0.0, error: Exception | None = None) -> None:
        self.names = list(names)
        self.delay = delay
        self.error = error
        self.calls = 0

    async def complete(self, body: dict, model: str):
        raise AssertionError("the listing tests never send a completion")

    async def models(self) -> list[str]:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return list(self.names)


def make_gateway(tmp_path, backends, clock=None, *, static=None, timeout_s=TIMEOUT_S, ttl_s=TTL_S) -> Gateway:
    return Gateway(backends, Meter(tmp_path / "usage.jsonl"), clock or FakeClock(), models_timeout_s=timeout_s,
                   models_cache_ttl_s=ttl_s, static_models=static or {})


def listed(payload: dict) -> list[str]:
    return sorted(m["id"] for m in payload["data"])


# ---- Timeout and fallback ------------------------------------------------------------------------

def test_a_slow_provider_is_cut_off_and_the_fast_list_is_still_served(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="bigbrane.gateway")
    gateway = make_gateway(tmp_path, {"chatgpt": ListBackend(["slow-model"], delay=HANG_S),
                                      "anthropic": ListBackend(["fast-model"])},
                           static={"chatgpt": ["static-slow-model"]})

    started = time.monotonic()
    payload = run(gateway.models())

    assert time.monotonic() - started < 5
    assert listed(payload) == ["anthropic/fast-model", "chatgpt/static-slow-model"]
    assert payload["upstream_errors"] == {"chatgpt": {"status": 504, "code": "upstream_timeout",
                                                      "message": "chatgpt model list did not answer within 0.05 s"}}
    [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert re.fullmatch(r"upstream chatgpt model list failed after \d+\.\d{2} s \(chatgpt model list did not answer "
                        r"within 0.05 s\); serving static list of 1 model", warning)


def test_a_timed_out_provider_serves_its_last_good_list_once_the_cache_is_stale(tmp_path):
    clock = FakeClock()
    chatgpt = ListBackend(["cached-model"])
    gateway = make_gateway(tmp_path, {"chatgpt": chatgpt, "anthropic": ListBackend(["fast-model"])}, clock,
                           static={"chatgpt": ["static-model"]})
    assert listed(run(gateway.models())) == ["anthropic/fast-model", "chatgpt/cached-model"]

    chatgpt.delay = HANG_S
    clock.now += TTL_S + 1
    payload = run(gateway.models())

    assert listed(payload) == ["anthropic/fast-model", "chatgpt/cached-model"]
    assert payload["upstream_errors"]["chatgpt"]["code"] == "upstream_timeout"


def test_a_provider_with_no_cache_and_no_static_ids_contributes_nothing(tmp_path):
    gateway = make_gateway(tmp_path, {"chatgpt": ListBackend(delay=HANG_S), "anthropic": ListBackend(["fast-model"])})

    payload = run(gateway.models())

    assert listed(payload) == ["anthropic/fast-model"]
    assert payload["upstream_errors"]["chatgpt"]["code"] == "upstream_timeout"


def test_a_failed_fetch_falls_back_to_the_cache_and_logs_one_warning_per_provider(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="bigbrane.gateway")
    clock = FakeClock()
    chatgpt = ListBackend(["cached-model"])
    anthropic = ListBackend(["fast-model"])
    gateway = make_gateway(tmp_path, {"chatgpt": chatgpt, "anthropic": anthropic}, clock)
    run(gateway.models())
    caplog.clear()

    chatgpt.error = UpstreamError("chatgpt returned HTTP 503: busy", status=502, code="upstream_error")
    clock.now += TTL_S + 1
    payload = run(gateway.models())

    assert listed(payload) == ["anthropic/fast-model", "chatgpt/cached-model"]
    assert payload["upstream_errors"] == {"chatgpt": {"status": 502, "code": "upstream_error",
                                                      "message": "chatgpt returned HTTP 503: busy"}}
    [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert re.fullmatch(r"upstream chatgpt model list failed after \d+\.\d{2} s \(chatgpt returned HTTP 503: busy\); "
                        r"serving cached list of 1 model, fetched \d+ s ago", warning)


def test_a_provider_without_a_login_is_named_and_its_cached_list_is_not_served(tmp_path):
    clock = FakeClock()
    chatgpt = ListBackend(["cached-model"])
    gateway = make_gateway(tmp_path, {"chatgpt": chatgpt}, clock)
    run(gateway.models())

    chatgpt.error = LoginRequired("chatgpt", "the stored login has no refresh token")
    clock.now += TTL_S + 1
    payload = run(gateway.models())

    assert payload["data"] == []
    assert "login chatgpt" in payload["not_logged_in"]["chatgpt"]
    assert "upstream_errors" not in payload


def test_an_error_that_is_not_an_upstream_failure_is_not_turned_into_a_fallback(tmp_path):
    gateway = make_gateway(tmp_path, {"chatgpt": ListBackend(error=RuntimeError("boom"))},
                           static={"chatgpt": ["static-model"]})

    with pytest.raises(RuntimeError, match="boom"):
        run(gateway.models())


# ---- Cache ---------------------------------------------------------------------------------------

def test_a_fresh_cached_list_makes_no_upstream_call(tmp_path):
    chatgpt = ListBackend(["gpt-model"])
    gateway = make_gateway(tmp_path, {"chatgpt": chatgpt})

    first = run(gateway.models())
    second = run(gateway.models())

    assert listed(first) == listed(second) == ["chatgpt/gpt-model"]
    assert chatgpt.calls == 1


def test_the_cache_expires_after_its_ttl_and_the_list_is_fetched_again(tmp_path):
    clock = FakeClock()
    chatgpt = ListBackend(["gpt-model"])
    gateway = make_gateway(tmp_path, {"chatgpt": chatgpt}, clock, ttl_s=600.0)
    run(gateway.models())

    clock.now += 599
    run(gateway.models())
    assert chatgpt.calls == 1

    clock.now += 1
    chatgpt.names = ["gpt-model-2"]
    payload = run(gateway.models())
    assert chatgpt.calls == 2
    assert listed(payload) == ["chatgpt/gpt-model-2"]


def test_a_zero_ttl_fetches_on_every_call(tmp_path):
    chatgpt = ListBackend(["gpt-model"])
    gateway = make_gateway(tmp_path, {"chatgpt": chatgpt}, ttl_s=0.0)
    run(gateway.models())
    run(gateway.models())
    assert chatgpt.calls == 2


@pytest.mark.parametrize("config_text, upstream_calls", [
    pytest.param("[models]\ncache_ttl_s = 0\n", 2, id="zero-ttl-fetches-every-call"),
    pytest.param("", 1, id="default-ttl-serves-the-second-call"),
])
def test_build_wires_the_models_cache_from_config_toml(tmp_path, config_text, upstream_calls):
    (tmp_path / "config.toml").write_text(config_text)
    seed(make_store(tmp_path), "chatgpt", meta={"client_id": "oaiapp_x"})
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, json={"models": [{"slug": "gpt-6.1-sol", "visibility": "list"}]})

    app = app_module.build(tmp_path, transport=httpx.MockTransport(handler), meter_file=tmp_path / "usage.jsonl",
                           clock=lambda: NOW, sleep=Sleeper())
    run(app.gateway.models())
    payload = run(app.gateway.models())

    assert len(requested) == upstream_calls
    assert "chatgpt/gpt-6.1-sol" in listed(payload)
    assert "anthropic/claude-opus-5-5" in listed(payload)


def test_a_chatgpt_list_that_times_out_while_queued_for_pacing_never_reaches_openai(tmp_path):
    (tmp_path / "config.toml").write_text("[models]\nupstream_timeout_s = 0.05\ncache_ttl_s = 0\n")
    seed(make_store(tmp_path), "chatgpt", meta={"client_id": "oaiapp_x"})
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, json={"models": [{"slug": "gpt-6.1-sol", "visibility": "list"}]})

    async def hold_for_the_pacing_gap(seconds: float) -> None:
        await asyncio.sleep(HANG_S)

    app = app_module.build(tmp_path, transport=httpx.MockTransport(handler), meter_file=tmp_path / "usage.jsonl",
                           clock=lambda: NOW, sleep=hold_for_the_pacing_gap)
    first = run(app.gateway.models())                # the first start is not delayed
    second = run(app.gateway.models())               # the 10 s gap after it outlives the timeout

    assert len(requested) == 1
    assert "chatgpt/gpt-6.1-sol" in listed(first) and "chatgpt/gpt-6.1-sol" in listed(second)
    assert second["upstream_errors"]["chatgpt"]["code"] == "upstream_timeout"


# ---- Configuration -------------------------------------------------------------------------------

def test_models_section_defaults_to_ten_seconds_and_ten_minutes(tmp_path):
    models = load_config(tmp_path).models
    assert (models.upstream_timeout_s, models.cache_ttl_s) == (10.0, 600.0)


def test_models_section_sets_the_timeout_and_ttl(tmp_path):
    (tmp_path / "config.toml").write_text("[models]\nupstream_timeout_s = 2.5\ncache_ttl_s = 0\n")
    models = load_config(tmp_path).models
    assert (models.upstream_timeout_s, models.cache_ttl_s) == (2.5, 0)


@pytest.mark.parametrize("line", [
    pytest.param("upstream_timeout_s = 0", id="zero-timeout"),
    pytest.param("upstream_timeout_s = true", id="bool-timeout"),
    pytest.param('upstream_timeout_s = "10"', id="string-timeout"),
    pytest.param("cache_ttl_s = -1", id="negative-ttl"),
    pytest.param("cache_ttl_s = false", id="bool-ttl"),
    pytest.param("retries = 2", id="unknown-key"),
])
def test_bad_models_values_are_configuration_errors(tmp_path, line):
    (tmp_path / "config.toml").write_text(f"[models]\n{line}\n")
    with pytest.raises(GatewayError):
        load_config(tmp_path)


# ---- Client disconnects and other server errors --------------------------------------------------

class FakeWriter:
    """The server side of one connection. ``drain_error`` is raised by the first response write, as asyncio does when
    the peer has already closed the socket."""

    def __init__(self, drain_error: Exception | None = None) -> None:
        self.drain_error = drain_error
        self.written = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        if self.drain_error is not None:
            raise self.drain_error

    def close(self) -> None:
        self.closed = True


def models_request() -> bytes:
    return b"GET /v1/models HTTP/1.1\r\nAuthorization: Bearer " + TOKEN.encode() + b"\r\n\r\n"


def server_for(tmp_path, backends) -> GatewayServer:
    return GatewayServer(make_gateway(tmp_path, backends), TOKEN, Redactor(), port=0)


async def handle(server: GatewayServer, writer: FakeWriter) -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(models_request())
    reader.feed_eof()
    await server._handle(reader, writer)


@pytest.mark.parametrize("error", [
    pytest.param(ConnectionResetError(54, "Connection reset by peer"), id="reset"),
    pytest.param(BrokenPipeError(32, "Broken pipe"), id="broken-pipe"),
])
def test_a_client_that_disconnects_mid_response_costs_one_info_line_and_no_traceback(tmp_path, caplog, error):
    caplog.set_level(logging.INFO, logger="bigbrane.gateway")
    server = server_for(tmp_path, {"chatgpt": ListBackend(["gpt-model"]), "anthropic": ListBackend(["claude-model"])})
    writer = FakeWriter(drain_error=error)

    run(handle(server, writer))

    [record] = caplog.records
    assert record.levelno == logging.INFO
    assert record.getMessage().startswith("GET /v1/models -> client disconnected after ")
    assert record.exc_info is None and "Traceback" not in caplog.text
    assert writer.closed


def test_an_unexpected_error_still_answers_500_and_logs_the_traceback(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bigbrane.gateway")
    server = server_for(tmp_path, {"chatgpt": ListBackend(error=RuntimeError("boom"))})
    writer = FakeWriter()

    run(handle(server, writer))

    assert writer.written.startswith(b"HTTP/1.1 500 ")
    assert b"internal error (RuntimeError)" in writer.written
    assert "Traceback" in caplog.text and "RuntimeError: boom" in caplog.text
