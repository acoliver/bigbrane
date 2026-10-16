"""Refresh behavior: single flight, rotation, crash safety, invalid_grant, and the 401 path."""
import asyncio
import json

import httpx
import pytest

from hybrid.gateway import app as app_module
from hybrid.gateway.anthropic_auth import AnthropicAuth, TOKEN_URL
from hybrid.gateway import fsstore
from hybrid.gateway.errors import LoginRequired, StoreError
from hybrid.gateway.redact import Redactor
from hybrid.gateway.store import Credentials
from hybrid.gateway.refresher import Refresher
from .support import NOW, anthropic_ok, make_app, make_store, meter_lines, run, seed

CHAT = {"model": "anthropic/claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}


class RotatingAnthropic:
    """Fake Anthropic: token endpoint that rotates refresh tokens and a messages endpoint that checks the bearer."""

    def __init__(self, store=None, delay=0.0):
        self.store = store
        self.delay = delay
        self.refresh_calls = 0
        self.valid_refresh = "refresh-0"
        self.message_bearers: list[str] = []
        self.stored_refresh_when_message_arrived: list[str] = []
        self.messages_status: list[int] = []        # scripted statuses; empty means 200

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            self.refresh_calls += 1
            await asyncio.sleep(self.delay)
            if form["refresh_token"] != self.valid_refresh:
                return httpx.Response(400, json={"error": "invalid_grant", "error_description": "Refresh token not found or invalid"})
            n = self.refresh_calls
            self.valid_refresh = f"refresh-{n}"
            return httpx.Response(200, json={"access_token": f"access-{n}", "refresh_token": f"refresh-{n}",
                                             "expires_in": 28800, "scope": "user:inference"})
        self.message_bearers.append(request.headers["authorization"])
        if self.store is not None:
            self.stored_refresh_when_message_arrived.append(self.store.load("anthropic").refresh_token)
        if self.messages_status:
            status = self.messages_status.pop(0)
            if status != 200:
                return httpx.Response(status, json={"error": {"type": "authentication_error", "message": "bad token"}})
        return anthropic_ok()


def expired_store(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", access="access-old", refresh="refresh-0", expires_at=NOW - 10)
    return store


def test_ten_concurrent_requests_on_an_expired_token_trigger_exactly_one_refresh(tmp_path):
    store = expired_store(tmp_path)
    upstream = RotatingAnthropic(delay=0.05)
    app = make_app(tmp_path, upstream)

    async def go():
        return await asyncio.gather(*[app.gateway.chat(CHAT) for _ in range(10)])

    results = run(go())
    assert len(results) == 10
    assert upstream.refresh_calls == 1
    assert upstream.message_bearers == ["Bearer access-1"] * 10


def test_token_close_to_expiry_is_refreshed_ahead_of_time(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", access="access-old", refresh="refresh-0", expires_at=NOW + 60)   # inside the skew
    upstream = RotatingAnthropic()
    run(make_app(tmp_path, upstream).gateway.chat(CHAT))
    assert upstream.refresh_calls == 1 and upstream.message_bearers == ["Bearer access-1"]


def test_valid_token_is_not_refreshed(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", access="access-live", refresh="refresh-0", expires_at=NOW + 7200)
    upstream = RotatingAnthropic()
    run(make_app(tmp_path, upstream).gateway.chat(CHAT))
    assert upstream.refresh_calls == 0 and upstream.message_bearers == ["Bearer access-live"]


def test_new_refresh_token_is_in_the_store_before_the_first_request_uses_the_new_access_token(tmp_path):
    store = expired_store(tmp_path)
    upstream = RotatingAnthropic(store=store)
    run(make_app(tmp_path, upstream).gateway.chat(CHAT))
    assert upstream.message_bearers == ["Bearer access-1"]
    assert upstream.stored_refresh_when_message_arrived == ["refresh-1"]


class SimulatedCrash(BaseException):
    pass


def test_crash_between_persist_and_use_leaves_a_usable_store(tmp_path):
    store = expired_store(tmp_path)
    upstream = RotatingAnthropic()

    async def crash(_provider):
        raise SimulatedCrash()

    crashing = Refresher(store, {"anthropic": AnthropicAuth(httpx.MockTransport(upstream), lambda: NOW)},
                         redactor=Redactor(), clock=lambda: NOW, on_persisted=crash)
    with pytest.raises(SimulatedCrash):
        run(crashing.credentials("anthropic"))
    assert upstream.refresh_calls == 1
    assert store.load("anthropic").refresh_token == "refresh-1"      # the rotated token survived the crash

    # A fresh process picks up the stored record and continues from the new refresh token.
    upstream2_calls = upstream.refresh_calls
    app = make_app(tmp_path, upstream, clock=lambda: NOW + 40000)     # new access token also expired by then
    run(app.gateway.chat(CHAT))
    assert upstream.refresh_calls == upstream2_calls + 1
    assert store.load("anthropic").refresh_token == "refresh-2"


@pytest.mark.parametrize("failing_step", ["fsync_file", "rename", "fsync_dir"])
def test_failed_write_is_an_error_and_the_new_token_is_never_used(tmp_path, monkeypatch, failing_step):
    store = expired_store(tmp_path)
    upstream = RotatingAnthropic()
    if failing_step == "rename":
        monkeypatch.setattr(fsstore.os, "rename", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    else:
        real_sync, calls = fsstore._sync, []

        def failing_sync(fd):
            calls.append(fd)
            if len(calls) == (1 if failing_step == "fsync_file" else 2):
                raise OSError("disk full")
            real_sync(fd)
        monkeypatch.setattr(fsstore, "_sync", failing_sync)

    with pytest.raises(StoreError, match="disk full"):
        run(make_app(tmp_path, upstream).gateway.chat(CHAT))
    assert upstream.message_bearers == []            # no request went out with the unpersisted token
    leftovers = sorted(p.name for p in (tmp_path / "oauth").iterdir())
    assert leftovers == ["anthropic.json", "anthropic.lock"]          # no staging file left behind
    if failing_step != "fsync_dir":                  # a directory-sync failure comes after the rename
        monkeypatch.undo()
        assert store.load("anthropic").refresh_token == "refresh-0"


def test_invalid_grant_fails_closed_names_the_login_command_and_does_not_retry(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", access="access-old", refresh="revoked", expires_at=NOW - 10)
    upstream = RotatingAnthropic()
    app = make_app(tmp_path, upstream)

    for _ in range(6):
        with pytest.raises(LoginRequired) as err:
            run(app.gateway.chat(CHAT))
        assert "python -m hybrid.gateway login anthropic" in str(err.value)
    assert upstream.refresh_calls == 1            # one attempt, then no network until the login changes
    assert upstream.message_bearers == []
    assert all(line["status"] == "error" and line["error"]["code"] == "login_required" for line in meter_lines(tmp_path))

    seed(store, "anthropic", access="fresh", refresh="refresh-0", expires_at=NOW + 3600)        # a new login
    run(app.gateway.chat(CHAT))
    assert upstream.message_bearers == ["Bearer fresh"]


def test_transient_refresh_failure_is_not_remembered_as_a_dead_login(tmp_path):
    store = expired_store(tmp_path)
    upstream = RotatingAnthropic()
    attempts = []

    async def flaky(request):
        if str(request.url) == TOKEN_URL and not attempts:
            attempts.append(1)
            return httpx.Response(503, json={"error": "temporarily_unavailable"})
        return await upstream(request)

    app = make_app(tmp_path, flaky)
    from hybrid.gateway.errors import UpstreamError
    with pytest.raises(UpstreamError):
        run(app.gateway.chat(CHAT))
    run(app.gateway.chat(CHAT))
    assert upstream.refresh_calls == 1 and upstream.message_bearers == ["Bearer access-1"]


def test_missing_login_names_the_login_command(tmp_path):
    with pytest.raises(LoginRequired) as err:
        run(make_app(tmp_path, RotatingAnthropic()).gateway.chat(CHAT))
    assert "python -m hybrid.gateway login anthropic" in str(err.value)


def test_api_401_on_a_live_token_forces_one_refresh_and_one_retry(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", access="access-old", refresh="refresh-0", expires_at=NOW + 7200)
    upstream = RotatingAnthropic()
    upstream.messages_status = [401]
    run(make_app(tmp_path, upstream).gateway.chat(CHAT))
    assert upstream.refresh_calls == 1
    assert upstream.message_bearers == ["Bearer access-old", "Bearer access-1"]


def test_api_401_twice_stops_after_one_retry(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", access="access-old", refresh="refresh-0", expires_at=NOW + 7200)
    upstream = RotatingAnthropic()
    upstream.messages_status = [401, 401, 401]
    from hybrid.gateway.errors import UpstreamError
    with pytest.raises(UpstreamError) as err:
        run(make_app(tmp_path, upstream).gateway.chat(CHAT))
    assert err.value.status == 401
    assert len(upstream.message_bearers) == 2 and upstream.refresh_calls == 1


def test_two_gateway_processes_sharing_one_store_cause_exactly_one_refresh(tmp_path):
    """Two Refresher instances stand in for two processes: they share only the files, so the flock serializes them."""
    store = expired_store(tmp_path)
    upstream = RotatingAnthropic(delay=0.05)
    first = make_app(tmp_path, upstream)
    second = make_app(tmp_path, upstream)

    async def go():
        return await asyncio.gather(first.gateway.chat(CHAT), second.gateway.chat(CHAT),
                                    first.gateway.chat(CHAT), second.gateway.chat(CHAT))

    run(go())
    assert upstream.refresh_calls == 1
    assert upstream.message_bearers == ["Bearer access-1"] * 4
    assert store.load("anthropic").refresh_token == "refresh-1"


def test_a_refresh_waits_for_a_login_command_holding_the_store_lock(tmp_path):
    store = expired_store(tmp_path)
    upstream = RotatingAnthropic()
    app = make_app(tmp_path, upstream)

    async def go():
        held = store.lock("anthropic")
        await asyncio.to_thread(held.acquire)
        task = asyncio.ensure_future(app.gateway.chat(CHAT))
        await asyncio.sleep(0.3)
        assert upstream.refresh_calls == 0           # blocked behind the lock holder
        # The holder (a login) publishes a fresh record, then releases.
        held.write(Credentials("login-access", "refresh-0", NOW + 7200))
        held.release()
        await task

    run(go())
    assert upstream.refresh_calls == 0               # the waiter re-read the login's record and used it
    assert upstream.message_bearers == ["Bearer login-access"]
