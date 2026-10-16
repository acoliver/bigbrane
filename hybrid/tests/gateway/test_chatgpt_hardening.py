"""ChatGPT (Sign in with ChatGPT) hardening: error diagnostics, the local stop after an admission 403, the
earliest_refresh_at window, the ChatGPT concurrency cap, revoking logout, tool placement, attribution text and the
new-registration login. Every upstream is a fake; no test reaches the network or the real token store."""
import argparse
import asyncio
import json
import logging
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from hybrid.gateway import __main__ as cli
from hybrid.gateway import app as app_module
from hybrid.gateway import chatgpt_auth, responses_translate as tr
from hybrid.gateway.backends import upstream_http_error
from hybrid.gateway.chatgpt_auth import ChatGPTAuth, TOKEN_URL
from hybrid.gateway.chatgpt_plan import ACCESS_DENIED_MESSAGE, LOGIN_NOTICE, STATUS_NOTICE
from hybrid.gateway.errors import BadRequest, GatewayError, UpstreamError
from hybrid.gateway.login import login_chatgpt, logout_chatgpt, status_lines
from hybrid.gateway.paths import ChatGPTConfig, host_id, load_config
from hybrid.gateway.store import Credentials
from .support import NOW, SENTINEL_REFRESH, Sleeper, anthropic_ok, make_app, make_store, meter_lines, responses_ok, run, seed
from .test_auth_flows import ISSUED, token_handler

CHAT = {"messages": [{"role": "user", "content": "hi"}]}
GPT = {**CHAT, "model": "chatgpt/gpt-6.1-sol"}
CLAUDE = {**CHAT, "model": "anthropic/claude-haiku-5-5"}
LEAD = "Manage usage: https://chatgpt.com/settings/usage. "
DENIED_HEADERS = {"x-request-id": "req_2c24", "x-openai-ide-root-error-code": "access_denied",
                  "x-openai-authorization-error": "403"}
ACCESS_DENIED_BODY = {"detail": {"error_code": "access_denied"}}


def denied() -> httpx.Response:
    return httpx.Response(403, json=ACCESS_DENIED_BODY, headers=DENIED_HEADERS)


class Upstream:
    """Counts requests per host and answers from a script. ``script`` returns a response for ChatGPT calls."""

    def __init__(self, script=None, delay: float = 0.0) -> None:
        self.script = script or (lambda n, request: responses_ok())
        self.delay = delay
        self.openai_requests: list[httpx.Request] = []
        self.other_requests: list[httpx.Request] = []
        self.current = self.peak = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if "openai.com" not in request.url.host:
            self.other_requests.append(request)
            self.current += 1
            self.peak = max(self.peak, self.current)
            try:
                await asyncio.sleep(self.delay)
            finally:
                self.current -= 1
            return anthropic_ok()
        self.openai_requests.append(request)
        self.current += 1
        self.peak = max(self.peak, self.current)
        try:
            await asyncio.sleep(self.delay)
            return self.script(len(self.openai_requests), request)
        finally:
            self.current -= 1

    @property
    def bearers(self) -> list[str]:
        return [r.headers["authorization"] for r in self.openai_requests if "authorization" in r.headers]


def seed_chatgpt(tmp_path, **kw):
    store = make_store(tmp_path)
    kw.setdefault("meta", {"client_id": ISSUED})
    seed(store, "chatgpt", **kw)
    return store


# ---- 1. error code and upstream headers ----------------------------------------------------------

@pytest.mark.parametrize("body", [
    pytest.param({"detail": {"error_code": "access_denied"}}, id="detail-error_code"),
    pytest.param({"error": {"error_code": "access_denied"}}, id="error-error_code"),
    pytest.param({"detail": {"error": {"code": "access_denied"}}}, id="detail-error-code"),
    pytest.param({"error_code": "access_denied"}, id="top-level"),
])
def test_error_code_is_read_from_the_nested_shapes_openai_uses(body):
    error = upstream_http_error("chatgpt", 403, json.dumps(body).encode())
    assert error.code == "access_denied" and error.detail.code == "access_denied"
    assert error.message == "chatgpt returned HTTP 403 (access_denied)"        # no message in the body: the code shows


def test_a_message_next_to_the_error_code_is_kept():
    error = upstream_http_error("chatgpt", 403, b'{"detail": {"error_code": "access_denied", "message": "Not allowed"}}')
    assert (error.code, error.detail.message) == ("access_denied", "Not allowed")
    assert error.message == "chatgpt returned HTTP 403: Not allowed"


def test_the_docs_403_reaches_the_log_the_meter_and_the_error_with_request_id_and_root_code(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bigbrane.gateway")
    seed_chatgpt(tmp_path)
    app = make_app(tmp_path, Upstream(lambda n, r: denied()))

    with pytest.raises(UpstreamError) as err:
        run(app.gateway.chat(GPT))

    assert err.value.code == "access_denied" and err.value.status == 403
    assert err.value.detail.diagnostics() == {"request_id": "req_2c24", "root_error_code": "access_denied",
                                              "authorization_error": "403"}
    assert "request_id=req_2c24 root_error_code=access_denied authorization_error=403" in caplog.text
    assert meter_lines(tmp_path)[0]["error"] == {"code": "access_denied", "status": 403, "request_id": "req_2c24",
                                                 "root_error_code": "access_denied", "authorization_error": "403"}


@pytest.mark.parametrize("status", [400, 429, 500])
def test_every_non_200_is_logged_and_metered_with_the_request_id(tmp_path, caplog, status):
    caplog.set_level(logging.INFO, logger="bigbrane.gateway")
    seed_chatgpt(tmp_path)
    reply = httpx.Response(status, json={"error": {"message": "no", "type": "t"}}, headers={"x-request-id": "req_x1"})
    app = make_app(tmp_path, Upstream(lambda n, r: reply))
    with pytest.raises(UpstreamError):
        run(app.gateway.chat(GPT))
    assert "request_id=req_x1" in caplog.text
    assert meter_lines(tmp_path)[0]["error"]["request_id"] == "req_x1"
    assert "root_error_code" not in meter_lines(tmp_path)[0]["error"]          # absent headers are not invented


def test_a_401_that_is_retried_after_a_refresh_still_logs_the_request_id(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bigbrane.gateway")
    seed_chatgpt(tmp_path, expires_at=NOW + 7200)
    stale = httpx.Response(401, json={"error": {"message": "stale"}}, headers={"x-request-id": "req_401"})

    def script(n, request):          # call 1 is the refused request, call 2 the refresh, call 3 the retry
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "r2", "expires_in": 3600})
        return stale if n == 1 else responses_ok()

    run(make_app(tmp_path, Upstream(script)).gateway.chat(GPT))
    assert "answered 401; refreshing once and retrying request_id=req_401" in caplog.text


def test_models_failure_carries_the_request_id_and_blocks_later_calls(tmp_path):
    store = seed_chatgpt(tmp_path)
    seed(store, "anthropic")
    upstream = Upstream(lambda n, r: denied())
    app = make_app(tmp_path, upstream)

    listing = run(app.gateway.models())
    assert listing["upstream_errors"]["chatgpt"]["request_id"] == "req_2c24"
    assert listing["upstream_errors"]["chatgpt"]["code"] == "access_denied"

    with pytest.raises(UpstreamError, match="plan access denied"):
        run(app.gateway.chat(GPT))
    assert len(upstream.openai_requests) == 1


def test_claude_errors_are_unchanged_by_the_chatgpt_diagnostics(tmp_path):
    seed(make_store(tmp_path), "anthropic")
    reply = httpx.Response(403, json={"error": {"type": "permission_error", "message": "no"}},
                           headers=DENIED_HEADERS)
    app = make_app(tmp_path, lambda request: reply)
    with pytest.raises(UpstreamError) as err:
        run(app.gateway.chat(CLAUDE))
    assert err.value.message == "anthropic returned HTTP 403: no"             # no Manage usage lead, no code suffix
    assert meter_lines(tmp_path)[0]["error"] == {"code": "permission_error", "status": 403}


# ---- 2. the local stop after an admission 403 ------------------------------------------------------

def test_after_an_admission_403_later_calls_fail_locally_without_contacting_openai(tmp_path):
    seed_chatgpt(tmp_path)
    upstream = Upstream(lambda n, r: denied())
    app = make_app(tmp_path, upstream)

    with pytest.raises(UpstreamError) as first:
        run(app.gateway.chat(GPT))
    assert first.value.code == "access_denied"
    for _ in range(3):
        with pytest.raises(UpstreamError) as later:
            run(app.gateway.chat(GPT))
        assert later.value.message == ("ChatGPT plan access denied; check https://chatgpt.com/settings/usage, "
                                       "then run login chatgpt")
        assert (later.value.status, later.value.code) == (403, "plan_access_denied")

    assert len(upstream.openai_requests) == 1                        # the one refused call, nothing after it
    assert [l["error"]["code"] for l in meter_lines(tmp_path)] == ["access_denied"] + ["plan_access_denied"] * 3


def test_the_stop_covers_a_token_that_would_otherwise_be_refreshed(tmp_path):
    seed_chatgpt(tmp_path, expires_at=NOW + 10)
    now = [NOW]

    def script(n, request):
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": "a2", "refresh_token": "r2", "expires_in": 3600})
        return denied()

    upstream = Upstream(script)
    app = make_app(tmp_path, upstream, clock=lambda: now[0])
    with pytest.raises(UpstreamError):
        run(app.gateway.chat(GPT))                                   # refresh, then the 403
    seen = len(upstream.openai_requests)
    now[0] = NOW + 7200                                              # the stored token is now expired
    with pytest.raises(UpstreamError, match="plan access denied"):
        run(app.gateway.chat(GPT))
    assert len(upstream.openai_requests) == seen                     # no refresh and no request after the stop


def test_a_changed_login_record_lifts_the_stop(tmp_path):
    store = seed_chatgpt(tmp_path, access="old-token")
    upstream = Upstream(lambda n, r: denied() if n == 1 else responses_ok())
    app = make_app(tmp_path, upstream)

    with pytest.raises(UpstreamError):
        run(app.gateway.chat(GPT))
    with pytest.raises(UpstreamError, match="plan access denied"):
        run(app.gateway.chat(GPT))
    seed(store, "chatgpt", access="new-token", refresh="new-refresh", meta={"client_id": ISSUED})     # a new login

    chat = run(app.gateway.chat(GPT))
    assert chat["choices"][0]["message"]["content"] == "Paris."
    assert upstream.bearers == ["Bearer old-token", "Bearer new-token"]


def test_deleting_the_record_does_not_lift_the_stop(tmp_path):
    seed_chatgpt(tmp_path)
    app = make_app(tmp_path, Upstream(lambda n, r: denied()))
    with pytest.raises(UpstreamError):
        run(app.gateway.chat(GPT))
    app.store.delete("chatgpt")
    with pytest.raises(UpstreamError) as err:
        run(app.gateway.chat(GPT))
    assert err.value.message == ACCESS_DENIED_MESSAGE                # not "login needed": the stop still holds


def test_the_stop_lasts_for_the_process_only(tmp_path):
    seed_chatgpt(tmp_path)
    upstream = Upstream(lambda n, r: denied() if n == 1 else responses_ok())
    with pytest.raises(UpstreamError):
        run(make_app(tmp_path, upstream).gateway.chat(GPT))
    run(make_app(tmp_path, upstream).gateway.chat(GPT))              # a new process asks OpenAI again
    assert len(upstream.openai_requests) == 2


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_other_failures_do_not_stop_chatgpt(tmp_path, status):
    seed_chatgpt(tmp_path, expires_at=NOW + 7200)
    upstream = Upstream(lambda n, r: httpx.Response(status, json={"error": {"message": "x"}}))
    app = make_app(tmp_path, upstream)
    for _ in range(2):
        with pytest.raises(UpstreamError) as err:
            run(app.gateway.chat(GPT))
        assert err.value.code != "plan_access_denied"
    assert len([r for r in upstream.openai_requests if str(r.url) != TOKEN_URL]) >= 2


def test_the_stop_does_not_touch_claude(tmp_path):
    store = seed_chatgpt(tmp_path)
    seed(store, "anthropic")
    app = make_app(tmp_path, Upstream(lambda n, r: denied()))
    with pytest.raises(UpstreamError):
        run(app.gateway.chat(GPT))
    assert run(app.gateway.chat(CLAUDE))["choices"][0]["message"]["content"] == "Paris."


def test_a_call_queued_behind_a_refused_one_fails_locally(tmp_path):
    seed_chatgpt(tmp_path)
    upstream = Upstream(lambda n, r: denied(), delay=0.05)
    app = make_app(tmp_path, upstream)

    async def go():
        return await asyncio.gather(*[app.gateway.chat(GPT) for _ in range(3)], return_exceptions=True)

    results = run(go())
    assert all(isinstance(r, UpstreamError) for r in results)
    assert sorted(r.code for r in results) == ["access_denied", "plan_access_denied", "plan_access_denied"]
    assert len(upstream.openai_requests) == 1


# ---- 3. earliest_refresh_at ---------------------------------------------------------------------

def token_endpoint(upstream_calls: list, earliest=None):
    def reply(n, request):
        upstream_calls.append(str(request.url))
        body = {"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600}
        if earliest is not None:
            body["earliest_refresh_at"] = earliest
        return httpx.Response(200, json=body)
    return reply


def window_upstream(token_calls: list, earliest=None):
    def script(n, request):
        if str(request.url) == TOKEN_URL:
            return token_endpoint(token_calls, earliest)(n, request)
        return responses_ok()
    return Upstream(script)


def test_before_earliest_refresh_at_the_current_token_is_used_even_inside_the_skew(tmp_path):
    seed_chatgpt(tmp_path, access="access-1", refresh="refresh-1", expires_at=NOW + 250,
                 meta={"client_id": ISSUED, "earliest_refresh_at": NOW + 100})
    token_calls: list = []
    upstream = window_upstream(token_calls)
    run(make_app(tmp_path, upstream, clock=lambda: NOW).gateway.chat(GPT))      # 250 s left, skew 300, window opens at +100
    assert token_calls == [] and upstream.bearers == ["Bearer access-1"]


@pytest.mark.parametrize("offset", [0, 40])
def test_from_earliest_refresh_at_on_a_token_near_expiry_is_refreshed(tmp_path, offset):
    store = seed_chatgpt(tmp_path, access="access-1", refresh="refresh-1", expires_at=NOW + 250,
                         meta={"client_id": ISSUED, "earliest_refresh_at": NOW + 100})
    token_calls: list = []
    upstream = window_upstream(token_calls, earliest=NOW + 100 + offset + 3240)
    now = NOW + 100 + offset                                                   # at the boundary and after it
    run(make_app(tmp_path, upstream, clock=lambda: now).gateway.chat(GPT))
    assert token_calls == [TOKEN_URL] and upstream.bearers[-1] == "Bearer access-2"
    assert store.load("chatgpt").meta["earliest_refresh_at"] == now + 3240      # the new window is stored


def test_inside_the_window_a_token_that_is_not_near_expiry_is_not_refreshed(tmp_path):
    seed_chatgpt(tmp_path, access="access-1", refresh="refresh-1", expires_at=NOW + 3000,
                 meta={"client_id": ISSUED, "earliest_refresh_at": NOW - 10})
    token_calls: list = []
    upstream = window_upstream(token_calls)
    run(make_app(tmp_path, upstream).gateway.chat(GPT))
    assert token_calls == []


def test_a_401_before_the_window_opens_does_not_attempt_a_refresh(tmp_path):
    seed_chatgpt(tmp_path, access="access-1", refresh="refresh-1", expires_at=NOW + 3000,
                 meta={"client_id": ISSUED, "earliest_refresh_at": NOW + 600})
    token_calls: list = []

    def script(n, request):
        if str(request.url) == TOKEN_URL:
            return token_endpoint(token_calls)(n, request)
        return httpx.Response(401, json={"error": {"message": "stale"}})

    with pytest.raises(UpstreamError) as err:
        run(make_app(tmp_path, Upstream(script)).gateway.chat(GPT))
    assert err.value.status == 401 and token_calls == []


@pytest.mark.parametrize("raw, expected", [
    (1_800_000_100, 1_800_000_100.0), (1_800_000_100.5, 1_800_000_100.5), ("1800000100", 1_800_000_100.0),
    ("2027-01-15T08:00:00Z", 1_800_000_000.0), ("not a time", None), (True, None), (None, None), ([], None),
])
def test_earliest_refresh_at_accepts_a_number_a_numeric_string_or_iso_time(raw, expected):
    meta = {} if raw is None else {"earliest_refresh_at": raw}
    assert Credentials("a", "r", NOW, meta=meta).earliest_refresh_at == expected


def test_a_record_without_the_field_refreshes_as_before(tmp_path):
    seed_chatgpt(tmp_path, access="access-1", refresh="refresh-1", expires_at=NOW + 60)
    token_calls: list = []
    run(make_app(tmp_path, window_upstream(token_calls)).gateway.chat(GPT))
    assert token_calls == [TOKEN_URL]


# ---- 4. the ChatGPT concurrency cap -------------------------------------------------------------

def build_app(tmp_path, upstream, **kw):
    return app_module.build(tmp_path, transport=httpx.MockTransport(upstream), clock=lambda: NOW,
                            meter_file=tmp_path / "usage.jsonl", sleep=Sleeper(), **kw)


@pytest.mark.parametrize("kwargs, expected_peak", [({}, 1), ({"chatgpt_concurrency": 1}, 1), ({"chatgpt_concurrency": 2}, 2)])
def test_chatgpt_calls_run_at_most_the_configured_number_at_a_time(tmp_path, kwargs, expected_peak):
    seed_chatgpt(tmp_path)
    upstream = Upstream(delay=0.05)
    app = build_app(tmp_path, upstream, **kwargs)

    async def go():
        return await asyncio.gather(*[app.gateway.chat(GPT) for _ in range(4)])

    assert len(run(go())) == 4
    assert upstream.peak == expected_peak and len(upstream.openai_requests) == 4


def test_the_cap_is_chatgpt_only_claude_calls_still_run_in_parallel(tmp_path):
    seed(make_store(tmp_path), "anthropic")
    upstream = Upstream(delay=0.05)
    app = build_app(tmp_path, upstream)

    async def go():
        return await asyncio.gather(*[app.gateway.chat(CLAUDE) for _ in range(3)])

    run(go())
    assert upstream.peak == 3


def test_config_toml_sets_the_cap_and_the_flag_overrides_it(tmp_path):
    (tmp_path / "config.toml").write_text("[chatgpt]\nmax_concurrency = 2\n")
    assert load_config(tmp_path).chatgpt.max_concurrency == 2
    seed_chatgpt(tmp_path)
    upstream = Upstream(delay=0.05)

    async def go(app):
        return await asyncio.gather(*[app.gateway.chat(GPT) for _ in range(3)])

    run(go(build_app(tmp_path, upstream)))
    assert upstream.peak == 2
    upstream.peak = 0
    run(go(build_app(tmp_path, upstream, chatgpt_concurrency=1)))
    assert upstream.peak == 1


@pytest.mark.parametrize("value", [0, 3, -1, "2"])
def test_a_cap_outside_one_to_two_is_a_configuration_error(tmp_path, value):
    (tmp_path / "config.toml").write_text(f"[chatgpt]\nmax_concurrency = {json.dumps(value)}\n")
    with pytest.raises(GatewayError, match="max_concurrency must be 1 or 2"):
        load_config(tmp_path)


def test_the_serve_flag_accepts_one_or_two_only():
    parser = cli._parser()
    assert parser.parse_args(["serve", "--chatgpt-concurrency", "2"]).chatgpt_concurrency == 2
    assert parser.parse_args(["serve"]).chatgpt_concurrency is None
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--chatgpt-concurrency", "3"])


def test_a_flag_value_outside_one_to_two_is_refused_by_build(tmp_path):
    with pytest.raises(GatewayError, match="must be 1 or 2"):
        build_app(tmp_path, Upstream(), chatgpt_concurrency=3)


# ---- 5. logout revokes -----------------------------------------------------------------------------

REVOCATION_ENDPOINT = "https://auth.openai.com/oauth/revoke"


def logout_upstream(*, discovery=None, revoke_statuses=(200,), seen=None, store=None):
    statuses = list(revoke_statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return discovery() if callable(discovery) else httpx.Response(
                200, json=discovery if discovery is not None else {"revocation_endpoint": REVOCATION_ENDPOINT})
        if seen is not None:
            seen.append({"url": str(request.url), "type": request.headers["content-type"],
                         "form": {k: v[0] for k, v in parse_qs(request.content.decode()).items()},
                         "record_present": store is not None and store.snapshot("chatgpt").raw is not None})
        status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        if status is None:
            raise httpx.ConnectError("down")
        return httpx.Response(status)
    return handler


def logout(store, handler, *, sleeper=None):
    said: list[str] = []
    run(logout_chatgpt(store, say=said.append, transport=httpx.MockTransport(handler), sleep=sleeper or Sleeper()))
    return said


def test_logout_posts_the_refresh_token_to_the_discovered_endpoint_then_deletes_the_file(tmp_path):
    store = seed_chatgpt(tmp_path, refresh=SENTINEL_REFRESH)
    seen: list = []
    said = logout(store, logout_upstream(seen=seen, store=store))

    assert seen == [{"url": REVOCATION_ENDPOINT, "type": "application/x-www-form-urlencoded",
                     "form": {"token": SENTINEL_REFRESH, "token_type_hint": "refresh_token", "client_id": ISSUED},
                     "record_present": True}]                   # the record was still there when the POST went out
    assert store.snapshot("chatgpt").raw is None
    assert said == ["chatgpt: refresh token revoked and local login removed"]
    assert SENTINEL_REFRESH not in " ".join(said)


@pytest.mark.parametrize("statuses, waits", [((500, 200), [0.5]), ((None, 503, 200), [0.5, 1.0])])
def test_logout_retries_network_failures_and_5xx_with_backoff(tmp_path, statuses, waits):
    store = seed_chatgpt(tmp_path)
    sleeper = Sleeper()
    seen: list = []
    said = logout(store, logout_upstream(revoke_statuses=statuses, seen=seen), sleeper=sleeper)
    assert len(seen) == len(statuses)                                 # one POST per attempt
    assert sleeper.waits == waits and said[-1].endswith("local login removed") and "revoked" in said[-1]
    assert store.snapshot("chatgpt").raw is None


def test_logout_deletes_the_file_and_says_so_when_revocation_cannot_be_confirmed(tmp_path):
    store = seed_chatgpt(tmp_path)
    sleeper = Sleeper()
    said = logout(store, logout_upstream(revoke_statuses=(503,)), sleeper=sleeper)
    assert sleeper.waits == [0.5, 1.0]                              # three attempts, then it gives up
    assert store.snapshot("chatgpt").raw is None
    assert "revocation not confirmed" in said[0]
    assert "remote revocation was not confirmed" in said[-1] and "https://chatgpt.com/settings/usage" in said[-1]


def test_logout_does_not_retry_a_client_error(tmp_path):
    store = seed_chatgpt(tmp_path)
    seen: list = []
    sleeper = Sleeper()
    said = logout(store, logout_upstream(revoke_statuses=(400,), seen=seen), sleeper=sleeper)
    assert len(seen) == 1 and sleeper.waits == [] and store.snapshot("chatgpt").raw is None
    assert "HTTP 400" in said[0]


@pytest.mark.parametrize("discovery", [
    {"issuer": "https://auth.openai.com"}, {"revocation_endpoint": "http://auth.openai.com/oauth/revoke"},
    {"revocation_endpoint": 7}, ["not", "an", "object"],
])
def test_logout_without_a_usable_revocation_endpoint_posts_nothing_and_still_deletes(tmp_path, discovery):
    store = seed_chatgpt(tmp_path)
    seen: list = []
    said = logout(store, logout_upstream(discovery=discovery, seen=seen))
    assert seen == [] and store.snapshot("chatgpt").raw is None
    assert "no https revocation_endpoint" in said[0]


def test_logout_with_no_record_contacts_nobody(tmp_path):
    store = make_store(tmp_path)

    def forbidden(request):
        raise AssertionError("no request expected")

    assert logout(store, forbidden) == ["chatgpt: no stored login to remove"]


def test_logout_clears_an_unreadable_record_without_any_request(tmp_path):
    store = seed_chatgpt(tmp_path)
    (tmp_path / "oauth" / "chatgpt.json").write_text("{not json")

    def forbidden(request):
        raise AssertionError("no request expected")

    said = logout(store, forbidden)
    assert "unreadable" in said[0] and store.snapshot("chatgpt").raw is None


def test_logout_cli_for_chatgpt_with_no_login_makes_no_request_and_anthropic_stays_local(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path))
    seed(make_store(tmp_path), "anthropic")
    assert cli.main(["logout", "chatgpt"]) == 0
    assert cli.main(["logout", "anthropic"]) == 0
    out = capsys.readouterr().out
    assert "chatgpt: no stored login to remove" in out
    assert "anthropic: local login removed (tokens were not revoked remotely)" in out
    assert make_store(tmp_path).snapshot("anthropic").raw is None


# ---- 6. tools -------------------------------------------------------------------------------------

TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Look up weather",
                                         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


def test_function_tools_travel_as_an_additional_tools_item_at_the_head_of_the_input():
    payload, _ = tr.build_request({**CHAT, "tools": [TOOL]}, "gpt-6.1-sol")
    assert "tools" not in payload                                    # no flat top-level function tools
    assert payload["input"][0] == {"type": "additional_tools", "role": "developer", "tools": [
        {"type": "function", "name": "get_weather", "description": "Look up weather",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}, "strict": None}]}
    assert payload["input"][1] == {"role": "user", "content": "hi"}
    assert payload["parallel_tool_calls"] is True


def test_no_tools_means_no_additional_tools_item_and_tool_choice_none_drops_them():
    plain, _ = tr.build_request(CHAT, "m")
    assert plain["input"] == [{"role": "user", "content": "hi"}] and "parallel_tool_calls" not in plain
    off, _ = tr.build_request({**CHAT, "tools": [TOOL], "tool_choice": "none"}, "m")
    assert off["input"] == [{"role": "user", "content": "hi"}] and "tool_choice" not in off


def test_tool_choice_is_sent_only_when_the_caller_sets_it():
    default, _ = tr.build_request({**CHAT, "tools": [TOOL]}, "m")
    assert "tool_choice" not in default
    required, _ = tr.build_request({**CHAT, "tools": [TOOL], "tool_choice": "required"}, "m")
    assert required["tool_choice"] == "required"
    with pytest.raises(BadRequest):
        tr.build_request({**CHAT, "tools": [TOOL], "tool_choice": "sometimes"}, "m")


def test_replayed_tool_calls_keep_the_plain_function_call_items():
    messages = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}]
    payload, _ = tr.build_request({"messages": messages, "tools": [TOOL]}, "m")
    assert [i.get("type") or i.get("role") for i in payload["input"]] == [
        "additional_tools", "user", "function_call", "function_call_output"]
    assert "namespace" not in payload["input"][2]


# ---- 7. attribution text and Manage usage ------------------------------------------------------------

def test_status_shows_the_plan_notice_only_for_a_chatgpt_login(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    assert STATUS_NOTICE not in status_lines(store, NOW)
    seed(store, "chatgpt", meta={"client_id": ISSUED})
    lines = status_lines(store, NOW)
    assert lines[2] == "Using ChatGPT plan (Manage usage: https://chatgpt.com/settings/usage)"
    assert lines[1].startswith("chatgpt: logged in")


def test_serve_startup_prints_the_plan_notice(tmp_path, capsys):
    seed_chatgpt(tmp_path)
    args = argparse.Namespace(meter=None, host="127.0.0.1", port=0, chatgpt_concurrency=None)
    with pytest.raises(asyncio.TimeoutError):
        run(asyncio.wait_for(cli._serve(tmp_path, args), 1.0))
    assert "Using ChatGPT plan (Manage usage: https://chatgpt.com/settings/usage)" in capsys.readouterr().out


@pytest.mark.parametrize("status, leads", [(403, True), (429, True), (400, False), (500, False), (401, False)])
def test_403_and_429_messages_lead_with_the_manage_usage_link(tmp_path, status, leads):
    seed_chatgpt(tmp_path, expires_at=NOW + 7200)
    app = make_app(tmp_path, Upstream(lambda n, r: httpx.Response(status, json={"error": {"message": "limit"}})))
    with pytest.raises(UpstreamError) as err:
        run(app.gateway.chat(GPT))
    assert err.value.message.startswith(LEAD) is leads


def test_an_in_stream_usage_limit_failure_leads_with_the_link():
    raw = ("event: response.failed\ndata: " + json.dumps({"type": "response.failed", "response": {"error": {
        "code": "subscription_sharing_usage_limit_exceeded", "message": "used up"}}}) + "\n\n")
    reply = httpx.Response(200, content=raw.encode(), headers={"content-type": "text/event-stream"})
    with pytest.raises(UpstreamError) as err:
        run(_feed(reply))
    assert err.value.status == 429 and err.value.message.startswith(LEAD + "chatgpt response failed:")


async def _feed(response):
    from hybrid.gateway.sse import iter_events
    accumulator = tr.ResponseAccumulator()
    async for event, data in iter_events(response):
        accumulator.feed(event, data)


def test_login_prints_the_plan_attribution_after_the_saved_line(tmp_path):
    said, holder, hits = [], {}, []

    def open_browser(url):
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        holder["pending"] = chatgpt_auth.PendingLogin(url, q["state"], q["nonce"], "unused", q["redirect_uri"], None,
                                                      q["ext_agent_host_id"])
        hits.append(asyncio.ensure_future(_hit(q)))

    async def _hit(q):
        async with httpx.AsyncClient() as client:
            await client.get(f"{q['redirect_uri']}?code=c&state={q['state']}&client_id={ISSUED}")

    async def go():
        creds = await login_chatgpt(make_store(tmp_path), tmp_path, ChatGPTConfig(callback_port=0), say=said.append,
                                    open_browser=open_browser,
                                    transport=httpx.MockTransport(lambda r: token_handler(holder["pending"])(r)),
                                    clock=lambda: NOW, timeout_s=20)
        await asyncio.gather(*hits)
        return creds

    run(go())
    assert said[-2].startswith("chatgpt: saved.")
    assert said[-1] == "You're using your ChatGPT plan. Manage usage: https://chatgpt.com/settings/usage" == LOGIN_NOTICE


# ---- 8. login chatgpt --new-registration --------------------------------------------------------------

def saved_login():
    return Credentials("a", "r", NOW, id_token="ID-TOKEN-VALUE-12345",
                       meta={"client_id": ISSUED, "email": "me@example.com"})


def test_new_registration_uses_the_dynamic_client_with_the_same_host_id_and_no_hints():
    pending = chatgpt_auth.begin_login(host_id="urn:uuid:keep-me", config=ChatGPTConfig(client_id="oaiapp_from_config"),
                                       saved=saved_login(), port=1455, new_registration=True)
    q = {k: v[0] for k, v in parse_qs(urlparse(pending.url).query).items()}
    assert q["client_id"] == "dynamic_agent_client" and q["agent_name_hint"] == "Bigbrane"
    assert q["ext_agent_host_id"] == "urn:uuid:keep-me"
    assert "id_token_hint" not in q and "login_hint" not in q
    assert pending.client_id is None                                 # so the callback must return an issued id


def test_without_the_flag_the_saved_client_is_still_reused():
    pending = chatgpt_auth.begin_login(host_id="urn:uuid:1", config=ChatGPTConfig(), saved=saved_login(), port=1455)
    assert parse_qs(urlparse(pending.url).query)["client_id"] == [ISSUED]


def test_new_registration_login_replaces_the_record_and_keeps_the_host_id_file(tmp_path):
    store = make_store(tmp_path)
    seed(store, "chatgpt", id_token="ID-TOKEN-VALUE-12345", meta={"client_id": "oaiapp_old", "email": "me@example.com"})
    said, holder, hits = [], {}, []
    host_before = host_id(tmp_path)

    def open_browser(url):
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        holder.update(q)
        holder["pending"] = chatgpt_auth.PendingLogin(url, q["state"], q["nonce"], "unused", q["redirect_uri"], None,
                                                      q["ext_agent_host_id"])
        hits.append(asyncio.ensure_future(_hit(q)))

    async def _hit(q):
        async with httpx.AsyncClient() as client:
            await client.get(f"{q['redirect_uri']}?code=c&state={q['state']}&client_id={ISSUED}")

    async def go():
        creds = await login_chatgpt(store, tmp_path, ChatGPTConfig(callback_port=0), say=said.append,
                                    open_browser=open_browser, new_registration=True,
                                    transport=httpx.MockTransport(lambda r: token_handler(holder["pending"])(r)),
                                    clock=lambda: NOW, timeout_s=20)
        await asyncio.gather(*hits)
        return creds

    run(go())
    assert holder["client_id"] == "dynamic_agent_client" and holder["ext_agent_host_id"] == host_before
    assert store.load("chatgpt").meta["client_id"] == ISSUED
    assert (tmp_path / "host-id").read_text().strip() == host_before
    assert any("only after OpenAI answered invalid_client" in line for line in said)


def test_the_flag_is_chatgpt_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path))
    assert cli.main(["login", "anthropic", "--new-registration"]) == 2
    assert "applies to the chatgpt login only" in capsys.readouterr().err
