"""SIWC durability: ChatGPT pacing, the refresh request shape, keeping a valid token after a refused refresh, the
proactive refresh scheduler, the refresh journal and the status lines."""
import asyncio
import json
from urllib.parse import parse_qsl

import httpx
import pytest

from hybrid.gateway import app as app_module
from hybrid.gateway.chatgpt_auth import TOKEN_URL
from hybrid.gateway.chatgpt_pacing import Pacer, PacingPolicy, cap_effort
from hybrid.gateway.errors import GatewayError, LoginRequired, RefreshFailed
from hybrid.gateway.login import chatgpt_durability_lines
from hybrid.gateway.paths import load_config
from hybrid.gateway.refresh_schedule import IDLE_POLL_S
from .support import NOW, SENTINELS, make_store, responses_ok, run, seed

CHAT = {"model": "chatgpt/gpt-6.1-sol", "messages": [{"role": "user", "content": "hi"}]}
META = {"client_id": "oaiapp_test", "earliest_refresh_at": NOW - 10}


class FakeTime:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self, now: float = NOW) -> None:
        self.now = now
        self.waits: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


class FakeOpenAI:
    """Token endpoint with scripted answers plus a Responses endpoint that records when and with what it was called."""

    def __init__(self, time: FakeTime, token_answers=None) -> None:
        self.time = time
        self.token_answers = list(token_answers or [])
        self.token_requests: list[httpx.Request] = []
        self.inference: list[tuple[float, str, dict]] = []      # (time, bearer, payload)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            self.token_requests.append(request)
            return self.token_answers.pop(0)
        if str(request.url).endswith("/v1/models"):
            self.inference.append((self.time.now, request.headers["authorization"], {}))
            return httpx.Response(200, json={"models": [{"slug": "gpt-6.1-sol", "visibility": "list"}]})
        self.inference.append((self.time.now, request.headers["authorization"], json.loads(request.content)))
        return responses_ok()


def rotated(n: int = 1, *, refresh: bool = True) -> httpx.Response:
    body = {"access_token": f"access-{n}", "token_type": "Bearer", "expires_in": 3600, "earliest_refresh_at": NOW + 3240,
            "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"}
    if refresh:
        body["refresh_token"] = f"refresh-{n}"
    return httpx.Response(200, json=body)


INVALIDATED = httpx.Response(400, json={"error": "invalid_grant", "error_reason": "refresh_token_invalidated",
                                        "error_description": "The refresh token has been invalidated. A fresh sign-in "
                                                             "is required."},
                             headers={"x-request-id": "req-abc123"})


def build(tmp_path, time: FakeTime, upstream, pacing: PacingPolicy | None = None):
    return app_module.build(tmp_path, transport=httpx.MockTransport(upstream), clock=time.clock, sleep=time.sleep,
                            meter_file=tmp_path / "usage.jsonl", pacing=pacing)


# ---- pacing ----------------------------------------------------------------------------------------

def test_request_starts_are_at_least_the_minimum_interval_apart():
    time = FakeTime()
    pacer = Pacer(PacingPolicy(min_start_interval_s=10, max_requests_per_hour=60), clock=time.clock, sleep=time.sleep)

    async def go():
        return [await pacer.admit() for _ in range(3)]

    assert run(go()) == [0.0, 10.0, 10.0]
    assert time.now == NOW + 20


def test_a_gap_already_longer_than_the_interval_costs_no_wait():
    time = FakeTime()
    pacer = Pacer(PacingPolicy(min_start_interval_s=10), clock=time.clock, sleep=time.sleep)
    run(pacer.admit())
    time.now += 25
    assert run(pacer.admit()) == 0.0


def test_requests_beyond_the_hourly_cap_wait_for_the_window_instead_of_failing():
    time = FakeTime()
    pacer = Pacer(PacingPolicy(min_start_interval_s=0, max_requests_per_hour=3), clock=time.clock, sleep=time.sleep)

    async def go():
        return [await pacer.admit() for _ in range(5)]

    assert run(go()) == [0.0, 0.0, 0.0, 3600.0, 0.0]
    assert pacer.starts_in_last_hour() == 2        # the first three left the window when the fourth started


def test_concurrent_callers_are_admitted_in_arrival_order_and_spaced():
    time = FakeTime()
    pacer = Pacer(PacingPolicy(min_start_interval_s=10), clock=time.clock, sleep=time.sleep)
    order: list[tuple[int, float]] = []

    async def caller(i):
        await pacer.admit()
        order.append((i, time.now))

    async def go():
        await asyncio.gather(*(caller(i) for i in range(4)))

    run(go())
    assert [i for i, _ in order] == [0, 1, 2, 3]
    starts = [t for _, t in order]
    assert all(b - a >= 10 for a, b in zip(starts, starts[1:]))


@pytest.mark.parametrize("requested, sent, replaced", [
    ("high", "medium", "high"), ("xhigh", "medium", "xhigh"), ("max", "medium", "max"),
    ("medium", "medium", None), ("low", "low", None), ("minimal", "minimal", None),
    (None, "medium", "(model default)"), ("turbo", "medium", "turbo"),
])
def test_reasoning_effort_is_lowered_to_the_ceiling(requested, sent, replaced):
    payload = {"reasoning": {"effort": requested}} if requested else {}
    assert cap_effort(payload, "medium") == replaced
    assert payload["reasoning"]["effort"] == sent


def test_an_empty_ceiling_leaves_the_effort_alone():
    payload = {"reasoning": {"effort": "high"}}
    assert cap_effort(payload, "") is None and payload == {"reasoning": {"effort": "high"}}


def test_gateway_spaces_chatgpt_calls_and_caps_their_effort(tmp_path):
    time = FakeTime()
    seed(make_store(tmp_path), "chatgpt", meta=dict(META))
    upstream = FakeOpenAI(time)
    app = build(tmp_path, time, upstream)

    async def go():
        await asyncio.gather(*(app.gateway.chat({**CHAT, "reasoning_effort": "high"}) for _ in range(3)))
        await app.gateway.models()

    run(go())
    starts = [t for t, _, _ in upstream.inference]
    assert starts == [NOW, NOW + 10, NOW + 20, NOW + 30]          # the catalog call is paced too
    assert [p["reasoning"] for _, _, p in upstream.inference[:3]] == [{"effort": "medium"}] * 3


def test_pacing_does_not_touch_anthropic(tmp_path):
    from .support import anthropic_ok
    time = FakeTime()
    seed(make_store(tmp_path), "anthropic")
    app = build(tmp_path, time, lambda request: anthropic_ok())
    chat = {"model": "anthropic/claude-haiku-5-5", "messages": [{"role": "user", "content": "hi"}]}
    run(app.gateway.chat(chat))
    run(app.gateway.chat(chat))
    assert time.waits == []


def test_pacing_settings_come_from_config_toml(tmp_path):
    (tmp_path / "config.toml").write_text("[chatgpt]\nmin_start_interval_s = 30\nmax_requests_per_hour = 20\n"
                                          "max_reasoning_effort = \"low\"\nproactive_refresh = false\n")
    config = load_config(tmp_path).chatgpt
    assert (config.min_start_interval_s, config.max_requests_per_hour, config.max_reasoning_effort,
            config.proactive_refresh) == (30, 20, "low", False)
    built = app_module.build(tmp_path)
    assert built.pacing == PacingPolicy(1, 30.0, 20, "low") and built.schedulers == ()


def test_defaults_are_the_conservative_policy(tmp_path):
    built = app_module.build(tmp_path)
    assert built.pacing == PacingPolicy(max_concurrency=1, min_start_interval_s=10.0, max_requests_per_hour=60,
                                        max_reasoning_effort="medium")
    assert len(built.schedulers) == 1


@pytest.mark.parametrize("line", ["min_start_interval_s = -1", "max_requests_per_hour = 0",
                                  "max_requests_per_hour = 1.5", "max_reasoning_effort = \"ultra\"",
                                  "proactive_refresh = 1"])
def test_bad_pacing_settings_are_rejected(tmp_path, line):
    (tmp_path / "config.toml").write_text(f"[chatgpt]\n{line}\n")
    with pytest.raises(GatewayError):
        load_config(tmp_path)


# ---- refresh request ---------------------------------------------------------------------------------

def test_refresh_request_matches_the_documented_and_devkit_request_byte_for_byte(tmp_path):
    time = FakeTime()
    seed(make_store(tmp_path), "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 60, meta=dict(META))
    upstream = FakeOpenAI(time, [rotated()])
    run(build(tmp_path, time, upstream).gateway.chat(CHAT))
    [request] = upstream.token_requests
    assert str(request.url) == "https://auth.openai.com/api/accounts/oauth/token"
    assert request.method == "POST"
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert request.headers["accept"] == "application/json"
    assert request.content.decode() == ("grant_type=refresh_token&client_id=oaiapp_test&refresh_token=refresh-0"
                                        "&resource=https%3A%2F%2Fapi.openai.com%2Fv1")
    assert "scope" not in dict(parse_qsl(request.content.decode()))
    assert upstream.inference[0][1] == "Bearer access-1"


def test_no_refresh_before_earliest_refresh_at_even_inside_the_skew(tmp_path):
    time = FakeTime()
    seed(make_store(tmp_path), "chatgpt", access="access-0", expires_at=NOW + 60,
         meta={**META, "earliest_refresh_at": NOW + 30})
    upstream = FakeOpenAI(time, [rotated()])
    run(build(tmp_path, time, upstream).gateway.chat(CHAT))
    assert upstream.token_requests == [] and upstream.inference[0][1] == "Bearer access-0"


def test_a_refresh_answer_without_a_replacement_never_keeps_the_consumed_refresh_token(tmp_path):
    time = FakeTime()
    store = make_store(tmp_path)
    seed(store, "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 60, meta=dict(META))
    run(build(tmp_path, time, FakeOpenAI(time, [rotated(refresh=False)])).gateway.chat(CHAT))
    saved = store.load("chatgpt")
    assert saved.access_token == "access-1" and saved.refresh_token is None


def test_a_refused_refresh_keeps_the_valid_access_token_until_it_expires(tmp_path):
    time = FakeTime()
    store = make_store(tmp_path)
    seed(store, "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 120, meta=dict(META))
    upstream = FakeOpenAI(time, [INVALIDATED])
    app = build(tmp_path, time, upstream, pacing=PacingPolicy(min_start_interval_s=0))
    run(app.gateway.chat(CHAT))
    run(app.gateway.chat(CHAT))
    assert [bearer for _, bearer, _ in upstream.inference] == ["Bearer access-0"] * 2
    assert len(upstream.token_requests) == 1                     # the dead refresh token is never sent again
    time.now = NOW + 121
    with pytest.raises(LoginRequired) as err:
        run(app.gateway.chat(CHAT))
    assert "refresh_token_invalidated" in str(err.value) and "req-abc123" in str(err.value)
    assert len(upstream.token_requests) == 1 and len(upstream.inference) == 2
    assert store.load("chatgpt").refresh_token == "refresh-0"    # kept for diagnosis; a login replaces it


def test_the_journal_records_each_refresh_outcome_without_tokens(tmp_path):
    time = FakeTime()
    seed(make_store(tmp_path), "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 60, meta=dict(META))
    upstream = FakeOpenAI(time, [rotated()])
    app = build(tmp_path, time, upstream)
    run(app.gateway.chat(CHAT))
    ok = app.journal.last("chatgpt")
    assert ok["outcome"] == "ok" and ok["rotated"] is True and ok["seconds_before_expiry"] == 60
    assert ok["expires_at"] == "2027-01-15T09:00:00Z" and ok["next_earliest_refresh_at"] == "2027-01-15T08:54:00Z"

    seed(make_store(tmp_path), "chatgpt", access="access-x", refresh="refresh-x", expires_at=time.now + 60,
         meta=dict(META))
    upstream.token_answers.append(INVALIDATED)
    run(app.gateway.chat(CHAT))
    bad = app.journal.last("chatgpt")
    assert {k: bad[k] for k in ("outcome", "http_status", "error", "error_reason", "request_id")} == {
        "outcome": "rejected", "http_status": 400, "error": "invalid_grant", "error_reason": "refresh_token_invalidated",
        "request_id": "req-abc123"}
    text = (tmp_path / "refresh.jsonl").read_text()
    assert not any(secret in text for secret in ("refresh-0", "refresh-1", "access-1", "refresh-x", *SENTINELS))


def test_a_transient_refresh_failure_is_journaled_and_retried_later(tmp_path):
    time = FakeTime()
    seed(make_store(tmp_path), "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 60, meta=dict(META))
    upstream = FakeOpenAI(time, [httpx.Response(503, json={"error": "temporarily_unavailable"}), rotated()])
    app = build(tmp_path, time, upstream)
    with pytest.raises(RefreshFailed):
        run(app.gateway.chat(CHAT))
    assert app.journal.last("chatgpt")["outcome"] == "failed"
    run(app.gateway.chat(CHAT))
    assert app.journal.last("chatgpt")["outcome"] == "ok"
    assert [bearer for _, bearer, _ in upstream.inference] == ["Bearer access-1"]


# ---- proactive refresh -----------------------------------------------------------------------------

def test_scheduler_sleeps_until_the_window_opens_then_refreshes_once_without_any_request(tmp_path):
    time = FakeTime()
    store = make_store(tmp_path)
    # Like the 2026-10-11 login: earliest_refresh_at six minutes before a one-hour expiry.
    seed(store, "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 3600,
         meta={**META, "earliest_refresh_at": NOW + 3240})
    upstream = FakeOpenAI(time, [rotated()])
    app = build(tmp_path, time, upstream)
    [scheduler] = app.schedulers
    due = NOW + 3600 - 300                         # max(earliest_refresh_at, expiry - skew)
    while time.now < due:
        delay = run(scheduler.step())
        assert upstream.token_requests == []
        time.now += delay
    assert time.now == due
    assert run(scheduler.step()) == IDLE_POLL_S
    assert len(upstream.token_requests) == 1 and store.load("chatgpt").access_token == "access-1"
    assert upstream.inference == []
    nxt = run(scheduler.step())                    # the next window is an hour away; nothing more is sent
    assert 0 < nxt <= IDLE_POLL_S and len(upstream.token_requests) == 1


def test_scheduler_never_refreshes_an_expired_token_before_earliest_refresh_at(tmp_path):
    time = FakeTime()
    seed(make_store(tmp_path), "chatgpt", expires_at=NOW - 5, meta={**META, "earliest_refresh_at": NOW + 40})
    upstream = FakeOpenAI(time, [rotated()])
    [scheduler] = build(tmp_path, time, upstream).schedulers
    assert run(scheduler.step()) == 40 and upstream.token_requests == []


def test_scheduler_stops_after_a_refusal_until_a_new_login_appears(tmp_path):
    time = FakeTime()
    store = make_store(tmp_path)
    seed(store, "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 60, meta=dict(META))
    upstream = FakeOpenAI(time, [INVALIDATED, rotated(2)])
    app = build(tmp_path, time, upstream)
    [scheduler] = app.schedulers
    run(scheduler.step())
    for _ in range(5):
        assert run(scheduler.step()) == IDLE_POLL_S
    assert len(upstream.token_requests) == 1
    seed(store, "chatgpt", access="access-new", refresh="refresh-new", expires_at=time.now + 30, meta=dict(META))
    run(scheduler.step())
    assert len(upstream.token_requests) == 2 and store.load("chatgpt").access_token == "access-2"


def test_scheduler_run_backs_off_on_transient_failures(tmp_path):
    time = FakeTime()
    seed(make_store(tmp_path), "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 60, meta=dict(META))
    fail = httpx.Response(503, json={"error": "temporarily_unavailable"})
    upstream = FakeOpenAI(time, [fail] * 20)
    [scheduler] = build(tmp_path, time, upstream).schedulers

    async def go():
        task = asyncio.create_task(scheduler.run())
        while len(upstream.token_requests) < 3:
            await asyncio.sleep(0)
        task.cancel()

    run(go())
    assert time.waits[:2] == [15.0, 30.0]


# ---- status -----------------------------------------------------------------------------------------

def test_status_lines_show_pacing_refresh_window_last_refresh_and_hourly_calls(tmp_path):
    time = FakeTime()
    store = make_store(tmp_path)
    seed(store, "chatgpt", access="access-0", refresh="refresh-0", expires_at=NOW + 60, meta=dict(META))
    app = build(tmp_path, time, FakeOpenAI(time, [rotated()]))
    run(app.gateway.chat(CHAT))
    lines = chatgpt_durability_lines(store.load("chatgpt"), pacing=app.pacing, skew_s=300, proactive_refresh=True,
                                     last_refresh=app.journal.last("chatgpt"), meter_file=app.meter.path,
                                     now=time.now)
    text = "\n".join(lines)
    assert "at least 10 s between request starts" in text and "at most 60 requests per hour" in text
    assert "reasoning effort capped at medium" in text and "concurrency 1" in text
    assert "earliest_refresh_at 2027-01-15T08:54:00Z, expires_at 2027-01-15T09:00:00Z" in text
    assert "refresh due from 2027-01-15T08:55:00Z" in text
    assert "chatgpt last refresh:" in text and " ok (" in text
    assert not any(secret in text for secret in ("access-0", "refresh-0", "access-1", "refresh-1"))


def test_status_counts_the_last_hours_chatgpt_calls_from_the_usage_log(tmp_path):
    from datetime import datetime, timezone
    def ts(epoch):
        return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    meter = tmp_path / "usage.jsonl"
    events = [("chatgpt", NOW - 4000), ("chatgpt", NOW - 3000), ("chatgpt", NOW - 10), ("anthropic", NOW - 5)]
    meter.write_text("".join(json.dumps({"ts": ts(t), "provider": p}) + "\n" for p, t in events) + "not json\n")
    lines = chatgpt_durability_lines(None, pacing=PacingPolicy(), skew_s=300, proactive_refresh=True,
                                     last_refresh=None, meter_file=meter, now=NOW)
    assert "chatgpt calls in the last hour (usage log): 2" in lines
    assert "chatgpt last refresh: none recorded yet" in lines
