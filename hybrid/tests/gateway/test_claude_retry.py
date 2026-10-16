"""Retry and rate-limit handling ported from LLxprt Code (anthropic_retry.py), with a fake upstream and a fake sleeper."""
import httpx
import pytest

from hybrid.gateway import anthropic_retry as retry
from hybrid.gateway.anthropic_retry import RetryPolicy
from hybrid.gateway.errors import UpstreamDetail, UpstreamError
from .support import NOW, Sleeper, anthropic_ok, make_app, make_store, meter_lines, responses_ok, run, seed, sse

BODY = {"model": "anthropic/claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
NO_JITTER = RetryPolicy(jitter=0.0)


def refusal(status, retry_after=None, error_type="rate_limit_error"):
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx.Response(status, json={"type": "error", "error": {"type": error_type, "message": "no"}}, headers=headers)


def stream_error(error_type="overloaded_error"):
    raw = sse(("message_start", {"type": "message_start", "message": {"id": "m", "model": "claude-opus-5-5",
                                                                     "role": "assistant", "usage": {"input_tokens": 1, "output_tokens": 1}}}),
              ("error", {"type": "error", "error": {"type": error_type, "message": "Overloaded"}}))
    return httpx.Response(200, content=raw, headers={"content-type": "text/event-stream"})


def scripted_app(tmp_path, script, policy=NO_JITTER, clock=lambda: NOW):
    """An app whose Anthropic upstream plays ``script`` (responses or exceptions) in order."""
    seed(make_store(tmp_path), "anthropic")
    calls, items = [], iter(script)

    def handler(request):
        calls.append(request)
        item = next(items)
        if isinstance(item, Exception):
            raise item
        return item
    sleeper = Sleeper()
    return make_app(tmp_path, handler, sleep=sleeper, anthropic_retry=policy, clock=clock), calls, sleeper


def chat(app):
    return run(app.gateway.chat(dict(BODY)))


def test_a_429_is_retried_after_the_retry_after_header_and_the_call_succeeds(tmp_path):
    app, calls, sleeper = scripted_app(tmp_path, [refusal(429, "7"), anthropic_ok()])
    assert chat(app)["choices"][0]["message"]["content"] == "Paris."
    assert len(calls) == 2 and sleeper.waits == [7.0]
    [line] = meter_lines(tmp_path)
    assert line["status"] == "ok"                       # one metered call, not one per attempt


def test_retry_after_is_capped_at_five_minutes(tmp_path):
    app, calls, sleeper = scripted_app(tmp_path, [refusal(429, "86400"), anthropic_ok()])
    chat(app)
    assert sleeper.waits == [300.0]


def test_retry_after_may_be_an_http_date(tmp_path):
    date = "Sun, 15 Jan 2027 08:00:30 GMT"            # NOW is 2027-01-15 08:00:00 UTC
    assert retry.parse_retry_after(date, NOW) == 30.0
    assert retry.parse_retry_after("12", NOW) == 12.0
    assert retry.parse_retry_after("Sun, 15 Jan 2027 07:00:00 GMT", NOW) == 0.0
    assert retry.parse_retry_after(None, NOW) is None and retry.parse_retry_after("soon", NOW) is None


def test_5xx_including_529_uses_exponential_backoff_from_four_seconds_up_to_thirty(tmp_path):
    script = [refusal(529, error_type="overloaded_error"), refusal(500), refusal(503), refusal(502), refusal(500), anthropic_ok()]
    app, calls, sleeper = scripted_app(tmp_path, script)
    chat(app)
    assert len(calls) == 6 and sleeper.waits == [4.0, 8.0, 16.0, 30.0, 30.0]


def test_a_5xx_retry_ignores_retry_after_like_llxprt(tmp_path):
    app, _, sleeper = scripted_app(tmp_path, [refusal(529, "99", "overloaded_error"), anthropic_ok()])
    chat(app)
    assert sleeper.waits == [4.0]


def test_six_attempts_are_made_and_the_last_error_is_reported(tmp_path):
    app, calls, sleeper = scripted_app(tmp_path, [refusal(429)] * 6)
    with pytest.raises(UpstreamError) as err:
        chat(app)
    assert len(calls) == 6 and len(sleeper.waits) == 5
    assert err.value.status == 429 and err.value.code == "rate_limit_error"
    [line] = meter_lines(tmp_path)
    assert line["status"] == "error" and line["error"] == {"code": "rate_limit_error", "status": 429}


@pytest.mark.parametrize("status", [400, 403, 404, 413])
def test_client_errors_are_not_retried(tmp_path, status):
    app, calls, sleeper = scripted_app(tmp_path, [refusal(status, error_type="invalid_request_error")])
    with pytest.raises(UpstreamError):
        chat(app)
    assert len(calls) == 1 and sleeper.waits == []


@pytest.mark.parametrize("error_type", ["overloaded_error", "rate_limit_error", "api_error"])
def test_an_error_event_inside_the_stream_is_retried(tmp_path, error_type):
    app, calls, sleeper = scripted_app(tmp_path, [stream_error(error_type), anthropic_ok()])
    chat(app)
    assert len(calls) == 2 and sleeper.waits == [4.0]


def test_an_invalid_request_error_event_inside_the_stream_is_not_retried(tmp_path):
    app, calls, _ = scripted_app(tmp_path, [stream_error("invalid_request_error")])
    with pytest.raises(UpstreamError) as err:
        chat(app)
    assert len(calls) == 1 and err.value.code == "invalid_request_error"


def test_a_stream_that_ends_early_is_retried(tmp_path):
    truncated = httpx.Response(200, content=sse(("message_start", {"type": "message_start", "message": {
        "id": "m", "model": "x", "role": "assistant", "usage": {}}})), headers={"content-type": "text/event-stream"})
    app, calls, _ = scripted_app(tmp_path, [truncated, anthropic_ok()])
    chat(app)
    assert len(calls) == 2


def test_a_transport_failure_is_retried(tmp_path):
    app, calls, sleeper = scripted_app(tmp_path, [httpx.ConnectError("refused"), anthropic_ok()])
    chat(app)
    assert len(calls) == 2 and sleeper.waits == [4.0]


def test_a_401_triggers_one_forced_refresh_and_is_not_otherwise_retried(tmp_path):
    seed(make_store(tmp_path), "anthropic")
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/v1/oauth/token":
            return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "r2", "expires_in": 3600})
        return httpx.Response(401, json={"error": {"type": "authentication_error", "message": "expired"}})
    sleeper = Sleeper()
    app = make_app(tmp_path, handler, sleep=sleeper, anthropic_retry=NO_JITTER)
    with pytest.raises(UpstreamError) as err:
        chat(app)
    messages = [c for c in calls if c.url.path == "/v1/messages"]
    assert err.value.status == 401 and len(messages) == 2 and sleeper.waits == []


def test_the_chatgpt_side_is_not_retried(tmp_path):
    seed(make_store(tmp_path), "chatgpt", meta={"client_id": "oaiapp_x"})
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"type": "rate_limit_error", "message": "no"}})
    sleeper = Sleeper()
    app = make_app(tmp_path, handler, sleep=sleeper)
    with pytest.raises(UpstreamError):
        run(app.gateway.chat({"model": "chatgpt/gpt-6.1-sol", "messages": [{"role": "user", "content": "hi"}]}))
    assert len(calls) == 1 and sleeper.waits == []


def test_jitter_stays_within_thirty_percent_of_the_delay():
    async def attempt(rand):
        waits, failures = [], [UpstreamError("x", status=429, code="rate_limit_error", detail=UpstreamDetail(429, "", None, None))]

        async def sleep(seconds):
            waits.append(seconds)

        async def call():
            if failures:
                raise failures.pop()
            return "ok"
        await retry.with_retries(call, RetryPolicy(), sleep=sleep, rand=rand)
        return waits[0]
    assert run(attempt(lambda: 0.0)) == pytest.approx(2.8)
    assert run(attempt(lambda: 0.5)) == pytest.approx(4.0)
    assert run(attempt(lambda: 1.0)) == pytest.approx(5.2)


# ---- rate-limit throttling ---------------------------------------------------------------------------------------

def limit_headers(remaining, limit=100, reset_in=30.0):
    from datetime import datetime, timezone
    reset = datetime.fromtimestamp(NOW + reset_in, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"anthropic-ratelimit-requests-limit": str(limit), "anthropic-ratelimit-requests-remaining": str(remaining),
            "anthropic-ratelimit-requests-reset": reset}


def ok_with(headers):
    response = anthropic_ok()
    response.headers.update(headers)
    return response


def test_the_next_call_waits_for_the_reset_when_the_last_answer_was_nearly_out_of_requests(tmp_path):
    app, _, sleeper = scripted_app(tmp_path, [ok_with(limit_headers(remaining=4)), anthropic_ok()])
    chat(app)
    assert sleeper.waits == []
    chat(app)
    assert sleeper.waits == [30.0]


def test_the_throttle_wait_is_capped_at_sixty_seconds(tmp_path):
    app, _, sleeper = scripted_app(tmp_path, [ok_with(limit_headers(remaining=1, reset_in=900)), anthropic_ok()])
    chat(app)
    chat(app)
    assert sleeper.waits == [60.0]


def test_no_throttle_wait_while_more_than_five_percent_remains(tmp_path):
    app, _, sleeper = scripted_app(tmp_path, [ok_with(limit_headers(remaining=5)), anthropic_ok()])
    chat(app)
    chat(app)
    assert sleeper.waits == []


def test_rate_limit_header_parsing_ignores_garbage():
    info = retry.parse_rate_limit_headers({"anthropic-ratelimit-requests-limit": "x",
                                           "anthropic-ratelimit-tokens-remaining": "10",
                                           "anthropic-ratelimit-tokens-reset": "not a date"})
    assert info == retry.RateLimitInfo(tokens_remaining=10)
    assert retry.throttle_wait_s(info, NOW) == 0.0
