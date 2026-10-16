"""Sentinel test: planted token values must not appear in any response, log line, meter record or status output."""
import json
import logging

import httpx

from hybrid.gateway.anthropic_auth import TOKEN_URL as ANTHROPIC_TOKEN_URL
from hybrid.gateway.login import status_lines
from hybrid.gateway.server import GatewayServer
from .support import (NOW, SENTINEL_ACCESS, SENTINEL_ID_TOKEN, SENTINEL_REFRESH, SENTINELS, anthropic_ok,
                      chat_fixture, make_app, make_store, responses_ok, run, seed, sse_response)

RUN_TOKEN = "SENTINEL-RUNTOKEN-90ab12cd34ef56a7"
NEW_ACCESS = "SENTINEL-NEWACCESS-5566778899aabbcc"
NEW_REFRESH = "SENTINEL-NEWREFRESH-ddeeff0011223344"
EVERYTHING = SENTINELS + (RUN_TOKEN, NEW_ACCESS, NEW_REFRESH)


def echoing_upstream(request: httpx.Request) -> httpx.Response:
    """A hostile upstream: every failure echoes the credentials it was sent back in the body."""
    url, bearer = str(request.url), request.headers.get("authorization", "")
    body = request.content.decode(errors="replace")
    if url == ANTHROPIC_TOKEN_URL:
        if SENTINEL_REFRESH in body:      # first refresh succeeds and rotates to new sentinel values
            return httpx.Response(200, json={"access_token": NEW_ACCESS, "refresh_token": NEW_REFRESH, "expires_in": 3600})
        return httpx.Response(400, json={"error": "invalid_grant", "error_description": f"bad refresh token {body}"})
    if url.endswith("/v1/messages"):
        payload = json.loads(body)
        mode = payload["messages"][-1]["content"][-1].get("text", "")
        if mode == "plain-400":
            return httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": f"rejected {bearer}"}})
        if mode == "html-500":
            return httpx.Response(500, text=f"<html>upstream crash for {bearer}</html>")
        if mode == "stream-error":
            return sse_response(("error", {"type": "error", "error": {"type": "api_error", "message": f"oops {bearer}"}}))
        if mode == "crash":
            raise RuntimeError(f"unexpected failure with {bearer}")
        return anthropic_ok()
    if url.endswith("/v1/responses"):
        return sse_response(("response.failed", {"type": "response.failed", "response": {
            "error": {"code": "weird_error", "message": f"failed for {bearer} and {SENTINEL_ID_TOKEN}"}}}))
    raise AssertionError(url)


def chat(model, text):
    return {"model": model, "messages": [{"role": "user", "content": text}]}


def test_no_token_value_appears_anywhere_a_user_or_log_reader_could_see(tmp_path, caplog, capsys):
    store = make_store(tmp_path)
    seed(store, "anthropic", access=SENTINEL_ACCESS, refresh=SENTINEL_REFRESH, id_token=SENTINEL_ID_TOKEN, expires_at=NOW - 10)
    seed(store, "chatgpt", access=SENTINEL_ACCESS, refresh=SENTINEL_REFRESH, id_token=SENTINEL_ID_TOKEN,
         meta={"client_id": "oaiapp_x"}, expires_at=NOW + 7200)
    app = make_app(tmp_path, echoing_upstream)
    server = GatewayServer(app.gateway, RUN_TOKEN, app.redactor, port=0)
    auth = {"authorization": f"Bearer {RUN_TOKEN}"}
    caplog.set_level(logging.DEBUG)

    async def go():
        port = await server.start()
        seen = []
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
                for mode in ("ok", "plain-400", "html-500", "stream-error", "crash"):
                    seen.append(await client.post("/v1/chat/completions", json=chat("anthropic/claude-opus-5-5", mode), headers=auth))
                seen.append(await client.post("/v1/chat/completions", json=chat("chatgpt/gpt-6.1-sol", "x"), headers=auth))
                seen.append(await client.post("/v1/chat/completions", json=chat("anthropic/claude-opus-5-5", "x"),
                                              headers={"authorization": f"Bearer {RUN_TOKEN}x"}))
                seen.append(await client.get("/v1/models", headers=auth))
                seen.append(await client.get("/health"))
        finally:
            await server.close()
        return seen

    responses = run(go())
    assert [r.status_code for r in responses][:5] == [200, 400, 502, 502, 500]
    assert responses[0].json()["choices"][0]["message"]["content"] == "Paris."

    # A dead refresh token path too: the token endpoint echoes the request body in its error.
    seed(store, "anthropic", access=SENTINEL_ACCESS, refresh="SENTINEL-DEADREFRESH-0011223344556677", expires_at=NOW - 10)
    app2 = make_app(tmp_path, echoing_upstream)
    server2 = GatewayServer(app2.gateway, RUN_TOKEN, app2.redactor, port=0)

    async def dead():
        port = await server2.start()
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
                return await client.post("/v1/chat/completions", json=chat("anthropic/claude-opus-5-5", "x"), headers=auth)
        finally:
            await server2.close()

    dead_response = run(dead())
    assert dead_response.status_code == 401 and "login anthropic" in dead_response.text

    surfaces = [r.text + json.dumps(dict(r.headers)) for r in responses + [dead_response]]
    surfaces.append(caplog.text)
    surfaces.append("\n".join(status_lines(app.store, NOW)))
    surfaces.append((tmp_path / "usage.jsonl").read_text())
    captured = capsys.readouterr()
    surfaces += [captured.out, captured.err, repr(app.store.load("anthropic")), repr(app.store.load("chatgpt"))]
    haystack = "\n".join(surfaces)

    for value in EVERYTHING + ("SENTINEL-DEADREFRESH-0011223344556677",):
        assert value not in haystack, f"{value} leaked"
    assert "[redacted]" in haystack          # the scrubber ran on at least one echoed credential
    assert "chatgpt" in haystack and "weird_error" in haystack      # diagnostics survive the scrubbing


def test_status_output_shows_expiry_and_never_token_values(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", expires_at=NOW + 600, id_token=SENTINEL_ID_TOKEN)
    seed(store, "chatgpt", expires_at=NOW - 60, meta={"client_id": "oaiapp_x", "email": "me@example.com"})
    lines = status_lines(store, NOW)
    text = "\n".join(lines)
    assert "anthropic: logged in" in lines[0] and "10 min left" in lines[0]
    assert "chatgpt: logged in, access token expired" in lines[1] and "me@example.com" in lines[1] and "oaiapp_x" in lines[1]
    assert not any(s in text for s in SENTINELS)


def test_status_reports_missing_logins_with_the_command(tmp_path):
    lines = status_lines(make_store(tmp_path), NOW)
    assert lines == ["anthropic: not logged in (run: python -m hybrid.gateway login anthropic)",
                     "chatgpt: not logged in (run: python -m hybrid.gateway login chatgpt)"]
