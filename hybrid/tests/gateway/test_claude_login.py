"""Claude login with a loopback callback racing the pasted code, as LLxprt Code does (anthropic_callback.py, login.py)."""
import asyncio
import socket
import threading
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from hybrid.gateway import anthropic_auth
from hybrid.gateway.anthropic_callback import CALLBACK_PORT_RANGE, start_listener
from hybrid.gateway.errors import BadRequest
from hybrid.gateway.login import login_anthropic
from .support import NOW, make_store, run

CONSOLE_REDIRECT = "https://console.anthropic.com/oauth/code/callback"


def query_of(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


class TokenEndpoint:
    """Records token-exchange bodies and answers like Anthropic."""

    def __init__(self):
        self.bodies = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        import json
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"access_token": "A", "refresh_token": "R", "expires_in": 100})


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def hit(url):
    async with httpx.AsyncClient() as client:
        return await client.get(url)


def blocking_until_cancelled(prompt, cancel: threading.Event):
    """A paste prompt nobody types into: returns None once the login calls it off."""
    cancel.wait(10)
    return None


# ---- authorize URL and exchange ----------------------------------------------------------------------------------

def test_the_authorize_url_carries_the_loopback_redirect_when_one_is_given():
    url, verifier = anthropic_auth.new_login("http://localhost:8766/callback")
    q = query_of(url)
    assert q["redirect_uri"] == "http://localhost:8766/callback" and q["state"] == verifier
    assert q["client_id"] == "9d1c250a-e61b-44d9-88ed-5944d1962f5e" and q["code_challenge_method"] == "S256"
    assert q["scope"] == "org:create_api_key user:profile user:inference" and q["code"] == "true"


def test_the_code_exchange_sends_the_redirect_the_authorize_url_used():
    endpoint = TokenEndpoint()
    auth = anthropic_auth.AnthropicAuth(httpx.MockTransport(endpoint), lambda: NOW)
    run(auth.exchange_code("c#v", "v", "http://localhost:8766/callback"))
    run(auth.exchange_code("c#v", "v"))
    assert [b["redirect_uri"] for b in endpoint.bodies] == ["http://localhost:8766/callback", CONSOLE_REDIRECT]


# ---- the listener ------------------------------------------------------------------------------------------------

def test_listener_answers_with_code_and_state_and_only_on_the_callback_path():
    async def go():
        listener = await start_listener("the-state", [0])
        try:
            assert urlparse(listener.redirect_uri).hostname == "localhost" and listener.redirect_uri.endswith("/callback")
            port = urlparse(listener.redirect_uri).port
            other = await hit(f"http://127.0.0.1:{port}/other?code=x&state=the-state")
            ok = await hit(f"http://127.0.0.1:{port}/callback?code=abc&state=the-state")
            return other.status_code, ok.status_code, await listener.wait()
        finally:
            await listener.close()
    assert run(go()) == (404, 200, "abc#the-state")


@pytest.mark.parametrize("query,message", [("code=abc&state=other", "state mismatch"), ("state=the-state", "missing code"),
                                           ("code=abc", "missing code")])
def test_listener_refuses_a_callback_that_is_not_for_this_login(query, message):
    async def go():
        listener = await start_listener("the-state", [0])
        try:
            response = await hit(f"http://127.0.0.1:{urlparse(listener.redirect_uri).port}/callback?{query}")
            with pytest.raises(BadRequest, match=message):
                await listener.wait()
            return response.status_code
        finally:
            await listener.close()
    assert run(go()) == 400


def test_listener_skips_busy_ports_and_fails_when_none_is_free():
    async def go():
        busy = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        busy_port = busy.sockets[0].getsockname()[1]
        try:
            listener = await start_listener("s", [busy_port, 0])
            assert urlparse(listener.redirect_uri).port != busy_port
            await listener.close()
            with pytest.raises(BadRequest, match="no free port"):
                await start_listener("s", [busy_port])
        finally:
            busy.close()
            await busy.wait_closed()
    run(go())


def test_llxprts_port_range_is_8765_to_8795():
    assert CALLBACK_PORT_RANGE == (8765, 8795)


# ---- the login ---------------------------------------------------------------------------------------------------

def test_the_browser_callback_completes_the_login_and_calls_off_the_paste_prompt(tmp_path):
    store, endpoint, said, opened, cancelled = make_store(tmp_path), TokenEndpoint(), [], [], []

    def ask_until(prompt, cancel):
        blocking_until_cancelled(prompt, cancel)
        cancelled.append(cancel.is_set())
        return None

    async def go():
        tasks = []

        def open_browser(url):
            opened.append(url)
            q = query_of(url)
            tasks.append(asyncio.ensure_future(hit(f"{q['redirect_uri']}?code=browser-code&state={q['state']}")))

        creds = await login_anthropic(store, ask=lambda p: pytest.fail("paste-only prompt used"), say=said.append,
                                      open_browser=open_browser, transport=httpx.MockTransport(endpoint),
                                      clock=lambda: NOW, callback_ports=[0], ask_until=ask_until)
        await asyncio.gather(*tasks)
        return creds

    creds = run(go())
    q = query_of(opened[0])
    assert q["redirect_uri"].startswith("http://localhost:") and q["redirect_uri"].endswith("/callback")
    assert endpoint.bodies == [{"grant_type": "authorization_code", "code": "browser-code", "state": q["state"],
                                "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e", "redirect_uri": q["redirect_uri"],
                                "code_verifier": q["state"]}]
    assert store.load("anthropic").access_token == "A" and creds.refresh_token == "R"
    assert cancelled == [True]
    paste_urls = [line for line in said if line.startswith("https://claude.ai/oauth/authorize")]
    assert [query_of(u)["redirect_uri"] for u in paste_urls] == [CONSOLE_REDIRECT]   # shown for a remote browser
    assert not any("browser-code" in line for line in said)


def test_a_pasted_code_wins_when_the_browser_never_calls_back(tmp_path):
    store, endpoint, opened = make_store(tmp_path), TokenEndpoint(), []

    def ask_until(prompt, cancel):
        q = query_of(opened[0])
        return f"pasted-code#{q['state']}"

    run(login_anthropic(store, ask=lambda p: pytest.fail("paste-only prompt used"), say=lambda s: None,
                        open_browser=opened.append, transport=httpx.MockTransport(endpoint), clock=lambda: NOW,
                        callback_ports=[0], ask_until=ask_until))
    assert endpoint.bodies[0]["code"] == "pasted-code" and endpoint.bodies[0]["redirect_uri"] == CONSOLE_REDIRECT
    assert store.load("anthropic").refresh_token == "R"
    port = urlparse(query_of(opened[0])["redirect_uri"]).port
    with socket.socket() as probe:                                    # the listener was closed
        assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_a_bad_browser_callback_does_not_end_the_paste_option(tmp_path):
    store, endpoint, said, opened = make_store(tmp_path), TokenEndpoint(), [], []
    release = threading.Event()

    def ask_until(prompt, cancel):
        release.wait(10)                                              # typed only after the bad callback was refused
        return f"pasted#{query_of(opened[0])['state']}"

    async def go():
        tasks = []

        def open_browser(url):
            opened.append(url)
            q = query_of(url)
            async def bad_callback():
                await hit(f"{q['redirect_uri']}?code=x&state=wrong")
                release.set()
            tasks.append(asyncio.ensure_future(bad_callback()))
        await login_anthropic(store, ask=lambda p: "", say=said.append, open_browser=open_browser,
                              transport=httpx.MockTransport(endpoint), clock=lambda: NOW, callback_ports=[0],
                              ask_until=ask_until)
        await asyncio.gather(*tasks)
    run(go())
    assert any("browser callback was refused" in line for line in said)
    assert endpoint.bodies[0]["code"] == "pasted"


def test_without_a_free_callback_port_the_login_falls_back_to_the_paste_flow(tmp_path):
    store, endpoint, said, opened = make_store(tmp_path), TokenEndpoint(), [], []

    async def go():
        busy = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        port = busy.sockets[0].getsockname()[1]
        try:
            await login_anthropic(store, ask=lambda p: f"pasted#{query_of(opened[0])['state']}", say=said.append,
                                  open_browser=opened.append, transport=httpx.MockTransport(endpoint),
                                  clock=lambda: NOW, callback_ports=[port], ask_until=blocking_until_cancelled)
        finally:
            busy.close()
            await busy.wait_closed()
    run(go())
    assert query_of(opened[0])["redirect_uri"] == CONSOLE_REDIRECT
    assert endpoint.bodies[0]["redirect_uri"] == CONSOLE_REDIRECT
    assert any("No loopback callback" in line for line in said)


def test_the_login_times_out_when_neither_arrives(tmp_path):
    with pytest.raises(BadRequest, match="no login arrived"):
        run(login_anthropic(make_store(tmp_path), ask=lambda p: "", say=lambda s: None, open_browser=lambda u: None,
                            transport=httpx.MockTransport(TokenEndpoint()), clock=lambda: NOW, callback_ports=[0],
                            ask_until=blocking_until_cancelled, callback_timeout_s=0.3))
    assert make_store(tmp_path).load("anthropic") is None


def test_paste_only_login_is_unchanged_without_callback_ports(tmp_path):
    store, endpoint, opened = make_store(tmp_path), TokenEndpoint(), []
    run(login_anthropic(store, ask=lambda p: f"c#{query_of(opened[0])['state']}", say=lambda s: None,
                        open_browser=opened.append, transport=httpx.MockTransport(endpoint), clock=lambda: NOW))
    assert query_of(opened[0])["redirect_uri"] == CONSOLE_REDIRECT and endpoint.bodies[0]["redirect_uri"] == CONSOLE_REDIRECT
