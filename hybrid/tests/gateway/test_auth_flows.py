"""Login flows for both providers against mocked token endpoints."""
import asyncio
import base64
import hashlib
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from hybrid.gateway import anthropic_auth, chatgpt_auth
from hybrid.gateway.errors import BadRequest, UpstreamError
from hybrid.gateway.login import login_anthropic, login_chatgpt
from hybrid.gateway.paths import ChatGPTConfig
from hybrid.gateway.refresher import RefreshRejected
from hybrid.gateway.store import Credentials, TokenStore
from .support import NOW, jwks, make_store, run, sign_id_token

ISSUED = "oaiapp_issued123"


# ---- Anthropic -----------------------------------------------------------------------------------

def test_anthropic_authorization_url_matches_llxprt():
    url, verifier = anthropic_auth.new_login()
    parsed = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://claude.ai/oauth/authorize"
    assert q["client_id"] == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    assert q["scope"] == "org:create_api_key user:profile user:inference"
    assert q["redirect_uri"] == "https://console.anthropic.com/oauth/code/callback"
    assert q["code"] == "true" and q["response_type"] == "code" and q["code_challenge_method"] == "S256"
    assert q["state"] == verifier           # the verifier doubles as state, as in LLxprt
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert q["code_challenge"] == expected


def test_anthropic_code_exchange_sends_json_like_llxprt():
    seen = {}

    def handler(request):
        seen["url"], seen["type"], seen["body"] = str(request.url), request.headers["content-type"], json.loads(request.content)
        return httpx.Response(200, json={"access_token": "A", "refresh_token": "R", "expires_in": 28800, "scope": "user:inference"})

    auth = anthropic_auth.AnthropicAuth(httpx.MockTransport(handler), lambda: NOW)
    creds = run(auth.exchange_code("the-code#the-verifier", "the-verifier"))
    assert seen["url"] == "https://console.anthropic.com/v1/oauth/token"
    assert seen["type"] == "application/json"
    assert seen["body"] == {"grant_type": "authorization_code", "code": "the-code", "state": "the-verifier",
                            "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                            "redirect_uri": "https://console.anthropic.com/oauth/code/callback", "code_verifier": "the-verifier"}
    assert creds.expires_at == NOW + 28800 and creds.refresh_token == "R"


def test_anthropic_exchange_rejects_a_code_from_another_attempt():
    auth = anthropic_auth.AnthropicAuth(httpx.MockTransport(lambda r: pytest.fail("must not call the endpoint")), lambda: NOW)
    with pytest.raises(BadRequest, match="state mismatch"):
        run(auth.exchange_code("code#other-state", "verifier"))


def test_anthropic_refresh_is_form_encoded_and_keeps_the_old_refresh_token_when_none_is_returned():
    seen = {}

    def handler(request):
        seen["type"], seen["form"] = request.headers["content-type"], parse_qs(request.content.decode())
        return httpx.Response(200, json={"access_token": "A2", "expires_in": 100})

    old = Credentials("A1", "R1", NOW - 1, meta={"scope": "s"})
    new = run(anthropic_auth.AnthropicAuth(httpx.MockTransport(handler), lambda: NOW).refresh(old))
    assert seen["type"] == "application/x-www-form-urlencoded"
    assert seen["form"] == {"grant_type": ["refresh_token"], "refresh_token": ["R1"],
                            "client_id": ["9d1c250a-e61b-44d9-88ed-5944d1962f5e"]}
    assert new.access_token == "A2" and new.refresh_token == "R1" and new.expires_at == NOW + 100


def test_login_anthropic_saves_to_the_store_with_hidden_paste(tmp_path):
    store = make_store(tmp_path)
    opened, said = [], []

    def handler(request):
        return httpx.Response(200, json={"access_token": "A", "refresh_token": "R", "expires_in": 100})

    def ask(prompt):
        verifier = parse_qs(urlparse(opened[0]).query)["state"][0]
        return f"pasted-code#{verifier}"

    creds = run(login_anthropic(store, ask=ask, say=said.append, open_browser=opened.append,
                                transport=httpx.MockTransport(handler), clock=lambda: NOW))
    assert store.load("anthropic").access_token == "A" and creds.refresh_token == "R"
    assert not any("A" == s or "pasted-code" in s for s in said)


# ---- ChatGPT -------------------------------------------------------------------------------------

def test_new_registration_url_uses_dynamic_client_and_host_id():
    pending = chatgpt_auth.begin_login(host_id="urn:uuid:1234", config=ChatGPTConfig(), saved=None, port=1455)
    parsed = urlparse(pending.url)
    q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://auth.openai.com/api/accounts/authorize"
    assert q["client_id"] == "dynamic_agent_client" and q["agent_name_hint"] == "Bigbrane"
    assert q["ext_agent_host_id"] == "urn:uuid:1234"
    assert q["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"
    assert q["scope"] == "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
    assert q["resource"] == "https://api.openai.com/v1" and q["response_type"] == "code"
    assert q["code_challenge_method"] == "S256" and q["state"] and q["nonce"] and q["code_challenge"]
    assert "id_token_hint" not in q


def test_returning_sign_in_reuses_the_issued_client_and_hides_the_id_token_hint():
    saved = Credentials("a", "r", NOW, id_token="ID-TOKEN-VALUE-12345", meta={"client_id": ISSUED, "email": "me@example.com"})
    pending = chatgpt_auth.begin_login(host_id="urn:uuid:1234", config=ChatGPTConfig(), saved=saved, port=54321)
    q = {k: v[0] for k, v in parse_qs(urlparse(pending.url).query).items()}
    assert q["client_id"] == ISSUED and "agent_name_hint" not in q
    assert q["id_token_hint"] == "ID-TOKEN-VALUE-12345" and q["login_hint"] == "me@example.com"
    assert "ID-TOKEN-VALUE-12345" not in pending.display_url()


def test_config_file_client_id_is_used_when_nothing_is_saved():
    pending = chatgpt_auth.begin_login(host_id="urn:uuid:1", config=ChatGPTConfig(client_id="oaiapp_from_config"), saved=None, port=1455)
    q = {k: v[0] for k, v in parse_qs(urlparse(pending.url).query).items()}
    assert q["client_id"] == "oaiapp_from_config" and "agent_name_hint" not in q


def id_claims(pending, **over):
    claims = {"iss": "https://auth.openai.com", "aud": ISSUED, "sub": "user-1", "email": "me@example.com",
              "exp": NOW + 3600, "iat": NOW, "nonce": pending.nonce}
    claims.update(over)
    return claims


def token_handler(pending, *, scope="openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
                  claims=None, tamper=False, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/token"):
            if seen is not None:
                seen.update({k: v[0] for k, v in parse_qs(request.content.decode()).items()})
                seen["content-type"] = request.headers["content-type"]
            return httpx.Response(200, json={
                "access_token": "chatgpt-access", "refresh_token": "chatgpt-refresh", "expires_in": 3600, "scope": scope,
                "token_type": "Bearer", "id_token": sign_id_token(claims or id_claims(pending), tamper=tamper)})
        if request.url.path.endswith("jwks.json"):
            return httpx.Response(200, json=jwks())
        raise AssertionError(f"unexpected {request.url}")
    return handler


def new_pending():
    return chatgpt_auth.begin_login(host_id="urn:uuid:1", config=ChatGPTConfig(), saved=None, port=1455)


def callback_for(pending, **over):
    params = {"code": "auth-code", "state": pending.state, "client_id": ISSUED}
    params.update(over)
    return {k: v for k, v in params.items() if v is not None}


def test_complete_login_exchanges_the_code_and_validates_the_id_token():
    pending, seen = new_pending(), {}
    auth = chatgpt_auth.ChatGPTAuth(httpx.MockTransport(token_handler(pending, seen=seen)), lambda: NOW)
    creds = run(auth.complete_login(pending, callback_for(pending)))
    assert seen["content-type"] == "application/x-www-form-urlencoded"
    assert {k: seen[k] for k in ("grant_type", "client_id", "code", "code_verifier", "redirect_uri", "resource")} == {
        "grant_type": "authorization_code", "client_id": ISSUED, "code": "auth-code", "code_verifier": pending.verifier,
        "redirect_uri": "http://127.0.0.1:1455/auth/callback", "resource": "https://api.openai.com/v1"}
    assert "client_secret" not in seen
    assert creds.meta["client_id"] == ISSUED and creds.meta["email"] == "me@example.com"
    assert creds.meta["ext_agent_host_id"] == "urn:uuid:1" and "chatgpt.tokens.use.direct" in creds.meta["scopes"]
    assert creds.refresh_token == "chatgpt-refresh" and creds.expires_at == NOW + 3600 and creds.id_token


@pytest.mark.parametrize("label,callback,handler_kwargs,message", [
    ("state mismatch", {"state": "forged"}, {}, "state does not match"),
    ("declined", {"error": "access_denied", "code": None}, {}, "consent was declined"),
    ("registration without issued id", {"client_id": None}, {}, "issued client id"),
    ("dynamic id echoed back", {"client_id": "dynamic_agent_client"}, {}, "issued client id"),
    ("no plan scope", {}, {"scope": "openid profile email offline_access"}, "chatgpt.tokens.use.direct"),
    ("bad signature", {}, {"tamper": True}, "signature does not verify"),
    ("wrong nonce", {}, {"claims_over": {"nonce": "other"}}, "nonce"),
    ("wrong audience", {}, {"claims_over": {"aud": "oaiapp_other"}}, "audience"),
    ("wrong issuer", {}, {"claims_over": {"iss": "https://evil.example"}}, "issuer"),
    ("expired id token", {}, {"claims_over": {"exp": NOW - 3600}}, "expired"),
])
def test_complete_login_rejections(label, callback, handler_kwargs, message):
    pending = new_pending()
    over = handler_kwargs.pop("claims_over", None)
    if over:
        handler_kwargs["claims"] = id_claims(pending, **over)
    auth = chatgpt_auth.ChatGPTAuth(httpx.MockTransport(token_handler(pending, **handler_kwargs)), lambda: NOW)
    with pytest.raises(BadRequest, match=message):
        run(auth.complete_login(pending, callback_for(pending, **callback)))


def test_reauthorization_rejects_a_callback_for_a_different_client():
    saved = Credentials("a", "r", NOW, id_token="x" * 20, meta={"client_id": ISSUED})
    pending = chatgpt_auth.begin_login(host_id="urn:uuid:1", config=ChatGPTConfig(), saved=saved, port=1455)
    auth = chatgpt_auth.ChatGPTAuth(httpx.MockTransport(lambda r: pytest.fail("no exchange expected")), lambda: NOW)
    with pytest.raises(BadRequest, match="different client id"):
        run(auth.complete_login(pending, callback_for(pending, client_id="oaiapp_other")))


def test_chatgpt_refresh_sends_documented_form_and_stores_the_replacement_refresh_token():
    seen = {}

    def handler(request):
        seen.update({k: v[0] for k, v in parse_qs(request.content.decode()).items()})
        return httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600,
                                         "scope": "openid chatgpt.tokens.use.direct offline_access"})

    old = Credentials("old-access", "old-refresh", NOW - 1, id_token="idtok", meta={"client_id": ISSUED})
    new = run(chatgpt_auth.ChatGPTAuth(httpx.MockTransport(handler), lambda: NOW).refresh(old))
    assert seen == {"grant_type": "refresh_token", "client_id": ISSUED, "refresh_token": "old-refresh",
                    "resource": "https://api.openai.com/v1"}          # no scope: omitted to retain the grant
    assert new.refresh_token == "new-refresh" and new.access_token == "new-access" and new.id_token == "idtok"
    assert new.meta["client_id"] == ISSUED


@pytest.mark.parametrize("code", ["invalid_grant", "refresh_token_reused", "refresh_token_expired", "refresh_token_invalidated",
                                  "invalid_refresh_token", "token_expired"])
def test_documented_unusable_refresh_token_codes_require_a_new_login(code):
    handler = lambda request: httpx.Response(400, json={"error": code})
    old = Credentials("a", "r", NOW - 1, meta={"client_id": ISSUED})
    with pytest.raises(RefreshRejected):
        run(chatgpt_auth.ChatGPTAuth(httpx.MockTransport(handler), lambda: NOW).refresh(old))


def test_invalid_client_is_a_configuration_error_not_a_login_problem():
    handler = lambda request: httpx.Response(401, json={"error": "invalid_client"})
    old = Credentials("a", "r", NOW - 1, meta={"client_id": ISSUED})
    with pytest.raises(UpstreamError) as err:
        run(chatgpt_auth.ChatGPTAuth(httpx.MockTransport(handler), lambda: NOW).refresh(old))
    assert "invalid_client" in str(err.value)


def test_login_chatgpt_end_to_end_over_a_real_loopback_callback(tmp_path):
    store = make_store(tmp_path)
    said, state = [], {}
    pending_holder = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        # The fake token endpoint signs an ID token over the nonce taken from the login URL.
        return token_handler(pending_holder["pending"])(request)

    async def go():
        callback_task = []

        def open_browser(url):
            q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
            state.update(q)
            pending_holder["pending"] = chatgpt_auth.PendingLogin(url, q["state"], q["nonce"], "unused", q["redirect_uri"], None, q["ext_agent_host_id"])
            async def hit():
                async with httpx.AsyncClient() as client:
                    await client.get(f"{q['redirect_uri']}?code=auth-code&state={q['state']}&client_id={ISSUED}")
            callback_task.append(asyncio.ensure_future(hit()))

        creds = await login_chatgpt(store, tmp_path, ChatGPTConfig(callback_port=0), say=said.append,
                                    open_browser=open_browser, transport=httpx.MockTransport(upstream),
                                    clock=lambda: NOW, timeout_s=20)
        await asyncio.gather(*callback_task)
        return creds

    creds = run(go())
    assert state["client_id"] == "dynamic_agent_client" and state["redirect_uri"].startswith("http://127.0.0.1:")
    stored = store.load("chatgpt")
    assert stored.refresh_token == "chatgpt-refresh" and stored.meta["client_id"] == ISSUED
    assert (tmp_path / "host-id").read_text().startswith("urn:uuid:")
    assert not any("chatgpt-access" in line or "chatgpt-refresh" in line for line in said)
    assert creds.meta["email"] == "me@example.com"


def test_login_chatgpt_times_out_when_no_callback_arrives(tmp_path):
    with pytest.raises(BadRequest, match="no login callback"):
        run(login_chatgpt(make_store(tmp_path), tmp_path, ChatGPTConfig(callback_port=0), say=lambda s: None,
                          open_browser=lambda url: None, timeout_s=0.2))


# ---- token lifetime ------------------------------------------------------------------------------

def _jwt(claims: dict) -> str:
    import base64
    enc = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return f"{enc(b'{}')}.{enc(json.dumps(claims).encode())}.sig"


def test_expiry_comes_from_expires_in_when_present():
    from hybrid.gateway.oauth_common import expires_at
    assert expires_at(NOW, 3600, "opaque-token") == NOW + 3600


def test_missing_expires_in_uses_the_access_token_jwt_exp_claim():
    from hybrid.gateway.oauth_common import expires_at
    assert expires_at(NOW, None, _jwt({"exp": int(NOW) + 7200})) == int(NOW) + 7200


@pytest.mark.parametrize("expires_in, token", [
    (None, "opaque-not-a-jwt"),                         # no lifetime is ever guessed
    (None, _jwt({"sub": "x"})),                         # JWT without exp
    (None, _jwt({"exp": "soon"})),                      # exp is not an integer
    (None, _jwt({"exp": int(NOW) - 5})),                # already expired
    (None, "a.!!!.c"),                                  # payload is not base64 JSON
    (0, "opaque"), (-5, "opaque"), ("3600", "opaque"), (True, "opaque"),
])
def test_unusable_lifetime_is_rejected_not_defaulted(expires_in, token):
    from hybrid.gateway.oauth_common import expires_at
    with pytest.raises(UpstreamError, match="expires_in|exp claim"):
        expires_at(NOW, expires_in, token)


def test_anthropic_refresh_response_without_expires_in_and_without_a_jwt_is_rejected():
    handler = lambda request: httpx.Response(200, json={"access_token": "sk-ant-opaque"})
    old = Credentials("A1", "R1", NOW - 1)
    with pytest.raises(UpstreamError, match="exp claim"):
        run(anthropic_auth.AnthropicAuth(httpx.MockTransport(handler), lambda: NOW).refresh(old))


def test_anthropic_refresh_response_without_expires_in_uses_the_jwt_exp():
    access = _jwt({"exp": int(NOW) + 900})
    handler = lambda request: httpx.Response(200, json={"access_token": access})
    new = run(anthropic_auth.AnthropicAuth(httpx.MockTransport(handler), lambda: NOW).refresh(Credentials("A1", "R1", NOW - 1)))
    assert new.expires_at == int(NOW) + 900 and new.refresh_token == "R1"
