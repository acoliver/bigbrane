"""Shared helpers for gateway tests: filesystem store seeding, SSE builders, fixtures, app construction."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path

import httpx

from hybrid.gateway import app as app_module
from hybrid.gateway.paths import oauth_dir
from hybrid.gateway.store import Credentials, TokenStore

FIXTURES = Path(__file__).parent / "fixtures"
NOW = 1_800_000_000.0

# Planted values. If any of them shows up in a response, log line or output, a token leaked.
SENTINEL_ACCESS = "SENTINEL-ACCESS-7f3a9c1e5b2d4a60"
SENTINEL_REFRESH = "SENTINEL-REFRESH-b81d6e0f2c9a47e3"
SENTINEL_ID_TOKEN = "SENTINEL-IDTOKEN-4d2e8a7c1b9f3065"
SENTINELS = (SENTINEL_ACCESS, SENTINEL_REFRESH, SENTINEL_ID_TOKEN)


def make_store(root: Path) -> TokenStore:
    """The real filesystem store under ``root/oauth``. ``root`` is a pytest tmp_path inside the workspace."""
    return TokenStore(oauth_dir(root))


def run(coro):
    return asyncio.run(coro)


def fixture_json(*parts: str) -> dict:
    return json.loads(FIXTURES.joinpath(*parts).read_text())


def chat_fixture(name: str, model: str) -> dict:
    body = fixture_json("golden", name)
    body["model"] = model
    return body


def sse(*events: tuple[str | None, dict]) -> bytes:
    out = []
    for name, data in events:
        out.append((f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n")
    return "".join(out).encode()


def sse_response(*events: tuple[str | None, dict], status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=sse(*events), headers={"content-type": "text/event-stream"})


def seed(store: TokenStore, provider: str, *, access: str = SENTINEL_ACCESS, refresh: str | None = SENTINEL_REFRESH,
         expires_at: float = NOW + 3600, id_token: str | None = None, meta: dict | None = None) -> Credentials:
    creds = Credentials(access_token=access, refresh_token=refresh, expires_at=expires_at, id_token=id_token,
                        meta=meta if meta is not None else {})
    store.save(provider, creds)
    return creds


class Sleeper:
    """Stands in for asyncio.sleep: records the requested waits and returns at once."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def make_app(tmp_path: Path, handler, *, clock=lambda: NOW, sleep=None, anthropic_retry=None):
    return app_module.build(tmp_path, transport=httpx.MockTransport(handler), clock=clock,
                            meter_file=tmp_path / "usage.jsonl", sleep=sleep or Sleeper(),
                            anthropic_retry=anthropic_retry)


def meter_lines(tmp_path: Path) -> list[dict]:
    path = tmp_path / "usage.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---- Anthropic stream fixtures -------------------------------------------------------------------

def anthropic_stream_text(text: str = "Paris.", *, usage=None) -> bytes:
    usage = usage or {"input_tokens": 25, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 10,
                      "output_tokens": 1}
    return sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_01", "model": "claude-opus-5-5",
                                                                "role": "assistant", "usage": usage}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("ping", {"type": "ping"}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}),
        ("message_stop", {"type": "message_stop"}),
    )


def anthropic_ok(text: str = "Paris.") -> httpx.Response:
    return httpx.Response(200, content=anthropic_stream_text(text), headers={"content-type": "text/event-stream"})


# ---- Responses stream fixtures -------------------------------------------------------------------

def responses_stream_text(text: str = "Paris.", *, usage="default") -> bytes:
    if usage == "default":
        usage = {"input_tokens": 120, "input_tokens_details": {"cached_tokens": 100}, "output_tokens": 9}
    response = {"id": "resp_1", "model": "gpt-6.1-sol", "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": text}]}]}
    if usage is not None:
        response["usage"] = usage
    return sse(
        ("response.created", {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "delta": text}),
        ("response.completed", {"type": "response.completed", "response": response}),
    )


def responses_ok(text: str = "Paris.") -> httpx.Response:
    return httpx.Response(200, content=responses_stream_text(text), headers={"content-type": "text/event-stream"})


# ---- Fake signed ID tokens -----------------------------------------------------------------------

def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def rsa_key() -> dict:
    return fixture_json("test_rsa.json")


def jwks() -> dict:
    key = rsa_key()
    to_b64 = lambda value: _b64(value.to_bytes((value.bit_length() + 7) // 8, "big"))
    return {"keys": [{"kty": "RSA", "kid": "test-key", "use": "sig", "alg": "RS256", "n": to_b64(key["n"]), "e": to_b64(key["e"])}]}


def sign_id_token(claims: dict, *, tamper: bool = False) -> str:
    key = rsa_key()
    header = _b64(json.dumps({"alg": "RS256", "kid": "test-key", "typ": "JWT"}).encode())
    payload = _b64(json.dumps(claims).encode())
    size = (key["n"].bit_length() + 7) // 8
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(f"{header}.{payload}".encode()).digest()
    em = b"\x00\x01" + b"\xff" * (size - len(digest_info) - 3) + b"\x00" + digest_info
    signature = pow(int.from_bytes(em, "big"), key["d"], key["n"]).to_bytes(size, "big")
    if tamper:
        payload = _b64(json.dumps({**claims, "sub": "someone-else"}).encode())
    return f"{header}.{payload}.{_b64(signature)}"
