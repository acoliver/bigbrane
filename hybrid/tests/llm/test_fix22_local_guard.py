"""FIX-22 (decision ENV): the per-call local-model guard of the openai_compat backend, against a fake LM Studio (an
httpx mock transport, and for the two-process case a local HTTP server). No live LM Studio is called."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from hybrid.llm import BackendError, LLMClient, TransportError
from hybrid.llm.backends import BackendRequest, OpenAICompatBackend
from hybrid.llm.backends.lmstudio_guard import EVENT
from hybrid.llm.config import BackendConfig

from .fakes import ListSink

KEY = "qwen3.6-35b-a3b-mlx"
OTHER = "gemma-4-e4b-it"
MIN = 131072
ROOT_DIR = Path(__file__).resolve().parents[3]


class FakeLMStudio:
    """LM Studio's model list, load, unload and chat completions. ``instances`` maps a model key to its loaded
    instances [(id, context)]. A load adds an instance (``key``, else ``key:2``, ...), as LM Studio does."""

    def __init__(self, instances: dict[str, list[tuple[str, int]]], *, load_status: int = 200,
                 load_reply: dict | None = None):
        self.instances = {key: list(value) for key, value in instances.items()}
        self.load_status = load_status
        self.load_reply = load_reply
        self.calls: list[tuple[str, str, dict | None]] = []

    def listing(self) -> dict:
        return {"models": [{"type": "llm", "key": key, "loaded_instances": [
            {"id": iid, "config": {"context_length": ctx, "parallel": 4}} for iid, ctx in found]}
            for key, found in self.instances.items()]}

    def handle(self, method: str, path: str, body: dict | None) -> tuple[int, dict]:
        self.calls.append((method, path, body))
        if method == "GET" and path == "/api/v1/models":
            return 200, self.listing()
        if method == "POST" and path == "/api/v1/models/unload":
            for key, found in self.instances.items():
                self.instances[key] = [item for item in found if item[0] != body["instance_id"]]
            return 200, {"instance_id": body["instance_id"]}
        if method == "POST" and path == "/api/v1/models/load":
            if self.load_status != 200:
                return self.load_status, {"error": "Failed to load model: insufficient memory"}
            if self.load_reply is not None:
                return 200, self.load_reply
            found = self.instances.setdefault(body["model"], [])
            ids = {iid for iid, _ in found}
            new = body["model"] if body["model"] not in ids else next(
                f"{body['model']}:{n}" for n in range(2, 99) if f"{body['model']}:{n}" not in ids)
            found.append((new, body["context_length"]))
            return 200, {"type": "llm", "instance_id": new, "load_time_seconds": 1.0, "status": "loaded"}
        if method == "POST" and path == "/v1/chat/completions":
            served = [iid for found in self.instances.values() for iid, _ in found]
            if body["model"] not in served:
                return 400, {"error": "Model unloaded."}
            return 200, {"id": "c1", "model": body["model"], "choices": [
                {"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        return 404, {"error": f"no route {method} {path}"}

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else None
            status, payload = self.handle(request.method, request.url.path, body)
            return httpx.Response(status, json=payload)
        return httpx.MockTransport(handler)

    def actions(self) -> list[tuple[str, dict]]:
        return [(path.rsplit("/", 1)[-1], body) for method, path, body in self.calls
                if path in ("/api/v1/models/load", "/api/v1/models/unload")]

    def sent_models(self) -> list[str]:
        return [body["model"] for _, path, body in self.calls if path == "/v1/chat/completions"]


def config(min_loaded_context=MIN, base_url="http://lmstudio.test/v1"):
    return BackendConfig(name="lmstudio", kind="openai_compat", concurrency=1, timeout_s=30, base_url=base_url,
                         min_loaded_context=min_loaded_context)


def request(model=KEY):
    return BackendRequest(call_id="call-1", model=model, system="S", user="U", allow_web=False,
                          max_output_tokens=None, timeout_s=30, scratch_dir=Path("/unused"))


def backend(server: FakeLMStudio, tmp_path: Path, events: list | None = None, **kw) -> OpenAICompatBackend:
    sink = (lambda kind, data: events.append((kind, data))) if events is not None else None
    return OpenAICompatBackend(config(**kw), server.transport(), guard_lock_dir=tmp_path / "slots", events=sink)


@pytest.mark.asyncio
async def test_an_undersized_instance_is_unloaded_alone_and_the_model_loaded_once(tmp_path):
    server = FakeLMStudio({KEY: [(KEY, 8192)], OTHER: [(OTHER, 4096)]})
    events: list = []
    reply = await backend(server, tmp_path, events).complete(request())
    assert reply.text == "pong" and reply.detail["local_instance"] == KEY
    assert server.actions() == [("unload", {"instance_id": KEY}), ("load", {"model": KEY, "context_length": MIN})]
    assert server.instances[OTHER] == [(OTHER, 4096)]          # another model is never touched
    assert server.sent_models() == [KEY]
    assert [kind for kind, _ in events] == [EVENT]
    data = events[0][1]
    assert data["model"] == KEY and data["found"] == [{"instance_id": KEY, "context_length": 8192}]
    assert data["unloaded"] == [KEY] and data["loaded_context"] == MIN and data["instance_id"] == KEY
    assert data["required_context"] == MIN and data["call_id"] == "call-1" and data["seconds"] >= 0
    # The next call finds the adequate instance and loads nothing.
    await backend(server, tmp_path, events).complete(request())
    assert len(server.actions()) == 2 and len(events) == 1


@pytest.mark.asyncio
async def test_an_adequate_instance_means_no_load(tmp_path):
    server = FakeLMStudio({KEY: [(KEY, MIN)]})
    events: list = []
    reply = await backend(server, tmp_path, events).complete(request())
    assert reply.text == "pong" and server.actions() == [] and events == []
    assert server.sent_models() == [KEY]


@pytest.mark.asyncio
async def test_a_missing_model_is_loaded_once(tmp_path):
    server = FakeLMStudio({KEY: []})
    events: list = []
    await backend(server, tmp_path, events).complete(request())
    assert server.actions() == [("load", {"model": KEY, "context_length": MIN})]
    assert events[0][1]["found"] == [] and events[0][1]["unloaded"] == []


@pytest.mark.asyncio
async def test_a_suffixed_adequate_instance_gets_the_request(tmp_path):
    # The incident's state after the manual repair: the bare key at 8192 and a second instance `key:2` at 131072.
    server = FakeLMStudio({KEY: [(KEY, 8192), (f"{KEY}:2", MIN)]})
    events: list = []
    reply = await backend(server, tmp_path, events).complete(request())
    assert server.sent_models() == [f"{KEY}:2"] and reply.detail["local_instance"] == f"{KEY}:2"
    assert server.actions() == [] and events == []             # an adequate instance is never unloaded or replaced


@pytest.mark.asyncio
async def test_the_bare_key_is_preferred_among_adequate_instances(tmp_path):
    server = FakeLMStudio({KEY: [(f"{KEY}:2", MIN), (KEY, 262144)]})
    await backend(server, tmp_path).complete(request())
    assert server.sent_models() == [KEY] and server.actions() == []


@pytest.mark.asyncio
async def test_a_failed_load_is_a_clear_backend_error(tmp_path):
    server = FakeLMStudio({KEY: [(KEY, 8192)]}, load_status=400)
    with pytest.raises(BackendError) as caught:
        await backend(server, tmp_path).complete(request())
    message = str(caught.value)
    assert f"could not load {KEY} (required context {MIN})" in message
    assert "/api/v1/models/load" in message and "insufficient memory" in message
    assert server.sent_models() == []


@pytest.mark.asyncio
async def test_a_load_that_does_not_report_loaded_is_a_backend_error(tmp_path):
    server = FakeLMStudio({KEY: []}, load_reply={"status": "loading"})
    with pytest.raises(BackendError, match="did not report status 'loaded'"):
        await backend(server, tmp_path).complete(request())


@pytest.mark.asyncio
async def test_a_load_that_leaves_no_adequate_instance_is_a_backend_error(tmp_path):
    server = FakeLMStudio({KEY: []}, load_reply={"status": "loaded", "instance_id": KEY})
    with pytest.raises(BackendError, match=f"lists its instances as none"):
        await backend(server, tmp_path).complete(request())


@pytest.mark.asyncio
async def test_a_server_without_a_model_list_is_a_transport_error(tmp_path):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []}))
    guarded = OpenAICompatBackend(config(), transport, guard_lock_dir=tmp_path / "slots")
    with pytest.raises(TransportError, match="has no model list"):
        await guarded.complete(request())


@pytest.mark.asyncio
async def test_the_guard_is_off_without_min_loaded_context_and_for_auto(tmp_path):
    server = FakeLMStudio({KEY: [(KEY, 8192)]})
    await backend(server, tmp_path, min_loaded_context=None).complete(request())
    assert [path for _, path, _ in server.calls] == ["/v1/chat/completions"]
    assert backend(server, tmp_path, min_loaded_context=None).guard is None
    gateway = OpenAICompatBackend(BackendConfig(name="gw", kind="gateway", concurrency=1, timeout_s=5,
                                                base_url="http://gw/v1", min_loaded_context=MIN))
    assert gateway.guard is None


@pytest.mark.asyncio
async def test_llm_client_records_the_reload_in_the_run_ledger(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(f"""
routing = "main"

[budgets]
max_calls = 10
max_tokens = 100000

[backends.lmstudio]
kind = "openai_compat"
base_url = "http://lmstudio.test/v1"
concurrency = 1
timeout_s = 30
min_loaded_context = {MIN}

[routes.main]
statement_match = [ {{ backend = "lmstudio", model = "{KEY}" }} ]
""")
    server = FakeLMStudio({KEY: [(KEY, 8192)]})
    sink = ListSink()
    client = LLMClient(str(path), str(tmp_path / "run"), sink, http_transport=server.transport(),
                       lanes_dir=tmp_path / "slots")
    result = await client.complete("statement_match", "S", "U")
    assert result.text == "pong"
    [event] = sink.of(EVENT)
    assert event["actor"] == "llm:lmstudio" and event["data"]["loaded_context"] == MIN
    assert (tmp_path / "slots" / "lmstudio-guard-lmstudio.0.lock").exists()


# ----- two processes -------------------------------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_state: FakeLMStudio
    lock = threading.Lock()
    load_delay_s = 1.0

    def _serve(self, method: str) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        if self.path == "/api/v1/models/load":
            time.sleep(self.load_delay_s)          # a load takes a while; a second loader would overlap it
        with self.lock:
            status, payload = self.server_state.handle(method, self.path, body)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._serve("GET")

    def do_POST(self):
        self._serve("POST")

    def log_message(self, *args):
        pass


CHILD = """
import asyncio, json, sys
from pathlib import Path
from hybrid.llm.backends import BackendRequest, OpenAICompatBackend
from hybrid.llm.config import BackendConfig

base_url, lock_dir, key, min_context = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
events = []
backend = OpenAICompatBackend(BackendConfig(name="lmstudio", kind="openai_compat", concurrency=1, timeout_s=30,
                                            base_url=base_url, min_loaded_context=min_context),
                              guard_lock_dir=Path(lock_dir), events=lambda kind, data: events.append(kind))
request = BackendRequest(call_id="c", model=key, system="S", user="U", allow_web=False, max_output_tokens=None,
                         timeout_s=30, scratch_dir=Path("/unused"))
reply = asyncio.run(backend.complete(request))
print(json.dumps({"text": reply.text, "instance": reply.detail["local_instance"], "events": events}))
"""


def test_two_processes_load_the_model_once(tmp_path):
    state = FakeLMStudio({KEY: [(KEY, 8192)]})
    handler = type("H", (_Handler,), {"server_state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        args = [sys.executable, "-c", CHILD, base_url, str(tmp_path / "slots"), KEY, str(MIN)]
        children = [subprocess.Popen(args, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for _ in range(2)]
        results = [child.communicate(timeout=60) for child in children]
    finally:
        httpd.shutdown()
        httpd.server_close()
    for child, (out, err) in zip(children, results):
        assert child.returncode == 0, err
    replies = [json.loads(out) for out, _ in results]
    assert all(r["text"] == "pong" and r["instance"] == KEY for r in replies)
    assert sorted(len(r["events"]) for r in replies) == [0, 1]
    assert state.actions() == [("unload", {"instance_id": KEY}), ("load", {"model": KEY, "context_length": MIN})]
    assert state.instances[KEY] == [(KEY, MIN)]
