"""LLMClient: role routing, fallback, JSON mode, budgets, concurrency, transcripts and events."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import jsonmode
from .backends import (AnthropicMessagesBackend, Backend, BackendReply, BackendRequest,
                       LlxprtBackend, OpenAICompatBackend, run_process)
from .backends.llxprt import ProcessRunner
from .budget import Budget
from .config import ModelsConfig, RoleEntry, load_config
from .errors import ConfigError, LLMError, OutputExhausted, SchemaValidationError, TransportError
from .lanes import Lanes
from .types import EventSink, LLMResult


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_sha256(system: str, user: str) -> str:
    return sha256_text(json.dumps([system, user]))


def cut_off(reply: BackendReply) -> str | None:
    """The backend's report that it stopped at the output limit, or None."""
    detail = reply.detail or {}
    if detail.get("finish_reason") == "length":
        return "finish_reason=length"
    if detail.get("stop_reason") == "max_tokens":
        return "stop_reason=max_tokens"
    return None


# One step lower effort for the single retry after a reply whose whole output cap went to reasoning (OutputExhausted).
# In the 2026-10-13 smoke Claude Opus and Sonnet at effort medium each thought through the prover's 64000-token cap
# on one hand-set construction statement and returned no text; the card-run ended with a stage error (FIX-5).
LOWER_EFFORT = {"high": "medium", "medium": "low"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class _Call:
    """Mutable record of one complete() call, serialised as the transcript."""

    def __init__(self, call_id: str, role: str, system: str, user: str, json_schema: dict | None,
                 allow_web: bool, max_output_tokens: int | None):
        self.call_id = call_id
        self.role = role
        self.system = system
        self.user = user
        self.json_schema = json_schema
        self.allow_web = allow_web
        self.max_output_tokens = max_output_tokens
        self.started_at = _now()
        self.started = time.monotonic()
        self.attempts: list[dict] = []
        self.fallbacks: list[dict] = []
        self.usage = {"input": 0, "output": 0, "cached": 0}
        self.backend: str | None = None
        self.model: str | None = None
        self.text: str | None = None
        self.parsed: dict | None = None
        self.error: str | None = None
        self.first_prompt_sha256: str | None = None
        self.length_retried = False

    def add_usage(self, usage: dict) -> None:
        for key in self.usage:
            self.usage[key] += usage.get(key, 0)


class LLMClient:
    def __init__(self, config_path: str, run_dir: str, ledger: EventSink, *,
                 routing: str | None = None,
                 backends: dict[str, Backend] | None = None,
                 process_runner: ProcessRunner = run_process,
                 http_transport: httpx.AsyncBaseTransport | None = None,
                 lanes_dir: str | Path | None = None):
        """``lanes_dir``: hold host-wide request lanes there (hybrid/llm/lanes.py), so requests from every client on
        the host share each backend's concurrency and the gateway's ChatGPT pacing, and wait for them before their
        timeout starts. None keeps the per-client semaphores."""
        self.config: ModelsConfig = load_config(config_path, routing)
        self.lanes = Lanes(self.config, Path(lanes_dir)) if lanes_dir is not None else None
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        self.sink = ledger
        self.llm_dir = self.run_dir / "llm"
        self.calls_dir = self.llm_dir / "calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.budget = Budget.from_transcripts(self.config.budgets, self.calls_dir)
        self._backends: dict[str, Backend] = dict(backends or {})
        for name, cfg in self.config.backends.items():
            if name in self._backends:
                continue
            if cfg.kind == "llxprt":
                self._backends[name] = LlxprtBackend(cfg, self.llm_dir / "llxprt-home", process_runner)
            elif cfg.kind in ("openai_compat", "gateway"):
                # FIX-22: the local-model guard reports a reload to this run's ledger and locks under the lanes dir.
                self._backends[name] = OpenAICompatBackend(
                    cfg, http_transport, guard_lock_dir=Path(lanes_dir) if lanes_dir is not None else None,
                    events=self._guard_event)
            else:
                self._backends[name] = AnthropicMessagesBackend(cfg, http_transport)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    def _guard_event(self, kind: str, data: dict) -> None:
        self.sink.append(kind, f"llm:{data.get('backend')}", data)

    # ----- public API -------------------------------------------------------------------

    def complete_sync(self, role: str, system: str, user: str, **kwargs) -> LLMResult:
        return asyncio.run(self.complete(role, system, user, **kwargs))

    async def complete(self, role: str, system: str, user: str, *,
                       json_schema: dict | None = None, allow_web: bool = False,
                       max_output_tokens: int | None = None) -> LLMResult:
        if role not in self.config.roles:
            raise ConfigError(f"unknown role '{role}'; known roles: {sorted(self.config.roles)}")
        if json_schema is not None:
            jsonmode.check_schema(json_schema)
        call = _Call(f"call-{uuid.uuid4().hex[:12]}", role, system, user, json_schema, allow_web,
                     max_output_tokens)
        try:
            await self._route(call, self.config.roles[role])
        except Exception as exc:
            call.error = f"{type(exc).__name__}: {exc}"
            self._finish(call)
            if isinstance(exc, LLMError):
                exc.call_id = call.call_id   # the transcript of the failed call, for the caller's records
            raise
        transcript_path = self._finish(call)
        return LLMResult(call_id=call.call_id, role=role, backend=call.backend, model=call.model,
                         text=call.text, parsed=call.parsed, usage=dict(call.usage),
                         latency_s=self._latency(call), transcript_path=transcript_path)

    # ----- routing ----------------------------------------------------------------------

    async def _route(self, call: _Call, entries: tuple[RoleEntry, ...]) -> None:
        """The entries in order until one answers. When every entry failed because its reply spent the whole output
        cap without text, the error is OutputExhausted, so a caller can answer with a smaller request."""
        failures: list[TransportError] = []
        for entry in entries:
            try:
                await self._attempt_entry(call, entry)
                return
            except TransportError as exc:
                failures.append(exc)
                failure = {"backend": entry.backend, "model": entry.model, "error": str(exc)}
                call.fallbacks.append(failure)
                self.sink.append("llm_fallback", f"llm:{call.role}", {"call_id": call.call_id, **failure})
        summary = "; ".join(f"{f['backend']}:{f['model']}: {f['error']}" for f in call.fallbacks)
        exhausted = bool(failures) and all(isinstance(f, OutputExhausted) for f in failures)
        error = OutputExhausted if exhausted else TransportError
        raise error(f"all backend entries failed for role '{call.role}': {summary}")

    async def _attempt_entry(self, call: _Call, entry: RoleEntry) -> None:
        """One entry; after its first reply that spent the whole output cap without text, the same entry once more
        at one step lower effort (once per call, recorded as an ``llm_length_retry`` event and an attempt of kind
        ``length_retry``). A role at effort low, or without an effort, falls back at once."""
        try:
            await self._run_entry(call, entry)
        except OutputExhausted as exc:
            effort = self.config.settings_for(call.role).effort
            lower = LOWER_EFFORT.get(effort)
            if lower is None or call.length_retried:
                raise
            call.length_retried = True
            self.sink.append("llm_length_retry", f"llm:{call.role}", {
                "call_id": call.call_id, "backend": entry.backend, "model": entry.model, "error": str(exc),
                "usage": exc.usage, "effort_from": effort, "effort_to": lower})
            await self._run_entry(call, entry, effort=lower, kind="length_retry")

    async def _run_entry(self, call: _Call, entry: RoleEntry, *, effort: str | None = None,
                         kind: str = "request") -> None:
        user = call.user
        if call.json_schema is not None:
            user += jsonmode.schema_instruction(call.json_schema)
        reply = await self._request(call, entry, user, kind, effort)
        if call.json_schema is None:
            self._succeed(call, entry, reply, None)
            return
        parsed, error = self._parse(call, reply)
        if parsed is None:
            retry_user = user + jsonmode.retry_instruction(reply.text, error, truncated=cut_off(reply))
            reply = await self._request(call, entry, retry_user, "schema_retry", effort)
            parsed, error = self._parse(call, reply)
            if parsed is None:
                call.backend = self._backends[entry.backend].kind
                call.model = reply.model
                call.text = reply.text
                raise SchemaValidationError(f"output invalid after one retry: {error}")
        self._succeed(call, entry, reply, parsed)

    @staticmethod
    def _parse(call: _Call, reply: BackendReply) -> tuple[dict | None, str | None]:
        """Parse and validate a JSON-mode reply; a reply the backend cut off at the output limit says so first, so
        neither the transcript nor the retry blames the schema for a truncation."""
        parsed, error = jsonmode.parse_and_validate(reply.text, call.json_schema)
        if parsed is None and cut_off(reply):
            error = (f"the reply was cut off at the output limit ({cut_off(reply)}, "
                     f"{reply.usage.get('output', 0)} output tokens); {error}")
        call.attempts[-1]["validation_error"] = error
        return parsed, error

    def _succeed(self, call: _Call, entry: RoleEntry, reply: BackendReply, parsed: dict | None) -> None:
        call.backend = self._backends[entry.backend].kind
        call.model = reply.model
        call.text = reply.text
        call.parsed = parsed

    async def _request(self, call: _Call, entry: RoleEntry, user: str, kind: str,
                       effort: str | None = None) -> BackendReply:
        """One backend request. ``effort`` overrides the role's effort (the length retry's lower effort)."""
        backend = self._backends[entry.backend]
        cfg = self.config.backends[entry.backend]
        timeout_s = entry.timeout_s or cfg.timeout_s
        settings = self.config.settings_for(call.role)
        effort = effort or settings.effort
        max_output_tokens = call.max_output_tokens if call.max_output_tokens is not None else settings.max_output_tokens
        self.budget.reserve_call()
        attempt = {"index": len(call.attempts), "kind": kind, "backend": entry.backend,
                   "backend_kind": backend.kind, "requested_model": entry.model,
                   "effort": effort, "max_output_tokens": max_output_tokens,
                   "user_sent": user if user != call.user else None,
                   "prompt_sha256": prompt_sha256(call.system, user), "started_at": _now()}
        call.attempts.append(attempt)
        if call.first_prompt_sha256 is None:
            call.first_prompt_sha256 = attempt["prompt_sha256"]
        waiting = time.monotonic()
        async with self._slot(entry, cfg) as allowance_s:
            if self.lanes is not None:
                attempt["lane_wait_s"] = round(time.monotonic() - waiting, 3)
            if allowance_s:
                attempt["timeout_allowance_s"] = allowance_s
            request = BackendRequest(call_id=call.call_id, model=entry.model, system=call.system, user=user,
                                     allow_web=call.allow_web, max_output_tokens=max_output_tokens,
                                     timeout_s=timeout_s + allowance_s,
                                     scratch_dir=self.llm_dir / "scratch" / call.call_id, effort=effort,
                                     run_id=self.run_id)
            started = time.monotonic()
            try:
                reply = await backend.complete(request)
            except Exception as exc:
                attempt.update(outcome=type(exc).__name__, error=str(exc),
                               latency_s=round(time.monotonic() - started, 3))
                spent = getattr(exc, "usage", None) if isinstance(exc, OutputExhausted) else None
                if spent:
                    # The backend did the work and reported it: an empty reply at the cap still costs its tokens.
                    attempt["usage"] = spent
                    self.budget.add_usage(spent)
                    call.add_usage(spent)
                raise
        attempt.update(outcome="ok", model=reply.model, raw_output=reply.text, usage=reply.usage,
                       response_sha256=sha256_text(reply.text), detail=reply.detail,
                       latency_s=round(time.monotonic() - started, 3))
        self.budget.add_usage(reply.usage)
        call.add_usage(reply.usage)
        return reply

    @asynccontextmanager
    async def _slot(self, entry: RoleEntry, cfg):
        """Wait for the request's turn; yields the seconds its timeout gets on top for waits at the server. With lanes
        the turn is host-wide and includes the gateway's ChatGPT pacing; without, the per-client semaphore."""
        if self.lanes is not None:
            async with self.lanes.hold(entry.backend, entry.model) as allowance_s:
                yield allowance_s
            return
        async with self._semaphore(entry.backend, cfg.concurrency):
            yield 0.0

    def _semaphore(self, backend: str, concurrency: int) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore_loop is not loop:
            self._semaphores = {}
            self._semaphore_loop = loop
        if backend not in self._semaphores:
            self._semaphores[backend] = asyncio.Semaphore(concurrency)
        return self._semaphores[backend]

    # ----- records ----------------------------------------------------------------------

    @staticmethod
    def _latency(call: _Call) -> float:
        return round(time.monotonic() - call.started, 3)

    def _finish(self, call: _Call) -> str:
        """Write the transcript and append the llm_call event. Returns the transcript path."""
        response_sha = sha256_text(call.text) if call.text is not None and call.error is None else None
        record = {
            "call_id": call.call_id, "role": call.role, "routing": self.config.routing,
            "started_at": call.started_at,
            "finished_at": _now(), "latency_s": self._latency(call),
            "backend": call.backend, "model": call.model,
            "allow_web": call.allow_web, "max_output_tokens": call.max_output_tokens,
            "json_schema": call.json_schema, "system": call.system, "user": call.user,
            "text": call.text, "parsed": call.parsed, "usage": call.usage, "error": call.error,
            "prompt_sha256": call.first_prompt_sha256, "response_sha256": response_sha,
            "fallbacks": call.fallbacks, "attempts": call.attempts,
        }
        path = self.calls_dir / f"{call.call_id}.json"
        payload = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        path.write_bytes(payload)
        summary = [{key: attempt.get(key) for key in
                    ("index", "kind", "backend", "requested_model", "model", "effort", "max_output_tokens",
                     "outcome", "error",
                     "validation_error", "prompt_sha256", "response_sha256", "usage", "latency_s")}
                   for attempt in call.attempts]
        self.sink.append("llm_call", f"llm:{call.role}", {
            "call_id": call.call_id, "role": call.role, "backend": call.backend, "model": call.model,
            "prompt_sha256": call.first_prompt_sha256, "response_sha256": response_sha,
            "usage": call.usage, "latency_s": record["latency_s"], "error": call.error,
            "transcript_path": str(path), "transcript_relpath": str(path.relative_to(self.run_dir)),
            "transcript_sha256": hashlib.sha256(payload).hexdigest(), "attempts": summary,
        })
        return str(path)
