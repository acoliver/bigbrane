"""OpenAI-compatible chat completions backend (LM Studio, vLLM, and the `gateway` preset)."""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from ..config import BackendConfig
from ..errors import BackendError, ConfigError, OutputExhausted, TransportError
from ..gateway_check import gateway_token, serve_command, token_file
from .base import BackendReply, BackendRequest
from .http import request_json
from .lmstudio_guard import EventSink, LocalModelGuard

RUN_ID_HEADER = "x-bigbrane-run-id"


class OpenAICompatBackend:
    def __init__(self, config: BackendConfig, transport: httpx.AsyncBaseTransport | None = None, *,
                 guard_lock_dir: Path | None = None, events: EventSink | None = None):
        """``guard_lock_dir`` and ``events``: where the per-call local-model guard (lmstudio_guard.py, FIX-22) takes
        its host-wide lock and where it reports a reload. The guard is on for an ``openai_compat`` backend with
        ``min_loaded_context``; its lock directory defaults to hybrid.slots.SLOTS_DIR."""
        self.config = config
        self.kind = config.kind          # "openai_compat" or "gateway"
        self.transport = transport
        self._auto_model: str | None = None
        self.guard: LocalModelGuard | None = None
        if self.kind == "openai_compat" and config.min_loaded_context is not None:
            if guard_lock_dir is None:
                from hybrid.slots import SLOTS_DIR

                guard_lock_dir = SLOTS_DIR
            self.guard = LocalModelGuard(config, transport, guard_lock_dir, events=events)

    def _headers(self, run_id: str | None = None) -> dict:
        headers = {"content-type": "application/json"}
        if self.kind == "gateway":
            token = gateway_token(self.config)
            if token is None:
                raise ConfigError(f"backend {self.config.name}: no gateway token ({self.config.api_key_env} is unset "
                                  f"and {token_file()} does not exist); start the gateway: "
                                  f"{serve_command(self.config)}")
            headers["authorization"] = f"Bearer {token.value}"
            if run_id:
                headers[RUN_ID_HEADER] = run_id
        elif self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if not key:
                raise ConfigError(f"backend {self.config.name}: env var {self.config.api_key_env} is not set")
            headers["authorization"] = f"Bearer {key}"
        return headers

    async def resolve_model(self, model: str, timeout_s: float) -> str:
        if model != "auto":
            return model
        if self._auto_model is None:
            payload = await request_json(self.transport, "GET", f"{self.config.base_url}/models",
                                         headers=self._headers(), body=None, timeout_s=timeout_s)
            ids = [item["id"] for item in payload.get("data", [])]
            candidates = [name for name in ids if "embed" not in name.lower()]
            if not candidates:
                raise TransportError(f"{self.config.name}: no chat model is loaded (GET /models gave {ids})")
            self._auto_model = candidates[0]
        return self._auto_model

    async def complete(self, request: BackendRequest) -> BackendReply:
        model = await self.resolve_model(request.model, request.timeout_s)
        instance = None
        if self.guard is not None and request.model != "auto":
            # Sent as `model`, the instance id reaches the adequate instance even when an under-sized `key:2` is loaded.
            instance = await self.guard.instance_for(model, call_id=request.call_id)
            model = instance
        body = {"model": model, "stream": False,
                "messages": [{"role": "system", "content": request.system},
                             {"role": "user", "content": request.user}]}
        if request.max_output_tokens is not None:
            # The gateway's ChatGPT plan route drops this field (listed under `dropped` in its meter line).
            body["max_tokens"] = request.max_output_tokens
        if request.effort is not None:
            body["reasoning_effort"] = request.effort
        payload = await request_json(self.transport, "POST", f"{self.config.base_url}/chat/completions",
                                     headers=self._headers(request.run_id), body=body, timeout_s=request.timeout_s)
        choices = payload.get("choices")
        if not choices:
            raise BackendError(f"{self.config.name}: response has no choices: {str(payload)[:300]}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        raw_usage = payload.get("usage") or {}
        details = raw_usage.get("prompt_tokens_details") or {}
        usage = {"input": int(raw_usage.get("prompt_tokens", 0)),
                 "output": int(raw_usage.get("completion_tokens", 0)),
                 "cached": int(details.get("cached_tokens", 0))}
        if not isinstance(text, str) or not text.strip():
            finish = choices[0].get("finish_reason")
            message = f"{self.config.name}: empty message content (finish_reason={finish!r})"
            if finish == "length":
                # The cap went to reasoning before any text (FIX-5: Claude adaptive thinking on the gateway).
                raise OutputExhausted(message, usage=usage)
            raise TransportError(message)
        detail = {"finish_reason": choices[0].get("finish_reason"), "id": payload.get("id"),
                  "reasoning_content": message.get("reasoning_content"),
                  "sent": {key: body[key] for key in ("max_tokens", "reasoning_effort") if key in body}}
        if instance is not None:
            detail["local_instance"] = instance
        return BackendReply(text=text, model=payload.get("model", model), usage=usage, detail=detail)
