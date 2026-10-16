"""Anthropic Messages API backend."""
from __future__ import annotations

import os

import httpx

from ..config import BackendConfig
from ..errors import BackendError, ConfigError, OutputExhausted, TransportError
from .base import BackendReply, BackendRequest
from .http import request_json


class AnthropicMessagesBackend:
    kind = "anthropic_messages"

    def __init__(self, config: BackendConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.transport = transport

    def _headers(self) -> dict:
        if not self.config.api_key_env:
            raise ConfigError(f"backend {self.config.name}: api_key_env is required")
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ConfigError(f"backend {self.config.name}: env var {self.config.api_key_env} is not set")
        return {"x-api-key": key, "anthropic-version": self.config.anthropic_version,
                "content-type": "application/json"}

    async def complete(self, request: BackendRequest) -> BackendReply:
        body = {"model": request.model,
                "max_tokens": request.max_output_tokens or self.config.default_max_output_tokens,
                "system": request.system,
                "messages": [{"role": "user", "content": request.user}]}
        payload = await request_json(self.transport, "POST", f"{self.config.base_url}/v1/messages",
                                     headers=self._headers(), body=body, timeout_s=request.timeout_s)
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise BackendError(f"{self.config.name}: response has no content blocks: {str(payload)[:300]}")
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        raw = payload.get("usage") or {}
        cached = int(raw.get("cache_read_input_tokens", 0))
        usage = {"input": int(raw.get("input_tokens", 0)) + cached + int(raw.get("cache_creation_input_tokens", 0)),
                 "output": int(raw.get("output_tokens", 0)), "cached": cached}
        if not text.strip():
            message = f"{self.config.name}: empty text (stop_reason={payload.get('stop_reason')!r})"
            if payload.get("stop_reason") == "max_tokens":
                raise OutputExhausted(message, usage=usage)
            raise TransportError(message)
        detail = {"stop_reason": payload.get("stop_reason"), "id": payload.get("id")}
        return BackendReply(text=text, model=payload.get("model", request.model), usage=usage, detail=detail)
