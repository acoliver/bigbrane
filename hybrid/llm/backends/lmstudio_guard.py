"""Per-call local-model guard for LM Studio (FIX-22, decision ENV).

Before each request to an ``openai_compat`` backend that sets ``min_loaded_context`` (models.toml), the guard lists
the routed model's loaded instances (``GET <server>/api/v1/models``, hybrid/localmodels.py) and returns the id of an
instance loaded with at least that context. The backend sends that id as the request's ``model``, so the request
reaches that instance whatever LM Studio does with a bare model key when a second instance (``key:2``) is loaded.

When no instance fits, the guard takes a host-wide file lock (``lmstudio-guard-<backend>`` under var/slots, so
parallel card-runs in other processes cannot load twice) and lists again: another process may have loaded it
meanwhile. Still none: it unloads only this model's under-sized instances (``POST /api/v1/models/unload``), loads
the model once at ``min_loaded_context`` (``POST /api/v1/models/load``), requires ``status: "loaded"`` and an
adequate instance in a fresh list, and reports a ``local_model_reloaded`` event through ``events`` (LLMClient appends
it to the calling run's ledger). Other models, adequate instances and LM Studio's settings are never touched.

The server's state is external input, so this is a check at that boundary. A failed listing is a TransportError (the
server did not answer: the role may fall back); a failed unload or load is a BackendError naming the model, the
context and the server's answer.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import httpx

from hybrid import localmodels
from hybrid.slots import FileSemaphore

from ..config import BackendConfig
from ..errors import BackendError, LLMError, TransportError
from .http import request_json

EVENT = "local_model_reloaded"
LIST_TIMEOUT_S = 30.0
LOCK_POLL_S = 0.25

EventSink = Callable[[str, dict], None]


class LocalModelGuard:
    def __init__(self, config: BackendConfig, transport: httpx.AsyncBaseTransport | None, lock_dir: Path, *,
                 events: EventSink | None = None, clock: Callable[[], float] = time.monotonic):
        if config.min_loaded_context is None:
            raise ValueError(f"backend {config.name}: the local-model guard needs min_loaded_context")
        self.config = config
        self.transport = transport
        self.server = localmodels.server_root(config.base_url)
        self.min_context = config.min_loaded_context
        self.lock = FileSemaphore(Path(lock_dir), f"lmstudio-guard-{config.name}", 1, poll_s=LOCK_POLL_S)
        self.events = events
        self.clock = clock

    async def instance_for(self, model: str, *, call_id: str | None = None) -> str:
        """The id of a loaded instance of ``model`` with at least min_loaded_context tokens, loading one if needed."""
        chosen = localmodels.adequate_instance(await self._instances(model), model, self.min_context)
        if chosen is not None:
            return chosen.id
        async with self.lock.hold_async():
            started = self.clock()
            found = await self._instances(model)
            chosen = localmodels.adequate_instance(found, model, self.min_context)
            if chosen is not None:
                return chosen.id            # another card-run loaded it while this one waited for the lock
            unloaded = []
            for instance in localmodels.undersized_instances(found, self.min_context):
                await self._post("unload", {"instance_id": instance.id}, model)
                unloaded.append(instance.id)
            reply = await self._post("load", {"model": model, "context_length": self.min_context}, model)
            if reply.get("status") != "loaded":
                raise BackendError(f"{self.config.name}: loading {model} at context {self.min_context} did not "
                                   f"report status 'loaded': {str(reply)[:300]}")
            after = await self._instances(model)
            chosen = localmodels.adequate_instance(after, model, self.min_context)
            if chosen is None:
                raise BackendError(f"{self.config.name}: loaded {model} at context {self.min_context}, but "
                                   f"{self.server} lists its instances as {self._describe(after)}")
            if self.events is not None:
                self.events(EVENT, {
                    "backend": self.config.name, "model": model, "call_id": call_id,
                    "found": [{"instance_id": i.id, "context_length": i.context} for i in found],
                    "unloaded": unloaded, "instance_id": chosen.id, "loaded_context": chosen.context,
                    "required_context": self.min_context, "seconds": round(self.clock() - started, 3)})
            return chosen.id

    async def _instances(self, model: str) -> list[localmodels.Instance]:
        payload = await request_json(self.transport, "GET", f"{self.server}/api/v1/models",
                                     headers={"content-type": "application/json"}, body=None,
                                     timeout_s=LIST_TIMEOUT_S)
        try:
            return localmodels.loaded_instances(payload, model)
        except localmodels.LocalModelUnavailable as exc:
            raise TransportError(f"{self.config.name}: {exc}") from exc

    async def _post(self, action: str, body: dict, model: str) -> dict:
        url = f"{self.server}/api/v1/models/{action}"
        try:
            return await request_json(self.transport, "POST", url, headers={"content-type": "application/json"},
                                      body=body, timeout_s=self.config.timeout_s)
        except LLMError as exc:
            raise BackendError(f"{self.config.name}: could not {action} {model} (required context "
                               f"{self.min_context}) with POST {url} {body}: {exc}") from exc

    @staticmethod
    def _describe(instances: list[localmodels.Instance]) -> str:
        return ", ".join(f"{i.id} (context {i.context})" for i in instances) or "none"
