"""Wiring: builds the store, refresher, backends, meter and gateway from a state directory."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from .anthropic_auth import AnthropicAuth
from .anthropic_models import model_ids as anthropic_model_ids
from .anthropic_retry import RetryPolicy
from .backends import AnthropicBackend, ChatGPTBackend
from .chatgpt_auth import ChatGPTAuth
from .chatgpt_pacing import Pacer, PacingPolicy
from .chatgpt_plan import PlanAccessGuard
from .gateway import Gateway
from .meter import Meter
from .errors import GatewayError
from .paths import CHATGPT_CONCURRENCY_CHOICES, ChatGPTConfig, load_config, meter_path, oauth_dir, refresh_journal_path
from .redact import Redactor
from .refresh_schedule import RefreshScheduler
from .refresher import RefreshJournal, Refresher
from .store import TokenStore


@dataclass
class App:
    store: TokenStore
    redactor: Redactor
    refresher: Refresher
    gateway: Gateway
    meter: Meter
    pacing: PacingPolicy
    journal: RefreshJournal
    chatgpt_config: ChatGPTConfig
    schedulers: tuple[RefreshScheduler, ...] = ()


def build(root: Path, *, transport: httpx.AsyncBaseTransport | None = None,
          meter_file: Path | None = None, clock: Callable[[], float] = time.time,
          sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, anthropic_retry: RetryPolicy | None = None,
          chatgpt_concurrency: int | None = None, pacing: PacingPolicy | None = None) -> App:
    """``chatgpt_concurrency`` (1 or 2) overrides ``max_concurrency`` from ``config.toml``; ``pacing`` replaces the
    whole ChatGPT pacing policy (tests)."""
    if chatgpt_concurrency is not None and chatgpt_concurrency not in CHATGPT_CONCURRENCY_CHOICES:
        raise GatewayError("the ChatGPT concurrency must be 1 or 2")
    loaded = load_config(root)
    config = loaded.chatgpt
    policy = pacing or PacingPolicy.from_config(config, chatgpt_concurrency)
    store = TokenStore(oauth_dir(root))
    redactor = Redactor()
    journal = RefreshJournal(refresh_journal_path(root), clock)
    refresher = Refresher(store, {"anthropic": AnthropicAuth(transport, clock), "chatgpt": ChatGPTAuth(transport, clock)},
                          redactor=redactor, clock=clock, journal=journal)
    meter = Meter(meter_file or meter_path(root))
    gateway = Gateway({"anthropic": AnthropicBackend(refresher, transport, clock, retry=anthropic_retry, sleep=sleep),
                       "chatgpt": ChatGPTBackend(refresher, transport, clock, guard=PlanAccessGuard(store),
                                                 pacer=Pacer(policy, clock=clock, sleep=sleep))}, meter, clock,
                      models_timeout_s=loaded.models.upstream_timeout_s,
                      models_cache_ttl_s=loaded.models.cache_ttl_s,
                      static_models={"anthropic": anthropic_model_ids()})
    schedulers = (RefreshScheduler(refresher, store, "chatgpt", clock=clock, sleep=sleep),) \
        if config.proactive_refresh else ()
    return App(store, redactor, refresher, gateway, meter, policy, journal, config, schedulers)
