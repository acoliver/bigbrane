"""ChatGPT plan usage: the attribution text OpenAI's UI guidelines ask for, and the local stop after a refusal.

OpenAI's errors page says a 403 at admission is a policy or permission refusal and that the app must not repeat the
same request or loop. ``PlanAccessGuard`` turns that into a rule for this process: after one admission 403 every
later ChatGPT call fails here, without contacting OpenAI, until a new login record replaces the one that was refused.
"""
from __future__ import annotations

import asyncio
import logging

from .errors import UpstreamError
from .store import Snapshot, TokenStore

log = logging.getLogger("bigbrane.gateway")

USAGE_URL = "https://chatgpt.com/settings/usage"
MANAGE_USAGE_LEAD = f"Manage usage: {USAGE_URL}. "
LOGIN_NOTICE = f"You're using your ChatGPT plan. Manage usage: {USAGE_URL}"
STATUS_NOTICE = f"Using ChatGPT plan (Manage usage: {USAGE_URL})"
ACCESS_DENIED_MESSAGE = f"ChatGPT plan access denied; check {USAGE_URL}, then run login chatgpt"
ACCESS_DENIED_CODE = "plan_access_denied"


def lead_with_manage_usage(message: str) -> str:
    """Put the Manage usage link first, as the UI guidelines ask for usage-limit and permission errors."""
    return MANAGE_USAGE_LEAD + message


class PlanAccessGuard:
    """Remembers an admission refusal for the life of the process.

    The refused record is remembered as the exact stored bytes. The block lifts only when the store holds a record
    that differs from it, which is what a new login (or any other rewrite of the record) produces. A missing
    record keeps the block; the caller then still sees the same message, which names the login command.
    """

    def __init__(self, store: TokenStore, provider: str = "chatgpt") -> None:
        self._store = store
        self._provider = provider
        self._refused: Snapshot | None = None

    @property
    def blocked(self) -> bool:
        return self._refused is not None

    async def check(self) -> None:
        """Raise the local refusal if OpenAI refused this login earlier and the record has not changed since."""
        if self._refused is None:
            return
        current = await asyncio.to_thread(self._store.snapshot, self._provider)
        if current.raw is not None and current.raw != self._refused.raw:
            log.info("%s: a new login record appeared; calls to OpenAI resume", self._provider)
            self._refused = None
            return
        raise UpstreamError(ACCESS_DENIED_MESSAGE, status=403, code=ACCESS_DENIED_CODE)

    async def block(self, error: UpstreamError) -> None:
        """Called with the 403 OpenAI answered at admission."""
        self._refused = await asyncio.to_thread(self._store.snapshot, self._provider)
        diagnostics = error.detail.diagnostics() if error.detail else {}
        log.error("%s: OpenAI refused admission (HTTP 403, code=%s%s); no further calls will be made until a new "
                  "login record appears. Check %s", self._provider, error.code,
                  "".join(f" {name}={value}" for name, value in diagnostics.items()), USAGE_URL)
