"""Resumable arXiv OAI-PMH harvest into the local metadata index.

Etiquette (https://info.arxiv.org/help/oa/index.html): follow resumption tokens, make one request at a
time with a pause between requests, wait out `Retry-After` on HTTP 503, and identify the client in the
User-Agent. The resumption token is stored with each page's records in one transaction, so a stop at
any moment, clean or not, resumes at the first page that was not stored.
"""
from __future__ import annotations

import email.utils
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import httpx

from hybrid.literature.arxiv_index import (
    STATUS_DEFERRED,
    STATUS_DONE,
    ArxivIndexWriter,
    ArxivRecord,
    PageCommit,
    SetState,
    dumps,
)
from hybrid.literature.oai import OaiError, OaiPage, parse_list_records

BASE_URL = "https://oaipmh.arxiv.org/oai"
METADATA_PREFIX = "arXiv"
USER_AGENT = "bigbrane-arxiv-oai-harvester/1.0 (local prior-art metadata index; OAI-PMH, one request at a time)"
DEFAULT_MIN_INTERVAL_S = 5.0
MAX_RETRY_AFTER_S = 3600.0
FIRST_BACKOFF_S = 30.0
MAX_BACKOFF_S = 600.0
MAX_ATTEMPTS_PER_REQUEST = 8
MANIFEST_EVERY_PAGES = 25

_LOG = logging.getLogger("hybrid.literature.harvest")


@dataclass(frozen=True)
class SetPlan:
    """A set to harvest. `categories` None keeps every record; otherwise a record is kept when it lists
    at least one of these categories (cross-lists count). Use it to narrow a wide set after harvest."""

    set_spec: str
    categories: frozenset[str] | None = None

    def keeps(self, record: ArxivRecord) -> bool:
        return self.categories is None or any(c in self.categories for c in record.categories)


# The target fields are whole archives for cs, stat, q-bio and nlin. For math, cond-mat and physics only
# a few categories matter, and arXiv publishes one OAI set per category, so those are harvested directly
# instead of streaming the wide sets (about 1.4 million records) to discard most of them.
DEFAULT_PLANS: tuple[SetPlan, ...] = (
    SetPlan("cs"),
    SetPlan("stat"),
    SetPlan("q-bio"),
    SetPlan("physics:nlin"),
    *(SetPlan(f"math:math:{code}") for code in ("ST", "PR", "OC", "DS", "IT", "CT", "AT")),
    SetPlan("physics:cond-mat:dis-nn"),
    SetPlan("physics:cond-mat:stat-mech"),
    SetPlan("physics:physics:bio-ph"),
    SetPlan("physics:physics:data-an"),
)


class HarvestError(RuntimeError):
    """One request failed for good: the server kept failing or answered something this harvester cannot
    interpret. The harvest loop defers the set; the saved state stays valid."""


class Stopped(Exception):
    """Raised inside the loop when a stop was requested during a wait."""


class Harvester:
    def __init__(
        self,
        writer: ArxivIndexWriter,
        http: httpx.Client,
        plans: Iterable[SetPlan] = DEFAULT_PLANS,
        *,
        stop: threading.Event | None = None,
        wait: Callable[[float], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        base_url: str = BASE_URL,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        from_date: str | None = None,
        max_pages: int | None = None,
        manifest_paths: Iterable[Path] = (),
        max_attempts: int = MAX_ATTEMPTS_PER_REQUEST,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._writer = writer
        self._http = http
        self._plans = tuple(plans)
        self._max_attempts = max_attempts
        self._stop = stop or threading.Event()
        # wait(seconds) sleeps and returns True when a stop was requested during the sleep.
        self._wait = wait or self._stop.wait
        self._clock = clock
        self._base_url = base_url
        self._min_interval_s = min_interval_s
        self._from_date = from_date
        self._max_pages = max_pages
        self._manifest_paths = tuple(manifest_paths)
        self._started_at = _utc_now()
        self._clock_start = clock()
        self._pages_this_run = 0
        self._seen_this_run = 0

    def run(self) -> str:
        """Returns "complete", "deferred" (sets kept failing after the retry pass; they resume on the next
        run), "stopped" (stop requested) or "page_limit" (max_pages reached). A set that keeps failing is
        deferred rather than ending the run. The state on disk is resumable in every case."""
        self._writer.ensure_sets([plan.set_spec for plan in self._plans], self._from_date)
        try:
            outcome = self._harvest_plans(self._plans)
            if outcome == "complete" and self.deferred_sets():
                _LOG.warning("retrying deferred sets once more: %s", ", ".join(self.deferred_sets()))
                outcome = self._harvest_plans(self._deferred_plans())
        except Stopped:
            outcome = "stopped"
        except BaseException:
            self._write_manifest("error")
            raise
        if outcome == "complete" and self.deferred_sets():
            outcome = "deferred"
        self._write_manifest(outcome)
        return outcome

    def deferred_sets(self) -> list[str]:
        return [plan.set_spec for plan in self._deferred_plans()]

    def _deferred_plans(self) -> list[SetPlan]:
        return [plan for plan in self._plans if self._writer.state(plan.set_spec).status == STATUS_DEFERRED]

    def _harvest_plans(self, plans: Iterable[SetPlan]) -> str:
        for plan in plans:
            if self._writer.state(plan.set_spec).status == STATUS_DONE:
                _LOG.info("set %s already done, skipping", plan.set_spec)
                continue
            try:
                outcome = self._harvest_set(plan)
            except HarvestError as exc:
                self._writer.mark_deferred(plan.set_spec)
                _LOG.warning("set %s deferred: %s; its resumption token is kept, moving on to the next set",
                             plan.set_spec, exc)
                continue
            if outcome != "complete":
                return outcome
        return "complete"

    def _harvest_set(self, plan: SetPlan) -> str:
        spec = plan.set_spec
        while True:
            state = self._writer.state(spec)
            if state.status == STATUS_DONE:
                _LOG.info("set %s done: %d seen, %d kept", spec, state.records_seen, state.records_kept)
                self._write_manifest("running")
                return "complete"
            if self._max_pages is not None and self._pages_this_run >= self._max_pages:
                return "page_limit"
            if self._stop.is_set():
                return "stopped"
            if self._pages_this_run:
                self._pause(self._min_interval_s)
            try:
                page = self._fetch_page(state)
            except OaiError as exc:
                if self._handle_oai_error(spec, exc):
                    continue
                raise
            self._store_page(plan, page)

    def _handle_oai_error(self, spec: str, exc: OaiError) -> bool:
        if exc.code == "noRecordsMatch":
            _LOG.info("set %s: no records match (%s)", spec, exc.message)
            self._writer.mark_done(spec, _utc_now())
            return True
        if exc.code == "badResumptionToken":
            # arXiv resumption tokens expire daily. Restarting is safe because the upserts are idempotent.
            _LOG.warning("set %s: resumption token rejected (%s); restarting the set from its first page", spec, exc)
            self._writer.restart_set(spec)
            return True
        return False

    def _fetch_page(self, state: SetState) -> OaiPage:
        if state.resumption_token:
            params = {"verb": "ListRecords", "resumptionToken": state.resumption_token}
        else:
            params = {"verb": "ListRecords", "metadataPrefix": METADATA_PREFIX, "set": state.set_spec}
            if state.from_date:
                params["from"] = state.from_date
        return parse_list_records(self._get(params))

    def _store_page(self, plan: SetPlan, page: OaiPage) -> None:
        kept = tuple(r for r in page.records if plan.keeps(r))
        commit = PageCommit(
            set_spec=plan.set_spec,
            records=kept,
            deleted_ids=page.deleted_ids,
            filtered=len(page.records) - len(kept),
            malformed=len(page.malformed_ids),
            resumption_token=page.resumption_token,
            response_date=page.response_date,
            complete_list_size=page.complete_list_size,
        )
        for identifier in page.malformed_ids:
            _LOG.warning("set %s: skipped malformed record %s", plan.set_spec, identifier)
        state = self._writer.commit_page(commit)
        self._pages_this_run += 1
        self._seen_this_run += commit.seen
        elapsed_min = max((self._clock() - self._clock_start) / 60.0, 1e-9)
        _LOG.info(
            "set %s page %d: %d records (%d kept, %d filtered, %d deleted, %d malformed); set total %d seen, %d kept;"
            " this run %d pages, %.0f records/min; response date %s%s",
            plan.set_spec, state.pages, commit.seen, len(kept), commit.filtered, len(commit.deleted_ids),
            commit.malformed, state.records_seen, state.records_kept, self._pages_this_run,
            self._seen_this_run / elapsed_min, page.response_date,
            f"; list size {state.complete_list_size}" if state.complete_list_size else "",
        )
        if state.status == STATUS_DONE or self._pages_this_run % MANIFEST_EVERY_PAGES == 0:
            self._write_manifest("running")

    def _get(self, params: dict[str, str]) -> str:
        backoff = FIRST_BACKOFF_S
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http.get(self._base_url, params=params)
            except httpx.TransportError as exc:
                delay, reason = backoff, f"transport error: {exc}"
            else:
                if response.status_code == 200:
                    return response.text
                if response.status_code in (429, 503):
                    retry_after = _retry_after_seconds(response)
                    delay = retry_after if retry_after is not None else backoff
                    reason = f"HTTP {response.status_code}, Retry-After {retry_after}"
                elif response.status_code >= 500:
                    delay, reason = backoff, f"HTTP {response.status_code}"
                else:
                    raise HarvestError(f"HTTP {response.status_code} from {self._base_url} for {params}")
            if attempt == self._max_attempts:
                raise HarvestError(f"gave up after {attempt} attempts: {reason}")
            delay = min(delay, MAX_RETRY_AFTER_S)
            _LOG.warning("%s; attempt %d of %d, waiting %.0f s", reason, attempt, self._max_attempts, delay)
            self._pause(delay)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        raise AssertionError("unreachable")

    def _pause(self, seconds: float) -> None:
        if self._wait(seconds):
            raise Stopped

    def _write_manifest(self, status: str) -> None:
        if not self._manifest_paths:
            return
        manifest = {
            "source": "arXiv OAI-PMH",
            "base_url": self._base_url,
            "metadata_prefix": METADATA_PREFIX,
            "user_agent": USER_AGENT,
            "harvest_started_at": self._started_at,
            "updated_at": _utc_now(),
            "run_status": status,
            "category_filters": {
                plan.set_spec: sorted(plan.categories) if plan.categories else "all" for plan in self._plans
            },
            **self._writer.summary(),
        }
        text = dumps(manifest) + "\n"
        for path in self._manifest_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)


def new_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(180.0, connect=30.0),
        follow_redirects=True,
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
