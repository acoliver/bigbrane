"""Offline differential measurement against the pinned pre-change transport."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from uuid import uuid4

import httpx

from hybrid.literature.cache import ResponseCache, request_key
from hybrid.literature.transport import Transport
from hybrid.tests.literature.support import FakeClock

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = "eeaba07"
RUN_ID = uuid4().hex
source = subprocess.check_output(
    ["git", "-C", str(ROOT), "show", f"{BASE}:hybrid/literature/transport.py"], text=True
)
baseline = types.ModuleType("baseline_transport_provenance_verification")
sys.modules[baseline.__name__] = baseline
exec(compile(source, "pinned-pre-change-transport.py", "exec"), baseline.__dict__)
URL = "https://export.arxiv.org/api/query"
EPOCH = datetime(2026, 10, 18, tzinfo=timezone.utc)


class EarlyClock(FakeClock):
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds - 1 if seconds > 1 else seconds


def run(cls, name, responses, *, header=None, url=URL, queries=("alpha",), early=False):
    clock = EarlyClock() if early else FakeClock()
    requests, records, results = [], [], []
    pending = iter(responses)

    def handler(request):
        code = next(pending)
        requests.append({"request_id": request_key("arxiv", str(request.url), {}), "started_s": clock.now,
                         "response": code,
                         "headers_hash": hashlib.sha256(json.dumps(dict(request.headers), sort_keys=True).encode()).hexdigest()})
        if code == "timeout":
            clock.now += 2
            raise httpx.ReadTimeout("synthetic private details", request=request)
        if code == 301:
            return httpx.Response(301, headers={"Location": URL + "?q=alpha"})
        return httpx.Response(code, text="synthetic answer", headers={"Retry-After": header} if header else {})

    cache = ResponseCache(OUT / "comparison-cache" / RUN_ID / name / cls.__module__)
    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    tr = cls(cache, http, clock=clock.clock, sleep=clock.sleep,
             now=lambda: (EPOCH + timedelta(seconds=clock.now)).isoformat())
    cached_key = request_key("arxiv", URL, {"q": "cached"})
    cache.put("arxiv", cached_key, {"source": "arxiv", "url": URL, "params": {"q": "cached"}, "status": 200,
                                  "fetched_at": EPOCH.isoformat(), "body": "synthetic cached answer"})
    for query in queries:
        request = baseline.Request(url, {"q": query})
        try:
            fetched = tr.get("arxiv", request)
        except RuntimeError as exc:
            reason = exc.reason
            outcome = "cooldown" if reason.startswith("skipped:") else "transport_error" if reason.startswith("transport error:") else reason
            results.append({"outcome": outcome})
            record = getattr(exc, "provenance", None)
        else:
            results.append({"body": fetched.body, "fetched_at": fetched.fetched_at, "from_cache": fetched.from_cache})
            record = getattr(fetched, "provenance", None)
        if record is not None:
            records.append(record)
    envelopes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.root.rglob("*.json")}
    return {"requests": requests, "sleeps_s": clock.sleeps, "results": results,
            "cache_envelope_hashes": envelopes, "final_clock_s": clock.now}, records


cases = [
    ("success-cache-pacer", [200, 200], {"queries": ("alpha", "alpha", "beta")}),
    ("retry-after-delta", [429, 200], {"header": "7"}),
    ("retry-after-cap", [429, 200], {"header": "1000"}),
    ("retry-after-short-pacer", [429, 200], {"header": "1"}),
    ("retry-after-date", [429, 200], {"header": "Sun, 18 Oct 2026 00:00:40 GMT"}),
    ("retry-after-invalid", [429, 200], {"header": "not a date"}),
    ("exhaustion-skip-cache", [429, 429, 429, 429], {"queries": ("alpha", "beta", "cached")}),
    ("timeout-retry", ["timeout", 200], {}),
    ("redirect", [301, 200], {"url": "http://export.arxiv.org/api/query"}),
    ("terminal-404", [404], {}),
    ("server-error", [503, 200], {}),
    ("early-sleep", [429, 200], {"header": "7", "early": True}),
]
comparison, examples = {}, {}
for name, responses, kwargs in cases:
    before, _ = run(baseline.Transport, name, responses, **kwargs)
    after, records = run(Transport, name, responses, **kwargs)
    assert before == after, f"Behavior changed in {name}"
    comparison[name] = {"unchanged": True, **after}
    if name in ("retry-after-delta", "retry-after-date", "exhaustion-skip-cache", "redirect", "timeout-retry"):
        examples[name] = records
(OUT / "behavior-comparison.json").write_text(json.dumps(
    {"baseline_commit": BASE, "baseline_transport_sha256": hashlib.sha256(source.encode()).hexdigest(),
     "case_count": len(cases), "all_unchanged": True, "cases": comparison}, indent=2) + "\n")
(OUT / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
print(f"{len(cases)} cases: request identities, headers, start times, sleeps, results and cache bytes unchanged")
