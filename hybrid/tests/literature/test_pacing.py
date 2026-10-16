"""Request pacing across processes: the production pacer keeps per-source lock and time files under the
workspace's var/slots, and processes that share the directory queue up on them. The in-memory pacer with
an injected clock keeps working for every other test."""
import json
import logging
import multiprocessing
import threading
import time

import httpx
import pytest

from hybrid.literature.client import PROCESS_PACER, Literature
from hybrid.literature.keys import ApiKey
from hybrid.literature.transport import MIN_INTERVAL_S, Pacer, Request, Transport
from hybrid.literature.cache import ResponseCache
from hybrid.slots import SLOTS_DIR
from hybrid.tests.literature.support import FakeClock, FakeWeb, fixture_text, ok, status

S2 = "api.semanticscholar.org"
S2_SECRET = "s2-secret-fedcba9876543210"


def _claim_slots(slots_dir, source, count, results):
    """Child process: take `count` slots of `source` and report each start time."""
    pacer = Pacer(time.time, slots_dir=slots_dir)
    for _ in range(count):
        results.put(pacer.wait_for_slot(source))


def _starts_from_processes(slots_dir, source, processes, per_process):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    children = [context.Process(target=_claim_slots, args=(str(slots_dir), source, per_process, results))
                for _ in range(processes)]
    for child in children:
        child.start()
    starts = sorted(results.get(timeout=60) for _ in range(processes * per_process))
    for child in children:
        child.join(timeout=60)
        assert child.exitcode == 0
    return starts


def test_s2_interval_is_below_the_key_limit_of_one_per_second():
    assert MIN_INTERVAL_S["s2"] == pytest.approx(1.1)
    assert MIN_INTERVAL_S["arxiv"] == pytest.approx(3.0)


def test_two_processes_never_start_s2_requests_closer_than_the_interval(tmp_path):
    starts = _starts_from_processes(tmp_path / "slots", "s2", processes=2, per_process=2)
    gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
    assert len(starts) == 4
    assert min(gaps) >= MIN_INTERVAL_S["s2"]


def test_arxiv_spacing_holds_across_processes(tmp_path):
    starts = _starts_from_processes(tmp_path / "slots", "arxiv", processes=2, per_process=1)
    assert starts[1] - starts[0] >= MIN_INTERVAL_S["arxiv"]


def test_sources_are_paced_independently(tmp_path):
    pacer = Pacer(time.time, slots_dir=tmp_path)
    started = time.time()
    pacer.wait_for_slot("arxiv")
    pacer.wait_for_slot("openalex")
    pacer.wait_for_slot("s2")
    assert time.time() - started < 1.0


def test_file_backed_pacer_works_with_a_fake_clock_and_creates_its_files(tmp_path):
    clock = FakeClock()
    pacer = Pacer(clock.clock, clock.sleep, slots_dir=tmp_path / "slots")
    assert pacer.wait_for_slot("s2") == 0.0
    assert pacer.wait_for_slot("s2") == pytest.approx(1.1)
    assert clock.sleeps == [pytest.approx(1.1)]
    assert (tmp_path / "slots" / "literature-s2.lock").is_file()
    assert float((tmp_path / "slots" / "literature-s2.last").read_text()) == pytest.approx(1.1)


def test_a_second_pacer_on_the_same_directory_sees_the_first_pacers_request(tmp_path):
    clock = FakeClock()
    first = Pacer(clock.clock, clock.sleep, slots_dir=tmp_path)
    second = Pacer(clock.clock, clock.sleep, slots_dir=tmp_path)
    first.wait_for_slot("arxiv")
    second.wait_for_slot("arxiv")
    assert clock.sleeps == [pytest.approx(3.0)]


def test_a_time_recorded_in_the_future_never_waits_longer_than_one_interval(tmp_path):
    (tmp_path / "literature-s2.last").write_text("5000.0")
    clock = FakeClock()
    clock.now = 100.0
    Pacer(clock.clock, clock.sleep, slots_dir=tmp_path).wait_for_slot("s2")
    assert clock.sleeps == [pytest.approx(1.1)]


def test_threads_sharing_a_file_backed_pacer_are_spaced(tmp_path):
    clock = FakeClock()
    pacer = Pacer(clock.clock, clock.sleep, slots_dir=tmp_path)
    starts: list[float] = []

    def worker():
        starts.append(pacer.wait_for_slot("s2"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(starts) == [pytest.approx(1.1 * i) for i in range(4)]


def test_the_production_pacer_is_file_backed_in_the_workspace_slots_directory():
    assert PROCESS_PACER._slots_dir == SLOTS_DIR
    assert SLOTS_DIR.parts[-2:] == ("var", "slots")


def test_a_429_retry_waits_for_its_slot_in_the_shared_pacer(tmp_path):
    clock = FakeClock()
    pacer = Pacer(clock.clock, clock.sleep, slots_dir=tmp_path / "slots")
    web = FakeWeb({S2: [status(429, {"Retry-After": "0"}), ok(fixture_text("s2_search.json"))]})
    attempts: list[float] = []
    inner = web.handler

    def handler(request):
        attempts.append(clock.now)
        return inner(request)

    transport = Transport(
        ResponseCache(tmp_path / "cache"),
        httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        clock=clock.clock,
        pacer=pacer,
    )
    fetched = transport.get("s2", Request("https://api.semanticscholar.org/graph/v1/paper/search", {"query": "alpha"}))
    assert not fetched.from_cache
    assert attempts == [pytest.approx(0.0), pytest.approx(1.1)]


def test_two_pacers_on_one_directory_space_requests_of_two_literature_instances(tmp_path):
    clock = FakeClock()
    web = FakeWeb({S2: ok(fixture_text("s2_search.json"))})
    shared = tmp_path / "slots"

    def literature(name):
        return Literature(
            cache_root=tmp_path / name,
            http=httpx.Client(transport=httpx.MockTransport(web.handler)),
            sleep=clock.sleep,
            clock=clock.clock,
            now=lambda: "2026-10-09T00:00:00+00:00",
            pacer=Pacer(clock.clock, clock.sleep, slots_dir=shared),
        )

    literature("a").search("alpha", sources=("s2",), limit=3)
    literature("b").search("beta", sources=("s2",), limit=3)
    assert clock.sleeps == [pytest.approx(1.1)]


def test_the_key_is_in_no_pacer_file_cache_envelope_url_or_log(tmp_path, caplog):
    clock = FakeClock()
    web = FakeWeb({S2: ok(fixture_text("s2_search.json"))})
    slots = tmp_path / "slots"
    literature = Literature(
        cache_root=tmp_path / "cache",
        http=httpx.Client(transport=httpx.MockTransport(web.handler)),
        sleep=clock.sleep,
        clock=clock.clock,
        now=lambda: "2026-10-09T00:00:00+00:00",
        api_keys={"s2": ApiKey("s2", S2_SECRET, "env test")},
        pacer=Pacer(clock.clock, clock.sleep, slots_dir=slots),
    )
    with caplog.at_level(logging.DEBUG):
        receipt = literature.search_with_receipt("alpha", sources=("s2",), limit=3)
        literature.search_with_receipt("beta", sources=("s2",), limit=3)
    assert receipt.papers
    assert any("literature request slot" in record.getMessage() for record in caplog.records)
    texts = [caplog.text, *[call.url.__str__() for call in web.calls]]
    texts += [path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file()]
    envelopes = [json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "cache").rglob("*.json")]
    assert envelopes
    assert all(S2_SECRET not in text for text in texts)
    assert all("x-api-key" not in json.dumps(envelope).lower() for envelope in envelopes)
