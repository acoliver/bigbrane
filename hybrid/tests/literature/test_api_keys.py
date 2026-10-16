"""API keys for OpenAlex and Semantic Scholar: sources and precedence, file permissions, request headers,
Semantic Scholar pacing, secrecy, the `keys` command, and the unchanged keyless behavior."""
import json
import logging
import threading

import httpx
import pytest

from hybrid.literature import __main__ as cli
from hybrid.literature.client import Literature, default_literature
from hybrid.literature.key_report import report_keys
from hybrid.literature.keys import ApiKey, ApiKeyError, load_api_key, load_api_keys
from hybrid.literature.transport import Pacer
from hybrid.tests.literature.support import FakeClock, FakeWeb, fixture_text, make_literature, ok, status

ARXIV = "export.arxiv.org"
OPENALEX = "api.openalex.org"
S2 = "api.semanticscholar.org"
OA_ENV = "BIGBRANE_OPENALEX_API_KEY"
S2_ENV = "BIGBRANE_S2_API_KEY"
OA_SECRET = "oa-secret-0123456789abcdef"
S2_SECRET = "s2-secret-fedcba9876543210"


def write_key(home, name, text, mode=0o600):
    directory = home / ".bigbrane" / "keys"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def both_keys():
    return {
        "openalex": ApiKey("openalex", OA_SECRET, "env test"),
        "s2": ApiKey("s2", S2_SECRET, "env test"),
    }


def keyed_literature(tmp_path, web, clock=None, **kwargs):
    clock = clock or FakeClock()
    return Literature(
        cache_root=tmp_path,
        http=httpx.Client(transport=httpx.MockTransport(web.handler)),
        sleep=clock.sleep,
        clock=clock.clock,
        now=lambda: "2026-10-09T00:00:00+00:00",
        api_keys=both_keys(),
        **kwargs,
    )


def keyed_web():
    return FakeWeb({OPENALEX: ok(fixture_text("openalex_search.json")), S2: ok(fixture_text("s2_search.json"))})


# Where the key comes from

def test_env_beats_file(tmp_path):
    write_key(tmp_path, "openalex", "file-key-aaaa\n")
    key = load_api_key("openalex", environ={OA_ENV: "env-key-bbbb"}, home=tmp_path)
    assert key.value == "env-key-bbbb"
    assert key.origin == f"env {OA_ENV}"


def test_file_is_used_without_env_and_whitespace_is_stripped(tmp_path):
    path = write_key(tmp_path, "semantic_scholar", "  file-key-cccc \n\n")
    key = load_api_key("s2", environ={}, home=tmp_path)
    assert key.value == "file-key-cccc"
    assert key.origin == f"file {path}"
    assert key.last4 == "cccc"


def test_blank_env_falls_through_to_the_file(tmp_path):
    write_key(tmp_path, "openalex", "file-key-dddd")
    assert load_api_key("openalex", environ={OA_ENV: "   "}, home=tmp_path).value == "file-key-dddd"


def test_no_key_anywhere_gives_none(tmp_path):
    assert load_api_key("openalex", environ={}, home=tmp_path) is None
    assert load_api_keys(environ={}, home=tmp_path) == {}


def test_each_source_reads_its_own_variable_and_file(tmp_path):
    write_key(tmp_path, "semantic_scholar", "s2-from-file")
    keys = load_api_keys(environ={OA_ENV: "oa-from-env"}, home=tmp_path)
    assert keys["openalex"].value == "oa-from-env"
    assert keys["s2"].value == "s2-from-file"


# File permissions

@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o660, 0o666])
def test_key_file_readable_by_group_or_others_is_refused(tmp_path, mode):
    path = write_key(tmp_path, "openalex", OA_SECRET, mode=mode)
    with pytest.raises(ApiKeyError) as raised:
        load_api_key("openalex", environ={}, home=tmp_path)
    message = str(raised.value)
    assert f"chmod 600 {path}" in message
    assert OA_SECRET not in message


def test_loose_key_file_is_refused_even_when_env_wins(tmp_path):
    write_key(tmp_path, "openalex", OA_SECRET, mode=0o644)
    with pytest.raises(ApiKeyError, match="chmod 600"):
        load_api_key("openalex", environ={OA_ENV: "env-key-eeee"}, home=tmp_path)


def test_owner_only_modes_are_accepted(tmp_path):
    write_key(tmp_path, "openalex", "ok-key-ffff", mode=0o400)
    assert load_api_key("openalex", environ={}, home=tmp_path).value == "ok-key-ffff"


def test_empty_key_file_is_refused(tmp_path):
    write_key(tmp_path, "openalex", " \n")
    with pytest.raises(ApiKeyError, match="empty"):
        load_api_key("openalex", environ={}, home=tmp_path)


def test_key_with_inner_whitespace_is_refused_without_echoing_it(tmp_path):
    with pytest.raises(ApiKeyError) as raised:
        load_api_key("s2", environ={S2_ENV: "abc def"}, home=tmp_path)
    assert "abc" not in str(raised.value)


def test_default_literature_refuses_to_start_on_a_loose_key_file(tmp_path, monkeypatch):
    write_key(tmp_path, "semantic_scholar", S2_SECRET, mode=0o644)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(S2_ENV, raising=False)
    with pytest.raises(ApiKeyError, match="chmod 600"):
        default_literature()


def test_default_literature_starts_with_private_key_files(tmp_path, monkeypatch):
    write_key(tmp_path, "semantic_scholar", S2_SECRET)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(S2_ENV, raising=False)
    monkeypatch.delenv(OA_ENV, raising=False)
    assert isinstance(default_literature(), Literature)


# Request headers

def test_openalex_gets_a_bearer_header_and_no_key_in_the_url(tmp_path):
    web = keyed_web()
    keyed_literature(tmp_path, web).search("alpha", sources=("openalex",), limit=3)
    (call,) = web.calls_to(OPENALEX)
    assert call.headers["Authorization"] == f"Bearer {OA_SECRET}"
    assert OA_SECRET not in str(call.url)
    assert "api_key" not in call.url.params


def test_s2_gets_the_x_api_key_header_and_no_key_in_the_url(tmp_path):
    web = keyed_web()
    keyed_literature(tmp_path, web).search("alpha", sources=("s2",), limit=3)
    (call,) = web.calls_to(S2)
    assert call.headers["x-api-key"] == S2_SECRET
    assert "Authorization" not in call.headers
    assert S2_SECRET not in str(call.url)


def test_each_key_goes_only_to_its_own_source(tmp_path):
    web = FakeWeb({ARXIV: ok(fixture_text("arxiv_search.xml")), OPENALEX: ok(fixture_text("openalex_search.json")),
                   S2: ok(fixture_text("s2_search.json"))})
    keyed_literature(tmp_path, web).search("alpha", sources=("arxiv", "openalex", "s2"), limit=3)
    (arxiv,) = web.calls_to(ARXIV)
    (openalex,) = web.calls_to(OPENALEX)
    (s2,) = web.calls_to(S2)
    assert "Authorization" not in arxiv.headers and "x-api-key" not in arxiv.headers
    assert "x-api-key" not in openalex.headers
    assert "Authorization" not in s2.headers


def test_mailto_is_still_sent_only_when_a_contact_is_set(tmp_path):
    web = keyed_web()
    keyed_literature(tmp_path / "with", web, contact_email="ops@example.org").search(
        "alpha", sources=("openalex",), limit=3)
    keyed_literature(tmp_path / "without", web).search("beta", sources=("openalex",), limit=3)
    with_email, without_email = web.calls_to(OPENALEX)
    assert with_email.url.params["mailto"] == "ops@example.org"
    assert "mailto" not in without_email.url.params


def test_cache_entries_are_shared_between_keyed_and_keyless_runs(tmp_path):
    web = keyed_web()
    keyed_literature(tmp_path, web).search("alpha", sources=("openalex",), limit=3)
    make_literature(tmp_path, web).search("alpha", sources=("openalex",), limit=3)
    assert len(web.calls_to(OPENALEX)) == 1


def test_keyed_related_lookups_also_carry_the_header(tmp_path):
    def s2_responder(request):
        return httpx.Response(200, json={"data": []})

    web = FakeWeb({OPENALEX: ok("{}"), S2: s2_responder})
    literature = keyed_literature(tmp_path, web, max_retries=0)
    from hybrid.literature.paper import Paper

    paper = Paper("s2", "p1", "Title", (), None, None, "u", None, None, "t")
    literature.related(paper)
    assert web.calls_to(S2)
    assert all(call.headers["x-api-key"] == S2_SECRET for call in web.calls_to(S2))
    assert all(call.headers["Authorization"] == f"Bearer {OA_SECRET}" for call in web.calls_to(OPENALEX))


# Semantic Scholar pacing

def test_keyed_s2_requests_are_spaced_below_the_key_limit(tmp_path):
    web = keyed_web()
    clock = FakeClock()
    literature = keyed_literature(tmp_path, web, clock=clock)
    for query in ("alpha", "beta", "gamma"):
        literature.search(query, sources=("s2",), limit=3)
    assert clock.sleeps == [pytest.approx(1.1), pytest.approx(1.1)]


def test_literature_instances_sharing_a_pacer_share_the_s2_budget(tmp_path):
    web = keyed_web()
    clock = FakeClock()
    pacer = Pacer(clock.clock, clock.sleep)
    first = keyed_literature(tmp_path / "a", web, clock=clock, pacer=pacer)
    second = keyed_literature(tmp_path / "b", web, clock=clock, pacer=pacer)
    first.search("alpha", sources=("s2",), limit=3)
    second.search("beta", sources=("s2",), limit=3)
    assert clock.sleeps == [pytest.approx(1.1)]


def test_parallel_threads_are_spaced_by_the_pacer():
    clock = FakeClock()
    pacer = Pacer(clock.clock, clock.sleep)
    starts: list[float] = []

    def worker():
        pacer.wait_for_slot("s2")
        starts.append(clock.now)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(starts) == [pytest.approx(1.1 * i) for i in range(5)]


def test_exhausted_429_with_a_key_still_cools_the_source_down(tmp_path):
    web = FakeWeb({S2: status(429)})
    literature = keyed_literature(tmp_path, web, max_retries=1)
    first = literature.search_with_receipt("alpha", sources=("s2",), limit=3)
    assert first.errors == {"s2": "HTTP 429 after 2 attempts"}
    second = literature.search_with_receipt("beta", sources=("s2",), limit=3)
    assert "cooling down" in second.errors["s2"]
    assert len(web.calls_to(S2)) == 2


# The key never leaves the process boundary

def test_key_never_appears_in_logs_receipts_errors_cache_or_events(tmp_path, caplog):
    def leaky(request):
        raise httpx.ConnectError(f"refused with headers {dict(request.headers)}", request=request)

    web = FakeWeb({OPENALEX: leaky, S2: status(401)})
    literature = keyed_literature(tmp_path, web, max_retries=1)
    with caplog.at_level(logging.DEBUG):
        receipt = literature.search_with_receipt("alpha", sources=("openalex", "s2"), limit=3)
        web_ok = FakeWeb({OPENALEX: ok(fixture_text("openalex_search.json")), S2: ok(fixture_text("s2_search.json"))})
        ok_receipt = keyed_literature(tmp_path / "ok", web_ok).search_with_receipt(
            "beta", sources=("openalex", "s2"), limit=3)
    assert set(receipt.errors) == {"openalex", "s2"}
    assert "***" in receipt.errors["openalex"]
    haystack = [
        caplog.text,
        json.dumps(receipt.event_data()),
        json.dumps(ok_receipt.event_data()),
        repr(receipt),
        *[(path.read_text(encoding="utf-8")) for path in tmp_path.rglob("*.json")],
    ]
    for secret in (OA_SECRET, S2_SECRET):
        for text in haystack:
            assert secret not in text


def test_api_key_text_forms_hide_the_value():
    key = ApiKey("openalex", OA_SECRET, "env test")
    assert OA_SECRET not in repr(key)
    assert OA_SECRET not in str(key)
    assert key.last4 == OA_SECRET[-4:]
    assert key.redact(f"x {OA_SECRET} y") == "x *** y"


def test_a_failed_key_check_error_names_the_file_not_the_key(tmp_path):
    write_key(tmp_path, "openalex", OA_SECRET, mode=0o644)
    with pytest.raises(ApiKeyError) as raised:
        load_api_keys(environ={}, home=tmp_path)
    assert OA_SECRET not in str(raised.value)


# Keyless behavior is unchanged

def test_without_keys_no_auth_header_is_sent(tmp_path):
    web = keyed_web()
    make_literature(tmp_path, web).search("alpha", sources=("openalex", "s2"), limit=3)
    for call in web.calls:
        assert "Authorization" not in call.headers
        assert "x-api-key" not in call.headers


def test_without_keys_s2_pacing_is_unchanged(tmp_path):
    web = keyed_web()
    clock = FakeClock()
    literature = make_literature(tmp_path, web, clock=clock)
    literature.search("alpha", sources=("s2",), limit=3)
    literature.search("beta", sources=("s2",), limit=3)
    assert clock.sleeps == [pytest.approx(1.1)]


# The keys command

def test_keys_command_reports_presence_and_last_four_only(tmp_path, monkeypatch, capsys):
    path = write_key(tmp_path, "semantic_scholar", S2_SECRET)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(OA_ENV, OA_SECRET)
    monkeypatch.delenv(S2_ENV, raising=False)
    assert cli.main(["keys"]) == 0
    out = capsys.readouterr().out
    assert f"openalex: key found from env {OA_ENV}, ends in {OA_SECRET[-4:]}" in out
    assert f"s2: key found from file {path}, ends in {S2_SECRET[-4:]}" in out
    assert "live test" not in out
    for secret in (OA_SECRET, S2_SECRET):
        assert secret not in out


def test_keys_command_says_where_it_looked_when_there_is_no_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(OA_ENV, raising=False)
    monkeypatch.delenv(S2_ENV, raising=False)
    assert cli.main(["keys"]) == 0
    out = capsys.readouterr().out
    assert f"openalex: no key (looked at env {OA_ENV}, then {tmp_path}/.bigbrane/keys/openalex)" in out
    assert f"s2: no key (looked at env {S2_ENV}, then {tmp_path}/.bigbrane/keys/semantic_scholar)" in out


def test_keys_command_exits_2_on_a_loose_key_file(tmp_path, monkeypatch, capsys):
    write_key(tmp_path, "openalex", OA_SECRET, mode=0o644)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(OA_ENV, raising=False)
    assert cli.main(["keys", "--live"]) == 2
    captured = capsys.readouterr()
    assert "chmod 600" in captured.err
    assert OA_SECRET not in captured.out + captured.err


def test_live_check_makes_one_authenticated_request_per_found_key(tmp_path):
    seen = []

    def responder(request):
        seen.append((request.url.host, dict(request.headers)))
        return httpx.Response(200 if request.url.host == OPENALEX else 401, text="{}")

    http = httpx.Client(transport=httpx.MockTransport(responder))
    lines = report_keys(both_keys(), live=True, http=http, home=tmp_path)
    by_source = {line.source: line for line in lines}
    assert by_source["openalex"].text.endswith("live test: HTTP 200") and by_source["openalex"].ok
    assert by_source["s2"].text.endswith("live test: HTTP 401") and not by_source["s2"].ok
    assert [host for host, _ in seen] == [OPENALEX, S2]
    assert seen[0][1]["authorization"] == f"Bearer {OA_SECRET}"
    assert seen[1][1]["x-api-key"] == S2_SECRET
    for line in lines:
        assert OA_SECRET not in line.text and S2_SECRET not in line.text


def test_live_check_skips_sources_without_a_key(tmp_path):
    calls = []
    http = httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200)))
    lines = report_keys({}, live=True, http=http, home=tmp_path)
    assert calls == []
    assert all("no key" in line.text and line.ok for line in lines)


def test_without_live_no_request_is_made(tmp_path):
    def forbidden(request):
        raise AssertionError("no network call expected without --live")

    http = httpx.Client(transport=httpx.MockTransport(forbidden))
    lines = report_keys(both_keys(), live=False, http=http, home=tmp_path)
    assert all("live test" not in line.text for line in lines)


def test_live_check_reports_transport_errors_without_the_key(tmp_path):
    def leaky(request):
        raise httpx.ConnectError(f"boom {request.headers['x-api-key']}", request=request)

    http = httpx.Client(transport=httpx.MockTransport(leaky))
    (line,) = [l for l in report_keys({"s2": ApiKey("s2", S2_SECRET, "env test")}, live=True, http=http,
                                       home=tmp_path) if l.source == "s2"]
    assert "transport error: boom ***" in line.text
    assert not line.ok
