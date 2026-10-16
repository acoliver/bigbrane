import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from hybrid.gateway import __main__ as cli
from hybrid.gateway import fsstore
from hybrid.gateway.errors import AlreadyRunning, StoreConflict, StoreError
from hybrid.gateway.login import login_anthropic
from hybrid.gateway.paths import oauth_dir
from hybrid.gateway.singleton import PidLock
from hybrid.gateway.store import PROVIDERS, Credentials, TokenStore
from .support import NOW, anthropic_ok, make_app, make_store, responses_ok, run, seed

REPO = Path(__file__).resolve().parents[3]
USER = os.geteuid()


def test_one_record_file_per_provider_with_private_modes(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    seed(store, "chatgpt")
    oauth = oauth_dir(tmp_path)
    assert PROVIDERS == ("anthropic", "chatgpt")
    assert sorted(p.name for p in oauth.iterdir()) == ["anthropic.json", "anthropic.lock", "chatgpt.json", "chatgpt.lock"]
    assert stat.S_IMODE(oauth.stat().st_mode) == 0o700
    for path in oauth.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_uid == USER


def test_a_gateway_run_writes_only_inside_its_own_state_directory(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", expires_at=NOW - 5, refresh="r0")
    seed(store, "chatgpt", meta={"client_id": "oaiapp_x"}, expires_at=NOW - 5, refresh="r0")

    def handler(request):
        url = str(request.url)
        if "oauth/token" in url:
            return httpx.Response(200, json={"access_token": "new-access-token", "refresh_token": "new-refresh-token", "expires_in": 3600})
        if url.endswith("/v1/models"):
            return httpx.Response(200, json={"models": [{"slug": "gpt-6.1-sol", "visibility": "list"}]})
        return anthropic_ok() if "anthropic.com" in url else responses_ok()

    app = make_app(tmp_path, handler)
    chat = {"messages": [{"role": "user", "content": "hi"}]}
    run(app.gateway.chat({**chat, "model": "anthropic/claude-opus-5-5"}))
    run(app.gateway.chat({**chat, "model": "chatgpt/gpt-6.1-sol"}))
    run(app.gateway.models())
    from hybrid.gateway.login import status_lines
    status_lines(app.store)
    app.store.delete("anthropic")

    created = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert created == ["oauth", "oauth/anthropic.lock", "oauth/chatgpt.json", "oauth/chatgpt.lock", "refresh.jsonl",
                       "usage.jsonl"]


def test_unreadable_record_fails_with_the_login_command(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    path = oauth_dir(tmp_path) / "anthropic.json"
    path.write_text("{not json")
    with pytest.raises(StoreError, match="login anthropic"):
        store.load("anthropic")


# ---- owner, mode and type checks --------------------------------------------------------------------

def meta(mode, uid=USER, nlink=1):
    return SimpleNamespace(st_mode=mode, st_uid=uid, st_nlink=nlink)


def test_check_file_accepts_only_private_single_link_regular_files_of_the_current_user():
    fsstore.check_file(meta(stat.S_IFREG | 0o600), uid=USER, label="f")
    with pytest.raises(StoreError, match="owned by uid 0, not the current user"):
        fsstore.check_file(meta(stat.S_IFREG | 0o600, uid=0), uid=USER, label="f")
    for mode in (0o644, 0o640, 0o604, 0o660, 0o400, 0o700):
        with pytest.raises(StoreError, match=f"mode {mode:04o}, expected 0600"):
            fsstore.check_file(meta(stat.S_IFREG | mode), uid=USER, label="f")
    with pytest.raises(StoreError, match="not a regular file"):
        fsstore.check_file(meta(stat.S_IFIFO | 0o600), uid=USER, label="f")
    with pytest.raises(StoreError, match="2 hard links"):
        fsstore.check_file(meta(stat.S_IFREG | 0o600, nlink=2), uid=USER, label="f")


def test_check_directory_requires_a_private_directory_and_a_non_writable_parent():
    fsstore.check_directory(meta(stat.S_IFDIR | 0o700), uid=USER, label="d", private=True)
    with pytest.raises(StoreError, match="owned by uid 0"):
        fsstore.check_directory(meta(stat.S_IFDIR | 0o700, uid=0), uid=USER, label="d", private=True)
    with pytest.raises(StoreError, match="mode 0755, expected 0700"):
        fsstore.check_directory(meta(stat.S_IFDIR | 0o755), uid=USER, label="d", private=True)
    with pytest.raises(StoreError, match="not a directory"):
        fsstore.check_directory(meta(stat.S_IFREG | 0o700), uid=USER, label="d", private=True)
    fsstore.check_directory(meta(stat.S_IFDIR | 0o755), uid=USER, label="parent", private=False)
    with pytest.raises(StoreError, match="writable by group or others"):
        fsstore.check_directory(meta(stat.S_IFDIR | 0o775), uid=USER, label="parent", private=False)


def test_record_with_a_wrong_mode_is_refused_on_read_and_not_repaired(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    path = oauth_dir(tmp_path) / "anthropic.json"
    path.chmod(0o644)
    with pytest.raises(StoreError, match="mode 0644, expected 0600"):
        store.load("anthropic")
    with pytest.raises(StoreError, match="mode 0644"):
        store.save("anthropic", Credentials("a", "r", NOW + 1))
    assert stat.S_IMODE(path.stat().st_mode) == 0o644          # never chmod-repaired


def test_oauth_directory_with_a_wrong_mode_is_refused(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    oauth_dir(tmp_path).chmod(0o755)
    with pytest.raises(StoreError, match="mode 0755, expected 0700"):
        store.load("anthropic")
    with pytest.raises(StoreError, match="mode 0755, expected 0700"):
        store.save("anthropic", Credentials("a", "r", NOW + 1))


def test_files_owned_by_someone_else_are_refused(tmp_path):
    seed(make_store(tmp_path), "anthropic")
    foreign = TokenStore(oauth_dir(tmp_path), uid=USER + 1)         # pretends the current user is a different uid
    with pytest.raises(StoreError, match="is owned by uid"):
        foreign.load("anthropic")
    with pytest.raises(StoreError, match="is owned by uid"):
        foreign.save("anthropic", Credentials("a", "r", NOW + 1))


def test_symlinked_record_is_refused_and_its_target_is_left_alone(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    record = oauth_dir(tmp_path) / "anthropic.json"
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    target.chmod(0o600)
    record.unlink()
    record.symlink_to(target)
    with pytest.raises(StoreError, match="symlinks are refused"):
        store.load("anthropic")
    with pytest.raises(StoreError, match="symlinks are refused"):
        store.save("anthropic", Credentials("a", "r", NOW + 1))
    assert target.read_text() == "{}"


def test_symlinked_lock_file_is_refused(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    lock = oauth_dir(tmp_path) / "anthropic.lock"
    other = tmp_path / "other.lock"
    other.write_text("")
    other.chmod(0o600)
    lock.unlink()
    lock.symlink_to(other)
    with pytest.raises(StoreError, match="symlinks are refused"):
        store.save("anthropic", Credentials("a", "r", NOW + 1))


def test_symlinked_oauth_directory_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    (root / "oauth").symlink_to(real)
    store = TokenStore(root / "oauth")
    with pytest.raises(StoreError, match="symlink or non-directory is refused"):
        store.load("anthropic")
    with pytest.raises(StoreError, match="symlink or non-directory is refused"):
        store.save("anthropic", Credentials("a", "r", NOW + 1))
    assert list(real.iterdir()) == []


def test_fifo_and_hard_linked_records_are_refused_without_hanging(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    record = oauth_dir(tmp_path) / "anthropic.json"
    link = tmp_path / "hardlink"
    os.link(record, link)
    with pytest.raises(StoreError, match="2 hard links"):
        store.load("anthropic")
    link.unlink()
    record.unlink()
    os.mkfifo(record, 0o600)
    with pytest.raises(StoreError, match="not a regular file"):
        store.load("anthropic")


def test_oversized_record_is_refused_by_the_bounded_read(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    record = oauth_dir(tmp_path) / "anthropic.json"
    record.write_bytes(b" " * (fsstore.MAX_BYTES + 1))
    with pytest.raises(StoreError, match="byte limit"):
        store.load("anthropic")
    with pytest.raises(StoreError, match="byte limit"):
        store.save("anthropic", Credentials("a", "r", NOW, meta={"pad": "x" * (fsstore.MAX_BYTES + 1)}))


def test_loading_before_any_login_creates_nothing(tmp_path):
    assert make_store(tmp_path).load("chatgpt") is None
    assert not oauth_dir(tmp_path).exists()


def test_stale_staging_files_from_a_crashed_writer_are_removed_on_the_next_write(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic")
    stale = oauth_dir(tmp_path) / ".anthropic.json.4242.deadbeef.tmp"
    stale.write_text("old")
    store.save("anthropic", Credentials("a", "r", NOW + 1))
    assert not stale.exists()


# ---- cross-process lock -----------------------------------------------------------------------------

LOCK_PROBE = """
import sys
from pathlib import Path
from hybrid.gateway.errors import StoreError
from hybrid.gateway.store import TokenStore
try:
    TokenStore(Path(sys.argv[1])).lock("anthropic").acquire(0.3)
except StoreError as exc:
    print("blocked:", exc)
    raise SystemExit(3)
print("acquired")
"""


def probe_lock(tmp_path):
    return subprocess.run([sys.executable, "-c", LOCK_PROBE, str(oauth_dir(tmp_path))], cwd=REPO,
                          capture_output=True, text=True, timeout=60)


def test_provider_lock_excludes_another_process_until_released(tmp_path):
    store = make_store(tmp_path)
    held = store.lock("anthropic")
    held.acquire()
    try:
        blocked = probe_lock(tmp_path)
        assert blocked.returncode == 3 and "held by another process" in blocked.stdout
        other_provider = store.lock("chatgpt")        # locks are per provider
        other_provider.acquire(1)
        other_provider.release()
    finally:
        held.release()
    free = probe_lock(tmp_path)
    assert free.returncode == 0 and "acquired" in free.stdout


def test_writes_need_the_lock(tmp_path):
    held = make_store(tmp_path).lock("anthropic")
    with pytest.raises(StoreError, match="not held"):
        held.write(Credentials("a", "r", NOW + 1))


# ---- login snapshot ----------------------------------------------------------------------------------

def test_login_publishes_when_the_record_is_unchanged(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", refresh="r0")
    started = store.snapshot("anthropic")
    store.publish_login("anthropic", Credentials("new", "r-login", NOW + 100), started)
    assert store.load("anthropic").refresh_token == "r-login"


def test_login_publishes_the_first_record_when_none_existed(tmp_path):
    store = make_store(tmp_path)
    started = store.snapshot("anthropic")
    assert started.raw is None
    store.publish_login("anthropic", Credentials("new", "r-login", NOW + 100), started)
    assert store.load("anthropic").access_token == "new"


def test_login_snapshot_conflict_keeps_the_newer_record(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", refresh="r0")
    started = store.snapshot("anthropic")
    store.save("anthropic", Credentials("refreshed", "r1", NOW + 500))          # a gateway rotated the token meanwhile
    with pytest.raises(StoreConflict, match="changed while this login was in progress"):
        store.publish_login("anthropic", Credentials("login", "r-login", NOW + 100), started)
    assert store.load("anthropic").refresh_token == "r1"


def test_login_snapshot_conflict_when_another_login_created_the_first_record(tmp_path):
    store = make_store(tmp_path)
    started = store.snapshot("chatgpt")
    seed(store, "chatgpt", refresh="from-other-login")
    with pytest.raises(StoreConflict):
        store.publish_login("chatgpt", Credentials("login", "r-login", NOW + 100), started)
    assert store.load("chatgpt").refresh_token == "from-other-login"


def test_login_command_fails_with_the_conflict_when_the_record_changes_during_the_browser_step(tmp_path):
    store = make_store(tmp_path)
    seed(store, "anthropic", refresh="r0")
    opened = []

    def handler(request):
        return httpx.Response(200, json={"access_token": "A", "refresh_token": "R", "expires_in": 100})

    def ask(prompt):
        store.save("anthropic", Credentials("rotated", "r-rotated", NOW + 900))      # refresh lands while the user is in the browser
        from urllib.parse import parse_qs, urlparse
        return f"code#{parse_qs(urlparse(opened[0]).query)['state'][0]}"

    with pytest.raises(StoreConflict):
        run(login_anthropic(store, ask=ask, say=lambda s: None, open_browser=opened.append,
                            transport=httpx.MockTransport(handler), clock=lambda: NOW))
    assert store.load("anthropic").refresh_token == "r-rotated"


def test_credentials_repr_hides_token_values():
    creds = Credentials(access_token="SECRET-ACCESS-VALUE", refresh_token="SECRET-REFRESH-VALUE", expires_at=1.0,
                        id_token="SECRET-ID-VALUE", meta={"client_id": "c"})
    assert "SECRET" not in repr(creds) and "SECRET" not in str(creds)


def test_second_gateway_fails_at_startup_on_the_pidfile_lock(tmp_path):
    first = PidLock(tmp_path / "gateway.pid")
    first.acquire()
    try:
        with pytest.raises(AlreadyRunning, match="only one gateway"):
            PidLock(tmp_path / "gateway.pid").acquire()
        assert PidLock.holder(tmp_path / "gateway.pid") == str(os.getpid())
    finally:
        first.release()
    assert PidLock.holder(tmp_path / "gateway.pid") is None
    again = PidLock(tmp_path / "gateway.pid")
    again.acquire()           # the lock is free after the first holder exits
    again.release()


def test_serve_command_in_a_second_process_exits_at_startup_while_the_first_holds_the_lock(tmp_path):
    lock = PidLock(tmp_path / "gateway.pid")
    lock.acquire()
    try:
        env = {**os.environ, "BIGBRANE_GATEWAY_HOME": str(tmp_path)}
        result = subprocess.run([sys.executable, "-m", "hybrid.gateway", "serve", "--port", "0"], cwd=REPO, env=env,
                                capture_output=True, text=True, timeout=60)
    finally:
        lock.release()
    assert result.returncode == 2
    assert "another gateway already holds" in result.stderr
    assert "listening" not in result.stdout


def test_serve_command_in_process_returns_error_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path))
    with PidLock(tmp_path / "gateway.pid"):
        assert cli.main(["serve", "--port", "0"]) == 2
    assert "another gateway" in capsys.readouterr().err


def test_a_cancelled_lock_waiter_does_not_keep_the_lock(tmp_path):
    import threading
    store = make_store(tmp_path)
    holder = store.lock("anthropic")
    holder.acquire()
    waiter = store.lock("anthropic")
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(_try_acquire(waiter)))
    thread.start()
    waiter.release()                      # the waiting task was cancelled
    thread.join(10)
    holder.release()
    assert outcome == ["StoreError"]
    again = store.lock("anthropic")
    again.acquire(1)                      # nothing leaked
    again.release()


def _try_acquire(lock):
    try:
        lock.acquire(30)
    except StoreError:
        return "StoreError"
    return "acquired"
