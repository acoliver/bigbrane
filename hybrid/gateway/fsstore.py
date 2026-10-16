"""Private filesystem storage for the gateway's OAuth records.

Layout: one directory (``<state dir>/oauth``, mode 0700) holding ``<provider>.json`` (mode 0600) and a stable
``<provider>.lock`` (mode 0600) per provider. The rules follow the explicit filesystem OAuth in llxprt-code-rs
(PR 309, ``src/model_api/local_oauth/store.rs``):

* Owner, mode, file type and link count are checked on every read and every write. A wrong state is an error
  that names the path. Nothing is ever repaired, overwritten or moved aside automatically.
* Files are opened relative to a directory descriptor with ``O_NOFOLLOW``, so a symlink is refused. Reads are
  bounded to ``MAX_BYTES``.
* Writers hold an exclusive ``flock`` on the stable lock file. The lock is separate from the record file, so
  replacing the record does not split synchronization.
* A write goes to a private temp file in the same directory, is fsynced, renamed over the record, and then the
  directory is fsynced. Readers never see a partial record and need no lock.

There is no other storage and no migration from anywhere else.
"""
from __future__ import annotations

import fcntl
import os
import secrets
import stat
import threading
import time
from pathlib import Path
from typing import Protocol

from .errors import StoreError

MAX_BYTES = 65_536
LOCK_WAIT_S = 120.0
LOCK_POLL_S = 0.05
DIR_MODE = 0o700
FILE_MODE = 0o600

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK     # O_NONBLOCK: opening a FIFO must not hang


class Metadata(Protocol):
    """The subset of ``os.stat_result`` the checks read, so tests can describe a wrong owner or mode."""

    st_mode: int
    st_uid: int
    st_nlink: int


def check_directory(meta: Metadata, *, uid: int, label: str, private: bool) -> None:
    """The oauth directory must be private (0700). Its parent only has to be ours and not group/other writable."""
    if not stat.S_ISDIR(meta.st_mode):
        raise StoreError(f"{label} is not a directory")
    if meta.st_uid != uid:
        raise StoreError(f"{label} is owned by uid {meta.st_uid}, not the current user (uid {uid})")
    mode = stat.S_IMODE(meta.st_mode)
    if private and mode != DIR_MODE:
        raise StoreError(f"{label} has mode {mode:04o}, expected {DIR_MODE:04o}; it is not repaired automatically "
                         f"(run: chmod 700 {label})")
    if not private and mode & 0o022:
        raise StoreError(f"{label} is writable by group or others (mode {mode:04o}); refusing to use it")


def check_file(meta: Metadata, *, uid: int, label: str) -> None:
    """Record and lock files must be single-link regular files owned by us with mode 0600."""
    if not stat.S_ISREG(meta.st_mode):
        raise StoreError(f"{label} is not a regular file")
    if meta.st_uid != uid:
        raise StoreError(f"{label} is owned by uid {meta.st_uid}, not the current user (uid {uid})")
    mode = stat.S_IMODE(meta.st_mode)
    if mode != FILE_MODE:
        raise StoreError(f"{label} has mode {mode:04o}, expected {FILE_MODE:04o}; it is not repaired automatically "
                         f"(run: chmod 600 {label})")
    if meta.st_nlink != 1:
        raise StoreError(f"{label} has {meta.st_nlink} hard links, expected 1")


def _why(exc: OSError) -> str:
    return exc.strerror or str(exc)


def _sync(fd: int) -> None:
    """Flush to stable storage. macOS needs F_FULLFSYNC; plain fsync does not reach the disk there."""
    if hasattr(fcntl, "F_FULLFSYNC"):
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    else:
        os.fsync(fd)


class PrivateDir:
    def __init__(self, path: Path, *, uid: int | None = None) -> None:
        self.path = Path(path)
        self._uid = os.geteuid() if uid is None else uid

    def _check_file(self, fd: int, name: str) -> None:
        check_file(os.fstat(fd), uid=self._uid, label=str(self.path / name))

    def open_dir(self, *, create: bool) -> int | None:
        """Descriptor of the verified directory. None when it does not exist and ``create`` is false."""
        created = False
        if create:
            try:
                os.mkdir(self.path, DIR_MODE)
                created = True
            except FileExistsError:
                pass
            except OSError as exc:
                raise StoreError(f"cannot create {self.path}: {_why(exc)}") from exc
        try:
            fd = os.open(self.path, _DIR_FLAGS)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StoreError(f"cannot open {self.path}: {_why(exc)}; a symlink or non-directory is refused") from exc
        try:
            if created:
                os.fchmod(fd, DIR_MODE)          # mkdir's mode is reduced by the umask
            check_directory(os.fstat(fd), uid=self._uid, label=str(self.path), private=True)
            parent = self.path.parent
            check_directory(os.stat(parent), uid=self._uid, label=str(parent), private=False)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def read(self, name: str) -> bytes | None:
        """The record's bytes, or None when the directory or record does not exist. No lock needed."""
        dirfd = self.open_dir(create=False)
        if dirfd is None:
            return None
        try:
            return self._read_in(dirfd, name)
        finally:
            os.close(dirfd)

    def _read_in(self, dirfd: int, name: str) -> bytes | None:
        try:
            fd = os.open(name, os.O_RDONLY | _FILE_FLAGS, dir_fd=dirfd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StoreError(f"cannot open {self.path / name}: {_why(exc)}; symlinks are refused") from exc
        try:
            self._check_file(fd, name)
            chunks, total = [], 0
            while True:
                chunk = os.read(fd, MAX_BYTES + 1 - total)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_BYTES:
                    raise StoreError(f"{self.path / name} exceeds the {MAX_BYTES} byte limit")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def lock(self, lock_name: str) -> "FileLock":
        return FileLock(self, lock_name)


class FileLock:
    """Exclusive cross-process lock on ``lock_name`` plus the directory handle needed to use it.

    ``read``, ``write`` and ``remove`` only work while the lock is held. ``release`` is safe to call from any
    thread, including while another thread is still waiting in ``acquire`` (a cancelled waiter).
    """

    def __init__(self, private: PrivateDir, lock_name: str) -> None:
        self._private = private
        self._lock_name = lock_name
        self._state = threading.Lock()
        self._closed = False
        self._dirfd: int | None = None
        self._lockfd: int | None = None

    def acquire(self, timeout_s: float = LOCK_WAIT_S) -> None:
        dirfd = self._private.open_dir(create=True)
        assert dirfd is not None
        lockfd = -1
        try:
            lockfd = self._open_lock_file(dirfd)
            self._flock(lockfd, timeout_s)
            with self._state:
                if self._closed:
                    raise StoreError("lock released before it was acquired")
                self._dirfd, self._lockfd = dirfd, lockfd
        except BaseException:
            if lockfd >= 0:
                os.close(lockfd)
            os.close(dirfd)
            raise

    def _open_lock_file(self, dirfd: int) -> int:
        flags = os.O_RDWR | _FILE_FLAGS
        try:
            fd = os.open(self._lock_name, flags | os.O_CREAT | os.O_EXCL, FILE_MODE, dir_fd=dirfd)
            os.fchmod(fd, FILE_MODE)             # creation mode is reduced by the umask
        except FileExistsError:
            try:
                fd = os.open(self._lock_name, flags, dir_fd=dirfd)
            except OSError as exc:
                raise StoreError(f"cannot open {self._private.path / self._lock_name}: {_why(exc)}; "
                                 f"symlinks are refused") from exc
        except OSError as exc:
            raise StoreError(f"cannot create {self._private.path / self._lock_name}: {_why(exc)}") from exc
        try:
            self._private._check_file(fd, self._lock_name)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _flock(self, fd: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                pass
            if self._closed:
                raise StoreError("lock wait cancelled")
            if time.monotonic() >= deadline:
                raise StoreError(f"{self._private.path / self._lock_name} is held by another process; "
                                 f"waited {timeout_s:g}s")
            time.sleep(LOCK_POLL_S)

    def release(self) -> None:
        with self._state:
            self._closed = True
            for fd in (self._lockfd, self._dirfd):
                if fd is not None:
                    os.close(fd)             # closing the lock descriptor drops the flock
            self._lockfd = self._dirfd = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def _held_dirfd(self) -> int:
        if self._dirfd is None:
            raise StoreError("the store lock is not held")
        return self._dirfd

    def read(self, name: str) -> bytes | None:
        return self._private._read_in(self._held_dirfd(), name)

    def write(self, name: str, data: bytes) -> None:
        """Publish ``data`` as ``name``: temp file, fsync, rename, directory fsync."""
        dirfd = self._held_dirfd()
        if len(data) > MAX_BYTES:
            raise StoreError(f"{name} would exceed the {MAX_BYTES} byte limit")
        self._private._read_in(dirfd, name)          # authenticate the destination before replacing it
        self._remove_stale_staging(dirfd, name)
        staging = f".{name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        try:
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_FLAGS, FILE_MODE, dir_fd=dirfd)
        except OSError as exc:
            raise StoreError(f"cannot create a private staging file in {self._private.path}: {_why(exc)}") from exc
        try:
            try:
                os.fchmod(fd, FILE_MODE)
                self._private._check_file(fd, staging)
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                _sync(fd)
            finally:
                os.close(fd)
            os.rename(staging, name, src_dir_fd=dirfd, dst_dir_fd=dirfd)
        except OSError as exc:
            self._discard(dirfd, staging)
            raise StoreError(f"cannot publish {self._private.path / name}: {_why(exc)}") from exc
        except BaseException:
            self._discard(dirfd, staging)
            raise
        self._sync_dir(dirfd)

    def remove(self, name: str) -> None:
        dirfd = self._held_dirfd()
        if self._private._read_in(dirfd, name) is None:
            return
        try:
            os.unlink(name, dir_fd=dirfd)
        except OSError as exc:
            raise StoreError(f"cannot remove {self._private.path / name}: {_why(exc)}") from exc
        self._sync_dir(dirfd)

    def _sync_dir(self, dirfd: int) -> None:
        try:
            _sync(dirfd)
        except OSError as exc:
            raise StoreError(f"cannot sync {self._private.path}: {_why(exc)}") from exc

    @staticmethod
    def _discard(dirfd: int, staging: str) -> None:
        try:
            os.unlink(staging, dir_fd=dirfd)
        except FileNotFoundError:
            pass

    @staticmethod
    def _remove_stale_staging(dirfd: int, name: str) -> None:
        """Staging files left by a crashed writer. Safe to remove: this writer holds the record's lock."""
        for entry in os.listdir(dirfd):
            if entry.startswith(f".{name}.") and entry.endswith(".tmp"):
                os.unlink(entry, dir_fd=dirfd)
