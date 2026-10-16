"""One gateway per machine, enforced with an exclusive flock on a pidfile."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from .errors import AlreadyRunning


class PidLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.pread(fd, 32, 0).decode(errors="replace").strip() or "unknown"
            os.close(fd)
            raise AlreadyRunning(f"another gateway already holds {self.path} (pid {holder}); only one gateway "
                                 f"may run per machine, because it is the only refresher of the OAuth tokens") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd

    @staticmethod
    def holder(path: Path) -> str | None:
        """Pid text of the running gateway, or None when no process holds the lock. Does not take the lock."""
        if not path.exists():
            return None
        fd = os.open(path, os.O_RDONLY)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return os.pread(fd, 32, 0).decode(errors="replace").strip() or "unknown"
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        finally:
            os.close(fd)

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)      # closing the descriptor drops the flock
            self._fd = None

    def __enter__(self) -> "PidLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
