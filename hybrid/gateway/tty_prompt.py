"""A hidden-input prompt that can be called off.

``getpass`` blocks until Enter, and a thread stuck in it leaves terminal echo switched off. The login races a paste
against a browser callback, so the loser has to be able to stop reading and restore the terminal.
"""
from __future__ import annotations

import getpass
import os
import select
import termios
import threading

POLL_S = 0.2


def read_hidden(prompt: str, cancel: threading.Event) -> str | None:
    """One line typed without echo on the controlling terminal, or None once ``cancel`` is set.

    With no controlling terminal this falls back to ``getpass`` (which then reads standard input and cannot be
    called off).
    """
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return getpass.getpass(prompt)
    try:
        original = termios.tcgetattr(fd)
        hidden = termios.tcgetattr(fd)
        hidden[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, hidden)
        try:
            os.write(fd, prompt.encode())
            buffered = b""
            while not cancel.is_set():
                if not select.select([fd], [], [], POLL_S)[0]:
                    continue
                chunk = os.read(fd, 4096)
                if not chunk:
                    return buffered.decode().strip()
                buffered += chunk
                if b"\n" in buffered or b"\r" in buffered:
                    return buffered.decode().splitlines()[0].strip()
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
            os.write(fd, b"\n")
    finally:
        os.close(fd)
