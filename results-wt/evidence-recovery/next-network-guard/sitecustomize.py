"""Prohibit network access in verification, including inherited Python subprocesses."""
import sys


def prohibit(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr'):
        raise AssertionError('network prohibited during historical evidence verification')


sys.addaudithook(prohibit)
