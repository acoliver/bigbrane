"""Process-wide network prohibition for the offline runner and all scoped tests."""
import socket
import sys


def prohibit_network(event, args):
    if event == 'socket.__new__' and args[1] == socket.AF_UNIX:
        return
    if event.startswith('socket.'):
        raise RuntimeError('Network prohibited: ' + event)


def install():
    sys.addaudithook(prohibit_network)
