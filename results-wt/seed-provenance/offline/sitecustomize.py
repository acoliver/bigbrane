"""Deny all real network connections during provenance verification."""
import sys

def offline(event, args):
    if event in {'socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.sendto'}:
        raise RuntimeError('network prohibited during seed provenance verification')

sys.addaudithook(offline)
