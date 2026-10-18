"""Run the unchanged seven transport tool tests with local temp space and no network."""
from pathlib import Path
import runpy
import socket
import sys
import tempfile

OUT = Path(__file__).resolve().parent
root = OUT / 'test-work'
root.mkdir(exist_ok=True)
tempfile.tempdir = str(root)

def prohibited(*args, **kwargs):
    raise AssertionError('Network prohibited during scoped offline tests')

socket.socket.connect = prohibited
socket.socket.connect_ex = prohibited
socket.create_connection = prohibited
prior = OUT.parent / 'transport-measurement'
sys.path.insert(0, str(prior))
sys.argv = [str(prior / 'test_measure.py'), '-v']
runpy.run_path(str(prior / 'test_measure.py'), run_name='__main__')
