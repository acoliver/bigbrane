"""Network-prohibited verification, with optional external-original isolation."""
import os
from pathlib import Path
import sys

ISOLATED=os.environ.get('COMPOSITES_ISOLATED')=='1'
ORIGINAL_ROOT='/Volumes/XS1000/acoliver/projects/bigbrane/runs/'


def prohibit(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr','socket.sendto'):
        raise RuntimeError('Network prohibited during checkpoint verification')
    if ISOLATED and event=='open' and args and isinstance(args[0],str) and args[0].startswith(ORIGINAL_ROOT):
        raise AssertionError('external original-run access prohibited in isolated package')


sys.addaudithook(prohibit)
if ISOLATED:
    original_exists=Path.exists
    original_is_file=Path.is_file
    def exists(self):
        return False if str(self).startswith(ORIGINAL_ROOT) else original_exists(self)
    def is_file(self):
        return False if str(self).startswith(ORIGINAL_ROOT) else original_is_file(self)
    Path.exists=exists
    Path.is_file=is_file
