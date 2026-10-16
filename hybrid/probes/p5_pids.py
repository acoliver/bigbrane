"""P0 probe 5a: fork until refused. argv[1] == 'catch' handles the failure and exits 0."""
import os
import sys
import time

catch = sys.argv[1] == "catch"
children = 0
try:
    for _ in range(100000):
        pid = os.fork()
        if pid == 0:
            time.sleep(30)
            os._exit(0)
        children += 1
except OSError as error:
    print("fork refused after %d children: errno %s" % (children, error.errno), flush=True)
    if not catch:
        raise
    sys.exit(0)
print("fork never refused", flush=True)
