"""Build a Bigbrane image: free-space precheck, labeled ``podman build``, then the lock file.

    python -m hybrid.executor.build tools --run phase1-20261012T120000Z
    python -m hybrid.executor.build executor --run p0-20261012T120000Z

Refuses (exit 2) when the internal disk holding Podman storage or the Podman VM disk is below the
thresholds in hybrid/diskguard.py. Every build carries ``bigbrane.owner=bigbrane``, ``bigbrane.run`` and
``bigbrane.image`` labels and uses ``--layers=false``, so no unlabeled intermediate images are left behind.
The tag is ``localhost/bigbrane/<image>:<run id>``; the lock file records the digest reference jobs use.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from hybrid import diskguard
from hybrid.executor import lock
from hybrid.executor.policy import OWNER_LABEL

HERE = Path(__file__).parent
REPOSITORY = "localhost/bigbrane/%s"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")


def image_tag(kind: str, run_id: str) -> str:
    return "%s:%s" % (REPOSITORY % kind, run_id)


def build_argv(kind: str, run_id: str) -> list[str]:
    """The exact ``podman build`` arguments. Pure, so the labels are unit-tested."""
    if kind not in lock.IMAGES:
        raise ValueError("unknown image %r" % (kind,))
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("bad run id %r" % (run_id,))
    labels = [OWNER_LABEL, ("bigbrane.run", run_id), ("bigbrane.image", kind)]
    return ["build", "--platform", "linux/arm64", "--layers=false",
            *[arg for key, value in labels for arg in ("--label", "%s=%s" % (key, value))],
            "-f", str(lock.IMAGES[kind]["containerfile"]), "-t", image_tag(kind, run_id), str(HERE)]


def main(argv: list[str] | None = None, *, check=diskguard.require_build_space, measure=diskguard.measure_build,
         runner=subprocess.run) -> int:
    parser = argparse.ArgumentParser(prog="python -m hybrid.executor.build")
    parser.add_argument("image", choices=sorted(lock.IMAGES))
    parser.add_argument("--run", required=True, help="run id, used in the tag and the bigbrane.run label")
    args = parser.parse_args(argv)
    argv_build = build_argv(args.image, args.run)
    if not diskguard.guard_cli(check, "the %s image build" % args.image, report=True):
        return 2
    print("+ podman " + " ".join(argv_build), flush=True)
    done = runner(["podman", *argv_build])
    if done.returncode != 0:
        print("podman build failed with exit %d" % done.returncode, file=sys.stderr)
        return done.returncode
    written = lock.write_lock(image_tag(args.image, args.run), kind=args.image)
    print("lock: %s -> %s" % (lock.IMAGES[args.image]["lock"].name, written["reference"]), flush=True)
    for measurement in measure():
        print("disk after build: " + measurement.describe(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
