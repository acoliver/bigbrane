"""Record and read the identity of a Bigbrane image (image.lock.json, image.tools.lock.json).

    python -m hybrid.executor.lock <image tag>                 # executor image: image.lock.json
    python -m hybrid.executor.lock --image tools <image tag>   # tools image: image.tools.lock.json
"""
import argparse
import json
import re
from pathlib import Path

from .podman import Podman

HERE = Path(__file__).parent
# Each image kind: its Containerfile and its lock file.
IMAGES = {
    "executor": {"containerfile": HERE / "Containerfile", "lock": HERE / "image.lock.json"},
    "tools": {"containerfile": HERE / "Containerfile.tools", "lock": HERE / "image.tools.lock.json"},
}
LOCK = IMAGES["executor"]["lock"]
CONTAINERFILE = IMAGES["executor"]["containerfile"]


def base_digest(kind="executor"):
    text = IMAGES[kind]["containerfile"].read_text()
    match = re.search(r"^FROM\s+\S+@(sha256:[0-9a-f]{64})(\s+AS\s+\S+)?\s*$", text, re.M | re.I)
    if not match:
        raise ValueError("%s FROM is not pinned by digest" % IMAGES[kind]["containerfile"].name)
    return match.group(1)


def write_lock(tag, podman=None, kind="executor", extra=None):
    image = (podman or Podman()).inspect("image", tag)
    repository = tag.rpartition(":")[0]
    lock = {
        "reference": "%s@%s" % (repository, image["Digest"]), "tag": tag, "digest": image["Digest"],
        "image_id": image["Id"], "architecture": image["Architecture"], "os": image["Os"],
        "base_digest": base_digest(kind), "labels": image["Labels"],
    }
    if extra:
        lock.update(extra)
    IMAGES[kind]["lock"].write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def read_lock(kind="executor"):
    return json.loads(IMAGES[kind]["lock"].read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hybrid.executor.lock")
    parser.add_argument("--image", choices=sorted(IMAGES), default="executor")
    parser.add_argument("tag")
    args = parser.parse_args(argv)
    print(json.dumps(write_lock(args.tag, kind=args.image), indent=2))


if __name__ == "__main__":
    main()
