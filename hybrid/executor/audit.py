"""Podman label audit (read only). ``scripts/podman-label-audit.sh`` runs it.

Lists every Podman resource labeled ``bigbrane.owner=bigbrane`` (containers, images, volumes, pods), then the
things our builds created or brought in that lack that label:

* images named ``localhost/bigbrane/*`` without the owner label;
* images carrying some ``bigbrane.*`` label but not ``bigbrane.owner=bigbrane``;
* build containers left by an interrupted ``podman build`` (``podman ps --external``, which has no labels)
  whose image is one of ours or a base image pinned in our Containerfiles.

Base images pinned by our Containerfiles are reported for information: a build pulls them and they carry no
Bigbrane label, so they are never removed by Bigbrane tooling. Nothing here changes any resource.
Exit 0 when nothing unlabeled of ours was found, 1 otherwise.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Callable

from hybrid.executor.lock import IMAGES

OWNER = "bigbrane.owner=bigbrane"
OWNER_FILTER = ("--filter", "label=" + OWNER)
OUR_REPOSITORY = "localhost/bigbrane/"


def pinned_base_digests() -> set[str]:
    digests = set()
    for spec in IMAGES.values():
        path = spec["containerfile"]
        if path.exists():
            digests |= set(re.findall(r"^FROM\s+\S+@(sha256:[0-9a-f]{64})", path.read_text(), re.M | re.I))
    return digests


def _owned(labels: dict | None) -> bool:
    return (labels or {}).get("bigbrane.owner") == "bigbrane"


def audit(podman: Callable[[list[str]], str], base_digests: set[str]) -> dict:
    def listing(argv):
        return json.loads(podman(argv) or "[]")

    labeled = {
        "containers": listing(["ps", "-a", *OWNER_FILTER, "--format", "json"]),
        "images": listing(["images", *OWNER_FILTER, "--format", "json"]),
        "volumes": listing(["volume", "ls", *OWNER_FILTER, "--format", "json"]),
        "pods": listing(["pod", "ls", *OWNER_FILTER, "--format", "json"]),
    }
    all_images = listing(["images", "--format", "json"])
    unlabeled_ours, stray_labels, bases = [], [], []
    our_ids = set()
    for image in all_images:
        names = image.get("Names") or []
        labels = image.get("Labels") or {}
        ours_by_name = any(name.startswith(OUR_REPOSITORY) for name in names)
        if ours_by_name or _owned(labels):
            our_ids.add(image["Id"])
        if ours_by_name and not _owned(labels):
            unlabeled_ours.append(image)
        elif not _owned(labels) and any(key.startswith("bigbrane.") for key in labels):
            stray_labels.append(image)
        if image.get("Digest") in base_digests or set(image.get("RepoDigests") or []) & {
                f"{name.rsplit('@', 1)[0]}@{d}" for name in names for d in base_digests}:
            bases.append(image)
    external = listing(["ps", "-a", "--external", "--format", "json"])
    build_leftovers = [c for c in external
                       if not _owned(c.get("Labels")) and (c.get("ImageID") in our_ids
                                                          or any(d in (c.get("Image") or "") for d in base_digests))
                       and c.get("Status", "").lower().startswith("storage")]
    return {"labeled": labeled, "unlabeled_ours": unlabeled_ours, "stray_bigbrane_labels": stray_labels,
            "build_leftovers": build_leftovers, "base_images": bases}


def _short(image: dict) -> str:
    names = ",".join(image.get("Names") or []) or "<untagged>"
    return f"{image['Id'][:12]} {names}"


def render(report: dict) -> list[str]:
    lines = ["Bigbrane-labeled resources (" + OWNER + "):"]
    labeled = report["labeled"]
    for kind in ("pods", "containers", "images", "volumes"):
        items = labeled[kind]
        lines.append(f"  {kind}: {len(items)}")
        for item in items:
            if kind == "images":
                lines.append("    " + _short(item))
            elif kind == "volumes":
                lines.append("    " + item.get("Name", "?"))
            else:
                names = item.get("Names") or item.get("Name") or "?"
                lines.append(f"    {item.get('Id', '?')[:12]} {names}")
    sections = [
        ("images named localhost/bigbrane/* without the owner label", report["unlabeled_ours"], _short),
        ("images with bigbrane.* labels but not the owner label", report["stray_bigbrane_labels"], _short),
        ("unlabeled build containers left by podman build", report["build_leftovers"],
         lambda c: f"{c.get('Id', '?')[:12]} {c.get('Names')} image {c.get('Image')}"),
    ]
    for title, items, fmt in sections:
        lines.append(f"{title}: {'none' if not items else len(items)}")
        lines += ["  " + fmt(item) for item in items]
    lines.append("base images pinned by our Containerfiles (pulled by builds, unlabeled, never removed by us): "
                 + (str(len(report["base_images"])) if report["base_images"] else "none"))
    lines += ["  " + _short(image) for image in report["base_images"]]
    return lines


def problems(report: dict) -> int:
    return len(report["unlabeled_ours"]) + len(report["stray_bigbrane_labels"]) + len(report["build_leftovers"])


def podman_cli(argv: list[str]) -> str:
    return subprocess.run(["podman", *argv], check=True, capture_output=True, text=True, timeout=120).stdout


def main() -> int:
    report = audit(podman_cli, pinned_base_digests())
    print("\n".join(render(report)))
    return 1 if problems(report) else 0


if __name__ == "__main__":
    sys.exit(main())
