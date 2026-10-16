"""Check-hash manifest (DISCOVERY-PIPELINE-PLAN.md section 5.5): the sha256 of every Python file under
hybrid/math, recorded in hybrid/math/MANIFEST.json and verified before any check runs, so a generator or prover
that edited a check would be caught.

    python -m hybrid.math.manifest            # verify; exit 1 and list the differences on a mismatch
    python -m hybrid.math.manifest --write    # record the current hashes (an operator action after a reviewed change)

``verify()`` raises ``ManifestMismatch`` on a changed, missing or unlisted file and returns the hashes otherwise.
``MathRunner.check_library`` calls it before staging anything, records the hashes in its result, and the driver in
the tools image (jobmain.py) re-hashes the staged files against them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MATH_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = MATH_DIR / "MANIFEST.json"
ALGORITHM = "sha256"


class ManifestMismatch(RuntimeError):
    def __init__(self, problems: list[str]):
        super().__init__("check files differ from MANIFEST.json: " + "; ".join(problems))
        self.problems = problems


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current(directory: Path | None = None) -> dict[str, str]:
    """Hashes of every .py file directly under ``directory`` (check modules, the driver and the tools)."""
    return {path.name: sha256_file(path) for path in sorted(Path(directory or MATH_DIR).glob("*.py"))}


def read(path: Path | None = None) -> dict[str, str]:
    path = path or MANIFEST_PATH
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("algorithm") != ALGORITHM or not isinstance(data.get("files"), dict):
        raise ValueError(f"{path} is not a {ALGORITHM} manifest")
    return data["files"]


def differences(recorded: dict[str, str], found: dict[str, str]) -> list[str]:
    problems = [f"{name}: changed" for name in sorted(recorded) if name in found and found[name] != recorded[name]]
    problems += [f"{name}: missing" for name in sorted(set(recorded) - set(found))]
    problems += [f"{name}: not in the manifest" for name in sorted(set(found) - set(recorded))]
    return problems


def verify(directory: Path | None = None, path: Path | None = None) -> dict[str, str]:
    recorded = read(path)
    problems = differences(recorded, current(directory))
    if problems:
        raise ManifestMismatch(problems)
    return recorded


def write(directory: Path | None = None, path: Path | None = None) -> dict[str, str]:
    files = current(directory)
    Path(path or MANIFEST_PATH).write_text(json.dumps({"algorithm": ALGORITHM, "files": files}, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hybrid.math.manifest")
    parser.add_argument("--write", action="store_true", help="record the current hashes")
    args = parser.parse_args(argv)
    if args.write:
        files = write()
        print(f"wrote {MANIFEST_PATH} with {len(files)} files")
        return 0
    try:
        files = verify()
    except ManifestMismatch as exc:
        print("\n".join(exc.problems))
        return 1
    print(f"{len(files)} check files match {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
