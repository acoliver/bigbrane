"""Runs once at the end of the tools image build (as root, with network still available).

Fails the build unless every Lean-side executable is a Linux aarch64 ELF (the lean4 issue 13306 regression
shipped an x86 leantar), then writes /opt/bigbrane/tools.json: tool versions, the Lake manifest revisions and
the measured size of the Mathlib build output."""
import json
import os
import subprocess
import sys
from pathlib import Path

EM_AARCH64 = 183
PROJECT = Path("/opt/mathproj")
OUT = Path("/opt/bigbrane/tools.json")


def elf_machine(path):
    head = Path(path).read_bytes()[:20]
    if head[:4] != b"\x7fELF":
        raise SystemExit("%s is not an ELF file" % path)
    return int.from_bytes(head[18:20], "little")


def tree_bytes(root):
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                total += os.path.getsize(path)
    return total


def main():
    checked = {}
    candidates = [Path("/opt/lean/bin/lean"), Path("/opt/lean/bin/lake")]
    candidates += sorted(Path("/opt/lean/bin").glob("leantar*"))
    candidates += sorted(Path("/root/.cache/mathlib").glob("leantar*"))
    for path in candidates:
        if path.is_file():
            machine = elf_machine(path)
            checked[str(path)] = machine
            if machine != EM_AARCH64:
                raise SystemExit("%s has ELF machine %d, not aarch64 (%d)" % (path, machine, EM_AARCH64))

    def out(*argv):
        return subprocess.run(argv, check=True, capture_output=True, text=True, cwd=PROJECT).stdout.strip()

    import mpmath
    import numpy
    import sympy
    import z3

    manifest = json.loads((PROJECT / "lake-manifest.json").read_text())
    packages = {p["name"]: p.get("rev") for p in manifest["packages"]}
    mathlib_build = PROJECT / ".lake" / "packages" / "mathlib" / ".lake" / "build"
    info = {
        "lean_version": out("/opt/lean/bin/lean", "--version"),
        "lake_version": out("/opt/lean/bin/lake", "--version"),
        "lean_toolchain": (PROJECT / "lean-toolchain").read_text().strip(),
        "lake_packages": packages,
        "elf_machines": checked,
        "mathlib_build_bytes": tree_bytes(mathlib_build),
        "packages_bytes": tree_bytes(PROJECT / ".lake" / "packages"),
        "lean_toolchain_bytes": tree_bytes("/opt/lean"),
        "python": {"z3": z3.get_version_string(), "sympy": sympy.__version__, "numpy": numpy.__version__,
                   "mpmath": mpmath.__version__},
    }
    OUT.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
