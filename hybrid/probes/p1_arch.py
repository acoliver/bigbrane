"""P0 probe 1: report CPU architecture as seen by the job."""
import json
import os
import platform
import struct
import sys


def elf_machine(path):
    with open(path, "rb") as handle:
        header = handle.read(20)
    assert header[:4] == b"\x7fELF", path
    return struct.unpack("<H", header[18:20])[0]


cpuinfo = open("/proc/cpuinfo").read()
result = {
    "platform_machine": platform.machine(),
    "uname_machine": os.uname().machine,
    "python": sys.version.split()[0],
    "python_elf_machine": elf_machine(os.path.realpath(sys.executable)),
    "cpuinfo_mentions_x86_vendor": any(v in cpuinfo for v in ("GenuineIntel", "AuthenticAMD")),
    "cpuinfo_mentions_aarch64_features": "Features" in cpuinfo,
}
json.dump(result, open("/out/result.json", "w"), sort_keys=True)
