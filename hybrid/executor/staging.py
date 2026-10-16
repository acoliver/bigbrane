"""Moving job inputs and outputs as tar streams through restricted containers.

Nothing from the host filesystem is bind-mounted into a job. Inputs go into a
VM-local volume and outputs come back out of one, so the external drive's
virtiofs mount is never visible to candidate code. The host side refuses
anything that is not a plain file or directory in both directions."""
import hashlib
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath

from .policy import JOB_GID, JOB_UID

# Run inside the stager container (executor image, no network, non-root).
STAGE_IN_SCRIPT = (
    "import sys, tarfile\n"
    "with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as t:\n"
    "    t.extractall('/in', filter='data')\n"
)
STAGE_OUT_SCRIPT = (
    "import sys, tarfile\n"
    "with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as t:\n"
    "    t.add('/out', arcname='.')\n"
)


class UnsafeInput(ValueError):
    pass


def write_input_tar(fileobj, input_dir):
    """Stream input_dir as tar. Symlinks, devices and other special files are refused."""
    input_dir = Path(input_dir)
    with tarfile.open(fileobj=fileobj, mode="w|") as archive:
        for path in sorted(input_dir.rglob("*")):
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise UnsafeInput("input contains a non-regular entry (symlink or special file): %s" % path)
            info = archive.gettarinfo(str(path), arcname=path.relative_to(input_dir).as_posix())
            info.uid, info.gid, info.uname, info.gname = JOB_UID, JOB_GID, "", ""
            info.mode = 0o755 if stat.S_ISDIR(mode) or mode & 0o111 else 0o644
            if info.isreg():
                with open(path, "rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


class OutputLimitExceeded(RuntimeError):
    pass


def read_output_tar(fileobj, destination, max_bytes):
    """Write regular files from a tar stream under destination.
    Returns ({relative path: {sha256, bytes}}, [rejected entries]). Never uses extractall."""
    destination = Path(destination)
    outputs, rejected, total = {}, [], 0
    with tarfile.open(fileobj=fileobj, mode="r|") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if str(name) == ".":
                continue
            if name.is_absolute() or ".." in name.parts:
                rejected.append({"path": member.name, "reason": "unsafe path"})
            elif member.isdir():
                (destination / name).mkdir(parents=True, exist_ok=True)
            elif member.isreg():
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with open(target, "wb") as out, archive.extractfile(member) as source:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        size += len(chunk)
                        total += len(chunk)
                        if total > max_bytes:
                            raise OutputLimitExceeded("output exceeds %d bytes" % max_bytes)
                        digest.update(chunk)
                        out.write(chunk)
                os.chmod(target, 0o644)
                outputs[name.as_posix()] = {"sha256": digest.hexdigest(), "bytes": size}
            else:
                rejected.append({"path": str(name), "reason": "not a regular file or directory (type %r)" % member.type})
    return outputs, rejected
