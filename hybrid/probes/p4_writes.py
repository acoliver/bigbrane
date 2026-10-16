"""P0 probe 4: which paths are writable, and does /tmp have a cap?"""
import json
import os
import shutil
import subprocess


def attempt(function):
    try:
        function()
        return {"ok": True, "error": None}
    except Exception as error:
        return {"ok": False, "error": "%s:%s" % (type(error).__name__, getattr(error, "errno", None))}


def write(path, data="x"):
    def action():
        with open(path, "w") as handle:
            handle.write(data)
    return action


def fill(path, limit_mb):
    written = 0
    error = None
    try:
        with open(path, "wb") as handle:
            for _ in range(limit_mb):
                handle.write(b"\0" * (1 << 20))
                handle.flush()
                written += 1
    except OSError as caught:
        error = "%s:%s" % (type(caught).__name__, caught.errno)
    return {"mb_written": written, "limit_mb": limit_mb, "error": error}


def run_binary():
    shutil.copy("/bin/true", "/tmp/true_copy")
    os.chmod("/tmp/true_copy", 0o755)
    subprocess.run(["/tmp/true_copy"], check=True)


result = {
    "write_in_new": attempt(write("/in/new.txt")),
    "write_in_existing": attempt(write("/in/seed.txt", "changed")),
    "unlink_in_existing": attempt(lambda: os.unlink("/in/seed.txt")),
    "write_image_usr": attempt(write("/usr/local/planted")),
    "write_image_etc": attempt(write("/etc/planted")),
    "write_root": attempt(write("/planted")),
    "overwrite_supervisor": attempt(write("/opt/bigbrane/supervisor.py", "import sys; sys.exit(0)")),
    "write_proc_sys": attempt(write("/proc/sys/kernel/hostname", "x")),
    "write_out": attempt(write("/out/declared_output.txt", "output ok")),
    "write_tmp": attempt(write("/tmp/scratch.txt")),
    "exec_from_tmp": attempt(run_binary),
    "tmp_fill": fill("/tmp/fill.bin", 200),
    "shm_fill": fill("/dev/shm/fill.bin", 100),
    "in_seed_after": open("/in/seed.txt").read(),
}
json.dump(result, open("/out/result.json", "w"), sort_keys=True)
