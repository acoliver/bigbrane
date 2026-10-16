"""P0 probe 6: who am I, what may I do, and what can I reach to control containers?"""
import ctypes
import json
import os
import stat

status = {}
for line in open("/proc/self/status"):
    key, _, value = line.partition(":")
    status[key] = value.strip()


def attempt(function):
    try:
        function()
        return {"ok": True, "error": None}
    except Exception as error:
        return {"ok": False, "error": "%s:%s" % (type(error).__name__, getattr(error, "errno", None))}


def libc_mount():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mount(b"tmpfs", b"/tmp", b"tmpfs", 0, None) != 0:
        raise OSError(ctypes.get_errno(), "mount")


sockets = []
for root in ("/run", "/var/run", "/tmp", "/out", "/in", "/"):
    for base, dirs, files in os.walk(root, onerror=lambda error: None):
        if base in ("/proc", "/sys", "/dev", "/usr", "/opt", "/lib", "/var/lib") or (root == "/" and base != "/"):
            dirs[:] = []
            continue
        for name in files + dirs:
            path = os.path.join(base, name)
            try:
                if stat.S_ISSOCK(os.lstat(path).st_mode):
                    sockets.append(path)
            except OSError:
                pass

credential_files = [
    "/root/.docker/config.json", "/root/.config/containers/auth.json", "/run/containers/0/auth.json",
    "/etc/containers/auth.json", "/root/.ssh",
]


def run_secrets():
    """Without the executor's mask, Podman copies the VM's /usr/share/rhel/secrets here."""
    return [os.path.join(base, name) for base, dirs, files in os.walk("/run/secrets") for name in dirs + files]


result = {
    "uid": os.getuid(), "euid": os.geteuid(), "gid": os.getgid(), "egid": os.getegid(),
    "caps": {k: status[k] for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")},
    "no_new_privs": status["NoNewPrivs"], "seccomp": status["Seccomp"],
    "setuid_root": attempt(lambda: os.setuid(0)),
    "mount_syscall": attempt(libc_mount),
    "sockets_found": sockets,
    "podman_socket_paths": {p: os.path.exists(p) for p in (
        "/run/podman/podman.sock", "/var/run/podman/podman.sock", "/run/user/1000/podman/podman.sock",
        "/var/run/docker.sock", "/run/docker.sock", "/run/.containerenv_sock")},
    "env_names": sorted(os.environ),
    "env_gpg_key": os.environ.get("GPG_KEY"),
    "credential_files": {p: os.path.exists(p) for p in credential_files},
    "run_secrets_entries": run_secrets(),
}
json.dump(result, open("/out/result.json", "w"), sort_keys=True)
