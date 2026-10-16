"""Thin wrapper over the podman CLI. Every nonzero exit raises unless the caller opts out."""
import json
import subprocess


class PodmanError(RuntimeError):
    def __init__(self, argv, returncode, stderr):
        super().__init__("podman %s failed (%s): %s" % (" ".join(argv[:3]), returncode, stderr.strip()))
        self.argv, self.returncode, self.stderr = argv, returncode, stderr


class Podman:
    def __init__(self, binary="podman"):
        self.binary = binary

    def run(self, argv, *, timeout=None, check=True):
        done = subprocess.run([self.binary, *argv], capture_output=True, text=True, timeout=timeout)
        if check and done.returncode != 0:
            raise PodmanError(argv, done.returncode, done.stderr)
        return done

    def popen(self, argv, **kwargs):
        return subprocess.Popen([self.binary, *argv], **kwargs)

    def inspect(self, kind, name):
        argv = [kind, "inspect", name] if kind != "container" else ["inspect", "--type", "container", name]
        return json.loads(self.run(argv).stdout)[0]
