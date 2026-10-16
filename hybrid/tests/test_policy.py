"""Unit tests that need no Podman: job validation, launch policy, classification."""
import tempfile
import unittest
from pathlib import Path

from hybrid.executor import policy, receipt
from hybrid.executor.job import Caps, Job

DIGEST = "localhost/bigbrane/executor@sha256:" + "a" * 64


def make_job(tmp, **overrides):
    inputs, outputs = Path(tmp, "in"), Path(tmp, "out")
    inputs.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    fields = dict(job_id="job-1", image=DIGEST, input_dir=inputs, output_dir=outputs,
                  command=("python3", "/in/x.py"), caps=Caps(wall_s=5, memory_mb=256, pids=64))
    fields.update(overrides)
    return Job(**fields)


class JobValidation(unittest.TestCase):
    def test_tag_only_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            make_job(tmp, image="localhost/bigbrane/executor:latest")

    def test_nonempty_output_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "out").mkdir()
            Path(tmp, "out", "stale").write_text("x")
            with self.assertRaises(ValueError):
                make_job(tmp)

    def test_caps_must_be_positive_and_tmp_below_memory(self):
        with self.assertRaises(ValueError):
            Caps(wall_s=0, memory_mb=128, pids=8)
        with self.assertRaises(ValueError):
            Caps(wall_s=1, memory_mb=64, pids=8, tmp_mb=64)

    def test_empty_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            make_job(tmp, command=())


class LaunchPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.argv = policy.job_create_argv(make_job(self.tmp.name), "p0-x", "c1", "vin", "vout")

    def pair(self, flag):
        return self.argv[self.argv.index(flag) + 1]

    def test_every_restriction_is_present(self):
        for flag in ("--network=none", "--read-only", "--cap-drop=ALL",
                     "--security-opt=no-new-privileges", "--pull=never", "--read-only-tmpfs=false"):
            self.assertIn(flag, self.argv)
        self.assertEqual(self.pair("--user"), "10001:10001")
        self.assertEqual(self.pair("--pids-limit"), "64")
        self.assertEqual(self.pair("--memory"), "256m")
        self.assertEqual(self.pair("--memory-swap"), "256m")
        self.assertEqual(self.pair("--cpus"), "1.0")

    def test_labels_on_every_resource(self):
        pairs = [self.argv[i + 1] for i, a in enumerate(self.argv) if a == "--label"]
        self.assertEqual(pairs, ["bigbrane.owner=bigbrane", "bigbrane.run=p0-x", "bigbrane.job=job-1"])
        volume = policy.volume_create_argv("vin", "p0-x", "job-1")
        self.assertIn("bigbrane.owner=bigbrane", volume)
        stager = policy.stager_run_argv(make_job(self.tmp.name), "p0-x", "s", "vin", "/in", False, "pass")
        self.assertIn("bigbrane.run=p0-x", stager)

    def test_only_declared_mounts(self):
        mounts = [self.argv[i + 1] for i, a in enumerate(self.argv) if a in ("--mount", "--tmpfs", "-v", "--volume")]
        self.assertEqual(len(mounts), 4)
        self.assertIn("/run/secrets:ro,noexec,nosuid,nodev,size=4k", mounts)
        self.assertIn("type=volume,source=vin,destination=/in,ro=true", mounts)
        self.assertIn("type=volume,source=vout,destination=/out", mounts)
        self.assertTrue(any(m.startswith("/tmp:rw,noexec,nosuid,nodev,size=64m") for m in mounts))
        for forbidden in ("--privileged", "--device", "--pid=host", "--network=host", "--volumes-from"):
            self.assertFalse(any(a.startswith(forbidden) for a in self.argv))
        self.assertNotIn("/var/run/podman", " ".join(self.argv))
        self.assertNotIn("/Volumes", " ".join(self.argv))

    def test_thread_env_is_one(self):
        envs = [self.argv[i + 1] for i, a in enumerate(self.argv) if a == "--env"]
        self.assertIn("OPENBLAS_NUM_THREADS=1", envs)
        self.assertIn("OMP_NUM_THREADS=1", envs)

    def test_image_and_command_come_last(self):
        self.assertEqual(self.argv[-3:], [DIGEST, "python3", "/in/x.py"])


class ImageVerification(unittest.TestCase):
    class FakePodman:
        def __init__(self, **image):
            self.image = {"Os": "linux", "Architecture": "arm64", "Digest": "sha256:" + "a" * 64, "Id": "x", **image}

        def inspect(self, kind, name):
            return self.image

    def verify(self, **image):
        from hybrid.executor.executor import Executor
        with tempfile.TemporaryDirectory() as tmp:
            return Executor("p0-unit", tmp, self.FakePodman(**image)).verify_image(DIGEST)

    def test_arm64_image_with_matching_digest_passes(self):
        self.assertEqual(self.verify()["Architecture"], "arm64")

    def test_amd64_image_is_refused(self):
        from hybrid.executor.executor import ImagePolicyError
        with self.assertRaises(ImagePolicyError):
            self.verify(Architecture="amd64")

    def test_digest_mismatch_is_refused(self):
        from hybrid.executor.executor import ImagePolicyError
        with self.assertRaises(ImagePolicyError):
            self.verify(Digest="sha256:" + "b" * 64)


class Classification(unittest.TestCase):
    base = dict(timed_out=False, oom_killed=False, started=True, runtime_error="", exit_code=0)

    def status(self, **change):
        return receipt.classify(**{**self.base, **change})

    def test_each_status(self):
        self.assertEqual(self.status(), "ok")
        self.assertEqual(self.status(exit_code=1), "candidate_error")
        self.assertEqual(self.status(exit_code=137, timed_out=True), "timeout")
        self.assertEqual(self.status(exit_code=137, oom_killed=True), "oom")
        self.assertEqual(self.status(exit_code=receipt.PIDS_LIMIT_EXIT), "pids")
        self.assertEqual(self.status(exit_code=receipt.SUPERVISOR_ERROR_EXIT), "infra_error")
        self.assertEqual(self.status(started=False, exit_code=-1), "infra_error")
        self.assertEqual(self.status(runtime_error="oci error"), "infra_error")


if __name__ == "__main__":
    unittest.main()
