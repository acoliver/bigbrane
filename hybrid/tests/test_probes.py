"""The eight P0 probes of HYBRID-PLAN.md, run inside real executor containers.

    BIGBRANE_P0_RUN=<run id> .venv/bin/python -m unittest hybrid.tests.test_probes -v

The probe programs only observe and write result.json. All assertions are here, on the host."""
import hashlib
import json
import re
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from hybrid.executor import lock as image_lock
from hybrid.executor.podman import Podman, PodmanError
from hybrid.fixtures import trial_fixture
from hybrid.tests.support import (DEFAULT_CAPS, EVIDENCE, EXECUTOR, FIXTURES, LOCK, PODMAN, PROBES, ROOT,
                                  RUN_ID, labeled, run_job, run_probe)
from hybrid.executor.job import Caps

EXPECTED_HMM_KL = 0.12230803167784307
EXPECTED_RNN_KL = 0.005312276268561993
ZERO_CAPS = "0000000000000000"


def setUpModule():
    PODMAN.run(["info"])
    image = PODMAN.inspect("image", LOCK["reference"])
    assert image["Id"] == LOCK["image_id"], "image.lock.json does not match the local image"


def tearDownModule():
    leftovers = {kind: labeled(kind) for kind in ("containers", "volumes", "pods")}
    assert not any(leftovers.values()), "executor left labeled resources behind: %r" % (leftovers,)


class Probe1Architecture(unittest.TestCase):
    def test_aarch64_and_digest_recorded(self):
        receipt, result, _ = run_probe("p1-arch", "p1_arch.py")
        self.assertEqual(receipt.status, "ok")
        self.assertEqual(result["platform_machine"], "aarch64")
        self.assertEqual(result["uname_machine"], "aarch64")
        self.assertEqual(result["python_elf_machine"], 183)  # EM_AARCH64; amd64 under Rosetta would be 62
        self.assertFalse(result["cpuinfo_mentions_x86_vendor"])
        self.assertEqual(receipt.image_architecture, "arm64")
        self.assertRegex(receipt.image_digest, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(receipt.image_digest, LOCK["digest"])
        self.assertTrue(receipt.image.endswith("@" + LOCK["digest"]))
        on_disk = json.loads((EVIDENCE / "receipts" / "p1-arch.receipt.json").read_text())
        self.assertEqual(on_disk["image_digest"], LOCK["digest"])

    def test_base_image_is_pinned_by_arm64_digest(self):
        self.assertEqual(image_lock.base_digest(), LOCK["base_digest"])
        base = PODMAN.inspect("image", "docker.io/library/python@" + LOCK["base_digest"])
        self.assertEqual((base["Os"], base["Architecture"]), ("linux", "arm64"))
        self.assertEqual(LOCK["architecture"], "arm64")


class Probe2Network(unittest.TestCase):
    def test_every_route_off_the_container_fails(self):
        receipt, result, _ = run_probe("p2-network", "p2_network.py")
        self.assertEqual(receipt.status, "ok")
        for name in ("dns_public_name", "dns_udp_to_public_resolver", "tcp_public_ip_1",
                     "tcp_public_ip_2", "tcp_cloud_metadata", "tcp_podman_default_gateway", "tcp_podman_host_alias_ip"):
            self.assertFalse(result[name]["connected"], "%s connected: %r" % (name, result[name]))
        self.assertTrue(result["host_gateway_name_resolves_to"], "expected podman's /etc/hosts entry for host.containers.internal")
        for target, outcome in result["tcp_host_gateway"].items():
            self.assertFalse(outcome["connected"], "host gateway %s connected" % target)
        self.assertTrue(result["loopback_control"]["connected"], "loopback must work or the probe proves nothing")
        self.assertEqual(result["interfaces"], ["lo"])
        self.assertEqual(result["default_routes"], [])
        self.assertEqual(receipt.effective_policy["network_mode"], "none")

    def test_probe_detects_connectivity_when_network_is_not_disabled(self):
        """Control: same kind of attempt on an unrestricted-network container succeeds, so the failures above are meaningful."""
        script = ("import socket; socket.setdefaulttimeout(8); "
                  "socket.getaddrinfo('pypi.org', 443); socket.create_connection(('1.1.1.1', 443)).close(); print('reached')")
        done = PODMAN.run(["run", "--rm", "--pull=never", "--label", "bigbrane.owner=bigbrane",
                           "--label", "bigbrane.run=" + RUN_ID, "--entrypoint", "python3",
                           LOCK["reference"], "-c", script], timeout=60)
        self.assertEqual(done.stdout.strip(), "reached")


class Probe3Filesystem(unittest.TestCase):
    ALLOWED_MOUNT_POINTS = {"/", "/in", "/out", "/tmp", "/etc/hosts", "/etc/hostname", "/etc/resolv.conf",
                            "/run/.containerenv", "/run/secrets"}

    @classmethod
    def setUpClass(cls):
        holdout_dir = EVIDENCE / "holdout-fixture"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        cls.holdout = holdout_dir / ("holdout-%s.secret" % uuid.uuid4().hex)
        cls.holdout.write_text("SECRET-" + uuid.uuid4().hex)
        assert cls.holdout.is_absolute() and cls.holdout.exists()
        cls.receipt, cls.result, cls.outputs = run_probe(
            "p3-filesystem", "p3_filesystem.py", extra_files={"holdout_path.txt": str(cls.holdout).encode()})

    def test_probe_ran(self):
        self.assertEqual(self.receipt.status, "ok", self.receipt.detail)

    def test_holdout_unreachable_from_inside(self):
        r = self.result
        self.assertTrue(self.holdout.exists())  # exists on the host
        self.assertFalse(r["holdout_exists"])
        self.assertFalse(r["read_holdout"]["ok"])
        self.assertFalse(r["read_holdout_via_in_dotdot"]["ok"])
        self.assertFalse(r["list_holdout_dir"]["ok"])
        self.assertEqual(r["existing_ancestors"], [])
        self.assertEqual(r["walk_found_holdout_name"], [])

    def test_no_host_directories_or_host_filesystems_visible(self):
        r = self.result
        self.assertFalse({"Users", "Volumes", "private", "Applications", "Library", "System"} & set(r["root_listing"]))
        for mount in r["mounts"]:
            self.assertNotRegex(mount["mount_point"], r"^/(Volumes|Users|private|var/folders)")
            self.assertNotIn(mount["fstype"], ("virtiofs", "9p", "nfs", "cifs"))
            self.assertFalse(mount["fstype"].startswith("fuse"), mount)
        visible = {m["mount_point"] for m in r["mounts"] if not m["mount_point"].startswith(("/proc", "/sys", "/dev"))}
        self.assertEqual(visible - self.ALLOWED_MOUNT_POINTS, set())

    def test_only_declared_input_is_visible_and_readable(self):
        self.assertEqual(self.result["in_listing"], ["holdout_path.txt", "p3_filesystem.py", "seed.txt"])
        self.assertTrue(self.result["read_in_seed"]["ok"])

    def test_podman_inspect_mounts_and_arguments_exclude_host_paths(self):
        receipt = self.receipt
        self.assertEqual(sorted(m["Destination"] for m in receipt.mounts), ["/in", "/out"])
        for mount in receipt.mounts:
            self.assertEqual(mount["Type"], "volume")
            self.assertIn("/containers/storage/volumes/bb-", mount["Source"])
        self.assertTrue(receipt.mounts[0]["RW"] != receipt.mounts[1]["RW"])
        flat = json.dumps([receipt.mounts, receipt.create_argv, receipt.effective_policy])
        self.assertNotIn(str(self.holdout), flat)
        self.assertNotIn(str(self.holdout.parent), flat)
        self.assertNotIn("/Volumes", flat)
        self.assertNotIn("/Users", flat)
        self.assertNotIn("sock", flat)

    def test_symlinks_written_to_output_are_refused_not_followed_on_host(self):
        rejected = {entry["path"] for entry in self.receipt.rejected_outputs}
        self.assertEqual(rejected, {"link_to_holdout", "link_to_in_parent"})
        for name in rejected:
            self.assertFalse((self.outputs / name).exists() or (self.outputs / name).is_symlink())
        self.assertEqual(self.receipt.outputs.keys(), {"result.json"})

    def test_symlink_in_input_is_refused_before_anything_runs(self):
        from hybrid.executor.staging import UnsafeInput
        link_target = {"escape": None}
        job_dir = EVIDENCE / "p3-input-symlink"
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        (job_dir / "in").mkdir(parents=True)
        (job_dir / "out").mkdir()
        (job_dir / "in" / "escape").symlink_to(self.holdout)
        from hybrid.executor.job import Job
        job = Job("p3-input-symlink", LOCK["reference"], job_dir / "in", job_dir / "out", ("true",), DEFAULT_CAPS)
        with self.assertRaises(UnsafeInput):
            EXECUTOR.run(job)
        self.assertEqual(labeled("containers"), [])
        self.assertEqual(labeled("volumes"), [])
        del link_target


class Probe4Writes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt, cls.result, cls.outputs = run_probe("p4-writes", "p4_writes.py")

    def test_probe_ran(self):
        self.assertEqual(self.receipt.status, "ok", self.receipt.detail)

    def test_inputs_and_image_root_are_read_only(self):
        r = self.result
        for name in ("write_in_new", "write_in_existing", "unlink_in_existing", "write_image_usr", "write_image_etc",
                     "write_root", "overwrite_supervisor", "write_proc_sys"):
            self.assertFalse(r[name]["ok"], "%s unexpectedly succeeded" % name)
        for name in ("write_in_new", "write_in_existing", "unlink_in_existing", "write_image_usr", "write_root",
                     "overwrite_supervisor"):
            self.assertIn("OSError:30", r[name]["error"].replace("PermissionError", "OSError").replace("OSError:13", "OSError:30"))
        self.assertEqual(r["in_seed_after"], "seed-data")
        self.assertTrue(self.receipt.effective_policy["read_only_rootfs"])
        self.assertFalse(next(m for m in self.receipt.mounts if m["Destination"] == "/in")["RW"])

    def test_out_and_tmp_are_writable_and_output_is_hashed(self):
        self.assertTrue(self.result["write_out"]["ok"])
        self.assertTrue(self.result["write_tmp"]["ok"])
        expected = hashlib.sha256(b"output ok").hexdigest()
        self.assertEqual(self.receipt.outputs["declared_output.txt"]["sha256"], expected)
        self.assertEqual(hashlib.sha256((self.outputs / "declared_output.txt").read_bytes()).hexdigest(), expected)

    def test_tmp_is_capped_and_not_executable(self):
        fill = self.result["tmp_fill"]
        self.assertLess(fill["mb_written"], fill["limit_mb"])
        self.assertLessEqual(fill["mb_written"], DEFAULT_CAPS.tmp_mb)
        self.assertEqual(fill["error"], "OSError:28")  # ENOSPC
        shm = self.result["shm_fill"]
        self.assertEqual((shm["mb_written"], shm["error"]), (0, "OSError:30"))  # /dev/shm is read-only under --read-only
        self.assertFalse(self.result["exec_from_tmp"]["ok"])


class Probe5Caps(unittest.TestCase):
    def test_fork_bomb_stops_at_pids_cap_and_is_classified_pids(self):
        for mode in ("raise", "catch"):
            with self.subTest(mode=mode):
                receipt, _, _ = run_probe("p5-pids-" + mode, "p5_pids.py", (mode,), caps=Caps(wall_s=60, memory_mb=256, pids=32))
                self.assertEqual(receipt.status, "pids", receipt.detail)
                self.assertEqual(receipt.exit_code, 254)
                self.assertEqual(receipt.effective_policy["pids_limit"], 32)
                stdout = Path(receipt.stdout_path).read_text()
                self.assertIn("fork refused after", stdout)
                children = int(stdout.split("fork refused after ")[1].split()[0])
                self.assertLess(children, 32)
                self.assertGreater(children, 5)

    def test_memory_hog_is_killed_and_classified_oom(self):
        receipt, _, _ = run_probe("p5-memory", "p5_memory.py", caps=Caps(wall_s=60, memory_mb=128, pids=64, tmp_mb=16))
        self.assertEqual(receipt.status, "oom", receipt.detail)
        self.assertEqual(receipt.exit_code, 137)
        self.assertTrue(receipt.effective_policy["oom_killed"])
        self.assertEqual(receipt.effective_policy["memory"], 128 * 1024 * 1024)
        self.assertEqual(receipt.effective_policy["memory_swap"], 128 * 1024 * 1024)
        held = int(Path(receipt.stdout_path).read_text().split()[-1])
        self.assertLessEqual(held, 128)

    def test_sleeping_past_the_wall_cap_is_classified_timeout(self):
        receipt, _, _ = run_probe("p5-wall", "p5_sleep.py", caps=Caps(wall_s=4, memory_mb=128, pids=64, tmp_mb=16))
        self.assertEqual(receipt.status, "timeout", receipt.detail)
        self.assertGreaterEqual(receipt.run_s, 4)
        self.assertLess(receipt.run_s, 30)
        self.assertEqual(Path(receipt.stdout_path).read_text().strip(), "sleeping")
        exists = subprocess.run(["podman", "container", "exists", receipt.container_name])
        self.assertEqual(exists.returncode, 1, "timed-out container was not removed")


class Probe6Identity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt, cls.result, _ = run_probe("p6-identity", "p6_identity.py")

    def test_non_root_with_no_capabilities(self):
        r = self.result
        self.assertEqual(self.receipt.status, "ok", self.receipt.detail)
        self.assertEqual((r["uid"], r["euid"], r["gid"], r["egid"]), (10001, 10001, 10001, 10001))
        self.assertEqual(r["caps"], {k: ZERO_CAPS for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")})
        self.assertEqual(r["no_new_privs"], "1")
        self.assertFalse(r["setuid_root"]["ok"])
        self.assertFalse(r["mount_syscall"]["ok"])
        self.assertEqual(self.receipt.effective_policy["user"], "10001:10001")
        self.assertIn("no-new-privileges", self.receipt.effective_policy["security_opt"])
        self.assertFalse(self.receipt.effective_policy["privileged"])

    def test_no_podman_socket_or_credentials(self):
        r = self.result
        self.assertEqual(r["sockets_found"], [])
        self.assertFalse(any(r["podman_socket_paths"].values()), r["podman_socket_paths"])
        self.assertFalse(any(r["credential_files"].values()), r["credential_files"])
        self.assertEqual(r["run_secrets_entries"], [])  # Podman's default VM-sourced /run/secrets copy is masked
        suspicious = [n for n in r["env_names"] if re.search(r"(^|_)HOST$|TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH|CONNECTION|DOCKER", n.upper())]
        # The python base image sets GPG_KEY to the public fingerprint of the release signing key (40 hex digits).
        self.assertEqual(suspicious, ["GPG_KEY"])
        self.assertRegex(r["env_gpg_key"], r"^[0-9A-F]{40}$")


class Probe7Threads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt, cls.result, _ = run_probe("p7-threads", "p7_threads.py")

    def test_blas_matmul_uses_one_thread(self):
        r = self.result
        self.assertEqual(self.receipt.status, "ok", self.receipt.detail)
        self.assertEqual(set(r["env"].values()), {"1"})
        self.assertEqual(r["numpy"]["worker_threads"], 0, r["numpy"])
        self.assertEqual(r["torch"]["configured_threads"], 1)
        self.assertEqual(r["torch"]["worker_threads"], 0, r["torch"])
        self.assertGreater(r["numpy"]["products"], 0)
        self.assertEqual(r["cgroup_cpu_max"], "100000 100000")  # --cpus=1

    def test_probe_sees_extra_threads_when_the_setting_is_raised(self):
        """Control: with OPENBLAS_NUM_THREADS=4 the same measurement reports worker threads."""
        self.assertGreaterEqual(self.result["control_numpy_four_threads"]["worker_threads"], 2, self.result)


class Probe8Fixture(unittest.TestCase):
    def test_trusted_fixture_is_bit_identical_in_container_and_on_host(self):
        receipt, result, outputs = run_probe(
            "p8-fixture", "p8_fixture.py", extra_files={"trial_fixture.py": FIXTURES / "trial_fixture.py"})
        self.assertEqual(receipt.status, "ok", receipt.detail)
        host_record = trial_fixture.evaluate(trial_fixture.CANDIDATES[0])
        host_text = json.dumps(host_record, sort_keys=True, allow_nan=False)
        container_text = (outputs / "record.json").read_text()
        container_record = json.loads(container_text)
        self.assertEqual(container_record["metrics"]["hmm_kl"], EXPECTED_HMM_KL)
        self.assertEqual(container_record["metrics"]["rnn_kl"], EXPECTED_RNN_KL)
        self.assertEqual(host_record["metrics"]["hmm_kl"], EXPECTED_HMM_KL)
        self.assertEqual(host_record["metrics"]["rnn_kl"], EXPECTED_RNN_KL)
        self.assertEqual(result["hmm_kl_hex"], host_record["metrics"]["hmm_kl"].hex())
        self.assertEqual(result["rnn_kl_hex"], host_record["metrics"]["rnn_kl"].hex())
        self.assertEqual(container_record["predictions_sha256"], host_record["predictions_sha256"])
        self.assertEqual(container_record["data_sha256"], host_record["data_sha256"])
        self.assertEqual(container_text, host_text)
        self.assertEqual(receipt.outputs["record.json"]["sha256"], hashlib.sha256(host_text.encode()).hexdigest())


class Probe8Diagnosis(unittest.TestCase):
    """Characterizes why probe 8 fails. These tests pass; they do not replace the strict bit-identity test."""
    DIAG = ROOT / "hybrid" / "probes" / "diag_confirm_cause.py"

    def host_hex(self, variant):
        done = subprocess.run([sys.executable, str(self.DIAG), str(FIXTURES / "trial_fixture.py"), variant],
                              capture_output=True, text=True, check=True)
        return json.loads(done.stdout)["hmm_kl_hex"]

    def container_hex(self, variant):
        receipt, _ = run_job("diag-cause-" + variant.replace("_", "-"),
                             ("python3", "/in/diag_confirm_cause.py", "/in/trial_fixture.py", variant),
                             {"diag_confirm_cause.py": self.DIAG, "trial_fixture.py": FIXTURES / "trial_fixture.py"})
        self.assertEqual(receipt.status, "ok", receipt.detail)
        return json.loads(Path(receipt.stdout_path).read_text())["hmm_kl_hex"]

    def test_native_and_container_hmm_kl_differ_by_ten_ulps(self):
        host, container = self.host_hex("as_is"), self.container_hex("as_is")
        self.assertEqual(host, float(EXPECTED_HMM_KL).hex())
        self.assertNotEqual(host, container)
        self.assertEqual(abs(float.fromhex(host) - float.fromhex(container)), 10 * 2.0 ** -56)  # 1.39e-16, 10 ulp

    def test_drift_disappears_when_only_the_matrix_vector_product_is_made_elementwise(self):
        self.assertEqual(self.host_hex("gemv_elementwise"), self.container_hex("gemv_elementwise"))


class ExecutorContract(unittest.TestCase):
    def test_candidate_error_keeps_exit_code(self):
        receipt, _ = run_job("c-candidate-error", ("python3", "-c", "import sys; print('boom'); sys.exit(3)"))
        self.assertEqual((receipt.status, receipt.exit_code), ("candidate_error", 3))
        self.assertEqual(Path(receipt.stdout_path).read_text().strip(), "boom")

    def test_missing_command_is_candidate_error(self):
        receipt, _ = run_job("c-missing-command", ("no-such-program",))
        self.assertEqual((receipt.status, receipt.exit_code), ("candidate_error", 127))

    def test_ok_receipt_has_timings_and_hashes(self):
        receipt, outputs = run_job(
            "c-ok", ("python3", "-c", "open('/out/a/b.bin','wb').write(bytes(range(256)))"),
            files=None)
        self.assertEqual(receipt.status, "candidate_error")  # /out/a does not exist: nested dirs are the job's duty
        receipt, outputs = run_job(
            "c-ok2", ("python3", "-c", "import os; os.makedirs('/out/a'); open('/out/a/b.bin','wb').write(bytes(range(256)))"))
        self.assertEqual(receipt.status, "ok")
        self.assertEqual(receipt.outputs["a/b.bin"], {"sha256": hashlib.sha256(bytes(range(256))).hexdigest(), "bytes": 256})
        self.assertGreater(receipt.run_s, 0)
        self.assertGreater(receipt.stage_in_s, 0)
        self.assertGreater(receipt.stage_out_s, 0)
        self.assertEqual(receipt.cleanup_errors, [])

    def test_runaway_logs_are_truncated_but_counted(self):
        receipt, _ = run_job("c-logs", ("python3", "-c", "print('x' * 3_000_000)"),
                             caps=Caps(wall_s=60, memory_mb=256, pids=64, max_log_bytes=65536))
        self.assertEqual(receipt.status, "ok")
        self.assertGreaterEqual(receipt.stdout_bytes, 3_000_000)
        self.assertEqual(Path(receipt.stdout_path).stat().st_size, 65536)

    def test_oversized_output_is_a_candidate_error(self):
        receipt, _ = run_job("c-big-output", ("python3", "-c", "open('/out/big','wb').write(b'0' * (8 << 20))"),
                             caps=Caps(wall_s=60, memory_mb=256, pids=64, max_output_bytes=1 << 20))
        self.assertEqual(receipt.status, "candidate_error")
        self.assertIn("exceeds", receipt.detail)

    def test_podman_failure_is_infra_error_and_still_cleans_up(self):
        class CreateFails(Podman):
            def run(self, argv, **kwargs):
                if argv[0] == "create":
                    raise PodmanError(argv, 125, "injected: engine unavailable")
                return super().run(argv, **kwargs)

        from hybrid.executor.executor import Executor
        failing = Executor(RUN_ID, EVIDENCE / "receipts", CreateFails())
        receipt, _ = run_job("c-infra", ("true",), executor=failing)
        self.assertEqual(receipt.status, "infra_error")
        self.assertIn("injected", receipt.detail)
        self.assertEqual(receipt.cleanup_errors, [])
        self.assertEqual(labeled("volumes"), [])

    def test_tag_reference_and_wrong_digest_are_refused(self):
        with self.assertRaises(ValueError):
            run_job("c-tag", ("true",), executor=EXECUTOR) if False else __import__("hybrid.executor.job", fromlist=["Job"]).Job(
                "c-tag", LOCK["tag"], ROOT, ROOT, ("true",), DEFAULT_CAPS)
        from hybrid.executor.job import Job
        import tempfile
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            job = Job("c-wrong", LOCK["reference"].split("@")[0] + "@sha256:" + "0" * 64, a, b, ("true",), DEFAULT_CAPS)
            with self.assertRaises(PodmanError):
                EXECUTOR.run(job)


if __name__ == "__main__":
    unittest.main()
