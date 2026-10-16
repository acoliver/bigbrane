"""Host-side executor: one fresh restricted container per job, one receipt per job."""
import datetime
import json
import subprocess
import threading
import time
from pathlib import Path

from . import policy, staging
from .job import Job
from .podman import Podman, PodmanError
from .receipt import CANDIDATE_ERROR, INFRA_ERROR, OK, Receipt, classify


class ImagePolicyError(RuntimeError):
    """The requested image is not a linux/arm64 image matching its pinned digest."""


class _Pump(threading.Thread):
    """Copies a stream to a file, keeping at most max_bytes but counting everything."""

    def __init__(self, stream, path, max_bytes):
        super().__init__(daemon=True)
        self.stream, self.path, self.max_bytes, self.total = stream, path, max_bytes, 0

    def run(self):
        with open(self.path, "wb") as out:
            for chunk in iter(lambda: self.stream.read(65536), b""):
                room = max(0, self.max_bytes - self.total)
                out.write(chunk[:room])
                self.total += len(chunk)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


class Executor:
    def __init__(self, run_id, receipts_dir, podman=None, slots=None):
        """``slots`` (hybrid.slots.JobSlots) bounds the jobs that run at once across every process on the host;
        None runs each job at once, as before."""
        self.run_id = run_id
        self.receipts_dir = Path(receipts_dir)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.podman = podman or Podman()
        self.slots = slots

    def verify_image(self, reference):
        """Fail fast unless the image exists locally, is linux/arm64 and matches its digest."""
        image = self.podman.inspect("image", reference)
        if (image["Os"], image["Architecture"]) != ("linux", "arm64"):
            raise ImagePolicyError("%s is %s/%s, not linux/arm64" % (reference, image["Os"], image["Architecture"]))
        if reference.rpartition("@")[2] != image["Digest"]:
            raise ImagePolicyError("%s does not match the local image digest %s" % (reference, image["Digest"]))
        return image

    def run(self, job: Job):
        """Run one job. With slots, the job waits for its share of the host-wide job slots first; its receipt times
        and its wall cap start once the slots are held, so the wait never counts against the job."""
        if self.slots is None:
            return self._run(job)
        with self.slots.hold(job.caps.memory_mb):
            return self._run(job)

    def _run(self, job: Job):
        image = self.verify_image(job.image)
        names = {kind: "bb-%s-%s-%s" % (self.run_id, job.job_id, kind) for kind in ("job", "in", "out", "stagein", "stageout")}
        base = self.receipts_dir / job.job_id
        paths = {"receipt": base.with_suffix(".receipt.json"), "stdout": base.with_suffix(".stdout"),
                 "stderr": base.with_suffix(".stderr")}
        create_argv = policy.job_create_argv(job, self.run_id, names["job"], names["in"], names["out"])
        state = {"detail": "", "exit_code": -1, "status": None, "mounts": [], "effective": {},
                 "outputs": {}, "rejected": [], "stdout_bytes": 0, "stderr_bytes": 0}
        timings = {"stage_in": 0.0, "run": 0.0, "stage_out": 0.0}
        for key in ("stdout", "stderr"):
            paths[key].write_bytes(b"")
        started_utc = _utc_now()
        try:
            self._stage_in(job, names, timings)
            self._run_container(job, names, create_argv, paths, state, timings)
            self._stage_out(job, names, state, timings)
        except PodmanError as error:
            state["status"], state["detail"] = INFRA_ERROR, str(error)
        except BaseException:
            self._cleanup(names)
            raise
        cleanup_errors = self._cleanup(names)
        if state["status"] is None:
            state["status"] = classify(**state["facts"])
        receipt = Receipt(
            job_id=job.job_id, run_id=self.run_id, status=state["status"], detail=state["detail"],
            exit_code=state["exit_code"], image=job.image, image_digest=image["Digest"], image_id=image["Id"],
            image_architecture=image["Architecture"], container_name=names["job"],
            started_utc=started_utc, finished_utc=_utc_now(),
            stage_in_s=timings["stage_in"], run_s=timings["run"], stage_out_s=timings["stage_out"],
            caps=json.loads(json.dumps(job.caps.__dict__)), command=list(job.command), create_argv=create_argv,
            effective_policy=state["effective"], mounts=state["mounts"],
            stdout_path=str(paths["stdout"]), stderr_path=str(paths["stderr"]),
            stdout_bytes=state["stdout_bytes"], stderr_bytes=state["stderr_bytes"],
            outputs=state["outputs"], rejected_outputs=state["rejected"], cleanup_errors=cleanup_errors)
        receipt.write(paths["receipt"])
        return receipt

    def _stage_in(self, job, names, timings):
        begin = time.monotonic()
        for volume in (names["in"], names["out"]):
            self.podman.run(policy.volume_create_argv(volume, self.run_id, job.job_id))
        argv = policy.stager_run_argv(job, self.run_id, names["stagein"], names["in"], policy.IN_MOUNT, False,
                                      staging.STAGE_IN_SCRIPT)
        with self.podman.popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            try:
                staging.write_input_tar(proc.stdin, job.input_dir)
            finally:
                proc.stdin.close()
            stdout, stderr = proc.stdout.read(), proc.stderr.read()
            if proc.wait() != 0:
                raise PodmanError(argv, proc.returncode, (stderr or stdout).decode(errors="replace"))
        timings["stage_in"] = time.monotonic() - begin

    def _run_container(self, job, names, create_argv, paths, state, timings):
        self.podman.run(create_argv)
        begin = time.monotonic()
        with self.podman.popen(["start", "--attach", names["job"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            pumps = [_Pump(proc.stdout, paths["stdout"], job.caps.max_log_bytes),
                     _Pump(proc.stderr, paths["stderr"], job.caps.max_log_bytes)]
            for pump in pumps:
                pump.start()
            timed_out = False
            try:
                proc.wait(timeout=job.caps.wall_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.podman.run(["kill", names["job"]])
                proc.wait(timeout=60)
            for pump in pumps:
                pump.join()
        timings["run"] = time.monotonic() - begin
        state["stdout_bytes"], state["stderr_bytes"] = pumps[0].total, pumps[1].total
        info = self.podman.inspect("container", names["job"])
        container = info["State"]
        state["exit_code"] = container["ExitCode"]
        state["mounts"] = [{k: m.get(k) for k in ("Type", "Name", "Source", "Destination", "RW")} for m in info["Mounts"]]
        host = info["HostConfig"]
        state["effective"] = {
            "network_mode": host["NetworkMode"], "read_only_rootfs": host["ReadonlyRootfs"],
            "cap_drop": host["CapDrop"], "security_opt": host["SecurityOpt"], "privileged": host["Privileged"],
            "pids_limit": host["PidsLimit"], "memory": host["Memory"], "memory_swap": host["MemorySwap"],
            "nano_cpus": host.get("NanoCpus"), "user": info["Config"]["User"], "tmpfs": host.get("Tmpfs"),
            "oom_killed": container["OOMKilled"]}
        state["facts"] = dict(
            timed_out=timed_out, oom_killed=container["OOMKilled"],
            started=not container["StartedAt"].startswith("0001-"),
            runtime_error=container.get("Error", ""), exit_code=container["ExitCode"])
        state["detail"] = "wall cap %ss exceeded" % job.caps.wall_s if timed_out else container.get("Error", "")

    def _stage_out(self, job, names, state, timings):
        begin = time.monotonic()
        argv = policy.stager_run_argv(job, self.run_id, names["stageout"], names["out"], policy.OUT_MOUNT, True,
                                      staging.STAGE_OUT_SCRIPT)
        over_limit = False
        with self.podman.popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            errors = []
            reader = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
            reader.start()
            try:
                state["outputs"], state["rejected"] = staging.read_output_tar(
                    proc.stdout, job.output_dir, job.caps.max_output_bytes)
            except staging.OutputLimitExceeded as error:
                over_limit = True
                proc.kill()
                state["rejected"].append({"path": "*", "reason": str(error)})
                if classify(**state["facts"]) == OK:  # a timeout or OOM remains the better explanation
                    state["status"], state["detail"] = CANDIDATE_ERROR, str(error)
            reader.join()
            proc.wait()
        if proc.returncode != 0 and not over_limit:
            raise PodmanError(argv, proc.returncode, b"".join(errors).decode(errors="replace"))
        timings["stage_out"] = time.monotonic() - begin

    def _cleanup(self, names):
        errors = []
        steps = [["rm", "--force", "--ignore", names[kind]] for kind in ("stagein", "stageout", "job")]
        steps += [["volume", "rm", "--force", names[kind]] for kind in ("in", "out")]
        for argv in steps:
            try:
                self.podman.run(argv)
            except PodmanError as error:
                errors.append(str(error))
        return errors

