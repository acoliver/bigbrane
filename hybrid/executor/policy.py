"""Launch policy: the exact podman arguments for job and staging containers.

Pure functions, no I/O, so the policy is reviewable and unit-testable. Every
resource carries the labels required by PODMAN-LIFECYCLE.md."""
OWNER_LABEL = ("bigbrane.owner", "bigbrane")
JOB_UID = 10001
JOB_GID = 10001
IN_MOUNT = "/in"
OUT_MOUNT = "/out"
TMP_MOUNT = "/tmp"
SHM_MB = 16
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def label_args(run_id, job_id):
    pairs = [OWNER_LABEL, ("bigbrane.run", run_id), ("bigbrane.job", job_id)]
    return [arg for key, value in pairs for arg in ("--label", "%s=%s" % (key, value))]


def volume_create_argv(volume, run_id, job_id):
    return ["volume", "create", *label_args(run_id, job_id),
            "--uid", str(JOB_UID), "--gid", str(JOB_GID), volume]


def _restrictions(memory_mb, pids, cpus):
    return [
        "--pull=never", "--network=none", "--read-only", "--read-only-tmpfs=false",
        "--user", "%d:%d" % (JOB_UID, JOB_GID),
        "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--ipc=private", "--shm-size=%dm" % SHM_MB,
        "--pids-limit", str(pids),
        "--memory", "%dm" % memory_mb, "--memory-swap", "%dm" % memory_mb,
        "--cpus", str(cpus),
        # Podman copies the VM's /usr/share/rhel/secrets (Red Hat CA bundles and rhsm.conf) into every
        # container at /run/secrets. An explicit empty read-only tmpfs masks it.
        "--tmpfs", "/run/secrets:ro,noexec,nosuid,nodev,size=4k",
    ]


def _volume_mount(volume, destination, read_only):
    spec = "type=volume,source=%s,destination=%s" % (volume, destination)
    return ["--mount", spec + (",ro=true" if read_only else "")]


def job_create_argv(job, run_id, container, in_volume, out_volume):
    caps = job.caps
    argv = ["create", "--name", container, *label_args(run_id, job.job_id),
            *_restrictions(caps.memory_mb, caps.pids, caps.cpus),
            "--tmpfs", "%s:rw,noexec,nosuid,nodev,size=%dm,mode=1777" % (TMP_MOUNT, caps.tmp_mb),
            *_volume_mount(in_volume, IN_MOUNT, True),
            *_volume_mount(out_volume, OUT_MOUNT, False),
            "--workdir", OUT_MOUNT]
    for key, value in THREAD_ENV.items():
        argv += ["--env", "%s=%s" % (key, value)]
    return [*argv, job.image, *job.command]


def stager_run_argv(job, run_id, container, volume, destination, read_only, script, memory_mb=256, pids=32):
    """A short-lived restricted container that moves a tar stream in or out of one volume."""
    return ["run", "--rm", "-i", "--name", container, *label_args(run_id, job.job_id),
            *_restrictions(memory_mb, pids, 1),
            *_volume_mount(volume, destination, read_only),
            "--entrypoint", "python3", job.image, "-c", script]
