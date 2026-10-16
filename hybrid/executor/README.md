# Bigbrane P0 executor

A host-side Python API that runs one job in a fresh, restricted Linux arm64 container on the Podman VM
`podman-machine-default` and returns a receipt. This is Phase P0 of [HYBRID-PLAN.md](../../HYBRID-PLAN.md).
Run id of the build described here: `p0-20261009T204723Z`. Results: [P0-RESULT.md](../../P0-RESULT.md).

## Files

| Path | Purpose |
| --- | --- |
| `Containerfile`, `requirements.txt`, `supervisor.py` | Image: python:3.12-slim-bookworm pinned by its linux/arm64 manifest digest, NumPy 2.5.3, CPU PyTorch 2.14.1 (from the PyTorch CPU index), user 10001, one-thread environment. `supervisor.py` is PID 1 and reports pids-limit hits through its exit code. |
| `image.lock.json` | Executor image reference, digest, image id, base digest, labels. Written by `python -m hybrid.executor.lock <tag>`. |
| `Containerfile.tools`, `requirements.tools.txt`, `tools/`, `image.tools.lock.json` | The tools image (Lean 4 with Mathlib, z3-solver, SymPy, NumPy): same base, user, supervisor and launch policy as the executor image. See [hybrid/math/README.md](../math/README.md). `read_lock("tools")` gives its reference. |
| `build.py` | `python -m hybrid.executor.build {executor,tools} --run <run id>`: free-space precheck (`hybrid/diskguard.py`; exit 2 below 30 GB on the internal disk or 20 GB on the VM disk), then `podman build --platform linux/arm64 --layers=false` with the `bigbrane.owner`, `bigbrane.run` and `bigbrane.image` labels, then the lock file. |
| `audit.py` | Read-only label audit, run by `scripts/podman-label-audit.sh`. |
| `job.py` | `Caps` and `Job`. Invalid jobs fail at construction (tag-only image, non-empty output dir, non-positive caps). |
| `policy.py` | Pure functions that build every `podman` argument list. The launch policy lives here and nowhere else. |
| `staging.py` | Tar streaming of inputs into and outputs out of VM-local volumes. Refuses symlinks and special files in both directions. |
| `executor.py` | `Executor.run(job)`: verify image, stage in, create and start, enforce the wall cap, stage out, clean up, write the receipt. |
| `receipt.py` | `Receipt` and `classify`. |
| `podman.py` | Thin CLI wrapper. |
| `../probes/` | Probe programs that run inside the container and only observe. `diag_*.py` are diagnostics for the probe 8 drift. |
| `../fixtures/trial_fixture.py` | Byte-identical copy of `platform-trials/20261008T003451/fixture.py` (sha256 `e099ea0b2ac8...910293`). |
| `../tests/` | `test_policy.py` (no Podman needed), `test_probes.py` (needs the VM and the image). |

## Usage

```python
from hybrid.executor.executor import Executor
from hybrid.executor.job import Caps, Job
from hybrid.executor.lock import read_lock

executor = Executor(run_id="p0-example", receipts_dir="results/p0-example/receipts")
job = Job(
    job_id="fit-001",
    image=read_lock()["reference"],            # name@sha256:<digest>, tags are refused
    input_dir="work/fit-001/in",               # mounted read-only at /in
    output_dir="work/fit-001/out",             # must exist and be empty; filled from /out afterwards
    command=("python3", "/in/candidate_runner.py"),
    caps=Caps(wall_s=900, memory_mb=2048, pids=256, cpus=1.0, tmp_mb=64),
)
receipt = executor.run(job)                    # also written to receipts_dir/fit-001.receipt.json
print(receipt.status, receipt.outputs)
```

### Launch policy applied to every job

`--network=none`, `--read-only` rootfs, `--read-only-tmpfs=false`, `--user 10001:10001`, `--cap-drop=ALL`,
`--security-opt=no-new-privileges`, `--pids-limit`, `--memory` with `--memory-swap` equal to it, `--cpus`, `--pull=never`,
`--ipc=private`, `--workdir /out`, thread variables set to 1. Mounts are exactly: VM-local volume at `/in` (read-only),
VM-local volume at `/out` (writable), tmpfs `/tmp` (noexec, size capped by `tmp_mb`), and an empty read-only tmpfs over
`/run/secrets` that masks a Podman default (see Findings). No bind mounts, no Podman socket, no credentials, nothing from `/Volumes`.
Every container, volume and staging container carries `bigbrane.owner=bigbrane`, `bigbrane.run=<run id>`, `bigbrane.job=<job id>`.
The wall timeout is enforced by the host (`podman kill` after `wall_s`).

Inputs reach the job through a short-lived staging container that runs the same image under the same restrictions, with a
tar stream on stdin. Outputs come back the same way and are written by host code that accepts only regular files and directories;
anything else (for example a symlink the job created in `/out`) is listed in `rejected_outputs` and never materialized.

### Receipt

`status` is one of `ok`, `candidate_error`, `timeout`, `oom`, `pids`, `infra_error`. Other fields: `exit_code`, `detail`, image reference, digest, image id and
architecture, container name, UTC start and finish, `stage_in_s`, `run_s`, `stage_out_s`, caps, command, the exact `create_argv`,
`effective_policy` and `mounts` read back from `podman inspect`, stdout and stderr paths and byte counts (files are truncated at
`max_log_bytes`, counts are not), per-file sha256 and size of outputs, `rejected_outputs`, `cleanup_errors`.

Classification order: engine or start failure is `infra_error`; host wall kill is `timeout`; kernel OOM kill (`State.OOMKilled`) is `oom`;
supervisor exit 254 (cgroup `pids.events` counted a limit hit) is `pids`; supervisor exit 253 is `infra_error`; exit 0 is `ok`; anything else is `candidate_error`.
A job can kill the supervisor, which only turns its own `pids` into `candidate_error`.

## Running the tests

```sh
podman machine start podman-machine-default
.venv-hybrid/bin/python -m hybrid.executor.build executor --run $RUN   # precheck, labeled build, lock
BIGBRANE_P0_RUN=$RUN .venv/bin/python -m unittest hybrid.tests.test_policy hybrid.tests.test_probes -v
```

Evidence is written under `results/$RUN/probes/<job id>/{in,out}` and `results/$RUN/probes/receipts/`. The module teardown fails
the run if any labeled container, volume or pod of the run id is left behind.

## Probes and results

| # | Probe (test class) | Result |
| --- | --- | --- |
| 1 | `Probe1Architecture`: `uname -m` aarch64, ELF machine of the interpreter is 183, image arch arm64, digest recorded, base pinned by arm64 digest | pass |
| 2 | `Probe2Network`: DNS, UDP DNS, TCP to two public IPs, 169.254.169.254, the addresses behind `host.containers.internal`, Podman bridge gateway; loopback works as a control; only `lo` exists; a control container with the network enabled reaches the internet | pass |
| 3 | `Probe3Filesystem`: host holdout file unreachable by absolute path and by `/in/../..`; ancestors absent; whole-filesystem walk finds no holdout name; no virtiofs mounts; receipt mounts hold only the two volumes; symlinks in output are rejected; symlinks in input are refused | pass |
| 4 | `Probe4Writes`: `/in` and image root read-only (EROFS), `/out` and `/tmp` writable, `/tmp` stops at 63 of 64 MB (ENOSPC), nothing executes from `/tmp`, supervisor cannot be overwritten | pass |
| 5 | `Probe5Caps`: fork bomb gives `pids` (exit 254) with and without the job catching the failure; memory hog gives `oom` (exit 137, OOMKilled); sleep gives `timeout` and the container is gone | pass |
| 6 | `Probe6Identity`: uid and gid 10001, all five capability sets zero, NoNewPrivs 1, setuid(0) and mount() fail, no sockets, no Podman or Docker socket paths, no credential files, no secrets in the environment | pass |
| 7 | `Probe7Threads`: a BLAS matmul runs with no worker threads beyond the calling thread for NumPy and PyTorch (`torch.get_num_threads()` is 1); a control child with 4 threads shows 3 workers; `cpu.max` is `100000 100000` | pass |
| 8 | `Probe8Fixture`: trusted fixture HMM KL 0.12230803167784307 and RNN KL 0.005312276268561993 bit-identical to the native run | **fail** (RNN identical, HMM KL differs) |

### Probe 8 in detail

| | HMM KL | RNN KL |
| --- | --- | --- |
| Native (macOS arm64, NumPy 2.5.3, Accelerate) | 0.12230803167784307 (`0x1.f4f9444182f0ep-4`) | 0.005312276268561993 |
| Container (Linux arm64, NumPy 2.5.3, scipy-openblas 0.3.34) | 0.12230803167784293 (`0x1.f4f9444182f04p-4`) | 0.005312276268561993 |

The data digest, the oracle matrix and the RNN predictions are bit-identical. 338 of 512 HMM predictions differ by at most one unit in the last place,
which moves the HMM KL by 1.39e-16 (10 ulp). The cause was isolated with `hybrid/probes/diag_confirm_cause.py`: replacing only the
matrix-vector product `A @ v` in the fixture's EM backward pass with an elementwise sum makes native and container agree on `0x1.f4f9444182f0ep-4`.
`Probe8Diagnosis` keeps both facts under test. The strict bit-identity assertion in `Probe8Fixture` is unchanged and fails.
An earlier suspect, `Generator.uniform` (fused multiply-add on macOS), does differ in raw bits but does not change the fixture's initial HMM parameters (diagnostic in `results/p0-20261009T204723Z/diag/`).

## Findings that shaped the policy

- Podman copies the VM's `/usr/share/rhel/secrets` (Red Hat CA bundles, `rhsm.conf`) into every container at `/run/secrets`. Probe 6 found it; an explicit empty read-only tmpfs masks it.
- With `--read-only`, `/dev/shm` is read-only too. Candidates cannot use POSIX shared memory.
- `host.containers.internal` resolves from Podman's `/etc/hosts` even with `--network=none`. The name resolving does not give a route, and every connection to its address fails.
- The python base image exports `GPG_KEY`, a public signing-key fingerprint. The test allows exactly that variable.
- `--cpus=1` is a CPU quota (`cpu.max`). The process still sees 8 CPUs in its affinity mask, which is why thread counts are set to 1 explicitly and measured.
- `requirements.txt` pins versions but not wheel hashes.
- `/out` is a plain volume with no disk quota. The host caps what it copies back (`max_output_bytes`); a job can still fill the VM disk until its wall cap ends it.
