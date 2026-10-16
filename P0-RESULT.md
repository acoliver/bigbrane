# P0 result: restricted Linux arm64 executor

**P0 does not close. Seven of the eight probes pass; probe 8 fails.** The executor works and enforces its limits on Podman 5.7.1 in the rootless applehv VM. The trusted fixture does not reproduce bit-identically inside the container: the RNN KL matches exactly, the HMM KL differs by 10 ulp (1.39e-16). The plan's rule for this case applies: the drift is documented, and all P4 and P5 scoring runs in the container only. The probe 8 assertion was not loosened and is still failing in the final run.

Run id `p0-20261009T204723Z`. Executor image `localhost/bigbrane/executor@sha256:7fa94b73a3f6fbd6b6544d075e24ff8bf28fae2662771e577a332348d4a00f1b` (linux/arm64, base `python:3.12-slim-bookworm` 3.12.15 pinned by arm64 digest `sha256:739ba32ae445e8d58f3d90feb85f83bebc8346f8dd280fa1eb5848f4ff1ed163`). Usage and policy: [hybrid/executor/README.md](hybrid/executor/README.md).

Final test run: 45 tests, 44 pass, 1 fails (`Probe8Fixture`). Log: `results/p0-20261009T204723Z/final/unittest-full.log`, per-test verdicts in `results/p0-20261009T204723Z/final/test-verdicts.txt`. Earlier runs, with the failures that led to policy changes, are `results/p0-20261009T204723Z/test-probes-run1.log` and `test-probes-run2.log`.

## Probe table

Evidence paths are relative to `results/p0-20261009T204723Z/probes/`. Each job directory holds `in/` and `out/` (`result.json` is the observation the test asserts on); receipts are in `receipts/<job>.receipt.json`, with stdout and stderr beside them.

| # | Probe | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Architecture aarch64, image digest recorded | pass | `p1-arch/out/result.json` (machine aarch64, interpreter ELF machine 183), `receipts/p1-arch.receipt.json` (arch arm64, digest `sha256:7fa94b73...`) |
| 2 | No network: DNS, public IPs, host gateway, metadata | pass | `p2-network/out/result.json`: every attempt failed, loopback control worked, only `lo` present. A control container with the network enabled reached pypi.org and 1.1.1.1 (`Probe2Network.test_probe_detects_connectivity...`) |
| 3 | No reads outside mounts, holdout absent | pass | `p3-filesystem/out/result.json`, `receipts/p3-filesystem.receipt.json` (mounts: two VM volumes only). Host holdout file in `holdout-fixture/` was unreachable and its name was not found anywhere in the container |
| 4 | Read-only inputs and root, `/out` and `/tmp` writable, `/tmp` capped | pass | `p4-writes/out/result.json` (EROFS on `/in` and root, `/tmp` stopped at 63 MB with ENOSPC), receipt output hash for `declared_output.txt` |
| 5a | Fork bomb stops at pids cap, classified `pids` | pass | `receipts/p5-pids-raise.receipt.json`, `p5-pids-catch.receipt.json` (status pids, exit 254, limit 32) |
| 5b | Memory hog killed, classified `oom` | pass | `receipts/p5-memory.receipt.json` (status oom, exit 137, OOMKilled true, 128 MB, swap equal) |
| 5c | Sleep past wall cap, classified `timeout` | pass | `receipts/p5-wall.receipt.json` (status timeout, run 4.14 s for a 4 s cap, container removed) |
| 6 | Non-root, zero capabilities, no Podman socket | pass | `p6-identity/out/result.json` (uid 10001, all capability sets zero, NoNewPrivs 1, no sockets, no credential files, `/run/secrets` empty) |
| 7 | One thread measured | pass | `p7-threads/out/result.json` (NumPy and PyTorch matmul: no worker threads; control with 4 threads: 3 workers) |
| 8 | Fixture bit-identical in container | **fail** | `p8-fixture/out/record.json` against the native run; diagnosis in `diag/` |

Other executor behavior tested in `ExecutorContract`: candidate error keeps its exit code, missing command, output hashes and timings, log truncation with full byte counts, oversized output, an injected Podman failure giving `infra_error` with cleanup, tag references and wrong digests refused, symlinks in input refused before anything runs. Receipts: `receipts/c-*.receipt.json`; summary table `final/receipts-summary.txt`.

## Probe 8

| | HMM KL | RNN KL |
| --- | --- | --- |
| Expected and native (macOS arm64, NumPy 2.5.3, Accelerate) | 0.12230803167784307 | 0.005312276268561993 |
| Container (Linux arm64, NumPy 2.5.3, scipy-openblas 0.3.34) | 0.12230803167784293 | 0.005312276268561993 |

Data digest, oracle matrix and RNN predictions are bit-identical. 338 of 512 HMM predictions differ by at most one ulp. The cause is the matrix-vector product `A @ v` in the fixture's EM backward pass: Accelerate and OpenBLAS round it differently. With that one line replaced by an elementwise sum, native and container both give `0x1.f4f9444182f0ep-4` (`diag/cause-*.json`, kept under test in `Probe8Diagnosis`). `Generator.uniform` also differs in raw bits between the platforms (`diag/uniform-*.json`), but the fixture's initial HMM parameters come out the same, so it is not the cause.

Consequences for the plan:
- Native and container numbers must not be mixed. Any KL that feeds a verdict gets computed from container predictions by one evaluator build, or the whole replay runs in the container. P1 acceptance test 2 (reproduce HMM KL 0.12230803167784307 through `submit_run`) cannot hold bitwise if the fit runs in the container; it needs the container baseline 0.12230803167784293, or a documented tolerance chosen by the user.
- The evaluator on the host reads container-written predictions as data, so cross-platform drift enters only through candidate code that runs in the container. Candidates should avoid relying on BLAS rounding when they need cross-platform equality.
- This is a user decision point only if bit-level host equality is required; otherwise the plan's stated response applies.

## What else the probes found

- Podman copies the VM's `/usr/share/rhel/secrets` (Red Hat CA files, `rhsm.conf`) into every container at `/run/secrets`. The policy masks it with an empty read-only tmpfs.
- `/dev/shm` is read-only under `--read-only`, so POSIX shared memory is unavailable to jobs.
- `host.containers.internal` resolves through Podman's `/etc/hosts` even with no network; every connection to it fails.
- `--cpus=1` is a quota (`cpu.max 100000 100000`); the affinity mask still shows 8 CPUs, so the one-thread setting is what limits BLAS, and probe 7 measures it.
- `pids` classification depends on a supervisor exit code, so a job that kills its own supervisor is classified `candidate_error` instead.
- `/out` has no disk quota; the host caps what it copies back. `requirements.txt` pins versions, not wheel hashes.
- First image build pulled CUDA wheels for torch from PyPI (aarch64 wheels there are CUDA builds). It was stopped and rebuilt with the PyTorch CPU index; no partial images remained.

## Podman state

Every resource created carried `bigbrane.owner=bigbrane` and `bigbrane.run=p0-20261009T204723Z`, except the registry pull noted below. Nothing was pruned, reset or reconfigured; no unlabeled resource was removed or modified.

- Kept: image `localhost/bigbrane/executor:p0-20261009T204723Z` (id `4ac1b48d2802`, labeled, 1 GB). It is the executor image the tests use.
- Removed: all job, staging and control containers (`--rm` or removed by the executor), all job volumes. At the end there were no labeled containers, volumes or pods (`final/podman-state-before-stop.txt`).
- Not removable and not labeled: base image `docker.io/library/python@sha256:739ba32a...` (id `eb7d16fec874`, arm64, 162 MB), pulled by the build. A registry pull cannot carry our labels, and it is the parent of the executor image. It was left in place; `scripts/podman-teardown.sh` does not touch it. The other 38 images present before this task are unchanged.
- `scripts/podman-teardown.sh --run p0-20261009T204723Z` in dry-run mode listed exactly one item, `podman rmi 4ac1b48d2802` (the kept image), and no containers, volumes or pods (`final/teardown-dryrun-run.log`). The unscoped dry run gave the same result (`final/teardown-dryrun-all.log`).
- VM: `podman machine stop podman-machine-default` succeeded. `podman machine inspect` reports State `stopped`; `podman info` exits 125 (`final/machine-state-after-stop.txt`, `final/machine-list-after-stop.txt`). `podman machine list` shows "Less than a second ago" under LAST UP, which per PODMAN-LIFECYCLE.md means stopped.
- To remove the kept image later: start the VM, then `scripts/podman-teardown.sh --run p0-20261009T204723Z --apply`.

## Not touched

`bigbrane/`, `tests/`, `EXPERIMENT.md`, `README.md` and old results were not modified (no file under `bigbrane/` or `tests/` is newer than the start of this run). New code is under `hybrid/`. The host `.venv` got no new packages. No GitHub actions and no OCR.
