# Podman lifecycle for Bigbrane

Bigbrane runs its container work on the Podman machine `podman-machine-default` (Apple hypervisor VM, 8 CPUs, 24 GiB memory, 100 GiB disk, rootless connection on 127.0.0.1:53342). This file records how to start it, check it, stop it, and remove everything Bigbrane created. Logs for the inventory below are in `results/podman-lifecycle-20261009T203425Z/`.

## Commands

| Goal | Command |
| --- | --- |
| Start the VM | `podman machine start podman-machine-default` |
| Check status | `podman machine list` (the running machine shows "Currently running" under LAST UP) |
| Confirm the engine answers | `podman info` |
| Stop the VM | `podman machine stop podman-machine-default` |
| Inspect state field | `podman machine inspect podman-machine-default` (look at `State`) |

Do not run `podman machine rm`, `podman machine reset`, `podman system reset`, or `podman system prune`. They delete resources that Bigbrane did not create, including the llxprt sandbox images.

The stop and start path was exercised on this machine: stop succeeded, start succeeded, and `podman info` returned afterwards (see `lifecycle-verify.log`). Right after a stop, `podman machine list` shows a recent timestamp under LAST UP instead of "Currently running". That timestamp means stopped; `podman machine inspect` `State` gives the explicit value.

## Label convention

Every container, image, volume, and pod that Bigbrane creates must carry both labels:

- `bigbrane.owner=bigbrane`
- `bigbrane.run=<run-id>` (for example `p0-20261009T180000Z`)

Examples:

```
podman run --rm --label bigbrane.owner=bigbrane --label bigbrane.run=$RUN_ID IMAGE CMD
podman build --label bigbrane.owner=bigbrane --label bigbrane.run=$RUN_ID -t bigbrane/NAME:$RUN_ID .
podman volume create --label bigbrane.owner=bigbrane --label bigbrane.run=$RUN_ID NAME
podman pod create --label bigbrane.owner=bigbrane --label bigbrane.run=$RUN_ID --name NAME
```

Prefer `--rm` for one-shot containers so nothing is left behind when a run ends.

## Cleaning up Bigbrane artifacts

List first:

```
podman ps -a --filter label=bigbrane.owner=bigbrane
podman images --filter label=bigbrane.owner=bigbrane
podman volume ls --filter label=bigbrane.owner=bigbrane
podman pod ls --filter label=bigbrane.owner=bigbrane
```

Remove (labels in the filter keep this away from anything else on the machine):

```
podman pod rm -f $(podman pod ls -q --filter label=bigbrane.owner=bigbrane)
podman rm -f $(podman ps -aq --filter label=bigbrane.owner=bigbrane)
podman rmi $(podman images -q --filter label=bigbrane.owner=bigbrane)
podman volume rm $(podman volume ls -q --filter label=bigbrane.owner=bigbrane)
podman image prune --filter label=bigbrane.owner=bigbrane
```

Add `--filter label=bigbrane.run=<run-id>` to any of these to scope to one run. `podman image prune` without `-a` only removes dangling images, and the label filter limits it to Bigbrane images. Dangling layers from builds that lack labels are not matched; label builds with `--label` so they are.

### Script

`scripts/podman-teardown.sh` does the label-filtered removal above.

```
scripts/podman-teardown.sh                          # dry run, prints only
scripts/podman-teardown.sh --run <run-id>           # dry run for one run
scripts/podman-teardown.sh --apply                  # remove all labeled Bigbrane resources
scripts/podman-teardown.sh --apply --stop-machine   # also stop the VM
```

It uses `set -euo pipefail`, defaults to dry run, never queries unlabeled resources, and exits with an error if the VM is unreachable. With `--apply --stop-machine` it refuses to stop the VM while any container is still running. Tested in dry run against the live machine (`script-dryrun.log`). It was also checked with a throwaway labeled container and an unlabeled one: only the labeled one was listed and removed, and the unlabeled one was then removed by hand.

### Label audit, retention and the free-space precheck (2026-10-12)

- `scripts/podman-label-audit.sh` (read only) lists every resource labeled `bigbrane.owner=bigbrane`, then anything our builds made that lacks the label: images named `localhost/bigbrane/*` without it, images with other `bigbrane.*` labels, and build containers left by an interrupted `podman build` (`podman ps --external`) on our images or our pinned base. The pinned base image (`python@sha256:739ba32a...`) is listed for information; builds pull it and it is never removed by Bigbrane tooling. Exit 1 when something unlabeled of ours exists.
- `scripts/retention.sh` (dry run by default, `--apply` to act) removes `tmp/pytest-*` older than a day, `results/bench-tests-*` beyond the newest 3, `__pycache__` under the root, `hybrid/` and `scripts/`, `tmp/**/.lake` older than a day, and tools images beyond the newest 2, chosen only through `label=bigbrane.owner=bigbrane` and `label=bigbrane.image=tools` and never the image in `image.tools.lock.json`. It never touches `runs/`, `reports/`, `data/` or any other `results/` entry.
- `python -m hybrid.diskguard` prints free space on the workspace volume (threshold 20 GB), the internal disk holding `~/.local/share/containers` (30 GB) and the VM's `/var` (20 GB). `python -m hybrid.research run` and `plumb` refuse below the first; `python -m hybrid.executor.build` refuses below the other two.
- Images are built with `python -m hybrid.executor.build {executor,tools} --run <run id>`, which adds the owner, run and image labels and `--layers=false`.
- `hybrid/tests/test_podman_hygiene.py` scans our code and fails on `system prune`, `system reset`, `machine rm`, `machine reset`, `prune -a`, an `image prune` without the owner label filter, or an image removal outside the teardown and retention scripts.

Disk effect of the tools image build (2026-10-12): free space on the internal disk fell from 157.2 GB to 119.4 GB during the build, while free space inside the VM fell from 79.5 GB to 67.2 GB. After the build the VM disk file `podman-machine-default-arm64.raw` occupies 65.5 GiB on the internal disk (`du`), while the VM reports about 32 GB in use, so blocks freed inside the VM (the build's temporary files) were not returned to macOS. Its size before the build was not measured, and other processes share the internal disk, so the whole 37.8 GB drop is not attributed to the build. `fstrim` inside the VM would likely return the freed blocks; it was not run.

## End-of-session teardown checklist

1. `podman ps` shows nothing Bigbrane started still running.
2. `scripts/podman-teardown.sh --run <run-id>` (dry run), read the list, then repeat with `--apply`.
3. `podman ps -a --filter label=bigbrane.owner=bigbrane` and the image, volume, and pod equivalents return nothing.
4. If nothing else uses the VM (`podman ps` empty and no other project needs it), stop it: `podman machine stop podman-machine-default`, then check `podman machine list`.
5. Check disk: `df -h /`, `df -h /Volumes/XS1000`, and `podman system df` while the VM is up.
6. Remove Bigbrane host outputs that are no longer needed from `results/` and `platform-trials/` (see sizes below). Keep run summaries.

## What the inventory found

Host (before starting the VM):

- Podman 5.7.1. `podman-machine-default` was **stopped**, created 2025-12-21, last up 2026-10-06. Two connections exist (rootless default, and root).
- The machine disk image `~/.local/share/containers/podman/machine/applehv/podman-machine-default-arm64.raw` is a sparse 100 GiB file; about 57 GiB is allocated per `ls -s` and `du` reports 29 GiB for the directory.
- Free space: `/` has 244 GiB free (5% used); `/Volumes/XS1000` has 323 GiB free (83% used).
- `~/.llxprt-code-cap-*` directories: 0 present. `telemetry.jsonl`: 528 KiB. `~/Library/Logs/llxprt-code/tmp/`: 2.8 GiB (not touched).
- Bigbrane: `results/` 1.0 GiB, `platform-trials/` 4.0 GiB, `.pytest_cache` 16 KiB.

Inside the VM (after starting it):

| Kind | Count | Notes |
| --- | --- | --- |
| Containers | 0 | Nothing left running or exited. |
| Pods | 0 | |
| Volumes | 0 | |
| Images | 38 (11.09 GB, all reported reclaimable because no container uses them) | None carry a Bigbrane label. |

Image classification:

- **llxprt sandbox** (not Bigbrane): `ghcr.io/vybestack/llxprt-code/sandbox` tags `0.11.0` and `0.11.0-nightly.260906.72ea679a5` (3.28 GB each), plus about 28 untagged build-layer images from 2026-09-02 (253 MB to 826 MB) that are the intermediate layers of those sandbox builds. They are labeled only `io.buildah.version`.
- **Another project's Python build** (not Bigbrane): `docker.io/library/python:3.14-slim` (a linux/amd64 image, 124 MB) and 4 untagged layers from 2026-10-06. The layers set `WORKDIR /app`, `AUTH_MODE=okta-proxy`, a job storage directory, and a pandoc install, which are not Bigbrane settings.
- **Base images** (shared, unknown owner): `docker.io/library/node:24-slim`, `docker.io/library/rust:1.98-bookworm` (1.49 GB).
- **Dangling** per `podman images --filter dangling=true`: `e076c0890c8d` (python layer, 2026-10-06) and `14ef90d86b4a` (sandbox layer, 2026-09-02). Both belong to other builds, so they were left alone.

## Cleanup performed

Nothing from the original inventory was removed. The VM held no containers, pods, or volumes, and every image is of non-Bigbrane or unknown origin. The only resources created and removed were two throwaway containers used to test the teardown script (`bigbrane-selftest`, labeled, removed by the script; `unlabeled-selftest`, removed by hand).

## Recommended cleanup (needs the user's approval)

These are not Bigbrane resources, so nothing was run. Review before running any of them.

- Dangling images only (safe in the sense that nothing references them, but they belong to other builds):
  `podman rmi e076c0890c8d 14ef90d86b4a`
- Old llxprt sandbox build layers and the older nightly tag, if the llxprt sandbox is no longer needed:
  `podman rmi ghcr.io/vybestack/llxprt-code/sandbox:0.11.0-nightly.260906.72ea679a5`
  `podman image prune` (dangling images only, unlabeled, affects other projects, run only with approval)
- Host-side: `~/Library/Logs/llxprt-code/tmp/` (2.8 GiB) accumulates session directories and can be cleared by the user. Bigbrane `platform-trials/` (4.0 GiB) and `results/` (1.0 GiB) can be trimmed once their summaries are recorded.
