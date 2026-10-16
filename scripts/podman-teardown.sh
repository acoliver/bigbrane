#!/usr/bin/env bash
# Label-filtered teardown of Bigbrane podman artifacts.
# Default is --dry-run: prints what would happen and changes nothing.
# Only resources labeled bigbrane.owner=bigbrane are ever touched.
#
# Usage:
#   scripts/podman-teardown.sh [--dry-run | --apply] [--run <run-id>] [--stop-machine]
set -euo pipefail

OWNER_LABEL="bigbrane.owner=bigbrane"
MACHINE="podman-machine-default"
APPLY=0
RUN_ID=""
STOP_MACHINE=0

usage() {
  sed -n "2,7p" "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) APPLY=0 ;;
    --apply) APPLY=1 ;;
    --stop-machine) STOP_MACHINE=1 ;;
    --run)
      [ $# -ge 2 ] || { echo "error: --run needs a run id" >&2; exit 2; }
      RUN_ID="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

command -v podman >/dev/null || { echo "error: podman not found" >&2; exit 1; }

# Fail fast if the VM is not reachable. Nothing to clean when it is down,
# and starting it here would defeat the purpose of a teardown.
podman info >/dev/null 2>&1 || { echo "error: podman machine is not reachable (is $MACHINE running?)" >&2; exit 1; }

FILTERS=(--filter "label=$OWNER_LABEL")
if [ -n "$RUN_ID" ]; then
  FILTERS+=(--filter "label=bigbrane.run=$RUN_ID")
fi

if [ "$APPLY" -eq 1 ]; then MODE="APPLY"; else MODE="DRY-RUN"; fi
echo "mode: $MODE  scope: $OWNER_LABEL${RUN_ID:+ bigbrane.run=$RUN_ID}"

CONTAINERS=$(podman ps -a "${FILTERS[@]}" --format '{{.ID}}')
IMAGES=$(podman images "${FILTERS[@]}" --format '{{.ID}}' | sort -u)
VOLUMES=$(podman volume ls "${FILTERS[@]}" --format '{{.Name}}')
PODS=$(podman pod ls "${FILTERS[@]}" --format '{{.ID}}')

show() { # title, ids
  local title="$1" ids="$2"
  if [ -z "$ids" ]; then echo "$title: none"; else echo "$title:"; echo "$ids" | sed 's/^/  /'; fi
}
show "pods" "$PODS"
show "containers" "$CONTAINERS"
show "images" "$IMAGES"
show "volumes" "$VOLUMES"

run() { # print always, execute only with --apply
  echo "+ $*"
  if [ "$APPLY" -eq 1 ]; then "$@"; fi
}

# Order matters: pods and containers first, then the images and volumes they used.
if [ -n "$PODS" ]; then
  # shellcheck disable=SC2086
  run podman pod rm -f $PODS
fi
if [ -n "$CONTAINERS" ]; then
  # shellcheck disable=SC2086
  run podman rm -f $CONTAINERS
fi
if [ -n "$IMAGES" ]; then
  # shellcheck disable=SC2086
  run podman rmi $IMAGES
fi
if [ -n "$VOLUMES" ]; then
  # shellcheck disable=SC2086
  run podman volume rm $VOLUMES
fi

if [ "$STOP_MACHINE" -eq 1 ]; then
  # Refuse to stop the VM while any container (labeled or not) is still running.
  OTHERS=$(podman ps --format '{{.Names}}')
  if [ "$APPLY" -eq 1 ] && [ -n "$OTHERS" ]; then
    echo "error: not stopping $MACHINE, containers still running:" >&2
    echo "$OTHERS" | sed 's/^/  /' >&2
    exit 1
  fi
  run podman machine stop "$MACHINE"
fi

echo "disk:"
df -h / | tail -1
if [ "$APPLY" -eq 0 ]; then echo "dry-run only; re-run with --apply to execute."; fi
