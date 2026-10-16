#!/usr/bin/env bash
# Read-only Podman label audit (hybrid/executor/audit.py): lists Bigbrane-labeled resources and anything our
# builds created that lacks bigbrane.owner=bigbrane. Changes nothing. Exit 1 when unlabeled resources of ours exist.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v podman >/dev/null || { echo "error: podman not found" >&2; exit 1; }
podman info >/dev/null 2>&1 || { echo "error: podman machine is not reachable" >&2; exit 1; }
exec "$ROOT/.venv-hybrid/bin/python" -m hybrid.executor.audit
