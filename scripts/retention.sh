#!/usr/bin/env bash
# Workspace retention for Bigbrane (rules in hybrid/retention.py). Dry run by default.
#
# Usage:
#   scripts/retention.sh [--apply] [--no-podman]
#
# Removes only: pytest-* directories at any depth under tmp/ older than 1 day, results/bench-tests-* beyond the
# newest 3, __pycache__ under the root, hybrid/ and scripts/, tmp/**/.lake older than 1 day, and tools images
# (labels bigbrane.owner=bigbrane and bigbrane.image=tools) beyond the newest 2. A path inside another planned
# path is not listed separately. Never touches runs/, reports/, data/ or any other results/ entry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/.venv-hybrid/bin/python" -m hybrid.retention "$@"
