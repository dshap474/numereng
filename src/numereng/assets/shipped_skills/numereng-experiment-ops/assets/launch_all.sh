#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/launch_all.py"

if [[ ! -f "${RUNNER}" ]]; then
  echo "Launcher not found: ${RUNNER}" >&2
  exit 1
fi

uv run python "${RUNNER}" "$@"
