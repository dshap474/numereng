#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/launch_all.py"

resolve_repo_root() {
  local start_dir="${1}"
  local search_dir="${start_dir}"
  while [[ "${search_dir}" != "/" ]]; do
    if [[ -f "${search_dir}/pyproject.toml" ]]; then
      printf "%s\n" "${search_dir}"
      return 0
    fi
    search_dir="$(dirname "${search_dir}")"
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root "${SCRIPT_DIR}" || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(resolve_repo_root "$(pwd)" || true)"
fi

if [[ -z "${REPO_ROOT}" ]]; then
  echo "Could not locate repo root (pyproject.toml) from ${SCRIPT_DIR} or $(pwd)" >&2
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "Launcher not found: ${RUNNER}" >&2
  exit 1
fi

(
  cd "${REPO_ROOT}"
  uv run python "${RUNNER}" "$@"
)
