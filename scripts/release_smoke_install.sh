#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/numereng-release-smoke.XXXXXX")}"
PORT="${NUMERENG_RELEASE_SMOKE_PORT:-8618}"

cd "${ROOT_DIR}"

uv build
WHEEL_PATH="$(ls -1t dist/*.whl | head -n 1)"
FIND_LINKS_PATH="$(cd dist && pwd -P)"

rm -rf "${WORKSPACE_DIR}"
mkdir -p "${WORKSPACE_DIR}"
WORKSPACE_DIR="$(cd "${WORKSPACE_DIR}" && pwd -P)"
uv venv "${WORKSPACE_DIR}/.venv"
uv pip install --python "${WORKSPACE_DIR}/.venv/bin/python" "${WHEEL_PATH}"

"${WORKSPACE_DIR}/.venv/bin/python" -c "import cloudpickle, numereng.api, numereng.features.serving"
UV_FIND_LINKS="${FIND_LINKS_PATH}" "${WORKSPACE_DIR}/.venv/bin/numereng" init --workspace "${WORKSPACE_DIR}" >/dev/null
"${WORKSPACE_DIR}/.venv/bin/numereng" serve --help >/dev/null

test -d "${WORKSPACE_DIR}/experiments"
test -d "${WORKSPACE_DIR}/notes"
test -d "${WORKSPACE_DIR}/custom_models"
test -d "${WORKSPACE_DIR}/research_programs"
test -d "${WORKSPACE_DIR}/.agents/skills"
test -d "${WORKSPACE_DIR}/.numereng"

"${WORKSPACE_DIR}/.venv/bin/numereng" viz --workspace "${WORKSPACE_DIR}" --host 127.0.0.1 --port "${PORT}" >/tmp/numereng-release-smoke.log 2>&1 &
VIZ_PID=$!
cleanup() {
  kill "${VIZ_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/tmp/numereng-release-smoke-health.json; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${PORT}/" >/tmp/numereng-release-smoke-index.html
grep -q "\"status\":\"ok\"" /tmp/numereng-release-smoke-health.json
grep -q "${WORKSPACE_DIR}" /tmp/numereng-release-smoke-health.json

echo "release_smoke_ok workspace=${WORKSPACE_DIR} wheel=${WHEEL_PATH} port=${PORT}"
