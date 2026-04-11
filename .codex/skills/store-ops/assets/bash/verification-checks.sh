#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <experiment_id> [<experiment_id> ...]" >&2
  exit 2
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-.}"

echo "== store doctor =="
numereng store doctor --workspace "$WORKSPACE_ROOT"

for EXPERIMENT_ID in "$@"; do
  echo "== details: ${EXPERIMENT_ID} =="
  numereng experiment details --id "$EXPERIMENT_ID" --format json --workspace "$WORKSPACE_ROOT"

  echo "== report: ${EXPERIMENT_ID} =="
  numereng experiment report --id "$EXPERIMENT_ID" --format json --workspace "$WORKSPACE_ROOT"

  echo "== impact summary: ${EXPERIMENT_ID} =="
  uv run python .agents/skills/store-ops/scripts/collect_store_impact.py \
    --workspace "$WORKSPACE_ROOT" \
    --experiment-id "$EXPERIMENT_ID"
done
