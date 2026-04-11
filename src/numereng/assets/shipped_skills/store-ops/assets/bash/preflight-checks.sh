#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <experiment_id> [<experiment_id> ...]" >&2
  exit 2
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-.}"

echo "== active writer check =="
WRITERS="$(ps aux | rg -i "numereng (run train|experiment train|cloud aws train submit|cloud modal train submit)" | rg -v rg || true)"
if [[ -n "$WRITERS" ]]; then
  echo "active training writer processes detected:" >&2
  echo "$WRITERS" >&2
else
  echo "none"
fi

echo "== store doctor =="
numereng store doctor --workspace "$WORKSPACE_ROOT"

for EXPERIMENT_ID in "$@"; do
  echo "== experiment: ${EXPERIMENT_ID} =="
  numereng experiment details --id "$EXPERIMENT_ID" --format json --workspace "$WORKSPACE_ROOT"
  numereng experiment report --id "$EXPERIMENT_ID" --format json --workspace "$WORKSPACE_ROOT"
  uv run python .agents/skills/store-ops/scripts/collect_store_impact.py \
    --workspace "$WORKSPACE_ROOT" \
    --experiment-id "$EXPERIMENT_ID"
done
