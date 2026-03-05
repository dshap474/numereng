#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <experiment_id> [<experiment_id> ...]" >&2
  exit 2
fi

STORE_ROOT="${STORE_ROOT:-.numereng}"

echo "== store doctor =="
uv run numereng store doctor --store-root "$STORE_ROOT"

for EXPERIMENT_ID in "$@"; do
  echo "== details: ${EXPERIMENT_ID} =="
  uv run numereng experiment details --id "$EXPERIMENT_ID" --format json --store-root "$STORE_ROOT"

  echo "== report: ${EXPERIMENT_ID} =="
  uv run numereng experiment report --id "$EXPERIMENT_ID" --format json --store-root "$STORE_ROOT"

  echo "== impact summary: ${EXPERIMENT_ID} =="
  uv run python .agents/skills/store-ops/scripts/collect_store_impact.py \
    --store-root "$STORE_ROOT" \
    --experiment-id "$EXPERIMENT_ID"
done
