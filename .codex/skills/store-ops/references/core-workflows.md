# Core Workflows

## Workflow 1: Baseline Health Check

```bash
uv run numereng store doctor
```

If healthy and no requested mutation, stop here.

## Workflow 2: Diagnose One Experiment

```bash
uv run numereng experiment details --id <experiment_id> --format json
uv run numereng experiment report --id <experiment_id> --format json
uv run python .agents/skills/store-ops/scripts/collect_store_impact.py \
  --experiment-id <experiment_id>
```

Outcome:
- run IDs in manifest
- run IDs on disk linked by `run.json.experiment_id`
- DB row counts by dependent table
- overlap risk with other experiments

## Workflow 3: Reindex One Run

Use when one run exists on disk but is missing/stale in SQLite.

```bash
uv run numereng store index --run-id <run_id>
uv run numereng store doctor
```

## Workflow 4: Full Store Rebuild

Use when many run rows are stale/missing.

```bash
uv run numereng store rebuild
uv run numereng store doctor
```

## Workflow 5: Reset Specific Run IDs

Dry-run first:

```bash
uv run python .agents/skills/store-ops/scripts/reset_runs.py \
  --run-id <run_id>
```

Execute after explicit confirmation:

```bash
uv run python .agents/skills/store-ops/scripts/reset_runs.py \
  --run-id <run_id> \
  --execute
```

Multiple run IDs:

```bash
uv run python .agents/skills/store-ops/scripts/reset_runs.py \
  --run-id <run_id_a> \
  --run-id <run_id_b> \
  --execute
```

## Workflow 6: Reset Experiment Run Data (Preserve Design Files)

Dry-run first:

```bash
uv run python .agents/skills/store-ops/scripts/reset_experiment_runs.py \
  --experiment-id <experiment_id>
```

Execute after explicit confirmation:

```bash
uv run python .agents/skills/store-ops/scripts/reset_experiment_runs.py \
  --experiment-id <experiment_id> \
  --execute
```

Multi-experiment reset:

```bash
uv run python .agents/skills/store-ops/scripts/reset_experiment_runs.py \
  --experiment-id <id_a> \
  --experiment-id <id_b> \
  --execute
```

## Workflow 7: Verification Bundle

```bash
bash .agents/skills/store-ops/assets/bash/verification-checks.sh <experiment_id> [<experiment_id> ...]
```

Or equivalent manual checks:

```bash
uv run numereng experiment details --id <experiment_id> --format json
uv run numereng experiment report --id <experiment_id> --format json
uv run numereng store doctor
```

## Workflow 8: SQL-First Investigation

When CLI output is insufficient, use `assets/sql/impact-queries.sql` with explicit IDs:

```bash
sqlite3 .numereng/numereng.db < .agents/skills/store-ops/assets/sql/impact-queries.sql
```

Prefer read-only queries unless user has explicitly approved mutation.
