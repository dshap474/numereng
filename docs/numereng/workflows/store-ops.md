# Store Operations

Use `numereng store` when filesystem artifacts and SQLite state need initialization, reconciliation, repair, or viz backfill.

## Initialize

```bash
uv run numereng store init
```

## Index Or Rebuild

Index one run:

```bash
uv run numereng store index --run-id <run_id>
```

Rebuild the full SQLite index from canonical artifacts:

```bash
uv run numereng store rebuild
```

## Diagnose Drift

```bash
uv run numereng store doctor
```

Clean targeted stray paths:

```bash
uv run numereng store doctor --fix-strays
```

### Restoring artifacts that were pruned

If `store doctor` reports `run_count_mismatch` with `filesystem_runs` far below `indexed_runs`, the SQLite index still has run rows but the on-disk `.numereng/runs/<run_id>/` directories are gone. The dashboard's Performance tab shows empty for those runs because it reads scoring parquet files off disk. If the runs originated on a remote target, recover via `remote experiment pull --mode scoring` — see [Remote Operations → Recovery](./remote-ops.md#recovery-restoring-missing-local-artifacts). Do not use `store rebuild` for this; rebuild only reconciles the SQLite index against whatever is already on disk.

## Repair And Backfill

Backfill missing execution metadata for exactly one scope:

```bash
# one run
uv run numereng store backfill-run-execution --run-id <run_id>

# all runs
uv run numereng store backfill-run-execution --all
```

Repair active lifecycle rows only:

```bash
uv run numereng store repair-run-lifecycles
```

Widen the sweep to all lifecycle rows:

```bash
uv run numereng store repair-run-lifecycles --all
```

## Viz Backfill

Materialize missing viz diagnostics:

```bash
uv run numereng store materialize-viz-artifacts \
  --kind scoring-artifacts \
  --run-id <run_id>
```

Or backfill older per-era compatibility artifacts:

```bash
uv run numereng store materialize-viz-artifacts \
  --kind per-era-corr \
  --experiment-id <experiment_id>
```

## High-Risk Gotchas

- filesystem artifacts are canonical; SQLite is an index over them
- use rebuild when broad reconciliation is needed, not repeated one-off indexing
- `backfill-run-execution` is strict XOR: pass exactly one of `--run-id` or `--all`
- `repair-run-lifecycles` defaults to active-only repair; `--all` broadens the sweep
- `--fix-strays` is a cleanup action; do not run it casually on a workspace you have not inspected

## Read Next

- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
- [Dashboard & Monitor](dashboard.md)
