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

## Repair And Backfill

Backfill missing execution metadata:

```bash
uv run numereng store backfill-run-execution --all
```

Repair lifecycle rows:

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
- `--fix-strays` is a cleanup action; do not run it casually on a workspace you have not inspected

## Read Next

- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
- [Dashboard & Monitor](dashboard.md)
