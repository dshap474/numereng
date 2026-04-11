# Store Operations

Numereng keeps filesystem artifacts and a SQLite index aligned under the store root.

Use the `store` command family when you need to initialize, reindex, rebuild, or diagnose that state.

## Initialize

```bash
uv run numereng store init
```

Use this once per store root to bootstrap the SQLite schema.

## Index One Run

```bash
uv run numereng store index --run-id <run_id>
```

Use this when a run artifact directory already exists and you want its manifest, metrics, and artifact paths re-ingested.

## Rebuild The Entire Index

```bash
uv run numereng store rebuild
```

Use this when you want SQLite state rebuilt from canonical filesystem artifacts under the store root.

## Diagnose Drift

```bash
uv run numereng store doctor
```

`store doctor` checks for:

- missing or corrupt database state
- required tables
- artifact/index mismatches
- stray top-level paths that do not belong under the canonical store layout
- retention-managed tmp remote-config staging under `.numereng/tmp/remote-configs/*.json`

## Cleanup Strays

```bash
uv run numereng store doctor --fix-strays
```

Use `--fix-strays` only when you explicitly want numereng to clean up detected stray store paths.

With `--fix-strays`, numereng now performs two conservative cleanup passes:

- deletes targeted stray top-level directories such as old logs/smoke roots
- prunes `.numereng/tmp/remote-configs/*.json` only when the file is older than 30 days and not referenced by an active run lifecycle

If the store DB is missing or unreadable, tmp remote-config cleanup is skipped rather than guessed.

## When To Use Each Command

- use `init` for a fresh store root
- use `index` after manual artifact recovery for one run
- use `rebuild` after large filesystem changes or when the SQLite index is untrusted
- use `doctor` when artifacts and metrics look inconsistent or you suspect drift

## High-Risk Gotchas

- keep `--workspace` consistent across train, experiment, HPO, ensemble, and store commands
- do not assume SQLite is the source of truth; canonical artifacts live on disk
- use rebuild when you need full reconciliation, not repeated one-off indexing
