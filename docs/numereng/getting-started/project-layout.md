# Project Layout

How numereng organizes runtime state under `.numereng/`.

## Canonical Store

Default store root: `.numereng`.

```text
.numereng/
  numereng.db
  runs/
    <run_id>/
      run.json
      resolved.json
      results.json
      metrics.json
      score_provenance.json
      artifacts/
        predictions/
  datasets/
  cloud/
  experiments/
    <experiment_id>/
      experiment.json
      EXPERIMENT.md
      configs/
      hpo/
      ensembles/
  hpo/
  ensembles/
  notes/
```

## Run Identity

Run IDs are deterministic hashes of resolved training config/context.

- Same resolved config/context -> same run ID
- Training indexing is mandatory after artifact write

## Store Commands

```bash
uv run numereng store init
uv run numereng store index --run-id <run_id>
uv run numereng store rebuild
uv run numereng store doctor
```

Use `store doctor --fix-strays` only when you intentionally want automatic cleanup.

## Path Expectations

- Training/HPO configs are explicit `.json` files.
- Runtime defaults target `.numereng` unless `--store-root` overrides it.
