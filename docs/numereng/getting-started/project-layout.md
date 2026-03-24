# Project Layout

Numereng has two layouts you need to keep straight:

- the repository layout under the repo root
- the runtime/store layout under `.numereng/`

## Repository Layout

Important top-level areas:

- `src/numereng/config/`: training and HPO contracts/loaders
- `src/numereng/platform/`: Numerai client, forum scraper, boundary errors
- `src/numereng/features/`: business logic slices such as training, scoring, experiments, ensemble, store, cloud, viz
- `src/numereng/api/`: stable Python facade and workflow entrypoints
- `src/numereng/cli/`: CLI parsing and command dispatch
- `src/numereng/features/models/custom_models/`: custom model plugin path
- `docs/numereng/`: user-facing docs set
- `docs/numerai/forum/`: forum scrape output, when used
- `viz/web/`: dashboard frontend

## Runtime Layout

Default store root: `.numereng`

```text
.numereng/
  numereng.db
  numereng.db-shm
  numereng.db-wal

  runs/
    <run_id>/
      run.json
      run.log
      resolved.json
      results.json
      metrics.json
      score_provenance.json
      artifacts/
        predictions/

  experiments/
    <experiment_id>/
      experiment.json
      EXPERIMENT.md
      configs/
      hpo/
      ensembles/

  hpo/
    <study_id>/

  ensembles/
    <ensemble_id>/

  datasets/
  cloud/
  notes/
  cache/
    derived_datasets/
    tabpfn/
```

## Run Artifacts

For a scored local run, the canonical minimum set is:

- `run.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json`
- `artifacts/predictions/*.parquet`

`run score` refreshes the scoring-side artifacts from saved predictions and reindexes the run.

## Experiment Artifacts

Experiments live under `.numereng/experiments/<experiment_id>/` and typically contain:

- `experiment.json`: manifest and run linkage
- `EXPERIMENT.md`: experiment notes/reporting
- `configs/*.json`: planned configs for the sweep
- `hpo/` and `ensembles/`: experiment-scoped derived work when present

## Store Commands

```bash
uv run numereng store init
uv run numereng store index --run-id <run_id>
uv run numereng store rebuild
uv run numereng store doctor
```

Use `store doctor --fix-strays` only when you explicitly want cleanup of detected stray store paths.

## Path Rules

- Training and HPO configs are JSON-only.
- Runtime defaults target `.numereng` unless `--store-root` overrides it.
- Managed cloud state paths should live under `.numereng/cloud/*.json`.
- Custom model discovery defaults to `src/numereng/features/models/custom_models/`.
