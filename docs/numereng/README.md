# Numereng

`numereng` is a package-first Numerai workflow engine for training, scoring, submissions, experiments, HPO, ensembles, cloud execution, and read-only monitoring.

Stable public interfaces:

- CLI: `numereng`
- Python facade: `import numereng.api`
- Workflow facade: `import numereng.api.pipeline`

## Core Workflows

- Train a run from a strict JSON config: `uv run numereng run train --help`
- Re-score an existing run from persisted predictions: `uv run numereng run score --help`
- Submit predictions or run artifacts: `uv run numereng run submit --help`
- Group work into experiments: `uv run numereng experiment --help`
- Run Optuna-backed HPO studies: `uv run numereng hpo --help`
- Build and inspect ensembles: `uv run numereng ensemble --help`
- Neutralize predictions as a separate stage: `uv run numereng neutralize --help`
- Build official-style dataset artifacts and maintain store state: `uv run numereng dataset-tools --help`, `uv run numereng store --help`
- Launch cloud workflows: `uv run numereng cloud --help`
- Query Numerai datasets/models/rounds and scrape the forum: `uv run numereng numerai --help`
- Launch the read-only dashboard: `make viz`

Default runtime state lives under `.numereng/`.

## Quick Start

```bash
uv sync --extra dev
uv run numereng --help
uv run numereng run train --config configs/run.json
uv run numereng experiment list
make viz
```

## Architecture

The codebase uses a strict dependency direction:

```text
config -> platform -> features -> api -> cli
```

High-level layout:

- `src/numereng/config/`: strict config contracts and loaders
- `src/numereng/platform/`: Numerai adapters, forum scraper, boundary errors
- `src/numereng/features/`: training, scoring, submission, experiments, hpo, ensemble, neutralization, dataset-tools, store, telemetry, cloud, viz, models
- `src/numereng/api/`: stable public Python facade and workflow entrypoints
- `src/numereng/cli/`: CLI parsing and command dispatch
- `viz/web/`: dashboard frontend

## Runtime Layout

Common artifacts under `.numereng/`:

```text
.numereng/
  numereng.db
  numereng.db-shm
  numereng.db-wal
  runs/<run_id>/
    run.json
    resolved.json
    results.json
    metrics.json
    score_provenance.json
    artifacts/predictions/*.parquet
  experiments/<experiment_id>/
    experiment.json
    EXPERIMENT.md
    configs/*.json
  hpo/
  ensembles/
  datasets/
  cloud/
  notes/
```

Forum scraping writes outside the store by default:

```text
docs/numerai/forum/
  INDEX.md
  posts/YYYY/MM/*.md
  .forum_scraper_manifest.json
  .forum_scraper_state.json
```

## Where To Go Next

- [Installation](getting-started/installation.md)
- [Train Your First Model](getting-started/first-model.md)
- [Project Layout](getting-started/project-layout.md)
- [Training Workflow](workflows/training.md)
- [Experiments](workflows/experiments.md)
- [Cloud Training](workflows/cloud-training.md)
- [Store Operations](workflows/store-ops.md)
- [Numerai Operations](workflows/numerai-ops.md)
- [Dashboard](workflows/dashboard.md)
- [CLI Reference](reference/cli.md)
- [Python API](reference/python-api.md)
- [Custom Models](reference/custom-models.md)
- [Configuration](reference/configuration.md)
- [Metrics](reference/metrics.md)
