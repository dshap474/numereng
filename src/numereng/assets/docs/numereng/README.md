# Numereng

`numereng` is a repo-local Numerai workspace runtime. Clone the repo, `cd` into it, use the repo-managed `.venv`, and work directly from this checkout.

Stable public interfaces:

- CLI: `numereng`
- Python facade: `import numereng.api`
- Workflow facade: `import numereng.api.pipeline`

## Golden Path

```bash
uv sync --extra dev
just viz
```

## Canonical Workspace Layout

Tracked authoring and extension roots:

- `src/numereng/features/models/custom_models/`
- `src/numereng/features/agentic_research/programs/`
- `.agents/skills/`

Hidden numereng-managed runtime state:

- `.numereng/numereng.db`
- `.numereng/experiments/`
- `.numereng/notes/`
- `.numereng/runs/`
- `.numereng/datasets/`
- `.numereng/cache/`
- `.numereng/tmp/`
- `.numereng/remote_ops/`

## Core Workflows

- Train a run from a strict JSON config: `numereng run train --help`
- Re-score an existing run from persisted predictions: `numereng run score --help`
- Submit predictions or run artifacts: `numereng run submit --help`
- Group work into experiments: `numereng experiment --help`
- Run Optuna-backed HPO studies: `numereng hpo --help`
- Build and inspect ensembles: `numereng ensemble --help`
- Neutralize predictions as a separate stage: `numereng neutralize --help`
- Maintain datasets and runtime state: `numereng dataset-tools --help`, `numereng store --help`
- Launch cloud workflows: `numereng cloud --help`
- Query Numerai datasets/models/rounds and forum data: `numereng numerai --help`
- Launch the read-only dashboard: `just viz`

## Where To Go Next

- [Installation](getting-started/installation.md)
- [Project Layout](getting-started/project-layout.md)
- [Training Workflow](workflows/training.md)
- [Experiments](workflows/experiments.md)
- [Cloud Training](workflows/cloud-training.md)
- [Store Operations](workflows/store-ops.md)
- [Dashboard](workflows/dashboard.md)
- [CLI Reference](reference/cli.md)
- [Python API](reference/python-api.md)
- [Custom Models](reference/custom-models.md)
- [Configuration](reference/configuration.md)
- [Metrics](reference/metrics.md)
