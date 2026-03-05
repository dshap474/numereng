# Numereng

CLI-first Numerai workflow package with stable public surfaces:

- CLI entrypoint: `numereng`
- Python API facade: `import numereng.api`

Numereng covers the full loop: run training, track experiments, run HPO studies, build ensembles, neutralize predictions, and submit to Numerai.

## Core Loop

```text
Prepare config -> Train run -> Analyze metrics -> Promote/ensemble -> Submit
       ^                                                          |
       +-------------------------- Iterate ------------------------+
```

## Key Capabilities

- Deterministic run IDs: identical resolved configs reuse the same run identity.
- Strict config contracts: training and HPO configs are JSON-only and reject unknown keys.
- Thin CLI, typed API: command handlers parse/dispatch only; feature logic runs behind `numereng.api`.
- Canonical store layout: `.numereng/` keeps runs, experiments, HPO studies, ensembles, cloud state, logs, and notes aligned with SQLite indexing.

## Quick Start

```bash
# From repo root
uv sync --extra dev

# Bootstrap store
uv run numereng store init

# Train one run
uv run numereng run train --config configs/run.json

# View experiment/run summaries
uv run numereng experiment list
uv run numereng experiment report --id <experiment_id>
```

## Next Steps

- [Installation](getting-started/installation.md)
- [Train Your First Model](getting-started/first-model.md)
- [Project Layout](getting-started/project-layout.md)
- [Training Workflow](workflows/training.md)
- [Submission Workflow](workflows/submission.md)
- [Metrics Reference](reference/metrics.md)
- [CLI Reference](reference/cli.md)
