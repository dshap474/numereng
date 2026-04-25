# Numereng

`numereng` is a repo-local Numerai workflow runtime. You clone the repo, work directly from that checkout, and let numereng manage model-development state under `.numereng/`.

Public surfaces:

- CLI: `numereng`
- Python facade: `import numereng.api`
- Training workflow facade: `import numereng.api.pipeline`

## Quick Start

```bash
git clone <repo-url> numereng
cd numereng
uv sync --extra dev
uv run numereng store init
uv run numereng --help
just viz
```

If you want a local copy of the official Numerai docs:

```bash
uv run numereng docs sync numerai
```

## How To Think About Numereng

- A **run** is one concrete training or scoring result under `.numereng/runs/<run_id>/`.
- An **experiment** groups related configs, runs, champions, reports, HPO studies, ensembles, and serving packages under `.numereng/experiments/<experiment_id>/`.
- **Agentic research** runs a persisted config-mutation loop inside one root experiment.
- **Serving** freezes a winning package for local live builds or Numerai-hosted model uploads.
- **Store** commands keep filesystem artifacts and SQLite state aligned.
- **Dashboard** and **monitor** are read-only inspection surfaces over the current checkout.

## Canonical Workspace Layout

Repo-local extension roots:

- `src/numereng/features/models/custom_models/` for custom model wrappers
- `src/numereng/features/agentic_research/PROGRAM.md` for agentic config-research policy
- `.agents/skills/` for local custom skills; this path is gitignored

Repo-local runtime state:

- `.numereng/numereng.db`
- `.numereng/experiments/`
- `.numereng/notes/`
- `.numereng/runs/`
- `.numereng/datasets/`
- `.numereng/cache/`
- `.numereng/tmp/`
- `.numereng/remote_ops/`

## Choose The Right Workflow

- Start with [Train Your First Model](getting-started/first-model.md) if you want one good end-to-end example.
- Use [Training Models](workflows/training.md) for one standalone run.
- Use [Experiments](workflows/experiments.md) for tracked model development.
- Use [Agentic Research](workflows/agentic-research.md) for autonomous config mutation loops.
- Use [Hyperparameter Optimization](workflows/optimization.md) for Optuna studies.
- Use [Ensembles](workflows/ensembles.md) when blending scored runs.
- Use [Serving & Model Uploads](workflows/serving.md) and [Submissions](workflows/submission.md) when you are ready to ship.
- Use [Remote Operations](workflows/remote-ops.md) or [Cloud Training](workflows/cloud-training.md) when local compute is not enough.
- Use [Store Operations](workflows/store-ops.md) and [Dashboard & Monitor](workflows/dashboard.md) to inspect and maintain the workspace.

## Read Next

- [Installation](getting-started/installation.md)
- [Project Layout](getting-started/project-layout.md)
- [CLI Commands](reference/cli.md)
- [Python API](reference/python-api.md)
- [Runtime Artifacts & Paths](reference/runtime-artifacts.md)
- [Configuration](reference/configuration.md)
- [Custom Models](reference/custom-models.md)
