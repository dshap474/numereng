![Numereng](docs/assets/numereng-banner.png)

# numereng

`numereng` is a repo-local Numerai workspace for training, experiments, submissions, serving, and monitoring.

`numereng` is a community-built, self-supported tool. It is not affiliated with, endorsed by, or supported by Numerai.

Stable public interfaces:

- CLI: `numereng`
- Python facade: `import numereng.api`
- Workflow facade: `import numereng.api.pipeline`

## Quick start

Requires Python `3.12+`.

Clone the repo, enter it, and use the repo-managed environment:

```bash
git clone <your-fork-or-local-path> numereng
cd numereng
uv sync --extra dev
```

First commands:

```bash
uv run numereng --help
uv run numereng experiment list
just viz
```

When you use `just viz`:

- dashboard UI: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- backend API: [http://127.0.0.1:8502](http://127.0.0.1:8502)

## Workspace model

The repo checkout is the canonical workspace.

Extension and authoring roots:

- `src/numereng/features/models/custom_models/` for built-in and default-discovered custom model wrappers
- `src/numereng/features/agentic_research/programs/` for built-in research programs
- `.agents/skills/` for repo-local custom skills; this path is local-only and gitignored

Repo-local runtime and experiment state:

- `.numereng/experiments/`
- `.numereng/notes/`
- `.numereng/runs/`
- `.numereng/datasets/`
- `.numereng/cache/`
- `.numereng/tmp/`
- `.numereng/remote_ops/`
- `.numereng/numereng.db`

## Core workflows

Use these command groups from the repo root:

- Train and rescore local runs: `uv run numereng run train --help`, `uv run numereng run score --help`
- Submit predictions or run outputs: `uv run numereng run submit --help`
- Manage experiments and reports: `uv run numereng experiment --help`
- Build model packages and hosted uploads: `uv run numereng serve --help`
- Run research and optimization loops: `uv run numereng research --help`, `uv run numereng ensemble --help`, `uv run numereng neutralize --help`
- Launch monitoring and dashboard views: `just viz`
- Query Numerai APIs and datasets: `uv run numereng numerai --help`

For deeper workflow documentation, use the docs links below instead of treating the README as the full command reference.

## Python API

The Python API is available for typed automation and power-user flows. For full local training orchestration, use `numereng.api.pipeline`.

## Docs and help

Start here:

- Agent usage guide: [AGENTS.md](AGENTS.md)
- Repo setup: [docs/numereng/getting-started/installation.md](docs/numereng/getting-started/installation.md)
- Workspace layout: [docs/numereng/getting-started/project-layout.md](docs/numereng/getting-started/project-layout.md)
- Custom models: [docs/numereng/reference/custom-models.md](docs/numereng/reference/custom-models.md)
- Dashboard: [docs/numereng/workflows/dashboard.md](docs/numereng/workflows/dashboard.md)
- Serving and hosted model uploads: [docs/numereng/workflows/serving.md](docs/numereng/workflows/serving.md)

Deeper system docs:

- Agent entrypoint: [docs/llms.txt](docs/llms.txt)
- Architecture map: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Contributing

Start with:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

Contributor setup:

```bash
uv sync --extra dev
just test
just build
```

The intended user path is clone the repo, `cd` into it, launch your agent, and work directly against this checkout.
