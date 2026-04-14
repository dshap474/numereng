![Numereng](docs/assets/numereng-banner.png)

# numereng

`numereng` is a package-first agentic development environment for Numerai. It gives you one installed runtime plus one workspace layout for experiments, notes, custom models, research programs, shipped agent skills, and the read-only dashboard.

`numereng` is a community-built, self-supported tool. It is not affiliated with, endorsed by, or supported by Numerai.

Stable public interfaces:

- CLI: `numereng`
- Python facade: `import numereng.api`
- Workflow facade: `import numereng.api.pipeline`

## Install

Requires Python `3.12+`.

```bash
uvx --from numereng numereng init --workspace numerai-dev
```

Alternative:

```bash
pip install numereng
python -m numereng.cli init --workspace numerai-dev
```

`numereng init` now creates the canonical workspace scaffold and provisions a workspace-local `uv` project with its own `.venv`. From then on, run commands from the workspace with `uv run numereng ...`.

The base install includes the public Python API, CLI, monitor stack, and the core local training/scoring runtime.

Optional extras:

- `pip install "numereng[training]"` for additional model backends and HPO tooling beyond the core local stack
- `pip install "numereng[mlops]"`

## Quick Start

Create a fresh workspace anywhere:

```bash
uvx --from numereng numereng init --workspace numerai-dev
cd numerai-dev
```

That creates the canonical workspace layout:

```text
numerai-dev/
  experiments/
  notes/
  custom_models/
  research_programs/
  .agents/skills/
  .numereng/
```

Then work from that directory:

```bash
uv run numereng --help
uv run numereng experiment list
uv run numereng viz
```

Default dashboard endpoint:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

## Workspace Model

Visible workspace roots:

- `experiments/`
- `notes/`
- `custom_models/`
- `research_programs/`
- `.agents/skills/`

Hidden numereng-managed runtime state:

- `.numereng/numereng.db`
- `.numereng/runs/`
- `.numereng/datasets/`
- `.numereng/cache/`
- `.numereng/tmp/`
- `.numereng/remote_ops/`

`numereng init` is idempotent and does not overwrite existing user-authored files. If the workspace runtime drifts after a package upgrade, run `uv run numereng workspace sync`.

## Core Workflows

- Train runs from strict JSON configs: `numereng run train --help`
- Re-score an existing run: `numereng run score --help`
- Submit predictions or run outputs: `numereng run submit --help`
- Manage experiments and experiment reports: `numereng experiment --help`
- Run agentic research campaigns: `numereng research --help`
- Run HPO studies: `numereng hpo --help`
- Build ensembles: `numereng ensemble --help`
- Build production submission packages and model uploads: `numereng serve --help`
- Neutralize predictions: `numereng neutralize --help`
- Maintain datasets and runtime state: `numereng dataset-tools --help`, `numereng store --help`
- Launch read-only monitoring: `numereng viz`
- Launch cloud jobs: `numereng cloud --help`
- Query Numerai APIs and forum data: `numereng numerai --help`

Python users can call the same flows through typed request/response contracts in `numereng.api.contracts`.

## Docs

- Product docs: [`docs/numereng`](docs/numereng)
- Agent entrypoint: [`docs/llms.txt`](docs/llms.txt)
- Architecture map: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

Start here:

- [`docs/numereng/getting-started/installation.md`](docs/numereng/getting-started/installation.md)
- [`docs/numereng/getting-started/project-layout.md`](docs/numereng/getting-started/project-layout.md)
- [`docs/numereng/reference/custom-models.md`](docs/numereng/reference/custom-models.md)
- [`docs/numereng/workflows/dashboard.md`](docs/numereng/workflows/dashboard.md)
- [`docs/numereng/workflows/serving.md`](docs/numereng/workflows/serving.md)

## Contributors

The repo remains the source of truth for implementation, tests, docs, and packaged assets.
End users do not need this checkout; they only need an installed `numereng` runtime plus a workspace.

Contributor setup:

```bash
uv sync --extra dev
just test
just build
```

Contributor/deep-system docs:

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`docs/llms.txt`](docs/llms.txt)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`SECURITY.md`](SECURITY.md)
