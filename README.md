# numereng

![Numereng](docs/assets/NUMERENG.png)

`numereng` is a package-first Numerai workflow system for training, scoring, submissions, experiments, HPO, ensembles, cloud execution, and read-only monitoring.

Stable public interfaces:

- CLI: `numereng`
- Python facade: `import numereng.api`
- Workflow facade: `import numereng.api.pipeline`

## Quick Start

```bash
uv sync --extra dev
uv run numereng --help
make oss-preflight
make test
uv build
```

Optional extras:

- `uv sync --extra training`
- `uv sync --extra mlops`

Default runtime state lives under `.numereng/`.

## Core Workflows

- Train runs from strict JSON configs: `uv run numereng run train --help`
- Re-score an existing run: `uv run numereng run score --help`
- Submit predictions or run outputs: `uv run numereng run submit --help`
- Manage experiments: `uv run numereng experiment --help`
- Run HPO studies: `uv run numereng hpo --help`
- Build ensembles: `uv run numereng ensemble --help`
- Neutralize predictions: `uv run numereng neutralize --help`
- Manage datasets and store state: `uv run numereng dataset-tools --help`, `uv run numereng store --help`
- Launch cloud jobs: `uv run numereng cloud --help`
- Query Numerai APIs and forum data: `uv run numereng numerai --help`

The CLI command families are:

- `run`
- `experiment`
- `hpo`
- `ensemble`
- `neutralize`
- `dataset-tools`
- `store`
- `cloud`
- `numerai`

Python users can call the same flows through typed request/response contracts in `numereng.api.contracts`.

## Architecture

The codebase follows a strict dependency direction:

```text
config -> platform -> features -> api -> cli
```

High-level layout:

- `src/numereng/config/`: strict training and HPO config contracts/loaders
- `src/numereng/platform/`: Numerai adapters, forum scraping, shared boundary errors
- `src/numereng/features/`: business logic by slice
- `src/numereng/api/`: stable Python facade and workflow entrypoints
- `src/numereng/cli/`: CLI parsing and command dispatch
- `src/numereng/features/viz/`: read-only dashboard backend
- `viz/web/`: dashboard frontend

Core feature slices include:

- training
- scoring
- submission
- experiments
- hpo
- ensemble
- feature neutralization
- store
- telemetry
- cloud (`aws`, `modal`)
- viz

## Runtime Layout

The default store root is `.numereng/`. Common artifacts include:

```text
.numereng/
  numereng.db
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
  datasets/
  cloud/
  notes/
```

## AWS Setup

To use the AWS cloud commands, first configure standard AWS credentials locally and create a local SSH keypair for EC2 access/debugging.

```bash
aws configure
ssh-keygen -t ed25519 -f ~/.ssh/numereng-aws
```

Use shared AWS credentials, `AWS_PROFILE`, or direct AWS credential env vars for auth. Then use `uv run numereng cloud --help` to supply the AWS-specific flags needed for your EC2 or managed AWS flow.

## Dashboard

Run the dashboard API and web app:

```bash
make viz
```

- Requires Node.js 20+ and `npm` for the web app.
- API: `http://127.0.0.1:8502`
- Web: `http://127.0.0.1:5173`
- Stop servers: `make kill-viz`

The dashboard is monitor-only. Launch and control operations still happen through the CLI or Python API.

## Docs

- Agent entrypoint: `docs/llms.txt`
- Deep architecture map: `docs/ARCHITECTURE.md`
- CLI reference: `docs/numereng/reference/cli.md`
- Training workflow: `docs/numereng/workflows/training.md`
- Experiments workflow: `docs/numereng/workflows/experiments.md`
- Ensembles workflow: `docs/numereng/workflows/ensembles.md`
- Cloud training workflow: `docs/numereng/workflows/cloud-training.md`

## Contributing and Support

- Contribution guide: `CONTRIBUTING.md`
- Support policy: `SUPPORT.md`
- Security policy: `SECURITY.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
