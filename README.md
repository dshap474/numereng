![Numereng](docs/assets/numereng-banner.png)

# numereng

`numereng` is a package-first Numerai agentic development engine for training, scoring, submissions, experiments, HPO, ensembles, cloud execution, and read-only monitoring.

Stable public interfaces:

- CLI: `numereng`
- Python facade: `import numereng.api`
- Workflow facade: `import numereng.api.pipeline`

## Quick Start

Requires Python `3.12+`.

```bash
uv sync --extra dev
make fmt
uv run numereng --help
make oss-preflight
make test
uv build
```

Optional extras:

- `uv sync --extra training`
- `uv sync --extra mlops`

Default runtime state lives under `.numereng/`.

## Development Tooling

The canonical local toolchain is `uv` for environment management, Ruff for linting and formatting,
`ty` for the repo's enforced type gate, and `pytest` for tests.

```bash
make fmt
make test
make test-all
```

`ty` adoption is intentionally staged in `pyproject.toml` through a scoped
`[tool.ty.src].include` list. The first enforced surface covers `config`,
`platform`, `features/submission`, a small public API subset, and mirrored smoke/unit tests while
backlog-heavy areas remain out of scope for this initial baseline.

## Core Workflows

- Train runs from strict JSON configs: `uv run numereng run train --help`
- Re-score an existing run: `uv run numereng run score --help`
- Submit predictions or run outputs: `uv run numereng run submit --help`
- Build or promote shared benchmark baselines from existing runs: `uv run numereng baseline --help`
- Manage experiments and pack experiment summaries: `uv run numereng experiment --help`
- Run agentic research campaigns rooted at one experiment: `uv run numereng research --help`
- Run HPO studies: `uv run numereng hpo --help`
- Build ensembles: `uv run numereng ensemble --help`
- Neutralize predictions: `uv run numereng neutralize --help`
- Manage datasets, store state, and backfill viz artifacts for historical runs: `uv run numereng dataset-tools --help`, `uv run numereng store --help`
- Build normalized read-only monitor snapshots for local or remote-backed stores: `uv run numereng monitor --help`
- Launch cloud jobs: `uv run numereng cloud --help`
- Query Numerai APIs and forum data: `uv run numereng numerai --help`

The CLI command families are:

- `run`
- `experiment`
- `baseline`
- `research`
- `hpo`
- `ensemble`
- `neutralize`
- `dataset-tools`
- `store`
- `monitor`
- `cloud`
- `numerai`

Python users can call the same flows through typed request/response contracts in `numereng.api.contracts`.

Agentic research defaults to a headless `codex exec` planner. To use OpenRouter instead, switch `ACTIVE_MODEL_SOURCE` in `src/numereng/config/openrouter/active-model.py` to `openrouter`.
Agentic research now centers on saved research programs: each session binds to one markdown program file, snapshots it into the experiment, and then runs against that persisted snapshot. The default tracked `numerai-experiment-loop` program is config-centric: each autonomous iteration picks one parent config, asks the planner for a small validated mutation, materializes one child config, and trains that single child run. Additional custom programs can be dropped into `src/numereng/features/agentic_research/programs/` and are ignored by git by default.

## Architecture

The codebase follows a strict dependency direction:

```text
config -> platform -> features -> api -> cli
```

High-level layout:

- `src/numereng/config/`: strict training and HPO config contracts/loaders
- `src/numereng/platform/`: Numerai adapters, forum scraping, shared boundary errors, remote monitor profile loading
- `src/numereng/features/`: business logic by slice
- `src/numereng/api/`: stable Python facade and workflow entrypoints
- `src/numereng/cli/`: CLI parsing and command dispatch
- `viz/api/numereng_viz/`: read-only dashboard backend package and monitor snapshot composition
- `src/numereng/features/viz/`: compatibility shim for the public viz boundary
- `viz/web/`: dashboard frontend

Core feature slices include:

- training
- scoring
- submission
- baseline
- experiments
- agentic research
- hpo
- ensemble
- dataset tools
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
    runtime.json
    resolved.json
    results.json
    metrics.json
    score_provenance.json
    artifacts/predictions/*.parquet
  datasets/
    baselines/<name>/
      baseline.json
      pred_<name>.parquet
  experiments/<experiment_id>/
    experiment.json
    EXPERIMENT.md
    EXPERIMENT.pack.md
    configs/*.json
    agentic_research/
      program.json
      lineage.json
      llm_trace.jsonl
      llm_trace.md
      rounds/rN/*
  cache/
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
Training and `run score` persist the primary per-era CORR artifact used by run-detail charts, and older runs can be backfilled with `uv run numereng store materialize-viz-artifacts --kind per-era-corr ...`.

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
