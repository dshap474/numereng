# numereng

`numereng` is a package-first Numerai workflow toolkit with two public interfaces:

- CLI: `numereng`
- Python facade: `import numereng.api as api_module`

It covers local and cloud training, submissions, experiments, HPO, ensembles, prediction neutralization, store maintenance, and read-only monitoring APIs for the dashboard.

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

## Core Workflows

- Train runs from strict JSON configs: `uv run numereng run train --help`
- Submit predictions or run outputs: `uv run numereng run submit --help`
- Manage experiments: `uv run numereng experiment --help`
- Run HPO studies: `uv run numereng hpo --help`
- Build ensembles: `uv run numereng ensemble --help`
- Neutralize predictions: `uv run numereng neutralize --help`
- Manage datasets and store state: `uv run numereng dataset-tools --help`, `uv run numereng store --help`
- Launch cloud jobs: `uv run numereng cloud --help`

By default, runtime artifacts live under `.numereng/`.

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

## Contributing and Support

- Contribution guide: `CONTRIBUTING.md`
- Support policy: `SUPPORT.md`
- Security policy: `SECURITY.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
