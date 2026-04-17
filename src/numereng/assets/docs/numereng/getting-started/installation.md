# Setup

`numereng` is meant to be used directly from a local source checkout.

## Prerequisites

- Python `3.12+`
- `uv`
- optional: `just`

## Clone And Bootstrap

```bash
git clone <your-fork-or-local-path> numereng
cd numereng
uv sync --extra dev
```

The repo-managed `.venv` is the canonical runtime.

## Verify

```bash
uv run numereng --help
uv run python -c "import numereng.api, cloudpickle"
```

## Configure Credentials

Set Numerai auth in your shell before dataset, model, round, or submission operations:

```bash
export NUMERAI_PUBLIC_ID=your_public_id
export NUMERAI_SECRET_KEY=your_secret_key
```

## Launch The Dashboard

From the repo root:

```bash
just viz
```

Default local endpoint:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

## Working Rule

Stay in the repo root. Run commands with `uv run ...` so they use the repo-managed environment and repo-local `.numereng/` state.
