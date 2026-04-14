# Installation

Bootstrap a workspace as a local `uv` project. The workspace owns its `.venv` and installed `numereng` runtime.

## Prerequisites

- Python `3.12+`
- `uv`

## Bootstrap A Workspace

Preferred:

```bash
uvx --from numereng numereng init --workspace numerai-dev
```

Contributor or local-source mode:

```bash
uvx --from /absolute/path/to/numereng numereng init --workspace numerai-dev --runtime-source path --runtime-path /absolute/path/to/numereng
```

The base install already includes the core local training/scoring stack. Optional extras remain available for additional model backends and HPO tooling:

```bash
cd numerai-dev
uv run numereng workspace sync --with-training
```

`numereng init` creates:

- `experiments/`
- `notes/`
- `custom_models/`
- `research_programs/`
- `.agents/skills/`
- `.numereng/`
- `pyproject.toml`
- `.python-version`
- `.venv/`

## Verify The Install

```bash
cd numerai-dev
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

From the workspace root:

```bash
uv run numereng viz
```

Default local endpoint:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

## Contributor Setup

If you are hacking on `numereng` itself rather than just using it, use the repo-managed environment instead:

```bash
uv sync --extra dev
just test
```
