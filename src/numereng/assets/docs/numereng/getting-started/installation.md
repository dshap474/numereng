# Installation

Set up `numereng` as an installed runtime, then initialize a workspace.

## Prerequisites

- Python `3.12+`
- `uv` or `pip`

## Install The Package

Preferred:

```bash
uv tool install numereng
```

Alternative:

```bash
pip install numereng
```

The base install already includes the core local training/scoring stack. Optional extras remain available for additional model backends and HPO tooling:

```bash
pip install "numereng[training]"
```

## Initialize A Workspace

```bash
mkdir numerai-dev
cd numerai-dev
numereng init
```

This creates:

- `experiments/`
- `notes/`
- `custom_models/`
- `research_programs/`
- `.agents/skills/`
- `.numereng/`

## Verify The Install

```bash
numereng --help
```

If you installed `numereng` into a Python environment rather than as a `uv tool`, you can also verify the public Python facade:

```bash
python -c "import numereng.api"
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
numereng viz
```

Default local endpoint:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

## Contributor Setup

If you are hacking on `numereng` itself rather than just using it, use the repo-managed environment instead:

```bash
uv sync --extra dev
just test
```
