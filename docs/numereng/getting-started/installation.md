# Installation

`numereng` is meant to be used directly from a cloned source checkout.

## Prerequisites

- Python `3.12+`
- `uv`
- optional: `just`
- optional: Node.js `20+` if you plan to work with the dashboard frontend

## Clone And Bootstrap

```bash
git clone <repo-url> numereng
cd numereng
uv sync --extra dev
```

The repo-managed `.venv` is the canonical environment. Do not create a separate ad hoc virtualenv.

## Initialize The Local Store

```bash
uv run numereng store init
```

This bootstraps `.numereng/numereng.db` and the canonical local runtime layout.

## Verify The Installation

```bash
uv run numereng --help
uv run python -c "import numereng.api, numereng.api.pipeline"
```

## Configure Numerai Credentials

Set Numerai auth in your shell before dataset, model, round, or submission operations:

```bash
export NUMERAI_PUBLIC_ID=your_public_id
export NUMERAI_SECRET_KEY=your_secret_key
```

## Optional: Sync The Official Numerai Docs

```bash
uv run numereng docs sync numerai
```

This populates `docs/numerai/` in the current checkout.

## Launch The Dashboard

```bash
just viz
```

This starts:

- dashboard UI on [http://127.0.0.1:5173](http://127.0.0.1:5173)
- backend API on [http://127.0.0.1:8502](http://127.0.0.1:8502)

Or launch the backend service directly:

```bash
uv run numereng viz --host 127.0.0.1 --port 8502
```

Direct backend endpoint:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

## Working Rule

Stay in the repo root and prefer `uv run numereng ...` so numereng uses the current checkout, the repo-managed `.venv`, and the repo-local `.numereng/` state.
