# Dashboard

Numereng ships a repo-local read-only monitoring stack.

The dashboard reads the current repo checkout and its `.numereng/` runtime state. Launch and control operations still happen through the CLI or Python API.

## Start

From the repo root:

```bash
just viz
```

Default local endpoint:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

Optional overrides:

```bash
uv run numereng viz --workspace /path/to/repo --host 127.0.0.1 --port 8600
```

## What The Dashboard Reads

- `.numereng/experiments/`
- `.numereng/notes/`
- `.numereng/runs/`
- `.numereng/numereng.db`
- `docs/numerai/`

It does not launch jobs or mutate experiment/store state.

## Backend Contract

The packaged FastAPI app exposes:

- `GET /healthz`
- `GET /`
- `GET /api/...` read-only routes for runs, experiments, studies, ensembles, docs, and notes

The health endpoint reports the resolved `workspace_root` and `.numereng` `store_root`.
