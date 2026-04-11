# Dashboard

Numereng ships a packaged read-only monitoring stack.

The dashboard reads the current workspace and its `.numereng/` runtime state. Launch and control operations still happen through the CLI or Python API.

## Start

From the workspace root:

```bash
numereng viz
```

Default local endpoint:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

Optional overrides:

```bash
numereng viz --workspace /path/to/workspace --host 127.0.0.1 --port 8600
```

## What The Dashboard Reads

- `experiments/`
- `notes/`
- `.numereng/runs/`
- `.numereng/numereng.db`
- packaged product docs

It does not launch jobs or mutate experiment/store state.

## Backend Contract

The packaged FastAPI app exposes:

- `GET /healthz`
- `GET /`
- `GET /api/...` read-only routes for runs, experiments, studies, ensembles, docs, and notes

The health endpoint reports the resolved `workspace_root` and `.numereng` `store_root`.
