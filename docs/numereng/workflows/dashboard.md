# Dashboard

Numereng ships a read-only monitoring stack:

- backend: `src/numereng/features/viz/`
- frontend: `viz/web/`

The dashboard reads store state. Launch and control actions still happen through the CLI or Python API.

## Start

```bash
make viz
```

Default local endpoints:

- API: `http://127.0.0.1:8502`
- web app: `http://127.0.0.1:5173`

Stop both:

```bash
make kill-viz
```

## Requirements

- Node.js 20+
- `npm`
- a populated `.numereng/` store for useful data

## What The Dashboard Reads

The viz backend exposes read-only views over:

- experiments and experiment configs
- run manifests, metrics, events, resources, and resolved configs
- HPO studies and trials
- ensembles and ensemble diagnostics
- docs trees
- notes

It does not launch jobs or mutate experiment/store state.

## Backend Contract

The FastAPI app reports:

- `GET /healthz`
- `GET /`
- `GET /api/...` read-only routes for runs, experiments, studies, ensembles, docs, and notes

The backend enables CORS for the local frontend and reports `read_only=true` from its health endpoint.

## Operational Notes

- the dashboard reflects whatever is indexed and present under the current store root
- if run artifacts are missing or not yet indexed, the dashboard surfaces them as unavailable rather than recomputing them
- for drift or reindex work, use the [Store Operations](store-ops.md) workflow
