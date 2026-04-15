# viz

`viz` is numereng’s read-only monitoring surface.

It has two parts:
- `viz/api/numereng_viz`: the FastAPI backend that builds workspace overviews from one local store plus optional SSH remote stores
- `viz/web`: the Svelte mission-control frontend

The key design choice is federated monitoring:
- every machine keeps its own `.numereng` store
- the local viz backend reads the local store directly
- enabled SSH remotes are queried through `numereng monitor snapshot --json`
- completed remote experiments can be pulled back into canonical local run storage under `.numereng/runs/<run_id>`
- the frontend renders one merged overview with local-primary canonical rows when experiment ids overlap
- remote experiment and run detail pages stay on the local viz server, but local detail requests now return canonical local experiment/run artifacts directly without probing remote overlays. Explicit remote source routes and genuine local misses still use source-aware SSH fallback, with legacy pulled-cache compatibility only as a last read-only bridge.

This keeps one dashboard for local and remote runs without syncing SQLite stores, while still allowing finished remote history to be materialized locally when that is the preferred operating mode.

The frontend is SSR-first again:
- browser requests still use `/api`
- server-side route loads default to `http://127.0.0.1:8502/api`
- set `VIZ_API_BASE` if the local API is bound somewhere else
- the root shell stays local-fast and does not fetch remote-aware mission-control data

## What Mission Control Assumes

- `/api/experiments/overview` is the only source of truth for the experiments page
- `/api/experiments/overview?include_remote=false` is the fast local-first load path for SSR and first paint
- experiment ids may exist on both local and remote stores
- the overview should collapse local+remote duplicates into one canonical local row when the experiment id matches
- experiment and run detail routes keep the existing local URL shape and add optional `source_kind` / `source_id` query params only when explicit remote identity is required
- remote snapshots can be slower than local snapshots
- short runs may move from live to terminal within a single manual observation window

## Common Failure Modes

- Duplicate experiment ids across local and remote stores:
  the backend canonicalizes these into one local-primary row, but remote-only rows still need source-aware frontend keys.
- Stale header pulse:
  use overview `generated_at`, not just experiment timestamps.
- Missing live rows during remote monitoring:
  check whether the browser tab is stale before assuming backend failure. Compare the page against `/api/experiments/overview`.
- Overlapping client polls:
  the experiments page should allow only one in-flight overview refresh at a time.
- Slow unrelated routes:
  verify they are not accidentally route-loading remote overview data from the global shell.

## Useful Local Checks

- Backend health: `curl http://127.0.0.1:8502/healthz`
- Current overview: `curl http://127.0.0.1:8502/api/experiments/overview`
- Local-only overview: `curl 'http://127.0.0.1:8502/api/experiments/overview?include_remote=false'`
- Partial run bundle: `curl 'http://127.0.0.1:8502/api/runs/<run_id>/bundle?sections=manifest,metrics,scoring_dashboard'`
- Remote snapshot directly:
  `ssh <target> "… uv run numereng monitor snapshot --workspace <workspace> --json"`

## Related Files

- Backend launcher: [`viz/api.py`](./api.py)
- Backend snapshot merge: [`viz/api/numereng_viz/monitor_snapshot.py`](./api/numereng_viz/monitor_snapshot.py)
- Frontend page: [`viz/web/src/routes/experiments/+page.svelte`](./web/src/routes/experiments/+page.svelte)
- Frontend route load: [`viz/web/src/routes/experiments/+page.ts`](./web/src/routes/experiments/+page.ts)
