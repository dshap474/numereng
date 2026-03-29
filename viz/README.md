# viz

`viz` is numereng’s read-only monitoring surface.

It has two parts:
- `viz/api/numereng_viz`: the FastAPI backend that builds workspace overviews from one local store plus optional SSH remote stores
- `viz/web`: the Svelte mission-control frontend

The key design choice is federated monitoring:
- every machine keeps its own `.numereng` store
- the local viz backend reads the local store directly
- enabled SSH remotes are queried through `numereng monitor snapshot --json`
- the frontend renders one merged overview

This avoids syncing SQLite stores between machines while still giving one dashboard for local and remote runs.

## What Mission Control Assumes

- `/api/experiments/overview` is the only source of truth for the experiments page
- experiment ids are not globally unique across sources
- remote snapshots can be slower than local snapshots
- short runs may move from live to terminal within a single manual observation window

## Common Failure Modes

- Duplicate experiment ids across local and remote stores:
  the frontend must key lists by source plus experiment id, not experiment id alone.
- Stale header pulse:
  use overview `generated_at`, not just experiment timestamps.
- Missing live rows during remote monitoring:
  check whether the browser tab is stale before assuming backend failure. Compare the page against `/api/experiments/overview`.
- Overlapping client polls:
  the experiments page should allow only one in-flight overview refresh at a time.

## Useful Local Checks

- Backend health: `curl http://127.0.0.1:8502/healthz`
- Current overview: `curl http://127.0.0.1:8502/api/experiments/overview`
- Remote snapshot directly:
  `ssh <target> "… uv run numereng monitor snapshot --store-root <store> --json"`

## Related Files

- Backend launcher: [api.py](/path/to/numereng/viz/api.py)
- Backend snapshot merge: [monitor_snapshot.py](/path/to/numereng/viz/api/numereng_viz/monitor_snapshot.py)
- Frontend page: [+page.svelte](/path/to/numereng/viz/web/src/routes/experiments/+page.svelte)
- Frontend route load: [+page.ts](/path/to/numereng/viz/web/src/routes/experiments/+page.ts)
