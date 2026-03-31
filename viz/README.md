# viz

`viz` is numereng’s read-only monitoring surface.

It has two parts:
- `viz/api/numereng_viz`: the FastAPI backend that builds workspace overviews from one local store plus optional SSH remote stores
- `viz/web`: the Svelte mission-control frontend

The key design choice is federated monitoring:
- every machine keeps its own `.numereng` store
- the local viz backend reads the local store directly
- enabled SSH remotes are queried through `numereng monitor snapshot --json`
- completed remote experiments can be pulled back as lightweight viz caches under `.numereng/cache/remote_ops/pulls/<target_id>/...`
- the frontend renders one merged overview with local-primary canonical rows when experiment ids overlap
- remote experiment and run detail pages stay on the local viz server, but default resolution is now local store first, then pulled cache, then source-aware SSH fallback when needed

This avoids syncing SQLite stores or full remote run directories between machines while still giving one dashboard for local and remote runs.

## What Mission Control Assumes

- `/api/experiments/overview` is the only source of truth for the experiments page
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

## Useful Local Checks

- Backend health: `curl http://127.0.0.1:8502/healthz`
- Current overview: `curl http://127.0.0.1:8502/api/experiments/overview`
- Remote snapshot directly:
  `ssh <target> "… uv run numereng monitor snapshot --store-root <store> --json"`

## Related Files

- Backend launcher: [api.py](/Users/daniel/Developer/numereng/viz/api.py)
- Backend snapshot merge: [monitor_snapshot.py](/Users/daniel/Developer/numereng/viz/api/numereng_viz/monitor_snapshot.py)
- Frontend page: [+page.svelte](/Users/daniel/Developer/numereng/viz/web/src/routes/experiments/+page.svelte)
- Frontend route load: [+page.ts](/Users/daniel/Developer/numereng/viz/web/src/routes/experiments/+page.ts)
