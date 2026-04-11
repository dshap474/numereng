# Viz monitoring

## Symptoms

- `/healthz` fails
- `/experiments` or experiment detail pages show stale state
- run detail charts are missing even though a run finished

## First checks

1. Start the dashboard with `numereng viz` from the workspace root. If you are debugging the source checkout itself, `just viz` remains a contributor wrapper.
2. Confirm the API health check: `http://127.0.0.1:8502/healthz`.
3. Inspect the backend stdout/stderr for the running `numereng viz` process. If you launched via `just viz`, inspect `viz/api.log` and `viz/vite.log`.
4. Confirm the underlying run artifacts exist under `.numereng/runs/<run_id>/`.

## Contract reminders

- Viz is monitor-only; launch/control flows remain CLI/API-driven.
- Progress state is backend-owned and sourced from lifecycle/store data.
- Persisted scoring artifacts are the preferred source for run charts.

## Common recovery paths

- If charts are missing for older runs, backfill canonical scoring artifacts through store tooling.
- If monitor state is stale, inspect lifecycle reconciliation before changing frontend code.
- If the API is healthy but the UI is stale, inspect the read-only backend routes before touching Svelte code.
