# viz Architecture

## Overview

`viz` is numereng’s workspace monitoring package.

It is intentionally read-only. It does not own run execution, store mutation, or remote state replication. Its job is to present one consistent monitoring surface over multiple independent numereng stores.

## High-Level Shape

```text
local browser
    |
    v
Svelte mission control (`viz/web`)
    |
    v
FastAPI backend (`viz/api/numereng_viz`)
    |
    +--> local store snapshot
    |      `build_monitor_snapshot(store_root=...)`
    |
    +--> SSH remote snapshot(s)
           `ssh <target> ... numereng monitor snapshot --json`
```

## Why Federation Exists

Numereng stores are SQLite plus filesystem run state. That is a good fit for one machine, but a bad fit for live multi-machine replication.

So `viz` uses federation:
- each machine keeps its own `.numereng`
- the local backend fetches a normalized snapshot from each store
- the UI renders one merged overview
- completed remote experiments can be pulled back into canonical local run storage under `.numereng/runs/<run_id>`

This avoids:
- SQLite-over-network problems
- active lifecycle replication and cross-machine DB coupling
- accidental cross-machine state corruption

## Backend Layers

### `viz/api.py`

Tiny launcher shim. No business logic should live here.

### `viz/api/numereng_viz/services.py`

Thin orchestration layer. For mission control it does:
- local snapshot build
- remote snapshot fetch
- merged overview return

### `viz/api/numereng_viz/monitor_snapshot.py`

This is the core mission-control integration layer.

Responsibilities:
- build one normalized store snapshot
- decorate entries with source metadata
- fetch SSH remote snapshots
- merge local + remote experiments/live activity
- canonicalize duplicate local+remote experiment ids into one local-primary row with remote overlay metadata
- emit one merged overview payload with `generated_at`

Important invariants:
- remote source success does not imply remote live runs
- overview merge must preserve remote-only rows but collapse exact local+remote experiment-id matches into one canonical local row
- merged overview must emit a current `generated_at`

### `viz/api/numereng_viz/store_adapter.py`

Local store reader and overview builder for one store.

Important invariants:
- live rows come from real lifecycle/cloud state, not stale active-looking jobs
- experiment attribution may be synthesized when remote/external config runs have no indexed experiment row
- config labels must tolerate Windows-style paths
- local-first detail reads resolve canonical local artifacts first, then pulled remote cache artifacts, then SSH fallback

## Frontend Layers

### `viz/web/src/routes/experiments/+page.ts`

Initial route load for mission control.

Important invariant:
- fetch the real `/api/experiments/overview` on load
- fallback-only parent experiment data is not sufficient for federated mission control

### `viz/web/src/routes/experiments/+page.svelte`

Live mission-control page.

Important invariants:
- treat `/api/experiments/overview` as the single source of truth
- allow only one in-flight overview refresh at a time
- hold the last good snapshot on failure
- keyed lists must be source-aware

Required keys:

```text
experiments      -> source_kind:source_id:experiment_id
live experiments -> source_kind:source_id:experiment_id
recent activity  -> source + job/run identity
```

If the page keys by `experiment_id` only, remote-only rows can still collide with each other. For canonical local+remote duplicates, the backend now collapses them before the frontend sees them.

## Polling and Freshness

Mission control uses a 3-second poll.

The critical client rule is single-flight polling:
- start one overview request
- do not start another until the first resolves
- if a request fails, keep the last good snapshot and mark feed state as holding

Why:
- SSH remote snapshots can be slower than local-only responses
- overlapping requests can cause the UI to discard slow successful responses and remain on an older payload

## Pulse Semantics

The mission-control header needs a reliable freshness signal.

Use:
- `overview.generated_at`

Do not rely only on:
- first experiment timestamp
- last activity of a single row

Reason:
- remote and local experiments can be terminal while the overview itself is still fresh
- pulse should represent overview freshness, not just workload freshness

## Debugging Checklist

When a user says “the live run is not showing”:

1. Check backend overview directly:
   - `curl http://127.0.0.1:8502/api/experiments/overview`
2. Check remote snapshot directly:
   - `ssh <target> ... numereng monitor snapshot --json`
3. Check whether the run simply finished before the screenshot:
   - inspect remote `runtime.json`
4. Check for client-side stale tab/session:
   - compare visible `Last pulse` against `overview.generated_at`
5. Check whether the backend collapsed a local+remote duplicate into one canonical row with overlay metadata

## Problems Solved in This Iteration

The federated mission-control work uncovered and fixed these concrete issues:

1. Duplicate local+remote experiment rows
- local and remote stores can both have the same experiment id
- fix: backend canonicalizes to one local-primary row and keeps remote freshness as overlay metadata

2. Fallback-only initial route load
- page initially rendered parent experiments only
- fix: fetch real overview in `+page.ts`

3. Misleading pulse
- header could show stale experiment timestamps instead of current overview freshness
- fix: emit and use merged `generated_at`

4. Overlapping client polls
- slow SSH overview refreshes could be overtaken by the next interval tick
- fix: single-flight polling

5. Remote attribution gaps
- remote/external-config runs needed synthesized experiment rows, Windows path-safe config labels, and a local-first fallback path for pulled cache artifacts

## Future Hardening

Useful next hardening steps:
- add an explicit frontend test that verifies canonical local+remote experiment collapse plus overlay rendering
- add a browser test that simulates a slow overview response and verifies single-flight polling
- expose poll latency in a lightweight debug panel or response header for mission-control troubleshooting
- consider marking the current overview source age directly in the UI when the feed is holding
