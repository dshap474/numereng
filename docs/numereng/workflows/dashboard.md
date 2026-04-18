# Dashboard & Monitor

Numereng ships a repo-local read-only monitoring stack over the current checkout and its `.numereng/` runtime state.

## Launch The Dashboard

Preferred shortcut:

```bash
just viz
```

This starts:

- dashboard UI on [http://127.0.0.1:5173](http://127.0.0.1:5173)
- backend API on [http://127.0.0.1:8502](http://127.0.0.1:8502)

Direct backend command:

```bash
uv run numereng viz --workspace . --host 127.0.0.1 --port 8502
```

Direct backend endpoint:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

## What The Dashboard Reads

- `.numereng/experiments/`
- `.numereng/notes/`
- `.numereng/runs/`
- `.numereng/numereng.db`
- `docs/numereng/`
- `docs/numerai/` when synced locally

The dashboard does not launch runs or mutate experiment/store state.

## Monitor Snapshot

For a CLI-readable summary of the same workspace:

```bash
uv run numereng monitor snapshot --json
```

Optional:

- `--workspace <path>`
- `--no-refresh-cloud`

## Remote-Aware Viz

If you use SSH remote targets, bootstrap the local viz runtime first:

```bash
uv run numereng remote bootstrap-viz
```

This keeps the local dashboard as the read surface while remote data is merged in through the backend.

## Read Next

- [Store Operations](store-ops.md)
- [Remote Operations](remote-ops.md)
