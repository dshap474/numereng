# Remote Operations

Use `numereng remote` when you want to orchestrate a separate SSH-accessible machine from the current checkout without copying the full `.numereng/` runtime store.

## Use This When

- local compute is not enough but you still want the current repo as the source of truth
- you want to sync experiment authoring files, launch remote training, and pull finished runs back into the local store
- you want the dashboard and monitor surfaces to read a merged local + remote view without starting a separate remote viz app

## Core Commands

Inspect configured targets:

```bash
uv run numereng remote list --format table
uv run numereng remote doctor --target <target_id>
```

Bootstrap the local dashboard’s remote-read support:

```bash
uv run numereng remote bootstrap-viz
```

Sync the current repo checkout to a remote target:

```bash
uv run numereng remote repo sync --target <target_id>
```

Push only one experiment’s authoring files:

```bash
uv run numereng remote experiment sync \
  --target <target_id> \
  --experiment-id <experiment_id>
```

Launch one experiment window remotely:

```bash
uv run numereng remote experiment launch \
  --target <target_id> \
  --experiment-id <experiment_id> \
  --start-index 1 \
  --end-index 5 \
  --score-stage post_training_core
```

Monitor and maintain a remote window:

```bash
uv run numereng remote experiment status --target <target_id> --experiment-id <experiment_id>
uv run numereng remote experiment maintain --target <target_id> --experiment-id <experiment_id>
uv run numereng remote experiment stop --target <target_id> --experiment-id <experiment_id>
```

Pull finished remote runs back into the local store. `--mode` is required and selects which artifact subset is copied:

```bash
# Dashboard-only: scoring artifacts + root manifest files. ~200 KB/run.
# Enables the Performance tab; no submit/ensemble/rescore locally.
uv run numereng remote experiment pull \
  --target <target_id> \
  --experiment-id <experiment_id> \
  --mode scoring

# Full: entire run directory including predictions parquets. ~140 MB/run.
# Required if you plan to submit, ensemble, or rescore the run locally.
uv run numereng remote experiment pull \
  --target <target_id> \
  --experiment-id <experiment_id> \
  --mode full
```

### Pull modes

| Mode      | What it copies                                                                                      | Typical size per run | Enables locally                          |
| --------- | --------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------- |
| `scoring` | `artifacts/scoring/*` + `run.json`, `resolved.json`, `results.json`, `metrics.json`, `run.log`, `score_provenance.json` | ~200 KB              | dashboard Performance tab, Run Ops metrics |
| `full`    | entire `.numereng/runs/<run_id>/` subtree including `artifacts/predictions/*`                      | ~140 MB              | everything above plus submit / ensemble / neutralize / rescore |

Mode interactions with already-materialized local runs:

- If local is already **full**, any `--mode` is a no-op (already materialized). Local is never downgraded.
- If local is **scoring** and you request `--mode scoring`, it's a no-op.
- If local is **scoring** and you request `--mode full`, the run is re-pulled and the existing local scoring tree is replaced by the full tree (predictions overlayed in). Those run ids come back in `partially_materialized_run_ids` so the caller can tell upgrades apart from fresh pulls.

Push one ad hoc config and run it remotely:

```bash
uv run numereng remote config push --target <target_id> --config configs/run.json
uv run numereng remote run train --target <target_id> --config configs/run.json
```

## Recovery: Restoring Missing Local Artifacts

If `store doctor` reports `run_count_mismatch` with `filesystem_runs < indexed_runs`, local artifacts were pruned or lost while SQLite still knows about the runs. The dashboard's Performance tab will render empty because it reads from `artifacts/scoring/*` on disk. If a remote target still has the artifacts, pull them back:

```bash
# Cheap, dashboard-only fidelity (~200 KB per run)
uv run numereng remote experiment pull \
  --target <target_id> \
  --experiment-id <experiment_id> \
  --mode scoring

# Full artifacts (~140 MB per run). Required if you plan to submit,
# ensemble, neutralize, or rescore those runs locally.
uv run numereng remote experiment pull \
  --target <target_id> \
  --experiment-id <experiment_id> \
  --mode full
```

Re-run the command per experiment. `scoring` is idempotent and also upgrades to `full` automatically on a later `--mode full` pull. A `full` local run is never downgraded by a subsequent `scoring` pull.

## What Syncs And What Does Not

- `remote repo sync` mirrors the git-visible working tree only.
- It does **not** sync the full `.numereng/` runtime store.
- `remote experiment sync` mirrors experiment authoring files only.
- `remote experiment pull --mode <scoring|full>` materializes eligible finished remote runs into local `.numereng/runs/<run_id>/`.

## Local State And Artifacts

Remote orchestration state is stored under:

- `.numereng/remote_ops/`
- `.numereng/cache/remote_ops/`

Pulled finished runs become normal local runs under `.numereng/runs/`, so later scoring, ensemble, serving, and dashboard flows can treat them like local runs.

## High-Risk Gotchas

- remote targets are declarative; the target definition must already exist and be valid
- repo sync excludes `.numereng/`, env files, and local machine-specific profiles
- experiment pull only materializes finished runs; incomplete remote runs stay remote
- remote experiment launch/status/maintain/stop operate on explicit experiment windows, so keep the same target, experiment ID, and index window across commands

## Read Next

- [Experiments](experiments.md)
- [Cloud Training](cloud-training.md)
- [Dashboard & Monitor](dashboard.md)
