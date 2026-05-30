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

## Experiment Record Direction Model

There are three `remote experiment` verbs with distinct data directions:

| Verb | Direction | What it moves |
| ---- | --------- | ------------- |
| `remote experiment sync` | PUSH ↑ | Authoring bundle: `EXPERIMENT.md`, `run_plan.csv`, `configs/`, `run_scripts/`. Does **not** push the `agentic_research/` record. |
| `remote experiment fetch` | PULL ↓ | Experiment record written by the controller: `agentic_research/` (state.json, trace.jsonl, rounds/), `configs/`, `EXPERIMENT.md`, `run_plan.csv`, `run_scripts/`. Does **not** pull `runs/` artifacts. |
| `remote experiment pull --mode scoring\|full` | PULL ↓ | Run artifacts only: `runs/<run_id>/` subtree. Does **not** touch the experiment record. |

**Direction gotcha — do not use `sync` to "pull" remote state.** Running `remote experiment sync` when the remote is ahead (e.g., the controller has updated `run_plan.csv` or `EXPERIMENT.md`) will overwrite the remote’s newer files with your stale local copies. Use `remote experiment fetch` when you want to bring the controller-written experiment record back to your local machine.

Push only one experiment’s authoring files:

```bash
uv run numereng remote experiment sync \
  --target <target_id> \
  --experiment-id <experiment_id>
```

Pull the experiment record (controller-written state) from remote down to the local store:

```bash
uv run numereng remote experiment fetch \
  --target <target_id> \
  --experiment-id <experiment_id>
```

The fetch is incremental (tracked via a `experiment_fetch__<id>.json` marker) and idempotent; re-running it is safe.

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
- `remote experiment sync` mirrors experiment authoring files **up** to the remote (PUSH). It does not pull.
- `remote experiment fetch` pulls the controller-written experiment record **down** from the remote (PULL). It does not pull `runs/` artifacts.
- `remote experiment pull --mode <scoring|full>` materializes eligible finished remote runs into local `.numereng/runs/<run_id>/`. It does not touch the experiment record.

## Local State And Artifacts

Remote orchestration state is stored under:

- `.numereng/remote_ops/`
- `.numereng/cache/remote_ops/`

Pulled finished runs become normal local runs under `.numereng/runs/`, so later scoring, ensemble, serving, and dashboard flows can treat them like local runs.

## High-Risk Gotchas

- remote targets are declarative; the target definition must already exist and be valid
- repo sync excludes `.numereng/`, env files, and local machine-specific profiles
- **do not run `remote experiment sync` to retrieve remote state** — it pushes your local authoring files to the remote and will clobber a remote `run_plan.csv` or `EXPERIMENT.md` that is ahead of your local copy; use `remote experiment fetch` instead
- experiment pull only materializes finished runs; incomplete remote runs stay remote
- remote experiment launch/status/maintain/stop operate on explicit experiment windows, so keep the same target, experiment ID, and index window across commands

## Read Next

- [Experiments](experiments.md)
- [Cloud Training](cloud-training.md)
- [Dashboard & Monitor](dashboard.md)
