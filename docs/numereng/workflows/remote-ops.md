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

Pull finished remote runs back into the local store:

```bash
uv run numereng remote experiment pull \
  --target <target_id> \
  --experiment-id <experiment_id>
```

Push one ad hoc config and run it remotely:

```bash
uv run numereng remote config push --target <target_id> --config configs/run.json
uv run numereng remote run train --target <target_id> --config configs/run.json
```

## What Syncs And What Does Not

- `remote repo sync` mirrors the git-visible working tree only.
- It does **not** sync the full `.numereng/` runtime store.
- `remote experiment sync` mirrors experiment authoring files only.
- `remote experiment pull` materializes eligible finished remote runs into local `.numereng/runs/<run_id>/`.

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
