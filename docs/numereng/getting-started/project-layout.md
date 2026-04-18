# Project Layout

The repo checkout is the workspace.

## Repo Roots

These are the repo-local extension directories you are expected to read and edit directly:

- `src/numereng/features/models/custom_models/`: built-in and default-discovered custom model wrappers
- `src/numereng/features/agentic_research/programs/`: built-in research program markdown
- `.agents/skills/`: local custom skills; gitignored by default

## Runtime Store

Numereng-managed runtime state lives under `.numereng/`:

```text
.numereng/
  numereng.db
  numereng.db-shm
  numereng.db-wal
  experiments/
  notes/
  runs/
  datasets/
    baselines/
  cache/
    cloud/
    remote_ops/
    tabpfn/
  tmp/
    remote-configs/
  remote_ops/
```

## Experiment Layout

Experiments live under `.numereng/experiments/<experiment_id>/`:

```text
.numereng/experiments/<experiment_id>/
  experiment.json
  EXPERIMENT.md
  EXPERIMENT.pack.md
  configs/
  run_plan.csv
  run_scripts/
  agentic_research/
  hpo/
  ensembles/
  submission_packages/
```

Archived experiments live under `.numereng/experiments/_archive/<experiment_id>/`.

## Run Artifacts

For a scored run, the canonical runtime artifacts live under `.numereng/runs/<run_id>/`:

- `run.json`
- `runtime.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json`
- `artifacts/scoring/*`
- `artifacts/predictions/*.parquet`
- `artifacts/model/*` for full-history refit or artifact-backed serving flows

## Other Important Roots

- `.numereng/datasets/baselines/`: named baselines plus `active_benchmark/`
- `.numereng/notes/`: local notes and rolling research memory
- `.numereng/cache/cloud/`: cloud pull and state staging
- `.numereng/cache/remote_ops/`: legacy-compatible remote pull/cache staging
- `docs/numerai/`: optional synced official Numerai docs mirror

## Path Rules

- Use `--workspace` only when you intentionally need to target another checkout's runtime store or docs mirror paths.
- Training and HPO configs are JSON-only.
- `src/numereng/features/models/custom_models/` is the canonical runtime discovery root for custom model plugins.
- `src/numereng/features/agentic_research/programs/` is the default research program catalog root.
- Managed runtime scratch state stays under `.numereng/`.
