# Project Layout

The repo checkout is the workspace.

## Repo Roots

These are the repo-local extension directories you are expected to read and edit directly:

- `src/numereng/features/models/custom_models/`: tracked custom model plugins
- `src/numereng/features/agentic_research/programs/`: tracked research program markdown
- `.agents/skills/`: repo-local skills

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
  cache/
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
  configs/*.json
  run_scripts/
  agentic_research/
  hpo/
  ensembles/
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
- `artifacts/predictions/*.parquet`

## Path Rules

- Use `--workspace` when you need to target another checkout's runtime store or docs mirror paths.
- Training and HPO configs are JSON-only.
- `src/numereng/features/models/custom_models/` is the canonical runtime discovery root for custom model plugins.
- `src/numereng/features/agentic_research/programs/` is the default research program catalog root.
- Managed runtime scratch state stays under `.numereng/`.
