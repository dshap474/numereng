# Project Layout

Numereng has two layouts you need to keep straight:

- the user-facing workspace root
- the hidden runtime store under `.numereng/`

## Workspace Root

These are the directories you are expected to read and edit directly:

- `experiments/`: experiment manifests, notes, configs, launcher scripts, HPO and ensemble outputs
- `notes/`: general workspace notes plus `__RESEARCH_MEMORY__/`
- `custom_models/`: user-authored model plugins
- `research_programs/`: user-authored research program markdown
- `.agents/skills/`: numereng-owned shipped skills plus any local skill additions

## Runtime Store

Numereng-owned runtime state lives under `.numereng/`:

```text
.numereng/
  numereng.db
  numereng.db-shm
  numereng.db-wal
  runs/
  datasets/
  cache/
    tabpfn/
  tmp/
    remote-configs/
  remote_ops/
```

## Experiment Layout

Experiments live under `experiments/<experiment_id>/`:

```text
experiments/<experiment_id>/
  experiment.json
  EXPERIMENT.md
  EXPERIMENT.pack.md
  configs/*.json
  run_scripts/
  agentic_research/
  hpo/
  ensembles/
```

Archived experiments live under `experiments/_archive/<experiment_id>/`.

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

- Use `--workspace` when you need to target a workspace other than the current directory.
- Training and HPO configs are JSON-only.
- `custom_models/` is the canonical runtime discovery root for custom model plugins.
- `research_programs/` is resolved before packaged built-ins.
- Managed runtime scratch state stays under `.numereng/`, not the workspace root.
