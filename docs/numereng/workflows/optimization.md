# Hyperparameter Optimization

Use `numereng hpo` when you want Optuna to search one config space while numereng persists study state, trial configs, and trial runs in the workspace.

## Create A Study

Preferred path:

```bash
uv run numereng hpo create --study-config configs/hpo/study.json
```

Inline alternative:

```bash
uv run numereng hpo create \
  --study-id ender20_lgbm_gpu_v1 \
  --study-name lgbm-sweep \
  --config configs/run.json \
  --search-space '{"model.params.learning_rate":{"type":"float","low":0.001,"high":0.05,"log":true}}' \
  --metric bmc_last_200_eras.mean \
  --direction maximize \
  --n-trials 50 \
  --sampler tpe
```

## Inspect Studies

```bash
uv run numereng hpo list
uv run numereng hpo details --study-id <study_id>
uv run numereng hpo trials --study-id <study_id>
```

## Artifact Locations

Studies are stored either:

- globally under `.numereng/hpo/<study_id>/`
- or under an experiment at `.numereng/experiments/<experiment_id>/hpo/<study_id>/`

Typical files:

- `study_spec.json`
- `study_summary.json`
- `optuna_journal.log`
- `configs/trial_0000.json`
- `trials_live.parquet`

Each trial still produces a normal run under `.numereng/runs/`.

## High-Risk Gotchas

- study and training configs are JSON-only
- rerunning `numereng hpo create` with the same immutable study spec resumes the study instead of creating a new one
- `--neutralize` requires `--neutralizer-path`
- trial indexing is mandatory; indexing failures fail the trial
- identical parameter draws may reuse completed deterministic results instead of retraining

## Read Next

- [Experiments](experiments.md)
- [Metrics](../reference/metrics.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
