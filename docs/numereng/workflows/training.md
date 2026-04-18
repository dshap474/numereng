# Training Models

Use `run train` when you want one local run that is not primarily managed as an experiment round.

## Use This When

- you are validating a single config quickly
- you do not need experiment-local reports, run plans, or champion tracking
- you want one run ID linked to one JSON config

For tracked model-development work, prefer [Experiments](experiments.md).

## Prerequisites

- `.numereng/` is initialized
- required Numerai dataset parquet files already exist locally
- your config path ends in `.json`

## Minimal Config

```json
{
  "data": {
    "data_version": "v5.2",
    "dataset_variant": "non_downsampled",
    "feature_set": "small",
    "target_col": "target"
  },
  "model": {
    "type": "LGBMRegressor",
    "params": {
      "n_estimators": 2000,
      "learning_rate": 0.01,
      "num_leaves": 64,
      "colsample_bytree": 0.1,
      "random_state": 42
    }
  },
  "training": {
    "engine": {
      "profile": "purged_walk_forward"
    },
    "post_training_scoring": "none"
  }
}
```

## Train

```bash
uv run numereng run train --config configs/run.json
```

Useful overrides:

- `--output-dir <path>`
- `--profile <simple|purged_walk_forward|full_history_refit>`
- `--experiment-id <id>`
- `--post-training-scoring <none|core|full|round_core|round_full>`

## Re-Score A Saved Run

```bash
uv run numereng run score --run-id <run_id>
```

Use `run score` when predictions already exist and you want to rebuild:

- `results.json`
- `metrics.json`
- `score_provenance.json`
- store index rows

## Run Outputs

Successful runs write under `.numereng/runs/<run_id>/`:

- `run.json`
- `runtime.json`
- `run.log`
- `resolved.json`
- `results.json`
- `metrics.json`
- `artifacts/predictions/*`
- optional `artifacts/scoring/*`
- optional `artifacts/model/*`

## High-Risk Gotchas

- training and HPO configs are JSON-only and reject unknown keys
- supported profiles are only `simple`, `purged_walk_forward`, and `full_history_refit`
- `full_history_refit` is final-fit only and emits no validation metrics
- `round_core` and `round_full` are experiment-oriented policies; use them with `experiment train`
- training requires successful run indexing before the command is considered successful
- default benchmark-relative scoring expects an active benchmark unless you configure `benchmark_source.source = "path"`

## Read Next

- [Configuration](../reference/configuration.md)
- [Experiments](experiments.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
