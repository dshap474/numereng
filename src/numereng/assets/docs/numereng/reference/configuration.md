# Configuration

Numereng is config-driven. Training and HPO use strict JSON contracts and reject unknown keys.

## Source Of Truth

- training configs are validated by the live numereng training contract
- HPO study configs are validated by the live numereng HPO contract

Generated or copied templates must stay aligned with those contracts.

## Non-Negotiable Rules

- training config paths must end in `.json`
- HPO study config paths must end in `.json`
- unknown keys are rejected
- legacy training engine knobs are rejected
- numereng does not support YAML runtime configs

## Training Config Shape

Top-level keys:

- `data` required
- `model` required
- `training` required
- `preprocessing` optional
- `output` optional

Minimal example:

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

## `data`

Important fields:

- `data_version` default `v5.2`
- `dataset_variant` required: `non_downsampled|downsampled`
- `feature_set` default `small`
- `target_col` default `target`
- `target_horizon` optional but preferred for purged walk-forward
- `era_col` default `era`
- `id_col` default `id`
- `benchmark_source` optional override block for benchmark scoring input
- `meta_model_data_path` optional override
- `meta_model_col` default `numerai_meta_model`
- `benchmark_source.source` default `active` (`active|path`)
- `benchmark_source.predictions_path` required only when `source=path`
- `benchmark_source.pred_col` default `prediction`
- `benchmark_source.name` optional provenance label

Benchmark source clarification:

- default `benchmark_source.source = "active"` requires a seeded shared
  artifact under `.numereng/datasets/baselines/active_benchmark/`
- use `benchmark_source.source = "path"` when bootstrapping a new machine or
  when you want scoring to use one explicit baseline parquet without changing
  the shared default
- the official `.numereng/datasets/<data_version>/*benchmark_models.parquet`
  files are separate dataset inputs and are not the same thing as the active
  benchmark scoring artifact
- see [Baselines & Active Benchmark](../workflows/baselines.md)

Dataset path behavior:

- `non_downsampled` defaults resolve under `.numereng/datasets/<data_version>/`
- `non_downsampled` canonical storage is split-source only: `train.parquet` and `validation.parquet`
- `downsampled` uses stored derived artifacts: `downsampled_full.parquet` and `downsampled_full_benchmark_models.parquet`
- dataset row paths are package-managed from `data_version`, `dataset_variant`, training profile, and `dataset_scope`
- `full_data_path` is removed from the public config contract
- `dataset-tools build-downsampled-full` materializes only the downsampled derived artifacts

## `model`

Required:

- `type`
- `params`

Optional:

- `x_groups`
- `data_needed`
- `module_path`
- `target_transform`
- `benchmark`
- `baseline`

Current model notes:

- built-in model type: `LGBMRegressor`
- plugin models can be loaded from `custom_models/`
- `module_path` is optional if the requested plugin type can be discovered in `custom_models/`

## `training`

### `training.engine.profile`

Supported values:

- `simple`
- `purged_walk_forward`
- `full_history_refit`

Current behavior:

- `purged_walk_forward` uses a fixed 156-era walk-forward window
- embargo defaults come from the target horizon: `20d -> 8`, `60d -> 16`
- if `target_horizon` is omitted and `target_col` is ambiguous, training fails
- `simple` requires split train/validation sources
- `full_history_refit` is final-fit only and emits no validation metrics

### `training.post_training_scoring`

Allowed values:

- `none` default
- `core`
- `full`
- `round_core`
- `round_full`

Behavior:

- `none` leaves post-training scoring deferred
- `core` auto-materializes `post_training_core` after training
- `full` auto-materializes inclusive `post_training_full` after training
- `round_core` and `round_full` are experiment-only policies that defer single-run
  scoring and instead trigger one round batch pass after the run is linked into
  the experiment manifest
- CLI or API overrides take precedence over the config value

### `training.resources`

Important fields:

- `parallel_folds`
- `parallel_backend` must be `joblib`
- `memmap_enabled`
- `max_threads_per_worker`
- `sklearn_working_memory_mib`

### `training.cache`

Important fields:

- `mode` must be `deterministic`
- `cache_fold_specs`
- `cache_features`
- `cache_labels`
- `cache_fold_matrices`

## `preprocessing`

Supported fields:

- `nan_missing_all_twos`
- `missing_value`

Constraint:

- `nan_missing_all_twos` and `missing_value` apply uniformly under the single materialized training loader

## `output`

Optional overrides:

- `output_dir`
- `baselines_dir`
- `predictions_name`
- `results_name`

## HPO Study Config Shape

HPO study configs are also JSON-only. The canonical model is `HpoStudyConfig`.

Important fields:

- `study_id`
- `study_name`
- `config_path`
- `experiment_id`
- `objective`
- `search_space`
- `sampler`
- `stopping`

Minimal example:

```json
{
  "study_id": "ender20_lgbm_gpu_v1",
  "study_name": "lgbm-sweep",
  "config_path": "configs/run.json",
  "objective": {
    "metric": "post_fold_champion_objective",
    "direction": "maximize",
    "neutralization": {
      "enabled": false,
      "neutralizer_path": null,
      "proportion": 0.5,
      "mode": "era",
      "neutralizer_cols": null,
      "rank_output": true
    }
  },
  "search_space": {
    "model.params.learning_rate": {
      "type": "float",
      "low": 0.001,
      "high": 0.05,
      "log": true
    }
  },
  "sampler": {
    "kind": "tpe",
    "seed": 1337,
    "n_startup_trials": 10,
    "multivariate": true,
    "group": false
  },
  "stopping": {
    "max_trials": 50,
    "max_completed_trials": null,
    "timeout_seconds": null,
    "plateau": {
      "enabled": false,
      "min_completed_trials": 15,
      "patience_completed_trials": 10,
      "min_improvement_abs": 0.00025
    }
  }
}
```

## High-Risk Gotchas

- numereng does not emit `payout_estimate_mean`
- benchmark predictions are metrics-only and are not generic training features
- `model.x_groups` and `model.data_needed` are feature-only by default
- `model.x_groups` rejects `era`, `id`, and benchmark aliases
- training always uses the materialized loader and the materialized scorer
- `training.post_training_scoring` defaults to `none`; `run train` only auto-scores when the resolved policy is `core` or `full`
- `round_core` and `round_full` require `experiment train` with an `rN_*` config stem
- canonical FNC neutralizes to `fncv3_features` and then correlates against the scoring target being evaluated
- benchmark and meta-model joins require strict era alignment
- if `objective.neutralization.enabled=true` in an HPO config, `neutralizer_path` is required
- `search_space` is required; numereng no longer infers HPO spaces from base model params
- HPO v2 config is a clean break from the old flat study config shape
- `sampler.kind=random` only allows `kind` and `seed`; TPE-only fields are invalid
- `stopping.timeout_seconds` is a per-`hpo create` invocation budget; resumed studies do not accumulate prior wall-clock time
- `post_fold_champion_objective` reads `post_fold_snapshots.parquet` first with `corr_ender20_fold_mean -> corr_native_fold_mean` and `bmc_fold_mean -> bmc_ender20_fold_mean` fallback, then falls back to `results.json`
- repeated identical HPO params reuse a completed trial value or a finished deterministic run when the scoring artifacts are intact; pre-existing non-finished run dirs fail loudly and are not reset automatically
