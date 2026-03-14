# Configuration

Numereng is config-driven. Training and HPO use strict JSON contracts and reject unknown keys.

## Source Of Truth

- training contract: `src/numereng/config/training/contracts.py`
- HPO contract: `src/numereng/config/hpo/contracts.py`

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
    "target_col": "target",
    "loading": {
      "mode": "materialized",
      "scoring_mode": "materialized",
      "era_chunk_size": 64
    }
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
    }
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
- `loading.mode`: `materialized|fold_lazy`
- `loading.scoring_mode`: `materialized|era_stream`
- `loading.era_chunk_size`: integer >= 1

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
- plugin models can be loaded from `src/numereng/features/models/custom_models/`
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

- `nan_missing_all_twos=true` is invalid when `data.loading.mode=fold_lazy`

## `output`

Optional overrides:

- `output_dir`
- `baselines_dir`
- `predictions_name`
- `results_name`

## HPO Study Config Shape

HPO study configs are also JSON-only. The canonical model is `HpoStudyConfig`.

Important fields:

- `study_name`
- `config_path`
- `experiment_id`
- `metric`
- `direction`
- `n_trials`
- `sampler`
- `seed`
- `search_space`
- `neutralization`

Minimal example:

```json
{
  "study_name": "lgbm-sweep",
  "config_path": "configs/run.json",
  "metric": "bmc_last_200_eras.mean",
  "direction": "maximize",
  "n_trials": 50,
  "sampler": "tpe"
}
```

## High-Risk Gotchas

- numereng does not emit `payout_estimate_mean`
- benchmark predictions are metrics-only and are not generic training features
- `model.x_groups` and `model.data_needed` are feature-only by default
- `model.x_groups` rejects `era`, `id`, and benchmark aliases
- for non-`full_history_refit` runs, metrics are computed from saved predictions in the post-run scoring stage
- canonical FNC neutralizes to `fncv3_features` and then correlates against the scoring target being evaluated
- benchmark and meta-model joins require strict era alignment
- if `neutralization.enabled=true` in an HPO config, `neutralizer_path` is required
