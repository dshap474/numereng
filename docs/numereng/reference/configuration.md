# Configuration

Training and HPO are config-driven, with strict JSON contracts.

## Non-Negotiable Rules

- Training config paths must end in `.json`.
- HPO study config paths must end in `.json`.
- Unknown keys are rejected (`extra=forbid`).
- There is no YAML config contract in the current runtime.

## Training Config Shape

Top-level keys:

- `data` (required)
- `model` (required)
- `training` (required)
- `preprocessing` (optional)
- `output` (optional)

## Minimal Example

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
    },
    "resources": {
      "parallel_folds": 1,
      "parallel_backend": "joblib",
      "memmap_enabled": true
    },
    "cache": {
      "mode": "deterministic",
      "cache_fold_specs": true,
      "cache_features": true,
      "cache_labels": true,
      "cache_fold_matrices": false
    }
  }
}
```

## `data`

Key fields:

- `data_version` (default `v5.2`)
- `dataset_variant` (required: `non_downsampled|downsampled`)
- `feature_set` (default `small`)
- `target_col` (default `target`)
- `target_horizon` (`20d|60d`, optional but preferred for purged walk-forward embargo defaults)
- `era_col` (default `era`)
- `id_col` (default `id`)
- `full_data_path` (optional)
- `benchmark_data_path` (optional)
- `meta_model_data_path` (optional)
- `meta_model_col` (default `numerai_meta_model`)
- `embargo_eras` (legacy compatibility fallback default `13`; profile behavior is driven by `training.engine.profile`)
- `benchmark_model` (default `v52_lgbm_ender20`)
- `baselines_dir` (optional)
- `loading.mode` (`materialized|fold_lazy`)
- `loading.scoring_mode` (`materialized|era_stream`)
- `loading.era_chunk_size` (integer >= 1)

Dataset path behavior:

- `dataset_variant=non_downsampled` default files resolve under `.numereng/datasets/<data_version>/`.
- `dataset_variant=downsampled` remaps `full.parquet` -> `downsampled_full.parquet` and `full_benchmark_models.parquet` -> `downsampled_full_benchmark_models.parquet`.
- Optional official-style downsampling artifacts can be built with `numereng dataset-tools build-full-datasets`.
- Downsample builder writes: `full.parquet`, `full_benchmark_models.parquet`, `downsampled_full.parquet`, `downsampled_full_benchmark_models.parquet`.

## `model`

Required:

- `type`
- `params`

Optional:

- `x_groups`
- `data_needed`
- `module_path`
- `target_transform`
- `prediction_transform`
- `era_weighting`
- `prediction_batch_size`
- `benchmark`
- `baseline` (`name`, `predictions_path`, optional `pred_col`)

## `training`

### `training.engine`

`profile` supports only:

- `simple`
- `purged_walk_forward`
- `submission`

Rules:

- `purged_walk_forward` uses a fixed 156-era walk-forward window; embargo is horizon-derived (`20d -> 8`, `60d -> 16`) and not user-configurable.
- For `purged_walk_forward`, horizon resolution is `target_horizon` first, then `target_col` name inference.
- If `target_horizon` is omitted and `target_col` is ambiguous, config execution fails (`training_engine_target_horizon_ambiguous`).
- Only `training.engine.profile` is accepted (`simple|purged_walk_forward|submission`); legacy engine parameters are rejected.
- Training never applies row-level subsampling; dataset size reduction must happen at dataset construction time.
- `simple` requires split train/validation sources (`dataset_variant=non_downsampled`); with `full_data_path`, sibling `train.parquet` and `validation.parquet` files must exist in the same directory.
- `submission` disables CV metrics.

### `training.resources`

- `parallel_folds` integer >= 1
- `parallel_backend` must be `joblib`
- `memmap_enabled` boolean
- `max_threads_per_worker` integer >= 1 or `"default"` (when `"default"` or omitted: `max(1, floor(available_cpus / parallel_folds))`; `null` is treated the same for backward compatibility)
- `sklearn_working_memory_mib` optional integer >= 1

### `training.cache`

- `mode` must be `deterministic`
- `cache_fold_specs` boolean
- `cache_features` boolean
- `cache_labels` boolean
- `cache_fold_matrices` boolean

## `preprocessing`

- `nan_missing_all_twos` (default `false`)
- `missing_value` (default `2.0`)

Constraint:

- `nan_missing_all_twos=true` is invalid when `data.loading.mode=fold_lazy`.

## `output`

Optional overrides:

- `output_dir`
- `baselines_dir`
- `predictions_name`
- `results_name`

## High-Risk Gotchas

- Benchmark model predictions are metrics-only and are not used as training features.
- `model.x_groups` rejects benchmark aliases (`benchmark`, `benchmarks`, `benchmark_models`) with
  `training_model_x_groups_benchmark_not_supported`.
- For non-`submission` runs, metrics are computed in a post-run scoring phase from the saved predictions parquet.
- If `x_groups` includes `baseline`, `id_col` must be present.
- Training config values are validated before execution; type coercion failures hard-fail.
- The canonical schema is generated from `src/numereng/config/training/contracts.py`.
