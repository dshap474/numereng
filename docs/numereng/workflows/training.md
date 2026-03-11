# Training Models

Train a single run from an explicit JSON config.

## When to Use

Use this workflow for any direct training run that is not tied to experiment metadata.

## Minimal Training Config

Training configs are JSON-only and must follow `src/numereng/config/training/contracts.py`.

```json
{
  "data": {
    "data_version": "v5.2",
    "dataset_variant": "non_downsampled",
    "feature_set": "small",
    "target_col": "target",
    "loading": {
      "mode": "materialized",
      "scoring_mode": "materialized"
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

## Run Training

```bash
uv run numereng run train --config configs/run.json
```

Optional overrides:

- `--output-dir <path>`
- `--profile <simple|purged_walk_forward|full_history_refit>`

## Outputs

Successful runs write artifacts under `.numereng/runs/<run_id>/`:

- `run.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json`
- `artifacts/predictions/*`

Run indexing is mandatory. If indexing fails, the command fails.

## High-Risk Gotchas

- Config path must be `.json`; other suffixes hard-fail.
- Unknown config keys hard-fail (`extra=forbid`).
- Training profiles are only `simple`, `purged_walk_forward`, `full_history_refit`.
- `purged_walk_forward` uses a fixed 156-era walk-forward window; embargo defaults to `8` (20D) or `16` (60D), with no legacy window/embargo overrides.
- `data.target_horizon` (`20d|60d`) is the preferred way to control purged walk-forward embargo defaults.
- If `target_horizon` is omitted and `target_col` name is ambiguous, training hard-fails.
- Training does not perform row-level subsampling; reduce size only via dataset-level downsampling.
- `simple` requires split sources (`train.parquet` + `validation.parquet`) and does not accept downsampled variants.
- If `full_data_path` is set with `simple`, sibling `train.parquet` and `validation.parquet` must exist beside that file.
- `full_history_refit` skips validation metrics and trains one model on full history.
- Canonical `model.x_groups` / `model.data_needed` are features-only by default; `era` and `id` are never auto-included and are not valid input groups.
- Post-run FNC always neutralizes to dataset feature set `fncv3_features`, then correlates against the scoring target being evaluated.
- Post-run scoring persists `score_provenance.json`.
- Numereng does not emit `payout_estimate_mean`.
- Benchmark and meta-model joins require strict era alignment; benchmark diagnostics are computed on overlapping rows only, and meta metrics are emitted on the available overlapping meta window whenever any overlap exists.
