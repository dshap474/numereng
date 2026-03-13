# Training Models

Use this workflow for a single local run that is not being launched through experiment metadata.

## Minimal Config

Training configs are strict JSON and validated against `src/numereng/config/training/contracts.py`.

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
- `--experiment-id <id>` if you want the run linked to an experiment while still using `run train`

## Re-Score A Saved Run

```bash
uv run numereng run score --run-id <run_id>
```

Use this when the predictions artifact already exists and you want to rebuild `results.json`, `metrics.json`, `score_provenance.json`, and the store index rows.

## Outputs

Successful scored runs write under `.numereng/runs/<run_id>/`:

- `run.json`
- `run.log`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json`
- `artifacts/predictions/*`

Run indexing is mandatory. If indexing fails, the command fails.

## High-Risk Gotchas

- config path must end in `.json`
- unknown config keys fail validation
- supported training profiles are only `simple`, `purged_walk_forward`, `full_history_refit`
- `purged_walk_forward` uses a fixed 156-era walk-forward window
- purged walk-forward embargo defaults are horizon-derived: `20d -> 8`, `60d -> 16`
- if `target_horizon` is omitted and `target_col` is ambiguous, training fails
- `simple` requires split train/validation sources and does not accept downsampled variants
- `full_history_refit` is final-fit only and emits no validation metrics
- post-run FNC always neutralizes to `fncv3_features`
- numereng does not emit `payout_estimate_mean`
