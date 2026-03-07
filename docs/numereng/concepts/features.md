# Features

Feature-set selection, model feature plumbing, and prediction-stage neutralization.

## Feature Sets

Numerai feature groups are configured by `data.feature_set`.

| Set | Approx. Features | Runtime | Typical Use |
|-----|-------------------|---------|-------------|
| `small` | ~300 | Fast | Baselines and iteration |
| `medium` | ~700 | Moderate | General development |
| `all` | ~2000+ | Slow | Full-search experiments |

## Model Feature Plumbing

Feature inputs are controlled through `model` fields:

- `x_groups`
- `data_needed`
- `baseline`
- `benchmark`

If `x_groups` includes `baseline`, ensure `data.id_col` is present.
If `model.x_groups` or `model.data_needed` is omitted, training uses feature columns only. `era` and `id` are never auto-included and are not valid training input groups.
`benchmark_models` aliases (`benchmark`, `benchmarks`, `benchmark_models`) are invalid in
`x_groups` and fail config validation.

## Neutralization Is Prediction-Stage

Neutralization is a separate workflow, not a training config block:

```bash
uv run numereng neutralize apply \
  --run-id <run_id> \
  --neutralizer-path data/neutralizer.parquet
```

Also available inline during submit/HPO/ensemble flows via neutralization flags.

## `colsample_bytree=0.1`

For tree models on correlated Numerai features, low column sampling is a strong default:

```json
{
  "model": {
    "type": "LGBMRegressor",
    "params": {
      "colsample_bytree": 0.1
    }
  }
}
```

## High-Risk Gotchas

- There is no `features.neutralization` runtime block in the current schema.
- `--neutralizer-cols` must resolve to numeric columns; empty selections fail.
