# Features

This page covers dataset feature sets, model feature plumbing, custom model expectations, and prediction-stage neutralization.

## Dataset Feature Sets

Numerai feature groups are selected with `data.feature_set`.

| Set | Approx. scale | Typical use |
| --- | --- | --- |
| `small` | smallest runtime | fast baselines and iteration |
| `medium` | moderate runtime | general development |
| `all` | largest runtime | full-search experiments |

The feature set controls which Numerai feature columns are loaded, but it does not change the scoring target contract.

## Model Feature Plumbing

Model inputs are controlled through `model` fields:

- `x_groups`
- `data_needed`
- `baseline`
- `benchmark`

Current rules:

- default model inputs are feature columns only
- `era` and `id` are never valid training feature groups
- `era` and `id` are never auto-included as model inputs
- benchmark aliases such as `benchmark`, `benchmarks`, and `benchmark_models` are not valid `x_groups`
- if `x_groups` includes `baseline`, `data.id_col` must be available

## Custom Models

Numereng supports plugin models loaded from `custom_models/`.

The canonical onboarding path is:

1. copy `custom_models/template_model.py`
2. rename the class and `MODEL_REGISTRY` key
3. implement `fit` and `predict`
4. reference the model with `model.type` and, when needed, `model.module_path`

See [Custom Models](../reference/custom-models.md) for the full contract.

## Neutralization Is A Separate Stage

Neutralization is not a training-config block. It is a prediction-stage workflow that can run:

- directly through `numereng neutralize apply`
- inline during submission
- inline during HPO objective scoring
- inline during ensemble construction

Example:

```bash
uv run numereng neutralize apply \
  --run-id <run_id> \
  --neutralizer-path data/neutralizer.parquet
```

Important constraints:

- `--neutralizer-cols` must resolve to numeric columns
- empty neutralizer column selections fail
- output ranking is enabled by default and can be disabled with `--no-neutralization-rank`

## Dataset Tools

Stored derived downsampled datasets are built with:

```bash
uv run numereng dataset-tools build-downsampled-full --data-version v5.2
```

That workflow materializes:

- `downsampled_full.parquet`
- `downsampled_full_benchmark_models.parquet`

under `.numereng/datasets/<data_version>/`.

Non-downsampled training data remains canonical as split sources:

- `train.parquet`
- `validation.parquet`
