# Custom Models

Numereng supports custom model adapters through `MODEL_REGISTRY` modules found by
the training model factory.

## Built-in model

- `LGBMRegressor` is the only built-in model type shipped in code
  (`src/numereng/features/models/lgbm.py`).

## Custom model directory

- Place custom files under `numereng/src/numereng/features/models/custom_models`.
- The directory is ignored by git by default so custom work stays modular.
- Discovery happens from:
  - explicit `model.module_path` in config, or
  - scanning `src/numereng/features/models/custom_models/**/*.py` when `module_path` is not set.

Each custom module must expose `MODEL_REGISTRY`, e.g.:

```python
MODEL_REGISTRY = {"MyModel": MyModel}
```

`MyModel` must accept `(feature_cols=None, **params)` in `__init__`
and provide `fit` and `predict` methods.

## Example config

```yaml
model:
  type: XGBoostRegressor
  module_path: src/numereng/features/models/custom_models/xgboost_model.py
  params:
    n_estimators: 500
    learning_rate: 0.05
```

```yaml
model:
  type: CatBoostRegressor
  module_path: src/numereng/features/models/custom_models/catboost_model.py
  params:
    depth: 8
    learning_rate: 0.07
    iterations: 1200
```

## Error behavior

- `xgboost` missing in model environment raises:
  - `training_model_backend_missing_xgboost`
- `catboost` missing in model environment raises:
  - `training_model_backend_missing_catboost`

The error is raised when the selected model class is initialized.
