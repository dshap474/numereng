# Custom Models

Numereng supports custom model adapters through `MODEL_REGISTRY` modules loaded by the
training model factory.

## Golden Path

If you want to add a custom model that works with numereng, use this path first:

1. Check `src/numereng/features/models/custom_models/` to see whether the estimator already has a
   tracked wrapper.
2. Start from the closest wrapper:
   - use `template_model.py` for a novel estimator API
   - use the nearest sklearn-style wrapper when the estimator shape is similar
   - use `xgboost_model.py` or `catboost_model.py` when the backend is optional and should raise
     a backend-missing error
3. Rename the class and `MODEL_REGISTRY` key to the real model type you want to expose.
4. Replace the placeholder or copied `fit` and `predict` bodies with your estimator logic.
5. Add a config with:
   - `model.type`
   - `model.params`
   - optional `model.module_path`
6. Run a smoke test:
   - `uv run numereng run train --config <config.json>`
7. Run the focused custom-model tests:
   - `uv run pytest src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_factory.py`
   - `uv run pytest src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_wrappers.py`
   - `uv run pytest src/numereng/features/models/custom_models/tests/test_custom_model_wrappers.py`

Start with the nearest wrapper. `template_model.py` remains the canonical lowest-level fallback.

## Built-in model

- `LGBMRegressor` is the only built-in model type shipped in code
  (`src/numereng/features/models/lgbm.py`).

## Custom model directory

- Place custom files under `src/numereng/features/models/custom_models/`.
- The directory is ignored by git for new files by default so custom work stays modular.
- `template_model.py` is the tracked exception and is the canonical example to copy from.
- This repo also contains many tracked wrapper examples and tests under `custom_models/`.
- Discovery happens from:
  - explicit `model.module_path` in config, or
  - scanning `src/numereng/features/models/custom_models/**/*.py` when `module_path` is not set.

## Required module shape

Each custom module must expose `MODEL_REGISTRY`, for example:

```python
MODEL_REGISTRY = {"MyModelType": MyModelClass}
```

Each registered class must:

- accept `feature_cols=None` and `**params` in `__init__`
- implement `fit`
- implement `predict`
- filter `X` by `feature_cols` when it is provided

Numereng may also normalize backend-specific asset paths when that keeps runtime
behavior deterministic. `TabPFNRegressor`, for example, routes cache-managed
checkpoint names into `.numereng/cache/tabpfn` instead of letting the backend
write into the current working directory.

Use `src/numereng/features/models/custom_models/template_model.py` as the exact starter.

## Config example

Start from the template, then reference it from config:

```json
{
  "data": {
    "data_version": "v5.2",
    "dataset_variant": "non_downsampled",
    "feature_set": "small",
    "target_col": "target"
  },
  "model": {
    "type": "YourModelType",
    "module_path": "src/numereng/features/models/custom_models/your_model.py",
    "params": {}
  },
  "training": {
    "engine": {
      "profile": "purged_walk_forward"
    }
  }
}
```

If your model file lives under `custom_models/` and the type is unique, `model.module_path` can be
omitted and numereng will discover it automatically.

## Tracked vs local files

- `template_model.py` is tracked on purpose so the repo has one stable reference example.
- User-specific new model files in `custom_models/` stay local by default because the directory is
  gitignored for new files.
- Many wrappers already committed in this repo are tracked examples and valid starting points.
- Commit a custom model file only when it is intended to become shared repo behavior.

## Tracked example wrappers

These files show real wrapper patterns already used in this repo:

- `linear_regression_model.py`
- `random_forest_model.py`
- `mlp_regressor_model.py`
- `xgboost_model.py`
- `catboost_model.py`

They are useful examples when the estimator shape is similar. Use `template_model.py` when the API
does not match an existing wrapper well.

## Focused tests

When you add or change a tracked wrapper, update the matching tests here:

- `src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_factory.py`
- `src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_wrappers.py`
- `src/numereng/features/models/custom_models/tests/test_custom_model_wrappers.py`

## Error behavior

- `training_model_custom_module_not_found:<path>`:
  `model.module_path` did not resolve to a file.
- `training_model_registry_missing_or_invalid:<path>`:
  the module did not expose a valid `MODEL_REGISTRY`.
- `training_model_type_not_supported:<type>`:
  numereng did not find the requested model type in built-ins or discovered plugins.
- `training_model_invalid_model_class:<type>`:
  the registry entry is missing `fit` or `predict`.
- `xgboost` missing in model environment raises:
  - `training_model_backend_missing_xgboost`
- `catboost` missing in model environment raises:
  - `training_model_backend_missing_catboost`

Backend-missing errors are raised when the selected model class initializes.
