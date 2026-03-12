# Custom Models

Numereng supports custom model adapters through `MODEL_REGISTRY` modules loaded by the
training model factory.

## Golden Path

If you want to add a custom model that works with numereng, use this path first:

1. Copy `src/numereng/features/models/custom_models/template_model.py` to a new file in the same
   directory.
2. Rename the class and `MODEL_REGISTRY` key to the real model type you want to expose.
3. Replace the placeholder `fit` and `predict` bodies with your estimator logic.
4. Add a config with:
   - `model.type`
   - `model.params`
   - optional `model.module_path`
5. Run a smoke test:
   - `uv run numereng run train --config <config.json>`

Start with `template_model.py`. Treat the other files in this directory as optional local examples,
not the main onboarding path.

## Built-in model

- `LGBMRegressor` is the only built-in model type shipped in code
  (`src/numereng/features/models/lgbm.py`).

## Custom model directory

- Place custom files under `src/numereng/features/models/custom_models/`.
- The directory is ignored by git by default so custom work stays modular.
- `template_model.py` is the tracked exception and is the canonical example to copy from.
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
- User-specific model files in `custom_models/` stay local by default because the directory is
  gitignored.
- Commit a custom model file only when it is intended to become shared repo behavior.

## Optional local examples

If present in your local checkout, these files show backend-specific wrappers:

- `xgboost_model.py`
- `catboost_model.py`

They are useful examples, but `template_model.py` is the canonical onboarding file.

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
