# Custom Models

Numereng supports plugin model adapters through `MODEL_REGISTRY` modules loaded by the training model factory.

## Golden Path

Start with the tracked template:

1. copy `src/numereng/features/models/custom_models/template_model.py`
2. rename the class and `MODEL_REGISTRY` key
3. implement `fit` and `predict`
4. reference the model from config with `model.type`
5. add `model.module_path` only when you need explicit resolution
6. smoke test with `uv run numereng run train --config <config.json>`

`template_model.py` is the canonical onboarding example.

## Plugin Location

Custom model modules live under:

- `src/numereng/features/models/custom_models/`

Discovery works in two modes:

- explicit `model.module_path`
- automatic scan of `custom_models/**/*.py` when `module_path` is omitted

## Required Module Shape

Every plugin module must expose `MODEL_REGISTRY`:

```python
MODEL_REGISTRY = {"MyModelType": MyModelClass}
```

Each registered class must:

- accept `feature_cols=None` and `**params` in `__init__`
- implement `fit`
- implement `predict`
- filter `X` by `feature_cols` when it is provided

Backend-specific path handling may also live in numereng. For example,
`TabPFNRegressor` uses the project-local cache root
`.numereng/cache/tabpfn`. A bare checkpoint filename in
`model.params.model_path` is treated as a cache-managed model name there
instead of a path relative to the current working directory.

## Config Example

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

If the file is discoverable under `custom_models/` and the type is unique, `module_path` can be omitted.

## Tracked Vs Local Files

- `template_model.py` is intentionally tracked
- user-specific custom model files remain local by default because the directory is gitignored
- commit a custom model file only when it is meant to become shared repo behavior

## Built-In Model

The built-in shipped model type is:

- `LGBMRegressor`

Custom models are the extension path for everything else.

## Validation Path

Recommended checks:

```bash
uv run numereng run train --config <config.json>
uv run pytest tests/unit/numereng/features/training/test_model_factory.py
```

## Common Errors

- `training_model_custom_module_not_found:<path>`
- `training_model_registry_missing_or_invalid:<path>`
- `training_model_type_not_supported:<type>`
- `training_model_invalid_model_class:<type>`
- `training_model_backend_missing_xgboost`
- `training_model_backend_missing_catboost`
