# Custom Models

Numereng supports plugin model adapters through `MODEL_REGISTRY` modules loaded by the training model factory.

## Golden Path

Start from the workspace template created by `numereng init`:

1. copy `custom_models/template_model.py`
2. rename the class and `MODEL_REGISTRY` key
3. implement `fit` and `predict`
4. reference the model from config with `model.type`
5. add `model.module_path` only when you need explicit resolution
6. smoke test with `numereng run train --config <config.json>`

## Plugin Location

Custom model modules live under:

- `custom_models/`

Discovery works in two modes:

- explicit `model.module_path`
- automatic scan of `custom_models/**/*.py` when `module_path` is omitted

Workspace-local models are the canonical runtime discovery source. Numereng does not auto-discover packaged repo custom-model wrappers.

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
    "module_path": "custom_models/your_model.py",
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

## Built-In Behavior

- `LGBMRegressor` is the built-in shipped model type.
- `TabPFNRegressor` resolves bare checkpoint names under `.numereng/cache/tabpfn` unless `TABPFN_MODEL_CACHE_DIR` is already set.

## Validation Path

Recommended checks:

```bash
numereng run train --config <config.json>
```

## Common Errors

- `training_model_custom_module_not_found:<path>`
- `training_model_registry_missing_or_invalid:<path>`
- `training_model_type_not_supported:<type>`
- `training_model_invalid_model_class:<type>`
- `training_model_backend_missing_xgboost`
- `training_model_backend_missing_catboost`
