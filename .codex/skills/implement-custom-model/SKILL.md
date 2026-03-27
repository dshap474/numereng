---
name: implement-custom-model
description: "Implement a numereng custom model plugin under features/models/custom_models using the existing wrapper, factory, and test patterns."
user-invocable: true
---

# Implement Custom Model

Use this skill to add or update a custom model that works with the current numereng training
pipeline. Default assumption: you are implementing a plugin under
`src/numereng/features/models/custom_models/` and wiring it through the existing model factory,
not changing the core training architecture.

Run from:
- `<repo>`

## Use when
- the user wants to add a new custom model type to numereng
- the user needs to know how wrappers under `custom_models/` should be structured
- the user needs to decide whether an existing tracked wrapper already covers the requested estimator
- the user needs the correct config and validation path for a new custom model adapter

## Do not use when
- the user is choosing experiment rounds or model strategy; use `experiment-design`
- the user needs experiment layout, config templates, or runtime contracts; use
  `numereng-experiment-ops`
- the user is debugging store drift, resets, or cleanup; use `store-ops`
- the user is preparing a winning experiment config for final refit or predictions submission; use `numereng-experiment-ops`

## Happy Path

Default path for a new custom model plugin:

1. Check `src/numereng/features/models/custom_models/` to see whether the estimator already exists.
2. Choose the closest starting point:
   - use `template_model.py` for a novel or unusual estimator API
   - use the closest tracked sklearn-style wrapper when the estimator is similar to an existing wrapper
   - use `xgboost_model.py` or `catboost_model.py` when the backend is optional and should raise a backend-missing error
3. Rename the class and `MODEL_REGISTRY` key to the real model type.
4. Keep the wrapper contract intact:
   - accept `feature_cols` and `**params` in `__init__`
   - filter `X` through `_filter_features(...)`
   - implement `fit(...)`
   - implement `predict(...)`
5. Add a config with:
   - `model.type`
   - `model.params`
   - optional `model.module_path`
6. Run:
   - `uv run numereng run train --config <config.json>`
7. Run the focused custom-model tests:
   - `uv run pytest src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_factory.py`
   - `uv run pytest src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_wrappers.py`
   - `uv run pytest src/numereng/features/models/custom_models/tests/test_custom_model_wrappers.py`

Use the nearest tracked wrapper first. Reach for built-in wiring only when the model should become a
first-class shared repo default rather than a custom-model plugin.

## Canonical Files

- Tracked starter template:
  - `src/numereng/features/models/custom_models/template_model.py`
- Tracked wrapper examples:
  - `src/numereng/features/models/custom_models/linear_regression_model.py`
  - `src/numereng/features/models/custom_models/random_forest_model.py`
  - `src/numereng/features/models/custom_models/xgboost_model.py`
  - `src/numereng/features/models/custom_models/catboost_model.py`
- Custom-model docs:
  - `src/numereng/features/models/CUSTOM_MODELS.md`
- Plugin discovery and built-in registry:
  - `src/numereng/features/training/model_factory.py`
- Training config contract:
  - `src/numereng/config/training/contracts.py`
- Focused tests:
  - `src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_factory.py`
  - `src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_wrappers.py`
  - `src/numereng/features/models/custom_models/tests/test_custom_model_wrappers.py`

## Implementation Workflow

1. Decide whether a new wrapper is actually needed.
- First inspect `src/numereng/features/models/custom_models/` for an existing tracked wrapper.
- If the requested estimator already exists, prefer reusing the existing `model.type` and params.
- Add a new wrapper only when the estimator or adapter behavior is genuinely new.

2. Choose model integration mode.
- Default: custom plugin in `src/numereng/features/models/custom_models/`
- Built-in: only when the model should ship as shared repo behavior via `_BUILTIN_MODELS`
- Prefer the plugin path unless the user explicitly wants a tracked built-in model.

3. Start from the closest tracked implementation.
- Novel API or unusual backend:
  - copy `src/numereng/features/models/custom_models/template_model.py`
- Sklearn-like estimator:
  - copy the closest wrapper already in `custom_models/`
- Optional third-party backend:
  - copy `xgboost_model.py` or `catboost_model.py` to preserve backend-missing error behavior
- Wrapper class requirements:
  - `__init__(self, feature_cols: list[str] | None = None, **params)`
  - `fit(self, X, y, **kwargs)`
  - `predict(self, X)`
- Filter `X` by `feature_cols` when it is provided.
- Export `MODEL_REGISTRY = {"YourModelType": YourModelClass}`.
- If an optional backend import is required, raise:
  - `TrainingModelError("training_model_backend_missing_<backend>")`

4. Decide whether explicit `module_path` is needed.
- Plugin path:
  - no `model_factory.py` change needed if `MODEL_REGISTRY` is present
  - use `model.module_path` when you want explicit resolution
  - omit `model.module_path` when discovery from `custom_models/` is sufficient
  - relative `model.module_path` values are resolved against:
    - `src/numereng/features/models/custom_models/`
    - current working directory
  - absolute paths also work
  - if the path omits `.py`, numereng also tries the `.py` suffix
- Built-in path:
  - add the wrapper under `src/numereng/features/models/`
  - register model in `_BUILTIN_MODELS` in `src/numereng/features/training/model_factory.py`
  - export the built-in from `src/numereng/features/models/__init__.py`
  - update built-in-facing docs if the shared built-in set changes
  - add focused coverage in the relevant training/model tests

5. Add a training config.
- Required keys:
  - `model.type`
  - `model.params`
- Optional for plugins:
  - `model.module_path`
- Important device rule:
  - `model.device` is currently valid only for `LGBMRegressor`
  - non-LGBM custom models must not rely on `model.device`
  - backend-specific runtime parameters for custom models must stay inside `model.params`

Example:

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

6. Validate end-to-end behavior.
- Train smoke:
  - `uv run numereng run train --config <config.json>`
- Focused unit test:
  - if you are adding or changing a tracked wrapper, update or extend:
    - `src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_factory.py`
    - `src/numereng/features/models/custom_models/tests/test_sklearn_custom_model_wrappers.py`
    - `src/numereng/features/models/custom_models/tests/test_custom_model_wrappers.py`
- If you added a tracked shared built-in:
  - add or update wrapper-specific unit tests beside the model implementation
- Optional broader gate:
  - `make test`

7. Verify numereng output.
- Confirm the run writes under `.numereng/runs/<run_id>/`.
- Confirm `run.json` records:
  - `model.type`
  - `model.params`
  - any model-specific extras that should survive config resolution

## Tracked vs local model files

- `template_model.py` is tracked on purpose so the repo always has one stable reference example.
- The `custom_models/` directory is gitignored for new files by default, but many wrappers in this
  repo are already tracked examples.
- New local wrappers still work at runtime even when untracked, as long as they live under
  `custom_models/` or are referenced by `model.module_path`.
- Commit a custom model file only when it is intended to become shared repo behavior. New tracked
  files under `custom_models/` may require a forced add because of the repo `.gitignore`.

## Troubleshooting

- `training_model_custom_module_not_found:<path>`
- `training_model_invalid_module_path`
- `training_model_custom_module_invalid:<path>`
- `training_model_custom_module_load_failed:<path>`
- `training_model_registry_missing_or_invalid:<path>`
- `training_model_invalid_registry_entry:<type>`
- `training_model_type_not_supported:<type>`
- `training_model_invalid_model_class:<type>`
- `training_model_backend_missing_lightgbm`
- `training_model_backend_missing_xgboost`
- `training_model_backend_missing_catboost`

Use these errors to decide whether the problem is:
- path resolution
- invalid `model.module_path` type or malformed module file
- import-time failure while loading the module
- missing `MODEL_REGISTRY`
- wrong `model.type`
- invalid wrapper shape
- missing optional backend dependency

## Output Contract

Return:
- whether a new wrapper is needed or an existing wrapper should be reused
- the chosen integration mode: plugin or built-in
- the exact file to start from
- the required config fields
- the exact smoke-test and unit-test commands
- the expected numereng artifacts to verify

## Done Criteria

- New model type can be selected with `model.type` in JSON config.
- For the default plugin path: dropping a valid module into `custom_models/` is enough for
  numereng to resolve it, with or without explicit `model.module_path`.
- Training succeeds via `uv run numereng run train --config ...`.
- Focused custom-model tests cover the new path and any wrapper-specific error behavior.
- No API/CLI contract regressions.
