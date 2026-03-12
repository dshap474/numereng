---
name: numerai-model-implementation
description: "Add a new numereng training model type by copying the tracked template, wiring config correctly, and validating it with the current training pipeline."
user-invocable: true
---

# Numerai Model Implementation

Use this skill to add a model that works with the current numereng training pipeline.

Run from:
- `<repo>`

## Use when
- the user wants to add a new model type to numereng
- the user needs to know how custom models under `custom_models/` should be written
- the user needs the correct config and validation path for a new model adapter

## Do not use when
- the user is choosing experiment rounds or model strategy; use `experiment-design`
- the user needs experiment layout, config templates, or runtime contracts; use
  `numereng-experiment-ops`
- the user is debugging store drift, resets, or cleanup; use `store-ops`
- the user is preparing submissions or pickle deployment; use `numerai-model-upload`

## Happy Path

Default path for a new custom model:

1. Copy `src/numereng/features/models/custom_models/template_model.py`.
2. Rename the class and `MODEL_REGISTRY` key to the real model type.
3. Replace the placeholder `fit` and `predict` logic with the real estimator.
4. Add a config with:
   - `model.type`
   - `model.params`
   - optional `model.module_path`
5. Run:
   - `uv run numereng run train --config <config.json>`
6. Run:
   - `uv run pytest tests/unit/numereng/features/training/test_model_factory.py`

Use the tracked template first. Reach for built-in wiring only when the model should become a
first-class shared repo default.

## Canonical Files

- Tracked starter template:
  - `src/numereng/features/models/custom_models/template_model.py`
- Local optional examples:
  - `src/numereng/features/models/custom_models/xgboost_model.py`
  - `src/numereng/features/models/custom_models/catboost_model.py`
- Plugin discovery and built-in registry:
  - `src/numereng/features/training/model_factory.py`
- Training config contract:
  - `src/numereng/config/training/contracts.py`
- Custom-model doc:
  - `src/numereng/features/models/CUSTOM_MODELS.md`
- Focused test:
  - `tests/unit/numereng/features/training/test_model_factory.py`

## Implementation Workflow

1. Choose model integration mode.
- Default: custom plugin in `src/numereng/features/models/custom_models/`
- Built-in: only when the model should ship as shared repo behavior via `_BUILTIN_MODELS`

2. Start from the tracked template.
- Copy `src/numereng/features/models/custom_models/template_model.py` to a new module file.
- Wrapper class requirements:
  - `__init__(self, feature_cols: list[str] | None = None, **params)`
  - `fit(self, X, y, **kwargs)`
  - `predict(self, X)`
- Filter `X` by `feature_cols` when it is provided.
- Export `MODEL_REGISTRY = {"YourModelType": YourModelClass}`.
- If an optional backend import is required, raise:
  - `TrainingModelError("training_model_backend_missing_<backend>")`

3. Decide whether explicit `module_path` is needed.
- Plugin path:
  - no `model_factory.py` change needed if `MODEL_REGISTRY` is present
  - use `model.module_path` when you want explicit resolution
  - omit `model.module_path` when discovery from `custom_models/` is sufficient
- Built-in path:
  - add the wrapper under `src/numereng/features/models/`
  - register model in `_BUILTIN_MODELS` in `src/numereng/features/training/model_factory.py`
  - add focused coverage in `tests/unit/numereng/features/training/test_model_factory.py`

4. Add a training config.
- Required keys:
  - `model.type`
  - `model.params`
- Optional for plugins:
  - `model.module_path`

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

5. Validate end-to-end behavior.
- Train smoke:
  - `uv run numereng run train --config <config.json>`
- Focused unit test:
  - `uv run pytest tests/unit/numereng/features/training/test_model_factory.py`
- Optional broader gate:
  - `make test`

6. Verify numereng output.
- Confirm the run writes under `.numereng/runs/<run_id>/`.
- Confirm `run.json` records:
  - `model.type`
  - `model.params`
  - any model-specific extras that should survive config resolution

## Tracked vs local model files

- `template_model.py` is tracked on purpose so the repo always has one stable reference example.
- Most files in `custom_models/` remain local by default because the directory is gitignored.
- Commit a custom model file only when it is intended to become shared repo behavior.

## Troubleshooting

- `training_model_custom_module_not_found:<path>`
- `training_model_registry_missing_or_invalid:<path>`
- `training_model_type_not_supported:<type>`
- `training_model_invalid_model_class:<type>`
- `training_model_backend_missing_lightgbm`
- `training_model_backend_missing_xgboost`
- `training_model_backend_missing_catboost`

Use these errors to decide whether the problem is:
- path resolution
- missing `MODEL_REGISTRY`
- wrong `model.type`
- invalid wrapper shape
- missing optional backend dependency

## Output Contract

Return:
- the chosen integration mode: plugin or built-in
- the exact file to start from
- the required config fields
- the exact smoke-test and unit-test commands
- the expected numereng artifacts to verify

## Done Criteria

- New model type can be selected with `model.type` in JSON config.
- Training succeeds via `uv run numereng run train --config ...`.
- `tests/unit/numereng/features/training/test_model_factory.py` covers the new path and error behavior.
- No API/CLI contract regressions.
