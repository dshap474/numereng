---
name: numerai-model-implementation
description: Add a new numereng training model type end-to-end through custom model wrappers or built-in registry wiring, then validate with run/train and unit tests.
user-invocable: true
---

# Numerai Model Implementation

Implement a new training model type for the current numereng architecture.

Run from:
- `<repo>`

## Contract Guardrails

- Keep dependency direction: `config -> platform -> features -> api -> cli`.
- Preferred extension path is plugin modules under:
  - `src/numereng/features/models/custom_models/`
- Training config input is JSON-only (`.json`), unknown keys are forbidden.
- Canonical model inputs are features-only; model wrappers must not depend on `era` or `id` being present in `X`.
- Keep CLI/API surfaces unchanged unless explicitly requested.
- Use project-managed env and commands:
  - `uv sync --extra dev`
  - `uv run ...`

## Canonical Extension Points

- Model wrappers:
  - built-in: `src/numereng/features/models/lgbm.py`
  - custom: `src/numereng/features/models/custom_models/*.py`
- Model dispatch:
  - `src/numereng/features/training/model_factory.py`
- Training model contract:
  - `src/numereng/config/training/contracts.py` (`model.type`, `model.params`, optional `model.module_path`)
- Existing tests:
  - `tests/unit/numereng/features/training/test_model_factory.py`

## Implementation Workflow

1. Choose model integration mode.
- Default: custom plugin module in `custom_models/` with `MODEL_REGISTRY`.
- Built-in: only when this model should ship as a first-class package default.

2. Add/implement model wrapper.
- Create `src/numereng/features/models/custom_models/<model_name>.py`.
- Wrapper class requirements:
  - `__init__(self, feature_cols: list[str] | None = None, **params)`
  - `fit(self, X, y, **kwargs)`
  - `predict(self, X)`
- Export `MODEL_REGISTRY = {"YourModelType": YourModelClass}`.
- Raise `TrainingModelError("training_model_backend_missing_<backend>")` when backend dependency import fails.

3. Wire model resolution (only if needed).
- Plugin path:
  - no model factory code change needed if `MODEL_REGISTRY` is present and config uses `module_path` or autodiscovery.
- Built-in path:
  - add wrapper file under `src/numereng/features/models/`
  - register model in `_BUILTIN_MODELS` in `src/numereng/features/training/model_factory.py`
  - add focused tests in `test_model_factory.py`

4. Add JSON config using the new model type.
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
    "type": "XGBoostRegressor",
    "module_path": "src/numereng/features/models/custom_models/xgboost_model.py",
    "params": {
      "n_estimators": 500,
      "learning_rate": 0.05
    }
  },
  "training": {
    "engine": {
      "profile": "purged_walk_forward"
    }
  }
}
```

5. Validate end-to-end behavior.
- Fast unit target:
  - `uv run pytest tests/unit/numereng/features/training/test_model_factory.py`
- Train smoke:
  - `uv run numereng run train --config <config.json>`
- Full fast gate:
  - `make test`

6. Verify run artifacts and model metadata.
- Confirm run indexed and artifacts written under:
  - `.numereng/runs/<run_id>/`
- Confirm `run.json`/results payload records expected:
  - `model.type`
  - `model.params`
  - implemented model extras copied from config (for example `target_transform`, when configured)

## Error Reference

- `training_model_custom_module_not_found:<path>`
- `training_model_registry_missing_or_invalid:<path>`
- `training_model_type_not_supported:<type>`
- `training_model_invalid_model_class:<type>`
- `training_model_backend_missing_lightgbm`
- `training_model_backend_missing_xgboost`
- `training_model_backend_missing_catboost`

## Done Criteria

- New model type can be selected with `model.type` in JSON config.
- Training succeeds via `uv run numereng run train --config ...`.
- `test_model_factory.py` covers the new path and error behavior.
- No API/CLI contract regressions.
