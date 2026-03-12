# Models Canonical Standards

This folder owns model implementations used by the training pipeline.
`features/training` orchestrates data/CV/artifacts, while `features/models` owns model classes.

## Source-of-Truth Modules

- `lgbm.py`: built-in LightGBM wrapper exposed as model type `LGBMRegressor`
- `custom_models/*.py`: user plugin modules loaded through `MODEL_REGISTRY`
- `__init__.py`: public exports for this feature slice
- `features/training/model_factory.py`: model resolution + plugin discovery entrypoint

## Built-in Model Contract

- Built-in type string is exactly: `LGBMRegressor`
- Resolved class location: `src/numereng/features/models/lgbm.py`
- The wrapper exposes `fit`/`predict` and supports optional `feature_cols` filtering.
- If LightGBM is not installed, training raises:
  - `training_model_backend_missing_lightgbm`

## Custom Model Location (Canonical)

User custom model modules must live under:

- `src/numereng/features/models/custom_models/`

The previous path `src/numereng/features/custom_models/` is no longer used.

Golden path:

1. Copy `src/numereng/features/models/custom_models/template_model.py`
2. Rename the class and `MODEL_REGISTRY` key
3. Replace the placeholder `fit` / `predict`
4. Add a config with `model.type` and optional `model.module_path`
5. Smoke test with `uv run numereng run train --config <config.json>`

Treat `template_model.py` as the tracked reference example. Treat other files in
`custom_models/` as optional local examples.

## How Custom Model Discovery Works

Model resolution happens in `features/training/model_factory.py`:

1. If `model.type` is built-in (`LGBMRegressor`), built-in class is used.
2. Otherwise plugin discovery runs:
  - If `model.module_path` is set:
    - absolute path: used directly
    - relative path: attempted relative to:
      - custom model root (`src/numereng/features/models/custom_models`)
      - current working directory
    - if no `.py` suffix is present, `.py` is also attempted
  - If `model.module_path` is not set:
    - scans `src/numereng/features/models/custom_models/**/*.py` (excluding `__init__.py`)

## Required Custom Module Shape

Each plugin file must define a `MODEL_REGISTRY` dict:

```python
MODEL_REGISTRY = {"MyModelType": MyModelClass}
```

Each registered model class must:

- accept `feature_cols=None` and `**params` in `__init__`
- implement `fit(...)`
- implement `predict(...)`
- filter `X` by `feature_cols` when it is provided

Minimal template:

```python
class TemplateModel:
    def __init__(self, feature_cols=None, **params):
        self.feature_cols = feature_cols
        self.params = params

    def fit(self, X, y, **kwargs):
        filtered_X = X if not self.feature_cols else X[self.feature_cols]
        return self

    def predict(self, X):
        filtered_X = X if not self.feature_cols else X[self.feature_cols]
        return [0.0] * len(filtered_X)


MODEL_REGISTRY = {"TemplateModel": TemplateModel}
```

## Config Example (JSON)

```json
{
  "model": {
    "type": "YourModelType",
    "module_path": "src/numereng/features/models/custom_models/your_model.py",
    "params": {
      "learning_rate": 0.05
    }
  }
}
```

## Common Errors

- `training_model_custom_module_not_found:<path>`:
  module path is wrong or file does not exist.
- `training_model_registry_missing_or_invalid:<path>`:
  module did not expose a dict named `MODEL_REGISTRY`.
- `training_model_type_not_supported:<type>`:
  requested `model.type` not present in built-ins or discovered plugin registries.
- `training_model_invalid_model_class:<type>`:
  resolved class is missing `fit` or `predict`.

## Git Behavior

By default, `.gitignore` ignores this directory:

- `src/numereng/features/models/custom_models/`

`template_model.py` is the tracked exception and should remain the canonical reference example.
User-specific model files stay local by default unless intentionally committed as shared repo code.
