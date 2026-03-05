# training config canonical contract

Purpose:
- `src/numereng/config/training` is the single source of truth for training config structure.
- Training configs are JSON-only and must validate against `TrainingConfig`.

Read order:
1. `contracts.py` - canonical typed fields and required keys
2. `loader.py` - file/extension checks + runtime validation entrypoint
3. `schema/training_config.schema.json` - machine-readable schema export

Core rules:
- Accepted training config format: `.json` only.
- Required top-level keys: `data`, `model`, `training`.
- Required nested keys: `model.type`, `model.params`.
- Unknown keys are forbidden at all levels (`extra="forbid"`).

Runtime flow:
1. CLI/API/cloud request contracts validate config path/URI shape (`.json`).
2. `features.training.repo.load_config` delegates to `load_training_config_json(...)`.
3. The loader parses JSON, validates against `TrainingConfig`, and returns normalized dict payload.
4. Training service consumes that validated payload.

Schema maintenance:
- Canonical schema path: `src/numereng/config/training/schema/training_config.schema.json`.
- Regenerate from model:
  - `uv run python -c "from numereng.config.training import canonical_schema_path, export_training_config_schema; export_training_config_schema(canonical_schema_path())"`
- Keep schema file updated whenever `contracts.py` changes.
