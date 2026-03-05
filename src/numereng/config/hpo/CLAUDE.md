# hpo study config canonical contract

Purpose:
- `src/numereng/config/hpo` is the single source of truth for HPO study config structure.
- HPO study configs are JSON-only and must validate against `HpoStudyConfig`.

Read order:
1. `contracts.py` - canonical typed fields and required keys
2. `loader.py` - file/extension checks + runtime validation entrypoint
3. `schema/hpo_study_config.schema.json` - machine-readable schema export

Core rules:
- Accepted HPO study config format: `.json` only.
- Required top-level keys: `study_name`, `config_path`.
- `config_path` must point to one training config `.json` file.
- Unknown keys are forbidden at all levels (`extra="forbid"`).

Runtime flow:
1. CLI request validates `--study-config` as one `.json` path.
2. `load_hpo_study_config_json(...)` parses JSON and validates against `HpoStudyConfig`.
3. CLI merges optional flag overrides (if provided) and forwards resolved fields to HPO execution.

Schema maintenance:
- Canonical schema path: `src/numereng/config/hpo/schema/hpo_study_config.schema.json`.
- Regenerate from model:
  - `uv run python -c "from numereng.config.hpo import canonical_schema_path, export_hpo_study_config_schema; export_hpo_study_config_schema(canonical_schema_path())"`
- Keep schema file updated whenever `contracts.py` changes.
