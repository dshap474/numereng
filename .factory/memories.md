# numereng project memory

## Stable public surfaces

- CLI entrypoint: `numereng = "numereng.cli:main"`
- Python facade: `import numereng.api`
- Public workflow entrypoint: `numereng.api.pipeline.run_training_pipeline(...)`
- Stable typed contracts live in `src/numereng/api/contracts.py`

## Core architectural invariants

- Dependency direction is strict: `config -> platform -> features -> api -> cli`
- `platform/*` must not import `features/*`
- CLI remains parse/dispatch/output only
- Business logic stays in `features/*` and is exposed through `api/*`
- `src/numereng/api/` and `src/numereng/cli/` are public surfaces and must stay small

## Runtime and store invariants

- Default local state root is `.numereng/`
- Canonical top-level store roots: `runs`, `datasets`, `cloud`, `experiments`, `notes`, `cache`
- Run IDs are deterministic hash-derived 12-char prefixes
- Training requires a successful pre-finalization `index_run`
- `run_lifecycles` and `runs/<run_id>/runtime.json` are the current-truth lifecycle surfaces

## Training and experiment invariants

- Training and HPO configs are JSON-only and reject unknown keys
- Training profile allowed only: `simple|purged_walk_forward|full_history_refit`
- Submission source is XOR: exactly one of `run_id` or `predictions_path`
- Neutralization source is XOR: exactly one of `run_id` or `predictions_path`
- `experiment train` enforces `output_dir == store_root` (or omitted)
- Archived experiments are read-only until unarchived

## Agent workflows

- Existing repo skills under `.agents/skills/*` are still canonical for long-form workflow knowledge
- `.factory/skills/*` mirror the highest-value workflows for Factory-native routing
- `docs/llms.txt` and `docs/ARCHITECTURE.md` remain the deep system maps

## Telemetry context memory

- Launch metadata is the current context contract for public training entrypoints
- Required fields today: `source`, `operation_type`, `job_type`
- Lower-level training code expects launch metadata to be bound before lifecycle bootstrap

## Known readiness focus areas

- Keep Factory-native repo memory (`AGENTS.md`, `.factory/*`) aligned with the real contracts
- Prefer machine-enforced checks over documented-only rules
- Preserve concise bootstrap and validation commands so agents can prove changes quickly
