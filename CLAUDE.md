# numereng

Read order:
1. `docs/llms.txt` (agent entrypoint)
2. `docs/ARCHITECTURE.md` (deep system map)

## Fast Commands
- `uv sync --extra dev`
- `make test`
- `make test-all`
- `uv build`

## Environment Rules
- Use only the project-managed env (`.venv`) via `uv sync --extra dev`.
- Prefer `uv run <tool>` for all commands (`pytest`, `ruff`, `mypy`, `numereng`, etc.).
- If direct Python is required, use `.venv/bin/python`.
- Do not create ad-hoc virtual envs.
- Numerai MCP auth is project-local via `.codex/config.toml` and expects `NUMERAI_MCP_AUTH` in the launching shell environment.
- If Numerai MCP tools are loaded but return missing-auth errors, run `eval "$(uv run python -m numereng.platform.export_numerai_mcp_auth)"` from the repo root, then relaunch Codex from that shell if MCP access is needed.

## Non-Negotiable Rules
- Preserve dependency direction: `config -> platform -> features -> api -> cli`.
- Keep CLI thin: parse/dispatch/output only. Business logic belongs in `features/*` behind `api/*`.
- `platform/*` must not import `features/*`.
- Public surfaces are `src/numereng/api/` and `src/numereng/cli/`.
- No source file in `src/numereng/api/` or `src/numereng/cli/` may exceed 500 lines.
- Update `docs/llms.txt` and `docs/ARCHITECTURE.md` when contracts/flows change.

## Boundary Contracts
- CLI entrypoint: `numereng = "numereng.cli:main"`.
- Python entrypoint: `import numereng.api as api_module`.
- Stable contracts live in `src/numereng/api/contracts.py`.
- Full local train/score pipeline entrypoint: `src/numereng/api/pipeline.py::run_training_pipeline(request)`.
  It runs `prepare_training_run -> load_training_data -> train_model -> score_predictions -> finalize_training_run`, then maps internal failures to `PackageError` and always performs cleanup.
- API boundary must translate internal errors to `PackageError`.
- CLI exit codes are fixed: `0` success/help, `1` runtime/boundary error, `2` parse/usage error.

## High-Risk Gotchas
- Submission source is XOR: exactly one of `run_id` or `predictions_path`.
- Neutralization source is XOR: exactly one of `run_id` or `predictions_path`.
- Training/HPO config files are JSON-only and reject unknown keys (`extra=forbid`).
- Training profile allowed only: `simple|purged_walk_forward|full_history_refit`; legacy `submission` profile references hard-fail with a rename error.
- Run IDs are deterministic hash-based IDs (12-char prefixes).
- Training requires successful pre-finalization `index_run`; if it fails, command fails.
- `experiment train` enforces `output_dir == store_root` (or omit output dir).
- `experiment pack` writes `.numereng/experiments/<id>/EXPERIMENT.pack.md` beside `EXPERIMENT.md` and overwrites it on each pack run.
- Telemetry is fail-open and opt-in via launch metadata binding.
- Launch metadata precedence: explicit outer binding first (CLI/API caller), API defaults only when unbound.
- `run train --experiment-id` explicitly scopes telemetry jobs to that experiment.
- If `experiment_id` is omitted, telemetry infers it when config path is under `.numereng/experiments/<id>/configs/*`.
- Dashboard is monitor-only: runs are launched via CLI/API, not frontend controls.
- Legacy runs may be backfilled with persisted per-era CORR artifacts via `numereng store materialize-viz-artifacts --kind per-era-corr ...`; viz otherwise uses a bounded write-through fallback on first miss.
- Canonical store roots: `runs`, `datasets`, `cloud`, `experiments`, `notes`.
- `cloud aws train submit` supports only `sagemaker|batch` and rejects `--spot` + `--on-demand` together.
- `cloud modal deploy` requires full ECR URI `<registry>/<repository>:<tag>`.
- `cloud modal data sync` requires config-required dataset files under local `.numereng/datasets`.

## Verification Anchors
- Fast gate: `make test`
- Full gate: `make test-all`
- Smoke surface contract: `tests/integration/test_smoke_structure.py`

## User Notes (Do Not Edit)
