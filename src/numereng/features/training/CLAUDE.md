# Training Canonical Standards

This folder defines the canonical training contract for Numereng.
Any downstream consumer (viz, reports, automation) must treat training artifacts
written by this pipeline as source of truth.

## Source-of-Truth Modules

- `service.py`: end-to-end pipeline orchestration and artifact writes
- `repo.py`: canonical file paths and persistence helpers
- `run_store.py`: run identity + manifest schema
- `metrics.py`: scoring computation, per-era math, provenance shape

## Canonical Run Artifact Contract

Each training run writes to `.numereng/runs/<run_id>/`.

Required artifacts:

- `run.json`
- `resolved.json`
- `metrics.json`
- `results.json`
- `artifacts/predictions/<predictions_name>.parquet`

Conditionally required:

- `score_provenance.json`
  - Present when post-run scoring succeeds.
  - May be absent if post-run scoring fails.

Not guaranteed by the canonical training pipeline:

- `artifacts/predictions/val_per_era_corr20v2.parquet`
- `artifacts/predictions/val_per_era_corr20v2.csv`
- `artifacts/predictions/val_per_era_payout_map.parquet`
- `artifacts/predictions/val_per_era_payout_map.csv`
- `artifacts/eval/feature_importance.csv`
- `artifacts/reports/trials.csv`
- `artifacts/reports/best_params.json`

Consumers must derive these read-time diagnostics from canonical artifacts when needed.

## Canonical Field Names

Target identity:

- `run.json -> data.target_col` is the canonical run target field.
- `results.json -> data.target` is the canonical target in results payload.

Training does not require `target_train` / `target_payout` for single-target runs.
Consumers must not assume those fields exist.

Canonical metric aliases (for viz/read APIs):

- `corr20v2_mean` from `corr.mean`
- `corr20v2_sharpe` from `corr.sharpe`
- `mmc_mean` from `mmc.mean`
- `bmc_mean` from `bmc.mean`
- `cwmm_mean` from `cwmm.mean`
- `max_drawdown` from `corr.max_drawdown`
- `payout_estimate_mean` from `payout_score`, or derived fallback

Canonical payout fallback formula:

- `payout_estimate_mean = clip(0.75 * corr20v2_mean + 2.25 * mmc_mean, +/-0.05)`

## Score Provenance Contract

When `score_provenance.json` exists, it must include:

- `columns`: id/era/target/prediction/meta/benchmark names used
- `sources`: fingerprinted paths for predictions, meta model, benchmark
- `joins`: overlap counters (rows, eras)

Canonical coverage ratio derivation:

- `mmc_coverage_ratio_rows = joins.meta_overlap_rows / joins.predictions_rows`

## Consumer Rules

- Prefer canonical artifacts first (`run.json`, `metrics.json`, `results.json`,
  `score_provenance.json`, predictions parquet).
- If precomputed visualization tables are absent, derive diagnostics from canonical inputs.
- Do not infer non-canonical schema fields that training did not write.

## Anti-Drift Policy

Any change to run artifact schema, metric shape, or canonical field naming must update:

- `src/numereng/features/training/CLAUDE.md`
- `docs/llms.txt`
- `docs/ARCHITECTURE.md`
- Relevant viz adapter mappings and tests
