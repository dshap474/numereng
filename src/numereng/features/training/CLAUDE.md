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
- `artifacts/eval/feature_importance.csv`
- `artifacts/reports/trials.csv`
- `artifacts/reports/best_params.json`

Consumers must not derive these diagnostics at request time. Read-only surfaces
should show them only when the precomputed visualization artifacts exist.

## Canonical Field Names

Target identity:

- `run.json -> data.target_col` is the canonical run target field.
- `results.json -> data.target` is the canonical target in results payload.
- `data.scoring_targets` is optional; when omitted, scoring defaults to the run
  target plus `target_ender_20`.

Training does not require `target_train` / `target_payout` for single-target runs.
Consumers must not assume those fields exist.

Canonical viz/read metric names:

- `corr_mean` from `corr.mean`
- `corr_sharpe` from `corr.sharpe`
- `mmc_mean` from `mmc.mean`
- extra scoring-target families may also be present as `corr_<alias>` /
  `mmc_<alias>` in `metrics.json` and `results.json`
- `bmc_mean` from `bmc.mean`
- `bmc_last_200_eras_mean` from `bmc_last_200_eras.mean`
- `cwmm_mean` from `cwmm.mean` when meta-model overlap exists
- `feature_exposure_mean` from `feature_exposure.mean`
- `max_feature_exposure` from `max_feature_exposure.mean`
- `max_drawdown` from `corr.max_drawdown`
- no payout estimate field is emitted by training scoring artifacts or viz APIs

## Score Provenance Contract

When `score_provenance.json` exists, it must include:

- `columns`: id/era/target/prediction/meta/benchmark names used
- `sources`: fingerprinted paths for predictions, meta model, benchmark
- `joins`: overlap counters (rows, eras)

Feature exposure diagnostics:

- Scoring reuses the canonical `fncv3_features` join path to compute rank-based per-era exposures.
- `feature_exposure` persists RMS exposure across features as `{mean,std,sharpe,max_drawdown}`.
- `max_feature_exposure` persists per-era max absolute exposure as `{mean,std,sharpe,max_drawdown}`.
- Viz/read APIs normalize `max_feature_exposure.mean` to the scalar key `max_feature_exposure`.

Canonical coverage ratio derivation:

- `mmc_coverage_ratio_rows = joins.meta_overlap_rows / joins.predictions_rows`
- benchmark metrics are computed on the available overlapping window when any strictly era-aligned benchmark overlap exists; meta-model metrics are computed on the available overlapping window when any strictly era-aligned meta overlap exists.

## Consumer Rules

- Prefer canonical artifacts first (`run.json`, `metrics.json`, `results.json`,
  `score_provenance.json`, predictions parquet).
- If precomputed visualization tables are absent, read-only consumers must surface them as unavailable.
- Do not infer non-canonical schema fields that training did not write.

## Anti-Drift Policy

Any change to run artifact schema, metric shape, or canonical field naming must update:

- `src/numereng/features/training/CLAUDE.md`
- `docs/llms.txt`
- `docs/ARCHITECTURE.md`
- Relevant viz adapter mappings and tests
