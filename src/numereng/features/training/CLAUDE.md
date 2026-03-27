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
- `runtime.json`
- `resolved.json`
- `metrics.json`
- `results.json`
- `artifacts/predictions/<predictions_name>.parquet`
  - parquet artifacts are written with `ZSTD` compression level `3`

Conditionally required:

- `score_provenance.json`
  - Present when deferred scoring has been materialized by `run score` or `experiment score-round`.
  - New training runs may legitimately omit it until deferred scoring happens.
- `artifacts/scoring/manifest.json`
- `artifacts/scoring/*.parquet`
  - Training guarantees only `post_fold` scoring artifacts during CV.
  - Deferred `post_training_core` / `post_training_full` artifacts appear after later scoring passes.
  - This bundle is the canonical persisted scoring surface for viz/report consumers.

Not guaranteed by the canonical training pipeline:

- `artifacts/reports/trials.parquet`
- `artifacts/reports/best_params.json`

CSV-only auxiliary artifacts are not part of the supported contract.

Consumers should treat `artifacts/scoring/manifest.json` as the scoring
artifact entrypoint and may use the store rescoring backfill path for legacy
runs. Other optional diagnostics must not be derived at request time.

Lifecycle contract:

- `runtime.json` is the canonical in-flight lifecycle snapshot and remains on disk as the final lifecycle snapshot after terminalization.
- `runtime.json` also carries canonical backend-owned progress fields:
  `progress_percent`, `progress_label`, `progress_current`, `progress_total`.
- `run.json` remains the canonical run artifact and terminal summary.
- Terminal `run.json.status` may be `FINISHED`, `FAILED`, `CANCELED`, or `STALE`.
- Terminal `run.json.lifecycle` carries `terminal_reason`, `terminal_detail`, `cancel_requested_at`, and `reconciled`.
- Progress percent is not a frontend guess. Training writes the fixed pipeline progress plan and fold-aware `train_model` updates into both `runtime.json` and `run_lifecycles`.

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
- If precomputed visualization tables are absent, read-only consumers may materialize canonical per-era CORR for legacy runs once, but should otherwise surface missing diagnostics as unavailable.
- Do not infer non-canonical schema fields that training did not write.

## Anti-Drift Policy

Any change to run artifact schema, metric shape, or canonical field naming must update:

- `src/numereng/features/training/CLAUDE.md`
- `docs/llms.txt`
- `docs/ARCHITECTURE.md`
- Relevant viz adapter mappings and tests
