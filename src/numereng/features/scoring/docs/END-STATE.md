# Scoring End State

## Purpose

This file describes the intended steady-state shape of Numereng scoring as it
exists today:

- one official persisted-run scorer
- one shared scoring computation pass
- a small set of canonical stage outputs
- a wider observability surface kept separate from the main selection scorecard

The canonical contract lives in:

- [SCORING_CONTRACT.md](/path/to/numereng/src/numereng/features/scoring/docs/SCORING_CONTRACT.md)

This file is the design-level summary of that contract.

## Numerai-Example Priorities

The vendored Numerai-example workflow emphasizes:

- primary: `bmc_last_200_eras.mean`
- secondary: `bmc.mean`
- sanity checks: `corr.mean`, `avg_corr_with_benchmark`
- stability checks: `sharpe`, `max_drawdown`

The standard visual is still the cumulative `corr` + BMC benchmark-comparison
plot.

## Current End State

Numereng now has one official persisted-run scorer.

- Training emits only `post_fold` pruning artifacts during CV.
- post-training scoring is optional and policy-driven; deferred runs keep
  `results.json` / `metrics.json` placeholders until scoring is materialized
- `run score` uses the persisted-run scorer to materialize deferred stages for one run.
- `experiment score-round` batch-materializes deferred stages for one experiment round.
- Older runs can be refreshed through the same path.
- Stage-selective rescoring writes into the canonical run scoring directory in
  place.

The scorer supports these canonical stage selectors:

- `all`
- `run_metric_series`
- `post_fold`
- `post_training_core`
- `post_training_full`

Stage-selective rescoring:

- rewrites only the selected canonical stage artifacts
- leaves non-selected existing stage artifacts untouched
- always refreshes `metrics.json`, `results.json`, `run.json`,
  `score_provenance.json`, and `artifacts/scoring/manifest.json`

## Core Policy

- Learn the native target with native `corr`.
- Judge business value on `Ender20`.
- Use cheap metrics early.
- Use heavy feature-neutral and exposure diagnostics late.
- Treat payout-target `mmc` / `bmc` as the default decision scorecard.

## Implemented Metric Surface

The implemented metric surface is intentionally broader than the core decision
surface.

Core metric families:

- `corr`
- `corr_<alias>`
- `mmc`
- `mmc_<alias>`
- `bmc`
- `bmc_last_200_eras`
- `bmc_<alias>`
- `bmc_last_200_eras_<alias>`

Meta-dependent helper metric:

- `cwmm`

Feature diagnostics:

- `fnc`
- `fnc_<alias>`
- `feature_exposure`
- `max_feature_exposure`

Observability helpers:

- `corr_with_benchmark`
- `corr_delta_vs_baseline_*`

`baseline_corr` is now treated as an internal/shared benchmark helper rather
than a run metric. Numereng still uses it to derive
`corr_delta_vs_baseline`, but it is not shown as a per-run scorecard chart or
persisted as a run metric family.

No payout estimate fields are emitted.

## Canonical Stages

### `post_fold`

Purpose: fast pruning signal.

Primary emitted metrics:

- `corr_<native>`
- `corr_ender20`
- `bmc`

Persisted artifacts:

1. `post_fold_per_era.parquet`
2. `post_fold_snapshots.parquet`

Notes:

- fold-time append remains an internal training optimization
- persisted-run rescoring can rebuild these artifacts from saved predictions
- `post_fold` may be unavailable for profiles that do not produce CV fold data

### `post_training_core`

Purpose: final model-selection scorecard.

Primary emitted metrics:

- `corr_<native>`
- `corr_ender20`
- `mmc`
- `bmc`
- `bmc_last_200_eras`
- `avg_corr_with_benchmark`
- `corr_delta_vs_baseline`

Persisted artifact:

- `post_training_core_summary.parquet`

Summary stats:

- `mean`
- `std`
- `sharpe`
- `max_drawdown`

Use this stage for keep/reject, ranking, champion decisions, and ensemble
selection.

### `post_training_full`

Purpose: heavy neutrality and exposure diagnostics.

This stage is inclusive: materializing it also refreshes `post_training_core`.

Primary emitted metrics:

- `fnc_<native>`
- `fnc_ender20`
- `feature_exposure`
- `max_feature_exposure`

Persisted artifact:

- `post_training_full_summary.parquet`

Use this stage for robustness and orthogonality analysis, not the first-pass
decision scorecard.

### `run_metric_series`

Purpose: broad observability and charting surface.

Persisted artifact:

- `run_metric_series.parquet`

This contains per-era and cumulative series for the broader emitted metric
surface, including diagnostics such as `corr_with_benchmark`, `cwmm`, FNC, and
exposure metrics. Contribution charts are intentionally narrowed to payout-backed
`bmc`, `mmc`, and `corr_delta_vs_baseline`; legacy native/baseline contribution
series remain read-compatible but hidden in viz.

## Default Practical Rule

- Observe per era.
- Monitor cumulatively.
- Prune from fold snapshots.
- Promote with `post_training_core`.
- Diagnose neutrality and exposure with `post_training_full`.

## Canonical Artifacts

The canonical scoring bundle is:

- `artifacts/scoring/manifest.json`
- `artifacts/scoring/run_metric_series.parquet`
- `artifacts/scoring/post_fold_per_era.parquet`
- `artifacts/scoring/post_fold_snapshots.parquet`
- `artifacts/scoring/post_training_core_summary.parquet`
- `artifacts/scoring/post_training_full_summary.parquet` when enabled

Associated refreshed files:

- `metrics.json`
- `results.json`
- `run.json`
- `score_provenance.json`

All numereng-written parquet artifacts use `ZSTD` compression level `3`.

## Example Target Mapping

For a non-Ender native target such as `Bravo20`:

- `post_fold` per-era artifact:
  - `corr_bravo20`
  - `corr_ender20`
  - `bmc`
- `post_fold` snapshot row:
  - `corr_bravo20.fold_mean`
  - `corr_ender20.fold_mean`
  - `bmc.fold_mean`
- `post_training_core`:
  - `corr_bravo20`
  - `corr_ender20`
  - `mmc`
  - `bmc`
  - `bmc_last_200_eras`
  - `avg_corr_with_benchmark`
  - `corr_delta_vs_baseline`
- `post_training_full`:
  - `fnc_bravo20`
  - `fnc_ender20`
  - exposure diagnostics
