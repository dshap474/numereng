# Metrics

Numereng writes canonical scoring outputs to run artifacts and exposes normalized metric views through the store and dashboard.

## Files

For scored runs, the primary scoring artifacts are:

- `results.json`
- `metrics.json`
- `score_provenance.json`

`run score` refreshes those files from persisted predictions and updates the run index rows in SQLite.

## Core Metric Families

`metrics.json` exposes these canonical families:

- `corr`
- `corr_<alias>`
- `fnc`
- `fnc_<alias>`
- `mmc`
- `mmc_<alias>`
- `cwmm`
- `bmc`
- `bmc_last_200_eras`
- `feature_exposure`
- `max_feature_exposure`
- `max_drawdown`

Numereng does not emit payout-estimate fields.

## Interpretation

- `corr`: Numerai correlation against the native or aliased scoring target
- `fnc`: feature-neutral correlation using `fncv3_features`
- `mmc`: meta-model contribution on the available overlapping meta window
- `cwmm`: diagnostic correlation between Numerai-transformed predictions and the raw meta-model series
- `bmc`: benchmark contribution against the configured benchmark model
- `bmc_last_200_eras`: benchmark contribution restricted to the trailing 200 eras
- `feature_exposure`: rank-based exposure summary against `fncv3_features`
- `max_feature_exposure`: maximum absolute exposure summary
- `max_drawdown`: drawdown derived from the relevant per-era score series

## Coverage And Provenance

`score_provenance.json` records the scoring-policy and join-coverage context used to produce the metrics.

Important fields include:

- `joins.predictions_rows`
- `joins.benchmark_source_rows`
- `joins.benchmark_overlap_rows`
- `joins.benchmark_missing_rows`
- `joins.benchmark_missing_eras`
- `joins.meta_source_rows`
- `joins.meta_overlap_rows`
- `joins.meta_overlap_eras`
- `policy.fnc_feature_set`
- `policy.fnc_target_policy`
- `policy.benchmark_min_overlap_ratio`
- `policy.include_feature_neutral_metrics`

Current rules:

- benchmark and meta-model joins require strict era alignment
- benchmark diagnostics score only overlapping rows
- meta metrics are emitted whenever there is usable overlap

## Dashboard Contract

The dashboard normalizes a subset of the metric space into common ranking fields, including:

- `corr_mean`
- `corr_sharpe`
- `fnc_mean`
- `mmc_mean`
- `bmc_mean`
- `feature_exposure_mean`
- `max_feature_exposure`

Aliased target families remain available in `metrics.json` even when the dashboard presents only canonical scalar keys.

## Special Case: `full_history_refit`

`full_history_refit` is final-fit only. It does not emit validation metrics because there is no scored validation window.
