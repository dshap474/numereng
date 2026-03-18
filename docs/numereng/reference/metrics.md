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
- `bmc_<alias>`
- `bmc_last_200_eras`
- `bmc_last_200_eras_<alias>`
- `corr_delta_vs_baseline`
- `corr_delta_vs_baseline_<alias>`
- `feature_exposure`
- `max_feature_exposure`
- `max_drawdown`

Numereng does not emit payout-estimate fields.

## Interpretation

- `corr`: Numerai correlation against the native or aliased scoring target
- `fnc`: feature-neutral correlation using `fncv3_features`
- `mmc`: payout-target meta-model contribution on the available overlapping meta window
- `mmc_<alias>`: explicit extra-target meta-model contribution
- `cwmm`: diagnostic correlation between Numerai-transformed predictions and the raw meta-model series
- `bmc`: payout-target benchmark contribution against the configured benchmark model
- `bmc_<alias>`: explicit extra-target benchmark contribution
- `bmc_last_200_eras`: payout-target benchmark contribution restricted to the trailing 200 eras
- `bmc_last_200_eras_<alias>`: explicit extra-target trailing-200 benchmark contribution
- `corr_delta_vs_baseline`: model payout-target CORR minus the benchmark model's payout-target CORR on the same scored eras
- `corr_delta_vs_baseline_<alias>`: the same delta on one explicit extra scoring target
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
- `baseline_corr.mode`

Current rules:

- benchmark and meta-model joins require strict era alignment
- benchmark diagnostics score only overlapping rows
- meta metrics are emitted whenever there is usable overlap
- `baseline_corr` itself is no longer persisted as a run metric; payout-target benchmark CORR is read from the shared active-benchmark artifact when available, or computed transiently for delta fallback

## Dashboard Contract

The dashboard normalizes a subset of the metric space into common ranking fields, including:

- `corr_mean`
- `corr_sharpe`
- `fnc_mean`
- `mmc_mean`
- `bmc_mean`
- `feature_exposure_mean`
- `max_feature_exposure`

`mmc_mean`, `bmc_mean`, and `bmc_last_200_eras_mean` normalize to the payout-backed surface for newly written runs, with legacy `*_ender20` fallback for historical runs.

Aliased target families remain available in `metrics.json` even when the dashboard presents only canonical scalar keys.

## Special Case: `full_history_refit`

`full_history_refit` is final-fit only. It does not emit validation metrics because there is no scored validation window.
