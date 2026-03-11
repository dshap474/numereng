# Metrics

Canonical run metrics written by Numereng and consumed by reporting/viz flows.

## Core Output Keys

`metrics.json` and normalized views expose these families:

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

No payout estimate fields are emitted.

## Interpretation

- `corr` / `corr_<alias>`: Numerai correlation against the native or aliased scoring target.
- `fnc` / `fnc_<alias>`: feature-neutral correlation using `fncv3_features`, then correlating against the scoring target being evaluated.
- `mmc` / `mmc_<alias>`: meta-model contribution metrics on the available overlapping meta window.
- `cwmm`: prediction/meta correlation diagnostic using the Numerai prediction transform and raw meta model series.
- `bmc` / `bmc_last_200_eras`: benchmark contribution diagnostics against one selected benchmark model.
- `feature_exposure` / `max_feature_exposure`: local diagnostics on rank-based exposure to `fncv3_features`.

## Coverage and Provenance

`score_provenance.json` is the source of scoring-policy and join-coverage context. Relevant fields:

- `joins.predictions_rows`
- `joins.benchmark_source_rows`
- `joins.benchmark_overlap_rows`
- `joins.benchmark_missing_rows`
- `joins.benchmark_missing_eras`
- `joins.meta_overlap_rows`
- `joins.meta_source_rows`
- `joins.meta_overlap_eras`
- `policy.fnc_feature_set`
- `policy.fnc_target_policy`
- `policy.benchmark_min_overlap_ratio`
- `policy.include_feature_neutral_metrics`

Canonical training requires strict era alignment for benchmark/meta joins, but not whole-run full coverage. Benchmark diagnostics are computed on overlapping rows only. Meta metrics are emitted on the maximum available overlapping meta-model window whenever any overlap exists.

## Dashboard Contract

Primary ranking/report keys currently exposed by viz include:

- `corr_mean`
- `corr_sharpe`
- `fnc_mean`
- `mmc_mean`
- `bmc_mean`
- `feature_exposure_mean`
- `max_feature_exposure`

Additional aliased families such as `corr_ender20.mean` or `fnc_ender20.mean` remain available in `metrics.json` even when viz normalizes only the native-target scalar keys.
