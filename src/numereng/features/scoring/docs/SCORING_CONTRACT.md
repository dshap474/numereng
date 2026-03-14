# Scoring Contract

This package defines the canonical post-training scoring contract for Numereng.
Training artifacts produced from this package are intended for local model
evaluation and diagnostics, not stake sizing or expected-payout estimation.

## Implemented Metric Families

Training scoring emits these native-target metric families when the required
join sources are available:

- `bmc`
- `bmc_ender20`
- `bmc_last_200_eras`
- `bmc_ender20_last_200_eras`

Training scoring emits these meta-dependent metric families only when there is
strictly era-aligned meta-model overlap:

- `cwmm`

When feature-neutral metrics are enabled, training scoring also emits:

- `fnc`
- `fnc_<alias>`
- `feature_exposure`
- `max_feature_exposure`

Training scoring also emits `corr` families for the resolved scoring
targets:

- native run target uses the unsuffixed key `corr`
- non-native scoring targets use suffixed keys such as `corr_ender20`,
  `corr_agnes20`

When there is strictly era-aligned meta-model overlap, training scoring also
emits `mmc` families for the resolved scoring targets:

- native run target uses the unsuffixed key `mmc`
- non-native scoring targets use suffixed keys such as `mmc_ender20`,
  `mmc_agnes20`

No payout estimate fields are emitted.

## Canonical Metric Semantics

- `corr`: per-era `numerai_tools.scoring.numerai_corr` against the native run target
- `corr_<alias>`: the same metric against one non-native scoring target
- `fnc`: per-era `numerai_tools.scoring.feature_neutral_corr` using `fncv3_features` against the native run target
- `fnc_<alias>`: the same metric against one non-native scoring target
- `mmc`: per-era `numerai_tools.scoring.correlation_contribution` against the meta model and native run target, computed on the overlapping meta-model window only
- `mmc_<alias>`: the same metric against the meta model and one non-native scoring target, computed on the overlapping meta-model window only
- `cwmm`: per-era Pearson correlation between the Numerai-transformed submission and the raw meta-model series, computed on the overlapping meta-model window only
- `bmc`: diagnostics-style benchmark contribution against a single selected benchmark model
- `bmc_ender20`: the same benchmark contribution evaluated against `target_ender_20`
- `bmc_last_200_eras`: the same BMC family restricted to the most recent 200 eras

## Local Diagnostics

These are useful Numereng diagnostics but are not canonical official Numerai
payout metrics:

- `feature_exposure`: per-era RMS rank-based feature exposure across `fncv3_features`
- `max_feature_exposure`: per-era max absolute rank-based feature exposure across `fncv3_features`
- `avg_corr_with_benchmark`: local diagnostic attached to `bmc` and `bmc_last_200_eras`; computed as per-era `numerai_tools.scoring.numerai_corr` between the submission and the selected benchmark series, then averaged across eras

## Fixed Policy

- default scoring targets: native run target plus opportunistic `target_ender_20`
- explicit config override: `data.scoring_targets`
  - when present, this list becomes the entire scoring-target set
- `fnc_feature_set`: `fncv3_features`
- `fnc_target_policy`: correlate the feature-neutralized submission against the scoring target being evaluated
- `include_feature_neutral_metrics`: `true` by default; when `false`, `fnc`,
  `feature_exposure`, and `max_feature_exposure` are omitted
- `benchmark_min_overlap_ratio`: `0.0`
- meta-model overlap policy: emit `mmc`, `mmc_<alias>`, and `cwmm` when there is
  any strictly era-aligned `(id, era)` overlap; compute them on the maximum
  available overlapping meta-model window
- diagnostics benchmark default: `.numereng/datasets/baselines/active_benchmark/predictions.parquet`
- diagnostics benchmark override: explicit `data.benchmark_source={source=\"path\", predictions_path=...}`
- diagnostics BMC semantics: contribution relative to the configured benchmark prediction source, not stake-weighted live benchmark meta semantics
- meta-dependent metrics (`mmc`, `mmc_<alias>`, `cwmm`) use the maximum available overlapping meta-model rows rather than requiring whole-run coverage

## Join and Coverage Rules

- non-native scoring targets are joined back from dataset files by `(id, era)`
- explicitly configured scoring-target joins require full `(id, era)` coverage
- opportunistic `target_ender_20` scoring is emitted only when the target column is available on the scoring universe
- FNC feature joins require full `(id, era)` coverage.
- Benchmark and meta-model joins require strict era alignment after matching on `id`.
- Benchmark and meta-model joins must have non-zero overlap.
- Benchmark joins do not require a whole-run minimum-overlap threshold; `bmc` /
  `bmc_last_200_eras` are computed on the available overlapping benchmark
  window whenever any strictly era-aligned overlap exists.
- Meta-model joins do not require a whole-run minimum-overlap threshold; `mmc` /
  `cwmm` are computed on the available overlapping meta-model window whenever
  any strictly era-aligned overlap exists.
- Era-stream scoring may skip chunks with zero benchmark/meta overlap after whole-run preflight has validated benchmark coverage and recorded the available meta overlap window.

## Summary Shape

Each emitted metric family persists a nested summary for each prediction column:

- `mean`
- `std`
- `sharpe`
- `max_drawdown`

`bmc` and `bmc_last_200_eras` also persist:

- `avg_corr_with_benchmark`

## Artifacts

This package writes or refreshes the following scoring artifacts:

- `metrics.json`
- `results.json`
- `run.json`
- `score_provenance.json`
- `artifacts/scoring/manifest.json`
- `artifacts/scoring/*.parquet`

## Provenance Contract

`score_provenance.json` records:

- selected prediction, native target, scoring-target list, id, era, benchmark,
  and meta-model columns
- target-metric alias mapping used to derive `corr_<alias>` / `fnc_<alias>` / `mmc_<alias>` keys
- `fnc_feature_set` and feature count
- fingerprinted prediction, benchmark, and meta-model sources
- feature source paths used for FNC
- join row and era counts
- benchmark/meta missing row and era counts when applicable
- overlap ratios for benchmark/meta joins
- whether meta-dependent metrics were emitted, and the omission reason when they were not
- fixed policy settings for benchmark/meta overlap behavior
- requested and effective scoring backend details for era-stream/materialized execution

## Payout Contract

Numereng does not emit `payout_estimate` or `payout_estimate_mean`.
There is no implemented claim that validation metrics can be converted into an
official expected payout estimate.
