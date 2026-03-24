# Scoring Contract

This package defines the canonical run-scoring contract for Numereng.
Training artifacts produced from this package are intended for local model
evaluation and diagnostics, not stake sizing or expected-payout estimation.

## Implemented Metric Families

Training scoring emits these benchmark-contribution metric families when the
required join sources are available:

- `bmc`
- `bmc_<alias>`
- `bmc_last_200_eras`
- `bmc_last_200_eras_<alias>`

Training scoring emits these meta-dependent metric families only when there is
strictly era-aligned meta-model overlap:

- `cwmm`

Feature-heavy scoring stages (`all` and `post_training_full`) also emit:

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

- the payout target uses the unsuffixed key `mmc`
- explicit extra scoring targets use suffixed keys such as `mmc_target`,
  `mmc_agnes20`

No payout estimate fields are emitted.

## Canonical Metric Semantics

Implementation note: these semantics are preserved by an internal NumPy/Numba
kernel layer in `features/scoring/_fastops.py`; the contract is mathematical
compatibility with the metrics below, not pandas callback execution.

- `corr`: per-era `numerai_tools.scoring.numerai_corr` against the native run target
- `corr_<alias>`: the same metric against one non-native scoring target
- `fnc`: per-era `numerai_tools.scoring.feature_neutral_corr` using `fncv3_features` against the native run target
- `fnc_<alias>`: the same metric against one non-native scoring target
- `mmc`: per-era `numerai_tools.scoring.correlation_contribution` against the meta model and payout target, computed on the overlapping meta-model window only
- `mmc_<alias>`: the same metric against the meta model and one explicit extra scoring target, computed on the overlapping meta-model window only
- `cwmm`: per-era Pearson correlation between the Numerai-transformed submission and the raw meta-model series, computed on the overlapping meta-model window only
- `bmc`: diagnostics-style benchmark contribution against a single selected benchmark model, evaluated on the payout target
- `bmc_<alias>`: the same benchmark contribution evaluated against one explicit extra scoring target
- `bmc_last_200_eras`: the same payout-target BMC family restricted to the most recent 200 eras
- `bmc_last_200_eras_<alias>`: the same explicit extra-target BMC family restricted to the most recent 200 eras

## Local Diagnostics

These are useful Numereng diagnostics but are not canonical official Numerai
payout metrics:

- `feature_exposure`: per-era RMS rank-based feature exposure across `fncv3_features`
- `max_feature_exposure`: per-era max absolute rank-based feature exposure across `fncv3_features`
- `avg_corr_with_benchmark`: local diagnostic attached to BMC summaries; computed as per-era `numerai_tools.scoring.numerai_corr` between the submission and the selected benchmark series, then averaged across eras
- `corr_delta_vs_baseline` / `corr_delta_vs_baseline_<alias>`: the per-era difference between the model's CORR and the benchmark model's own CORR on the same scoring target; positive values mean the model outperformed the benchmark on that era

`baseline_corr` is now an internal/shared helper rather than a persisted run
metric:

- active-benchmark payout delta reads `benchmark.json -> artifacts.per_era_corr_target_ender_20` when available
- explicit `benchmark_source.source=path` runs fall back to transient local benchmark-CORR computation for delta only
- `baseline_corr` is not persisted in `metrics.json`, `results.json`, summary parquet files, or `run_metric_series.parquet`

## Fixed Policy

- default `corr` / `fnc` targets: native run target plus opportunistic `target_ender_20`
- default contribution targets: payout target only (`target_ender_20`)
- explicit config override: `data.scoring_targets`
  - `corr` / `fnc` keep using the resolved scoring-target set
  - contribution families add alias-suffixed extra targets from the explicit list while reserving the unsuffixed keys for the payout target
- `fnc_feature_set`: `fncv3_features`
- `fnc_target_policy`: correlate the feature-neutralized submission against the scoring target being evaluated
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
- default contribution families are omitted when `target_ender_20` is unavailable
- FNC feature joins require full `(id, era)` coverage.
- Benchmark and meta-model joins require strict era alignment after matching on `id`.
- Benchmark and meta-model joins must have non-zero overlap.
- Benchmark joins do not require a whole-run minimum-overlap threshold; `bmc` /
  `bmc_last_200_eras` are computed on the available overlapping benchmark
  window whenever any strictly era-aligned overlap exists.
- Boundary-era partial benchmark coverage is expected in some local datasets;
  this does not invalidate native CORR scoring and benchmark-relative metrics
  are computed on the aligned overlap window only.
- Meta-model joins do not require a whole-run minimum-overlap threshold; `mmc` /
  `cwmm` are computed on the available overlapping meta-model window whenever
  any strictly era-aligned overlap exists.
## Summary Shape

Each emitted metric family persists a nested summary for each prediction column:

- `mean`
- `std`
- `sharpe`
- `max_drawdown`

BMC summary families also persist:

- `avg_corr_with_benchmark`

## Artifacts

This package writes or refreshes the following scoring artifacts:

- `metrics.json`
- `results.json`
- `run.json`
- `score_provenance.json`
- `artifacts/scoring/manifest.json`
- `artifacts/scoring/run_metric_series.parquet`
- `artifacts/scoring/post_fold_per_era.parquet`
- `artifacts/scoring/post_fold_snapshots.parquet`
- `artifacts/scoring/post_training_core_summary.parquet`
- `artifacts/scoring/post_training_full_summary.parquet` when feature-neutral diagnostics are materialized

`run train` automatically refreshes `post_training_core` after predictions are
written. `run score` and `experiment score-round` can later refresh any stage,
including the inclusive feature-heavy `post_training_full`. Historical runs may still expose the
legacy filenames `post_training_summary.parquet` and
`post_training_features_summary.parquet`, which remain read-compatible.

There is one official persisted-run scorer. It supports these canonical stage
selectors:

- `all`
- `run_metric_series`
- `post_fold`
- `post_training_core`
- `post_training_full`

Stage-selective rescoring writes only the selected canonical stage artifacts
into `artifacts/scoring/` and leaves non-selected existing stage files
untouched. Each scoring invocation still refreshes `metrics.json`,
`results.json`, `run.json`, `score_provenance.json`, and
`artifacts/scoring/manifest.json`.

`post_training_full` is inclusive and refreshes both the core summary and the
feature-heavy full summary in one pass.

`post_training_core_summary.parquet` is the flattened scorecard stage. For new
runs it includes the native and payout CORR summaries, payout-backed `mmc`,
`bmc`, `bmc_last_200_eras`, scalar `avg_corr_with_benchmark`, and
`corr_delta_vs_baseline` summary stats when the required sources are available.

`run_metric_series.parquet` is intentionally narrower for contribution metrics:

- it charts payout-backed `bmc`, `mmc`, and unsuffixed `corr_delta_vs_baseline`
- it does not chart `baseline_corr`
- it hides legacy native-vs-Ender20 contribution duplication while keeping legacy runs read-compatible in viz

All numereng-written parquet artifacts use `ZSTD` compression level `3`.

New runs do not persist the legacy per-metric `*_per_era.parquet` /
`*_cumulative.parquet` fanout. Viz adapts those older files read-only for
historical runs that have not been rescored yet.

## Provenance Contract

`score_provenance.json` records:

- selected prediction, native target, resolved scoring-target lists, id, era, benchmark,
  and meta-model columns
- target-metric alias mapping used to derive `corr_<alias>` / `fnc_<alias>` / `mmc_<alias>` / `bmc_<alias>` keys
- `fnc_feature_set` and feature count
- fingerprinted prediction, benchmark, and meta-model sources
- feature source paths used for FNC
- join row and era counts
- baseline-delta provenance describing whether payout benchmark CORR came from the shared active-benchmark artifact or transient fallback computation
- benchmark/meta missing row and era counts when applicable
- overlap ratios for benchmark/meta joins
- whether meta-dependent metrics were emitted, and the omission reason when they were not
- fixed policy settings for benchmark/meta overlap behavior
- requested canonical stage selection and refreshed canonical stages
- preserved benchmark alias metadata used by Numereng consumers even though the
  underlying scorer executes through array kernels

## Payout Contract

Numereng does not emit `payout_estimate` or `payout_estimate_mean`.
There is no implemented claim that validation metrics can be converted into an
official expected payout estimate.
