# Scoring Comparison: Numereng vs `numerai-example` Crossover Only

## Scope

This note compares only the scoring outputs that exist in both:

- `src/numereng/features/scoring/metrics.py`
- `src/numereng/features/scoring/numerai-example/numerai_metrics.py`

It intentionally excludes numereng-only metrics and workflow code such as:

- `fnc`
- `feature_exposure`
- `max_feature_exposure`
- `cwmm`
- target-alias metric families
- provenance, persisted-run rescoring, and service-layer orchestration

The comparison target is the vendored `numerai-example` snapshot in this repo.

Research anchor: both codepaths delegate the shared core math to
`numerai_tools.scoring` (`numerai_corr` and `correlation_contribution`):
https://pypi.org/project/numerai-tools/

## Actual Shared Surface

These are the only shared outputs worth comparing:

| Shared output | `numerai-example` | numereng | Primary code refs | Same math? |
| --- | --- | --- | --- | --- |
| `corr` | `per_era_corr(...)` | `per_era_corr(...)` | example `60-74`, numereng `225-248` | Yes |
| `mmc` | `per_era_mmc(...)` | `per_era_mmc(...)` | example `77-94`, numereng `351-379` | Yes |
| `bmc` | `per_era_bmc(...)` | `per_era_bmc(...)` | example `114-131`, numereng `320-348` | Yes |
| `avg_corr_with_benchmark` | `per_era_pred_corr(...)` then era mean | `per_era_reference_corr(...)` then era mean | example `97-111` and `411-420`, numereng `282-305` and `2014-2016` | Yes |
| `bmc_last_200_eras` | `_last_n_eras(..., 200)` + `summarize_scores(...)` | `_last_n_eras(..., 200)` + `summarize_scores(...)` | example `53-57` and `423-431`, numereng `170-174` and `2019-2027` | Yes |
| summary stats | `mean`, `std`, `sharpe`, `max_drawdown` | `mean`, `std`, `sharpe`, `max_drawdown` | example `134-167`, numereng `501-538` | Yes |

## Code Comparison For The Shared Surface

### `corr`

`numerai-example`

- `per_era_corr(...)` drops NaNs once for all prediction columns, then runs
  `df.groupby(era_col).apply(_corr)` where `_corr` calls
  `numerai_corr(group[pred_cols], group[target_col])`.

numereng

- `per_era_corr(...)` loops prediction columns one at a time and routes each
  column through `_single_prediction_per_era(...)`.
- the scorer still calls `numerai_corr(group[[col]], group[target_col])`.

Conclusion:

- same scoring math
- numereng is longer because it uses a per-column helper instead of one
  groupby call over all prediction columns at once

### `mmc`

`numerai-example`

- `per_era_mmc(...)` drops NaNs once, groups by era, then calls
  `correlation_contribution(group[pred_cols], group[meta_col], group[target_col])`.

numereng

- `per_era_mmc(...)` uses the same `correlation_contribution(...)` call
  shape, but again routes each prediction column through
  `_single_prediction_per_era(...)`.

Conclusion:

- same scoring math
- numereng is structurally more defensive, not mathematically different

### `bmc`

`numerai-example`

- `per_era_bmc(...)` is the benchmark analogue of `per_era_mmc(...)`.
- it uses `correlation_contribution(group[pred_cols], group[benchmark_col], group[target_col])`.

numereng

- `per_era_bmc(...)` uses the same core formula and the same helper pattern as
  numereng `mmc`.

Conclusion:

- same scoring math
- numereng is longer for the same reason as `mmc`: single-column helper path,
  extra type/error checks, and local normalization

### `avg_corr_with_benchmark`

`numerai-example`

- computes per-era benchmark correlation with `per_era_pred_corr(...)`
- that helper uses `numerai_corr(group[pred_cols], group[benchmark_col])`
- then averages the per-era series and attaches the result to the `bmc`
  summary row as `avg_corr_with_benchmark`

numereng

- computes the same quantity through `per_era_reference_corr(...)`
- that helper also uses `numerai_corr(group[[col]], group[reference_col])`
- then averages the per-era series and attaches it to `bmc` and
  `bmc_last_200_eras`

Conclusion:

- same metric semantics
- different helper name only
- numereng's `per_era_reference_corr(...)` is the crossover equivalent of
  `numerai-example`'s `per_era_pred_corr(...)`

### `bmc_last_200_eras`

`numerai-example`

- computes per-era `bmc`
- slices the last `200` eras via `_last_n_eras(...)`
- summarizes that window with `summarize_scores(...)`
- computes a matching last-`200` benchmark-correlation mean and stores it as
  `avg_corr_with_benchmark`

numereng

- follows the same sequence:
  `bmc_per_era -> _last_n_eras(..., 200) -> summarize_scores(...)`
- also computes the matching recent benchmark-correlation mean and stores it as
  `avg_corr_with_benchmark`

Conclusion:

- same report-level behavior
- numereng names the benchmark-correlation helper differently, but the last-200
  summary logic matches the example

### Shared Summary Statistics

Both implementations use the same formulas for:

- `mean`
- `std` with `ddof=0`
- `sharpe = mean / std`
- `max_drawdown` as drawdown on cumulative per-era score sums

The implementations are effectively the same in:

- `max_drawdown(...)`
- `score_summary(...)`
- `summarize_scores(...)`

## Why Numereng Is Longer Even On The Shared Surface

If we compare only the crossover code, the size gap is much smaller than the
current full-package comparison suggests.

Approximate shared-surface line counts:

- `numerai-example`
  - `numerai_metrics.py:32-167`
  - `numerai_metrics.py:392-431`
  - about `176` lines
- numereng
  - `metrics.py:157-379`
  - `metrics.py:501-538`
  - `metrics.py:2013-2027`
  - about `276` lines

Why numereng still uses more lines for the same outputs:

- it centralizes per-column scoring in `_single_prediction_per_era(...)`
- it adds `_groupby_apply_per_era(...)` for pandas compatibility around
  `include_groups=False`
- it keeps local type/error validation around per-era callback return shapes
- it splits benchmark-correlation naming into
  `per_era_reference_corr(...)` instead of reusing an example-specific name

So the fair comparison is not:

- `432` lines vs `2372` lines

It is closer to:

- `176` lines vs `276` lines for the overlapping metric surface

## Bottom Line

For the metrics that actually overlap, numereng and `numerai-example` match on
the scoring formulas and summary semantics.

The real crossover set is:

- `corr`
- `mmc`
- `bmc`
- `bmc_last_200_eras`
- `avg_corr_with_benchmark`
- `mean`
- `std`
- `sharpe`
- `max_drawdown`

Numereng is still longer on that shared surface, but mostly because of helper
abstraction, compatibility handling, and package-grade validation. It is not
because the shared metric formulas are materially different.
