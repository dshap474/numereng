# Metrics

Canonical run metrics written by Numereng and consumed by reporting/viz flows.

## Core Output Keys

`metrics.json` and normalized views expose these families:

- `corr20v2_*`
- `fnc_*`
- `mmc_*`
- `cwmm_*`
- `bmc_*`
- `bmc_last_200_eras_*`
- `payout_estimate_*`
- `mmc_coverage_*`
- `validation_profile`

## Interpretation

- `corr20v2_*`: primary tournament-aligned correlation metrics.
- `fnc_*`: feature-neutral correlation diagnostics using the fixed `all` neutralization set.
- `mmc_*`: meta-model contribution metrics.
- `cwmm_*`: prediction/meta correlation diagnostics.
- `bmc_*`: benchmark contribution diagnostics (informational, not payout axis).
- `payout_estimate_*`: clipped blend of CORR and MMC.

## Payout Estimate

For Numerai Classic (as of January 1, 2026), canonical training persists `payout_estimate_*` only for runs scored on `target_ender_20`, using:

```text
clip(0.75 * corr20v2 + 2.25 * mmc, -0.05, +0.05)
```

For non-`target_ender_20` runs, `payout_estimate_mean` is intentionally `null`.

The same weights are exposed via compat payloads. Viz fallback logic only applies to older runs that predate persisted payout metrics.

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
- `policy.benchmark_overlap_policy`
- `policy.meta_overlap_policy`

`mmc_coverage_ratio_rows` is effectively `meta_overlap_rows / predictions_rows` when available.
Low coverage means MMC and payout estimates are less reliable even when headline values look strong.
Canonical training requires benchmark overlap, not benchmark full coverage. Partial benchmark overlap is preserved in provenance and benchmark diagnostics are computed on overlapping rows only. Meta-model coverage remains strict.

## Dashboard Contract

Primary ranking/report keys:

- `corr20v2_sharpe`
- `corr20v2_mean`
- `mmc_mean`
- `payout_estimate_mean`
- `mmc_coverage_ratio_rows`
- `bmc_mean` (diagnostic)

Payout map axes are `corr20v2_mean` (x) and `mmc_mean` (y).
