# Metrics

Canonical run metrics written by Numereng and consumed by reporting/viz flows.

## Core Output Keys

`metrics.json` and normalized views expose these families:

- `corr20v2_*`
- `mmc_*`
- `cwmm_*`
- `bmc_*`
- `bmc_last_200_eras_*`
- `payout_estimate_*`
- `mmc_coverage_*`
- `validation_profile`

## Interpretation

- `corr20v2_*`: primary tournament-aligned correlation metrics.
- `mmc_*`: meta-model contribution metrics.
- `cwmm_*`: prediction/meta correlation diagnostics.
- `bmc_*`: benchmark contribution diagnostics (informational, not payout axis).
- `payout_estimate_*`: clipped blend of CORR and MMC.

## Payout Estimate

Fallback formula:

```text
clip(0.75 * corr20v2 + 2.25 * mmc, -0.05, +0.05)
```

The same weights are exposed via compat payloads and viz fallback logic.

## Coverage and Provenance

`score_provenance.json` is the source of MMC join coverage context. Relevant fields:

- `joins.predictions_rows`
- `joins.meta_overlap_rows`
- `joins.meta_overlap_eras`

`mmc_coverage_ratio_rows` is effectively `meta_overlap_rows / predictions_rows` when available.
Low coverage means MMC and payout estimates are less reliable even when headline values look strong.

## Dashboard Contract

Primary ranking/report keys:

- `corr20v2_sharpe`
- `corr20v2_mean`
- `mmc_mean`
- `payout_estimate_mean`
- `mmc_coverage_ratio_rows`
- `bmc_mean` (diagnostic)

Payout map axes are `corr20v2_mean` (x) and `mmc_mean` (y).
