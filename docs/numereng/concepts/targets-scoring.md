# Targets & Scoring

Numereng keeps target/scoring behavior aligned with compat metadata and persisted run provenance.

## Targets

Training config target keys:

- `data.target_col`: training target used for model fit and core scoring context.
- `data.target_horizon` (optional `20d|60d`): explicit official-horizon control used for embargo defaults.

Official default mapping:
- `20d -> embargo 8`
- `60d -> embargo 16`

Horizon resolution order:
1. `data.target_horizon` (when set)
2. infer from `data.target_col`
3. fail if still ambiguous

## Canonical Metric Families

Run scoring produces:

- `corr20v2_*`
- `mmc_*`
- `cwmm_*`
- `bmc_*`
- `payout_estimate_*`

`bmc_*` is diagnostic; payout alignment is driven by CORR + MMC.

## Payout Estimate

```text
clip(0.75 * CORR20v2 + 2.25 * MMC, -0.05, +0.05)
```

Compat payload endpoint:

- `GET /api/system/numerai-classic-compat`

## Provenance

`score_provenance.json` records source fingerprints and join coverage. In particular:

- `joins.predictions_rows`
- `joins.meta_overlap_rows`

These fields are used to reason about MMC coverage quality (for example, `mmc_coverage_ratio_rows`).

## Viz Alignment

Dashboard payout map uses:

- X axis: `corr20v2_mean`
- Y axis: `mmc_mean`

If `payout_estimate_mean` is missing in persisted metrics, viz computes it from the same payout formula.
