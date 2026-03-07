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
- `fnc_*`
- `mmc_*`
- `cwmm_*`
- `bmc_*`
- `bmc_last_200_eras_*`
- `payout_estimate_*`

`bmc_*` is diagnostic; payout alignment is driven by CORR + MMC.

## Payout Estimate

For Numerai Classic (as of January 1, 2026), payout estimate semantics are tied to `target_ender_20`.

```text
clip(0.75 * CORR20v2 + 2.25 * MMC, -0.05, +0.05)
```

Compat payload endpoint:

- `GET /api/system/numerai-classic-compat`

## Provenance

`score_provenance.json` records source fingerprints, fixed scoring policy, and join coverage. In particular:

- `joins.predictions_rows`
- `joins.benchmark_source_rows`
- `joins.benchmark_overlap_rows`
- `joins.benchmark_missing_rows`
- `joins.benchmark_missing_eras`
- `joins.meta_overlap_rows`
- `joins.meta_source_rows`
- `policy.fnc_feature_set`
- `policy.benchmark_overlap_policy`
- `policy.meta_overlap_policy`

These fields are used to reason about MMC coverage quality (for example, `mmc_coverage_ratio_rows`).
Canonical training scoring uses `fnc_feature_set=all`, requires benchmark overlap, and keeps meta-model overlap strict. Partial benchmark overlap is tolerated and benchmark diagnostics are computed on overlapping rows only; partial or misaligned meta-model joins are scoring errors.

## Viz Alignment

Dashboard payout map uses:

- X axis: `corr20v2_mean`
- Y axis: `mmc_mean`

Canonical training persists `payout_estimate_mean` for `target_ender_20` runs. For non-`target_ender_20` runs this field is intentionally `null`.
Viz computes the same fallback formula only for older runs that predate persisted payout metrics.
